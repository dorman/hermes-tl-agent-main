"""
Semantic session search — vector-embedding-based recall for past conversations.

Augments the existing FTS5 keyword search with cosine similarity over
per-session embeddings so Hermes can surface "that time I debugged a similar
CORS issue" without requiring the user to remember exact words.

Architecture
------------
- One embedding per root session (title + first user message + first assistant
  response), stored as a float32 BLOB in the session_embeddings table.
- Embeddings are computed lazily on first search and cached permanently.
- At search time, the query is embedded with the same model and compared
  against all cached session embeddings via cosine similarity.
- Results are merged with FTS5 results using Reciprocal Rank Fusion.

Embedding library support (tried in order, first available wins):
  1. fastembed  — ONNX-based, no PyTorch, ~50 MB first-run model download
  2. sentence_transformers — PyTorch-based, larger install footprint
  3. None → semantic search gracefully unavailable; FTS5 still works.

Thread safety
-------------
Embedding computation is CPU-bound.  The module-level _embed_fn and
_model_name are initialized once under a lock and reused across calls.
DB writes use SessionDB._execute_write (jitter-retry, BEGIN IMMEDIATE).
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding library detection
# ---------------------------------------------------------------------------

_embed_lock = threading.Lock()
_embed_fn: Optional[callable] = None   # (texts: list[str]) -> list[list[float]]
_embed_model_name: Optional[str] = None
_embed_initialized: bool = False

_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
_ST_MODEL = "all-MiniLM-L6-v2"


def _init_embedding() -> None:
    """Initialize embedding backend once. Must be called under _embed_lock."""
    global _embed_fn, _embed_model_name, _embed_initialized
    if _embed_initialized:
        return
    _embed_initialized = True

    # Tier 1: fastembed (ONNX, no PyTorch)
    try:
        from fastembed import TextEmbedding  # type: ignore[import]
        _model = TextEmbedding(_FASTEMBED_MODEL)
        _embed_fn = lambda texts: [list(v) for v in _model.embed(texts)]
        _embed_model_name = _FASTEMBED_MODEL
        logger.debug("semantic_search: using fastembed / %s", _FASTEMBED_MODEL)
        return
    except Exception:
        pass

    # Tier 2: sentence_transformers (PyTorch)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        _model = SentenceTransformer(_ST_MODEL)
        _embed_fn = lambda texts: _model.encode(texts, show_progress_bar=False).tolist()
        _embed_model_name = _ST_MODEL
        logger.debug("semantic_search: using sentence_transformers / %s", _ST_MODEL)
        return
    except Exception:
        pass

    logger.debug(
        "semantic_search: no embedding library available "
        "(install fastembed or sentence-transformers to enable semantic search)"
    )
    _embed_fn = None
    _embed_model_name = None


def get_embed_fn():
    """Return the active embedding callable, or None if unavailable."""
    with _embed_lock:
        if not _embed_initialized:
            _init_embedding()
        return _embed_fn


def get_model_name() -> Optional[str]:
    """Return the active model name, or None if unavailable."""
    with _embed_lock:
        if not _embed_initialized:
            _init_embedding()
        return _embed_model_name


def is_available() -> bool:
    """True if an embedding backend is available."""
    return get_embed_fn() is not None


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------

def _to_bytes(vec: Sequence[float]) -> bytes:
    """Serialize a float list to little-endian float32 bytes."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _from_bytes(blob: bytes) -> List[float]:
    """Deserialize little-endian float32 bytes to a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns 0 on zero norm."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
) -> List[str]:
    """Merge multiple ranked lists of session IDs using Reciprocal Rank Fusion.

    Each list is a ranking (index 0 = best). The output is a single merged
    ranking sorted by descending RRF score.  k=60 is the standard constant
    that de-emphasizes low-rank items without over-penalising them.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, sid in enumerate(ranking):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda s: scores[s], reverse=True)


# ---------------------------------------------------------------------------
# Embedding computation and caching
# ---------------------------------------------------------------------------

def _embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed a batch of texts. Returns None if no backend is available."""
    fn = get_embed_fn()
    if fn is None:
        return None
    try:
        return fn(texts)
    except Exception as exc:
        logger.warning("Embedding computation failed: %s", exc)
        return None


def compute_and_store_embeddings(
    db,
    session_ids: List[str],
    batch_size: int = 32,
) -> int:
    """Compute and store embeddings for the given session IDs.

    Processes sessions in batches to keep memory usage bounded.
    Silently skips sessions with no embeddable text.
    Returns the number of embeddings successfully stored.
    """
    fn = get_embed_fn()
    model = get_model_name()
    if fn is None or model is None:
        return 0

    stored = 0
    # Build (session_id, text) pairs
    pairs: List[tuple[str, str]] = []
    for sid in session_ids:
        try:
            text = db.get_session_text_for_embedding(sid)
            if text:
                pairs.append((sid, text))
        except Exception as exc:
            logger.debug("Could not get text for session %s: %s", sid, exc)

    # Process in batches
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        texts = [p[1] for p in batch]
        try:
            vecs = fn(texts)
        except Exception as exc:
            logger.warning("Batch embedding failed (offset %d): %s", i, exc)
            continue

        for (sid, text), vec in zip(batch, vecs):
            try:
                db.store_session_embedding(sid, _to_bytes(vec), model, text)
                stored += 1
            except Exception as exc:
                logger.debug("Failed to store embedding for %s: %s", sid, exc)

    return stored


# ---------------------------------------------------------------------------
# Semantic search entry point
# ---------------------------------------------------------------------------

_MAX_EMBED_PER_SEARCH = 200   # cap on how many new sessions to embed per call
_SEARCH_TIMEOUT_S = 8.0       # don't let embedding block for more than N seconds


def semantic_search_sessions(
    db,
    query: str,
    exclude_sources: Optional[Sequence[str]] = None,
    exclude_session_ids: Optional[set] = None,
    limit: int = 10,
) -> List[str]:
    """
    Return root session IDs ranked by semantic similarity to *query*.

    Steps:
    1. Check embedding backend is available (fast path: return [] if not)
    2. Embed any sessions that haven't been embedded yet (up to _MAX_EMBED_PER_SEARCH)
    3. Embed the query
    4. Load all stored embeddings, compute cosine similarity, rank
    5. Return top *limit* session IDs (excluding current session lineage)

    This function never raises; on any error it returns [].
    """
    fn = get_embed_fn()
    model = get_model_name()
    if fn is None or model is None:
        return []

    try:
        t0 = time.monotonic()
        exclude_src = list(exclude_sources) if exclude_sources else []

        # --- Step 1: Embed missing sessions (lazily, capped) ---
        try:
            missing = db.get_sessions_without_embeddings(
                exclude_sources=exclude_src,
                limit=_MAX_EMBED_PER_SEARCH,
            )
        except Exception as exc:
            logger.debug("Could not query missing embeddings: %s", exc)
            missing = []

        if missing:
            elapsed = time.monotonic() - t0
            if elapsed < _SEARCH_TIMEOUT_S - 1.0:
                n = compute_and_store_embeddings(db, missing)
                if n:
                    logger.debug("semantic_search: embedded %d new sessions", n)

        # --- Step 2: Embed the query ---
        elapsed = time.monotonic() - t0
        if elapsed >= _SEARCH_TIMEOUT_S:
            logger.debug("semantic_search: timed out before query embedding")
            return []

        try:
            query_vecs = fn([query])
            if not query_vecs:
                return []
            query_vec = query_vecs[0]
        except Exception as exc:
            logger.warning("Query embedding failed: %s", exc)
            return []

        # --- Step 3: Load stored embeddings and rank ---
        try:
            rows = db.get_session_embeddings(exclude_sources=exclude_src or None)
        except Exception as exc:
            logger.debug("Could not load session embeddings: %s", exc)
            return []

        if not rows:
            return []

        # Filter to sessions using the same model and not excluded
        scored: List[tuple[float, str]] = []
        excluded = exclude_session_ids or set()
        for row in rows:
            sid = row["session_id"]
            if sid in excluded:
                continue
            if row.get("model") != model:
                continue
            try:
                vec = _from_bytes(row["embedding"])
                sim = cosine_similarity(query_vec, vec)
                scored.append((sim, sid))
            except Exception as exc:
                logger.debug("Could not score session %s: %s", sid, exc)

        # Sort descending by similarity
        scored.sort(reverse=True)
        return [sid for _, sid in scored[:limit]]

    except Exception as exc:
        logger.warning("semantic_search_sessions failed: %s", exc, exc_info=True)
        return []
