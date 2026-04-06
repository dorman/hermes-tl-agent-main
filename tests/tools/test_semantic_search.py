#!/usr/bin/env python3
"""
Tests for semantic session search (tools/semantic_search.py and
the hermes_state.py embedding storage methods).

Uses an in-memory SQLite SessionDB so no real state.db is required.
Embedding backend is mocked so no ML library install is needed.

Run with:
  python3 -m pytest tests/tools/test_semantic_search.py -v \
    --override-ini="addopts=-m 'not integration'"
"""

import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_state import SessionDB
from tools.semantic_search import (
    _to_bytes,
    _from_bytes,
    cosine_similarity,
    reciprocal_rank_fusion,
    compute_and_store_embeddings,
    semantic_search_sessions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> SessionDB:
    """Create a fresh in-memory SessionDB (uses temp file for WAL compat)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return SessionDB(Path(tmp.name))


def _seed_session(db: SessionDB, session_id: str, title: str, user_msg: str, source: str = "cli") -> None:
    """Insert a minimal session + one user message."""
    db.create_session(session_id, source=source, model="test-model")
    db.set_session_title(session_id, title)
    db.append_message(session_id, role="user", content=user_msg)


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------

class TestVectorUtils(unittest.TestCase):

    def test_round_trip_bytes(self):
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = _to_bytes(vec)
        back = _from_bytes(blob)
        self.assertEqual(len(back), len(vec))
        for a, b in zip(vec, back):
            self.assertAlmostEqual(a, b, places=5)

    def test_cosine_identical(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_cosine_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0)

    def test_cosine_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0)

    def test_cosine_zero_norm(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        self.assertEqual(cosine_similarity(a, b), 0.0)

    def test_rrf_single_ranking(self):
        ranking = ["A", "B", "C"]
        merged = reciprocal_rank_fusion([ranking])
        self.assertEqual(merged, ranking)

    def test_rrf_two_rankings_merge(self):
        # A appears in both at rank 0 → highest RRF score
        r1 = ["A", "B", "C"]
        r2 = ["A", "D", "E"]
        merged = reciprocal_rank_fusion([r1, r2])
        self.assertEqual(merged[0], "A")

    def test_rrf_deduplication(self):
        r1 = ["A", "B"]
        r2 = ["A", "B"]
        merged = reciprocal_rank_fusion([r1, r2])
        self.assertEqual(len(merged), 2)
        self.assertNotIn("A", merged[1:])  # only once

    def test_rrf_empty_input(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])

    def test_rrf_preserves_items_not_in_all_lists(self):
        r1 = ["A", "B"]
        r2 = ["C", "D"]
        merged = reciprocal_rank_fusion([r1, r2])
        self.assertIn("A", merged)
        self.assertIn("C", merged)


# ---------------------------------------------------------------------------
# hermes_state embedding methods
# ---------------------------------------------------------------------------

class TestSessionDBEmbeddingMethods(unittest.TestCase):

    def setUp(self):
        self.db = _make_db()
        _seed_session(self.db, "sess-1", "CORS debugging", "I'm getting a CORS error on the API")
        _seed_session(self.db, "sess-2", "Docker setup", "How do I configure Docker networking?")
        _seed_session(self.db, "sess-3", "Python logging", "Can you help me set up logging in Python?")

    def tearDown(self):
        self.db.close()

    def test_get_session_text_includes_title(self):
        text = self.db.get_session_text_for_embedding("sess-1")
        self.assertIsNotNone(text)
        self.assertIn("CORS debugging", text)

    def test_get_session_text_includes_first_user_message(self):
        text = self.db.get_session_text_for_embedding("sess-1")
        self.assertIn("CORS error", text)

    def test_get_session_text_missing_session(self):
        text = self.db.get_session_text_for_embedding("nonexistent")
        self.assertIsNone(text)

    def test_store_and_retrieve_embedding(self):
        vec = [0.1] * 384
        blob = _to_bytes(vec)
        self.db.store_session_embedding("sess-1", blob, "test-model", "some text")
        rows = self.db.get_session_embeddings(["sess-1"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session_id"], "sess-1")
        self.assertEqual(rows[0]["model"], "test-model")
        recovered = _from_bytes(rows[0]["embedding"])
        self.assertAlmostEqual(recovered[0], 0.1, places=5)

    def test_store_embedding_upsert(self):
        blob1 = _to_bytes([0.1] * 384)
        blob2 = _to_bytes([0.9] * 384)
        self.db.store_session_embedding("sess-1", blob1, "model-a", "text")
        self.db.store_session_embedding("sess-1", blob2, "model-b", "text")
        rows = self.db.get_session_embeddings(["sess-1"])
        self.assertEqual(len(rows), 1)
        # Second upsert wins
        recovered = _from_bytes(rows[0]["embedding"])
        self.assertAlmostEqual(recovered[0], 0.9, places=5)

    def test_get_all_embeddings(self):
        for sid in ["sess-1", "sess-2"]:
            self.db.store_session_embedding(sid, _to_bytes([0.5] * 10), "m", "t")
        rows = self.db.get_session_embeddings()
        self.assertEqual(len(rows), 2)

    def test_get_embeddings_empty_list(self):
        rows = self.db.get_session_embeddings([])
        self.assertEqual(rows, [])

    def test_get_sessions_without_embeddings(self):
        # None embedded yet
        missing = self.db.get_sessions_without_embeddings()
        self.assertIn("sess-1", missing)
        self.assertIn("sess-2", missing)

        # Embed sess-1
        self.db.store_session_embedding("sess-1", _to_bytes([0.1]*10), "m", "t")
        missing_after = self.db.get_sessions_without_embeddings()
        self.assertNotIn("sess-1", missing_after)
        self.assertIn("sess-2", missing_after)

    def test_get_sessions_without_embeddings_excludes_sources(self):
        _seed_session(self.db, "sess-tool", "Tool session", "Internal work", source="tool")
        missing = self.db.get_sessions_without_embeddings(exclude_sources=["tool"])
        self.assertNotIn("sess-tool", missing)

    def test_get_all_root_session_ids(self):
        ids = self.db.get_all_root_session_ids()
        self.assertIn("sess-1", ids)
        self.assertIn("sess-2", ids)

    def test_schema_version_is_7(self):
        from hermes_state import SCHEMA_VERSION
        self.assertEqual(SCHEMA_VERSION, 7)


# ---------------------------------------------------------------------------
# compute_and_store_embeddings
# ---------------------------------------------------------------------------

class TestComputeAndStore(unittest.TestCase):

    def setUp(self):
        self.db = _make_db()
        _seed_session(self.db, "s1", "Auth bug", "JWT token keeps expiring")
        _seed_session(self.db, "s2", "Deployment", "How to deploy to AWS")

    def tearDown(self):
        self.db.close()

    def _mock_embed(self, model_name: str):
        """Return a patcher that makes the embed_fn return deterministic vectors."""
        dim = 4
        fake_vecs = {
            "Auth bug\nJWT token keeps expiring": [1.0, 0.0, 0.0, 0.0],
            "Deployment\nHow to deploy to AWS": [0.0, 1.0, 0.0, 0.0],
        }

        def _fn(texts):
            return [fake_vecs.get(t, [0.5, 0.5, 0.0, 0.0]) for t in texts]

        return (
            patch("tools.semantic_search._embed_fn", _fn),
            patch("tools.semantic_search._embed_model_name", model_name),
            patch("tools.semantic_search._embed_initialized", True),
        )

    def test_stores_embeddings_for_sessions(self):
        patches = self._mock_embed("test-model")
        with patches[0], patches[1], patches[2]:
            n = compute_and_store_embeddings(self.db, ["s1", "s2"])
        self.assertEqual(n, 2)
        rows = self.db.get_session_embeddings()
        stored_ids = {r["session_id"] for r in rows}
        self.assertIn("s1", stored_ids)
        self.assertIn("s2", stored_ids)

    def test_skips_sessions_with_no_text(self):
        # Empty session with no messages
        self.db.create_session("empty", source="cli", model="m")
        patches = self._mock_embed("test-model")
        with patches[0], patches[1], patches[2]:
            n = compute_and_store_embeddings(self.db, ["empty"])
        self.assertEqual(n, 0)

    def test_no_embed_fn_returns_zero(self):
        with patch("tools.semantic_search._embed_fn", None), \
             patch("tools.semantic_search._embed_initialized", True):
            n = compute_and_store_embeddings(self.db, ["s1"])
        self.assertEqual(n, 0)


# ---------------------------------------------------------------------------
# semantic_search_sessions
# ---------------------------------------------------------------------------

class TestSemanticSearchSessions(unittest.TestCase):

    def setUp(self):
        self.db = _make_db()
        _seed_session(self.db, "auth-sess", "Auth debugging", "JWT token expiry issue in production")
        _seed_session(self.db, "deploy-sess", "Deployment", "Docker compose networking setup")
        _seed_session(self.db, "cors-sess", "CORS fix", "Cross-origin request blocked on API")

        # Pre-load deterministic embeddings
        embeddings = {
            "auth-sess":   [1.0, 0.0, 0.0, 0.0],
            "deploy-sess": [0.0, 1.0, 0.0, 0.0],
            "cors-sess":   [0.0, 0.0, 1.0, 0.0],
        }
        for sid, vec in embeddings.items():
            self.db.store_session_embedding(sid, _to_bytes(vec), "test-model", "text")

        # Mock embed fn returns query-specific vectors
        self._query_vec = [1.0, 0.0, 0.0, 0.0]  # points toward auth-sess

    def tearDown(self):
        self.db.close()

    def _patch_embed(self, query_vec):
        def _fn(texts):
            return [query_vec for _ in texts]
        return (
            patch("tools.semantic_search._embed_fn", _fn),
            patch("tools.semantic_search._embed_model_name", "test-model"),
            patch("tools.semantic_search._embed_initialized", True),
        )

    def test_returns_most_similar_session_first(self):
        patches = self._patch_embed([1.0, 0.0, 0.0, 0.0])
        with patches[0], patches[1], patches[2]:
            results = semantic_search_sessions(self.db, "JWT authentication", limit=3)
        self.assertEqual(results[0], "auth-sess")

    def test_excludes_current_session(self):
        patches = self._patch_embed([1.0, 0.0, 0.0, 0.0])
        with patches[0], patches[1], patches[2]:
            results = semantic_search_sessions(
                self.db, "JWT", exclude_session_ids={"auth-sess"}, limit=3
            )
        self.assertNotIn("auth-sess", results)

    def test_returns_empty_when_no_backend(self):
        with patch("tools.semantic_search._embed_fn", None), \
             patch("tools.semantic_search._embed_initialized", True):
            results = semantic_search_sessions(self.db, "anything")
        self.assertEqual(results, [])

    def test_excludes_tool_source_sessions(self):
        _seed_session(self.db, "tool-sess", "Tool", "Internal tool session", source="tool")
        self.db.store_session_embedding("tool-sess", _to_bytes([1.0, 0.0, 0.0, 0.0]), "test-model", "t")
        patches = self._patch_embed([1.0, 0.0, 0.0, 0.0])
        with patches[0], patches[1], patches[2]:
            results = semantic_search_sessions(
                self.db, "JWT", exclude_sources=["tool"], limit=10
            )
        self.assertNotIn("tool-sess", results)

    def test_limit_respected(self):
        patches = self._patch_embed([0.5, 0.5, 0.0, 0.0])
        with patches[0], patches[1], patches[2]:
            results = semantic_search_sessions(self.db, "anything", limit=2)
        self.assertLessEqual(len(results), 2)

    def test_model_mismatch_excluded(self):
        # Store an embedding under a different model name — should be ignored
        self.db.store_session_embedding(
            "auth-sess", _to_bytes([1.0, 0.0, 0.0, 0.0]), "other-model", "t"
        )
        patches = self._patch_embed([1.0, 0.0, 0.0, 0.0])
        with patches[0], patches[1], patches[2]:
            results = semantic_search_sessions(self.db, "JWT", limit=3)
        # auth-sess has a "test-model" embedding stored in setUp and an "other-model" one from above.
        # The upsert replaced it with "other-model", so now it should be excluded from "test-model" results.
        self.assertNotIn("auth-sess", results)

    def test_never_raises(self):
        """semantic_search_sessions must not propagate exceptions."""
        with patch("tools.semantic_search._embed_fn", side_effect=RuntimeError("boom")), \
             patch("tools.semantic_search._embed_initialized", True):
            results = semantic_search_sessions(self.db, "query")
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion integration with session search
# ---------------------------------------------------------------------------

class TestRRFIntegration(unittest.TestCase):

    def test_semantic_only_session_appears_in_result(self):
        """A session found only by semantic (not FTS5) should still be returned."""
        fts_ranking = ["sess-a", "sess-b"]
        sem_ranking = ["sess-c", "sess-a"]
        merged = reciprocal_rank_fusion([fts_ranking, sem_ranking])
        self.assertIn("sess-c", merged)

    def test_session_in_both_ranks_higher(self):
        """A session in both FTS5 and semantic lists should rank above session in only one."""
        fts_ranking = ["sess-both", "sess-fts-only"]
        sem_ranking = ["sess-both", "sess-sem-only"]
        merged = reciprocal_rank_fusion([fts_ranking, sem_ranking])
        idx_both = merged.index("sess-both")
        idx_fts = merged.index("sess-fts-only")
        idx_sem = merged.index("sess-sem-only")
        self.assertLess(idx_both, idx_fts)
        self.assertLess(idx_both, idx_sem)


if __name__ == "__main__":
    unittest.main()
