"""
Shared memory — a session-scoped scratch space for parallel subagents.

Created per delegation call when the parent sets shared_memory=True.
All subagents in that batch share one thread-safe key-value store.
Entries are never persisted to disk and never touch MEMORY.md / USER.md.
The parent receives a snapshot of the final state alongside the results.

Lifecycle:
  delegate_task(shared_memory=True)
    └─ creates SharedMemory()
    └─ passes store to each child via _build_child_agent
    └─ _run_single_child: set_thread_shared_memory(store) before run_conversation
    └─ parallel children call shared_memory_write / shared_memory_read freely
    └─ after all children finish: store.snapshot() → result["shared_memory"]
    └─ SharedMemory discarded (GC)

Thread safety:
  All read/write/delete operations are protected by a threading.RLock.
  Multiple parallel subagents can safely call write concurrently.

Capacity limits (prevent abuse):
  MAX_ENTRIES  = 100 key-value pairs per delegation call
  MAX_KEY_LEN  = 64 characters
  MAX_VALUE_LEN = 2000 characters

Security:
  Values are scanned with the same injection-detection logic used by
  long-term memory (tools/memory_tool.py) before being accepted.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MAX_ENTRIES = 100
MAX_KEY_LEN = 64
MAX_VALUE_LEN = 2000


# ---------------------------------------------------------------------------
# Core store
# ---------------------------------------------------------------------------

class SharedMemory:
    """Thread-safe key-value scratch space for one delegation call."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def write(self, key: str, value: str) -> Dict[str, Any]:
        """Upsert a key-value entry. Returns {ok, key, entry_count} or error."""
        key = key.strip()
        value = value.strip()

        if not key:
            return {"ok": False, "error": "key cannot be empty"}
        if len(key) > MAX_KEY_LEN:
            return {"ok": False, "error": f"key too long ({len(key)} > {MAX_KEY_LEN} chars)"}
        if len(value) > MAX_VALUE_LEN:
            return {"ok": False, "error": f"value too long ({len(value)} > {MAX_VALUE_LEN} chars)"}

        # Re-use long-term memory's injection scanner
        from tools.memory_tool import _scan_memory_content
        scan_err = _scan_memory_content(value)
        if scan_err:
            return {"ok": False, "error": scan_err}

        with self._lock:
            is_new = key not in self._store
            if is_new and len(self._store) >= MAX_ENTRIES:
                return {
                    "ok": False,
                    "error": f"store full ({MAX_ENTRIES} entries max). Delete an entry first.",
                }
            self._store[key] = value
            action = "added" if is_new else "updated"
            return {"ok": True, "key": key, "action": action, "entry_count": len(self._store)}

    def delete(self, key: str) -> Dict[str, Any]:
        """Remove an entry by key. Returns {ok} or error."""
        key = key.strip()
        with self._lock:
            if key not in self._store:
                return {"ok": False, "error": f"key '{key}' not found", "keys": list(self._store)}
            del self._store[key]
            return {"ok": True, "key": key, "entry_count": len(self._store)}

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Read one key or all entries.

        Returns {ok, value} for a single key, or {ok, entries, entry_count}
        when key is None.
        """
        with self._lock:
            if key is not None:
                key = key.strip()
                if key not in self._store:
                    return {
                        "ok": False,
                        "error": f"key '{key}' not found",
                        "available_keys": list(self._store),
                    }
                return {"ok": True, "key": key, "value": self._store[key]}
            return {
                "ok": True,
                "entries": dict(self._store),
                "entry_count": len(self._store),
            }

    def snapshot(self) -> Dict[str, str]:
        """Return a point-in-time copy of the store (called by parent after children finish)."""
        with self._lock:
            return dict(self._store)


# ---------------------------------------------------------------------------
# Thread-local store registry
#
# Each child thread sets its own store before run_conversation() starts.
# Handlers read from this thread-local so parallel subagents never
# cross-contaminate each other's store references.
# ---------------------------------------------------------------------------

_tl = threading.local()


def set_thread_shared_memory(store: Optional[SharedMemory]) -> None:
    """Call from the child's thread before run_conversation() starts."""
    _tl.store = store


def get_thread_shared_memory() -> Optional[SharedMemory]:
    """Called by tool handlers to retrieve this thread's shared memory store."""
    return getattr(_tl, "store", None)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _handle_write(args: dict, **_kw) -> str:
    store = get_thread_shared_memory()
    if store is None:
        return json.dumps({
            "ok": False,
            "error": "Shared memory is not enabled for this delegation. "
                     "The parent must set shared_memory=true when calling delegate_task.",
        })
    result = store.write(
        key=str(args.get("key") or ""),
        value=str(args.get("value") or ""),
    )
    return json.dumps(result, ensure_ascii=False)


def _handle_read(args: dict, **_kw) -> str:
    store = get_thread_shared_memory()
    if store is None:
        return json.dumps({"ok": False, "error": "Shared memory is not enabled for this delegation."})
    result = store.read(key=args.get("key"))
    return json.dumps(result, ensure_ascii=False)


def _handle_delete(args: dict, **_kw) -> str:
    store = get_thread_shared_memory()
    if store is None:
        return json.dumps({"ok": False, "error": "Shared memory is not enabled for this delegation."})
    result = store.delete(key=str(args.get("key") or ""))
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SHARED_MEMORY_WRITE_SCHEMA = {
    "name": "shared_memory_write",
    "description": (
        "Write a fact or discovery to the shared scratch space so other subagents "
        "(and the parent) can read it. Use this for findings that are relevant to "
        "parallel workstreams — e.g. a root cause, a discovered file path, a "
        "decision that affects other tasks.\n\n"
        "Keys should be short snake_case identifiers (e.g. 'root_cause', "
        "'affected_endpoints'). Values are plain text up to 2000 characters. "
        "Writing the same key twice updates the value in place.\n\n"
        "This scratch space is session-scoped: it does not persist between "
        "delegation calls and does not write to long-term memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Short snake_case identifier (max 64 chars). E.g. 'cors_root_cause'.",
            },
            "value": {
                "type": "string",
                "description": "The fact or finding to share (max 2000 chars).",
            },
        },
        "required": ["key", "value"],
    },
}

SHARED_MEMORY_READ_SCHEMA = {
    "name": "shared_memory_read",
    "description": (
        "Read from the shared scratch space. Omit 'key' to see all current entries. "
        "Use this to check what other subagents have already discovered before "
        "duplicating work, or to build on their findings.\n\n"
        "Returns the value for a specific key, or a dict of all entries when no "
        "key is given. Returns an error with available keys if the key is not found."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key to retrieve. Omit to read all entries.",
            },
        },
        "required": [],
    },
}

SHARED_MEMORY_DELETE_SCHEMA = {
    "name": "shared_memory_delete",
    "description": (
        "Remove an entry from the shared scratch space. Use this when a finding "
        "turns out to be wrong or no longer relevant, so other subagents don't act "
        "on stale information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key of the entry to delete.",
            },
        },
        "required": ["key"],
    },
}

SHARED_MEMORY_TOOL_NAMES = frozenset([
    "shared_memory_write",
    "shared_memory_read",
    "shared_memory_delete",
])

# ---------------------------------------------------------------------------
# Registry — toolset "shared_memory" is in BLOCKED_TOOLSET_NAMES so the parent
# never sees these tools in its prompt. They're injected into children's
# valid_tool_names explicitly in _build_child_agent when shared_memory=True.
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402

registry.register(
    name="shared_memory_write",
    toolset="shared_memory",
    schema=SHARED_MEMORY_WRITE_SCHEMA,
    handler=_handle_write,
    emoji="📝",
)

registry.register(
    name="shared_memory_read",
    toolset="shared_memory",
    schema=SHARED_MEMORY_READ_SCHEMA,
    handler=_handle_read,
    emoji="📋",
)

registry.register(
    name="shared_memory_delete",
    toolset="shared_memory",
    schema=SHARED_MEMORY_DELETE_SCHEMA,
    handler=_handle_delete,
    emoji="🗑️",
)
