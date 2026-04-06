"""
Subagent result verification.

After a subagent returns, this module cross-checks its summary claims against
observable reality — primarily file existence on disk.  It also compares what
the summary claims against what the tool trace actually shows, flagging
summaries that describe file writes when no write tools appear in the trace.

Design principles:
- No LLM calls — filesystem checks and regex only
- No side effects — read-only (no command execution)
- Fast — completes in milliseconds
- Additive — produces a 'verification' dict; never mutates the result status
- The parent decides what to do with verification failures
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Tools whose presence in the trace means a file write was attempted
WRITE_TOOL_NAMES = frozenset(["write_file", "patch"])

# Tools whose presence means the agent ran shell commands
RUN_TOOL_NAMES = frozenset(["terminal", "run_command", "bash"])

# Verbs that signal the summary is claiming a file was created or modified
_WRITE_VERB_RE = re.compile(
    r"\b(?:creat(?:ed?|ing)|wrote|written|modif(?:ied|ying)|updat(?:ed?|ing)|"
    r"generat(?:ed?|ing)|sav(?:ed?|ing)|add(?:ed|ing)|implement(?:ed|ing))\b",
    re.IGNORECASE,
)

# A file path token: starts with ./ or / or a word char, contains a dot extension,
# does not look like a URL or semver.
_PATH_TOKEN_RE = re.compile(
    r"(?<![:/.\w])"              # not preceded by URL scheme chars, dot, or word chars
    r"((?:\.{0,2}/)?[\w./\-]+\.\w{1,8})"  # optional ./ or ../ prefix + path + extension
    r"(?![:\w])",                # not followed by : (URL) or more word chars
)

_IGNORE_RE = re.compile(
    r"(?:https?://|\.com\b|\.org\b|\.io\b|\.net\b|\d+\.\d+\.\d+)",
    re.IGNORECASE,
)

# Extensions that are plausibly source/config files (vs domain names, versions)
_FILE_EXTENSIONS = frozenset([
    "py", "js", "ts", "tsx", "jsx", "go", "rs", "java", "c", "cpp", "h",
    "json", "yaml", "yml", "toml", "txt", "md", "sh", "bash", "env",
    "html", "css", "scss", "sass", "sql", "rb", "php", "swift", "kt",
    "r", "jl", "lua", "ex", "exs", "tf", "hcl", "dockerfile", "lock",
    "cfg", "ini", "conf", "xml", "proto", "graphql",
])


def extract_claimed_paths(summary: str) -> List[str]:
    """
    Extract file paths from a subagent summary that appear to be claimed as
    created or modified.

    For each path-like token found by regex:
    - Accept if it contains a directory separator (strong structural signal)
    - Accept if a write verb appears within 120 characters before it in the text

    Intentionally does NOT split on '.' to avoid destroying file extensions.
    Returns deduplicated list in order of appearance.
    """
    if not summary:
        return []

    found: list[str] = []
    seen: set[str] = set()

    for m in _PATH_TOKEN_RE.finditer(summary):
        candidate = m.group(1)
        if _IGNORE_RE.search(candidate):
            continue
        ext = candidate.rsplit(".", 1)[-1].lower() if "." in candidate else ""
        if ext not in _FILE_EXTENSIONS:
            continue

        has_slash = "/" in candidate

        # Check for a write verb in the 120 characters leading up to this token
        ctx_start = max(0, m.start() - 120)
        has_verb = bool(_WRITE_VERB_RE.search(summary[ctx_start:m.start()]))

        if has_slash or has_verb:
            if candidate not in seen:
                seen.add(candidate)
                found.append(candidate)

    return found


def paths_from_trace(tool_trace: List[Dict[str, Any]]) -> List[str]:
    """
    Extract file paths that the subagent actually wrote, as recorded in the
    tool trace.  These are populated by _enrich_trace_entry() in delegate_tool.py
    at trace-build time from the raw tool call arguments.

    Returns deduplicated list preserving order of first write.
    """
    seen: set[str] = set()
    result: list[str] = []
    for entry in tool_trace:
        if entry.get("tool") not in WRITE_TOOL_NAMES:
            continue
        for path in entry.get("paths_written", []):
            if path and path not in seen:
                seen.add(path)
                result.append(path)
    return result


def verify_result(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run verification checks on a completed subagent result entry.

    Checks (in order of reliability):
    1. Trace-derived paths: files the trace shows were written → check existence
    2. Summary-derived paths: files the summary claims were written → check existence
    3. Hallucination signal: summary contains write verbs but trace shows zero
       write-tool calls (strong indicator the summary fabricated file writes)

    Returns:
    {
        "verified": bool,          # True iff all checks pass and no suspicion
        "checks": [                # Per-path results (deduped, trace first)
            {"path": str, "exists": bool, "source": "trace"|"summary"},
        ],
        "write_tools_used": int,   # Write tool call count from trace
        "run_tools_used": int,     # Terminal/command tool call count from trace
        "suspicious": bool,        # Summary claims writes but trace shows none
        "note": str,               # Human-readable one-line verdict
    }
    """
    summary = entry.get("summary") or ""
    tool_trace = entry.get("tool_trace") or []

    write_tools_used = sum(1 for t in tool_trace if t.get("tool") in WRITE_TOOL_NAMES)
    run_tools_used = sum(1 for t in tool_trace if t.get("tool") in RUN_TOOL_NAMES)

    # Collect paths — trace is authoritative, summary is secondary
    trace_paths = paths_from_trace(tool_trace)
    summary_paths = extract_claimed_paths(summary)

    # Merge: trace paths first, then summary-only paths
    seen: set[str] = set()
    all_checks: list[Dict[str, Any]] = []
    for path in trace_paths:
        if path not in seen:
            seen.add(path)
            all_checks.append({"path": path, "source": "trace"})
    for path in summary_paths:
        if path not in seen:
            seen.add(path)
            all_checks.append({"path": path, "source": "summary"})

    # Run existence checks
    for check in all_checks:
        check["exists"] = os.path.exists(check["path"])

    all_exist = all(c["exists"] for c in all_checks) if all_checks else True

    # Hallucination signal: write verbs in summary but no write tools in trace
    has_write_claims = bool(summary_paths) or bool(_WRITE_VERB_RE.search(summary))
    suspicious = has_write_claims and write_tools_used == 0

    verified = all_exist and not suspicious

    # Build note
    if suspicious:
        note = (
            f"Suspicious: summary describes file writes but trace shows "
            f"0 write-tool calls."
        )
    elif not all_checks:
        note = "No file claims to verify."
    elif not all_exist:
        missing = [c["path"] for c in all_checks if not c["exists"]]
        note = f"{len(missing)} claimed file(s) not found on disk: {', '.join(missing[:3])}"
        if len(missing) > 3:
            note += f" (+{len(missing) - 3} more)"
    else:
        note = f"All {len(all_checks)} claimed file(s) verified present on disk."

    return {
        "verified": verified,
        "checks": all_checks,
        "write_tools_used": write_tools_used,
        "run_tools_used": run_tools_used,
        "suspicious": suspicious,
        "note": note,
    }
