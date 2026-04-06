#!/usr/bin/env python3
"""
Tests for the subagent result verification module (tools/verification.py)
and the verification integration in delegate_task.

Run with: python3 -m pytest tests/tools/test_verification.py -v --override-ini="addopts=-m 'not integration'"
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.verification import (
    extract_claimed_paths,
    paths_from_trace,
    verify_result,
    WRITE_TOOL_NAMES,
    RUN_TOOL_NAMES,
)
from tools.delegate_tool import (
    _enrich_trace_entry,
    delegate_task,
    DELEGATE_TASK_SCHEMA,
)


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    return parent


# ---------------------------------------------------------------------------
# extract_claimed_paths
# ---------------------------------------------------------------------------

class TestExtractClaimedPaths(unittest.TestCase):

    def test_empty_summary(self):
        self.assertEqual(extract_claimed_paths(""), [])

    def test_none_handled(self):
        self.assertEqual(extract_claimed_paths(None), [])

    def test_write_verb_with_py_file(self):
        paths = extract_claimed_paths("I created tests/test_auth.py with full coverage.")
        self.assertIn("tests/test_auth.py", paths)

    def test_modified_with_path(self):
        paths = extract_claimed_paths("Modified src/utils/helpers.py to fix the bug.")
        self.assertIn("src/utils/helpers.py", paths)

    def test_no_verb_no_slash_ignored(self):
        # A bare filename with no verb and no slash should not be extracted
        paths = extract_claimed_paths("The file auth.py is important.")
        self.assertNotIn("auth.py", paths)

    def test_slash_path_without_verb_included(self):
        # A path with slash is reliable enough to include even without a write verb
        paths = extract_claimed_paths("See src/main.py for details.")
        self.assertIn("src/main.py", paths)

    def test_url_ignored(self):
        paths = extract_claimed_paths("Created https://example.com/api.json endpoint.")
        self.assertEqual(paths, [])

    def test_semver_ignored(self):
        paths = extract_claimed_paths("Updated version to 1.2.3 in the config.")
        self.assertEqual(paths, [])

    def test_multiple_paths_deduplicated(self):
        summary = "Created src/foo.py. Also wrote src/foo.py again."
        paths = extract_claimed_paths(summary)
        self.assertEqual(paths.count("src/foo.py"), 1)

    def test_multiple_distinct_paths(self):
        summary = "Created src/auth.py and wrote tests/test_auth.py."
        paths = extract_claimed_paths(summary)
        self.assertIn("src/auth.py", paths)
        self.assertIn("tests/test_auth.py", paths)

    def test_yaml_and_toml_extensions(self):
        paths = extract_claimed_paths("Updated config/settings.yaml and pyproject.toml.")
        self.assertIn("config/settings.yaml", paths)
        self.assertIn("pyproject.toml", paths)


# ---------------------------------------------------------------------------
# paths_from_trace
# ---------------------------------------------------------------------------

class TestPathsFromTrace(unittest.TestCase):

    def test_empty_trace(self):
        self.assertEqual(paths_from_trace([]), [])

    def test_non_write_tool_ignored(self):
        trace = [{"tool": "read_file", "paths_written": ["should_not_appear.py"]}]
        # read_file is not in WRITE_TOOL_NAMES so its paths_written (if any) won't appear
        # Actually we only call paths_from_trace on write tool entries — let's verify the filter
        self.assertEqual(paths_from_trace([{"tool": "read_file", "args_bytes": 10}]), [])

    def test_write_file_path_extracted(self):
        trace = [
            {"tool": "write_file", "args_bytes": 50, "paths_written": ["src/auth.py"]},
        ]
        self.assertEqual(paths_from_trace(trace), ["src/auth.py"])

    def test_patch_paths_extracted(self):
        trace = [
            {"tool": "patch", "args_bytes": 200, "paths_written": ["a.py", "b.py"]},
        ]
        self.assertEqual(paths_from_trace(trace), ["a.py", "b.py"])

    def test_deduplication(self):
        trace = [
            {"tool": "write_file", "args_bytes": 10, "paths_written": ["foo.py"]},
            {"tool": "patch", "args_bytes": 20, "paths_written": ["foo.py", "bar.py"]},
        ]
        result = paths_from_trace(trace)
        self.assertEqual(result.count("foo.py"), 1)
        self.assertIn("bar.py", result)

    def test_entry_without_paths_written(self):
        trace = [{"tool": "write_file", "args_bytes": 10}]
        self.assertEqual(paths_from_trace(trace), [])


# ---------------------------------------------------------------------------
# _enrich_trace_entry
# ---------------------------------------------------------------------------

class TestEnrichTraceEntry(unittest.TestCase):

    def test_write_file_adds_paths_written(self):
        entry = {"tool": "write_file", "args_bytes": 40}
        _enrich_trace_entry(entry, json.dumps({"path": "src/foo.py", "content": "hello"}))
        self.assertEqual(entry["paths_written"], ["src/foo.py"])

    def test_patch_replace_mode_adds_path(self):
        entry = {"tool": "patch", "args_bytes": 50}
        _enrich_trace_entry(entry, json.dumps({"mode": "replace", "path": "src/bar.py", "old_string": "x", "new_string": "y"}))
        self.assertEqual(entry["paths_written"], ["src/bar.py"])

    def test_patch_v4a_mode_extracts_embedded_paths(self):
        patch_text = (
            "*** Begin Patch\n"
            "*** Update File: src/auth.py\n"
            "@@ login @@\n"
            "-old\n+new\n"
            "*** Add File: tests/test_auth.py\n"
            "+new test\n"
            "*** End Patch"
        )
        entry = {"tool": "patch", "args_bytes": len(patch_text)}
        _enrich_trace_entry(entry, json.dumps({"mode": "patch", "patch": patch_text}))
        self.assertIn("src/auth.py", entry["paths_written"])
        self.assertIn("tests/test_auth.py", entry["paths_written"])

    def test_non_write_tool_not_enriched(self):
        entry = {"tool": "terminal", "args_bytes": 10}
        _enrich_trace_entry(entry, json.dumps({"cmd": "ls"}))
        self.assertNotIn("paths_written", entry)

    def test_invalid_json_does_not_raise(self):
        entry = {"tool": "write_file", "args_bytes": 5}
        _enrich_trace_entry(entry, "not-json{{{")
        self.assertNotIn("paths_written", entry)

    def test_empty_args_does_not_raise(self):
        entry = {"tool": "write_file", "args_bytes": 0}
        _enrich_trace_entry(entry, "")
        self.assertNotIn("paths_written", entry)


# ---------------------------------------------------------------------------
# verify_result
# ---------------------------------------------------------------------------

class TestVerifyResult(unittest.TestCase):

    def _entry(self, summary="", tool_trace=None):
        return {
            "status": "completed",
            "summary": summary,
            "tool_trace": tool_trace or [],
        }

    def test_no_claims_verified_true(self):
        result = verify_result(self._entry("Task complete. No files changed."))
        self.assertTrue(result["verified"])
        self.assertEqual(result["checks"], [])
        self.assertFalse(result["suspicious"])

    def test_file_exists_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name
        try:
            entry = self._entry(
                summary=f"Created {path}.",
                tool_trace=[{"tool": "write_file", "args_bytes": 10, "paths_written": [path]}],
            )
            result = verify_result(entry)
            self.assertTrue(result["verified"])
            checks = {c["path"]: c for c in result["checks"]}
            self.assertIn(path, checks)
            self.assertTrue(checks[path]["exists"])
        finally:
            os.unlink(path)

    def test_missing_file_fails(self):
        path = "/tmp/this_file_does_not_exist_ever_12345.py"
        entry = self._entry(
            summary=f"Created {path}.",
            tool_trace=[{"tool": "write_file", "args_bytes": 10, "paths_written": [path]}],
        )
        result = verify_result(entry)
        self.assertFalse(result["verified"])
        checks = {c["path"]: c for c in result["checks"]}
        self.assertIn(path, checks)
        self.assertFalse(checks[path]["exists"])
        self.assertIn(path, result["note"])

    def test_suspicious_write_claims_no_trace_tools(self):
        entry = self._entry(
            summary="I created src/auth.py with the new login logic.",
            tool_trace=[{"tool": "read_file", "args_bytes": 10}],  # only reads
        )
        result = verify_result(entry)
        self.assertTrue(result["suspicious"])
        self.assertFalse(result["verified"])
        self.assertIn("suspicious", result["note"].lower())

    def test_not_suspicious_when_write_tool_used(self):
        path = "/tmp/verify_test_exists_abc.py"
        try:
            with open(path, "w") as f:
                f.write("# test")
            entry = self._entry(
                summary=f"I created {path}.",
                tool_trace=[{"tool": "write_file", "args_bytes": 30, "paths_written": [path]}],
            )
            result = verify_result(entry)
            self.assertFalse(result["suspicious"])
            self.assertEqual(result["write_tools_used"], 1)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_trace_source_preferred_over_summary_source(self):
        """Paths from the trace are labeled 'trace'; summary-only paths are 'summary'."""
        path = "/tmp/trace_sourced_abc123.py"
        try:
            with open(path, "w") as f:
                f.write("x")
            entry = self._entry(
                summary=f"Wrote {path} and also created src/other.py.",
                tool_trace=[{"tool": "write_file", "args_bytes": 20, "paths_written": [path]}],
            )
            result = verify_result(entry)
            sources = {c["path"]: c["source"] for c in result["checks"]}
            self.assertEqual(sources[path], "trace")
            # src/other.py is only in the summary (no slash-relative path used here)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_run_tools_counted(self):
        entry = self._entry(
            summary="Ran tests.",
            tool_trace=[
                {"tool": "terminal", "args_bytes": 20},
                {"tool": "terminal", "args_bytes": 15},
            ],
        )
        result = verify_result(entry)
        self.assertEqual(result["run_tools_used"], 2)

    def test_write_tools_counted(self):
        entry = self._entry(
            summary="Done.",
            tool_trace=[
                {"tool": "write_file", "args_bytes": 50, "paths_written": []},
                {"tool": "patch", "args_bytes": 80, "paths_written": []},
            ],
        )
        result = verify_result(entry)
        self.assertEqual(result["write_tools_used"], 2)

    def test_output_keys_always_present(self):
        result = verify_result(self._entry("All done."))
        for key in ("verified", "checks", "write_tools_used", "run_tools_used", "suspicious", "note"):
            self.assertIn(key, result)


# ---------------------------------------------------------------------------
# Integration: verify= in delegate_task
# ---------------------------------------------------------------------------

class TestVerifyIntegration(unittest.TestCase):

    def test_verify_in_schema(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("verify", props)
        self.assertEqual(props["verify"]["type"], "boolean")

    def test_verify_in_task_item_schema(self):
        task_item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        self.assertIn("verify", task_item["properties"])

    @patch("tools.delegate_tool._run_single_child")
    def test_verify_false_no_verification_key(self, mock_run):
        """With verify=False (default), results have no 'verification' key."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0,
            "tool_trace": [],
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Do stuff", parent_agent=parent))
        self.assertNotIn("verification", result["results"][0])

    @patch("tools.delegate_tool._run_single_child")
    def test_verify_true_adds_verification_key(self, mock_run):
        """With verify=True, completed results include 'verification' dict."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Task complete with no file changes.",
            "api_calls": 1, "duration_seconds": 1.0,
            "tool_trace": [],
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Do stuff", verify=True, parent_agent=parent))
        entry = result["results"][0]
        self.assertIn("verification", entry)
        v = entry["verification"]
        self.assertIn("verified", v)
        self.assertIn("note", v)

    @patch("tools.delegate_tool._run_single_child")
    def test_verify_skipped_for_failed_tasks(self, mock_run):
        """Verification only runs for status='completed', not 'failed' or 'error'."""
        mock_run.return_value = {
            "task_index": 0, "status": "failed",
            "summary": "", "api_calls": 1, "duration_seconds": 1.0,
            "tool_trace": [],
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Do stuff", verify=True, parent_agent=parent))
        self.assertNotIn("verification", result["results"][0])

    @patch("tools.delegate_tool._run_single_child")
    def test_per_task_verify_overrides_toplevel(self, mock_run):
        """Per-task verify=True overrides top-level verify=False."""
        def side_effect(**kwargs):
            return {
                "task_index": kwargs.get("task_index", 0), "status": "completed",
                "summary": "No file changes.", "api_calls": 1, "duration_seconds": 1.0,
                "tool_trace": [],
            }
        mock_run.side_effect = side_effect
        parent = _make_mock_parent()
        result = json.loads(delegate_task(
            tasks=[
                {"goal": "Task A", "verify": True},
                {"goal": "Task B"},
            ],
            verify=False,
            parent_agent=parent,
        ))
        results = {r["task_index"]: r for r in result["results"]}
        self.assertIn("verification", results[0])    # task A has verify=True
        self.assertNotIn("verification", results[1]) # task B inherits top-level False

    @patch("tools.delegate_tool._run_single_child")
    def test_hallucination_signal_surfaced(self, mock_run):
        """A summary with write verbs but empty trace triggers suspicious=True."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "I created src/auth.py and wrote tests/test_auth.py.",
            "api_calls": 3, "duration_seconds": 2.0,
            "tool_trace": [{"tool": "read_file", "args_bytes": 10}],
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Write auth code", verify=True, parent_agent=parent))
        v = result["results"][0]["verification"]
        self.assertTrue(v["suspicious"])
        self.assertFalse(v["verified"])


if __name__ == "__main__":
    unittest.main()
