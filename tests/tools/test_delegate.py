#!/usr/bin/env python3
"""
Tests for the subagent delegation tool.

Uses mock AIAgent instances to test the delegation logic without
requiring API keys or real LLM calls.

Run with:  python -m pytest tests/test_delegate.py -v
   or:     python tests/test_delegate.py
"""

import json
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_BLOCKED_TOOLS,
    DELEGATE_TASK_SCHEMA,
    MAX_CONCURRENT_CHILDREN,
    MAX_DEPTH,
    check_delegate_requirements,
    delegate_task,
    _build_child_agent,
    _build_child_system_prompt,
    _strip_blocked_tools,
    _resolve_delegation_credentials,
)


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
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


class TestDelegateRequirements(unittest.TestCase):
    def test_always_available(self):
        self.assertTrue(check_delegate_requirements())

    def test_schema_valid(self):
        self.assertEqual(DELEGATE_TASK_SCHEMA["name"], "delegate_task")
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("goal", props)
        self.assertIn("tasks", props)
        self.assertIn("context", props)
        self.assertIn("toolsets", props)
        self.assertIn("max_iterations", props)
        self.assertEqual(props["tasks"]["maxItems"], 3)


class TestChildSystemPrompt(unittest.TestCase):
    def test_goal_only(self):
        prompt = _build_child_system_prompt("Fix the tests")
        self.assertIn("Fix the tests", prompt)
        self.assertIn("YOUR TASK", prompt)
        self.assertNotIn("CONTEXT", prompt)

    def test_goal_with_context(self):
        prompt = _build_child_system_prompt("Fix the tests", "Error: assertion failed in test_foo.py line 42")
        self.assertIn("Fix the tests", prompt)
        self.assertIn("CONTEXT", prompt)
        self.assertIn("assertion failed", prompt)

    def test_empty_context_ignored(self):
        prompt = _build_child_system_prompt("Do something", "  ")
        self.assertNotIn("CONTEXT", prompt)


class TestStripBlockedTools(unittest.TestCase):
    def test_removes_blocked_toolsets(self):
        result = _strip_blocked_tools(["terminal", "file", "delegation", "clarify", "memory", "code_execution"])
        self.assertEqual(sorted(result), ["file", "terminal"])

    def test_preserves_allowed_toolsets(self):
        result = _strip_blocked_tools(["terminal", "file", "web", "browser"])
        self.assertEqual(sorted(result), ["browser", "file", "terminal", "web"])

    def test_empty_input(self):
        result = _strip_blocked_tools([])
        self.assertEqual(result, [])


class TestDelegateTask(unittest.TestCase):
    def test_no_parent_agent(self):
        result = json.loads(delegate_task(goal="test"))
        self.assertIn("error", result)
        self.assertIn("parent agent", result["error"])

    def test_depth_limit(self):
        parent = _make_mock_parent(depth=2)
        result = json.loads(delegate_task(goal="test", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("depth limit", result["error"].lower())

    def test_no_goal_or_tasks(self):
        parent = _make_mock_parent()
        result = json.loads(delegate_task(parent_agent=parent))
        self.assertIn("error", result)

    def test_empty_goal(self):
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="  ", parent_agent=parent))
        self.assertIn("error", result)

    def test_task_missing_goal(self):
        parent = _make_mock_parent()
        result = json.loads(delegate_task(tasks=[{"context": "no goal here"}], parent_agent=parent))
        self.assertIn("error", result)

    @patch("tools.delegate_tool._run_single_child")
    def test_single_task_mode(self, mock_run):
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done!", "api_calls": 3, "duration_seconds": 5.0
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Fix tests", context="error log...", parent_agent=parent))
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["status"], "completed")
        self.assertEqual(result["results"][0]["summary"], "Done!")
        mock_run.assert_called_once()

    @patch("tools.delegate_tool._run_single_child")
    def test_batch_mode(self, mock_run):
        mock_run.side_effect = [
            {"task_index": 0, "status": "completed", "summary": "Result A", "api_calls": 2, "duration_seconds": 3.0},
            {"task_index": 1, "status": "completed", "summary": "Result B", "api_calls": 4, "duration_seconds": 6.0},
        ]
        parent = _make_mock_parent()
        tasks = [
            {"goal": "Research topic A"},
            {"goal": "Research topic B"},
        ]
        result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["summary"], "Result A")
        self.assertEqual(result["results"][1]["summary"], "Result B")
        self.assertIn("total_duration_seconds", result)

    @patch("tools.delegate_tool._run_single_child")
    def test_batch_capped_at_3(self, mock_run):
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
        }
        parent = _make_mock_parent()
        tasks = [{"goal": f"Task {i}"} for i in range(5)]
        result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))
        # Should only run 3 tasks (MAX_CONCURRENT_CHILDREN)
        self.assertEqual(mock_run.call_count, 3)

    @patch("tools.delegate_tool._run_single_child")
    def test_batch_ignores_toplevel_goal(self, mock_run):
        """When tasks array is provided, top-level goal/context/toolsets are ignored."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(
            goal="This should be ignored",
            tasks=[{"goal": "Actual task"}],
            parent_agent=parent,
        ))
        # The mock was called with the tasks array item, not the top-level goal
        call_args = mock_run.call_args
        self.assertEqual(call_args.kwargs.get("goal") or call_args[1].get("goal", call_args[0][1] if len(call_args[0]) > 1 else None), "Actual task")

    @patch("tools.delegate_tool._run_single_child")
    def test_failed_child_included_in_results(self, mock_run):
        mock_run.return_value = {
            "task_index": 0, "status": "error",
            "summary": None, "error": "Something broke",
            "api_calls": 0, "duration_seconds": 0.5
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Break things", parent_agent=parent))
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertIn("Something broke", result["results"][0]["error"])

    def test_depth_increments(self):
        """Verify child gets parent's depth + 1."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test depth", parent_agent=parent)
            self.assertEqual(mock_child._delegate_depth, 1)

    def test_active_children_tracking(self):
        """Verify children are registered/unregistered for interrupt propagation."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test tracking", parent_agent=parent)
            self.assertEqual(len(parent._active_children), 0)

    def test_child_inherits_runtime_credentials(self):
        parent = _make_mock_parent(depth=0)
        parent.base_url = "https://chatgpt.com/backend-api/codex"
        parent.api_key = "codex-token"
        parent.provider = "openai-codex"
        parent.api_mode = "codex_responses"

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test runtime inheritance", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["base_url"], parent.base_url)
            self.assertEqual(kwargs["api_key"], parent.api_key)
            self.assertEqual(kwargs["provider"], parent.provider)
            self.assertEqual(kwargs["api_mode"], parent.api_mode)


class TestToolNamePreservation(unittest.TestCase):
    """Verify _last_resolved_tool_names is restored after subagent runs."""

    def test_global_tool_names_restored_after_delegation(self):
        """The process-global _last_resolved_tool_names must be restored
        after a subagent completes so the parent's execute_code sandbox
        generates correct imports."""
        import model_tools

        parent = _make_mock_parent(depth=0)
        original_tools = ["terminal", "read_file", "web_search", "execute_code", "delegate_task"]
        model_tools._last_resolved_tool_names = list(original_tools)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test tool preservation", parent_agent=parent)

        self.assertEqual(model_tools._last_resolved_tool_names, original_tools)

    def test_global_tool_names_restored_after_child_failure(self):
        """Even when the child agent raises, the global must be restored."""
        import model_tools

        parent = _make_mock_parent(depth=0)
        original_tools = ["terminal", "read_file", "web_search"]
        model_tools._last_resolved_tool_names = list(original_tools)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.side_effect = RuntimeError("boom")
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Crash test", parent_agent=parent))
            self.assertEqual(result["results"][0]["status"], "error")

        self.assertEqual(model_tools._last_resolved_tool_names, original_tools)

    def test_build_child_agent_does_not_raise_name_error(self):
        """Regression: _build_child_agent must not reference _saved_tool_names.

        The bug introduced by the e7844e9c merge conflict: line 235 inside
        _build_child_agent read `list(_saved_tool_names)` where that variable
        is only defined later in _run_single_child.  Calling _build_child_agent
        standalone (without _run_single_child's scope) must never raise NameError.
        """
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent"):
            try:
                _build_child_agent(
                    task_index=0,
                    goal="regression check",
                    context=None,
                    toolsets=None,
                    model=None,
                    max_iterations=10,
                    parent_agent=parent,
                )
            except NameError as exc:
                self.fail(
                    f"_build_child_agent raised NameError — "
                    f"_saved_tool_names leaked back into wrong scope: {exc}"
                )

    def test_saved_tool_names_set_on_child_before_run(self):
        """_run_single_child must set _delegate_saved_tool_names on the child
        from model_tools._last_resolved_tool_names before run_conversation."""
        import model_tools

        parent = _make_mock_parent(depth=0)
        expected_tools = ["read_file", "web_search", "execute_code"]
        model_tools._last_resolved_tool_names = list(expected_tools)

        captured = {}

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()

            def capture_and_return(user_message):
                captured["saved"] = list(mock_child._delegate_saved_tool_names)
                return {"final_response": "ok", "completed": True, "api_calls": 1}

            mock_child.run_conversation.side_effect = capture_and_return
            MockAgent.return_value = mock_child

            delegate_task(goal="capture test", parent_agent=parent)

        self.assertEqual(captured["saved"], expected_tools)


class TestDelegateObservability(unittest.TestCase):
    """Tests for enriched metadata returned by _run_single_child."""

    def test_observability_fields_present(self):
        """Completed child should return tool_trace, tokens, model, exit_reason."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 5000
            mock_child.session_completion_tokens = 1200
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 3,
                "messages": [
                    {"role": "user", "content": "do something"},
                    {"role": "assistant", "tool_calls": [
                        {"id": "tc_1", "function": {"name": "web_search", "arguments": '{"query": "test"}'}}
                    ]},
                    {"role": "tool", "tool_call_id": "tc_1", "content": '{"results": [1,2,3]}'},
                    {"role": "assistant", "content": "done"},
                ],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test observability", parent_agent=parent))
            entry = result["results"][0]

            # Core observability fields
            self.assertEqual(entry["model"], "claude-sonnet-4-6")
            self.assertEqual(entry["exit_reason"], "completed")
            self.assertEqual(entry["tokens"]["input"], 5000)
            self.assertEqual(entry["tokens"]["output"], 1200)

            # Tool trace
            self.assertEqual(len(entry["tool_trace"]), 1)
            self.assertEqual(entry["tool_trace"][0]["tool"], "web_search")
            self.assertIn("args_bytes", entry["tool_trace"][0])
            self.assertIn("result_bytes", entry["tool_trace"][0])
            self.assertEqual(entry["tool_trace"][0]["status"], "ok")

    def test_tool_trace_detects_error(self):
        """Tool results containing 'error' should be marked as error status."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 0
            mock_child.session_completion_tokens = 0
            mock_child.run_conversation.return_value = {
                "final_response": "failed",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [
                    {"role": "assistant", "tool_calls": [
                        {"id": "tc_1", "function": {"name": "terminal", "arguments": '{"cmd": "ls"}'}}
                    ]},
                    {"role": "tool", "tool_call_id": "tc_1", "content": "Error: command not found"},
                ],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test error trace", parent_agent=parent))
            trace = result["results"][0]["tool_trace"]
            self.assertEqual(trace[0]["status"], "error")

    def test_parallel_tool_calls_paired_correctly(self):
        """Parallel tool calls should each get their own result via tool_call_id matching."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 3000
            mock_child.session_completion_tokens = 800
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [
                    {"role": "assistant", "tool_calls": [
                        {"id": "tc_a", "function": {"name": "web_search", "arguments": '{"q": "a"}'}},
                        {"id": "tc_b", "function": {"name": "web_search", "arguments": '{"q": "b"}'}},
                        {"id": "tc_c", "function": {"name": "terminal", "arguments": '{"cmd": "ls"}'}},
                    ]},
                    {"role": "tool", "tool_call_id": "tc_a", "content": '{"ok": true}'},
                    {"role": "tool", "tool_call_id": "tc_b", "content": "Error: rate limited"},
                    {"role": "tool", "tool_call_id": "tc_c", "content": "file1.txt\nfile2.txt"},
                    {"role": "assistant", "content": "done"},
                ],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test parallel", parent_agent=parent))
            trace = result["results"][0]["tool_trace"]

            # All three tool calls should have results
            self.assertEqual(len(trace), 3)

            # First: web_search → ok
            self.assertEqual(trace[0]["tool"], "web_search")
            self.assertEqual(trace[0]["status"], "ok")
            self.assertIn("result_bytes", trace[0])

            # Second: web_search → error
            self.assertEqual(trace[1]["tool"], "web_search")
            self.assertEqual(trace[1]["status"], "error")
            self.assertIn("result_bytes", trace[1])

            # Third: terminal → ok
            self.assertEqual(trace[2]["tool"], "terminal")
            self.assertEqual(trace[2]["status"], "ok")
            self.assertIn("result_bytes", trace[2])

    def test_exit_reason_interrupted(self):
        """Interrupted child should report exit_reason='interrupted'."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 0
            mock_child.session_completion_tokens = 0
            mock_child.run_conversation.return_value = {
                "final_response": "",
                "completed": False,
                "interrupted": True,
                "api_calls": 2,
                "messages": [],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test interrupt", parent_agent=parent))
            self.assertEqual(result["results"][0]["exit_reason"], "interrupted")

    def test_exit_reason_max_iterations(self):
        """Child that didn't complete and wasn't interrupted hit max_iterations."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 0
            mock_child.session_completion_tokens = 0
            mock_child.run_conversation.return_value = {
                "final_response": "",
                "completed": False,
                "interrupted": False,
                "api_calls": 50,
                "messages": [],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test max iter", parent_agent=parent))
            self.assertEqual(result["results"][0]["exit_reason"], "max_iterations")


class TestBlockedTools(unittest.TestCase):
    def test_blocked_tools_constant(self):
        for tool in ["delegate_task", "clarify", "memory", "send_message", "execute_code"]:
            self.assertIn(tool, DELEGATE_BLOCKED_TOOLS)

    def test_constants(self):
        self.assertEqual(MAX_CONCURRENT_CHILDREN, 3)
        self.assertEqual(MAX_DEPTH, 2)


class TestDelegationCredentialResolution(unittest.TestCase):
    """Tests for provider:model credential resolution in delegation config."""

    def test_no_provider_returns_none_credentials(self):
        """When delegation.provider is empty, all credentials are None (inherit parent)."""
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "", "provider": ""}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertIsNone(creds["provider"])
        self.assertIsNone(creds["base_url"])
        self.assertIsNone(creds["api_key"])
        self.assertIsNone(creds["api_mode"])
        self.assertIsNone(creds["model"])

    def test_model_only_no_provider(self):
        """When only model is set (no provider), model is returned but credentials are None."""
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "google/gemini-3-flash-preview", "provider": ""}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "google/gemini-3-flash-preview")
        self.assertIsNone(creds["provider"])
        self.assertIsNone(creds["base_url"])
        self.assertIsNone(creds["api_key"])

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_provider_resolves_full_credentials(self, mock_resolve):
        """When delegation.provider is set, full credentials are resolved."""
        mock_resolve.return_value = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-test-key",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "google/gemini-3-flash-preview", "provider": "openrouter"}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "google/gemini-3-flash-preview")
        self.assertEqual(creds["provider"], "openrouter")
        self.assertEqual(creds["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(creds["api_key"], "sk-or-test-key")
        self.assertEqual(creds["api_mode"], "chat_completions")
        mock_resolve.assert_called_once_with(requested="openrouter")

    def test_direct_endpoint_uses_configured_base_url_and_api_key(self):
        parent = _make_mock_parent(depth=0)
        cfg = {
            "model": "qwen2.5-coder",
            "provider": "openrouter",
            "base_url": "http://localhost:1234/v1",
            "api_key": "local-key",
        }
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "qwen2.5-coder")
        self.assertEqual(creds["provider"], "custom")
        self.assertEqual(creds["base_url"], "http://localhost:1234/v1")
        self.assertEqual(creds["api_key"], "local-key")
        self.assertEqual(creds["api_mode"], "chat_completions")

    def test_direct_endpoint_falls_back_to_openai_api_key_env(self):
        parent = _make_mock_parent(depth=0)
        cfg = {
            "model": "qwen2.5-coder",
            "base_url": "http://localhost:1234/v1",
        }
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}, clear=False):
            creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["api_key"], "env-openai-key")
        self.assertEqual(creds["provider"], "custom")

    def test_direct_endpoint_does_not_fall_back_to_openrouter_api_key_env(self):
        parent = _make_mock_parent(depth=0)
        cfg = {
            "model": "qwen2.5-coder",
            "base_url": "http://localhost:1234/v1",
        }
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "env-openrouter-key",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        ):
            with self.assertRaises(ValueError) as ctx:
                _resolve_delegation_credentials(cfg, parent)
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_nous_provider_resolves_nous_credentials(self, mock_resolve):
        """Nous provider resolves Nous Portal base_url and api_key."""
        mock_resolve.return_value = {
            "provider": "nous",
            "base_url": "https://inference-api.nousresearch.com/v1",
            "api_key": "nous-agent-key-xyz",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "hermes-3-llama-3.1-8b", "provider": "nous"}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["provider"], "nous")
        self.assertEqual(creds["base_url"], "https://inference-api.nousresearch.com/v1")
        self.assertEqual(creds["api_key"], "nous-agent-key-xyz")
        mock_resolve.assert_called_once_with(requested="nous")

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_provider_resolution_failure_raises_valueerror(self, mock_resolve):
        """When provider resolution fails, ValueError is raised with helpful message."""
        mock_resolve.side_effect = RuntimeError("OPENROUTER_API_KEY not set")
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "some-model", "provider": "openrouter"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_delegation_credentials(cfg, parent)
        self.assertIn("openrouter", str(ctx.exception).lower())
        self.assertIn("Cannot resolve", str(ctx.exception))

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_provider_resolves_but_no_api_key_raises(self, mock_resolve):
        """When provider resolves but has no API key, ValueError is raised."""
        mock_resolve.return_value = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "some-model", "provider": "openrouter"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_delegation_credentials(cfg, parent)
        self.assertIn("no API key", str(ctx.exception))

    def test_missing_config_keys_inherit_parent(self):
        """When config dict has no model/provider keys at all, inherits parent."""
        parent = _make_mock_parent(depth=0)
        cfg = {"max_iterations": 45}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertIsNone(creds["model"])
        self.assertIsNone(creds["provider"])


class TestDelegationProviderIntegration(unittest.TestCase):
    """Integration tests: delegation config → _run_single_child → AIAgent construction."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_config_provider_credentials_reach_child_agent(self, mock_creds, mock_cfg):
        """When delegation.provider is configured, child agent gets resolved credentials."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "google/gemini-3-flash-preview",
            "provider": "openrouter",
        }
        mock_creds.return_value = {
            "model": "google/gemini-3-flash-preview",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-delegation-key",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test provider routing", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["model"], "google/gemini-3-flash-preview")
            self.assertEqual(kwargs["provider"], "openrouter")
            self.assertEqual(kwargs["base_url"], "https://openrouter.ai/api/v1")
            self.assertEqual(kwargs["api_key"], "sk-or-delegation-key")
            self.assertEqual(kwargs["api_mode"], "chat_completions")

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_cross_provider_delegation(self, mock_creds, mock_cfg):
        """Parent on Nous, subagent on OpenRouter — full credential switch."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "google/gemini-3-flash-preview",
            "provider": "openrouter",
        }
        mock_creds.return_value = {
            "model": "google/gemini-3-flash-preview",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-key",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        parent.provider = "nous"
        parent.base_url = "https://inference-api.nousresearch.com/v1"
        parent.api_key = "nous-key-abc"

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Cross-provider test", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            # Child should use OpenRouter, NOT Nous
            self.assertEqual(kwargs["provider"], "openrouter")
            self.assertEqual(kwargs["base_url"], "https://openrouter.ai/api/v1")
            self.assertEqual(kwargs["api_key"], "sk-or-key")
            self.assertNotEqual(kwargs["base_url"], parent.base_url)
            self.assertNotEqual(kwargs["api_key"], parent.api_key)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_direct_endpoint_credentials_reach_child_agent(self, mock_creds, mock_cfg):
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "qwen2.5-coder",
            "base_url": "http://localhost:1234/v1",
            "api_key": "local-key",
        }
        mock_creds.return_value = {
            "model": "qwen2.5-coder",
            "provider": "custom",
            "base_url": "http://localhost:1234/v1",
            "api_key": "local-key",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Direct endpoint test", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["model"], "qwen2.5-coder")
            self.assertEqual(kwargs["provider"], "custom")
            self.assertEqual(kwargs["base_url"], "http://localhost:1234/v1")
            self.assertEqual(kwargs["api_key"], "local-key")
            self.assertEqual(kwargs["api_mode"], "chat_completions")

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_empty_config_inherits_parent(self, mock_creds, mock_cfg):
        """When delegation config is empty, child inherits parent credentials."""
        mock_cfg.return_value = {"max_iterations": 45, "model": "", "provider": ""}
        mock_creds.return_value = {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test inherit", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["model"], parent.model)
            self.assertEqual(kwargs["provider"], parent.provider)
            self.assertEqual(kwargs["base_url"], parent.base_url)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_credential_error_returns_json_error(self, mock_creds, mock_cfg):
        """When credential resolution fails, delegate_task returns a JSON error."""
        mock_cfg.return_value = {"model": "bad-model", "provider": "nonexistent"}
        mock_creds.side_effect = ValueError(
            "Cannot resolve delegation provider 'nonexistent': Unknown provider"
        )
        parent = _make_mock_parent(depth=0)

        result = json.loads(delegate_task(goal="Should fail", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("Cannot resolve", result["error"])
        self.assertIn("nonexistent", result["error"])

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_batch_mode_all_children_get_credentials(self, mock_creds, mock_cfg):
        """In batch mode, all children receive the resolved credentials."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "meta-llama/llama-4-scout",
            "provider": "openrouter",
        }
        mock_creds.return_value = {
            "model": "meta-llama/llama-4-scout",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-batch",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)

        # Patch _build_child_agent since credentials are now passed there
        # (agents are built in the main thread before being handed to workers)
        with patch("tools.delegate_tool._build_child_agent") as mock_build, \
             patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_child = MagicMock()
            mock_build.return_value = mock_child
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
            }

            tasks = [{"goal": "Task A"}, {"goal": "Task B"}]
            delegate_task(tasks=tasks, parent_agent=parent)

            self.assertEqual(mock_build.call_count, 2)
            for call in mock_build.call_args_list:
                self.assertEqual(call.kwargs.get("model"), "meta-llama/llama-4-scout")
                self.assertEqual(call.kwargs.get("override_provider"), "openrouter")
                self.assertEqual(call.kwargs.get("override_base_url"), "https://openrouter.ai/api/v1")
                self.assertEqual(call.kwargs.get("override_api_key"), "sk-or-batch")
                self.assertEqual(call.kwargs.get("override_api_mode"), "chat_completions")

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_model_only_no_provider_inherits_parent_credentials(self, mock_creds, mock_cfg):
        """Setting only model (no provider) changes model but keeps parent credentials."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "google/gemini-3-flash-preview",
            "provider": "",
        }
        mock_creds.return_value = {
            "model": "google/gemini-3-flash-preview",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Model only test", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            # Model should be overridden
            self.assertEqual(kwargs["model"], "google/gemini-3-flash-preview")
            # But provider/base_url/api_key should inherit from parent
            self.assertEqual(kwargs["provider"], parent.provider)
            self.assertEqual(kwargs["base_url"], parent.base_url)


class TestRoles(unittest.TestCase):
    """Tests for tools/roles.py role registry."""

    def test_all_builtin_roles_exist(self):
        from tools.roles import get_role_config, VALID_ROLES
        for role_name in ["researcher", "coder", "reviewer", "tester"]:
            self.assertIn(role_name, VALID_ROLES)
            cfg = get_role_config(role_name)
            self.assertEqual(cfg.name, role_name)
            self.assertTrue(cfg.identity)
            self.assertTrue(cfg.guidance)
            self.assertTrue(cfg.toolsets)

    def test_unknown_role_raises_valueerror(self):
        from tools.roles import get_role_config
        with self.assertRaises(ValueError) as ctx:
            get_role_config("wizard")
        self.assertIn("wizard", str(ctx.exception))
        self.assertIn("Valid roles", str(ctx.exception))

    def test_reviewer_has_blocked_write_tools(self):
        from tools.roles import get_role_config
        cfg = get_role_config("reviewer")
        self.assertIn("write_file", cfg.blocked_tools)
        self.assertIn("edit_file", cfg.blocked_tools)
        self.assertIn("run_command", cfg.blocked_tools)

    def test_reviewer_toolset_is_file_only(self):
        from tools.roles import get_role_config
        cfg = get_role_config("reviewer")
        self.assertEqual(cfg.toolsets, ["file"])

    def test_researcher_toolset_includes_web(self):
        from tools.roles import get_role_config
        cfg = get_role_config("researcher")
        self.assertIn("web", cfg.toolsets)

    def test_role_configs_are_frozen(self):
        """RoleConfig is frozen=True so roles can't be mutated at runtime."""
        from tools.roles import get_role_config
        cfg = get_role_config("coder")
        with self.assertRaises(Exception):
            cfg.name = "hacked"  # type: ignore[misc]


class TestRolePromptBuilding(unittest.TestCase):
    """Tests for _build_child_system_prompt with role_config."""

    def test_no_role_uses_generic_opener(self):
        prompt = _build_child_system_prompt("Do something")
        self.assertIn("focused subagent", prompt)

    def test_role_identity_replaces_generic_opener(self):
        from tools.roles import get_role_config
        role = get_role_config("researcher")
        prompt = _build_child_system_prompt("Research quantum computing", role_config=role)
        self.assertNotIn("focused subagent", prompt)
        self.assertIn("research specialist", prompt)

    def test_role_guidance_injected_after_goal(self):
        from tools.roles import get_role_config
        role = get_role_config("coder")
        prompt = _build_child_system_prompt("Fix the login bug", role_config=role)
        self.assertIn("APPROACH", prompt)
        self.assertIn("minimal change", prompt)

    def test_goal_always_present(self):
        from tools.roles import get_role_config
        for role_name in ["researcher", "coder", "reviewer", "tester"]:
            role = get_role_config(role_name)
            prompt = _build_child_system_prompt("MY GOAL TEXT", role_config=role)
            self.assertIn("MY GOAL TEXT", prompt)

    def test_context_included_with_role(self):
        from tools.roles import get_role_config
        role = get_role_config("reviewer")
        prompt = _build_child_system_prompt("Review auth.py", context="Focus on SQL injection", role_config=role)
        self.assertIn("CONTEXT", prompt)
        self.assertIn("SQL injection", prompt)

    def test_reviewer_identity_mentions_read_only(self):
        from tools.roles import get_role_config
        role = get_role_config("reviewer")
        prompt = _build_child_system_prompt("Review this code", role_config=role)
        self.assertIn("only read and report", prompt)


class TestRoleIntegration(unittest.TestCase):
    """Integration tests: role= in delegate_task wires through correctly."""

    def test_unknown_role_returns_json_error(self):
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Do stuff", role="wizard", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("wizard", result["error"])

    def test_role_in_schema(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("role", props)
        self.assertEqual(props["role"]["type"], "string")
        self.assertIn("enum", props["role"])
        self.assertIn("reviewer", props["role"]["enum"])

    def test_role_in_task_item_schema(self):
        task_item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        self.assertIn("role", task_item["properties"])

    @patch("tools.delegate_tool._run_single_child")
    def test_role_reaches_child_agent_via_build(self, mock_run):
        """role= triggers role_config to be passed into _build_child_agent."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
        }
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child._delegate_saved_tool_names = []
            mock_build.return_value = mock_child

            delegate_task(goal="Review my code", role="reviewer", parent_agent=parent)

            mock_build.assert_called_once()
            call_kwargs = mock_build.call_args.kwargs
            role_cfg = call_kwargs.get("role_config")
            self.assertIsNotNone(role_cfg)
            self.assertEqual(role_cfg.name, "reviewer")

    @patch("tools.delegate_tool._run_single_child")
    def test_per_task_role_overrides_top_level(self, mock_run):
        """Per-task role in tasks array overrides top-level role."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
        }
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child._delegate_saved_tool_names = []
            mock_build.return_value = mock_child

            delegate_task(
                tasks=[{"goal": "Write tests", "role": "tester"}],
                role="coder",
                parent_agent=parent,
            )

            call_kwargs = mock_build.call_args.kwargs
            self.assertEqual(call_kwargs["role_config"].name, "tester")

    @patch("tools.delegate_tool._run_single_child")
    def test_role_toolsets_constrained_by_parent(self, mock_run):
        """Role toolsets are intersected with parent's enabled toolsets."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
        }
        parent = _make_mock_parent()
        # Parent only has 'file' — researcher wants 'web' too, but can't get it
        parent.enabled_toolsets = ["file"]

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Research topic", role="researcher", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            # 'web' must not appear because parent doesn't have it
            self.assertNotIn("web", kwargs["enabled_toolsets"])
            self.assertIn("file", kwargs["enabled_toolsets"])

    @patch("tools.delegate_tool._run_single_child")
    def test_reviewer_blocked_tools_removed_from_valid_tool_names(self, mock_run):
        """reviewer role removes write tools from child's valid_tool_names."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
        }
        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.valid_tool_names = {"read_file", "write_file", "edit_file", "web_search"}
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Review auth.py", role="reviewer", parent_agent=parent)

            self.assertNotIn("write_file", mock_child.valid_tool_names)
            self.assertNotIn("edit_file", mock_child.valid_tool_names)
            self.assertIn("read_file", mock_child.valid_tool_names)


class TestSharedMemory(unittest.TestCase):
    """Tests for the shared_memory=True delegation parameter."""

    # ------------------------------------------------------------------
    # SharedMemory unit tests
    # ------------------------------------------------------------------

    def test_write_and_read_single_key(self):
        from tools.shared_memory import SharedMemory
        store = SharedMemory()
        result = store.write("root_cause", "null pointer in auth.py:42")
        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "added")
        r = store.read("root_cause")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], "null pointer in auth.py:42")

    def test_write_updates_existing_key(self):
        from tools.shared_memory import SharedMemory
        store = SharedMemory()
        store.write("k", "v1")
        result = store.write("k", "v2")
        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "updated")
        self.assertEqual(store.read("k")["value"], "v2")

    def test_read_all_entries(self):
        from tools.shared_memory import SharedMemory
        store = SharedMemory()
        store.write("a", "1")
        store.write("b", "2")
        r = store.read()
        self.assertTrue(r["ok"])
        self.assertEqual(r["entry_count"], 2)
        self.assertEqual(r["entries"]["a"], "1")

    def test_read_missing_key_returns_error(self):
        from tools.shared_memory import SharedMemory
        store = SharedMemory()
        r = store.read("nonexistent")
        self.assertFalse(r["ok"])
        self.assertIn("available_keys", r)

    def test_delete_existing_key(self):
        from tools.shared_memory import SharedMemory
        store = SharedMemory()
        store.write("tmp", "data")
        d = store.delete("tmp")
        self.assertTrue(d["ok"])
        self.assertEqual(d["entry_count"], 0)
        self.assertFalse(store.read("tmp")["ok"])

    def test_delete_missing_key_returns_error(self):
        from tools.shared_memory import SharedMemory
        store = SharedMemory()
        d = store.delete("ghost")
        self.assertFalse(d["ok"])
        self.assertIn("keys", d)

    def test_capacity_limit(self):
        from tools.shared_memory import SharedMemory, MAX_ENTRIES
        store = SharedMemory()
        for i in range(MAX_ENTRIES):
            r = store.write(f"key_{i}", "v")
            self.assertTrue(r["ok"])
        overflow = store.write("overflow_key", "v")
        self.assertFalse(overflow["ok"])
        self.assertIn("full", overflow["error"])

    def test_key_too_long_rejected(self):
        from tools.shared_memory import SharedMemory, MAX_KEY_LEN
        store = SharedMemory()
        r = store.write("x" * (MAX_KEY_LEN + 1), "value")
        self.assertFalse(r["ok"])
        self.assertIn("too long", r["error"])

    def test_value_too_long_rejected(self):
        from tools.shared_memory import SharedMemory, MAX_VALUE_LEN
        store = SharedMemory()
        r = store.write("k", "x" * (MAX_VALUE_LEN + 1))
        self.assertFalse(r["ok"])
        self.assertIn("too long", r["error"])

    def test_empty_key_rejected(self):
        from tools.shared_memory import SharedMemory
        store = SharedMemory()
        r = store.write("", "value")
        self.assertFalse(r["ok"])
        self.assertIn("empty", r["error"])

    def test_snapshot_returns_copy(self):
        from tools.shared_memory import SharedMemory
        store = SharedMemory()
        store.write("x", "1")
        snap = store.snapshot()
        store.write("x", "2")
        # snapshot is a point-in-time copy, not a live view
        self.assertEqual(snap["x"], "1")

    def test_thread_local_isolation(self):
        """Two threads with different stores don't contaminate each other."""
        from tools.shared_memory import SharedMemory, set_thread_shared_memory, get_thread_shared_memory
        store_a = SharedMemory()
        store_b = SharedMemory()
        store_a.write("key", "A")
        store_b.write("key", "B")

        results = {}

        def _run(name, store):
            set_thread_shared_memory(store)
            results[name] = get_thread_shared_memory().read("key")["value"]
            set_thread_shared_memory(None)

        t1 = threading.Thread(target=_run, args=("t1", store_a))
        t2 = threading.Thread(target=_run, args=("t2", store_b))
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.assertEqual(results["t1"], "A")
        self.assertEqual(results["t2"], "B")

    # ------------------------------------------------------------------
    # Tool handler tests (no shared memory → error)
    # ------------------------------------------------------------------

    def test_write_handler_no_store_returns_error(self):
        from tools.shared_memory import _handle_write, set_thread_shared_memory
        set_thread_shared_memory(None)
        r = json.loads(_handle_write({"key": "k", "value": "v"}))
        self.assertFalse(r["ok"])
        self.assertIn("not enabled", r["error"])

    def test_read_handler_no_store_returns_error(self):
        from tools.shared_memory import _handle_read, set_thread_shared_memory
        set_thread_shared_memory(None)
        r = json.loads(_handle_read({}))
        self.assertFalse(r["ok"])

    def test_delete_handler_no_store_returns_error(self):
        from tools.shared_memory import _handle_delete, set_thread_shared_memory
        set_thread_shared_memory(None)
        r = json.loads(_handle_delete({"key": "k"}))
        self.assertFalse(r["ok"])

    def test_write_handler_with_store(self):
        from tools.shared_memory import _handle_write, set_thread_shared_memory, SharedMemory
        store = SharedMemory()
        set_thread_shared_memory(store)
        try:
            r = json.loads(_handle_write({"key": "finding", "value": "auth bug"}))
            self.assertTrue(r["ok"])
            self.assertEqual(store.read("finding")["value"], "auth bug")
        finally:
            set_thread_shared_memory(None)

    # ------------------------------------------------------------------
    # Integration: delegate_task with shared_memory=True
    # ------------------------------------------------------------------

    @patch("tools.delegate_tool._run_single_child")
    def test_shared_memory_snapshot_in_result(self, mock_run):
        """shared_memory=True → result contains 'shared_memory' snapshot."""
        from tools.shared_memory import set_thread_shared_memory

        def _fake_run(task_index, goal, child=None, parent_agent=None, **kw):
            # Simulate the child writing to the shared store
            store = getattr(child, '_shared_memory_store', None)
            if store:
                store.write("root_cause", "auth.py line 42")
            return {
                "task_index": task_index, "status": "completed",
                "summary": "Done", "api_calls": 1, "duration_seconds": 1.0,
            }

        mock_run.side_effect = _fake_run
        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.valid_tool_names = {"read_file", "write_file"}
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            result = json.loads(
                delegate_task(goal="Find bug", shared_memory=True, parent_agent=parent)
            )

        self.assertIn("shared_memory", result)
        self.assertEqual(result["shared_memory"].get("root_cause"), "auth.py line 42")

    @patch("tools.delegate_tool._run_single_child")
    def test_no_shared_memory_key_when_disabled(self, mock_run):
        """shared_memory=False (default) → result does NOT contain 'shared_memory'."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0,
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Do thing", parent_agent=parent))
        self.assertNotIn("shared_memory", result)

    @patch("tools.delegate_tool._run_single_child")
    def test_shared_memory_tools_injected_into_child(self, mock_run):
        """shared_memory=True → SHARED_MEMORY_TOOL_NAMES added to child.valid_tool_names."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0,
        }
        parent = _make_mock_parent()
        from tools.shared_memory import SHARED_MEMORY_TOOL_NAMES

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.valid_tool_names = {"read_file"}
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Find bug", shared_memory=True, parent_agent=parent)

        for name in SHARED_MEMORY_TOOL_NAMES:
            self.assertIn(name, mock_child.valid_tool_names)

    @patch("tools.delegate_tool._run_single_child")
    def test_shared_memory_tools_not_injected_when_disabled(self, mock_run):
        """shared_memory=False → SHARED_MEMORY_TOOL_NAMES NOT added to child.valid_tool_names."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0,
        }
        parent = _make_mock_parent()
        from tools.shared_memory import SHARED_MEMORY_TOOL_NAMES

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.valid_tool_names = {"read_file"}
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Find bug", shared_memory=False, parent_agent=parent)

        for name in SHARED_MEMORY_TOOL_NAMES:
            self.assertNotIn(name, mock_child.valid_tool_names)

    def test_schema_has_shared_memory_field(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("shared_memory", props)
        self.assertEqual(props["shared_memory"]["type"], "boolean")

    def test_shared_memory_toolset_blocked(self):
        """_strip_blocked_tools removes 'shared_memory' toolset."""
        result = _strip_blocked_tools(["file", "shared_memory", "terminal"])
        self.assertNotIn("shared_memory", result)
        self.assertIn("file", result)
        self.assertIn("terminal", result)


if __name__ == "__main__":
    unittest.main()
