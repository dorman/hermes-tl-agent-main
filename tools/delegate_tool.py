#!/usr/bin/env python3
"""
Delegate Tool -- Subagent Architecture

Spawns child AIAgent instances with isolated context, restricted toolsets,
and their own terminal sessions. Supports single-task and batch (parallel)
modes. The parent blocks until all children complete.

Each child gets:
  - A fresh conversation (no parent history)
  - Its own task_id (own terminal session, file ops cache)
  - A restricted toolset (configurable, with blocked tools always stripped)
  - A focused system prompt built from the delegated goal + context

The parent's context only sees the delegation call and the summary result,
never the child's intermediate tool calls or reasoning.
"""

import json
import logging
logger = logging.getLogger(__name__)
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tools.subagent_channel import (
    SubagentChannel,
    run_drain_loop,
    REPORT_TO_PARENT_SCHEMA,
    ASK_PARENT_SCHEMA,
)
from tools.roles import RoleConfig, get_role_config, VALID_ROLES
from tools.verification import verify_result, WRITE_TOOL_NAMES as _WRITE_TOOL_NAMES
from tools.shared_memory import (
    SharedMemory,
    set_thread_shared_memory,
    SHARED_MEMORY_TOOL_NAMES,
)


# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset([
    "delegate_task",   # no recursive delegation
    "clarify",         # no user interaction
    "memory",          # no writes to shared MEMORY.md
    "send_message",    # no cross-platform side effects
    "execute_code",    # children should reason step-by-step, not write scripts
])

MAX_CONCURRENT_CHILDREN = 3
MAX_DEPTH = 2  # parent (0) -> child (1) -> grandchild rejected (2)
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_TOOLSETS = ["terminal", "file", "web"]


def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


def _build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    role_config: Optional[RoleConfig] = None,
) -> str:
    """Build a focused system prompt for a child agent.

    When role_config is provided, the role's identity replaces the generic
    opener and its behavioral guidance is injected after the goal/context.
    """
    opener = role_config.identity if role_config else "You are a focused subagent working on a specific delegated task."
    parts = [opener, "", f"YOUR TASK:\n{goal}"]

    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context}")

    if role_config and role_config.guidance:
        parts.append(f"\nAPPROACH:\n{role_config.guidance}")

    parts.append(
        "\nWhen finished, provide a clear, concise summary of:\n"
        "- What you did\n"
        "- What you found or accomplished\n"
        "- Any files you created or modified\n"
        "- Any issues encountered\n\n"
        "Be thorough but concise -- your response is returned to the "
        "parent agent as a summary."
    )
    return "\n".join(parts)


def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets that contain only blocked tools."""
    blocked_toolset_names = {
        "delegation", "clarify", "memory", "code_execution", "shared_memory",
    }
    return [t for t in toolsets if t not in blocked_toolset_names]


def _enrich_trace_entry(entry: Dict[str, Any], raw_args: str) -> None:
    """
    Augment a tool trace entry with information extracted from the raw JSON
    arguments — specifically, file paths written by write_file and patch calls.

    This is called at trace-build time (inside _run_single_child) so that the
    verification pass has authoritative path data without re-parsing messages.

    Mutates entry in place; never raises.
    """
    if entry.get("tool") not in _WRITE_TOOL_NAMES:
        return
    try:
        args = json.loads(raw_args or "{}")
    except Exception:
        return

    paths: list[str] = []
    tool = entry["tool"]

    if tool == "write_file":
        p = args.get("path")
        if p:
            paths.append(str(p))

    elif tool == "patch":
        # replace mode: explicit path arg
        p = args.get("path")
        if p:
            paths.append(str(p))
        # patch (V4A) mode: paths embedded in the patch text
        patch_text = args.get("patch") or ""
        if patch_text:
            import re as _re
            for m in _re.finditer(
                r"^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+)$",
                patch_text,
                _re.MULTILINE,
            ):
                paths.append(m.group(1).strip())

    if paths:
        entry["paths_written"] = paths


def _build_child_progress_callback(task_index: int, parent_agent, task_count: int = 1) -> Optional[callable]:
    """Build a callback that relays child agent tool calls to the parent display.

    Two display paths:
      CLI:     prints tree-view lines above the parent's delegation spinner
      Gateway: batches tool names and relays to parent's progress callback

    Returns None if no display mechanism is available, in which case the
    child agent runs with no progress callback (identical to current behavior).
    """
    spinner = getattr(parent_agent, '_delegate_spinner', None)
    parent_cb = getattr(parent_agent, 'tool_progress_callback', None)

    if not spinner and not parent_cb:
        return None  # No display → no callback → zero behavior change

    # Show 1-indexed prefix only in batch mode (multiple tasks)
    prefix = f"[{task_index + 1}] " if task_count > 1 else ""

    # Gateway: batch tool names, flush periodically
    _BATCH_SIZE = 5
    _batch: List[str] = []

    def _callback(tool_name: str, preview: str = None):
        # Special "_thinking" event: model produced text content (reasoning)
        if tool_name == "_thinking":
            if spinner:
                short = (preview[:55] + "...") if preview and len(preview) > 55 else (preview or "")
                try:
                    spinner.print_above(f" {prefix}├─ 💭 \"{short}\"")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            # Don't relay thinking to gateway (too noisy for chat)
            return

        # Regular tool call event
        if spinner:
            short = (preview[:35] + "...") if preview and len(preview) > 35 else (preview or "")
            from agent.display import get_tool_emoji
            emoji = get_tool_emoji(tool_name)
            line = f" {prefix}├─ {emoji} {tool_name}"
            if short:
                line += f"  \"{short}\""
            try:
                spinner.print_above(line)
            except Exception as e:
                logger.debug("Spinner print_above failed: %s", e)

        if parent_cb:
            _batch.append(tool_name)
            if len(_batch) >= _BATCH_SIZE:
                summary = ", ".join(_batch)
                try:
                    parent_cb("subagent_progress", f"🔀 {prefix}{summary}")
                except Exception as e:
                    logger.debug("Parent callback failed: %s", e)
                _batch.clear()

    def _flush():
        """Flush remaining batched tool names to gateway on completion."""
        if parent_cb and _batch:
            summary = ", ".join(_batch)
            try:
                parent_cb("subagent_progress", f"🔀 {prefix}{summary}")
            except Exception as e:
                logger.debug("Parent callback flush failed: %s", e)
            _batch.clear()

    _callback._flush = _flush
    return _callback


def _build_child_agent(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    max_iterations: int,
    parent_agent,
    # Credential overrides from delegation config (provider:model resolution)
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
    channel: Optional[SubagentChannel] = None,
    role_config: Optional[RoleConfig] = None,
    shared_memory_store: Optional[SharedMemory] = None,
):
    """
    Build a child AIAgent on the main thread (thread-safe construction).
    Returns the constructed child agent without running it.

    When override_* params are set (from delegation config), the child uses
    those credentials instead of inheriting from the parent.  This enables
    routing subagents to a different provider:model pair (e.g. cheap/fast
    model on OpenRouter while the parent runs on Nous Portal).
    """
    from run_agent import AIAgent

    # When no explicit toolsets given, fall back to: role defaults > parent toolsets.
    # Always intersect with parent — subagent cannot gain tools the parent lacks.
    parent_toolsets = set(getattr(parent_agent, "enabled_toolsets", None) or DEFAULT_TOOLSETS)
    if toolsets:
        child_toolsets = _strip_blocked_tools([t for t in toolsets if t in parent_toolsets])
    elif role_config and role_config.toolsets:
        child_toolsets = _strip_blocked_tools([t for t in role_config.toolsets if t in parent_toolsets])
    elif parent_agent and getattr(parent_agent, "enabled_toolsets", None):
        child_toolsets = _strip_blocked_tools(parent_agent.enabled_toolsets)
    else:
        child_toolsets = _strip_blocked_tools(DEFAULT_TOOLSETS)

    child_prompt = _build_child_system_prompt(goal, context, role_config)
    # Extract parent's API key so subagents inherit auth (e.g. Nous Portal).
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    # Build progress callback to relay tool calls to parent display
    child_progress_cb = _build_child_progress_callback(task_index, parent_agent)

    # Each subagent gets its own iteration budget capped at max_iterations
    # (configurable via delegation.max_iterations, default 50).  This means
    # total iterations across parent + subagents can exceed the parent's
    # max_iterations.  The user controls the per-subagent cap in config.yaml.

    # Resolve effective credentials: config override > role default > parent inherit
    effective_model = model or (role_config.model if role_config else None) or parent_agent.model
    effective_provider = override_provider or getattr(parent_agent, "provider", None)
    effective_base_url = override_base_url or parent_agent.base_url
    effective_api_key = override_api_key or parent_api_key
    effective_api_mode = override_api_mode or getattr(parent_agent, "api_mode", None)
    effective_acp_command = getattr(parent_agent, "acp_command", None)
    effective_acp_args = list(getattr(parent_agent, "acp_args", []) or [])

    child = AIAgent(
        base_url=effective_base_url,
        api_key=effective_api_key,
        model=effective_model,
        provider=effective_provider,
        api_mode=effective_api_mode,
        acp_command=effective_acp_command,
        acp_args=effective_acp_args,
        max_iterations=max_iterations,
        max_tokens=getattr(parent_agent, "max_tokens", None),
        reasoning_config=getattr(parent_agent, "reasoning_config", None),
        prefill_messages=getattr(parent_agent, "prefill_messages", None),
        enabled_toolsets=child_toolsets,
        quiet_mode=True,
        ephemeral_system_prompt=child_prompt,
        log_prefix=f"[subagent-{task_index}]",
        platform=parent_agent.platform,
        skip_context_files=True,
        skip_memory=True,
        clarify_callback=None,
        session_db=getattr(parent_agent, '_session_db', None),
        providers_allowed=parent_agent.providers_allowed,
        providers_ignored=parent_agent.providers_ignored,
        providers_order=parent_agent.providers_order,
        provider_sort=parent_agent.provider_sort,
        tool_progress_callback=child_progress_cb,
        iteration_budget=None,  # fresh budget per subagent
    )
    # Set delegation depth so children can't spawn grandchildren
    child._delegate_depth = getattr(parent_agent, '_delegate_depth', 0) + 1

    # Apply role-level tool blocks (e.g. reviewer must not write files)
    if role_config and role_config.blocked_tools:
        if hasattr(child, 'valid_tool_names') and child.valid_tool_names is not None:
            child.valid_tool_names = set(child.valid_tool_names) - role_config.blocked_tools

    # Inject shared memory tools if a store is provided.
    # shared_memory tools are in BLOCKED_TOOLSET_NAMES so they are never
    # part of the standard tool set — they're only added explicitly here.
    if shared_memory_store is not None:
        child._shared_memory_store = shared_memory_store
        if hasattr(child, 'valid_tool_names') and child.valid_tool_names is not None:
            child.valid_tool_names = set(child.valid_tool_names) | SHARED_MEMORY_TOOL_NAMES

    # Register child for interrupt propagation
    if hasattr(parent_agent, '_active_children'):
        lock = getattr(parent_agent, '_active_children_lock', None)
        if lock:
            with lock:
                parent_agent._active_children.append(child)
        else:
            parent_agent._active_children.append(child)

    # Wire up two-way channel if provided.
    # Registers report_to_parent and ask_parent as ad-hoc tools on the child
    # by injecting them into the child's tool registry before first use.
    if channel is not None:
        child._subagent_channel = channel
        _register_channel_tools(child, channel)

    return child


def _register_channel_tools(child_agent, channel: SubagentChannel) -> None:
    """
    Ensure report_to_parent and ask_parent are enabled for the child agent.

    The actual tool handlers live in the global registry and read the channel
    from a thread-local (set in _run_single_child before run_conversation).
    Here we just make sure these tool names are in the child's valid_tool_names
    so the model is allowed to call them.
    """
    # Add channel tool names to the child's enabled set if it has one
    if hasattr(child_agent, 'valid_tool_names') and child_agent.valid_tool_names is not None:
        child_agent.valid_tool_names = set(child_agent.valid_tool_names) | {
            "report_to_parent", "ask_parent"
        }

def _run_single_child(
    task_index: int,
    goal: str,
    child=None,
    parent_agent=None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Run a pre-built child agent. Called from within a thread.
    Returns a structured result dict.
    """
    child_start = time.monotonic()

    # Get the progress callback from the child agent
    child_progress_cb = getattr(child, 'tool_progress_callback', None)

    # Activate shared memory for this thread if a store was attached
    _sm_store = getattr(child, '_shared_memory_store', None)
    if _sm_store is not None:
        set_thread_shared_memory(_sm_store)

    # Set up two-way channel if the child has one
    channel: Optional[SubagentChannel] = getattr(child, '_subagent_channel', None)
    drain_thread = None
    done_event = threading.Event()

    if channel is not None:
        from tools.subagent_channel import set_thread_channel
        set_thread_channel(channel)

        parent_progress_cb = getattr(parent_agent, 'tool_progress_callback', None)
        task_count = 1  # will be overridden by caller context; fine for display
        prefix = f"[{task_index + 1}] "

        drain_thread = threading.Thread(
            target=run_drain_loop,
            kwargs=dict(
                channel=channel,
                done_event=done_event,
                progress_callback=parent_progress_cb,
                reply_callback=None,  # extend here to add parent-replies-to-child
                task_prefix=prefix,
            ),
            daemon=True,
            name=f"channel-drain-{task_index}",
        )
        drain_thread.start()

    # Restore parent tool names using the value saved before child construction
    # mutated the global. This is the correct parent toolset, not the child's.
    import model_tools
    _saved_tool_names = getattr(child, "_delegate_saved_tool_names",
                                list(model_tools._last_resolved_tool_names))

    try:
        result = child.run_conversation(user_message=goal)

        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, '_flush'):
            try:
                child_progress_cb._flush()
            except Exception as e:
                logger.debug("Progress callback flush failed: %s", e)

        duration = round(time.monotonic() - child_start, 2)

        summary = result.get("final_response") or ""
        completed = result.get("completed", False)
        interrupted = result.get("interrupted", False)
        api_calls = result.get("api_calls", 0)

        if interrupted:
            status = "interrupted"
        elif summary:
            # A summary means the subagent produced usable output.
            # exit_reason ("completed" vs "max_iterations") already
            # tells the parent *how* the task ended.
            status = "completed"
        else:
            status = "failed"

        # Build tool trace from conversation messages (already in memory).
        # Uses tool_call_id to correctly pair parallel tool calls with results.
        tool_trace: list[Dict[str, Any]] = []
        trace_by_id: Dict[str, Dict[str, Any]] = {}
        messages = result.get("messages") or []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "assistant":
                    for tc in (msg.get("tool_calls") or []):
                        fn = tc.get("function", {})
                        raw_args = fn.get("arguments", "")
                        entry_t = {
                            "tool": fn.get("name", "unknown"),
                            "args_bytes": len(raw_args),
                        }
                        # Enrich write-tool entries with paths_written for verification
                        _enrich_trace_entry(entry_t, raw_args)
                        tool_trace.append(entry_t)
                        tc_id = tc.get("id")
                        if tc_id:
                            trace_by_id[tc_id] = entry_t
                elif msg.get("role") == "tool":
                    content = msg.get("content", "")
                    is_error = bool(
                        content and "error" in content[:80].lower()
                    )
                    result_meta = {
                        "result_bytes": len(content),
                        "status": "error" if is_error else "ok",
                    }
                    # Match by tool_call_id for parallel calls
                    tc_id = msg.get("tool_call_id")
                    target = trace_by_id.get(tc_id) if tc_id else None
                    if target is not None:
                        target.update(result_meta)
                    elif tool_trace:
                        # Fallback for messages without tool_call_id
                        tool_trace[-1].update(result_meta)

        # Determine exit reason
        if interrupted:
            exit_reason = "interrupted"
        elif completed:
            exit_reason = "completed"
        else:
            exit_reason = "max_iterations"

        # Extract token counts (safe for mock objects)
        _input_tokens = getattr(child, "session_prompt_tokens", 0)
        _output_tokens = getattr(child, "session_completion_tokens", 0)
        _model = getattr(child, "model", None)

        entry: Dict[str, Any] = {
            "task_index": task_index,
            "status": status,
            "summary": summary,
            "api_calls": api_calls,
            "duration_seconds": duration,
            "model": _model if isinstance(_model, str) else None,
            "exit_reason": exit_reason,
            "tokens": {
                "input": _input_tokens if isinstance(_input_tokens, (int, float)) else 0,
                "output": _output_tokens if isinstance(_output_tokens, (int, float)) else 0,
            },
            "tool_trace": tool_trace,
        }
        if status == "failed":
            entry["error"] = result.get("error", "Subagent did not produce a response.")

        return entry

    except Exception as exc:
        duration = round(time.monotonic() - child_start, 2)
        logging.exception(f"[subagent-{task_index}] failed")
        return {
            "task_index": task_index,
            "status": "error",
            "summary": None,
            "error": str(exc),
            "api_calls": 0,
            "duration_seconds": duration,
        }

    finally:
        # Signal drain thread to finish and wait for it
        done_event.set()
        if drain_thread is not None:
            drain_thread.join(timeout=5.0)

        # Clear thread-local channel so it doesn't leak to reused threads
        if channel is not None:
            from tools.subagent_channel import set_thread_channel
            set_thread_channel(None)

        # Clear thread-local shared memory reference
        if _sm_store is not None:
            set_thread_shared_memory(None)

        # Restore the parent's tool names so the process-global is correct
        # for any subsequent execute_code calls or other consumers.
        import model_tools

        saved_tool_names = getattr(child, "_delegate_saved_tool_names", None)
        if isinstance(saved_tool_names, list):
            model_tools._last_resolved_tool_names = list(saved_tool_names)

        # Unregister child from interrupt propagation
        if hasattr(parent_agent, '_active_children'):
            try:
                lock = getattr(parent_agent, '_active_children_lock', None)
                if lock:
                    with lock:
                        parent_agent._active_children.remove(child)
                else:
                    parent_agent._active_children.remove(child)
            except (ValueError, UnboundLocalError) as e:
                logger.debug("Could not remove child from active_children: %s", e)

def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    role: Optional[str] = None,
    verify: bool = False,
    shared_memory: bool = False,
    parent_agent=None,
) -> str:
    """
    Spawn one or more child agents to handle delegated tasks.

    Supports two modes:
      - Single: provide goal (+ optional context, toolsets)
      - Batch:  provide tasks array [{goal, context, toolsets}, ...]

    Returns JSON with results array, one entry per task.
    """
    if parent_agent is None:
        return json.dumps({"error": "delegate_task requires a parent agent context."})

    # Depth limit
    depth = getattr(parent_agent, '_delegate_depth', 0)
    if depth >= MAX_DEPTH:
        return json.dumps({
            "error": (
                f"Delegation depth limit reached ({MAX_DEPTH}). "
                "Subagents cannot spawn further subagents."
            )
        })

    # Resolve top-level role (applies to all tasks unless overridden per-task)
    top_role_config: Optional[RoleConfig] = None
    if role:
        try:
            top_role_config = get_role_config(role)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

    # Shared memory store — one instance shared by all children in this call
    shared_memory_store: Optional[SharedMemory] = SharedMemory() if shared_memory else None

    # Load config
    cfg = _load_config()
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)

    # Resolve delegation credentials (provider:model pair).
    # When delegation.provider is configured, this resolves the full credential
    # bundle (base_url, api_key, api_mode) via the same runtime provider system
    # used by CLI/gateway startup.  When unconfigured, returns None values so
    # children inherit from the parent.
    try:
        creds = _resolve_delegation_credentials(cfg, parent_agent)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    # Normalize to task list
    if tasks and isinstance(tasks, list):
        task_list = tasks[:MAX_CONCURRENT_CHILDREN]
    elif goal and isinstance(goal, str) and goal.strip():
        task_list = [{"goal": goal, "context": context, "toolsets": toolsets}]
    else:
        return json.dumps({"error": "Provide either 'goal' (single task) or 'tasks' (batch)."})

    if not task_list:
        return json.dumps({"error": "No tasks provided."})

    # Validate each task has a goal
    for i, task in enumerate(task_list):
        if not task.get("goal", "").strip():
            return json.dumps({"error": f"Task {i} is missing a 'goal'."})

    overall_start = time.monotonic()
    results = []

    n_tasks = len(task_list)
    # Track goal labels for progress display (truncated for readability)
    task_labels = [t["goal"][:40] for t in task_list]

    # Save parent tool names BEFORE any child construction mutates the global.
    # _build_child_agent() calls AIAgent() which calls get_tool_definitions(),
    # which overwrites model_tools._last_resolved_tool_names with child's toolset.
    import model_tools as _model_tools
    _parent_tool_names = list(_model_tools._last_resolved_tool_names)

    # Build all child agents on the main thread (thread-safe construction)
    # Wrapped in try/finally so the global is always restored even if a
    # child build raises (otherwise _last_resolved_tool_names stays corrupted).
    children = []
    try:
        for i, t in enumerate(task_list):
            # Resolve per-task role: task-level overrides top-level
            task_role_name = t.get("role") or role
            task_role_config: Optional[RoleConfig] = top_role_config
            if task_role_name and task_role_name != role:
                try:
                    task_role_config = get_role_config(task_role_name)
                except ValueError as exc:
                    return json.dumps({"error": f"Task {i}: {exc}"})
            elif task_role_name == role:
                task_role_config = top_role_config  # already resolved above

            # Per-task max_iterations: explicit > role default > delegation config default
            task_max_iter = (
                t.get("max_iterations")
                or max_iterations
                or (task_role_config.max_iterations if task_role_config else None)
                or default_max_iter
            )

            channel = SubagentChannel()
            child = _build_child_agent(
                task_index=i, goal=t["goal"], context=t.get("context"),
                toolsets=t.get("toolsets") or toolsets, model=creds["model"],
                max_iterations=task_max_iter, parent_agent=parent_agent,
                override_provider=creds["provider"], override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                channel=channel,
                role_config=task_role_config,
                shared_memory_store=shared_memory_store,
            )
            # Override with correct parent tool names (before child construction mutated global)
            child._delegate_saved_tool_names = _parent_tool_names
            children.append((i, t, child))
    finally:
        # Authoritative restore: reset global to parent's tool names after all children built
        _model_tools._last_resolved_tool_names = _parent_tool_names

    if n_tasks == 1:
        # Single task -- run directly (no thread pool overhead)
        _i, _t, child = children[0]
        result = _run_single_child(0, _t["goal"], child, parent_agent)
        task_verify = _t.get("verify", verify)
        if task_verify and result.get("status") == "completed":
            result["verification"] = verify_result(result)
        results.append(result)
    else:
        # Batch -- run in parallel with per-task progress lines
        completed_count = 0
        spinner_ref = getattr(parent_agent, '_delegate_spinner', None)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHILDREN) as executor:
            futures = {}
            for i, t, child in children:
                future = executor.submit(
                    _run_single_child,
                    task_index=i,
                    goal=t["goal"],
                    child=child,
                    parent_agent=parent_agent,
                )
                futures[future] = i

            for future in as_completed(futures):
                try:
                    entry = future.result()
                except Exception as exc:
                    idx = futures[future]
                    entry = {
                        "task_index": idx,
                        "status": "error",
                        "summary": None,
                        "error": str(exc),
                        "api_calls": 0,
                        "duration_seconds": 0,
                    }
                task_idx = entry["task_index"]
                task_def = task_list[task_idx] if task_idx < len(task_list) else {}
                task_verify = task_def.get("verify", verify)
                if task_verify and entry.get("status") == "completed":
                    entry["verification"] = verify_result(entry)
                results.append(entry)
                completed_count += 1

                # Print per-task completion line above the spinner
                idx = entry["task_index"]
                label = task_labels[idx] if idx < len(task_labels) else f"Task {idx}"
                dur = entry.get("duration_seconds", 0)
                status = entry.get("status", "?")
                icon = "✓" if status == "completed" else "✗"
                remaining = n_tasks - completed_count
                completion_line = f"{icon} [{idx+1}/{n_tasks}] {label}  ({dur}s)"
                if spinner_ref:
                    try:
                        spinner_ref.print_above(completion_line)
                    except Exception:
                        print(f"  {completion_line}")
                else:
                    print(f"  {completion_line}")

                # Update spinner text to show remaining count
                if spinner_ref and remaining > 0:
                    try:
                        spinner_ref.update_text(f"🔀 {remaining} task{'s' if remaining != 1 else ''} remaining")
                    except Exception as e:
                        logger.debug("Spinner update_text failed: %s", e)

        # Sort by task_index so results match input order
        results.sort(key=lambda r: r["task_index"])

    # Notify parent's memory provider of delegation outcomes
    if parent_agent and hasattr(parent_agent, '_memory_manager') and parent_agent._memory_manager:
        for entry in results:
            try:
                _task_goal = task_list[entry["task_index"]]["goal"] if entry["task_index"] < len(task_list) else ""
                parent_agent._memory_manager.on_delegation(
                    task=_task_goal,
                    result=entry.get("summary", "") or "",
                    child_session_id=getattr(children[entry["task_index"]][2], "session_id", "") if entry["task_index"] < len(children) else "",
                )
            except Exception:
                pass

    total_duration = round(time.monotonic() - overall_start, 2)

    output: Dict[str, Any] = {
        "results": results,
        "total_duration_seconds": total_duration,
    }
    if shared_memory_store is not None:
        output["shared_memory"] = shared_memory_store.snapshot()

    return json.dumps(output, ensure_ascii=False)


def _resolve_delegation_credentials(cfg: dict, parent_agent) -> dict:
    """Resolve credentials for subagent delegation.

    If ``delegation.base_url`` is configured, subagents use that direct
    OpenAI-compatible endpoint. Otherwise, if ``delegation.provider`` is
    configured, the full credential bundle (base_url, api_key, api_mode,
    provider) is resolved via the runtime provider system — the same path used
    by CLI/gateway startup. This lets subagents run on a completely different
    provider:model pair.

    If neither base_url nor provider is configured, returns None values so the
    child inherits everything from the parent agent.

    Raises ValueError with a user-friendly message on credential failure.
    """
    configured_model = str(cfg.get("model") or "").strip() or None
    configured_provider = str(cfg.get("provider") or "").strip() or None
    configured_base_url = str(cfg.get("base_url") or "").strip() or None
    configured_api_key = str(cfg.get("api_key") or "").strip() or None

    if configured_base_url:
        api_key = (
            configured_api_key
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        if not api_key:
            raise ValueError(
                "Delegation base_url is configured but no API key was found. "
                "Set delegation.api_key or OPENAI_API_KEY."
            )

        base_lower = configured_base_url.lower()
        provider = "custom"
        api_mode = "chat_completions"
        if "chatgpt.com/backend-api/codex" in base_lower:
            provider = "openai-codex"
            api_mode = "codex_responses"
        elif "api.anthropic.com" in base_lower:
            provider = "anthropic"
            api_mode = "anthropic_messages"

        return {
            "model": configured_model,
            "provider": provider,
            "base_url": configured_base_url,
            "api_key": api_key,
            "api_mode": api_mode,
        }

    if not configured_provider:
        # No provider override — child inherits everything from parent
        return {
            "model": configured_model,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

    # Provider is configured — resolve full credentials
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested=configured_provider)
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve delegation provider '{configured_provider}': {exc}. "
            f"Check that the provider is configured (API key set, valid provider name), "
            f"or set delegation.base_url/delegation.api_key for a direct endpoint. "
            f"Available providers: openrouter, nous, zai, kimi-coding, minimax."
        ) from exc

    api_key = runtime.get("api_key", "")
    if not api_key:
        raise ValueError(
            f"Delegation provider '{configured_provider}' resolved but has no API key. "
            f"Set the appropriate environment variable or run 'hermes login'."
        )

    return {
        "model": configured_model,
        "provider": runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
    }


def _load_config() -> dict:
    """Load delegation config from CLI_CONFIG or persistent config.

    Checks the runtime config (cli.py CLI_CONFIG) first, then falls back
    to the persistent config (hermes_cli/config.py load_config()) so that
    ``delegation.model`` / ``delegation.provider`` are picked up regardless
    of the entry point (CLI, gateway, cron).
    """
    try:
        from cli import CLI_CONFIG
        cfg = CLI_CONFIG.get("delegation", {})
        if cfg:
            return cfg
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config
        full = load_config()
        return full.get("delegation", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    "description": (
        "Spawn one or more subagents to work on tasks in isolated contexts. "
        "Each subagent gets its own conversation, terminal session, and toolset. "
        "Only the final summary is returned -- intermediate tool results "
        "never enter your context window.\n\n"
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets)\n"
        "2. Batch (parallel): provide 'tasks' array with up to 3 items. "
        "All run concurrently and results are returned together.\n\n"
        "WHEN TO USE delegate_task:\n"
        "- Reasoning-heavy subtasks (debugging, code review, research synthesis)\n"
        "- Tasks that would flood your context with intermediate data\n"
        "- Parallel independent workstreams (research A and B simultaneously)\n\n"
        "WHEN NOT TO USE (use these instead):\n"
        "- Mechanical multi-step work with no reasoning needed -> use execute_code\n"
        "- Single tool call -> just call the tool directly\n"
        "- Tasks needing user interaction -> subagents cannot use clarify\n\n"
        "IMPORTANT:\n"
        "- Subagents have NO memory of your conversation. Pass all relevant "
        "info (file paths, error messages, constraints) via the 'context' field.\n"
        "- Subagents CANNOT call: delegate_task, clarify, memory, send_message, "
        "execute_code.\n"
        "- Each subagent gets its own terminal session (separate working directory and state).\n"
        "- Results are always returned as an array, one entry per task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What the subagent should accomplish. Be specific and "
                    "self-contained -- the subagent knows nothing about your "
                    "conversation history."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Background information the subagent needs: file paths, "
                    "error messages, project structure, constraints. The more "
                    "specific you are, the better the subagent performs."
                ),
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Toolsets to enable for this subagent. "
                    "Default: inherits your enabled toolsets. "
                    "Common patterns: ['terminal', 'file'] for code work, "
                    "['web'] for research, ['terminal', 'file', 'web'] for "
                    "full-stack tasks."
                ),
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Task goal"},
                        "context": {"type": "string", "description": "Task-specific context"},
                        "toolsets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Toolsets for this specific task",
                        },
                        "role": {
                            "type": "string",
                            "enum": VALID_ROLES,
                            "description": "Role for this task (overrides top-level role)",
                        },
                        "verify": {
                            "type": "boolean",
                            "description": "Verify this task's result (overrides top-level verify)",
                        },
                    },
                    "required": ["goal"],
                },
                "maxItems": 3,
                "description": (
                    "Batch mode: up to 3 tasks to run in parallel. Each gets "
                    "its own subagent with isolated context and terminal session. "
                    "When provided, top-level goal/context/toolsets are ignored."
                ),
            },
            "max_iterations": {
                "type": "integer",
                "description": (
                    "Max tool-calling turns per subagent (default: 50). "
                    "Only set lower for simple tasks."
                ),
            },
            "role": {
                "type": "string",
                "enum": VALID_ROLES,
                "description": (
                    "Pre-configured agent persona applied to all tasks. "
                    "Each role carries a tailored system prompt and default toolset:\n"
                    "- researcher: web + file search, synthesis, citation\n"
                    "- coder: read-modify-run cycle, minimal targeted changes\n"
                    "- reviewer: read-only code critique with file+line citations (no writes)\n"
                    "- tester: write and run tests, report coverage\n"
                    "Individual tasks in a batch can override with their own 'role' field."
                ),
            },
            "verify": {
                "type": "boolean",
                "description": (
                    "When true, run a verification pass after each subagent completes. "
                    "Checks that files the summary claims were created or modified actually "
                    "exist on disk, and flags summaries that describe file writes when the "
                    "tool trace shows no write-tool calls (hallucination signal). "
                    "Adds a 'verification' dict to each result: {verified, checks, "
                    "write_tools_used, run_tools_used, suspicious, note}. "
                    "Does not change the result status — the parent decides how to act. "
                    "Default: false. Individual batch tasks can set their own 'verify' field."
                ),
            },
            "shared_memory": {
                "type": "boolean",
                "description": (
                    "When true, all subagents in this call share a session-scoped "
                    "key-value scratch space. Subagents can read and write facts "
                    "(shared_memory_write / shared_memory_read / shared_memory_delete) "
                    "without touching long-term memory or MEMORY.md. "
                    "Useful for parallel workstreams that discover related facts — "
                    "e.g. one subagent finds the root cause while another is still "
                    "running and can read it to focus its own search. "
                    "After all subagents finish, the final store snapshot is included "
                    "in the result as 'shared_memory': {key: value, ...}. "
                    "The store is discarded when the delegation call ends. "
                    "Default: false."
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        toolsets=args.get("toolsets"),
        tasks=args.get("tasks"),
        max_iterations=args.get("max_iterations"),
        role=args.get("role"),
        verify=bool(args.get("verify", False)),
        shared_memory=bool(args.get("shared_memory", False)),
        parent_agent=kw.get("parent_agent")),
    check_fn=check_delegate_requirements,
    emoji="🔀",
)
