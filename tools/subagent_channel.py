"""
SubagentChannel — Two-way message passing between a parent agent and a subagent.

The child agent gets two tools wired to its channel instance:

  report_to_parent(message)      — non-blocking status update
  ask_parent(question, timeout)  — blocking question; returns parent's reply or None

The parent drains messages from a background thread while the child runs,
relaying them to the parent's existing progress callback and optionally
answering blocking questions via a parent_reply_callback.

Architecture:
    child thread                    drain thread (in _run_single_child)
    ──────────────────────          ──────────────────────────────────
    report_to_parent("...")  ──▶    inbox.get() → progress_callback(...)
    ask_parent("?")          ──▶    inbox.get() → reply_callback("?") → slot[0] = answer
                                                                       ↓
                             ◀──    reply_event.set()  (unblocks child)
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class StatusMessage:
    """Non-blocking update from child → parent."""
    content: str


@dataclass
class QuestionMessage:
    """Blocking question from child → parent.  Child waits for reply_event."""
    content: str
    reply_event: threading.Event = field(default_factory=threading.Event)
    reply_slot: list = field(default_factory=lambda: [None])  # parent writes answer here

    def answer(self, reply: str) -> None:
        """Called by the parent (drain thread) to unblock the child."""
        self.reply_slot[0] = reply
        self.reply_event.set()

    def reply(self) -> Optional[str]:
        """Returns the parent's answer (None if timed out)."""
        return self.reply_slot[0]


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

class SubagentChannel:
    """
    Thread-safe two-way message channel.  One instance per subagent.

    Child-side API (called from tool handlers inside the child thread):
        channel.send_status(msg)              — fire-and-forget
        channel.ask_parent(question, timeout) — blocks until answered

    Parent-side API (called from the drain thread):
        channel.drain_nowait()  — returns all pending messages without blocking
        channel.close()         — signals child that channel is gone (ask_parent → None)
    """

    DEFAULT_QUESTION_TIMEOUT = 60.0  # seconds child will wait for parent reply

    def __init__(self) -> None:
        self._inbox: queue.Queue = queue.Queue()
        self._closed = False

    # ------------------------------------------------------------------
    # Child-side
    # ------------------------------------------------------------------

    def send_status(self, message: str) -> None:
        """Non-blocking: enqueue a status update for the parent to display."""
        if not self._closed:
            self._inbox.put(StatusMessage(content=message))

    def ask_parent(self, question: str, timeout: float = DEFAULT_QUESTION_TIMEOUT) -> Optional[str]:
        """
        Blocking: send a question, wait up to `timeout` seconds for parent reply.
        Returns the parent's answer string, or None on timeout / closed channel.
        """
        if self._closed:
            return None
        msg = QuestionMessage(content=question)
        self._inbox.put(msg)
        answered = msg.reply_event.wait(timeout=timeout)
        return msg.reply() if answered else None

    # ------------------------------------------------------------------
    # Parent-side (drain thread)
    # ------------------------------------------------------------------

    def drain_nowait(self) -> list:
        """Return all currently queued messages without blocking."""
        messages = []
        while True:
            try:
                messages.append(self._inbox.get_nowait())
            except queue.Empty:
                break
        return messages

    def drain_blocking(self, timeout: float = 0.1):
        """
        Block up to `timeout` seconds for the next message.
        Returns a list with 0 or 1 items.
        """
        try:
            return [self._inbox.get(timeout=timeout)]
        except queue.Empty:
            return []

    def close(self) -> None:
        """
        Signal that the channel is closing.  Any pending ask_parent() calls
        in the child will return None rather than blocking forever.
        """
        self._closed = True
        # Unblock any waiting QuestionMessages by draining and auto-answering None
        for msg in self.drain_nowait():
            if isinstance(msg, QuestionMessage):
                msg.answer("")


# ---------------------------------------------------------------------------
# Drain loop (runs in a background thread inside _run_single_child)
# ---------------------------------------------------------------------------

def run_drain_loop(
    channel: SubagentChannel,
    done_event: threading.Event,
    progress_callback: Optional[Callable] = None,
    reply_callback: Optional[Callable[[str], str]] = None,
    task_prefix: str = "",
) -> None:
    """
    Continuously drain the channel until done_event is set.

    Args:
        channel:           The SubagentChannel to drain.
        done_event:        Set by _run_single_child when the child finishes.
        progress_callback: parent_agent.tool_progress_callback — for status display.
        reply_callback:    Optional callable(question: str) -> str that generates
                           a parent reply.  When None, blocking questions time out.
        task_prefix:       Display prefix, e.g. "[2] ".
    """
    while not done_event.is_set():
        messages = channel.drain_blocking(timeout=0.1)
        for msg in messages:
            _handle_message(msg, progress_callback, reply_callback, task_prefix)

    # Drain any final messages after child completes
    for msg in channel.drain_nowait():
        _handle_message(msg, progress_callback, reply_callback, task_prefix)

    channel.close()


def _handle_message(
    msg,
    progress_callback: Optional[Callable],
    reply_callback: Optional[Callable[[str], str]],
    task_prefix: str,
) -> None:
    if isinstance(msg, StatusMessage):
        text = f"{task_prefix}📢 {msg.content}"
        if progress_callback:
            try:
                progress_callback("subagent_status", text)
            except Exception:
                pass

    elif isinstance(msg, QuestionMessage):
        text = f"{task_prefix}❓ {msg.content}"
        if progress_callback:
            try:
                progress_callback("subagent_question", text)
            except Exception:
                pass

        if reply_callback:
            try:
                answer = reply_callback(msg.content) or ""
            except Exception:
                answer = ""
        else:
            answer = ""  # no reply callback → child gets empty string and continues

        msg.answer(answer)


# ---------------------------------------------------------------------------
# Thread-local channel registry
#
# Tool handlers are registered globally, but each child thread sets its own
# channel via set_thread_channel() before calling run_conversation().
# The handlers read from this thread-local so they never cross-contaminate
# parallel subagents running in sibling threads.
# ---------------------------------------------------------------------------

import threading as _threading

_thread_local = _threading.local()


def set_thread_channel(channel: Optional[SubagentChannel]) -> None:
    """Call from the child's thread before run_conversation() starts."""
    _thread_local.channel = channel


def get_thread_channel() -> Optional[SubagentChannel]:
    """Called by tool handlers to retrieve this thread's channel."""
    return getattr(_thread_local, "channel", None)


# ---------------------------------------------------------------------------
# Tool schemas (registered in delegate_tool.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global tool handlers (read channel from thread-local)
# ---------------------------------------------------------------------------

def _handle_report_to_parent(args: dict, **_kw) -> str:
    channel = get_thread_channel()
    if channel is None:
        return json.dumps({"ok": False, "error": "No active subagent channel"})
    message = str(args.get("message") or "").strip()
    if message:
        channel.send_status(message)
    return json.dumps({"ok": True})


def _handle_ask_parent(args: dict, **_kw) -> str:
    channel = get_thread_channel()
    if channel is None:
        return json.dumps({"reply": "", "error": "No active subagent channel"})
    question = str(args.get("question") or "").strip()
    if not question:
        return json.dumps({"reply": ""})
    raw_timeout = args.get("timeout")
    try:
        timeout = min(float(raw_timeout), 120.0) if raw_timeout is not None else 60.0
    except (TypeError, ValueError):
        timeout = 60.0
    reply = channel.ask_parent(question, timeout=timeout)
    return json.dumps({"reply": reply or ""})


REPORT_TO_PARENT_SCHEMA = {
    "name": "report_to_parent",
    "description": (
        "Send a non-blocking status update to the parent agent. "
        "Use this to share discoveries, progress milestones, or warnings "
        "without pausing your work. The parent sees the message in real-time "
        "above the delegation spinner.\n\n"
        "Good uses:\n"
        "- 'Found 3 failing tests in auth module'\n"
        "- 'Writing output to /tmp/report.json'\n"
        "- 'Warning: rate limit hit, retrying in 5s'\n\n"
        "This tool always returns immediately — use ask_parent if you need a reply."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Status update to send to the parent agent. Keep it concise (1-2 sentences).",
            }
        },
        "required": ["message"],
    },
}

ASK_PARENT_SCHEMA = {
    "name": "ask_parent",
    "description": (
        "Ask the parent agent a question and wait for its reply before continuing. "
        "Use this when you are genuinely blocked and need guidance — not for "
        "optional information you could discover yourself.\n\n"
        "Good uses:\n"
        "- 'Should I delete the old config files or archive them?'\n"
        "- 'I found two conflicting schemas. Which takes precedence?'\n"
        "- 'The target directory already exists. Overwrite or skip?'\n\n"
        "This tool BLOCKS until the parent replies (up to 60 seconds). "
        "If no reply arrives, it returns an empty string and you should make "
        "a reasonable default decision and continue."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The question for the parent. Include all relevant context "
                    "so the parent can answer without asking follow-ups."
                ),
            },
            "timeout": {
                "type": "number",
                "description": "Seconds to wait for a reply (default: 60, max: 120).",
            },
        },
        "required": ["question"],
    },
}


# ---------------------------------------------------------------------------
# Registry — register tools globally; handlers resolve channel via thread-local
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402

registry.register(
    name="report_to_parent",
    toolset="delegation",
    schema=REPORT_TO_PARENT_SCHEMA,
    handler=_handle_report_to_parent,
    emoji="📢",
)

registry.register(
    name="ask_parent",
    toolset="delegation",
    schema=ASK_PARENT_SCHEMA,
    handler=_handle_ask_parent,
    emoji="❓",
)
