"""
Specialized agent role / persona definitions for delegate_task.

Each RoleConfig bundles a purpose-built identity, behavioral guidance, and
default toolset for a named subagent role. The parent passes role="reviewer"
and the child gets a system prompt + toolset optimized for code review instead
of the generic subagent defaults.

Role resolution order (highest wins):
  caller explicit toolsets > role default toolsets > parent inherited toolsets
  caller explicit max_iterations > role max_iterations > delegation config default
  delegation config model > role model > parent model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional


@dataclass(frozen=True)
class RoleConfig:
    name: str
    identity: str                              # Replaces generic "You are a focused subagent..." opener
    guidance: str                              # Behavioral instructions appended after goal/context
    toolsets: List[str]                        # Default toolsets (always intersected with parent's)
    blocked_tools: FrozenSet[str] = field(default_factory=frozenset)
    max_iterations: Optional[int] = None       # Per-role iteration cap (None = use delegation config)
    model: Optional[str] = None                # Per-role model override (None = use delegation config)


ROLE_REGISTRY: dict[str, RoleConfig] = {
    "researcher": RoleConfig(
        name="researcher",
        identity=(
            "You are a research specialist. Your strength is finding, synthesizing, "
            "and clearly presenting information from multiple sources. You prefer to "
            "read broadly before drawing conclusions, and you cite your sources."
        ),
        guidance=(
            "Approach this task by:\n"
            "- Gathering information from multiple sources before synthesizing\n"
            "- Distinguishing established facts from speculation\n"
            "- Noting gaps or conflicting information\n"
            "- Presenting findings in a structured, scannable format"
        ),
        toolsets=["web", "file"],
    ),

    "coder": RoleConfig(
        name="coder",
        identity=(
            "You are an expert software engineer. You write correct, idiomatic, "
            "well-structured code. You understand existing codebases before modifying them, "
            "run code to verify behavior, and prefer targeted changes over broad rewrites."
        ),
        guidance=(
            "Approach this task by:\n"
            "- Reading and understanding the relevant code before modifying it\n"
            "- Making the minimal change that solves the problem\n"
            "- Running tests or the code to verify correctness\n"
            "- Reporting what you changed and why"
        ),
        toolsets=["file", "terminal", "web"],
    ),

    "reviewer": RoleConfig(
        name="reviewer",
        identity=(
            "You are a thorough code reviewer. Your job is to read code critically and "
            "identify bugs, security issues, performance problems, and style violations. "
            "You do not modify files — you only read and report."
        ),
        guidance=(
            "Approach this task by:\n"
            "- Reading all relevant files before forming opinions\n"
            "- Flagging bugs, security vulnerabilities, and correctness issues first\n"
            "- Then noting performance concerns and style issues\n"
            "- Being specific: include file name, line numbers, and the exact problem\n"
            "- Suggesting concrete fixes, not just identifying problems"
        ),
        toolsets=["file"],
        blocked_tools=frozenset([
            "write_file", "edit_file", "create_file",
            "run_command", "bash", "terminal",
        ]),
    ),

    "tester": RoleConfig(
        name="tester",
        identity=(
            "You are a test engineer. You write comprehensive tests that cover happy paths, "
            "edge cases, and error conditions. You understand the code under test before "
            "writing tests, and you run the test suite to confirm everything passes."
        ),
        guidance=(
            "Approach this task by:\n"
            "- Reading the code under test to understand its contract\n"
            "- Writing tests for happy paths, edge cases, and error conditions\n"
            "- Running the tests to confirm they pass\n"
            "- Reporting test coverage and any issues found"
        ),
        toolsets=["file", "terminal"],
    ),
}

VALID_ROLES = sorted(ROLE_REGISTRY)


def get_role_config(role: str) -> RoleConfig:
    """Look up a role by name. Raises ValueError for unknown roles."""
    config = ROLE_REGISTRY.get(role)
    if config is None:
        raise ValueError(
            f"Unknown role '{role}'. Valid roles: {', '.join(VALID_ROLES)}."
        )
    return config
