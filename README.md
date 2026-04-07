<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
</p>

**A fork of [Hermes Agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com), extended with features not yet in the official release.** The upstream agent has a built-in learning loop — it creates skills from experience, improves them during use, nudges itself to persist knowledge, and builds a deepening model of who you are across sessions. This fork adds on top of that:

- **Specialized subagent roles** — delegate with `role="reviewer"` or `role="coder"` and get a subagent with a purpose-built prompt, toolset, and enforced constraints instead of the generic default
- **Subagent result verification** — after a subagent returns, the parent cross-checks file existence claims and flags hallucinated completions before acting on them
- **Semantic session search** — past conversations are vector-embedded and searched by meaning, not just keywords, so Hermes can find "that CORS debugging session" without exact word matches
- **Shared memory between subagents** — a session-scoped scratch space parallel subagents can read and write to share discovered facts without touching `MEMORY.md`

Everything else — multiplatform gateway, cron scheduling, memory, skills, terminal backends — comes from upstream and is documented at [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/).

Use any model you want — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200+ models), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), OpenAI, or your own endpoint. Switch with `hermes model` — no code changes, no lock-in.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. Hybrid keyword + semantic (vector embedding) session search for cross-session recall — finds "that time I debugged a CORS issue" even without exact words. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams with specialized roles (researcher, coder, reviewer, tester) — each with a tailored system prompt, toolset, and constraints. Parallel subagents share a session-scoped scratch space (<code>shared_memory=True</code>) to exchange findings without polluting long-term memory. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Six terminal backends — local, Docker, SSH, Daytona, Singularity, and Modal. Daytona and Modal offer serverless persistence — your agent's environment hibernates when idle and wakes on demand, costing nearly nothing between sessions. Run it on a $5 VPS or a GPU cluster.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, Atropos RL environments, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Works on Linux, macOS, and WSL2. The installer handles everything — Python, Node.js, dependencies, and the `hermes` command. No prerequisites except git.

> **Windows:** Native Windows is not supported. Please install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and run the command above.

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

---

## Getting Started

```bash
hermes              # Interactive CLI — start a conversation
hermes model        # Choose your LLM provider and model
hermes tools        # Configure which tools are enabled
hermes config set   # Set individual config values
hermes gateway      # Start the messaging gateway (Telegram, Discord, etc.)
hermes setup        # Run the full setup wizard (configures everything at once)
hermes claw migrate # Migrate from OpenClaw (if coming from OpenClaw)
hermes update       # Update to the latest version
hermes doctor       # Diagnose any issues
```

📖 **[Full documentation →](https://hermes-agent.nousresearch.com/docs/)**

## CLI vs Messaging Quick Reference

Hermes has two entry points: start the terminal UI with `hermes`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action | CLI | Messaging platforms |
|---------|-----|---------------------|
| Start chatting | `hermes` | Run `hermes gateway setup` + `hermes gateway start`, then send the bot a message |
| Start fresh conversation | `/new` or `/reset` | `/new` or `/reset` |
| Change model | `/model [provider:model]` | `/model [provider:model]` |
| Set a personality | `/personality [name]` | `/personality [name]` |
| Retry or undo the last turn | `/retry`, `/undo` | `/retry`, `/undo` |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| Browse skills | `/skills` or `/<skill-name>` | `/skills` or `/<skill-name>` |
| Interrupt current work | `Ctrl+C` or send a new message | `/stop` or send a new message |
| Platform-specific status | `/platforms` | `/status`, `/sethome` |

For the full command lists, see the [CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli) and the [Messaging Gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Documentation

All documentation lives at **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Section | What's Covered |
|---------|---------------|
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | Install → setup → first conversation in 2 minutes |
| [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | Commands, keybindings, personalities, sessions |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | Config file, providers, models, all options |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security) | Command approval, DM pairing, container isolation |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ tools, toolset system, terminal backends |
| [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | Procedural memory, Skills Hub, creating skills |
| [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | Persistent memory, user profiles, best practices |
| [MCP Integration](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | Connect any MCP server for extended capabilities |
| [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | Scheduled tasks with platform delivery |
| [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Project context that shapes every conversation |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | Project structure, agent loop, key classes |
| [Contributing](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | Development setup, PR process, code style |
| [CLI Reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | All commands and flags |
| [Environment Variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Complete env var reference |

---

## Migrating from OpenClaw

If you're coming from OpenClaw, Hermes can automatically import your settings, memories, skills, and API keys.

**During first-time setup:** The setup wizard (`hermes setup`) automatically detects `~/.openclaw` and offers to migrate before configuration begins.

**Anytime after install:**

```bash
hermes claw migrate              # Interactive migration (full preset)
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
```

What gets imported:
- **SOUL.md** — persona file
- **Memories** — MEMORY.md and USER.md entries
- **Skills** — user-created skills → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — approval patterns
- **Messaging settings** — platform configs, allowed users, working directory
- **API keys** — allowlisted secrets (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — workspace audio files
- **Workspace instructions** — AGENTS.md (with `--workspace-target`)

See `hermes claw migrate --help` for all options, or use the `openclaw-migration` skill for an interactive agent-guided migration with dry-run previews.

---

## Semantic Session Search

Hermes searches past conversations using both keyword matching (FTS5) and vector embeddings, merged with Reciprocal Rank Fusion. This means it can surface "that time I debugged a similar CORS issue" automatically — without requiring you to remember the exact words used.

**How it works:**

1. Each past session is embedded once (title + first user message + first assistant response) and stored in `~/.hermes/state.db`.
2. When you search, the query is embedded with the same model and compared against all stored session embeddings by cosine similarity.
3. Semantic results are merged with FTS5 keyword results — sessions appearing in both rank higher.
4. The top matches are summarized by an auxiliary LLM and returned.

Embeddings are computed **lazily** — missing sessions are embedded at search time (up to 200 per call, within 8 seconds) and cached permanently. No background jobs, no blocking session writes.

**Enabling semantic search:**

Requires one optional library (no PyTorch needed for the recommended option):

```bash
pip install fastembed          # recommended — ONNX-based, ~50 MB model download on first use
# or
pip install sentence-transformers  # PyTorch-based alternative
```

If neither is installed, `session_search` falls back transparently to FTS5 keyword search — same behavior as before.

**Query style:**

| Goal | Example query |
|---|---|
| Semantic (natural language) | `"debugging auth token expiry"` |
| Keyword (exact terms) | `"JWT OR token OR expiry"` |
| Phrase | `"docker networking"` |
| Boolean | `"python NOT java"` |

The result JSON includes a `search_mode` field — `"keyword"` when only FTS5 ran, `"hybrid"` when both ran.

---

## Subagent Roles

When delegating tasks, the parent agent can specify a `role` to give the subagent a purpose-built identity, behavioral guidance, and a matching default toolset — instead of the generic "focused subagent" prompt.

```
delegate_task(
  goal="Review the authentication module for security issues",
  role="reviewer"
)
```

Four built-in roles:

| Role | Identity | Default toolset | Notable constraints |
|---|---|---|---|
| `researcher` | Research specialist — synthesizes information from multiple sources, cites evidence | `web`, `file` | — |
| `coder` | Expert software engineer — reads before modifying, makes targeted changes, verifies with tests | `file`, `terminal`, `web` | — |
| `reviewer` | Code reviewer — reads critically, reports bugs/security issues with file + line citations | `file` only | Write tools structurally blocked |
| `tester` | Test engineer — writes tests covering happy paths, edge cases, and errors; runs the suite | `file`, `terminal` | — |

**How roles compose with other parameters:**

- **Toolsets:** Caller-explicit `toolsets` > role defaults > parent's inherited toolsets. A role can never grant tools the parent doesn't have.
- **`max_iterations`:** Caller-explicit > role default (if set) > `delegation.max_iterations` config.
- **Model:** `delegation.model` config > role default (if set) > parent model.
- **Batch tasks:** Each task in a `tasks` array can set its own `role`, overriding the top-level role for that task only.

The `reviewer` role enforces read-only behavior structurally: write tools (`write_file`, `edit_file`, `create_file`, `run_command`) are removed from the child's valid tool list at construction time, not just by prompt instruction.

To add a custom role, extend `ROLE_REGISTRY` in `tools/roles.py` with a `RoleConfig` dataclass entry.

---

## Shared Memory Between Subagents

When parallel subagents need to share discovered facts without contaminating long-term memory, enable `shared_memory=True` on a delegation call.

```
delegate_task(
  tasks=[
    {"goal": "Find the root cause of the auth regression", "role": "researcher"},
    {"goal": "Audit all endpoints that touch the auth module", "role": "reviewer"},
  ],
  shared_memory=True
)
```

All subagents in the batch share a single session-scoped key-value store. Each subagent can read from and write to it concurrently using three tools:

| Tool | Purpose |
|---|---|
| `shared_memory_write` | Store a fact under a short snake_case key (max 2000 chars) |
| `shared_memory_read` | Read one key, or all entries if no key given |
| `shared_memory_delete` | Remove a stale or incorrect entry |

**How it works:**

- The store is created when `delegate_task` is called and destroyed when the call ends — nothing persists to `MEMORY.md` or any database.
- Values are scanned with the same injection-detection logic as long-term memory before being accepted.
- All reads and writes are thread-safe (backed by `threading.RLock`), so parallel subagents can't corrupt each other's data.
- After all subagents finish, the final store contents are returned to the parent in the result JSON as `shared_memory: {key: value, ...}`.

**Capacity limits** (per delegation call): 100 entries, 64-char keys, 2000-char values.

The shared memory toolset is structurally blocked from the parent and from non-shared-memory delegation calls — a subagent cannot access or pollute another call's store.

---

## Contributing

We welcome contributions! See the [Contributing Guide](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) for development setup, code style, and PR process.

Quick start for contributors:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

> **RL Training (optional):** To work on the RL/Tinker-Atropos integration:
> ```bash
> git submodule update --init tinker-atropos
> uv pip install -e "./tinker-atropos"
> ```

---

## Community

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 💡 [Discussions](https://github.com/NousResearch/hermes-agent/discussions)

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com).
