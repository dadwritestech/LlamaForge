---
title: MCP Server
section: guides
order: 11
---

# MCP Server

LlamaForge ships an MCP server, so Claude Code, Codex or any MCP client can drive it: see what's loaded, load and unload models, check fit, search and download from Hugging Face, and hand a task to a local model.

## What it does

`backend/mcp_server.py` is a stdio MCP server (newline-delimited JSON-RPC, stdlib only). It's a thin client over the panel's own HTTP API on `127.0.0.1:<panel_port>`. The panel admits a loopback caller that sends no `Origin`, so there's no token to manage and no new way in. It reads `config.json` for the panel port, pi's location and the router key, and never writes it.

### Tools

| Tool | What it does |
|---|---|
| `status` | Version, engine, loaded models (role, OpenAI-compatible endpoint, context size), models still loading, GPU memory, multi-model mode. |
| `list_models` | Every installed model with status, size on disk, context size, engine and pinned build. |
| `load_model` | Loads a model and waits until it's serving, or returns the diagnosed cause if it fails. |
| `unload_model` | Unloads a model and frees its memory. |
| `fit_check` | Multi-model mode: would this model fit beside what's loaded, on which GPUs, and what would have to be evicted. |
| `stats` | Per-model throughput and request counts from the router's `/metrics`. |
| `diagnose` | Why a model failed to load (known causes with a suggested fix) plus the tail of the router log. |
| `search_models` | Searches Hugging Face for GGUF repos and marks the ones already installed. |
| `list_files` | GGUF files in a repo: size, shard count, whether each fits this machine's GPUs, plus mmproj (vision) and MTP companions. |
| `download_model` | Starts a download. The model registers itself when done, ready for `load_model`. |
| `download_progress` | Progress of the current or last download. |
| `ask` | One prompt to a loaded local model, no tools. |
| `pi_run` | Hands a whole task to [pi](https://github.com/earendil-works/pi) (Mario Zechner's MIT coding agent) running on a loaded model, and returns its final answer. |
| `forge_start` | Starts an [Auto-Forge](autoforge.md) run on one model: it measures GPU placements, KV cache types and drafters and keeps the fastest. It unloads every model for the run, so the description tells the agent to ask you first. |
| `forge_status` | The run's stage and the top five setups (tokens/s, VRAM, context). Once it's done, the before/after numbers and a postable summary with no paths, keys or hosts in it. |
| `forge_apply` | Writes the winning setup to the model. The agent only calls it when you say so. |
| `forge_undo` | Puts back the settings the last apply replaced. |

### Agents don't take your main model

In multi-model mode, `load_model` loads beside what's running (`role=worker`) by default, and only if it fits. It evicts other workers only when called with `evict=true`. Replacing your main model takes `role=main`, and the tool description tells the agent to do that only when you ask. `ask` and `pi_run` default to a loaded worker in multi-model mode, otherwise the loaded model.

### pi_run tool sets

| `tools` | pi can |
|---|---|
| `read` (default) | read, grep, find, ls. Changes nothing. |
| `edit` | also edit and write files. |
| `full` | also run shell commands. |

pi works in `cwd`, which defaults to where the MCP server was started (usually your project). It needs pi installed: **Setup → pi coding agent → Install pi**, or `npm install -g @earendil-works/pi-coding-agent`.

## How to use it

Open **Setup → MCP server**. It shows ready-to-paste configs for this install, with the Python and script paths already filled in:

- **Claude Code**: a `claude mcp add --scope user llamaforge -- <python> <script>` command.
- **Codex**: a `[mcp_servers.llamaforge]` block for `config.toml`. It sets `tool_timeout_sec = 900`, because `pi_run` can work for minutes and Codex's default tool timeout is 60s.
- **Any other client**: an `mcpServers` JSON block.

None of them contain secrets; the server reads the router key from `config.json` itself. The LlamaForge panel has to be running for the tools to work.

## Troubleshooting

- **Every tool fails to connect**: the panel isn't running, or it's on a different port than `config.json` says. Start LlamaForge and try again.
- **`pi_run` says pi isn't installed**: install it from **Setup → pi coding agent**.
- **`pi_run` times out in Codex**: raise `tool_timeout_sec` in the Codex config.

See also [Connect an Agent](agents.md), [Embers](embers.md) and [HTTP API](api.md).
