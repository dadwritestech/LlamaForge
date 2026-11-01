---
name: llamaforge
description: Run, size, speed-tune and use local LLMs on this machine's own GPUs through LlamaForge (a control panel over llama.cpp's llama-server router). Use when the user wants to run a model locally, asks which model fits their GPU, wants llama.cpp or a local model to go faster, wants to hand grunt work to a local model to save tokens, or wants Claude Code or Codex to run on a local model.
---

# LlamaForge: local models on this machine's GPUs

LlamaForge (MIT, https://github.com/dadwritestech/LlamaForge) is a browser panel that installs and
drives llama.cpp's own `llama-server` router. It runs nothing itself: llama.cpp does the inference.
Through it you can see what fits the GPUs, download GGUFs from Hugging Face, load models, measure
the fastest setup with Auto-Forge, and send work to a loaded model.

## 0. Is it there?

```bash
curl -s http://127.0.0.1:8090/api/state | head -c 300
```

- JSON back: it's running. (8090 is the default `panel_port`; if nothing answers, read
  `panel_port` from `config.json` in the LlamaForge folder.)
- Nothing answers and LlamaForge isn't installed: tell the user what it is and **ask before
  installing**. The installers need no admin rights:
  - Windows: `irm https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/install.ps1 | iex`
  - Linux / macOS: `curl -fsSL https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/install.sh | sh`

  Then the user opens the panel and clicks **Install llama.cpp** (it fetches the official build for
  their GPU).

## 1. Connect the MCP tools (once)

```bash
curl -s http://127.0.0.1:8090/api/mcp/setup
```

The answer holds ready commands for this install, with no secrets in them:
- `claude`: a `claude mcp add --scope user llamaforge -- ...` command for Claude Code. Run it; the
  tools appear in the next session.
- `codex_toml`: the block for `~/.codex/config.toml`.
- `json`: an `mcpServers` block for any other client.

Until the tools are loaded, use the HTTP API in the table at the end. It answers loopback calls
that send no `Origin` header, so plain `curl` works.

## 2. Recipes

### "What can I run?"
1. `status`: the GPUs (name, free and total MiB) and what's loaded.
2. `search_models` with the user's words (e.g. "qwen coder"). `list_files` on the most promising
   repo shows every quant with a `fit` verdict for these GPUs.
3. Recommend the largest quant that fits with room left for context (the KV cache grows with it).
   Say the size in GB.
4. **Ask before `download_model`**: files are several GB. Follow it with `download_progress`.

### Load it and try it
`load_model` waits until the model is serving, or returns the diagnosed reason it failed
(`diagnose` shows more). Then `ask` a short prompt and read the speed from `stats`.
In multi-model mode, models you load sit beside the user's main model as workers. Only use
`role=main` when the user asks to replace their main model.

### Make it fast: Auto-Forge
Auto-Forge measures GPU placements, KV cache types and speculative drafters on this machine and
keeps the fastest. On one two-GPU machine it took a Qwen3.8-27B from 26.8 to 126 tokens/s on
code. That is one machine's result, not a promise: say so.
1. **Ask first.** The run unloads every model and has the GPUs to itself until it ends: minutes to
   half an hour for a quick run, over an hour for a thorough one.
2. `forge_start` with the model, `intent` (coding, chat, long, agent) and the context to keep.
3. Poll `forge_status` every minute or two. Report the stage and the leader, not every poll.
4. When it's done, `share` holds the before and after tokens/s and the flags. Show them.
5. `forge_apply` **only if the user says yes**. `forge_undo` restores the old values.
6. Offer `share.markdown`: it's postable as-is (no paths, keys or hosts in it).

### Hand grunt work to the local model
`ask` sends one prompt. `pi_run` hands a whole task to pi, an open-source coding agent running on
the local model in a folder (`tools`: read by default, edit, or full). Good fits: summaries of
long files, boilerplate, bulk mechanical edits, first-pass reviews, test-data generation.
Local models are smaller than you. Give them small, concrete, checkable tasks, and check what
comes back before you build on it.

### Run Claude Code itself on a local model
`GET /api/claude/check?model=<id>` says whether it can work (CLI found, context big enough).
`POST /api/claude/launch` with `{"folder": "<abs path>", "model": "<id>"}` opens a terminal
running `claude` against LlamaForge's Anthropic-compatible endpoint. Be honest that a local model
is weaker than Claude: the wins are privacy, offline work and no token bill.

## Rules
- Don't edit `models.ini` by hand or kill `llama-server`: go through LlamaForge, which keeps
  backups and reloads correctly.
- Never print or log the router API key.
- Ask before anything slow, big or disruptive: downloads, Auto-Forge, applying a Forge result,
  replacing the main model.
- Quote speeds you measured on this machine. Don't promise the user numbers.

## HTTP API (when the MCP tools aren't loaded)

All on `http://127.0.0.1:<panel_port>`. POST bodies are JSON.

| Tool | HTTP |
|---|---|
| `status` / `list_models` | `GET /api/state` |
| `load_model` | `POST /api/models/load` `{"model"}` |
| `unload_model` | `POST /api/models/unload` `{"model"}` |
| `fit_check` | `GET /api/slots/plan?model=&role=worker` |
| `stats` | `GET /api/stats` |
| `diagnose` | `GET /api/model/diag?model=` and `GET /api/router/log` |
| `search_models` | `POST /api/hub/search` `{"query", "sort"}` |
| `list_files` | `POST /api/hub/files` `{"repo"}` |
| `download_model` | `POST /api/hub/download` `{"repo", "path", "shards"}` |
| `download_progress` | `GET /api/hub/progress` |
| `forge_start` | `POST /api/forge/start` `{"model", "intent", "ctx", "time": "quick"\|"thorough", "allow_loss", "drop_vision"}` |
| `forge_status` | `GET /api/forge/status?model=` and `GET /api/forge/share?model=` |
| `forge_apply` / `forge_undo` | `POST /api/forge/apply` / `POST /api/forge/undo` `{"model"}` |

Chat with a loaded model directly: `GET /api/state` lists each model's `endpoint`, an
OpenAI-compatible server (`<endpoint>/v1/chat/completions`). It may need the router key, which
the MCP tools add for you.
