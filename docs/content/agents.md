---
title: Connect an Agent
section: guides
order: 8
---

# Connect an Agent

Point a coding agent — Claude Code, Codex, or pi.dev — at your locally loaded models with one click, using either the Anthropic-compatible shim or the OpenAI-compatible router directly.

## What it does

LlamaForge exposes two HTTP surfaces a coding agent can talk to:

- The **router** (`router_port`, default 8080) speaks the OpenAI `/v1/chat/completions` shape natively — this is what llama.cpp's router already serves.
- The **Anthropic shim** (`backend/anthropic_shim.py`, served from the panel on `panel_port`, default 8090) translates Anthropic's `/v1/messages` Messages API into the router's OpenAI request shape and back, including streaming SSE and tool use, so an Anthropic-API client like Claude Code can drive a local model without knowing it isn't talking to Anthropic. `to_openai_request()` maps system/messages/tools/tool_choice/stop_sequences into the OpenAI body; `to_anthropic_response()` and `stream_anthropic_events()` map the router's response (or SSE chunk stream) back into Anthropic message/content-block events. `/v1/messages/count_tokens` is served too, but only as an estimate (`count_tokens_estimate()`: ~4 characters per token over the translated prompt text, not a real tokenizer count).

Shim auth (`backend/server.py` `_shim_auth_ok()`) accepts either header an Anthropic client might send: `x-api-key: <key>` or `Authorization: Bearer <key>`. The panel itself remains loopback-only. Network Access requires a usable router key for every newly configured LAN router; the shim forwards the configured credential to the router where needed.

**Agent setup** (`backend/agentsetup.py`) generates the config each agent needs to point at LlamaForge, and can optionally write it in place:

| Agent | Config file | Format | Endpoint it's given |
|---|---|---|---|
| `claude-code` | `~/.claude/settings.json` | JSON (`env` block) | The loopback Anthropic endpoint, `http://127.0.0.1:<panel_port>` — always local, never the LAN. |
| `codex` | `~/.codex/config.toml` | TOML (appended provider block) | Direct mode uses the router's OpenAI-compatible endpoint, `http://<host>:<router_port>/v1`; injected mode uses the loopback panel endpoint `http://127.0.0.1:<panel_port>/v1`. |
| `pi` | `~/.pi/agent/models.json` | JSON (`providers` block) | The same direct-router or injected-loopback choice as Codex. |

Claude Code's generated `settings.json` sets `ANTHROPIC_BASE_URL` to the shim endpoint, `ANTHROPIC_AUTH_TOKEN` to the router API key (or the literal string `llamaforge` if none is set — the shim needs a non-empty token even when auth is effectively open), `ANTHROPIC_MODEL`, and `ANTHROPIC_SMALL_FAST_MODEL`. Codex's TOML block declares a `[model_providers.llamaforge]` section with `wire_api = "chat"` and, if a router API key exists, an `env_key` pointing at a `LLAMAFORGE_API_KEY` environment variable the user must set — the key itself is never written to the TOML file. pi's `models.json` sets `api: "openai-completions"` with the key embedded directly in the config.

Codex and pi can optionally be routed through the loopback panel proxy with
`inject=true` — this is how those two agents pick up wiki context injection (see
[Context Wiki](context-wiki.md)). An injected endpoint is local-machine-only;
direct Codex/pi configuration uses the router endpoint instead.

Preview generation returns config content and a human-readable target path/instructions without touching disk (used for "copy this into your config" display). Apply writes it — JSON targets are deep-merged into any existing file (`_deep_merge()`, so unrelated existing keys survive), and the Codex TOML target is appended only if the `[model_providers.llamaforge]` block isn't already present, commenting out any conflicting top-level `model`/`model_provider` line rather than deleting it. Apply backs up the target once, to `<path>.llamaforge.bak`, before its first write.

The preview may deliberately contain the Claude Code or pi credential needed for
that agent's native file. Treat it like Client Config: it is shown only after the
explicit POST request. Apply may use the stored key to write a file, but its
response never returns the key.

## How to use it

1. Open the **Setup** tab and pick Claude Code, Codex, or pi.dev.
2. Choose a model from the active llama-family engine (and, for Claude Code, an optional small/fast model). vLLM agent setup is deferred.
3. Press **Show configuration**. This is the only preview action: it is an explicit POST and is not fetched on render or selection change. For Codex or pi.dev, choose injection only when the agent runs on this same machine.
4. Review or copy the preview, then click **Apply** only to write the config into the agent's real local config file.
5. Launch the agent. Claude Code must run on the same machine as LlamaForge — its endpoint is always `127.0.0.1`.

## Reference

| Concept | Source | Behavior |
|---|---|---|
| Anthropic request translation | `backend/anthropic_shim.py` `to_openai_request()` | Anthropic Messages body -> OpenAI chat body (system, messages incl. tool_use/tool_result, tools, tool_choice, stop_sequences). |
| Anthropic response translation | `backend/anthropic_shim.py` `to_anthropic_response()` / `stream_anthropic_events()` | OpenAI response/SSE -> Anthropic message / streaming content-block events. |
| Token estimate | `backend/anthropic_shim.py` `count_tokens_estimate()` | Advisory only: `len(text) // 4` over the translated prompt. |
| Shim auth | `backend/server.py` `_shim_auth_ok()` | Accepts `x-api-key` or `Authorization: Bearer <key>`; skipped when `router_host` is `127.0.0.1` or no `router_api_key` is set. |
| Endpoint per agent | `backend/routes.py` `_agent_endpoint_for()` | Claude Code -> loopback Anthropic endpoint; direct Codex/pi -> router at `<host>:router_port/v1`; injected Codex/pi -> loopback panel at `127.0.0.1:panel_port/v1`. |
| Config preview | `POST /api/agent/config` | Explicit POST with `{agent, model, backend, small, inject}`; may deliberately contain Claude/pi credentials and does not write files. |
| Config apply | `POST /api/agent/apply` | Same targeting fields; writes to the real target path, JSON is deep-merged and Codex TOML appended if absent, original is backed up once. Apply never returns a key. |

## Troubleshooting

If Claude Code can't reach LlamaForge, confirm it's running on the same machine — the shim endpoint is hardcoded to `127.0.0.1` and never the LAN IP, by design (`_agent_endpoint()`: "shim binds localhost only"). If Codex or pi.dev requests fail auth after changing `router_host`, make sure `LLAMAFORGE_API_KEY` (Codex) or the embedded `apiKey` (pi) actually matches the current `router_api_key` in config — a stale key set before you edited config will keep failing. If re-applying a config seems to have no effect, check `<path>.llamaforge.bak` next to the target file: it holds the original from before LlamaForge ever touched it, since the backup is only ever written once.

See also [Context Wiki](context-wiki.md) for how injected context reaches requests that go through the shim or router proxy.
