# Security

## Default: local only

By default, both processes bind to `127.0.0.1` and are not reachable from
your network:

- The LlamaForge dashboard (`backend/server.py`), on `panel_port` (default 8090).
- The llama.cpp router (`llama-server.exe --models-preset ...`), on `router_port`
  (default 8080).

## Opt-in: LAN access for the router

The Setup tab's **Network Access** panel lets you switch the llama.cpp
router's bind address from `127.0.0.1` to `0.0.0.0`, making it reachable from
other devices on your network (e.g. to use the chat UI from your phone, or
point another machine's OpenAI-compatible client at it).

When enabled:

- An **API key is strongly recommended**. A **Require an API key** toggle
  (on by default) makes the panel refuse to enable LAN access until you set
  or generate one; the panel can generate a random key for you. If you
  uncheck the toggle and leave the key blank, the router is served to your
  network **unauthenticated** - anyone who can reach the port can use it.
  When a key is set, clients must send `Authorization: Bearer <key>`.
- `GET /models` (a metadata listing, no prompts or completions) is not
  covered by the key - this is llama.cpp's own behavior, not
  LlamaForge-specific. All inference endpoints (`/completion`,
  `/v1/chat/completions`, etc.) are.
- The setting is saved in `config.json` and re-applied on every restart
  (including autostart), so it persists until you turn it back off.
- Windows Firewall must allow inbound connections to `llama-server.exe` on
  your network profile; if you're prompted by Windows the first time, allow
  it for the profile you're actually on (Private/Public).

The **LlamaForge dashboard itself** (port 8090) always stays local-only -
it can trigger rebuilds, install prerequisites, and edit configuration, so
it is intentionally not exposed by this feature. If you need to administer
LlamaForge from another device, use remote desktop / SSH to this machine
rather than exposing port 8090.

## Reporting

This is a personal/local tool, not a hosted service. If you find a security
issue, please open an issue on the repo.
