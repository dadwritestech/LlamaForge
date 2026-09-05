# Security

## Control plane and router scope

The LlamaForge dashboard and its management API always bind to `127.0.0.1`
(the `panel_port`, default `8090`). They are a local control plane and are not
made reachable on the LAN. The separate llama.cpp router (`router_port`, default
`8080`) has just two LlamaForge-managed scopes: local `127.0.0.1` or LAN
`0.0.0.0`.

The Setup tab's **Network Access** card changes the router scope, not the
dashboard. Use remote desktop or SSH to administer LlamaForge from another
machine; do not expose the panel port.

## LAN access fails closed

Every newly configured LAN router requires a usable API key, and every
LlamaForge-owned router start or restart repeats that policy check before it
stops or starts a process. There is no unauthenticated-LAN override. Clients use
`Authorization: Bearer <key>` when the upstream router requires it.

Network Access has explicit key actions: keep the current key, generate a new
key, replace it, or clear it (clear is available only for local access). Rotating
a key invalidates clients that use the old one. Switching back to local keeps a
key unless the operator separately confirms its removal.

Older printable keys may continue to protect an existing LAN configuration and
are labelled `protected_legacy` with a rotation recommendation. Historical or
manually edited unsupported hosts, and LAN configurations with an absent or
invalid key, are assessed read-only as `unsafe_legacy`: LlamaForge does not
silently rewrite them, but blocks any new start or restart until they are
repaired. If a port is already occupied, the dashboard leaves its listener
alone; seeing a listener is not verification of its process identity or of its
authentication policy.

## Management boundary and credentials

`/api/state`, `/api/config`, and routine management responses never include the
router key. Their public config projection is an explicit allowlist plus the
non-secret `router_api_key_configured` boolean. **Client Config**, **Show
configuration** for an agent, and server-side **Generate** are deliberate,
no-store reveal actions; they return credentials only to the initiating explicit
POST where applicable. Every JSON response has `Cache-Control: no-store`.

The panel treats every HTTP request as untrusted input:

- Host and Origin checks keep requests tied to this loopback service and defend
  against cross-site requests and DNS rebinding.
- When a POST declares `Content-Type`, it must be `application/json`; declared
  form or other non-JSON types are rejected with 415. This is defense in depth
  against form posts. A missing `Content-Type` is not rejected by this guard.
- POST framing requires one valid `Content-Length`; transfer encoding, malformed
  or duplicate lengths, short bodies, and oversized requests are rejected and
  the connection is closed. Management JSON is capped at 4 MiB; the
  `/v1/messages` and `/v1/chat/completions` inference proxies have a 64 MiB cap
  for realistic multimodal payloads.
- `POST /api/config` accepts only a type-checked allowlist. Request data is not
  interpolated into shell commands.

## Local limitations

`config.json` stores `router_api_key` as plaintext; it is not encrypted and is
not an OS credential vault. A process running under the same OS account may be
able to read that file and inspect the llama-server command line, because the
upstream process currently receives `--api-key` in argv. Explicit configuration
previews improve resistance to accidental ambient disclosure, not isolation from
such a local peer process.

## Reporting

This is an early-preview personal/local tool, not a hosted service or a security
certification. If you find a security issue, please open an issue on the repo.
