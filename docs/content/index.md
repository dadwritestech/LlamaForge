---
title: Introduction
section: getting-started
order: 1
---

# LlamaForge

LlamaForge runs the real [llama.cpp](https://github.com/ggml-org/llama.cpp) server from a browser tab. Pick a model, see if it fits your GPU, load it, chat. Every `llama-server` flag is still there when you want it, without hand-editing `models.ini` or long command lines.

LlamaForge contains no llama.cpp source code. The backend (`backend/server.py`, pure Python standard library) drives llama.cpp's own router API, edits `models.ini`, fetches the official llama.cpp build for your GPU, and shells out to `git` / `cmake` only if you build from source. It also drives **vLLM** as a second, optional backend for full-precision and safetensors models.

## Who it is for

LlamaForge is built for people who want llama.cpp's speed and control but would rather not memorize flags, edit config files by hand, or babysit build commands. Install is one line with no admin and no compiler, and **Install llama.cpp** in the dashboard fetches the official build for your GPU.

Windows with an NVIDIA GPU is the most-tested path; AMD/Intel (Vulkan) and CPU-only work too. Linux and macOS (Apple Silicon, Metal) are an early preview: the same dashboard and a one-line installer, CI-tested but with little real-hardware use so far.

> [!NOTE]
> If you'd rather have polish than control, [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com), or [Jan](https://jan.ai) are more mature. LlamaForge trades that for direct, per-model control over the real llama.cpp server.

## Architecture

LlamaForge runs three local HTTP services:

| Component | Default address | Role |
|---|---|---|
| Dashboard (panel) | `http://127.0.0.1:8090` | The LlamaForge backend and web UI — Models, Stats, Discover, Build, Setup tabs. Always binds to `127.0.0.1` only. |
| Router | `http://127.0.0.1:8080` | llama.cpp's own server process, started by LlamaForge with `--models-preset models.ini`. Serves the OpenAI-compatible API. Always runs with an API key. |
| Chat | `http://127.0.0.1:8091` | llama.cpp's chat UI on its own origin, shown in the dashboard's Chat tab. The router key is added server-side. |

The ports and the router's bind address are configured by the `panel_port`, `router_port`, `chat_port` and `router_host` keys in `config.json` (defaults `8090`, `8080`, `8091` and `127.0.0.1`). The Setup tab's Network Access panel supports only local `127.0.0.1` and LAN `0.0.0.0` router scopes. LAN requires a usable API key and LlamaForge-owned starts fail closed until it is configured; the dashboard itself never leaves `127.0.0.1`.

Clients — `curl`, an OpenAI SDK, or any OpenAI-compatible chat client — talk to the router, not the dashboard. The dashboard's job is configuration: it writes model presets into `models.ini`, starts and stops the router, and reads back the router's own metrics endpoint for the Stats tab.

When vLLM is enabled (Windows only, via WSL2), it runs as another process inside WSL and is bridged back to a Windows localhost port (`vllm_port` in `config.json`); it shares the same Models list, Discover tab, and stats as llama.cpp.

## Where to go next

Start with [Installation](install.md) for the one-line installer, then [First Run](first-run.md) for the onboarding wizard and auto-tune.
