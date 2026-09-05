---
title: Introduction
section: getting-started
order: 1
---

# LlamaForge

LlamaForge is a graphical control panel for [llama.cpp](https://github.com/ggml-org/llama.cpp). It builds llama.cpp for your machine, keeps it current with upstream, discovers models that fit your hardware, tunes every server parameter per model, and runs it — all from a browser instead of hand-editing `models.ini` and long `llama-server` command lines.

LlamaForge contains no llama.cpp source code. The backend (`backend/server.py`, pure Python standard library) proxies llama.cpp's own router API, edits `models.ini`, and shells out to `git`, `cmake`, `nvidia-smi`, and the platform package manager. It also drives **vLLM** as a second, optional backend for full-precision and safetensors models.

## Who it is for

LlamaForge is built for people who want llama.cpp's speed and control but would rather not memorize flags, edit config files by hand, or babysit build commands. It assumes you are comfortable running a setup script once and building llama.cpp for your machine — both are guided from the dashboard.

Windows with an NVIDIA GPU is the primary target (CPU-only also works). Linux (NVIDIA/CPU) and macOS (Apple Silicon, Metal) are supported as an early preview — the same dashboard, launched with `bootstrap.sh` instead of `bootstrap.ps1`.

> [!NOTE]
> If you want a zero-config, double-click installer with no compile step, [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com), or [Jan](https://jan.ai) will get you running faster. LlamaForge trades that convenience for direct, per-model control over the real llama.cpp server.

## Architecture

LlamaForge runs two local HTTP services:

| Component | Default address | Role |
|---|---|---|
| Dashboard (panel) | `http://127.0.0.1:8090` | The LlamaForge backend and web UI — Models, Stats, Discover, Build, Setup tabs. Always binds to `127.0.0.1` only. |
| Router | `http://127.0.0.1:8080` | llama.cpp's own server process, started by LlamaForge with `--models-preset models.ini`. Serves the OpenAI-compatible API and llama.cpp's chat UI. |

Both ports, along with the bind address, are configured by the `panel_port`, `router_port`, and `router_host` keys in `config.json` (defaults `8090`, `8080`, and `127.0.0.1`). The Setup tab's Network Access panel supports only local `127.0.0.1` and LAN `0.0.0.0` router scopes. LAN requires a usable API key and LlamaForge-owned starts fail closed until it is configured; the dashboard itself never leaves `127.0.0.1`.

Clients — `curl`, an OpenAI SDK, or any OpenAI-compatible chat client — talk to the router, not the dashboard. The dashboard's job is configuration: it writes model presets into `models.ini`, starts and stops the router, and reads back the router's own metrics endpoint for the Stats tab.

When vLLM is enabled (Windows only, via WSL2), it runs as a third process inside WSL and is bridged back to a Windows localhost port (`vllm_port` in `config.json`); it shares the same Models list, Discover tab, and stats as llama.cpp.

## Where to go next

Start with [Installation](install.md) to get the bootstrap script running on your platform, then [First Run](first-run.md) for the onboarding wizard and auto-tune.
