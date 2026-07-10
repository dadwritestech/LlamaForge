<p align="center">
  <img src="docs/hero.png" alt="LlamaForge - the full GUI experience on top of llama.cpp" width="100%">
</p>

<p align="center">
  <a href="https://github.com/ggml-org/llama.cpp"><img alt="powered by llama.cpp" src="https://img.shields.io/badge/powered%20by-llama.cpp-ffb000?style=flat-square&labelColor=0f1315"></a>
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows%2010%2F11-3fd7e6?style=flat-square&labelColor=0f1315">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B%20%C2%B7%20zero%20deps-39d98a?style=flat-square&labelColor=0f1315">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-c8d2d4?style=flat-square&labelColor=0f1315"></a>
  <img alt="status" src="https://img.shields.io/badge/status-early%20preview-ff5c57?style=flat-square&labelColor=0f1315">
</p>

<p align="center">
  <a href="https://github.com/dadwritestech/LlamaForge/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/dadwritestech/LlamaForge/ci.yml?branch=master&style=flat-square&labelColor=0f1315&color=39d98a&label=CI"></a>
  <a href="https://github.com/dadwritestech/LlamaForge/stargazers"><img alt="stars" src="https://img.shields.io/github/stars/dadwritestech/LlamaForge?style=flat-square&labelColor=0f1315&color=ffb000&cacheSeconds=1800"></a>
  <a href="https://github.com/dadwritestech/LlamaForge/network/members"><img alt="forks" src="https://img.shields.io/github/forks/dadwritestech/LlamaForge?style=flat-square&labelColor=0f1315&color=3fd7e6&cacheSeconds=1800"></a>
  <a href="https://github.com/dadwritestech/LlamaForge/issues"><img alt="open issues" src="https://img.shields.io/github/issues/dadwritestech/LlamaForge?style=flat-square&labelColor=0f1315&color=39d98a"></a>
  <a href="https://github.com/dadwritestech/LlamaForge/pulls"><img alt="pull requests" src="https://img.shields.io/github/issues-pr/dadwritestech/LlamaForge?style=flat-square&labelColor=0f1315&color=39d98a"></a>
  <a href="https://github.com/dadwritestech/LlamaForge/commits/master"><img alt="last commit" src="https://img.shields.io/github/last-commit/dadwritestech/LlamaForge?style=flat-square&labelColor=0f1315&color=c8d2d4"></a>
  <img alt="repo size" src="https://img.shields.io/github/repo-size/dadwritestech/LlamaForge?style=flat-square&labelColor=0f1315&color=6b7a7e&cacheSeconds=1800">
</p>

# LlamaForge

A graphical control panel that sits on top of [llama.cpp](https://github.com/ggml-org/llama.cpp):
build it, keep it current with upstream, discover models that fit your hardware,
tune **every** server parameter per model, and run - all from a browser, no
command line.

> LlamaForge is an independent wrapper and is **not affiliated with llama.cpp / ggml-org**.
> All inference, model support, and performance come from llama.cpp (MIT, (c) The ggml
> authors). See [NOTICE](NOTICE). Please support the upstream project.

## Features

| Tab | What it does |
|-----|--------------|
| **Models** | Every model on your machine in one list with live GPU VRAM/util/temp meters. Expand a model to edit all **~220 llama.cpp knobs** (context, KV-cache type, speculative decoding, tensor split, sampling, rope, ...), grouped and searchable. Save hot-reloads with no restart; load/unload in a click. |
| **Discover** | Search **huggingface.co** for GGUF models (newest / most downloaded / most liked). Every quant is rated against your total VRAM - **FITS / TIGHT / CPU OFFLOAD** - before you download. One click streams the download (multi-shard + vision mmproj handled) and registers it in your registry. |
| **Build / Update** | Shows your current llama.cpp commit, checks GitHub for how far behind you are, and rebuilds via CMake with flags **auto-detected for your CPU/GPU** (CUDA arch, AVX-512, quantized-KV flash attention). Prior binaries are backed up; the build streams live. |
| **Setup** | Checks prerequisites (Git, CMake, Ninja, Python, MSVC, CUDA), installs missing ones **with your permission** (winget/choco) or links official downloads. Detects hardware and scans all drives for existing GGUF models. **Check for deleted models** prunes registry entries whose file has since been removed from disk. |

## Screenshots

| Setup & hardware detection | Discover with VRAM-fit ratings |
|---|---|
| ![Setup tab](docs/screenshot.png) | ![Discover tab](docs/screenshot-discover.png) |

## Quick start (new machine)

```powershell
git clone https://github.com/dadwritestech/LlamaForge
cd LlamaForge
powershell -ExecutionPolicy Bypass -File bootstrap.ps1
```

`bootstrap.ps1` ensures Python + Git (asking before installing anything), fetches
llama.cpp if you don't have it, writes `config.json`, and opens the dashboard. From
there: **Setup** to install any missing compiler/CUDA and scan your drives, **Build**
to compile llama.cpp for your hardware, **Models** to tune and run.

## Daily use

Double-click **`LlamaForge.vbs`**. It starts the llama.cpp router and the dashboard
hidden, then opens your browser. For autostart, put a shortcut to it in your Startup
folder (`Win+R` -> `shell:startup`).

- Dashboard: http://127.0.0.1:8090
- llama.cpp chat UI + OpenAI-compatible API: http://127.0.0.1:8080

To shut everything down, run **`stop.ps1`**. It reads the ports from `config.json`
and stops the dashboard, the router, and every model instance the router spawned:

```powershell
powershell -ExecutionPolicy Bypass -File stop.ps1
```

## Requirements

- Windows 10/11
- Python 3.10+ (backend is **pure stdlib** - nothing to `pip install`)
- NVIDIA GPU for CUDA acceleration (CPU-only builds also supported)
- Everything else (Git, CMake, Ninja, MSVC Build Tools, CUDA) is detected and can be
  installed from the Setup tab

## Configuration

All machine-specific paths live in `config.json` (see `config.example.json`):

| key | meaning |
|-----|---------|
| `llama_src` | your llama.cpp git checkout |
| `build_dir` | CMake build directory |
| `server_bin` | path to `llama-server.exe` |
| `models_ini` | the router preset file LlamaForge edits |
| `model_dirs` | directories to scan for GGUFs (empty = all fixed drives) |
| `router_port` / `panel_port` | ports for llama.cpp and the dashboard |
| `router_host` | `127.0.0.1` (default, local only) or `0.0.0.0` (reachable on your LAN) |
| `router_api_key` | key clients send as `Authorization: Bearer <key>`; strongly recommended (and enforceable) whenever `router_host` isn't `127.0.0.1` |

By default everything binds to `127.0.0.1` only. The Setup tab has a **Network
Access** panel to opt into serving the llama.cpp API/chat UI to other devices on
your network (e.g. `http://192.168.1.x:8080/`) and restarts the router for you,
no manual editing needed. A **Require an API key** toggle (on by default) blocks
LAN access until you set or generate a key; leaving it unchecked exposes the
router unauthenticated. See [SECURITY.md](SECURITY.md).

## How it works

LlamaForge contains **no llama.cpp source code**. The backend
(`backend/server.py`, pure Python stdlib) proxies llama.cpp's own router API, edits
`models.ini`, and shells out to `git` / `cmake` / `nvidia-smi` / `winget`. The knob
list is parsed live from `llama-server --help`, so it stays correct across llama.cpp
versions automatically. HuggingFace downloads are streamed by the backend, so they
work even when llama.cpp is built without SSL.

When models are registered, LlamaForge reads each GGUF's trained context length
straight from its header and writes sensible `ctx-size` defaults into `models.ini`
(a **150k** global baseline; **100k** for models that can't reach it, capped at the
model's own trained length so nothing is over-extended). Per-model settings you set
by hand always win.

## Credits & license

LlamaForge is MIT-licensed ([LICENSE](LICENSE)). It builds and drives
**[llama.cpp](https://github.com/ggml-org/llama.cpp)** - MIT, (c) The ggml authors -
see [NOTICE](NOTICE) and [LICENSE.llama.cpp.txt](LICENSE.llama.cpp.txt).
The hard part is theirs; please star and support the upstream project.
