---
title: Installation
section: getting-started
order: 2
---

# Installation

One line, no git, no admin, no compiler. Re-run it any time to update.

**Windows** (PowerShell):

```powershell
irm https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/install.ps1 | iex
```

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/install.sh | sh
```

The installer:

1. finds Python 3.10+ (the backend is pure standard library, nothing to `pip install`). On Windows, if you have none, it drops a private, SHA-256-pinned copy of python.org's embeddable Python;
2. downloads the latest LlamaForge release into `%LOCALAPPDATA%\LlamaForge` (Windows) or `~/.local/share/llamaforge` (Linux/macOS). Updating keeps your config, models and engines;
3. adds a Start menu entry and an Apps & Features uninstaller (Windows), a `llamaforge` command plus an app-menu entry (Linux), or `~/Applications/LlamaForge.app` (macOS);
4. starts LlamaForge and opens the dashboard.

In the dashboard, click **Install llama.cpp**. It fetches the official llama.cpp release for your GPU (CUDA, Vulkan, Metal or CPU), verifies it and starts the router. Then continue to [First Run](first-run.md).

### Installer options

Set these environment variables before running the one-liner:

| Variable | Effect |
|---|---|
| `LLAMAFORGE_HOME` | install directory |
| `LLAMAFORGE_REF` | a release tag (e.g. `v0.14.0`) to install or roll back to; default is the latest release |
| `LLAMAFORGE_ARCHIVE` | install from a local `.zip` / `.tar.gz` instead of downloading |
| `LLAMAFORGE_NO_LAUNCH` | don't start LlamaForge at the end |
| `LLAMAFORGE_NO_SHORTCUTS` | skip the command, menu entry and app bundle |
| `LLAMAFORGE_NO_STOP` | don't stop a running copy before updating it |

## Daily use

Open **LlamaForge** from the Start menu or app menu, or run `llamaforge`. It starts the llama.cpp router and the dashboard if they aren't already running, then opens your browser at `http://127.0.0.1:8090`.

To shut down the dashboard, the router and the models it spawned, run `llamaforge stop` (Linux/macOS) or `stop.ps1` in the install directory (Windows). Only processes LlamaForge started are stopped; any other `llama-server` you run is left alone. On Windows it also stops a `vllm serve` it started inside WSL.

## Uninstall

Windows: **Settings → Apps → Installed apps → LlamaForge → Uninstall**. Linux/macOS: `llamaforge uninstall`. Both ask before removing your settings or models, and only remove files the installer put there.

## From source (git clone)

Use this if you want to hack on LlamaForge or build llama.cpp from source.

```powershell
git clone https://github.com/dadwritestech/LlamaForge
cd LlamaForge
powershell -ExecutionPolicy Bypass -File bootstrap.ps1   # Windows
./bootstrap.sh                                           # Linux / macOS
```

The bootstrap script checks for Python and Git (asking before installing anything), asks whether to use an existing llama.cpp checkout, writes `config.json` and a starter `models.ini`, then launches the dashboard. Set `LLAMAFORGE_LLAMA_SRC` to supply the checkout non-interactively. Afterwards, start it with `LlamaForge.vbs` (Windows) or `./run.sh`, and stop it with `stop.ps1` / `./stop.sh`.

Building llama.cpp from source needs more tools, all detected on the **Setup** tab:

- **Git**, **CMake**, **Ninja**;
- a **C++ compiler**: MSVC on Windows, `clang++`/`g++` on Linux/macOS;
- **CUDA**, only for NVIDIA builds (macOS uses Metal).

On Windows and macOS the Setup tab can install missing tools after you confirm (winget/choco, Homebrew). On Linux, LlamaForge never runs `sudo`: it shows the exact `apt`/`dnf`/`pacman` command for you to run.

## What the launcher does

The launcher reads `config.json` and starts the llama.cpp router (`llama-server --models-preset <models.ini> --models-max 1 --offline --host <router_host> --port <router_port> --metrics --api-key <key>`) and the dashboard backend, each only if its port isn't already in use, then opens the dashboard. Running it twice is safe. Launch output goes to `logs/` in the install directory, so a router that fails to start leaves a log you can read.
