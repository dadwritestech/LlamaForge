---
title: Installation
section: getting-started
order: 2
---

# Installation

LlamaForge ships as a source checkout plus a set of launcher scripts — there is no installer package. A `bootstrap` script gets a fresh machine to the point where the dashboard can take over; `run` starts the dashboard and router on subsequent launches; `stop` shuts everything down.

## Prerequisites

The bootstrap script checks and, with your consent, installs:

- **Python 3.10+** — required to run the backend (pure standard library; nothing to `pip install`).
- **Git** — required to fetch and update the llama.cpp source.

Additional build prerequisites are detected by `backend/prereqs.py` and surfaced on the dashboard's **Setup** tab (not installed by bootstrap itself):

- **Git**, **CMake**, **Ninja**, **Python** — checked via `git --version`, `cmake --version`, `ninja --version`, `python --version`.
- A **C++ compiler** — MSVC on Windows (located via `vswhere.exe`, or a `cl.exe` scan under `Program Files`), `clang++`/`g++` on Linux/macOS.
- **CUDA** — optional, needed only for NVIDIA GPU builds; detected via `CUDA_PATH` or `nvcc`. Not applicable on macOS (Metal is used instead).

On Windows and macOS, the Setup tab can install missing tools for you (winget/choco on Windows, Homebrew on macOS) after you confirm. On Linux, LlamaForge never runs `sudo`: it shows you the exact `apt`/`dnf`/`pacman` command to run yourself.

## Windows

```powershell
git clone https://github.com/dadwritestech/LlamaForge
cd LlamaForge
powershell -ExecutionPolicy Bypass -File bootstrap.ps1
```

`bootstrap.ps1` checks for Python and Git (offering to install Python 3.12 via winget if missing), then asks whether to use an existing llama.cpp checkout (Enter keeps the bundled `<LlamaForge>/llama.cpp` default). It derives `build_dir`, detects an existing `llama-server` in common build layouts, writes `config.json`, offers to clone llama.cpp when the selected checkout is absent, writes a starter `models.ini`, then launches `run.ps1`. Set `LLAMAFORGE_LLAMA_SRC` before running the script to supply the checkout non-interactively.

For daily use after the first run, double-click **`LlamaForge.vbs`** — it starts the router and dashboard hidden and opens your browser. To autostart it, put a shortcut to `LlamaForge.vbs` in your Startup folder (`Win+R` -> `shell:startup`).

To shut everything down — dashboard, router, and every model instance the router spawned:

```powershell
powershell -ExecutionPolicy Bypass -File stop.ps1
```

## Linux / macOS

```bash
git clone https://github.com/dadwritestech/LlamaForge
cd LlamaForge
./bootstrap.sh
```

`bootstrap.sh` checks for `python3` and `git` (printing the platform-appropriate install command if either is missing — `brew install python@3.12` on macOS, `sudo apt-get install -y python3` as a Linux example), asks whether to use an existing llama.cpp checkout, derives its build and server paths, writes `config.json`, offers to clone llama.cpp when needed, writes a starter `models.ini`, then execs `run.sh`. `LLAMAFORGE_LLAMA_SRC=/path/to/llama.cpp ./bootstrap.sh` supplies the checkout non-interactively.

For daily use after the first run:

```bash
./run.sh
```

To shut everything down:

```bash
./stop.sh
```

## What the scripts do

`run.ps1` / `run.sh` read `config.json`, start the llama.cpp router (`llama-server --models-preset <models.ini> --models-max 1 --offline --host <router_host> --port <router_port> --metrics`, plus `--api-key` if one is set) if it is not already listening, start the dashboard backend (`backend/server.py`) if it is not already listening, then open `http://127.0.0.1:<panel_port>/` in your browser. Both scripts are safe to run repeatedly — each checks whether its port is already in use before starting anything.

`stop.ps1` / `stop.sh` kill the process listening on `panel_port`, the process listening on `router_port`, and sweep any remaining `llama-server` processes the router spawned to serve individual models. On Windows, `stop.ps1` additionally kills any `vllm serve` process running inside WSL.

> [!NOTE]
> Building llama.cpp itself, installing a C++ compiler/CUDA, and installing the vLLM backend are deliberately left out of bootstrap — they are heavier, consent-gated steps performed from the dashboard's **Setup** and **Build** tabs after first launch.

Once the dashboard is open, continue to [First Run](first-run.md) for the onboarding wizard.
