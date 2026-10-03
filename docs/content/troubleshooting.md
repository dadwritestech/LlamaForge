---
title: Troubleshooting
section: troubleshooting
order: 1
---

# Troubleshooting

## Inline load-failure diagnosis

When a model fails to load, the Models tab shows the reason inline instead of making you scroll the Router Log. It calls `GET /api/model/diag?model=<id>`, which hands the router log to `diagnose()` in `backend/diag.py`.

`diagnose()` reads only **this model's last load attempt**. The router logs `spawning server instance with name=<id> on port <P>` for each load; the child's own output comes back prefixed `[<P>]`, and the router ends with `instance name=<id> exited with status N`. Lines from earlier loads and from other models are ignored, so an old out-of-memory error is never blamed for today's failure.

It then matches llama.cpp's actual error strings, most specific first. Healthy startup lines such as `ggml_cuda_init: found 2 CUDA devices` or `n_ctx = 65536` are not errors and never match. The error shown is llama.cpp's own line; the fix is LlamaForge's:

| llama.cpp says | Suggested fix |
|---|---|
| `unknown model architecture` / `unknown pre-tokenizer type` | Update llama.cpp (Build / Update tab). The model is newer than your build. |
| `error while handling argument "--x"` / `invalid argument: --x` | Clear that setting (the flag is named). If it's a newer flag, update llama.cpp. |
| `... requires flash_attn to be enabled` | Set flash-attn on (or auto), or set cache-type-v back to f16. |
| `cudaMalloc failed`, `out of memory`, `failed to allocate ... buffer` | If the model pins `n-gpu-layers`, `ctx-size` or `tensor-split`, those switch off llama.cpp's automatic fit: clear them and load again. Otherwise pick a smaller quant or a smaller ctx-size. |
| `context type MTP requested ...` / `failed to create MTP context` | Clear spec-type and spec-draft-model. |
| `failed to load multimodal model` | The mmproj doesn't match: clear it, or use the one from the model's own repo. |
| `failed to open GGUF file`, `failed to read magic`, `No such file` | Missing, moved or half-downloaded file: re-scan from Setup or fix the path. |
| `error loading model` / `failed to load model` | llama.cpp's own line is the reason; the full log is at the bottom of Models. |

If nothing matches but the instance exited with a non-zero status, that status is shown. If the log has no load attempt for this model at all, no diagnosis is shown rather than a guess. A `common_fit_params: failed to fit params` warning on its own is not a failure: llama.cpp prints it and then loads anyway.

> [!TIP]
> The Router Log panel at the bottom of the Models tab has the full output. `diagnose()` reads the last 800 lines of each router log file.

## Setup tab: missing prerequisites

The **Setup** tab (`backend/prereqs.py`) checks for `git`, `cmake`, `ninja`, and `python` (via `<tool> --version`), a C++ compiler, and CUDA, and reports whether each can be auto-installed on your platform.

| Issue | Cause | Fix |
|---|---|---|
| A tool shows as missing (git/cmake/ninja/python) | Not found on `PATH` (`shutil.which`) | On Windows/macOS, click install from the Setup tab (winget/choco, or Homebrew). On Linux, the dashboard never runs `sudo` — it shows the exact `apt`/`dnf`/`pacman` command to run yourself. |
| C++ compiler not found | Windows: no MSVC install with the "Desktop development with C++" workload found via `vswhere.exe` (or a `cl.exe` scan under `Program Files`). Linux/macOS: neither `clang++` nor `g++` on `PATH`. | Windows: install Build Tools for Visual Studio with the C++ workload. macOS: `xcode-select --install`. Linux: install `g++` or `clang++` via your package manager. |
| CUDA not found | No `CUDA_PATH` set and `nvcc` not on `PATH`. Not applicable on macOS (Metal is used instead — the row is hidden). | Only needed for NVIDIA GPU builds; install from https://developer.nvidia.com/cuda-downloads, or build CPU-only. |
| Install button does nothing / fails on Windows | Neither `winget` nor `choco` is available, or the install command exited non-zero. | Install the tool manually from the URL shown for that tool in the Setup tab. |
| Install fails on macOS | Homebrew isn't installed. | Install Homebrew (https://brew.sh) first, or install the tool manually. |

## Bootstrap script issues

`bootstrap.ps1` (Windows) and `bootstrap.sh` (Linux/macOS) get a fresh checkout to the point where the dashboard can take over.

- **"Python not found" (Windows) / "python3 not found" (Linux/macOS)** — Python 3.10+ is required to run the backend. `bootstrap.ps1` offers to install Python 3.12 via winget; without winget it points you to https://www.python.org/downloads/windows/. `bootstrap.sh` prints `brew install python@3.12` on macOS or a `sudo apt-get install -y python3`-style hint on Linux, then exits — it does not install Python for you.
- **"Git not found"** — Git is recommended for fetching/updating llama.cpp. `bootstrap.ps1` offers to install it via winget; `bootstrap.sh` prints the platform-appropriate install command. Bootstrap continues either way; you can clone llama.cpp manually later.
- **"llama.cpp source not found at `<path>`"** — `config.json`'s `llama_src` doesn't point at a git checkout. Answer `y` to clone it (needs Git), or point `llama_src` in `config.json` at an existing checkout and re-run.
- **Re-run bootstrap after installing Python** — `bootstrap.ps1` exits after installing Python and asks you to open a new terminal so `PATH` picks it up, then run bootstrap again.

## First run

- **`config.json` / `models.ini` missing** — no longer fatal. The launchers copy `config.example.json` to `config.json` on a fresh checkout, and a `[*]`-only `models.ini` is created if absent (the router won't start without it). Set your real paths in the Setup tab afterward.
- **Router port already in use** — port `8080` collides with XAMPP/Apache and other dev servers. `run.ps1`/`run.sh` now name the process holding `router_port` and skip starting the router instead of leaving every model showing "offline" for no visible reason; free the port or change `router_port` in Setup.
- **All models "offline" / zero models after moving the folder** — a relative `models_ini` (the shipped default `./models.ini`) is now anchored to the repo root, so the router no longer reads an empty registry when launched from another directory. If you still see this, check `models_ini` in `config.json` points at the right file.
- **Build shows "built, with warnings"** — `llama-server` built fine but a non-essential step (usually the npm/`sharp` UI assets on Windows) failed. The binary is usable; the failing step is in the Build Log. This is distinct from a red **BUILD FAILED**, which means no fresh `llama-server` was produced.

## Everything else

- **Dashboard won't open** — `run.ps1`/`run.sh` skip starting the router or dashboard if something is already listening on `router_port` (default `8080`) or `panel_port` (default `8090`); check nothing else on your machine is bound to those ports.
- **Shutting everything down cleanly** — use `stop.ps1` (Windows) or `stop.sh` (Linux/macOS) rather than killing windows manually; they stop the dashboard, the router, and every `llama-server` process the router spawned (and, on Windows, any `vllm serve` process running inside WSL).
- **vLLM startup looks stuck** — vLLM has no hot reload, so saving knobs on a loaded model restarts it; startup can take 1–5 minutes. Watch the vLLM Log panel rather than assuming it hung.
