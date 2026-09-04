---
title: FAQ
section: faq
order: 1
---

# FAQ

## How is LlamaForge different from LM Studio, Ollama, or Jan?

Those tools are zero-config, double-click installers with no compile step. LlamaForge is not: it assumes you're comfortable running a setup script once and building llama.cpp for your machine — both guided from the dashboard. In exchange, it gives you direct, per-model control over the real llama.cpp server rather than an abstraction over it. If you want something running with no compile step, [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com), or [Jan](https://jan.ai) will get you there faster.

## What platforms does it run on?

Windows with an NVIDIA GPU is the primary target (CPU-only also works). Linux (NVIDIA/CPU) and macOS (Apple Silicon, Metal) are supported as an early preview — same dashboard, `bootstrap.sh` instead of `bootstrap.ps1`. See [Installation](install.md) for the per-OS commands.

## What GPU do I need?

An NVIDIA GPU for CUDA acceleration, or Apple Silicon for Metal on macOS. CPU-only builds are supported everywhere too. Everything else needed to build (Git, CMake, Ninja, a C++ compiler, CUDA) is detected and can be installed from the **Setup** tab where a package manager allows it.

## Does LlamaForge come with any models?

No. LlamaForge contains no llama.cpp source code and bundles no models. The backend (`backend/server.py`, pure Python stdlib) proxies llama.cpp's own router API and shells out to `git`/`cmake`/`nvidia-smi` and your platform's package manager. Models are found two ways: the **Setup** tab scans your drives (or `$HOME` and mounts) for GGUFs you already have, and the **Discover** tab searches huggingface.co for GGUF (llama.cpp) or safetensors (vLLM) models to download, rated **FITS / TIGHT / CPU OFFLOAD** against your VRAM before you pull one down.

## What is the second "engine," vLLM, and why does it need WSL2?

LlamaForge is a llama.cpp control panel first, but it can also drive [vLLM](https://github.com/vllm-project/vllm) as a second backend for full-precision/safetensors models (FP16, BF16, AWQ, GPTQ, FP8, NVFP4). On Windows, vLLM runs inside WSL2 with GPU passthrough — installed from the **Setup** tab (uv plus a standalone Python into `~/.llamaforge/vllm-venv`, no `sudo`), with the dashboard bridging WSL's localhost port back to Windows. It runs one model at a time and has no hot reload, so saving knobs on a loaded model restarts it (startup can take 1–5 minutes). On Linux and macOS, vLLM is a Windows/WSL2-only feature — its tab and Discover's safetensors mode are hidden automatically, and llama.cpp (CUDA/CPU on Linux, Metal on Apple Silicon) is the engine there. If you only ever want llama.cpp, nothing about vLLM is installed unless you ask.

## Is my API key or model traffic sent anywhere?

The dashboard and its management API always bind to `127.0.0.1`. The separate
llama.cpp router is local by default; the Setup tab can make that router LAN
accessible at `0.0.0.0`, but a usable API key is backend-enforced for every new
LAN configuration and LlamaForge-owned start. There is no unauthenticated-LAN
toggle. See [SECURITY.md](https://github.com/dadwritestech/LlamaForge/blob/master/SECURITY.md) in the repository.

## Where do I go if something breaks?

See [Troubleshooting](troubleshooting.md) for the real failure messages LlamaForge's load-failure diagnosis recognizes, plus common install/build issues.
