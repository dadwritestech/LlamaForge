# Roadmap

LlamaForge is an **early preview**. This is a direction, not a set of promises —
priorities shift with feedback, and there are no dates. If something here matters
to you, open an issue or 👍 an existing one; that's the strongest signal for what
gets built next.

## Now (shipped)

- **llama.cpp control panel** — per-model tuning of every `llama-server` flag
  (~220, parsed live from `--help`); saving hot-reloads the model, no restart.
- **VRAM-fit model discovery** — search HuggingFace GGUFs, each quant rated
  **FITS / TIGHT / CPU OFFLOAD** against your real VRAM before you download.
- **Guided build & update** — current commit vs upstream, rebuild with CMake
  flags auto-detected for your CPU/GPU.
- **Automatic `ctx-size` defaults** — read each GGUF's trained context length and
  write sane per-model context sizes.
- **Setup** — detect/install prereqs (winget/choco), scan drives for GGUFs, and
  prune registry entries whose files were deleted.
- **Usage stats** — per-model tokens, runs, average tok/s, daily activity —
  and optional **LAN sharing** with an API-key toggle.
- **Linux & macOS (early preview)** — `bootstrap.sh` / `run.sh` / `stop.sh`,
  portable process control and drive scanning, Metal build flags and
  unified-memory VRAM-fit ratings on Apple Silicon, package-manager-aware
  Setup (brew; exact install hints on Linux). The vLLM backend remains
  Windows/WSL2-only for now.
- **Discover platform tags** — every result shows which OSes its backend runs
  on, plus GATED and INSTALLED badges.
- **Agent-friendly API** — OpenAI-compatible endpoint plus load/unload so agents
  can swap models on demand.

## Next (in progress)

- **vLLM backend (WSL2)** — a second inference engine alongside llama.cpp.
  Design spec landed; implementation underway on `feature/vllm-backend`.
- **Backend abstraction** — the piece that makes multi-engine real: LlamaForge
  manages an engine's **process lifecycle directly** instead of delegating to
  llama.cpp's `--models-preset` router. vLLM and ik-llama both need this, because
  neither ships that router. Building it once unlocks both.

## Planned

- **ik-llama support** — rides the backend abstraction above. ik_llama.cpp is on
  an older base without llama.cpp's router (`--models-preset`, `/models`
  load/unload), so it needs process-managed integration rather than a binary
  swap. Its flag set still parses from `--help`, so per-model tuning comes along.
- **Named launch profiles** — save a *model + engine + settings* combo and launch
  it in one click.
- **Linux/macOS hardening** — the port shipped as an early preview; next is
  CI coverage on ubuntu/macos runners and native (non-WSL) vLLM on Linux.

## Under consideration

- More engines as demand shows (TabbyAPI/ExLlama, etc.).
- Import/export of profiles and per-model settings.

---

Not affiliated with ggml-org. All inference is done by the underlying engines
([llama.cpp](https://github.com/ggml-org/llama.cpp) and others) — LlamaForge just
drives them.
