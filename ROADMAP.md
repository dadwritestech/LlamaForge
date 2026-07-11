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
- **Usage stats** and optional **LAN sharing** with an API-key toggle.
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
- **Linux support** — the most-requested item. Today's Windows-specific bits
  (process control, drive scanning, winget/choco installs) need portable
  equivalents. It's a real port, not a flag flip, but it's genuinely wanted.
- **macOS support** — Metal builds and Apple-silicon VRAM-fit ratings.

## Under consideration

- More engines as demand shows (TabbyAPI/ExLlama, etc.).
- Import/export of profiles and per-model settings.

---

Not affiliated with ggml-org. All inference is done by the underlying engines
([llama.cpp](https://github.com/ggml-org/llama.cpp) and others) — LlamaForge just
drives them.
