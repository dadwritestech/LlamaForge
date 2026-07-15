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
  Setup (brew; exact install hints on Linux), with CI running the full test
  suite on windows / ubuntu / macos runners.
- **vLLM backend (WSL2)** — a second inference engine alongside llama.cpp for
  safetensors / AWQ / GPTQ / FP8 / NVFP4 models, sharing the same model list,
  Discover tab, and stats. Windows/WSL2-only for now (hidden on Linux/macOS).
- **Discover platform tags** — every result shows which OSes its backend runs
  on, plus GATED and INSTALLED badges.
- **Agent-friendly API** — OpenAI-compatible endpoint plus load/unload so agents
  can swap models on demand.
- **Quality-of-life pass** — quick-load from the row + a sequential load queue,
  named knob **presets** (apply to any model), side-by-side **model compare**,
  a **GGUF metadata card** (arch/params/quant/ctx/rope), **inline load-failure
  diagnosis** with a suggested fix, copy-paste **client config** (curl / OpenAI /
  JSON), a keyboard map, **download pause/resume**, **auto-load a model on
  launch**, and an optional system-tray icon.

## Next (in progress)

- **Named launch profiles** — save a *model + engine + settings* combo and launch
  it in one click. Knob **presets** shipped as the first step; the next is
  [binding a preset as a model's default](https://github.com/dadwritestech/LlamaForge/issues/2)
  so a dialed-in model just loads and goes.
- **ik-llama support** — needs the process-managed backend path that vLLM proved
  out. ik_llama.cpp is on an older base without llama.cpp's router
  (`--models-preset`, `/models` load/unload), so it's driven as a managed process
  rather than a binary swap. Its flag set still parses from `--help`, so per-model
  tuning comes along.

## Planned

- **Smarter VRAM-fit ratings** — the current rating is a naive size-vs-VRAM
  heuristic that mislabels **MoE** and CPU/expert-offload setups. Make it
  offload-aware and add community-sourced fit reports.
  ([#4](https://github.com/dadwritestech/LlamaForge/issues/4))
- **Auto-wire MTP draft models** — `mtp-*` sidecars are recognized today but not
  attached; wire them into speculative-decoding flags automatically, the way
  mmproj already is. ([#3](https://github.com/dadwritestech/LlamaForge/issues/3))
- **Native (non-WSL) vLLM on Linux** — vLLM currently rides WSL2 on Windows only.

## Under consideration

- More engines as demand shows (TabbyAPI/ExLlama, etc.).
- Import/export of profiles and per-model settings.

---

Not affiliated with ggml-org. All inference is done by the underlying engines
([llama.cpp](https://github.com/ggml-org/llama.cpp) and others) — LlamaForge just
drives them.
