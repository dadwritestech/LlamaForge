# Roadmap

LlamaForge is an **early preview**. This is a direction, not a set of promises —
priorities shift with feedback, and there are no dates. If something here matters
to you, open an issue or 👍 an existing one; that's the strongest signal for what
gets built next.

## Now (shipped)

- **llama.cpp control panel** — per-model tuning of every `llama-server` flag
  (~220, parsed live from `--help`); saving hot-reloads the model, no restart.
- **VRAM-fit model discovery** — search HuggingFace GGUFs, each quant rated
  **FITS / TIGHT / CPU OFFLOAD** against your real VRAM before you download. The
  rating is **offload-aware**: it defers to a physics estimate that accounts for
  MoE active-vs-total params, so a big MoE that runs fast with experts on CPU is
  no longer mislabeled CPU OFFLOAD.
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
- **ik_llama engine** — build and drive **ik_llama.cpp** as a second
  llama-family engine, switched from the Build tab. The switch is gated on a
  router-mode capability probe (a binary without `--models-preset` is refused
  rather than taking the router down); ik_llama keeps its own `models.ini`
  sibling registry, and per-model tuning rides its own parsed `--help` schema.
- **Bind a preset as a model's default** — a named preset can be pinned to a
  model so its knobs travel with it, and editing the preset re-syncs every bound
  model. ([#2](https://github.com/dadwritestech/LlamaForge/issues/2))
- **Auto-wired MTP draft models** — an `mtp-*` sidecar is attached as the
  speculative draft model on scan, enabling `spec-type=draft-mtp` only when the
  file declares NextN layers. ([#3](https://github.com/dadwritestech/LlamaForge/issues/3))
- **Robust first run** — `config.json`/`models.ini` auto-created, relative paths
  anchored, router port-conflict surfaced, freshly installed tools detected
  without a restart, and partial builds reported as "built, with warnings."
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
- **Lite & Advanced modes + guided first run** — a first-run wizard (engine →
  hardware → model → tune → load) and a hardware **auto-tune** that sizes
  GPU-layer offload, KV-cache type, context ceiling, and intent presets
  (balanced / speed / context / coding) to your VRAM. Lite hides the deep knobs;
  Advanced exposes every flag.
- **Anthropic-compatible endpoint** — a `POST /v1/messages` shim (SSE streaming +
  tool use) that translates to the local OpenAI-style router, so Anthropic
  clients (Claude Code and others) run against your local models.
- **One-click agent setup** — a *Connect an Agent* panel that generates and
  optionally writes config for **Claude Code**, **Codex**, and **pi.dev**
  (Claude Code scoped to `127.0.0.1`; existing files backed up before any change).
- **Context Wiki** — a directory of Markdown context docs composed into named
  **profiles**, selected per model, and delivered by proxy injection (Anthropic
  shim + OpenAI proxy) or exported into `CLAUDE.md` / `AGENTS.md` (marker region).
  The stable prefix rides the router's prompt cache.
- **Light/dark + colorblind-safe theming** — a Light theme adapting the terminal
  identity, plus an orthogonal colorblind-safe mode (universal Okabe–Ito status
  palette + non-color glyph/label cues). Layered persistence: localStorage >
  `config.json` > OS.
- **In-app documentation + published site** — a full docs corpus rendered from one
  Markdown source into an in-app **Help** view and a static **GitHub Pages** site.
- **Collapsible sidebar UI** — navigation moved from a top tab bar to a two-panel
  layout: a left sidebar that collapses between an icon rail and labeled state
  (settings pinned at the bottom), with a responsive overlay drawer on narrow
  windows.

## Next (in progress)

- **Named launch profiles** — save a *model + engine + settings* combo and launch
  it in one click. Knob **presets** and [binding one as a model's default](https://github.com/dadwritestech/LlamaForge/issues/2)
  shipped as the first steps; the remaining piece is a single named profile that
  also carries the engine and model choice.

## Planned

- **Local-only VRAM-fit predictions** — fit ratings remain entirely on-device.
  LlamaForge will not build hosted or crowdsourced performance reporting.
- **Native (non-WSL) vLLM on Linux** — vLLM currently rides WSL2 on Windows only.

## Under consideration

- More engines as demand shows (TabbyAPI/ExLlama, etc.).
- Import/export of profiles and per-model settings.

---

Not affiliated with ggml-org. All inference is done by the underlying engines
([llama.cpp](https://github.com/ggml-org/llama.cpp) and others) — LlamaForge just
drives them.
