---
title: First Run
section: getting-started
order: 3
---

# First Run

The first time LlamaForge starts with a fresh `config.json` (no `ui_mode` key on disk), `config.py`'s `migrate()` classifies the install: if `server_bin` is already set (an existing llama.cpp build was found), the config is marked `ui_mode: "advanced"` and `onboarded: true`; otherwise it is marked `ui_mode: "lite"` and `onboarded: false`, which is what causes the onboarding wizard to appear. This check is idempotent — a config that already carries `ui_mode` is left untouched on every later startup.

## Fresh-install safety net

Several things that used to derail a first run are now handled automatically, in both `run.ps1` / `run.sh` and the backend:

- **No `config.json`** — the launchers copy `config.example.json` to `config.json` and tell you to set your paths in Setup, instead of crashing on a missing file.
- **No `models.ini`** — `config.ensure_models_ini()` creates a minimal one with a `[*]` section (the router refuses to start without it). A relative `models_ini` is anchored to the repo root so the detached router can't read an empty registry from the wrong directory.
- **Router port already in use** — if something else holds `router_port` (port `8080` collides with XAMPP/Apache and other dev servers), the launcher names the process holding it and skips starting the router, rather than leaving the dashboard showing every model "offline" with no explanation. The backend's `router_ctl.start()` refuses a bound port the same way.
- **A just-installed build tool still shows MISSING** — the Setup **Install** flow re-reads `PATH` from the registry (`osplat.refresh_path()`) so a freshly installed `ninja`/`cmake` is detected without restarting; only if that genuinely isn't enough does it ask for a restart.
- **`server_bin` after a build** — a finished build records where `llama-server` actually landed (`bin/Release` under MSVC, `bin` elsewhere), correcting the pre-build guess so you don't have to hand-edit `config.json`.

## Lite vs Advanced mode

LlamaForge has two UI densities, toggled at any time from the mode switch in the dashboard header (`applyMode()` in `web/js/ui.js` toggles a `mode-lite` class on `<body>` and persists the choice via `PUT /api/config` with `ui_mode`):

- **Lite** — a reduced set of controls, aimed at getting a model loaded quickly.
- **Advanced** — every flag your `llama-server --help` lists (200+ on current builds) and every tab exposed.

Finishing the wizard sets `ui_mode` to `"lite"`; skipping it sets `ui_mode` to `"advanced"`. You can switch between them afterward at any time.

## The onboarding wizard

`wizMaybeStart()` shows the wizard automatically whenever the server reports `onboarding.onboarded` as false. It walks five steps, defined in `WIZ.steps` in `web/js/wizard.js`:

1. **Engine** — "Do you already have a llama.cpp build?" Choose *Yes, I have one built* or *No — clone & build it for me*, and a build flavor (official llama.cpp; a mainline fork; ik_llama is listed but disabled, marked "coming soon"). The clone path hands off to the same flow as the Build tab.
2. **Hardware** — a read-only summary of detected GPUs and their VRAM (or "No GPU detected — CPU mode" if none).
3. **Model** — pick an already-registered model from a dropdown. If none are registered, the step instead links to the Discover tab and closes the wizard so you can download one.
4. **Tune** — choose a goal (Balanced, Max speed, Max context, or Coding) and click **Auto-tune** to call `/api/autotune/recommend` for that model and intent; the resulting knobs and their rationale are shown in a table. An optional **Refine with a quick test (~1 min)** button calls `/api/autotune/refine` to try a few high-impact variants (e.g. alternate `ubatch-size`/`batch-size`) with one short generation each and keep the fastest. It is a quick check, not llama-bench: results move with whatever else the machine is doing.
5. **Ready** — confirms the chosen settings will be applied to the selected model and it will be loaded.

Finishing the wizard saves the recommended knobs with `/api/save`, loads the model with `/api/load`, and marks the config `onboarded: true, ui_mode: "lite"` regardless of whether the load itself succeeded (a failed load surfaces a toast telling you to load it manually from the Models tab). **Skip** instead marks the config `onboarded: true, ui_mode: "advanced"` and closes the wizard without touching any model.

## What auto-tune decides

`backend/autotune.py`'s `recommend(meta, hw, intent, size_bytes)` is a pure function that turns a GGUF's header metadata and the detected hardware into a small set of knobs — everything else is left at llama.cpp's own defaults. Four intents are supported: `balanced`, `speed`, `context`, `coding`.

- **Memory sizing is left to llama.cpp.** `fit` is set to `on`, and `ctx-size`, `n-gpu-layers` and `tensor-split` are left blank. llama.cpp's `--fit` then picks the context, the GPU layers and the multi-GPU split from your real free VRAM at load, and moves MoE experts to CPU when it must. Pinning any of those three turns fit off, which is why auto-tune clears them (this also undoes a pin written by an older LlamaForge).
- **Flash attention** is set to `auto`: llama.cpp enables it wherever the backend supports it.
- **Threads** are set to the CPU's hardware thread (or core) count, when known.
- **Intent-specific shaping:** `context` sets `fit-ctx` (a floor, not a pin: fit moves layers to CPU before it goes below the model's trained length, capped at 150000) and an 8-bit KV cache (`q8_0`, roughly half the memory per token). `speed` uses an `f16` KV cache and raises `batch-size`/`ubatch-size` to `2048`/`512`. `coding` lowers `temp` to `0.2` and `top-p` to `0.9`.

Every knob `recommend()` sets comes with a plain-language reason, which is what populates the rationale column in the wizard's Tune step.

> [!TIP]
> Auto-tune only writes the handful of knobs above. Anything else you set by hand on the Models tab is kept. If you pin `ctx-size`, `n-gpu-layers` or `tensor-split` yourself, llama.cpp's fit switches off for that model, and a load that runs out of memory will tell you so.
