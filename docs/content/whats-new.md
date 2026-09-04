---
title: What's New
section: whats-new
order: 1
---

# What's New

This page summarizes the most recent additions to LlamaForge. Each entry links to the full reference for that capability. For the longer-term direction, see the project's `ROADMAP.md`.

## Fail-closed network and explicit credentials

Network Access now distinguishes the loopback-only dashboard from the router it
controls. LlamaForge accepts only local (`127.0.0.1`) or LAN (`0.0.0.0`) router
scope, requires a usable key for every newly configured LAN router, and refuses
LlamaForge-owned starts/restarts when that policy is unsafe. Existing manual or
legacy settings are shown for repair rather than silently rewritten. Routine
dashboard state is redacted; Client Config, Agent Config, and generated keys are
deliberate no-store actions. This is tested product behavior in an early preview,
not a formal security standard or audit certification.

See [Security](https://github.com/dadwritestech/LlamaForge/blob/master/SECURITY.md),
[Setup](setup.md), [HTTP API](api.md), and [Connect an Agent](agents.md).

## ik_llama as a second llama-family engine

The Build tab now builds and drives **ik_llama.cpp** alongside stock llama.cpp, switched with one control (`POST /api/engine/switch`). The switch is gated on a capability probe: a binary whose `llama-server` lacks router mode (`--models-preset`) is refused with an explanation rather than taking the router down, and ik_llama keeps its own `models.ini` registry (a `-ikllama` sibling of the main one). Per-model tuning comes along automatically because the knob schema is parsed from whichever binary is active.

See [Build & Update](build.md) and [models.ini Format](models-ini.md).

## Bind a preset as a model's default

Named presets can now be **bound** to a model, not just applied once. Binding writes the preset's knobs into the model's section and — the point of it — editing a bound preset re-syncs every model using it. A ◉/○ dot on each preset chip toggles the binding; the bound chip is highlighted. Hand edits still win, and unbinding leaves the knobs in place.

See [Models & Tuning](models.md).

## Auto-wired MTP draft models

A scan now attaches an `mtp-*` speculative draft sidecar to its parent model the way `mmproj` already is, filling `spec-draft-model`. It enables `spec-type=draft-mtp` only when the sidecar actually declares NextN layers — the signal llama.cpp itself gates on — so a model that can't use MTP is never broken. The wiring is additive: it never overwrites a `spec-type` you set by hand (e.g. an ngram mode).

See [models.ini Format](models-ini.md).

## Offload-aware VRAM-fit rating

The **FITS / TIGHT / CPU OFFLOAD** badge in Discover now defers to the same physics estimate as the Will-it-run panel instead of a raw size-vs-VRAM guess. A large MoE that runs fast with its experts on CPU reads as **TIGHT** rather than being mislabeled **CPU OFFLOAD** just for being bigger than VRAM. The size heuristic remains the fallback when no prediction is available.

See [Discover](discover.md).

## A more forgiving first run

Several fresh-install rough edges are gone. A missing `config.json` is seeded from `config.example.json` instead of crashing; `models.ini` is created if absent; a relative `models_ini` resolves against the repo root so the router never comes up with zero models; a busy router port is reported by name instead of failing silently; a freshly installed tool is detected without a restart; and a build whose `llama-server` succeeded but whose UI-asset step failed reports **"built, with warnings"** instead of BUILD FAILED.

See [First Run](first-run.md), [Setup](setup.md), and [Troubleshooting](troubleshooting.md).

## Lite and Advanced modes, with a guided first run

A first-run wizard now walks a new install through engine detection, hardware review, model selection, and a recommended tune — then applies it and loads the model. The dashboard runs in one of two modes: **Lite** presents a reduced, task-oriented set of controls; **Advanced** exposes every server flag. A hardware **auto-tune** proposes per-model settings (GPU-layer offload, KV-cache type, context ceiling, and intent presets for balanced / speed / context / coding) sized to the detected VRAM.

See [First Run](first-run.md) and [Models & Tuning](models.md).

## Anthropic-compatible endpoint

The panel exposes an Anthropic-compatible `POST /v1/messages` endpoint that translates to the local OpenAI-style router, with full streaming and tool-use support. Combined with the existing OpenAI-compatible surface, LlamaForge can serve clients written for either API against your local models.

See [HTTP API](api.md).

## One-click agent setup

A **Connect an Agent** panel generates — and optionally writes — the configuration for **Claude Code**, **Codex**, and **pi.dev**, pointing each at your local endpoint. Claude Code is scoped to `127.0.0.1`; generated files are backed up before any change is written.

See [Connect an Agent](agents.md).

## Context Wiki

A working directory of Markdown context documents, composed into named **profiles** and selected **per model**. An active profile is delivered two ways: injected into requests through the Anthropic shim and the OpenAI proxy, or exported into an agent's native context file (`CLAUDE.md` / `AGENTS.md`) inside a managed marker region. Because the injected prefix is stable, the router's prompt-cache reuses it across requests.

See [Context Wiki](context-wiki.md).

## Light and dark themes, plus a colorblind-safe mode

The dashboard now offers a **Light** theme alongside the original dark one, and an independent **Colorblind-safe** mode that applies a universal Okabe–Ito status palette and adds non-color cues (glyphs and labels) so status never depends on hue alone. The two axes are orthogonal — all four combinations are valid — and each choice persists per device.

See [Theming & Accessibility](theming.md).

## In-app documentation

The documentation you are reading is available inside the dashboard under the **Help** view and is also published as a static site. Both surfaces render from a single Markdown source, so they never drift.

## A collapsible sidebar layout

Navigation moved from a top tab bar to a left **sidebar** that collapses between a compact icon rail and a labeled, expanded state. Settings are pinned at the bottom; the layout adapts to narrow windows with an overlay drawer. All existing navigation, theming, and keyboard shortcuts are unchanged.

See [Keyboard Shortcuts](keymap.md).

## Refine benchmark in the Models panel

A **Refine** button now sits inline in each model's editor (beside the Presets bar). Pick an intent (balanced / speed / context / coding), click **Run**, and it auto-generates knob recommendations, benchmarks candidates with real completion requests (~200 tokens each), and applies the fastest config. A results table shows tok/s per candidate and which was chosen. The same autotune engine used in the first-run wizard is now available anytime from the main Models tab.

See [Models & Tuning](models.md).

## VRAM "Will-it-run" panel

Before downloading, the **Discover** tab now shows a **Will-it-run** panel that predicts whether a model quant will fit your GPU and at what approximate speed. It factors in MoE active-vs-total parameters, your GPU's memory bandwidth (with manual overrides in Setup), and the quant's size — then rates it as **FITS**, **TIGHT**, or **CPU OFFLOAD** with an estimated tok/s. The same estimate appears as a badge when you expand a model in Discover.

See [Discover](discover.md).
