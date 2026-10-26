---
title: What's New
section: whats-new
order: 1
---

# What's New

This page summarizes the most recent additions to LlamaForge. Each entry links to the full reference for that capability. For the longer-term direction, see the project's `ROADMAP.md`.

## v0.17.0: Auto-Forge, Claude Code on a folder

- **Auto-Forge.** Finds the fastest way to run a model on your GPUs by measuring it, not guessing. In a private sandbox it tries GPU placements, KV cache types and speculative drafters (DFlash, MTP) at the context you ask for, benchmarks code, prose and prompt speed, and checks token for token that a drafter gives the same answers as the model without it. Pick what you use the model for (coding, chat, long documents, agents), quick or thorough, and optionally allow lossy KV or try without the vision projector. **Apply** writes only what it changed into `models.ini`; **Undo** puts your values back. On a Qwen3.8-27B with an RTX 5080 and an RTX 5060 Ti it went from 26.8 to 126.0 tokens/s on code and from 26.5 to 71.0 on prose (one machine, one model). NVIDIA only, and new. See [Auto-Forge](autoforge.md). API: `/api/forge/status`, `/api/forge/start`, `/api/forge/cancel`, `/api/forge/apply`, `/api/forge/undo`.
- **Claude Code on a folder, one click.** A **Claude Code** button on a loaded model checks that it's ready (CLI found, Anthropic endpoint on, context size, jinja, a small model for background requests), lets you pick a folder, and opens a terminal there running `claude` against the local model. The endpoint and key are set on that one process, so your other Claude Code sessions and `~/.claude/settings.json` are untouched, and an inherited cloud API key never reaches the local endpoint. See [API](api.md). API: `/api/claude/check`, `/api/claude/pick-folder`, `/api/claude/launch`.
- **Fixes.** A metrics scrape on Stats and a slot read by Elastic VRAM never load a model any more. A router that can't start because its port is taken now says so, with a fix. Controls you could only click are keyboard buttons now, and inputs that only had a placeholder have labels. Python and JavaScript lint run in CI. The docs are also served as [llms.txt](https://dadwritestech.github.io/LlamaForge/llms.txt).

## v0.16.4: Elastic VRAM, live RAM

- **Elastic VRAM.** When another app needs your GPU, LlamaForge moves a loaded model's layers to the CPU instead of unloading it (MoE experts first), keeps the conversation through llama.cpp's slot save and restore (after one router restart; the router gets a save folder only while Elastic is on), and moves the layers back once the memory is free. A resized model is much slower, and the dashboard row says so; click its chip to grow it back. NVIDIA only, off by default, and new: not yet tested on many machines. See [Setup > Elastic VRAM](setup.md#elastic-vram). API: `/api/elastic`, `/api/elastic/grow`.
- **Live system RAM on Stats.** A System RAM meter sits next to the GPU memory in every look, also on machines without a GPU. "Used" leaves out cache the OS can hand back, like Task Manager's "In use". See [Stats](stats.md).
- **A warning before an unknown architecture fails.** If a model's architecture is newer than the llama.cpp checkout that would load it, the model editor says so (and to update the build) instead of the load failing with `unknown model architecture`. Embedding models such as EmbeddingGemma now have a section in [models.ini](models-ini.md#embedding-models).
- **Fixes.** `/v1/audio/speech` answers an unsupported `response_format` (say `mp3`) with a 400 instead of WAV under the wrong name. Ember quotes copied from PDFs match across line-break and soft hyphens. Reloading `models.ini` works on router builds that serve it on a different route (thanks @Vergillll), and `allowed values:` lists that wrap over several lines in `--help` are read whole (thanks @Vergillll).

## v0.16.3: Game mode

- **Game mode.** Start a game and LlamaForge unloads your models; quit it and they load again. Games are recognised by their Steam, Epic, GOG, Xbox, Riot, EA or Ubisoft folder, by names you add, or (Windows) by exclusive full screen. Pick models to keep loaded through it. Off by default. See [Setup](setup.md#game-mode).
- **Discover shows VRAM in use** next to the fit rating, so you can see what's already taken.

## v0.16.2: load progress, Thinking and Vision switches

- **Real load progress.** A loading model shows `loading N%` and a bar through its stages (weights, draft model, vision projector) instead of a stopwatch. The numbers are llama.cpp's own load callback, which the router streams on `GET /models/sse`; LlamaForge listens only while something is loading. A router build without that stream keeps the elapsed-seconds display. See [Models & Tuning](models.md).
- **Thinking: Auto / On / Off, plus a budget.** A one-click row in the model editor over llama.cpp's `--reasoning` and `--reasoning-budget`. The model's chat template decides the hint: a template with a thinking switch (`enable_thinking` and the like) really turns thinking off; a template that only shows `<think>` or `[THINK]` tags may keep thinking anyway. Models with neither don't get the row.
- **Vision on/off.** Turning a model's vision off removes its `mmproj` line and remembers the path, so the projector's VRAM is free and the file stays on disk. A rescan or re-download doesn't turn it back on; **On** restores the line.

## v0.16.1

- **Model Scout runs overnight again.** An ember with no model of its own now uses the auto-load model from Setup when nothing is loaded, instead of skipping its jobs. See [Embers](embers.md).

## v0.16: Embers, several models at once, an MCP server, Voice

- **Voice: text to speech on your GPU.** A new Voice tab speaks through llama.cpp's own `llama-tts` with Qwen3-TTS (10 languages) or the tiny English Pocket TTS (faster than real time, even on a CPU), optionally in a voice you record or upload (only clone voices you may use). The panel also serves an OpenAI-style `POST /v1/audio/speech`, so OpenAI clients get local speech. No VRAM is held between requests. See [Voice](voice.md).
- **Several models at once.** Multi-model mode places your main model on the fastest GPU that fits it and keeps worker models off that GPU. Fit is checked against footprints measured on this machine, and each verdict says whether it's measured, predicted or a rough estimate. Workers are evicted least-recently-used, and only after you agree to it. A model with no context size set is refused, with the reason. The model rows get **Load as worker**, **Make main** and **Unload X and load**, and Setup has a Multi-model card. API: `/api/slots`, `/api/slots/plan`, `/api/slots/main`, `/api/slots/apply`.
- **Per-model builds.** Pin a model to any installed llama.cpp build. An ik_llama build runs as its own `llama-server` on `127.0.0.1:8100` and up, with arguments translated between the builds. Crash exit codes are named instead of shown as raw numbers.
- **"Runs on" for GGUFs.** The file card says whether a GGUF runs on any build, mainline only or ik_llama only. The diagnosis explains `invalid ggml type N` instead of leaving you with the number.
- **Embers.** Small local agents that keep watch on a topic, keep a wiki and write you a brief on a schedule. A wiki item only counts if it quotes its source verbatim. Model Scout needs no setup, and **Forge** builds a new ember by interviewing you (tested on 8 models from 3B to 27B). See [Embers](embers.md).
- **MCP server.** Claude Code, Codex or any MCP client can check status, load and unload models, check fit, search and download from Hugging Face, and hand a task to a local model. See [MCP Server](mcp.md).
- **pi from Setup.** The pi coding agent card installs, updates and removes pi under `<root>/agents/pi` (with `--ignore-scripts`), and can install Node.js through winget, Chocolatey or Homebrew. See [Connect an Agent](agents.md).
- **Stowage, the new default look.** Each GPU on Models and Stats is drawn as a cargo hold ruled in 1 GiB cells, with every loaded model stowed in it at its measured size. Open a model and it shows where it would go: if it won't fit, the part that doesn't fit hangs past the end of the hold and anything that would be unloaded is marked. A ledger adds up used, booked and free to the tenth. Light, dark and colorblind-safe. **Hearth** and **Classic** are one click away. Motion is subtle, never runs on the views that refresh every few seconds, and is off when your system asks for reduced motion. See [Theming & Accessibility](theming.md).
- **No font CDN.** The panel's fonts ship with it, so it renders the same offline and no third party learns you opened it.
- **Stats count every loaded model**, including models running as their own process, with live tokens per second for each. See [Stats](stats.md).
- **Vision projectors pair again.** A scan only attached an `mmproj` file to models on a fixed list of architectures, so Gemma 4, Qwen 3.5 and other current vision models lost theirs ([Discussion #16](https://github.com/dadwritestech/LlamaForge/discussions/16)). A projector now pairs when its width matches the model's, the same check llama.cpp makes when it loads one. Several quants of one model can share a projector, and with several projectors F16 wins. Files named `<model>-mmproj-*.gguf` count as projectors, in a scan and in Discover.
- **Logs stay small.** With `verbose = true` a router log could grow to gigabytes, and the panel read all of it every time the log view refreshed, using tens of GB of RAM ([#26](https://github.com/dadwritestech/LlamaForge/issues/26)). The panel now reads only the end of a log, and once a router, vLLM or model log passes 50 MB it moves to `.1` (7 old copies kept), both when the process starts and every minute while it runs. On Windows, a router started by `run.ps1` is rotated from its next restart from the panel.
- **Docs search reads the whole page**, not only titles and headings, and says so when nothing matches.
- **Fixes.** An out-of-memory load on a model with `split-mode = tensor` now says that tensor split turns llama.cpp's fit off and suggests `split-mode = layer`. A `pi_run` that gave no answer and had every tool call fail is reported as a failure. The ik_llama update check reads ik_llama's own default branch (`main`). A busy router port points you at `config.json`.

## v0.15: llama.cpp sizes memory, honest diagnosis, safer by default

- **llama.cpp's `--fit` decides context, GPU layers and split.** LlamaForge no longer pins `ctx-size`, `n-gpu-layers` or `tensor-split`. Auto-tune clears them, and the old `[*] ctx-size = 150000` pin is removed once on upgrade. See [First Run](first-run.md) and [models.ini Format](models-ini.md).
- **Load failures quote the attempt that failed.** The diagnosis reads only the last load of that model from the router log and quotes llama.cpp's own error line. If a value you pinned turned fit off, the out-of-memory hint names it. See [Troubleshooting](troubleshooting.md).
- **Split GGUFs count all their shards** in sizes and fit estimates. Speed and fit badges are labelled as rough estimates.
- **A download ends in Load & Chat**, and a first run with no models suggests starters sized to your VRAM.
- **Security:** the router always runs with an API key, and CORS is localhost-only unless you opt into LAN. Recipes import only allowlisted tuning knobs. `stop` and uninstall touch only processes and files LlamaForge owns. Vulnerabilities can be reported privately (see `SECURITY.md`).
- The running version is shown next to the logo and requested in bug reports.

## Community recipe gallery

Click **browse recipes** on the Profiles strip to see tested setups people have shared: the model, the hardware it ran on, what the key knobs do, and an **on this machine** badge when you already have the file. **Import** runs the normal recipe import, so a missing model downloads from Hugging Face and the same knob filter applies. The list comes live from the repo's `recipes/` folder, so a merged recipe appears without a release; offline, LlamaForge shows the copy that shipped with your version. To add yours, export it with ↗ and open a pull request (see `recipes/README.md`).

See [Models & Tuning](models.md) and [HTTP API](api.md).

## Shareable recipes

Any llama.cpp profile can be shared as a **recipe**: click ↗ on its chip and copy readable JSON with the model file, the Hugging Face repo it came from, its knobs, and the llama.cpp build it was made on. Paste a recipe into **+ import recipe** to get the same preset and profile. If you don't have the model, **Download from <repo> & import** fetches it. Only tuning knobs are imported; paths, hosts, keys and logging flags are always dropped.

See [Models & Tuning](models.md) and [HTTP API](api.md).

## One-click launch profiles

Save a model, a preset, and (optionally) a specific installed llama.cpp build as a named **profile** from the model's editor, then launch the whole combination from a ▶ chip above the model list. If the pinned build isn't the active one, LlamaForge switches to it, waits for the router, applies the preset, and loads the model. That's useful when a new llama.cpp release regresses one model and you want to keep it on the build that worked. Builds a profile pins are kept out of the automatic pruning after an update.

See [Models & Tuning](models.md) and [HTTP API](api.md).

## Focused scanning and a lighter Models tab

Setup can now persist one or more model folders instead of walking every drive, while an explicit blank list still selects the platform defaults. A llama.cpp model can be **Unregistered** from `models.ini` without deleting its GGUF file. The Models view also avoids polling collapsed logs, shares GPU telemetry through a short cache instead of spawning `nvidia-smi` per browser poll, and keeps the CRT scanline look without the expensive full-viewport blend mode.

Fresh bootstrap runs now accept an existing llama.cpp checkout and derive its build/server paths, removing the hand-edit step reported by WSL2 users.

See [Setup](setup.md), [Models & Tuning](models.md), and [Install](install.md).

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
