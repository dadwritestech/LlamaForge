---
title: config.json Reference
section: reference
order: 1
---

# config.json Reference

`config.json` lives at the repository root and holds every machine-specific setting LlamaForge needs — nothing is hardcoded into the source. It is created by the bootstrap scripts on first run and updated by the dashboard (`POST /api/config`, `POST /api/network`) as you use it. `backend/config.py` defines the defaults; any key missing from the file on disk falls back to its default at load time.

## Keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `llama_src` | string | `""` | Path to a git checkout of `llama.cpp`. |
| `build_dir` | string | `""` | CMake build directory for `llama.cpp` (usually `<llama_src>/build`). |
| `server_bin` | string | `""` | Path to the built `llama-server` (or `llama-server.exe`) binary. |
| `models_ini` | string | `<repo root>/models.ini` | Path to the `models.ini` preset file passed to `llama-server --models-preset`. |
| `model_dirs` | list | `[]` | Directories the Discover/scan feature searches for GGUF files. |
| `router_port` | int | `8080` | Port `llama-server` (the router) listens on. |
| `panel_port` | int | `8090` | Port the LlamaForge dashboard (`backend/server.py`) listens on. |
| `chat_port` | int | `8091` | Port llama.cpp's own chat UI is proxied on, on its own origin (the Chat tab). |
| `router_host` | string | `"127.0.0.1"` | Router bind address. The Network Access UI supports only `127.0.0.1` (local) and `0.0.0.0` (LAN). |
| `router_api_key` | string | `""` | Plaintext API key required for LAN. It is not returned in ordinary dashboard state. |
| `router_local_key` | string | `""` | The key LlamaForge generates for itself when you set none, so the router always runs keyed. Minted at startup; never shown in the dashboard. |
| `wsl_distro` | string | `""` | WSL distro that runs vLLM. Empty string auto-picks the default distro. |
| `vllm_port` | int | `8081` | Port vLLM serves on inside WSL (localhost-forwarded to Windows). |
| `cmake_flags` | object | `{}` | Persisted CMake build flags, normally seeded from hardware detection. |
| `git_remote` | string | `"https://github.com/ggml-org/llama.cpp"` | Remote used to clone/update the `llama.cpp` source. |
| `active_engine` | string | `"llamacpp"` | Which llama-family binary the router uses: `"llamacpp"` or `"ikllama"`. |
| `ik_llama_src` | string | `""` | Path to a git checkout of `ik_llama.cpp`. |
| `ik_llama_build_dir` | string | `""` | CMake build directory for ik_llama. |
| `ik_llama_server_bin` | string | `""` | Path to ik_llama's built `llama-server`; empty leaves the engine disabled. |
| `ik_llama_models_ini` | string | `""` | ik_llama's own registry; empty resolves to a `-ikllama` sibling of `models_ini`. |
| `ik_llama_git_remote` | string | `"https://github.com/ikawrakow/ik_llama.cpp"` | Remote used to clone/update ik_llama. |
| `ik_llama_cmake_flags` | object | `{}` | Persisted CMake build flags for the ik_llama build. |
| `auto_load_model` | string | `""` | Model id to load automatically on launch. Empty string disables auto-load. |
| `pi_bin` | string | `""` | The pi coding agent: a `.js` entry, a package directory or a binary. Empty resolves the copy LlamaForge installed, then `PATH`. See [Connecting Coding Agents](agents.md). |
| `presets` | object | `{}` | Named knob sets: `{name: {knob: value}}`, managed from the dashboard. |
| `profiles` | object | `{}` | Launch profiles — a model plus an optional preset and an optional pinned prebuilt engine, started in one click: `{name: {model, backend, preset, engine}}`. `engine` is an install directory name under `engines/`, never a path. |
| `preset_bindings` | object | `{}` | Preset bound as each model's default, scoped by llama-family engine: `{engine: {model_id: preset_name}}`. |
| `preset_binding_snapshots` | object | `{}` | Engine-scoped values materialized by a preset binding: `{engine: {model_id: {knob: value}}}`. LlamaForge uses these snapshots to retain model values you subsequently change yourself. |
| `mtp_auto_owned` | object | `{}` | Bookkeeping for automatic MTP wiring: `{engine: {model_id: {key: value}}}` records the `models.ini` values LlamaForge wrote itself, so a value you changed or deleted is treated as yours and left alone. |
| `ui_mode` | string | `"lite"` | `"lite"` shows a curated knob set; `"advanced"` exposes every flag your `llama-server --help` lists (200+ on current builds). |
| `onboarded` | bool | `False` | Whether the first-run wizard has already been shown; set to `True` once dismissed. |
| `anthropic_default_model` | string | `""` | Fallback local model id used by the Anthropic-compatible shim when a request doesn't map to one. |
| `anthropic_shim_enabled` | bool | `True` | Whether `/v1/messages` (Anthropic-compatible) is served. |
| `wiki_dir` | string | `""` | Context-doc directory for the wiki feature. Empty string resolves to `<repo root>/wiki`. |
| `wiki_profiles` | object | `{}` | Named context-doc profiles: `{name: {"docs": [...], "description": str}}`. |
| `wiki_active` | object | `{}` | Active profile per model: `{model_id: profile_name}`. |
| `theme` | string | `""` | Light or dark. Empty string follows the OS/`localStorage`; otherwise `"light"` or `"dark"`. |
| `cvd` | bool | `False` | Enables the colorblind-safe palette and non-color status cues. |
| `skin` | string | `""` | UI skin. Empty string is the default, `"stowage"`; the others are `"hearth"` and `"classic"`. See [Theming](theming.md). |
| `vram_bandwidths` | object | `{}` | Optional `{vram_bw, ram_bw, disk_bw}` GB/s overrides for the VRAM-fit estimate; empty uses GPU presets/defaults. |
| `vram_predict_enabled` | bool | `True` | Whether the offline VRAM-fit/tok-s estimate is computed (Discover, on expand). |
| `docs_dir` | string | `""` | Directory the in-app docs viewer reads from. Empty string resolves to `<repo root>/docs/content`. |
| `embers_dir` | string | `""` | Where ember wikis and `embers.db` live. Empty string resolves to `<repo root>/embers`. |
| `tts_dir` | string | `""` | Where the speech model (`models/`) and voice clips (`voices/`) live. Empty string resolves to `<repo root>/tts`. |
| `tts_default_voice` | string | `""` | Voice clip name used when a speech request names no known voice. Empty = the model's own default voice. |
| `embers_scheduler` | bool | `True` | Run due ember jobs inside the dashboard process. |
| `embers_swap_models` | bool | `True` | Let an ember load its pinned model when the router is idle. |
| `multi_model` | bool | `False` | Let the router hold several models at once. Off keeps one model at a time. Changing it restarts a running router. |
| `slot_cap` | int | `3` | Most models loaded together, 2–4. Changing it restarts a running router. |
| `slot_headroom_mib` | int | `1536` | VRAM kept free per GPU beyond every plan, in MiB (0–32768). |
| `slot_autoload` | bool | `False` | Let client requests load models themselves, which bypasses the placement planner. Changing it restarts a running router. |
| `slots` | object | `{"main": "", "placed": {}}` | Multi-model bookkeeping: the main model id, and the `models.ini` keys LlamaForge wrote per placed model (so turning `multi_model` off can put the file back the way you wrote it). |
| `vision_off` | object | `{}` | Models whose vision projector is turned off, scoped by llama-family engine: `{engine: {model_id: mmproj path}}`. Written by the Vision switch; a rescan keeps these off. |
| `model_builds` | object | `{}` | Per-model engine pin: `{model_id: install dir name or "ik_llama"}`. A model pinned to a build other than the router's own runs in its own process. |
| `slot_port_base` | int | `8100` | First port used for those per-model processes. |
| `game_mode` | bool | `False` | Unload models while a game runs (Setup > Game mode). |
| `game_mode_keep` | list | `[]` | Model ids Game mode leaves loaded. |
| `game_mode_games` | list | `[]` | Extra executable names that count as games, e.g. `eldenring.exe`. |
| `router_watchdog` | bool | `True` | Restart the llama.cpp router if it crashes, backing off and giving up after 5 quick deaths with the reason from its log. |
| `game_mode_restore` | bool | `True` | Load the unloaded models again when the game closes. |
| `elastic_vram` | bool | `False` | Move a loaded model's layers to the CPU when other apps need VRAM, and back after (Setup > Elastic VRAM). NVIDIA only. |
| `elastic_grow` | bool | `True` | Grow resized models back automatically once the VRAM is free. Off: only the dashboard chip grows them back. |
| `elastic_keep_conversation` | bool | `True` | Save and restore the model's conversation (slot state) around each resize. Needs a router started with `--slot-save-path`, which LlamaForge passes only while this and `elastic_vram` are on: restart the router once after turning them on. |
| `elastic_headroom_mib` | int | `1536` | VRAM to keep free on each GPU, 512-16384 MiB. A model shrinks when less than half of it is free. |
| `elastic_placed` | object | `{}` | Internal: the `models.ini` keys Elastic VRAM wrote per model and the values they replaced, so they can be taken out again, plus the free memory per GPU the model had at your own settings (so it can grow back there after a LlamaForge restart). Not writable from the browser. |

64 keys total, matching `DEFAULTS` in `backend/config.py`.

`prebuilt_channel` (`"nightly"` or `"stable"`, the channel the Build / Update tab's prebuilt-engine card follows) is not in `DEFAULTS`: it is written the first time you install a prebuilt engine, and the card reads it as `"nightly"` until then.

`POST /api/config` accepts only the settings the dashboard itself changes: `ui_mode`, `theme`, `cvd`, `skin`, `onboarded`, `auto_load_model`, `wsl_distro`, `vllm_port`, `model_dirs`, `anthropic_default_model`, `anthropic_shim_enabled`, `vram_bandwidths`, `vram_predict_enabled`, `multi_model`, `slot_cap`, `slot_headroom_mib`, `slot_autoload`, `elastic_vram`, `elastic_grow`, `elastic_keep_conversation`, `elastic_headroom_mib`, `game_mode`, `game_mode_keep`, `game_mode_games` and `game_mode_restore`. Any other key is refused and named in the response; the rest are edited by their own endpoints or by hand in `config.json`.

## Preset bindings

Binding a preset makes it the default for a model in the active llama-family
engine (`llamacpp` or `ikllama`). When a binding is created or its preset is
edited, LlamaForge materializes only preset knobs that the model has not set.
`preset_binding_snapshots` records those materialized values so unbinding,
deleting a preset, or removing a knob cleans up only values that are still
unchanged; manual model overrides remain in `models.ini`. Older flat binding
maps are migrated to the engine that was active when they were saved.

## Loading and saving

`config.load()` deep-copies `DEFAULTS` and overlays whatever is present in `config.json` on disk, so a config file written before a new key was added still works — the new key simply falls back to its default. `config.save()` writes the full in-memory dict back to disk as indented JSON.

`config.migrate()` runs once at server startup (`backend/server.py` `main()`) to classify pre-existing installs: a config file with no `ui_mode` key is treated as a legacy install. If `server_bin` is already set, it is stamped `ui_mode: "advanced"` and `onboarded: True`; otherwise it gets `ui_mode: "lite"` and `onboarded: False`, so the onboarding wizard shows. The migration is idempotent — a config that already has `ui_mode` is returned unchanged.

See also [models.ini Format](models-ini.md) for the preset file `models_ini` points at, and [HTTP API](api.md) for the endpoints that read and write these keys.

## Network Access and historical configuration

`POST /api/network` is the supported way to change the router scope and key. It
uses explicit `keep`, `generate`, `replace`, and local-only `clear` actions;
newly configured LAN access requires a usable key and LlamaForge-owned starts
fail closed without one. `config.json` is a normal plaintext file, not an OS
credential vault.

LlamaForge assesses existing values read-only. A printable older LAN key can
remain in use as `protected_legacy` and is marked for rotation. A manually edited
unsupported host, or LAN with an absent or invalid key, remains visible as
`unsafe_legacy`; it is not auto-rewritten, but future starts and restarts are
blocked until you generate/replace a key or return the router to local access.
