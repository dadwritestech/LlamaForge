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
| `router_host` | string | `"127.0.0.1"` | Router bind address. The Network Access UI supports only `127.0.0.1` (local) and `0.0.0.0` (LAN). |
| `router_api_key` | string | `""` | Plaintext API key required for LAN. It is not returned in ordinary dashboard state. |
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
| `presets` | object | `{}` | Named knob sets: `{name: {knob: value}}`, managed from the dashboard. |
| `preset_bindings` | object | `{}` | Preset bound as each model's default, scoped by llama-family engine: `{engine: {model_id: preset_name}}`. |
| `preset_binding_snapshots` | object | `{}` | Engine-scoped values materialized by a preset binding: `{engine: {model_id: {knob: value}}}`. LlamaForge uses these snapshots to retain model values you subsequently change yourself. |
| `ui_mode` | string | `"lite"` | `"lite"` shows a curated knob set; `"advanced"` exposes all ~220 llama-server flags. |
| `onboarded` | bool | `False` | Whether the first-run wizard has already been shown; set to `True` once dismissed. |
| `anthropic_default_model` | string | `""` | Fallback local model id used by the Anthropic-compatible shim when a request doesn't map to one. |
| `anthropic_shim_enabled` | bool | `True` | Whether `/v1/messages` (Anthropic-compatible) is served. |
| `wiki_dir` | string | `""` | Context-doc directory for the wiki feature. Empty string resolves to `<repo root>/wiki`. |
| `wiki_profiles` | object | `{}` | Named context-doc profiles: `{name: {"docs": [...], "description": str}}`. |
| `wiki_active` | object | `{}` | Active profile per model: `{model_id: profile_name}`. |
| `theme` | string | `""` | UI theme. Empty string follows OS/`localStorage`; otherwise `"light"` or `"dark"`. |
| `cvd` | bool | `False` | Enables the colorblind-safe palette and non-color status cues. |
| `vram_bandwidths` | object | `{}` | Optional `{vram_bw, ram_bw, disk_bw}` GB/s overrides for the VRAM-fit estimate; empty uses GPU presets/defaults. |
| `vram_predict_enabled` | bool | `True` | Whether the offline VRAM-fit/tok-s estimate is computed (Discover, on expand). |
| `docs_dir` | string | `""` | Directory the in-app docs viewer reads from. Empty string resolves to `<repo root>/docs/content`. |

36 keys total, matching `DEFAULTS` in `backend/config.py`.

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
