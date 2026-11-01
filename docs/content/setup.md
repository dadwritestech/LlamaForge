---
title: Setup
section: guides
order: 4
---

# Setup

Check build prerequisites, install what is missing with your permission, scan your drives for GGUF models already on disk, and prune registry entries whose files are gone.

## What it does

The Setup tab is a column of cards. The four described first (prerequisites, installing missing tools, drive scan, registry prune) each have their own backend module; the cards for [Network Access](#network-access), [Connect an Agent](#connect-an-agent), [Game mode](#game-mode), [Elastic VRAM](#elastic-vram), [Sleep idle models](#sleep-idle-models), multi-model, speed estimates and vLLM follow below.

**Prerequisite detection.** `backend/prereqs.py` checks for four command-line tools — `git`, `cmake`, `ninja`, and `python` (via `shutil.which`, running `--version` to confirm and capture the installed version) — plus the platform C++ compiler and the CUDA toolkit. On Windows, the compiler check (`find_msvc()`) shells out to `vswhere.exe` to locate an MSVC install with the C++ desktop workload, falling back to a glob for `cl.exe` under `Program Files\Microsoft Visual Studio`. On macOS/Linux it looks for `clang++` or `g++` on `PATH`. CUDA detection reads `nvcc --version` (or `$CUDA_PATH`) and is skipped entirely on macOS, where Metal is used instead.

**Installing missing tools.** Which package manager runs depends on the OS, verified directly in `prereqs.py`:

- **Windows:** `install()` tries `winget install --id <pkg> -e --accept-source-agreements --accept-package-agreements` first; if `winget` isn't on `PATH` or the install fails, it falls back to `choco install <pkg> -y`.
- **macOS:** `install()` requires Homebrew and runs `brew install <pkg>` — no `sudo` needed.
- **Linux:** the dashboard never runs `sudo`. `install()` returns `ok=False` with the exact command to paste into a terminal (`sudo apt-get install -y <pkg>`, `sudo dnf install -y <pkg>`, or `sudo pacman -S --noconfirm <pkg>`, picked by `osplat.linux_pkg_manager()` probing `apt-get` → `dnf` → `pacman` in that order).

Each tool in `prereqs.TOOLS` carries its own `winget`/`choco`/`brew`/`pkg` identifiers and a fallback download URL, used whichever path applies. `_can_auto_install()` gates the **Install** button: on Windows it needs `winget` or `choco` present; on macOS it needs `brew`; on Linux the button never appears — only the hint.

After a successful install, the running process's `PATH` is refreshed from the registry (`osplat.refresh_path()`, Windows-only) and the tool is re-probed, so a freshly installed `ninja` or `cmake` is detected immediately. Only if it still isn't visible does the result ask you to restart LlamaForge — the old behavior (always MISSING until a restart) is gone.

**Folder or drive scan for GGUFs.** `backend/scanner.py`'s `scan()` (via `POST /api/scan`) walks the folders entered in Setup, one path per line. Saving the list persists it as `model_dirs`; clearing it restores the platform defaults. Those defaults are every fixed drive letter on Windows (`GetLogicalDrives` + `GetDriveTypeW == DRIVE_FIXED`, so removable/network drives are excluded), `$HOME` plus `/Volumes` on macOS, and `$HOME` plus any of `/mnt`, `/media`, `/srv`, `/data` on Linux. The scanner finds `.gguf` files at least 50 MB while skipping recycle bins, `.git`, and `.trash` folders. Multi-shard GGUF sets collapse to their first shard, `mmproj*` files attach to a sibling model of a known vision architecture instead of appearing standalone, `mtp-*` speculative draft sidecars attach as a `spec-draft-model` (see [models.ini Format](models-ini.md)), and files with `embed` in the name are flagged as embedding endpoints. Confirmed results are written into `models.ini` via `POST /api/scan/apply`.

**Registry prune.** `POST /api/scan/prune` ("Check for deleted models") takes a list of model IDs, re-checks each one's `model` path in `models.ini` against disk, and — only for entries whose file no longer exists — unloads it from the router if currently loaded, then removes its section from `models.ini` with `config.remove_section()`. An entry whose file has reappeared since the check is left alone. This edits `models.ini` only; it never touches files on disk.

The tab also surfaces `hardware.recommend()`'s detected CPU/GPU (shared with the Build tab), lets you pick a model under **Startup** to auto-load on launch (`auto_load_model` in `config.json`), and — on Windows — the vLLM/WSL2 install flow described in [vLLM Backend](vllm.md).

**Speed Estimates.** An optional card (marked advanced) with **VRAM GB/s**, **RAM GB/s** and **Disk GB/s** fields. They override the memory-bandwidth figures behind the "Will it run?" panel and Discover's speed badges (`vram_bandwidths` in `config.json`); blank means the detected preset or default.

**Multi-model.** The **Multi-model (llama.cpp)** card turns on loading several models at once (`multi_model`), with **at most this many** (`slot_cap`, 2-4), **VRAM kept free per GPU (MiB)** (`slot_headroom_mib`) and **let client requests load models** (`slot_autoload`, which bypasses the VRAM check). Changing the switch, the count or client loading restarts the router and unloads every loaded model, so the card asks first. It needs llama.cpp: ik_llama has no router mode. See [Models & Tuning](models.md).

## How to use it

1. Open the **Setup** tab. **Prerequisites** lists Git, CMake, Ninja, Python, your C++ compiler, and CUDA (if applicable), each marked present/missing with its detected version.
2. Click **Install** next to a missing tool to install it with your OS's package manager (Windows: winget, falling back to choco; macOS: Homebrew). On Linux, copy the shown command into a terminal yourself — the dashboard never runs `sudo`.
3. Review **Detected Hardware** — your CPU and any GPUs found, shared with the Build tab's flag recommendations.
4. Under **Scan Drives for Models**, enter one folder per line and optionally click **Save folders**, then click **Scan for GGUF models (and Ollama's)**. Leave the list blank to scan all fixed drives (or `$HOME` plus mounted volumes); review the results and click **Add N models to config** to register them. Models Ollama downloaded are listed too, marked **from Ollama** (see [below](#models-you-already-have-in-ollama)).
5. Click **Check for deleted models** to find registry entries whose backing file no longer exists on disk, then click **Remove N missing** to prune them.
6. Optionally pick a model under **Startup** to auto-load when LlamaForge launches.

## Models you already have in Ollama

Ollama keeps each model as a plain GGUF file with no `.gguf` suffix (`~/.ollama/models/blobs/sha256-...`, or wherever `OLLAMA_MODELS` points), named by a manifest. A drive scan can't recognize those files by name, so the scan reads Ollama's manifests instead and lists every model it has downloaded, marked **from Ollama**, with its vision projector attached when the manifest has one.

**Add** points `models.ini` at Ollama's own file. Nothing is copied or moved, so there's no second copy on disk and Ollama keeps working. If you `ollama rm` the model later, **Check for deleted models** finds the entry.

Most models load as is. A model Ollama converted for its own engine can hold tensors upstream llama.cpp doesn't know; that one fails to load with llama.cpp's own error, and the same model from Hugging Face (**Discover**) is the fix.

## Game mode

Tick **unload models while a game is running** and LlamaForge frees your GPU when you start a game, then loads the same models again when you quit it.

- **What counts as a game:** anything running from a Steam, Epic, GOG, Xbox, Riot, EA or Ubisoft game folder; any program you list in the box (one name per line, `.exe` optional); and, on Windows, an app in exclusive full screen. Launchers themselves (Steam, the Epic launcher, the Riot client) don't count.
- **Exceptions:** tick models under **Keep these loaded during games** and they stay loaded; every other model is unloaded. A model a client loads while you play is unloaded too, unless it's on the list.
- **Timing:** it checks every 5 seconds and needs two sightings to start, so a launcher splash doesn't trigger it. It waits about 15 seconds after the game closes before loading models back, so a loading screen doesn't either.
- **Load them again when the game closes** is on by default; untick it to leave the GPU empty afterwards.
- Turning Game mode off mid-game brings the models back right away. The **now** line shows the detected game and how many models it unloaded.

## Elastic VRAM

New and not yet tested on many machines: treat it as an early preview. Off by default.

When another app (a game, Blender, ComfyUI, a video editor) needs your GPU, Elastic VRAM moves a loaded model's layers to the CPU instead of unloading it, and moves them back when the app is done. The conversation can be kept: LlamaForge saves the model's slot state before the resize and restores it after, so the next message doesn't re-read the whole chat.

- **When it shrinks:** it checks every 5 seconds. A model shrinks after two checks in a row with less than half the **VRAM to keep free per GPU** left on a GPU, and only if other apps use more of it than before: a model that fills the GPU on its own is left alone. "Before" starts with its first check after you turn it on (or after LlamaForge restarts), so an app that is already squeezing the GPU then only causes a shrink once it grows further. In multi-model mode, a model that is still loading counts as other apps' memory until it has loaded.
- **When it grows back:** after 30 seconds (six checks) in which the model fits at a bigger size with the VRAM to keep free plus 1 GB to spare (back at your own settings, the free memory you had there before is enough), and at least 2 minutes after the last change. If an app takes the memory back within 5 minutes of a grow, that wait doubles, up to 30 minutes; an hour without changes resets it to 2 minutes. It also holds off while the apps that caused the shrink still use most of what they used then.
- **What it changes:** MoE models move their experts to the CPU first (`n-cpu-moe`, a quarter of the layers at a time), then whole layers (`n-gpu-layers`); other models move layers only, a quarter at a time. A model never goes above the GPU layers you set. Context size, quant and cache type never change.
- **A resized model is much slower.** One real measurement: a 27B ran at 23 tokens/s fully on the GPU and 4 tokens/s at 40 of 64 layers. The dashboard shows **Elastic · 40/64 on GPU** on its row; click it to grow it back now, if it fits.
- **A resize reloads the model:** a few seconds for a small one, about 20 for a 27B. A request that arrives meanwhile waits or fails.
- **Your settings win.** It only edits the llama.cpp `models.ini`, records exactly which keys it wrote and what they replaced (`elastic_placed`, along with the free memory you had at your own settings, so a resized model still grows back after LlamaForge restarts), and takes them out again when the model grows back. A key you edit by hand in the meantime is yours: it is neither overwritten nor reverted. A resized model that gets unloaded gets your values back.
- **Turning it off isn't instant.** Each resized model goes back to full size once it fits again, after the same wait. The **now** line lists the models still waiting.
- **Grow back automatically** is on by default. Untick it and a resized model stays small until you click its dashboard chip.
- **Game mode comes first.** While Game mode is on and a game is running, Elastic VRAM stands by.
- **Turning it on takes effect at once**, but **keeping the conversation** needs one router restart afterwards (Network Access: **Apply & Restart Router**, keeping the key as it is). LlamaForge gives the router a save folder (`--slot-save-path`) only when it starts it while Elastic VRAM and keeping conversations are both on, and the binary has the flag; toggling them never restarts the router by itself, since that would unload the model it's protecting. The files go in a `slot-saves` folder beside `logs`, are deleted right after each resize, and any left by a crash are removed at startup. Until the restart, or on a router started some other way (the run script), the model is still resized, but the conversation isn't kept, and the log says so.
- **Where it works:** NVIDIA GPUs only, since it reads free memory from `nvidia-smi`. On a Mac, or without an NVIDIA GPU, the card says so and the setting does nothing. Only llama.cpp router models are resized: not models pinned to a build of their own, ik_llama or vLLM.

The card shows the current state on its **now** line and a log of the last 20 resizes: when, which model, from and to which size, why, how long it took and whether the conversation was kept.

## Sleep idle models

A model you loaded an hour ago and haven't used since still holds its VRAM. Turn on **Sleep idle models**, pick a number of minutes (1 to 1440), and Save: a model that gets no request for that long frees its VRAM and RAM but stays in the list as **sleeping**, with a hollow light. The next request to it, from Chat or any app, wakes it, usually in a second or a few since the file is still in the OS cache.

- It's llama.cpp's own `--sleep-idle-seconds`, written to the `[*]` section of `models.ini`, so it applies to every llama.cpp model **from its next load**. A model that's loaded now keeps running as before until you unload it or it's reloaded.
- It needs a llama.cpp build that has the flag; on an older one the card says to update it in Build. ik_llama and vLLM models aren't affected.
- The Stats poller only asks loaded models for numbers, so it doesn't keep a sleeping model awake.

## Network Access

The Network Access card controls the llama.cpp router, never the dashboard: the
panel and management API stay on `127.0.0.1`. Choose **This computer only** for
the router's `127.0.0.1` scope or **Devices on my local network** for its
`0.0.0.0` scope. LAN selection requires a usable key before Apply is enabled.
While [Elastic VRAM](#elastic-vram) keeps conversations, the router has a slot save
folder, so anyone holding the router API key can also use its slot save and restore.

Choose one unambiguous key action: **Keep the configured key**, **Generate a new strong key**,
**Replace with a key I provide**, or **Remove the key (local-only)**. Remove is available only with local access; moving
from LAN back to local otherwise retains the key. Generating or replacing over an
existing key, and removing an existing key, each require confirmation. Rotation
warns because clients using the previous key will stop authenticating. A generated
key is returned only by that explicit Generate action: it starts masked, can be
copied without displaying plaintext, and its one-time reveal expires after 30
seconds. Retry, Done, navigation, or a Setup rerender clears the value and its
copy closure.

Older printable LAN keys are retained as `protected_legacy` so an upgrade does
not force an immediate client outage, but the card recommends rotation. Unsupported
manual hosts and LAN configurations with an absent or invalid key are displayed as
`unsafe_legacy`, not silently rewritten. Repair them by generating/replacing the
key or returning to local-only; new router starts and restarts remain blocked until
then.

Applying saves a validated safe configuration and requests a restart. The status
separates saved settings from observed listener state: a port being occupied does
not prove that it belongs to LlamaForge or enforces the expected key. If restart
fails, the saved protected configuration remains, but the router is stopped or
unverified rather than claimed as active: “A protected configuration was saved,
but the router is stopped; LAN protection is not currently active or verified.”

## Connect an Agent

Changing an agent or model selection does not fetch credentials. Press **Show
configuration** to make the explicit POST preview, then choose **Apply** only if
you want LlamaForge to write the agent's local config file. Agent setup supports
the active llama-family backend; vLLM agent setup is deferred. Context injection
for Codex and pi uses the loopback panel endpoint, so it is local-machine-only. The same card installs the pi coding agent and lists the MCP server snippets; see [Connect an Agent](agents.md) and [MCP Server](mcp.md).

## Screenshot

![Setup tab](docs/img/setup.png)

## Reference

| Concept | Source | Behavior |
|---|---|---|
| Prereq detection | `prereqs.status()` | `git`, `cmake`, `ninja`, `python` via `shutil.which` + `--version`; C++ compiler and CUDA toolkit detected separately. |
| Windows install | `prereqs.install()` | `winget install --id <id> -e ...`, falling back to `choco install <id> -y` if winget is absent or fails. |
| macOS install | `prereqs.install()` | `brew install <pkg>` — requires Homebrew, no sudo. |
| Linux install | `prereqs.install()` | Never runs sudo; returns the exact `apt-get`/`dnf`/`pacman` command to run yourself. |
| Drive scan roots | `scanner.list_drives()` | Windows: all fixed drive letters. macOS: `$HOME` + `/Volumes`. Linux: `$HOME` + any of `/mnt`, `/media`, `/srv`, `/data` that exist. |
| Selected scan roots | `config.json: model_dirs` | One or more directories persisted from Setup. An explicit empty list selects the platform defaults. |
| GGUF discovery | `scanner.find_ggufs()` | `.gguf` files ≥ 50 MB; skips recycle bin/`.git`/`.trash`; collapses multi-shard sets; attaches `mmproj` and `mtp-*` siblings; flags `embed` files. |
| PATH refresh after install | `osplat.refresh_path()` | Windows-only; re-reads PATH from the registry so a just-installed tool is detected without a restart. |
| Registry prune | `POST /api/scan/prune` | Removes a `models.ini` section only if its `model` file no longer exists on disk; unloads it from the router first if loaded. |
| Auto-load | `config.json: auto_load_model` | Model ID to load automatically once the router is ready after launch. |

## Troubleshooting

If **Install** doesn't appear for a missing tool, no supported package manager was found for your OS (Windows without winget or choco, macOS without Homebrew) — use the tool's download URL shown next to it instead. If a Windows install reports "winget failed" then falls through to choco, check the combined output shown in the toast/log for the underlying error (often a source-agreement prompt or an already-installed conflicting version). If the drive scan finds nothing on Windows, confirm your models sit on a fixed (non-removable, non-network) drive — `list_drives()` skips those by design. If **Check for deleted models** reports an entry you know still exists, verify the `model` path in `models.ini` matches the file's real location; a moved file reads as "deleted" until you re-scan and re-apply it.

If Network Access says settings were saved but router start is not verified,
read the displayed start error and router log, then use Retry after fixing it.
The port check only observes a listener; it cannot prove listener ownership or
authentication. An `unsafe_legacy` warning means LlamaForge deliberately refused
to start/restart until the configured host and key are repaired.

See also [Build & Update](build.md) for the same hardware detection used to pick CMake flags, and [vLLM Backend](vllm.md) for the WSL2 install flow shown at the bottom of this tab on Windows.
