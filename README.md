<p align="center">
  <picture>
    <source srcset="docs/hero.webp" type="image/webp">
    <img src="docs/hero.png" alt="LlamaForge - a control panel for llama.cpp" width="100%">
  </picture>
</p>

<p align="center"><b>The real llama.cpp server, run from a browser tab.</b><br>
Pick a model, see if it fits your GPU, load it, chat. Every llama-server flag is still there when you want it.</p>

<p align="center">
  <a href="https://github.com/dadwritestech/LlamaForge/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/dadwritestech/LlamaForge/ci.yml?branch=master&style=flat-square&labelColor=0f1315&color=39d98a&label=CI"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-c8d2d4?style=flat-square&labelColor=0f1315"></a>
  <img alt="status" src="https://img.shields.io/badge/status-early%20preview-ff5c57?style=flat-square&labelColor=0f1315">
  <a href="https://github.com/dadwritestech/LlamaForge/stargazers"><img alt="stars" src="https://img.shields.io/github/stars/dadwritestech/LlamaForge?style=flat-square&labelColor=0f1315&color=ffb000&cacheSeconds=1800"></a>
</p>

```powershell
irm https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/install.ps1 | iex   # Windows, no admin
```
```bash
curl -fsSL https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/install.sh | sh   # Linux / macOS
```
Then **Install llama.cpp** (the official build for your GPU, no compiler) → **Discover** → **Chat**.
Early preview: Windows + NVIDIA is the most-tested path. Linux and macOS pass CI but have had little real-hardware use.

<p align="center">
  <img src="docs/demo.gif" alt="Install, fetch llama.cpp, find a model that fits, chat" width="100%">
</p>

LlamaForge runs no models itself. It installs and drives llama.cpp's own `llama-server` router and
edits its `models.ini` for you. Not affiliated with ggml-org. If you'd rather have polish than
control, use [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com) or [Jan](https://jan.ai).

## Why LlamaForge

New model architectures land in llama.cpp first. Desktop apps pass them on when they next
update their bundled engine. LlamaForge runs the **official llama.cpp release itself** (or
your own build or fork), so a new model is one **Update** click away, and it puts a UI over
every server flag instead of a curated subset.

<p align="center">
  <img src="docs/screenshot-new-this-week.png" alt="Discover opens on New this week: model architectures llama.cpp just merged, each marked IN YOUR ENGINE or UPDATE ENGINE against your build, above Hugging Face GGUFs trending in the last 14 days" width="100%">
  <br><sub>Discover opens on what llama.cpp just learned to run, checked against the engine you have.</sub>
</p>

| | **LlamaForge** | LM Studio | Ollama |
|---|---|---|---|
| Open source | ✅ MIT | ❌ proprietary app | ✅ MIT |
| Engine | official upstream llama.cpp builds, any version, or your own fork | LM Studio's bundled llama.cpp / MLX runtimes | Ollama's own engine on ggml |
| Per-model settings | every flag your `llama-server --help` lists (200+ on current builds) | many, curated | Modelfile parameters |
| Any GGUF from Hugging Face, rated for your VRAM before download | ✅ (a rough estimate) | ✅ | pulls GGUFs, no fit rating |
| OpenAI + Anthropic-compatible API | ✅, plus one-click Claude Code / Codex / pi.dev config | ✅ | ✅ |
| Backend dependencies | none (Python stdlib) | – | – |
| Native desktop app | ❌ runs in your browser | ✅ | ✅ |
| Maturity | **early preview** | mature | mature |

<sub>As of October 2026, to the best of our knowledge. Spot something wrong? A PR to fix this table is very welcome.</sub>

## What's in it

- **Models**: every model on your machine in one list, with live VRAM/util/temp per GPU. Load, unload, or open Chat from the row.
  - Expand a model to edit every llama-server flag, grouped and searchable, next to a GGUF metadata card (architecture, quant, trained context, layers).
  - Save reloads the model in place. A failed load shows the last error from the router log with a best-guess hint.
  - Presets, launch profiles (model + preset + pinned llama.cpp build in one click), side-by-side compare, and copy-paste client snippets.
- **Chat**: llama.cpp's own chat UI (markdown, reasoning, images) inside the dashboard, on its own port, with the API key added for you.
- **Discover**: Hugging Face GGUF search that opens on what's new this week. Every quant gets a rough fit rating for your VRAM before you download (FITS / TIGHT / CPU OFFLOAD). Downloads resume after interruption, register themselves, and end in **Load & Chat**.
- **Will it run?**: pick a repo and quant, get the fit and a rough speed estimate.
- **Build / Update**: one-click official llama.cpp builds with rollback, or build from source with flags detected for your GPU. Also drives [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and, on Windows, [vLLM](https://github.com/vllm-project/vllm) in WSL2.
- **Stats**: per-model tokens, speed and run counts from the router's own metrics. Per-client stats aren't possible, because clients talk to the router directly.
- **Context**: Markdown context docs composed into profiles, injected into requests or written into `CLAUDE.md` / `AGENTS.md`.
- **Recipes**: share a profile as readable JSON; others import it in one paste and LlamaForge downloads the model if missing. There's a [community gallery](recipes/).

A first-run wizard and a **Lite / Advanced** toggle keep the deep knobs out of the way until you want them.

## Use it from other apps

Anything that speaks the OpenAI API (Open WebUI, SillyTavern, Continue, Cline, Aider, the OpenAI SDKs) works against
`http://127.0.0.1:8080/v1`. The router always runs with an API key, and **Client Config** on any model gives you the
base URL, key and model id ready to paste.

- **Anthropic-compatible** `POST /v1/messages` on the panel, with streaming and tool use.
- **Connect an agent** writes the config for **Claude Code**, **Codex** and **pi.dev** (any file it touches is backed up first).
- Load/unload endpoints let an agent swap models on demand.

## Screenshots

| Models | Discover with fit ratings |
|---|---|
| ![Models](docs/content/img/models.png) | ![Discover](docs/content/img/discover.png) |

| Build / Update | Setup |
|---|---|
| ![Build / Update](docs/content/img/build.png) | ![Setup](docs/content/img/setup.png) |

## Install

The one-liners above need no git, admin or compiler. Re-run them any time to update.

The installer finds Python 3.10+ (on Windows it drops a private, SHA-256-pinned copy of
python.org's embeddable Python if you have none), downloads the latest release, adds a
Start menu / app-menu entry (`~/Applications/LlamaForge.app` on macOS, plus a `llamaforge`
command on Linux/macOS) and opens the dashboard. Updating keeps your settings, models and
engines. Uninstall from **Apps & Features** on Windows, or `llamaforge uninstall`; it asks
before touching your settings or models.

<details><summary>From source (git clone)</summary>

```powershell
git clone https://github.com/dadwritestech/LlamaForge
cd LlamaForge
powershell -ExecutionPolicy Bypass -File bootstrap.ps1   # Windows
./bootstrap.sh                                           # Linux / macOS
```

The bootstrap script checks for Python and Git (asking before installing anything),
writes `config.json` and opens the dashboard.
</details>

**Daily use:** open **LlamaForge** from the Start menu / app menu, or run `llamaforge`.
It starts the router and the dashboard and opens your browser.

- Dashboard: http://127.0.0.1:8090
- Chat: http://127.0.0.1:8091 (also the dashboard's **Chat** tab)
- API for your other apps: http://127.0.0.1:8080/v1

`llamaforge stop` (or `stop.ps1` / `stop.sh`) shuts down the dashboard, the router and
the models it spawned. It stops only processes LlamaForge started; any other
llama-server you run is left alone.

**Requirements:** Windows 10/11, Linux, or macOS on Apple Silicon. Python 3.10+ (stdlib only;
the Windows installer brings its own). NVIDIA (CUDA), AMD/Intel (Vulkan) or Apple (Metal) GPUs
are used when present; CPU-only works. Building from source also needs Git, CMake, Ninja and
a C++ compiler, which the Setup tab can install where a package manager allows.

## How it works

LlamaForge contains no llama.cpp code. A pure-stdlib Python backend drives llama.cpp's own
router API, edits `models.ini`, and fetches official builds (or shells out to `git` / `cmake`).
The knob list is parsed live from `llama-server --help`, so it tracks whatever build you run.

LlamaForge doesn't pin context size, GPU layers or the multi-GPU split. llama.cpp's `--fit`
(on by default) sizes all three to your free VRAM at load, and moves MoE experts to CPU when
needed. Pinning any of them turns fit off, so they stay unset unless you set them yourself.

**Security:** the dashboard only listens on `127.0.0.1`. The router is keyed even when
local, so a web page you visit can't drive it. LAN access is opt-in from Setup and always
needs a key. Details and how to report a vulnerability privately: [SECURITY.md](SECURITY.md).

## Docs

Everything is in the in-app **Help** tab and at **[dadwritestech.github.io/LlamaForge](https://dadwritestech.github.io/LlamaForge/)**:
[configuration](docs/content/config.md), [keyboard shortcuts](docs/content/keymap.md),
[themes & colorblind-safe mode](docs/content/theming.md), [vLLM](docs/content/vllm.md),
[troubleshooting](docs/content/troubleshooting.md), and [what's new](docs/content/whats-new.md).
[ROADMAP.md](ROADMAP.md) has what's shipped and planned; it's an early preview, so priorities follow feedback.

## Credits & license

LlamaForge is MIT-licensed ([LICENSE](LICENSE)). It builds and drives
**[llama.cpp](https://github.com/ggml-org/llama.cpp)** - MIT, (c) The ggml authors -
see [NOTICE](NOTICE) and [LICENSE.llama.cpp.txt](LICENSE.llama.cpp.txt).
The hard part is theirs; please star and support the upstream project.
