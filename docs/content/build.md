---
title: Build & Update
section: guides
order: 3
---

# Build & Update

Check whether your local llama.cpp checkout is behind upstream, and rebuild it with CMake flags auto-detected for your machine's GPU and CPU — without hand-typing `cmake` commands.

## What it does

The Build tab (labelled **Build / Update** in the sidebar) does three things: installs an official prebuilt llama.cpp (no compiler), reports how your local `llama_src` checkout compares to upstream `github.com/ggml-org/llama.cpp`, and shows the CMake flags this machine should build with.

**Official prebuilt builds.** On the llama.cpp target, the top card, **Engine · official llama.cpp builds**, installs a binary from `github.com/ggml-org/llama.cpp` releases with no compiler or checkout needed (`backend/prebuilt.py`, `engines/llama.cpp/` under the LlamaForge folder). It picks the variant for this PC (the **variant** dropdown lists the alternatives; the default is `auto`) and a **channel**: `nightly` (every upstream merge) or `stable` (curated releases). The install is checked against GitHub's SHA-256 and test-run before the router is switched to it; the button reads **Install llama.cpp <tag>**, **Update to <tag>** or **Reinstall**, and **Check now** refreshes the lookup. Older installs stay on disk (the newest 3 are kept) and each has a **Use** button to roll back. The source-build cards below it are headed **Or build from source**.

**Compare before you switch.** With more than one build installed, the Engine card has a **compare speed** row: pick one of your other builds and a model, then click **Compare with your build**. It's the same `llama-bench` comparison as a PR build's [Compare speed](#is-the-pr-actually-faster): every model unloaded, pp512 and tg128 three times on each build, the change marked *within noise* when it's inside the spread. So you can see whether a newer nightly is slower on your own model before pressing **Use**. Both builds need `llama-bench` next to `llama-server`; the official zips include it.

**Current vs. upstream.** `backend/builder.py`'s `BuildManager.current_commit()` reads the local checkout's HEAD (`git log -1`) and branch. `check_updates()` runs `git fetch --quiet origin`, works out the remote's default branch for that checkout (`origin/HEAD`, else `origin/main`, else `origin/master`; llama.cpp uses `master`, ik_llama.cpp uses `main`), then runs `git rev-list --count HEAD..<that branch>` to report how many commits you're behind, plus the latest upstream commit's hash and subject. This check is cached for 15 minutes (`UPDATE_TTL = 900` seconds) so opening the Build tab doesn't `git fetch` GitHub on every visit; a failed fetch is retried sooner (`UPDATE_TTL_FAIL = 60` seconds). The **Check GitHub now** button bypasses the cache with `force=1`, and a finished build clears the cache outright so the next check is fresh.

**Auto-detected build flags.** `backend/hardware.py`'s `recommend()` inspects your GPU(s) (via `nvidia-smi`, or Apple's unified memory on macOS) and CPU, then returns a dict of `{cmake_flags, notes, runtime, gpus, cpu}`:

- On **macOS**, it returns `GGML_METAL=ON` and `GGML_NATIVE=ON`, with a note that Metal uses unified memory as VRAM. No CUDA branch runs on Mac.
- If one or more **NVIDIA GPUs** are detected, it sets `GGML_CUDA=ON`, and if compute capabilities were readable, `CMAKE_CUDA_ARCHITECTURES` to the sorted, deduplicated list of detected architectures (e.g. `86;89`). It also sets `GGML_CUDA_FA_ALL_QUANTS=ON` to enable flash attention across all quantized KV cache combinations.
- If **no NVIDIA GPU** is detected, it configures a CPU-only build (no CUDA flags) and notes that fact.
- On Windows/Linux either way, it always sets `GGML_NATIVE=ON`. If the CPU looks like it supports AVX-512 (`avx512_hint` — on Linux read from real CPU flags; on Windows a name-based heuristic for Ryzen 7/9 (7000- and 9000-series), Xeon, and Threadripper parts), it additionally sets `GGML_AVX512=ON`, `GGML_AVX512_VNNI=ON`, `GGML_AVX512_VBMI=ON`, and `GGML_AVX512_BF16=ON`.
- It also returns a `runtime` recommendation (not a CMake flag, but a suggested per-model default): `fit=on` and `flash-attn=auto` on every platform. Memory sizing (context, GPU layers, split) is left to llama.cpp's own `--fit` rather than pinned.

These flags are shown as pills on the Build tab and used as the default for a rebuild; if you've previously saved custom flags (`cmake_flags` in `config.json`), those are shown instead until you clear them.

A rebuild (`BuildManager.run_build()`) runs in a background thread: it first validates that `llama_src`/`build_dir` are set and the source exists (an unset path is reported plainly instead of running `cmake -B  -S ` and surfacing CMake's own error), optionally `git pull --ff-only origin`, backs up the current binaries directory (`bin/Release` on Windows/MSVC, `bin` elsewhere) to a timestamped copy beside it (e.g. `bin/Release-backup-20261006-120000`) before touching anything, runs `cmake -B <build_dir> -S <llama_src> -DCMAKE_BUILD_TYPE=Release` with your flags, then `cmake --build <build_dir> --config Release --parallel <jobs>`. Output streams to a log file the dashboard polls live. On success it records where `llama-server` actually landed into `config.json`, so a first build doesn't leave you hand-fixing `server_bin`.

**"Built, with warnings."** `cmake --build` returns a single exit code for the whole build, so a failing non-essential target (the npm/`sharp` UI-asset step on Windows is the common one) used to be reported as a hard **BUILD FAILED** even though `llama-server` itself built. Now, if the build step fails but a *freshly built* `llama-server` is present (freshness proven by comparing its mtime to the build's start, so a stale binary from a previous build can't mask a real compile failure), the build is reported as **built, with warnings** — the binary is recorded and usable, and the failing step is left in the log for you to read. A build that produced no fresh binary is still a hard failure.

**Two llama-family engines.** The Build tab has a **Build Target** toggle (**llama.cpp** / **ik_llama**). Building the `ik_llama` target uses its own `ik_llama_src` / `ik_llama_build_dir` / `ik_llama_cmake_flags` and produces a separate binary. When the target isn't the active engine, a **Switch to <engine>** button (`POST /api/engine/switch`) points the router at it by setting `active_engine` and restarting the router. The switch is gated on a capability probe (`router_ctl.supports_router_mode()`): because LlamaForge drives the router as `<server_bin> --models-preset ...`, a binary whose `llama-server` predates router mode is **refused** with an explanation rather than being switched to and taking the router down. Each engine reads its own `models.ini` (ik_llama uses a `-ikllama` sibling), and the knob editor reflects whichever engine is active.

## How to use it

1. Open the **Build / Update** tab. To skip the compiler, use the **Engine** card at the top (llama.cpp target) and click **Install llama.cpp**. To build from source, the **Or build from source** card (headed **Current Build** on the ik_llama target) shows your checkout's commit, branch, and date; **Upstream** shows whether you're behind the remote's default branch and by how many commits (an error is shown if that branch can't be found, never a false "up to date").
2. Click **Check GitHub now** to force a fresh upstream check instead of waiting for the 15-minute cache.
3. Review the **Build Flags** pills — these are auto-detected for your GPU/CPU (or your previously saved custom flags, if any).
4. Leave **git pull first** checked (it's checked by default when you're behind) to pull the latest commits before building, or uncheck it to rebuild the current checkout as-is.
5. Click **Pull latest & Rebuild** (or **Rebuild current** if already up to date). The build runs in the background; watch progress and any errors in the Build Log panel below, which polls live until the build finishes or fails.
6. On Windows, the same tab has a **vLLM (pip package in WSL)** card showing its installed vs. latest PyPI version, with an **Update vLLM to <version>** button when a newer release is available.

## Try a llama.cpp PR

Someone opened a pull request that makes your GPU faster, or adds the model you've been waiting for? You don't have to wait for it to merge.

1. In **Build / Update** (llama.cpp target), paste the PR number or its `github.com/ggml-org/llama.cpp/pull/...` link into **Try a llama.cpp pull request** and click **Look it up**. LlamaForge shows the PR's title, author and commit, and says so if it's already merged, closed, or a draft.
2. Click **Build PR #N**. LlamaForge fetches only that PR's commit into a temporary folder, builds `llama-server` and `llama-bench` with your **Build Flags**, and installs them beside your other builds. It needs the same git, CMake and compiler as building from source. The temporary sources are deleted afterwards.
3. Pick a model under the new build and click **Run it on this build**. That model now loads on the PR in its own process; the router and every other model stay on your build. **unpin** puts it back.
4. **Remove** deletes the PR build, after unloading and unpinning any model on it.

The commit is fixed when you look the PR up: if the author pushes again before the build reaches CMake, the build stops and asks you to look it up again, so you always run the code you saw. Only PRs on `ggml-org/llama.cpp` can be built. PR builds are never pruned by updates and never become the router's build.

### Is the PR actually faster?

Pick a model under the PR build and click **Compare speed**. LlamaForge unloads every model, runs `llama-bench` (pp512 and tg128, 3 runs each) on that model with your build, then with the PR build, and loads your models back. Both runs use the same model file and the same GPU settings from its models.ini section. The table shows each build's tokens per second with llama-bench's own spread, and the change in percent; a change smaller than the two spreads added together is marked *within noise*. **Copy as markdown** gives you the table to paste into the PR's thread. It refuses to start while a PR build, another comparison or Auto-Forge is running. The last result stays on the card after a restart.

## Screenshot

![Build tab](docs/img/build.png)

## Reference

| Concept | Source | Behavior |
|---|---|---|
| Current commit | `BuildManager.current_commit()` | `git log -1` (hash, subject, date) + current branch on `llama_src`. |
| Upstream check | `BuildManager.check_updates()` | `git fetch` + `git rev-list --count HEAD..<origin default branch>` (resolved per checkout via `resolve_remote_branch()`); cached 15 min (60s on failure), bypassed with `force=1`. |
| Recommended flags | `hardware.recommend()` | Returns `cmake_flags`, human-readable `notes`, a `runtime` suggestion, and the detected `gpus`/`cpu`. |
| macOS flags | `hardware.recommend()` | `GGML_METAL=ON`, `GGML_NATIVE=ON`. |
| NVIDIA GPU flags | `hardware.recommend()` | `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=<archs>` (if readable), `GGML_CUDA_FA_ALL_QUANTS=ON`. |
| CPU-only flags | `hardware.recommend()` | No CUDA flags; `GGML_NATIVE=ON` still set. |
| AVX-512 flags | `hardware.recommend()` | `GGML_AVX512`, `GGML_AVX512_VNNI`, `GGML_AVX512_VBMI`, `GGML_AVX512_BF16` — all `ON` when `avx512_hint` is true. |
| Runtime suggestion | `hardware.recommend()["runtime"]` | `fit=on`, `flash-attn=auto` on every platform. |
| Rebuild | `BuildManager.run_build()` | Validates paths, optional `git pull --ff-only`, backs up prior binaries, `cmake` configure + build (Release, parallel jobs = CPU count by default), records the built `server_bin`. |
| Partial success | `BuildManager.run_build()` | Build-step failure with a fresh `llama-server` present → `done_warnings` (amber "built with warnings"); no fresh binary → hard `failed`. |
| Engine target / switch | Build tab toggle, `POST /api/engine/switch` | Builds `llama.cpp` or `ik_llama`; switching sets `active_engine` and restarts the router, refused if the target binary has no router mode. |
| Try a PR | `backend/prtry.py`, `/api/prtry*` | Shallow-fetches `pull/N/head`, checks it is the looked-up SHA, builds `llama-server` + `llama-bench` with your flags into `engines/llama.cpp/pr-N-sha7` (channel `pr`); models pin to it like any build. `backend/prbench.py` runs the llama-bench A/B. |
| Prebuilt engine | Engine card, `backend/prebuilt.py`, `/api/engine/prebuilt*` | Installs an official ggml-org release (`nightly` or `stable` channel), SHA-256 checked and test-run first; keeps the newest 3 installs, **Use** rolls back. |

## Troubleshooting

If the upstream status shows "check failed," `git fetch` couldn't reach GitHub (network issue, or `llama_src` isn't a valid checkout) — check `llama_src` in `config.json` points at a real git clone. If a build fails, the Build Log panel shows the failing `cmake` step's output; prior binaries are always backed up first (to a `<binaries dir>-backup-<timestamp>` folder beside them, e.g. `bin/Release-backup-…`) so a bad build doesn't leave you without a working server. If flag detection looks wrong (e.g. no GPU found on a machine that has one), verify `nvidia-smi` is on `PATH` and reports the GPU — `hardware.detect_gpus()` shells out to it directly.

See also [Models & Tuning](models.md) for the flags the resulting `llama-server` binary exposes, and [config.json Reference](config.md) for `llama_src`, `build_dir`, `server_bin`, and `cmake_flags`.
