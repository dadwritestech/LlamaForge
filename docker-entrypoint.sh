#!/usr/bin/env bash
# Reimplements bootstrap.sh + run.sh for a headless, single-container deploy:
# seeds config.json from env vars on first boot only, fetches/builds
# llama.cpp if missing, starts the router in the background, then execs the
# dashboard in the foreground so the container has a real PID 1.
set -e

APP_DIR="${APP_DIR:-/opt/llamaforge}"
DATA_DIR="${DATA_DIR:-/data}"

ROUTER_PORT="${ROUTER_PORT:-8080}"
PANEL_PORT="${PANEL_PORT:-8090}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_API_KEY="${ROUTER_API_KEY:-}"
MODEL_DIRS="${MODEL_DIRS:-/data/models}"
AUTO_LOAD_MODEL="${AUTO_LOAD_MODEL:-}"
UI_MODE="${UI_MODE:-advanced}"
THEME="${THEME:-}"
CVD="${CVD:-false}"
ANTHROPIC_SHIM_ENABLED="${ANTHROPIC_SHIM_ENABLED:-false}"
ANTHROPIC_DEFAULT_MODEL="${ANTHROPIC_DEFAULT_MODEL:-}"
GIT_REMOTE_LLAMACPP="${GIT_REMOTE_LLAMACPP:-https://github.com/ggml-org/llama.cpp}"
AUTO_BUILD_LLAMACPP="${AUTO_BUILD_LLAMACPP:-true}"
ENABLE_CUDA="${ENABLE_CUDA:-true}"

mkdir -p "$DATA_DIR/logs" "$DATA_DIR/models" "$DATA_DIR/wiki"

CFG="$DATA_DIR/config.json"
LLAMA_SRC="$DATA_DIR/llama.cpp"
BUILD_DIR="$LLAMA_SRC/build"
SERVER_BIN="$BUILD_DIR/bin/llama-server"
MODELS_INI="$DATA_DIR/models.ini"
WIKI_DIR="$DATA_DIR/wiki"

# --- 1. Seed config.json on first run only. After that it's dashboard-owned,
#        same as running the app natively - env vars won't fight you over
#        settings you changed in the UI on later restarts. ---
if [ ! -f "$CFG" ]; then
  echo "[entrypoint] no config.json in $DATA_DIR yet - creating one from env vars"
  MODEL_DIRS="$MODEL_DIRS" ROUTER_PORT="$ROUTER_PORT" PANEL_PORT="$PANEL_PORT" \
  ROUTER_HOST="$ROUTER_HOST" ROUTER_API_KEY="$ROUTER_API_KEY" \
  GIT_REMOTE_LLAMACPP="$GIT_REMOTE_LLAMACPP" AUTO_LOAD_MODEL="$AUTO_LOAD_MODEL" \
  UI_MODE="$UI_MODE" THEME="$THEME" CVD="$CVD" \
  ANTHROPIC_SHIM_ENABLED="$ANTHROPIC_SHIM_ENABLED" \
  ANTHROPIC_DEFAULT_MODEL="$ANTHROPIC_DEFAULT_MODEL" \
  python3 - "$CFG" "$LLAMA_SRC" "$BUILD_DIR" "$SERVER_BIN" "$MODELS_INI" "$WIKI_DIR" <<'PY'
import json, os, sys
cfg_path, llama_src, build_dir, server_bin, models_ini, wiki_dir = sys.argv[1:7]
model_dirs = [d.strip() for d in os.environ.get("MODEL_DIRS", "").split(",") if d.strip()]
cfg = {
    "llama_src": llama_src,
    "build_dir": build_dir,
    "server_bin": server_bin,
    "models_ini": models_ini,
    "model_dirs": model_dirs,
    "router_port": int(os.environ.get("ROUTER_PORT", 8080)),
    "panel_port": int(os.environ.get("PANEL_PORT", 8090)),
    "router_host": os.environ.get("ROUTER_HOST", "0.0.0.0"),
    "router_api_key": os.environ.get("ROUTER_API_KEY", ""),
    "cmake_flags": {},
    "git_remote": os.environ.get("GIT_REMOTE_LLAMACPP", "https://github.com/ggml-org/llama.cpp"),
    "auto_load_model": os.environ.get("AUTO_LOAD_MODEL", ""),
    "presets": {},
    "ui_mode": os.environ.get("UI_MODE", "advanced"),
    "theme": os.environ.get("THEME", ""),
    "cvd": os.environ.get("CVD", "false").lower() == "true",
    "anthropic_shim_enabled": os.environ.get("ANTHROPIC_SHIM_ENABLED", "false").lower() == "true",
    "anthropic_default_model": os.environ.get("ANTHROPIC_DEFAULT_MODEL", ""),
    "wiki_dir": wiki_dir,
    "wiki_profiles": {},
    "wiki_active": {},
}
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print("wrote", cfg_path)
PY
else
  echo "[entrypoint] using existing config.json from $DATA_DIR (dashboard-managed - env vars ignored)"
fi

# run.sh/bootstrap.sh hardcode config.json and logs/ next to the script itself;
# symlink those into the persistent volume instead of fighting that assumption.
ln -sfn "$CFG" "$APP_DIR/config.json"
ln -sfn "$DATA_DIR/logs" "$APP_DIR/logs"

getcfg() { python3 -c "import json;print(json.load(open('$CFG')).get('$1',''))"; }

# --- 2. fetch llama.cpp source if missing ---
if [ ! -d "$LLAMA_SRC/.git" ]; then
  echo "[entrypoint] cloning llama.cpp into $LLAMA_SRC"
  git clone --depth 1 "$(getcfg git_remote)" "$LLAMA_SRC"
fi

# --- 3. build llama-server if missing (the dashboard's Build tab can also do
#        this later - AUTO_BUILD_LLAMACPP=false to leave it to the UI) ---
if [ ! -x "$SERVER_BIN" ] && [ "$AUTO_BUILD_LLAMACPP" = "true" ]; then
  echo "[entrypoint] llama-server not found - building now (first boot can take several minutes)"
  cmake_args=(-S "$LLAMA_SRC" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)
  if [ "$ENABLE_CUDA" = "true" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "[entrypoint] GPU visible - building with CUDA support"
    cmake_args+=(-DGGML_CUDA=ON)
  else
    echo "[entrypoint] building CPU-only (no GPU visible or ENABLE_CUDA=false)"
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"
elif [ ! -x "$SERVER_BIN" ]; then
  echo "[entrypoint] llama-server not built and AUTO_BUILD_LLAMACPP=false - build it from the dashboard's Build tab"
fi

# --- 4. ensure a models.ini exists (llama-server refuses to start without one) ---
if [ ! -f "$MODELS_INI" ]; then
  cat > "$MODELS_INI" <<'INI'
version = 1

[*]
ctx-size = 150000
flash-attn = on
jinja = true
n-gpu-layers = 99
load-on-startup = false
INI
  echo "[entrypoint] created starter $MODELS_INI"
fi

# --- 5. start the llama.cpp router in the background ---
if [ -x "$SERVER_BIN" ]; then
  args=(--models-preset "$MODELS_INI" --models-max 1 --offline
        --host "$(getcfg router_host)" --port "$(getcfg router_port)" --metrics)
  api_key="$(getcfg router_api_key)"
  [ -n "$api_key" ] && args+=(--api-key "$api_key")
  echo "[entrypoint] starting llama.cpp router on $(getcfg router_host):$(getcfg router_port)"
  nohup "$SERVER_BIN" "${args[@]}" \
    >>"$DATA_DIR/logs/router.out.log" 2>>"$DATA_DIR/logs/router.err.log" </dev/null &
else
  echo "[entrypoint] WARNING: server_bin still missing - router was not started"
fi

# --- 6. run the dashboard in the foreground so the container stays alive ---
echo "[entrypoint] starting LlamaForge dashboard on port $(getcfg panel_port)"
cd "$APP_DIR/backend"
exec python3 server.py
