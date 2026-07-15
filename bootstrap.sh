#!/usr/bin/env bash
# LlamaForge first-run bootstrap (Linux / macOS).
# Gets the machine to the point where the GUI can take over: checks Python +
# Git, fetches llama.cpp if absent, writes config.json, then launches the app.
# Heavy steps (compiler/CUDA install, the actual build) are done from the
# dashboard's Setup and Build tabs. Never runs sudo.
set -e
here="$(cd "$(dirname "$0")" && pwd)"

have() { command -v "$1" >/dev/null 2>&1; }

echo "=== LlamaForge bootstrap ==="

# --- Python (required to run the GUI backend) ---
if ! have python3; then
  echo "python3 not found."
  if [ "$(uname)" = "Darwin" ]; then
    echo "Install it with:  brew install python@3.12   (or https://www.python.org/downloads/)"
  else
    echo "Install it with your package manager, e.g.:  sudo apt-get install -y python3"
  fi
  exit 1
fi

# --- Git (required to fetch/update llama.cpp) ---
if ! have git; then
  echo "git not found - recommended for building/updating llama.cpp."
  if [ "$(uname)" = "Darwin" ]; then
    echo "Install it with:  xcode-select --install   or   brew install git"
  else
    echo "Install it with your package manager, e.g.:  sudo apt-get install -y git"
  fi
fi

# --- config.json ---
cfg="$here/config.json"
if [ -f "$cfg" ]; then
  echo "Using existing config.json"
else
  src="$here/llama.cpp"
  python3 - "$cfg" "$src" "$here" <<'PY'
import json, os, sys
cfg_path, src, here = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = {
    "llama_src":   src,
    "build_dir":   os.path.join(src, "build"),
    "server_bin":  os.path.join(src, "build", "bin", "llama-server"),
    "models_ini":  os.path.join(here, "models.ini"),
    "model_dirs":  [],
    "router_port": 8080,
    "panel_port":  8090,
    "router_host": "127.0.0.1",
    "router_api_key": "",
    "cmake_flags": {},
    "git_remote":  "https://github.com/ggml-org/llama.cpp",
}
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
PY
  echo "Wrote config.json (edit paths there if your models live elsewhere)."
fi

# --- fetch llama.cpp source if missing ---
src="$(python3 -c "import json;print(json.load(open('$cfg'))['llama_src'])")"
if [ ! -d "$src/.git" ]; then
  if have git; then
    read -r -p "llama.cpp source not found at $src. Clone it now? (y/n) " ans
    if [ "$ans" = "y" ]; then
      git clone --depth 1 "$(python3 -c "import json;print(json.load(open('$cfg'))['git_remote'])")" "$src"
    else
      echo "Skipped. Clone later or point config.json 'llama_src' at an existing checkout."
    fi
  fi
fi

# --- ensure a models.ini with a global section exists ---
ini="$(python3 -c "import json;print(json.load(open('$cfg'))['models_ini'])")"
if [ ! -f "$ini" ]; then
  cat > "$ini" <<'INI'
version = 1

[*]
ctx-size = 8192
flash-attn = on
jinja = true
n-gpu-layers = 99
load-on-startup = false
INI
  echo "Created a starter models.ini"
fi

echo ""
echo "Bootstrap done. Launching dashboard..."
echo "In the app: Setup tab -> check prereqs and scan for models;"
echo "            Build tab -> build/update llama.cpp; Models tab -> tune + load."
exec "$here/run.sh"
