#!/usr/bin/env bash
# LlamaForge one-click runner (Linux / macOS).
# Reads config.json, starts the llama.cpp router + the LlamaForge backend,
# then opens the dashboard in your browser. Safe to run repeatedly.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
cfg="$here/config.json"

# config.json is per-machine and deliberately not in the repo. Without this the
# first run died on a raw Python traceback from getcfg that said nothing about
# config.example.json sitting right next to it.
if [ ! -f "$cfg" ]; then
  if [ ! -f "$here/config.example.json" ]; then
    echo "config.json is missing and config.example.json was not found in $here." >&2
    echo "Re-clone the repo, or create config.json by hand." >&2
    exit 1
  fi
  cp "$here/config.example.json" "$cfg"
  echo "config.json not found - created one from config.example.json."
  echo "Set your llama.cpp paths and model folders in the dashboard's Setup tab."
fi

getcfg() { python3 -c "import json;print(json.load(open('$cfg')).get('$1',''))"; }

listening() { lsof -ti "tcp:$1" -sTCP:LISTEN >/dev/null 2>&1; }

router_port="$(getcfg router_port)"
panel_port="$(getcfg panel_port)"
server_bin="$(getcfg server_bin)"
models_ini="$(getcfg models_ini)"
router_host="$(getcfg router_host)"; [ -n "$router_host" ] || router_host=127.0.0.1
api_key="$(getcfg router_api_key)"

# Mirror config._abs(): the router inherits this shell's CWD, and
# config.example.json ships "./models.ini", so a relative value resolved against
# wherever the user ran this from - the router read an empty registry and loaded
# 0 models.
case "$models_ini" in
  /*|"") ;;
  *) models_ini="$here/${models_ini#./}" ;;
esac

logdir="$here/logs"
mkdir -p "$logdir"

# Mirror config.ensure_models_ini(): llama-server refuses to start without this
# file, and the router is launched here, before the backend can make one.
if [ -n "$models_ini" ] && [ ! -f "$models_ini" ]; then
  mkdir -p "$(dirname "$models_ini")"
  cat >"$models_ini" <<'EOF'
; LlamaForge model registry - read by llama-server's router.
; Sections are model ids; keys are llama-server flags.
version = 1

[*]
ctx-size = 150000
EOF
  echo "created $models_ini"
fi

# 1. llama.cpp router (only if not already up)
if ! listening "$router_port"; then
  if python3 "$here/backend/network_policy.py" --preflight "$cfg"; then
    if [ -x "$server_bin" ]; then
      args=(--models-preset "$models_ini" --models-max 1 --offline
            --host "$router_host" --port "$router_port" --metrics)
      [ -n "$api_key" ] && args+=(--api-key "$api_key")
      nohup "$server_bin" "${args[@]}" \
        >>"$logdir/router.out.log" 2>>"$logdir/router.err.log" </dev/null &
      echo "started llama.cpp router on $router_host:$router_port"
    else
      echo "server_bin not found ($server_bin) - open the dashboard Build tab to build llama.cpp first."
    fi
  else
    echo "Router not started: repair Network Access in the dashboard." >&2
  fi
else
  # Something already holds the router port. If it isn't a llama-server, the
  # dashboard would come up with every model "offline" and no stated reason.
  owner="$(lsof -ti "tcp:$router_port" -sTCP:LISTEN 2>/dev/null | head -1)"
  owner_name="$(ps -p "${owner:-0}" -o comm= 2>/dev/null || true)"
  case "$owner_name" in
    *llama*|"") ;;
    *) echo "port $router_port is already in use by '$owner_name' (PID $owner)."
       echo "The router was not started. Stop that process, or change router_port in the Setup tab." ;;
  esac
fi

# 2. LlamaForge backend (dashboard)
if ! listening "$panel_port"; then
  (cd "$here/backend" && nohup python3 server.py \
    >>"$logdir/panel.out.log" 2>>"$logdir/panel.err.log" </dev/null &)
  echo "started LlamaForge dashboard on port $panel_port"
fi

# 3. open the dashboard
if [ -z "${LLAMAFORGE_NO_BROWSER:-}" ]; then
  sleep 2
  url="http://127.0.0.1:$panel_port/"
  if [ "$(uname)" = "Darwin" ]; then
    open "$url"
  else
    xdg-open "$url" >/dev/null 2>&1 || echo "open $url"
  fi
fi
