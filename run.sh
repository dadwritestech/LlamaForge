#!/usr/bin/env bash
# LlamaForge one-click runner (Linux / macOS).
# Reads config.json, starts the llama.cpp router + the LlamaForge backend,
# then opens the dashboard in your browser. Safe to run repeatedly.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
cfg="$here/config.json"

getcfg() { python3 -c "import json;print(json.load(open('$cfg')).get('$1',''))"; }

listening() { lsof -ti "tcp:$1" -sTCP:LISTEN >/dev/null 2>&1; }

router_port="$(getcfg router_port)"
panel_port="$(getcfg panel_port)"
server_bin="$(getcfg server_bin)"
models_ini="$(getcfg models_ini)"
router_host="$(getcfg router_host)"; [ -n "$router_host" ] || router_host=127.0.0.1
api_key="$(getcfg router_api_key)"

logdir="$here/logs"
mkdir -p "$logdir"

# 1. llama.cpp router (only if not already up)
if ! listening "$router_port"; then
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
fi

# 2. LlamaForge backend (dashboard)
if ! listening "$panel_port"; then
  (cd "$here/backend" && nohup python3 server.py \
    >>"$logdir/panel.out.log" 2>>"$logdir/panel.err.log" </dev/null &)
  echo "started LlamaForge dashboard on port $panel_port"
fi

# 3. open the dashboard
sleep 2
url="http://127.0.0.1:$panel_port/"
if [ "$(uname)" = "Darwin" ]; then open "$url"; else xdg-open "$url" >/dev/null 2>&1 || echo "open $url"; fi
