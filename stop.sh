#!/usr/bin/env bash
# LlamaForge one-click shutdown (Linux / macOS). Mirror of run.sh.
# Stops the llama.cpp router, every model instance it spawned, and the
# LlamaForge dashboard backend. Safe to run repeatedly.
here="$(cd "$(dirname "$0")" && pwd)"
cfg="$here/config.json"

getcfg() { python3 -c "import json;print(json.load(open('$cfg')).get('$1',''))"; }

kill_port() {
  local port="$1" label="$2" pids
  pids="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null)"
  for pid in $pids; do
    kill "$pid" 2>/dev/null && echo "stopped $label (pid $pid on port $port)"
  done
}

# 1. dashboard backend
panel_port="$(getcfg panel_port)"
kill_port "$panel_port" "LlamaForge dashboard"

# 2. router
router_port="$(getcfg router_port)"
kill_port "$router_port" "llama.cpp router"

# 3. sweep any llama-server model instances the router spawned on random ports
if pkill -x llama-server 2>/dev/null; then
  echo "stopped model instance(s)"
fi

echo "LlamaForge stopped."
