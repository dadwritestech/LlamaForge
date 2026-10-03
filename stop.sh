#!/usr/bin/env bash
# LlamaForge one-click shutdown (Linux / macOS). Mirror of run.sh.
# Stops the router this copy started, its model instances, and the dashboard -
# nothing else on the machine. The rules live in backend/procs.py.
here="$(cd "$(dirname "$0")" && pwd)"

# The one-line installer records which Python it found (python3 may be too old on macOS).
PY="${LLAMAFORGE_PYTHON:-$(cat "$here/.lf-python" 2>/dev/null || echo python3)}"
exec "$PY" "$here/backend/procs.py" --stop "$here"
