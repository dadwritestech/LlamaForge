#!/bin/sh
# LlamaForge uninstaller (Linux / macOS):  llamaforge uninstall
# Removes the app and the llama.cpp builds it downloaded. Your settings and
# downloaded models stay unless you say otherwise.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
cd /

echo "Uninstalling LlamaForge from $here"
if [ -f "$here/config.json" ] && [ -z "${LLAMAFORGE_NO_STOP:-}" ]; then
  bash "$here/stop.sh" >/dev/null 2>&1 || true
fi
rm -f "$HOME/.local/bin/llamaforge" \
      "${XDG_DATA_HOME:-$HOME/.local/share}/applications/llamaforge.desktop"
rm -rf "$HOME/Applications/LlamaForge.app"

printf 'Also delete your settings and the models LlamaForge downloaded into %s? (y/N) ' "$here"
read -r all || all=""
if [ "$all" = "y" ]; then
  rm -rf "$here"
else
  py="$(cat "$here/.lf-python" 2>/dev/null || echo python3)"
  # Only files the installer put there (its manifest), plus what it downloaded.
  "$py" - "$here" <<'EOF' || true
import json, os, sys
root = sys.argv[1]
try:
    files = json.load(open(os.path.join(root, ".lf-files.json")))["files"]
except Exception:
    files = []
for rel in files:
    parts = rel.split("/")
    if rel.startswith("/") or ".." in parts or "\\" in rel:
        continue
    try:
        os.remove(os.path.join(root, *parts))
    except OSError:
        pass
import shutil
for dirpath, dirnames, filenames in sorted(os.walk(root), key=lambda w: -len(w[0])):
    if os.path.basename(dirpath) == "__pycache__":
        shutil.rmtree(dirpath, ignore_errors=True)
    elif dirpath != root and not os.listdir(dirpath):
        os.rmdir(dirpath)
EOF
  rm -rf "$here/engines" "$here/logs" "$here/.lf-files.json" "$here/.lf-python" "$here/stats.json"
  echo "Kept your settings and models in $here"
fi
echo "LlamaForge is uninstalled."
