#!/bin/sh
# LlamaForge uninstaller (Linux / macOS):  llamaforge uninstall
# Removes the app and the llama.cpp builds it downloaded. Your settings and
# downloaded models stay unless you say otherwise.
# Only a folder the installer made (it has .lf-files.json) is touched; what
# gets deleted is decided by backend/appinstall.py, never a blind rm -rf.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
cd /

if [ ! -f "$here/.lf-files.json" ]; then
  echo "$here was not made by the LlamaForge installer (no .lf-files.json) - nothing removed." >&2
  echo "For a git checkout, stop it with stop.sh and delete the folder yourself." >&2
  exit 1
fi
py="$(cat "$here/.lf-python" 2>/dev/null || echo python3)"

echo "Uninstalling LlamaForge from $here"
if [ -f "$here/config.json" ] && [ -z "${LLAMAFORGE_NO_STOP:-}" ]; then
  bash "$here/stop.sh" >/dev/null 2>&1 || true
fi

printf 'Also delete your settings and the models LlamaForge downloaded into %s? (y/N) ' "$here"
read -r all || all=""
if [ "$all" = "y" ]; then
  "$py" "$here/backend/appinstall.py" --uninstall "$here" --all || exit 1
else
  "$py" "$here/backend/appinstall.py" --uninstall "$here" || exit 1
  echo "Kept your settings and models in $here"
fi
rm -f "$HOME/.local/bin/llamaforge" \
      "${XDG_DATA_HOME:-$HOME/.local/share}/applications/llamaforge.desktop"
rm -rf "$HOME/Applications/LlamaForge.app"
echo "LlamaForge is uninstalled."
