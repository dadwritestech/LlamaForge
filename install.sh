#!/bin/sh
# LlamaForge one-line installer for Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/install.sh | sh
#
# No git, no sudo, no build tools. It:
#   1. finds Python 3.10+ (the backend is pure stdlib - nothing to pip install)
#   2. downloads the latest LlamaForge release and lays it over the install
#      dir (backend/appinstall.py: your config, models and engines are kept)
#   3. adds a `llamaforge` command, plus an app-menu entry (Linux) or
#      ~/Applications/LlamaForge.app (macOS)
#   4. starts LlamaForge; the panel then offers the official llama.cpp build
#      for your machine in one click
# Re-run it (or `llamaforge update`) any time to update.
#
# Options (environment variables):
#   LLAMAFORGE_HOME        install dir (default ~/.local/share/llamaforge)
#   LLAMAFORGE_REF         release tag or branch (default: latest release)
#   LLAMAFORGE_ARCHIVE     install from a local .tar.gz/.zip instead of downloading
#   LLAMAFORGE_NO_LAUNCH   don't start LlamaForge at the end
#   LLAMAFORGE_NO_SHORTCUTS  skip the command, menu entry and app bundle
#   LLAMAFORGE_NO_STOP     don't stop a running copy before updating it
set -eu

REPO="dadwritestech/LlamaForge"
DEST="${LLAMAFORGE_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/llamaforge}"
BIN_DIR="$HOME/.local/bin"

say() { printf '  %s\n' "$*"; }
die() { printf '  error: %s\n' "$*" >&2; exit 1; }

fetch() {  # fetch URL FILE
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then wget -qO "$2" "$1"
  else die "need curl or wget"; fi
}

printf '\n  \033[33mLlamaForge installer\033[0m\n'
say "-> $DEST"
echo
mkdir -p "$DEST"
TMP="$(mktemp -d 2>/dev/null || mktemp -d -t lfinstall)"
trap 'rm -rf "$TMP"' EXIT INT TERM

# ---- 1. Python ----------------------------------------------------------
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
  p="$(command -v "$cand" 2>/dev/null || true)"
  [ -n "$p" ] || continue
  if "$p" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$p"; break
  fi
done
if [ -z "$PY" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    die "Python 3.10+ not found. Install it with 'brew install python@3.12' (or from python.org), then re-run."
  else
    die "Python 3.10+ not found. Install it with your package manager (e.g. 'sudo apt install python3'), then re-run."
  fi
fi
say "[1/4] Python: $PY"

# ---- 2. LlamaForge -------------------------------------------------------
if [ -n "${LLAMAFORGE_ARCHIVE:-}" ]; then
  ARCHIVE="$LLAMAFORGE_ARCHIVE"
  VERSION="${LLAMAFORGE_REF:-local}"
  say "[2/4] LlamaForge: from $ARCHIVE"
else
  REF="${LLAMAFORGE_REF:-}"
  if [ -z "$REF" ]; then
    fetch "https://api.github.com/repos/$REPO/releases/latest" "$TMP/latest.json" 2>/dev/null || true
    REF="$("$PY" -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["tag_name"])
except Exception: print("master")' "$TMP/latest.json")"
  fi
  case "$REF" in v[0-9]*) KIND=tags ;; *) KIND=heads ;; esac
  VERSION="$REF"
  ARCHIVE="$TMP/llamaforge.tar.gz"
  say "[2/4] LlamaForge: downloading $REF"
  fetch "https://github.com/$REPO/archive/refs/$KIND/$REF.tar.gz" "$ARCHIVE"
fi
mkdir -p "$TMP/src"
case "$ARCHIVE" in
  *.zip) "$PY" -m zipfile -e "$ARCHIVE" "$TMP/src" ;;
  *)     "$PY" -m tarfile -e "$ARCHIVE" "$TMP/src" ;;
esac
if [ -f "$DEST/stop.sh" ] && [ -f "$DEST/config.json" ] && [ -z "${LLAMAFORGE_NO_STOP:-}" ]; then
  say "      stopping the running copy to update it"
  bash "$DEST/stop.sh" >/dev/null 2>&1 || true
fi
INSTALLER="$(find "$TMP/src" -path '*/backend/appinstall.py' | head -n 1)"
[ -n "$INSTALLER" ] || die "that archive is not a LlamaForge release"
"$PY" "$INSTALLER" --from "$TMP/src" --to "$DEST" --version "$VERSION"
chmod +x "$DEST"/*.sh 2>/dev/null || true
printf '%s\n' "$PY" > "$DEST/.lf-python"     # run.sh/stop.sh use this Python

# ---- 3. launchers --------------------------------------------------------
if [ -z "${LLAMAFORGE_NO_SHORTCUTS:-}" ]; then
  say "[3/4] llamaforge command + app launcher"
  mkdir -p "$BIN_DIR"
  cat > "$BIN_DIR/llamaforge" <<EOF
#!/bin/sh
# LlamaForge launcher (written by install.sh)
case "\${1:-}" in
  stop)      exec bash "$DEST/stop.sh" ;;
  update)    curl -fsSL https://raw.githubusercontent.com/$REPO/master/install.sh | sh ;;
  uninstall) exec sh "$DEST/uninstall.sh" ;;
  ""|start)  exec bash "$DEST/run.sh" ;;
  *)         echo "usage: llamaforge [start|stop|update|uninstall]"; exit 2 ;;
esac
EOF
  chmod +x "$BIN_DIR/llamaforge"
  if [ "$(uname -s)" = "Darwin" ]; then
    APP="$HOME/Applications/LlamaForge.app/Contents"
    mkdir -p "$APP/MacOS" "$APP/Resources"
    cp "$DEST/web/icons/llamaforge.icns" "$APP/Resources/llamaforge.icns"
    printf '#!/bin/bash\nexec bash "%s/run.sh"\n' "$DEST" > "$APP/MacOS/LlamaForge"
    chmod +x "$APP/MacOS/LlamaForge"
    cat > "$APP/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>LlamaForge</string>
  <key>CFBundleIdentifier</key><string>io.github.dadwritestech.llamaforge</string>
  <key>CFBundleExecutable</key><string>LlamaForge</string>
  <key>CFBundleIconFile</key><string>llamaforge</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION#v}</string>
  <key>LSUIElement</key><true/>
</dict></plist>
EOF
  else
    APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$APPS"
    cat > "$APPS/llamaforge.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=LlamaForge
Comment=Local LLMs on llama.cpp
Exec=bash "$DEST/run.sh"
Icon=$DEST/web/icons/llamaforge-512.png
Terminal=false
Categories=Development;Utility;
EOF
  fi
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) say "      note: add $BIN_DIR to your PATH to use the 'llamaforge' command" ;;
  esac
else
  say "[3/4] launchers skipped"
fi

# ---- 4. launch -----------------------------------------------------------
echo
printf '  \033[32mLlamaForge %s is installed.\033[0m\n' "$VERSION"
if [ -z "${LLAMAFORGE_NO_LAUNCH:-}" ]; then
  say "[4/4] Starting it - your browser will open the panel."
  bash "$DEST/run.sh" || say "start it later with: llamaforge"
else
  say "[4/4] Start it any time with: llamaforge"
fi
