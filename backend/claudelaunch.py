"""Open Claude Code in a folder, pointed at a local model (pure stdlib).

Claude Code talks to the panel's Anthropic shim (/v1/messages). Unlike
agentsetup.apply(), which merges the endpoint into ~/.claude/settings.json and so
redirects every Claude Code session on the machine, this sets it on one process:
the terminal window LlamaForge opens. The user's own Claude Code config is never
read or written.

Everything here is a pure function over its inputs (which/run/isfile are
injected) except spawn(), which writes the launch script and starts the window.
"""
import os
import re
import shlex
import shutil
import subprocess
import tempfile

import osplat

CREATE_NEW_CONSOLE = 0x00000010
_NO_WINDOW = 0x08000000
PICK_TIMEOUT = 600
CTX_MIN = 32768          # below this Claude Code's system prompt + tools barely fit
CTX_GOOD = 65536
RECENT_CAP = 8

INSTALL_HINT = {
    "win": "irm https://claude.ai/install.ps1 | iex",
    "mac": "curl -fsSL https://claude.ai/install.sh | bash",
    "linux": "curl -fsSL https://claude.ai/install.sh | bash",
}

# Inherited settings that would send the session to the cloud, or add to what we
# send: a real ANTHROPIC_API_KEY must never reach the local endpoint.
_STRIP_EXACT = ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")

# (binary, flag before the command; "" = none)
_LINUX_TERMINALS = (("x-terminal-emulator", "-e"), ("gnome-terminal", "--"),
                    ("konsole", "-e"), ("xfce4-terminal", "-x"), ("kitty", ""),
                    ("alacritty", "-e"), ("xterm", "-e"))

TITLE = "Folder for Claude Code"


def os_name():
    return "win" if osplat.IS_WIN else "mac" if osplat.IS_MAC else "linux"


def _run(argv, timeout=15):
    """(returncode, stdout); never raises."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
                           creationflags=_NO_WINDOW if osplat.IS_WIN else 0)
        return r.returncode, r.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return -1, ""


# ---------------- finding Claude Code ----------------

def locate(which=shutil.which, home=None, is_win=osplat.IS_WIN, isfile=os.path.isfile,
           env=os.environ):
    """Path to the claude CLI, or "". PATH first, then where the installers put it."""
    hit = which("claude")
    if hit:
        return hit
    home = home or os.path.expanduser("~")
    if is_win:
        cands = [os.path.join(home, ".local", "bin", "claude.exe"),
                 os.path.join(home, ".claude", "local", "claude.cmd")]
        if env.get("APPDATA"):
            cands.append(os.path.join(env["APPDATA"], "npm", "claude.cmd"))
    else:
        cands = [os.path.join(home, ".local", "bin", "claude"),
                 os.path.join(home, ".claude", "local", "claude")]
    hit = next((c for c in cands if isfile(c)), "")
    if not hit and is_win and env.get("APPDATA"):
        hit = _desktop_bundle(os.path.join(env["APPDATA"], "Claude", "claude-code"), isfile)
    return hit


def _desktop_bundle(root, isfile):
    """The newest CLI the Claude desktop app keeps for itself:
    <root>/<version>/<hash>/claude.exe. Last resort: it's the app's own copy."""
    def key(v):
        return tuple(int(n) for n in v.split(".")) if re.fullmatch(r"\d+(\.\d+)*", v) else None
    try:
        versions = sorted((v for v in os.listdir(root) if key(v)), key=key, reverse=True)
    except OSError:
        return ""
    for v in versions:
        try:
            subs = sorted(os.listdir(os.path.join(root, v)))
        except OSError:
            continue
        for h in subs:
            exe = os.path.join(root, v, h, "claude.exe")
            if isfile(exe):
                return exe
    return ""


def version(path, run=_run):
    if not path:
        return ""
    rc, out = run([path, "--version"])
    m = re.search(r"\d+\.\d+\.\d+", out or "") if rc == 0 else None
    return m.group(0) if m else ""


# ---------------- the session's environment ----------------

def env_changes(base, endpoint, token, model, small):
    """(variables to set, inherited variable names to remove)."""
    small = small or model
    sets = {
        "ANTHROPIC_BASE_URL": endpoint,
        "ANTHROPIC_AUTH_TOKEN": token or "llamaforge",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": small,
        "ANTHROPIC_SMALL_FAST_MODEL": small,          # older Claude Code versions
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "API_TIMEOUT_MS": "1800000",                  # a long local prefill isn't a hang
    }
    unsets = sorted(k for k in base
                    if k not in sets and (k.startswith("ANTHROPIC_") or k in _STRIP_EXACT))
    return sets, unsets


def session_env(base, endpoint, token, model, small):
    sets, unsets = env_changes(base, endpoint, token, model, small)
    env = {k: v for k, v in base.items() if k not in unsets}
    env.update(sets)
    return env


# ---------------- readiness ----------------

def checks(cli, shim_enabled, loaded, n_ctx, settings, model, small):
    """([{id, level, text}], ready). ready = no "error" item."""
    items = []

    def add(cid, level, text):
        items.append({"id": cid, "level": level, "text": text})

    add("cli", "ok" if cli else "error",
        f"Claude Code found: {cli}" if cli else
        "Claude Code isn't installed (or isn't where LlamaForge looks). Install it, then check again.")
    add("shim", "ok" if shim_enabled else "error",
        "The panel's Anthropic endpoint (/v1/messages) is on." if shim_enabled else
        "The panel's Anthropic endpoint is turned off (anthropic_shim_enabled in config.json).")
    add("loaded", "ok" if loaded else "warn",
        f"{model} is loaded." if loaded else
        f"{model} isn't loaded; it will load on Claude Code's first message.")
    if not n_ctx:
        add("ctx", "warn", "Couldn't read the model's context size. Claude Code needs 64K or more.")
    elif n_ctx < CTX_MIN:
        add("ctx", "error", f"Context is {n_ctx:,} tokens. Claude Code's system prompt and tools "
                            f"alone fill most of that: set ctx-size to 65536 or more.")
    elif n_ctx < CTX_GOOD:
        add("ctx", "warn", f"Context is {n_ctx:,} tokens. That works for short tasks; 65536 or "
                           f"more is recommended.")
    else:
        add("ctx", "ok", f"Context is {n_ctx:,} tokens.")
    if str((settings or {}).get("no-jinja", "")).lower() in ("1", "true", "on", "yes"):
        add("jinja", "error", "no-jinja is set. Claude Code needs tool calls, which need the jinja "
                              "chat template: remove no-jinja.")
    else:
        add("jinja", "ok", "The jinja chat template is on (needed for tool calls).")
    add("small", "ok",
        f"Background requests (titles, summaries) go to {small}, so they don't push the main "
        f"conversation out of {model}'s prompt cache." if small and small != model else
        f"Background requests also go to {model}. Load a small worker beside it to keep them "
        f"off the main model.")
    return items, not any(i["level"] == "error" for i in items)


def choose_small(rows, main):
    """The smallest other loaded model (a multi-model worker), else the main one."""
    others = [r for r in rows or () if r.get("status") == "loaded" and r.get("id") != main]
    if not others:
        return main
    return min(others, key=lambda r: r.get("file_gib") or 0)["id"]


# ---------------- launching ----------------

def _ps(s):
    return "'" + str(s).replace("'", "''") + "'"


def _manual_command(name, claude, folder, sets, unsets):
    if name == "win":
        parts = [f"Remove-Item Env:{k} -ErrorAction SilentlyContinue" for k in unsets]
        parts += [f"$env:{k}={_ps(v)}" for k, v in sets.items()]
        parts += [f"Set-Location -LiteralPath {_ps(folder)}", f"& {_ps(claude)}"]
        return "; ".join(parts)
    steps = [f"cd {shlex.quote(folder)}"]
    if unsets:
        steps.append("unset " + " ".join(unsets))
    steps.append("export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in sets.items()))
    steps.append(f"exec {shlex.quote(claude)}")
    return "(" + " && ".join(steps) + ")"


def _script(claude, folder, sets, unsets):
    lines = ["#!/bin/sh", 'rm -f -- "$0"']           # holds the panel key: gone once read
    if unsets:
        lines.append("unset " + " ".join(unsets))
    lines += [f"export {k}={shlex.quote(v)}" for k, v in sets.items()]
    lines += [f"cd {shlex.quote(folder)} || exit 1", shlex.quote(claude),
              "echo", "printf 'Claude Code exited. Press Enter to close. '", "read _", ""]
    return "\n".join(lines)


def launch_plan(name, claude, folder, env, sets, unsets, script_dir, which=shutil.which,
                comspec=None):
    """How to open the window. kind "spawn" (argv/cwd/env/creationflags, plus a
    script to write first on macOS/Linux) or "manual" (no terminal found).
    command is always a line the user can paste themselves."""
    plan = {"kind": "spawn", "argv": None, "cwd": folder, "env": env, "creationflags": 0,
            "script": None, "script_path": None,
            "command": _manual_command(name, claude, folder, sets, unsets)}
    if name == "win":
        # The env goes straight through CreateProcess: no file holds the key, and a
        # default terminal (Windows Terminal) hosting the console keeps it intact.
        # cmd /k keeps the window open if claude exits with an error.
        plan.update(argv=[comspec or os.environ.get("COMSPEC", "cmd.exe"), "/k", claude],
                    creationflags=CREATE_NEW_CONSOLE)
        return plan
    plan["script"] = _script(claude, folder, sets, unsets)
    if name == "mac":
        plan["script_path"] = os.path.join(script_dir, "claude-code.command")
        plan["argv"] = ["open", "-a", "Terminal", plan["script_path"]]
        return plan
    plan["script_path"] = os.path.join(script_dir, "claude-code.sh")
    for binary, flag in _LINUX_TERMINALS:
        exe = which(binary)
        if exe:
            plan["argv"] = [exe] + ([flag] if flag else []) + [plan["script_path"]]
            return plan
    plan.update(kind="manual", script=None, script_path=None)
    return plan


def spawn(plan, popen=subprocess.Popen):
    """Start the window. Raises OSError on failure."""
    if plan["script"]:
        with open(plan["script_path"], "w", encoding="utf-8", newline="\n") as f:
            f.write(plan["script"])
        os.chmod(plan["script_path"], 0o700)
    popen(plan["argv"], cwd=plan["cwd"], env=plan["env"], creationflags=plan["creationflags"],
          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
          close_fds=True)


def launch(claude, folder, base_env, endpoint, token, model, small, name=None,
           which=shutil.which, popen=subprocess.Popen):
    """Plan and start the session. Returns the plan (kind "manual" = not started)."""
    name = name or os_name()
    sets, unsets = env_changes(base_env, endpoint, token, model, small)
    env = session_env(base_env, endpoint, token, model, small)
    script_dir = tempfile.mkdtemp(prefix="lf-claude-") if name != "win" else ""
    plan = launch_plan(name, claude, folder, env, sets, unsets, script_dir, which)
    if plan["kind"] == "spawn":
        spawn(plan, popen)
    elif script_dir:
        os.rmdir(script_dir)
    return plan


# ---------------- folder picker ----------------

_PS_PICK = (
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
    "Add-Type -AssemblyName System.Windows.Forms;"
    # The panel is a background process, so Windows won't give this window the
    # foreground. An invisible TopMost owner that is actually shown puts the dialog
    # above the browser anyway (z-order isn't subject to the foreground lock).
    "$o=New-Object System.Windows.Forms.Form -Property @{TopMost=$true;ShowInTaskbar=$false;"
    "FormBorderStyle='None';Opacity=0;StartPosition='CenterScreen';Size='1,1'};"
    "$o.Show();$o.Activate();"
    "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
    f"$d.Description='{TITLE}';$d.ShowNewFolderButton=$true;"
    "$r=$d.ShowDialog($o);$o.Close();"
    "if($r -eq 'OK'){[Console]::Out.Write($d.SelectedPath)}")

_TK_PICK = ("import tkinter as t, tkinter.filedialog as f; r=t.Tk(); r.withdraw(); "
            f"r.attributes('-topmost', 1); print(f.askdirectory(title='{TITLE}') or '', end='')")


def pick_folder_argv(name, which=shutil.which, python=None):
    """argv for a native folder dialog run as its own process (Tk needs the main
    thread on macOS; the embedded Windows Python has no tkinter). None = no picker."""
    if name == "win":
        ps = which("powershell") or which("pwsh")
        if ps:
            return [ps, "-NoProfile", "-STA", "-Command", _PS_PICK]
    elif name == "mac":
        osa = which("osascript")
        if osa:
            return [osa, "-e", f'POSIX path of (choose folder with prompt "{TITLE}")']
    else:
        z = which("zenity")
        if z:
            return [z, "--file-selection", "--directory", f"--title={TITLE}"]
        k = which("kdialog")
        if k:
            return [k, "--getexistingdirectory", os.path.expanduser("~"), "--title", TITLE]
    return [python, "-c", _TK_PICK] if python else None


def pick_folder(argv, run=None):
    """The chosen folder, or "" when cancelled."""
    run = run or (lambda a: _run(a, timeout=PICK_TIMEOUT))
    rc, out = run(argv)
    return out.strip() if rc == 0 else ""


def clean_folder(text):
    """A typed or pasted folder: trimmed, minus one pair of wrapping quotes
    (Explorer's "Copy as path" adds them)."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def remember(recent, folder, cap=RECENT_CAP):
    """Recent folders, newest first, without duplicates."""
    key = lambda p: os.path.normcase(os.path.normpath(p))
    keep = [p for p in recent or () if isinstance(p, str) and p and key(p) != key(folder)]
    return [folder] + keep[:cap - 1]
