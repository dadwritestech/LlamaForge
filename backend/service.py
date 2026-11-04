"""Opt-in "start at login" for Linux (systemd --user), macOS (launchd) and
Windows (a script in the user's Startup folder).

Off unless you run `llamaforge service on`; nothing here runs on its own. The
service just calls run.sh (run.ps1 on Windows) with LLAMAFORGE_NO_BROWSER=1, so
it starts exactly what the launcher starts (router + dashboard, the dashboard on
127.0.0.1), and stop.sh / stop.ps1 stops it. `llamaforge service off` removes
the file it wrote.

    python3 backend/service.py on|off|status <install dir>
"""
import ntpath, os, platform, plistlib, posixpath, subprocess, sys

LABEL = "io.github.dadwritestech.llamaforge"
UNIT = "llamaforge.service"
_UNSAFE = '"\\\n\r'   # can't be quoted in a unit file's Exec/Environment line


class Unsupported(Exception):
    pass


def _run(argv):
    return subprocess.call(argv)


def _checked(here):
    if not posixpath.isabs(here) or any(c in here for c in _UNSAFE):
        raise ValueError(f"can't write a service for {here!r}: use an absolute path "
                         "without quotes, backslashes or newlines")
    return here.rstrip("/")


def _sd(s):
    """systemd expands % specifiers and $VARS in Exec lines; make both literal."""
    return s.replace("%", "%%").replace("$", "$$")


def _path(path):
    """This shell's PATH (so run.sh finds the same python3), minus any entry a
    unit file couldn't quote."""
    return ":".join(p for p in path.split(":") if p and not any(c in p for c in _UNSAFE))


def systemd_unit(here, path):
    here = _checked(here)
    return ("[Unit]\n"
            "Description=LlamaForge (llama.cpp router + dashboard on 127.0.0.1)\n"
            "After=network-online.target\n\n"
            "[Service]\n"
            # run.sh starts both in the background and exits; they stay in this
            # unit's cgroup, and stop.sh (then systemd) stops them.
            "Type=oneshot\nRemainAfterExit=yes\n"
            "Environment=LLAMAFORGE_NO_BROWSER=1\n"
            f'Environment="PATH={_sd(_path(path))}"\n'
            f'ExecStart=/bin/bash "{_sd(here)}/run.sh"\n'
            f'ExecStop=/bin/bash "{_sd(here)}/stop.sh"\n\n'
            "[Install]\nWantedBy=default.target\n")


def launchd_plist(here, path):
    here = _checked(here)
    return plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": ["/bin/bash", here + "/run.sh"],
        "RunAtLoad": True,
        # run.sh exits once both are started; without this launchd kills them.
        "AbandonProcessGroup": True,
        "EnvironmentVariables": {"LLAMAFORGE_NO_BROWSER": "1", "PATH": _path(path)},
    })


def windows_vbs(here):
    """Runs run.ps1 hidden, without waiting, the way LlamaForge.vbs does."""
    if not ntpath.isabs(here) or not ntpath.splitdrive(here)[0] or any(c in here for c in '"\n\r'):
        raise ValueError(f"can't write a startup script for {here!r}: use an absolute path "
                         "with a drive and without quotes or newlines")
    here = here.rstrip("\\")
    return ("' LlamaForge start at login (llamaforge service on). Delete this file to turn it off.\r\n"
            'Set sh = CreateObject("WScript.Shell")\r\n'
            'sh.Environment("Process")("LLAMAFORGE_NO_BROWSER") = "1"\r\n'
            'sh.Run "powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass '
            f'-File ""{here}\\run.ps1""", 0, False\r\n')


def _file(home, system):
    if system == "Windows":
        # %APPDATA% can be redirected; a home given by the caller (tests) wins.
        roaming = (os.path.join(home, "AppData", "Roaming") if home
                   else os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming"))
        return os.path.join(roaming, "Microsoft", "Windows", "Start Menu", "Programs", "Startup", "LlamaForge.vbs")
    home = home or os.path.expanduser("~")
    if system == "Linux":
        return os.path.join(home, ".config", "systemd", "user", UNIT)
    if system == "Darwin":
        return os.path.join(home, "Library", "LaunchAgents", LABEL + ".plist")
    raise Unsupported("start at login is only for Linux (systemd), macOS and Windows")


def status(home=None, system=None):
    return os.path.isfile(_file(home, system or platform.system()))


def enable(here, home=None, system=None, path=None, uid=None):
    system = system or platform.system()
    path = os.environ.get("PATH", "/usr/bin:/bin") if path is None else path
    target = _file(home, system)
    if system == "Windows":
        body = windows_vbs(here)
    else:
        body = systemd_unit(here, path) if system == "Linux" else launchd_plist(here, path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as f:
        f.write(body.encode("utf-8") if isinstance(body, str) else body)
    if system == "Windows":
        return (f"LlamaForge now starts when you log in ({target}).\n"
                "Turn it off: delete that file, or untick Setup > Startup > start LlamaForge when you log in.")
    if system == "Linux":
        _run(["systemctl", "--user", "daemon-reload"])
        _run(["systemctl", "--user", "enable", UNIT])
        return (f"LlamaForge now starts when you log in ({target}).\n"
                f"Start it now: systemctl --user start {UNIT}\n"
                "To start at boot before anyone logs in: loginctl enable-linger $USER\n"
                "Turn it off: llamaforge service off")
    uid = os.getuid() if uid is None else uid
    _run(["launchctl", "bootout", f"gui/{uid}/{LABEL}"])   # a stale copy, if any
    _run(["launchctl", "bootstrap", f"gui/{uid}", target])
    return (f"LlamaForge now starts when you log in ({target}).\n"
            "Turn it off: llamaforge service off")


def disable(home=None, system=None, uid=None):
    system = system or platform.system()
    target = _file(home, system)
    if not os.path.isfile(target):
        return "Start at login is not set up; nothing to remove."
    if system == "Linux":
        _run(["systemctl", "--user", "disable", UNIT])
    elif system == "Darwin":
        _run(["launchctl", "bootout", f"gui/{os.getuid() if uid is None else uid}/{LABEL}"])
    os.remove(target)
    if system == "Linux":
        _run(["systemctl", "--user", "daemon-reload"])
    stop = "stop.ps1" if system == "Windows" else "llamaforge stop"
    return f"LlamaForge no longer starts at login. A running copy keeps running; stop it with: {stop}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("on", "off", "status"):
        print("usage: service.py on|off|status <install dir>", file=sys.stderr)
        return 2
    try:
        if argv[0] == "on":
            print(enable(os.path.abspath(argv[1])))
        elif argv[0] == "off":
            print(disable())
        else:
            print("on" if status() else "off")
    except (Unsupported, ValueError, OSError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
