"""Detect and (with consent) install build prerequisites on Windows.

Order of preference for install: winget -> choco. If neither works, returns
manual instructions with links. Never installs without an explicit request.
"""
import os, shutil, subprocess, glob

# name -> {check: how to detect, winget id, choco id, manual hint+url}
TOOLS = {
    "git": {
        "cmd": "git", "args": ["--version"],
        "winget": "Git.Git", "choco": "git",
        "url": "https://git-scm.com/download/win",
    },
    "cmake": {
        "cmd": "cmake", "args": ["--version"],
        "winget": "Kitware.CMake", "choco": "cmake",
        "url": "https://cmake.org/download/",
    },
    "ninja": {
        "cmd": "ninja", "args": ["--version"],
        "winget": "Ninja-build.Ninja", "choco": "ninja",
        "url": "https://github.com/ninja-build/ninja/releases",
    },
    "python": {
        "cmd": "python", "args": ["--version"],
        "winget": "Python.Python.3.12", "choco": "python",
        "url": "https://www.python.org/downloads/windows/",
    },
}

def _which(cmd):
    return shutil.which(cmd)

def _version(spec):
    exe = _which(spec["cmd"])
    if not exe:
        return None
    try:
        out = subprocess.run([exe] + spec["args"], capture_output=True, text=True, timeout=8)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        return exe

def find_msvc():
    """MSVC's cl.exe lives inside a VS install, not on PATH. Locate via vswhere."""
    vswhere = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
    if os.path.exists(vswhere):
        try:
            out = subprocess.run([vswhere, "-latest", "-products", "*",
                                  "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                  "-property", "installationPath"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if out:
                return {"present": True, "path": out}
        except Exception:
            pass
    # fallback: look for cl.exe under common VS dirs
    for base in (r"C:\Program Files\Microsoft Visual Studio",
                 r"C:\Program Files (x86)\Microsoft Visual Studio"):
        hits = glob.glob(base + r"\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe")
        if hits:
            return {"present": True, "path": hits[0]}
    return {"present": False, "path": "",
            "url": "https://visualstudio.microsoft.com/downloads/ (Build Tools for Visual Studio -> Desktop development with C++)"}

def find_cuda():
    base = os.environ.get("CUDA_PATH") or ""
    nvcc = _which("nvcc")
    if nvcc or (base and os.path.exists(os.path.join(base, "bin", "nvcc.exe"))):
        ver = ""
        try:
            src = nvcc or os.path.join(base, "bin", "nvcc.exe")
            out = subprocess.run([src, "--version"], capture_output=True, text=True, timeout=8).stdout
            import re
            m = re.search(r"release ([\d.]+)", out)
            ver = m.group(1) if m else ""
        except Exception:
            pass
        return {"present": True, "version": ver, "path": base or nvcc}
    return {"present": False,
            "url": "https://developer.nvidia.com/cuda-downloads (needed only for NVIDIA GPU builds)"}

def status():
    """Full prerequisite report."""
    tools = {}
    for name, spec in TOOLS.items():
        v = _version(spec)
        tools[name] = {"present": v is not None, "version": v or "",
                       "winget": spec["winget"], "choco": spec["choco"], "url": spec["url"]}
    return {
        "tools": tools,
        "msvc": find_msvc(),
        "cuda": find_cuda(),
        "installers": {"winget": bool(_which("winget")), "choco": bool(_which("choco"))},
    }

def install(name):
    """Install one tool by name via winget then choco. Returns (ok, log)."""
    spec = TOOLS.get(name)
    if not spec:
        return False, f"unknown tool: {name}"
    if _which("winget"):
        r = subprocess.run(["winget", "install", "--id", spec["winget"], "-e",
                            "--accept-source-agreements", "--accept-package-agreements"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True, r.stdout or "installed via winget"
        log = "winget failed:\n" + (r.stdout + r.stderr)
    else:
        log = "winget not available\n"
    if _which("choco"):
        r = subprocess.run(["choco", "install", spec["choco"], "-y"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True, log + "\ninstalled via choco"
        return False, log + "\nchoco failed:\n" + (r.stdout + r.stderr)
    return False, log + f"\nNo package manager. Install manually: {spec['url']}"

if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2))
