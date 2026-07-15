"""Detect and (with consent) install the vLLM stack inside WSL2.

Mirrors prereqs.py but for the Linux side. Install is deliberately sudo-free:
we fetch the static `uv` binary, let it manage its own standalone Python, and
create an isolated venv at ~/.llamaforge/vllm-venv. Nothing touches the system
Python or apt. The install runs as a streamed background job (see server.py),
same UX as the CMake build log.
"""
import re

import wsl

VENV = "~/.llamaforge/vllm-venv"
UV   = "~/.llamaforge/bin/uv"


def status(distro):
    """Report WSL / distro / GPU-passthrough / vLLM presence for the UI."""
    distros = wsl.list_distros()
    if not distros:
        return {"wsl": {"present": False}, "distros": [],
                "gpu": {"present": False}, "vllm": {"present": False, "version": ""}}
    chosen = distro or (distros[0]["name"])
    gpu = _gpu(chosen)
    vllm = _vllm_version(chosen)
    return {
        "wsl": {"present": True},
        "distros": distros,
        "chosen": chosen,
        "gpu": gpu,
        "vllm": vllm,
    }


def _gpu(distro):
    code, out, _err = wsl.run("nvidia-smi -L 2>/dev/null | head -n 2", distro=distro, timeout=15)
    if code == 0 and "GPU" in out:
        return {"present": True, "info": out.strip()}
    return {"present": False, "info": ""}


def _vllm_version(distro):
    code, out, _err = wsl.run(f"{VENV}/bin/vllm --version 2>/dev/null", distro=distro, timeout=20)
    if code == 0:
        m = re.search(r"(\d+\.\d+\.\d+)", out)
        return {"present": True, "version": m.group(1) if m else out.strip()}
    return {"present": False, "version": ""}


def install_script():
    """The sudo-free bash that installs uv + a vLLM venv. Run via wsl.popen so
    its output streams to the vLLM setup log."""
    return "\n".join([
        "set -e",
        "mkdir -p ~/.llamaforge/bin",
        "export UV_INSTALL_DIR=~/.llamaforge/bin",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"{UV} venv {VENV} --python 3.12",
        f"source {VENV}/bin/activate",
        f"{UV} pip install vllm",
        "echo VLLM_INSTALL_DONE",
    ])


def update_script():
    """Upgrade vLLM in the existing venv."""
    return "\n".join([
        "set -e",
        f"{UV} pip install --python {VENV}/bin/python -U vllm",
        "echo VLLM_UPDATE_DONE",
    ])


import time as _time
_PYPI_TTL = 3600           # PyPI releases don't warrant a hit per Build-tab open
_pypi_cache = {"at": 0.0, "version": ""}

def latest_pypi_version(force=False):
    """Latest vLLM version on PyPI (for the 'update available' indicator),
    cached for an hour. force=True refetches (the UI's Refresh button)."""
    import json, urllib.request
    now = _time.time()
    if not force and _pypi_cache["version"] and now - _pypi_cache["at"] < _PYPI_TTL:
        return _pypi_cache["version"]
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/vllm/json", timeout=15) as r:
            v = json.loads(r.read().decode())["info"]["version"]
        _pypi_cache.update(at=now, version=v)
        return v
    except Exception:
        return _pypi_cache["version"]   # fall back to last known, if any
