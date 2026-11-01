"""GPU memory on machines where nvidia-smi has nothing to say (AMD, Intel).

Two sources, both stdlib:
- `llama-server --list-devices`: whatever backend the build has (Vulkan, ROCm,
  SYCL, CUDA) prints each device with its total and free MiB. This is the
  engine's own view, so fit ratings match what llama.cpp will actually use.
- Linux amdgpu sysfs (`/sys/class/drm/card*/device/mem_info_vram_*`), for an
  AMD card before any GPU build is installed.

Results have routes._gpu_telemetry()'s shape plus "via" (where they came from);
utilization and temperature aren't known here, so they're None. Elastic VRAM
stays NVIDIA-only and skips anything with "via".
"""
import os, platform, re, subprocess

IS_LINUX = platform.system() == "Linux"
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_LINE = re.compile(r"^\s*([A-Za-z]+)(\d+):\s*(.+)\s\((\d+) MiB,\s*(\d+) MiB free\)\s*$", re.M)
# A card one build reaches through two backends is listed twice; keep the
# backend the engine would rather use.
_PREFER = ("CUDA", "ROCm", "SYCL", "Vulkan")


# Integrated GPUs report shared system RAM as their memory (this box's Ryzen
# iGPU: "AMD Radeon(TM) Graphics (16412 MiB, ...)"), which would add a fake
# 16 GB to every fit rating. --list-devices has no UMA flag, so go by name.
_INTEGRATED = re.compile(r"\bRadeon(\(TM\))?\s+(\d{3,4}S?M?\s+)?Graphics\b|\bUHD\b|\bIris\b"
                         r"|Intel\(R\) (Arc\(TM\) )?Graphics$", re.I)
_SOFTWARE = re.compile(r"llvmpipe|lavapipe|SwiftShader", re.I)


def integrated(name):
    return bool(_INTEGRATED.search(name or ""))


def parse_list_devices(text):
    return [{"backend": b, "n": int(n), "name": name.strip(), "total": int(t), "free": int(f)}
            for b, n, name, t, f in _LINE.findall(text or "")]


def telemetry(devices):
    if not devices:
        return []
    devices = [d for d in devices if not _SOFTWARE.search(d["name"])]
    if not devices:
        return []
    have = {d["backend"] for d in devices}
    pick = next((b for b in _PREFER if b in have), devices[0]["backend"])
    devices = _drop_integrated([d for d in devices if d["backend"] == pick])
    return [{"index": i, "name": d["name"], "used": max(0, d["total"] - d["free"]),
             "total": d["total"], "util": None, "temp": None, "via": d["backend"]}
            for i, d in enumerate(devices)]


def _drop_integrated(devices):
    """An iGPU beside a real card: drop it. An iGPU on its own (a laptop, Strix
    Halo): keep it, llama.cpp really does run there."""
    discrete = [d for d in devices if not integrated(d["name"])]
    return discrete or devices


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def amdgpu_sysfs(root="/sys/class/drm"):
    try:
        cards = sorted((n for n in os.listdir(root) if re.fullmatch(r"card\d+", n)),
                       key=lambda n: int(n[4:]))
    except OSError:
        return []
    out = []
    for card in cards:
        dev = os.path.join(root, card, "device")
        total, used = _read(os.path.join(dev, "mem_info_vram_total")), \
            _read(os.path.join(dev, "mem_info_vram_used"))
        if _read(os.path.join(dev, "vendor")).lower() != "0x1002" or not total.isdigit():
            continue
        name = _read(os.path.join(dev, "product_name")) or f"GPU ({card})"
        if not name.upper().startswith("AMD"):
            name = "AMD " + name
        out.append({"index": len(out), "name": name,
                    "used": int(used) // 1048576 if used.isdigit() else 0,
                    "total": int(total) // 1048576, "util": None, "temp": None,
                    "via": "amdgpu"})
    out = _drop_integrated(out)
    return [dict(g, index=i) for i, g in enumerate(out)]


def _list_devices(server_bin):
    try:
        r = subprocess.run([server_bin, "--list-devices"], capture_output=True, text=True,
                           timeout=20, creationflags=_CREATE_NO_WINDOW)
        return (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def other_gpus(server_bin):
    """The engine's devices, else amdgpu sysfs on Linux, else []."""
    if server_bin and os.path.isfile(server_bin):
        found = telemetry(parse_list_devices(_list_devices(server_bin)))
        if found:
            return found
    return amdgpu_sysfs() if IS_LINUX else []
