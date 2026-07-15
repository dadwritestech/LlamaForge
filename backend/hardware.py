"""Detect CPU/GPU and recommend CMake build flags + runtime defaults.

Windows + Linux: NVIDIA CUDA is the primary accelerator, CPU-only fallback.
macOS: Apple Silicon unified memory with a Metal build (no CUDA).
Platform branching lives in osplat; this module just asks it.
"""
import re, subprocess

import osplat

def _run(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""

# Compute-capability -> CUDA arch number used by CMAKE_CUDA_ARCHITECTURES
def detect_gpus():
    if osplat.IS_MAC:
        return []                       # no NVIDIA on Apple Silicon; Metal instead
    out = _run(["nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits"])
    gpus = []
    for ln in out.strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 3:
            cc = f[3] if len(f) > 3 and f[3] and f[3] != "[N/A]" else ""
            gpus.append({"index": int(f[0]), "name": f[1],
                         "vram_mib": int(f[2]) if f[2].isdigit() else None,
                         "compute_cap": cc})
    return gpus

def _detect_cpu_windows():
    # wmic is removed on recent Windows 11; use PowerShell CIM.
    out = _run(["powershell", "-NoProfile", "-Command",
                "$c=Get-CimInstance Win32_Processor|Select-Object -First 1;"
                "'{0}|{1}|{2}' -f $c.Name,$c.NumberOfCores,$c.NumberOfLogicalProcessors"])
    info = {"name": "", "cores": None, "threads": None}
    parts = out.strip().split("|")
    if len(parts) == 3:
        info["name"] = parts[0].strip()
        info["cores"] = int(parts[1]) if parts[1].strip().isdigit() else None
        info["threads"] = int(parts[2]) if parts[2].strip().isdigit() else None
    # crude AVX-512 hint: recent AMD Zen4/5 and Intel server/HEDT
    n = info["name"].lower()
    info["avx512_hint"] = any(x in n for x in ["ryzen 7 9", "ryzen 9 9", "ryzen 7 7", "ryzen 9 7", "xeon", "threadripper"])
    return info

def detect_cpu():
    if osplat.IS_LINUX:
        c = osplat.linux_cpu()
        return {"name": c["name"], "cores": c["cores"], "threads": c["threads"],
                "avx512_hint": c["avx512"]}      # real flag, not a name heuristic
    if osplat.IS_MAC:
        c = osplat.mac_cpu()
        return {"name": c["name"], "cores": c["cores"], "threads": c["threads"],
                "avx512_hint": False}
    return _detect_cpu_windows()

def recommend(gpus=None, cpu=None):
    """Return {cmake_flags:{...}, notes:[...], runtime:{...}} for this machine."""
    gpus = detect_gpus() if gpus is None else gpus
    cpu  = detect_cpu()  if cpu  is None else cpu
    flags, notes = {}, []

    if osplat.IS_MAC:
        flags["GGML_METAL"] = "ON"
        notes.append("Apple Silicon detected - Metal build (uses unified memory as VRAM).")
        runtime = {"n-gpu-layers": "99", "flash-attn": "on"}
        flags["GGML_NATIVE"] = "ON"
        return {"cmake_flags": flags, "notes": notes, "runtime": runtime,
                "gpus": gpus, "cpu": cpu}

    if gpus:
        archs = sorted({g["compute_cap"].replace(".", "") for g in gpus if g["compute_cap"]})
        flags["GGML_CUDA"] = "ON"
        if archs:
            flags["CMAKE_CUDA_ARCHITECTURES"] = ";".join(archs)
            notes.append(f"CUDA build for arch(s) {', '.join(archs)} ({len(gpus)} GPU(s)).")
        flags["GGML_CUDA_FA_ALL_QUANTS"] = "ON"   # quantized-KV flash attention
        notes.append("Enabled flash-attention for all quant KV combos.")
    else:
        notes.append("No NVIDIA GPU detected - configuring a CPU-only build.")

    flags["GGML_NATIVE"] = "ON"
    if cpu.get("avx512_hint"):
        for f in ("GGML_AVX512", "GGML_AVX512_VNNI", "GGML_AVX512_VBMI", "GGML_AVX512_BF16"):
            flags[f] = "ON"
        notes.append("Enabled AVX-512 (+VNNI/VBMI/BF16) for this CPU.")

    runtime = {
        "n-gpu-layers": "99" if gpus else "0",
        "flash-attn": "on" if gpus else "off",
    }
    return {"cmake_flags": flags, "notes": notes, "runtime": runtime,
            "gpus": gpus, "cpu": cpu}

if __name__ == "__main__":
    import json
    print(json.dumps(recommend(), indent=2))
