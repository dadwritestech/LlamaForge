"""Detect CPU/GPU and recommend CMake build flags + runtime defaults.

Windows-focused (this build target). NVIDIA CUDA is the primary accelerator;
falls back to a CPU-only build when no supported GPU is found.
"""
import re, subprocess

def _run(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""

# Compute-capability -> CUDA arch number used by CMAKE_CUDA_ARCHITECTURES
def detect_gpus():
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

def detect_cpu():
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

def recommend(gpus=None, cpu=None):
    """Return {cmake_flags:{...}, notes:[...], runtime:{...}} for this machine."""
    gpus = detect_gpus() if gpus is None else gpus
    cpu  = detect_cpu()  if cpu  is None else cpu
    flags, notes = {}, []

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
