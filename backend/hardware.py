"""Detect CPU/GPU and recommend CMake build flags + runtime defaults.

Windows + Linux: NVIDIA CUDA is the primary accelerator, CPU-only fallback.
macOS: Apple Silicon unified memory with a Metal build (no CUDA).
Platform branching lives in osplat; this module just asks it.
"""
import csv
import io
import re
import subprocess

import osplat

def _run(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""

def detect_amd_gpus():
    """Detect AMD GPUs via rocminfo (primary) or rocm-smi."""
    # Primary: rocminfo provides accurate names and architecture strings
    out = _run(["rocminfo"])
    if out.strip():
        gpus = []
        idx = 0
        # Parse each GPU section - look for Marketing Name and gfx architecture
        sections = re.split(r'Agent\s+\d+\s*\n', out)
        for section in sections:
            # Skip non-GPU sections (CPU, etc.)
            if "Marketing Name:" not in section:
                continue
            # Extract marketing name
            name_match = re.search(r'Marketing Name:\s*(.+)', section)
            if not name_match:
                continue
            name = name_match.group(1).strip()
            # Skip CPU entries
            if name.startswith("AMD Ryzen") or name.startswith("AMD EPYC"):
                continue
            # Extract architecture from the section
            arch_match = re.search(r'Name:\s*(gfx\d+)', section)
            arch = arch_match.group(1) if arch_match else _amd_arch_from_name(name)
            # Extract global memory size from the section (in KB)
            # rocminfo format: "Size:                    16695296 (0xfec000) KB"
            vram_mib = None
            mem_match = re.search(r'Size:\s+(\d+)\s+\(0x[0-9a-f]+\)\s+KB', section)
            if mem_match:
                vram_kb = int(mem_match.group(1))
                vram_mib = vram_kb // 1024
            gpus.append({
                "index": idx,
                "name": name,
                "vram_mib": vram_mib,
                "compute_cap": arch,
                "backend": "amd",
            })
            idx += 1
        if gpus:
            return gpus

    # Fallback: rocm-smi --csv
    out = _run(["rocm-smi", "--csv"])
    if out.strip() and "index" in out.lower():
        gpus = []
        try:
            reader = csv.DictReader(io.StringIO(out.strip()))
            for row in reader:
                name = row.get("name", "").strip()
                if not name:
                    continue
                try:
                    idx = int(row.get("index", 0))
                except ValueError:
                    continue
                try:
                    vram = int(row.get("vram_total", 0))
                except ValueError:
                    vram = None
                gpus.append({
                    "index": idx,
                    "name": name,
                    "vram_mib": vram,
                    "compute_cap": _amd_arch_from_name(name),
                    "backend": "amd",
                })
            if gpus:
                return gpus
        except csv.Error:
            pass

    return []

def _amd_arch_from_name(name):
    """Map AMD GPU product name to gfx architecture string."""
    n = name.lower()
    if "rx 7900" in n or "7900" in n:
        return "gfx1100"
    if "rx 7800" in n or "7800" in n:
        return "gfx1101"
    if "rx 7700" in n or "7700" in n:
        return "gfx1101"
    if "rx 7600" in n or "7600" in n:
        return "gfx1102"
    if "rx 6800" in n or "6800" in n:
        return "gfx1030"
    if "rx 6700" in n or "6700" in n:
        return "gfx1030"
    if "mi200" in n or "mi300" in n:
        return "gfx90a"
    if "mi100" in n:
        return "gfx908"
    return ""

def detect_gpus():
    if osplat.IS_MAC:
        return []                       # no NVIDIA on Apple Silicon; Metal instead
    # Try NVIDIA first
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
                          "compute_cap": cc,
                          "backend": "nvidia"})
    if gpus:
        return gpus
    # Fall back to AMD
    return detect_amd_gpus()

def detect_ram_gb():
    """Total system RAM in GB (SI, /1e9 to match GPU vendor GB units), 0 if unknown."""
    return round(osplat.total_ram_bytes() / 1e9, 1)


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
        runtime = {"fit": "on", "flash-attn": "auto"}
        flags["GGML_NATIVE"] = "ON"
        return {"cmake_flags": flags, "notes": notes, "runtime": runtime,
                "gpus": gpus, "cpu": cpu}

    if gpus:
        # Determine backend(s)
        nvidia_gpus = [g for g in gpus if g.get("backend") == "nvidia"]
        amd_gpus = [g for g in gpus if g.get("backend") == "amd"]

        if nvidia_gpus and amd_gpus:
            # Mixed system: prefer CUDA, warn about multi-backend
            notes.append("Mixed NVIDIA and AMD GPUs detected - using CUDA build. "
                         "ROCm build would be needed for AMD GPU acceleration.")
            archs = sorted({g["compute_cap"].replace(".", "") for g in nvidia_gpus if g["compute_cap"]})
            flags["GGML_CUDA"] = "ON"
            if archs:
                flags["CMAKE_CUDA_ARCHITECTURES"] = ";".join(archs)
                notes.append(f"CUDA build for arch(s) {', '.join(archs)} ({len(nvidia_gpus)} NVIDIA GPU(s)).")
            flags["GGML_CUDA_FA_ALL_QUANTS"] = "ON"
        elif nvidia_gpus:
            # NVIDIA only
            archs = sorted({g["compute_cap"].replace(".", "") for g in nvidia_gpus if g["compute_cap"]})
            flags["GGML_CUDA"] = "ON"
            if archs:
                flags["CMAKE_CUDA_ARCHITECTURES"] = ";".join(archs)
                notes.append(f"CUDA build for arch(s) {', '.join(archs)} ({len(nvidia_gpus)} GPU(s)).")
            flags["GGML_CUDA_FA_ALL_QUANTS"] = "ON"
            notes.append("Enabled flash-attention for all quant KV combos.")
        else:
            # AMD only
            archs = sorted({g["compute_cap"] for g in amd_gpus if g["compute_cap"]})
            if not archs:
                archs = ["gfx1030"]  # fallback
            flags["GGML_HIP"] = "ON"
            flags["GPU_TARGETS"] = ",".join(archs)
            notes.append(f"ROCm build for arch(s) {', '.join(archs)} ({len(amd_gpus)} AMD GPU(s)).")
            notes.append("Requires ROCm SDK 5.4+ and hipconfig. Build with: "
                         "HIPCXX=\"$(hipconfig -l)/clang\" HIP_PATH=\"$(hipconfig -R)\" cmake ...")
    else:
        notes.append("No GPU detected - configuring a CPU-only build.")

    flags["GGML_NATIVE"] = "ON"
    if cpu.get("avx512_hint"):
        for f in ("GGML_AVX512", "GGML_AVX512_VNNI", "GGML_AVX512_VBMI", "GGML_AVX512_BF16"):
            flags[f] = "ON"
        notes.append("Enabled AVX-512 (+VNNI/VBMI/BF16) for this CPU.")

    # memory sizing (ctx, GPU layers, split) is llama.cpp's --fit; never pinned here
    runtime = {"fit": "on", "flash-attn": "auto"}
    return {"cmake_flags": flags, "notes": notes, "runtime": runtime,
            "gpus": gpus, "cpu": cpu}

if __name__ == "__main__":
    import json
    print(json.dumps(recommend(), indent=2))
