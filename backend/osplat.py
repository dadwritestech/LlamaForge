"""Single choke point for OS differences (Windows / Linux / macOS).

Mirrors wsl.py's role: the rest of the backend asks this module "what platform
am I on / what does this platform's output mean" instead of sprinkling
sys.platform checks around. All parsers are pure functions over text so they
are unit-testable on any OS. Pure stdlib.
"""
import os, re, subprocess, sys

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

# Fraction of unified memory Metal will realistically let llama.cpp use.
# macOS caps the GPU working set around 70-75% of RAM on Apple Silicon.
METAL_BUDGET = 0.70


def current():
    if IS_WIN:
        return "windows"
    if IS_MAC:
        return "macos"
    return "linux"


def run_text(cmd, timeout=10):
    """Run a command, return stdout ("" on any failure)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:
        return ""


# ---------- Linux: /proc/cpuinfo ----------

def parse_proc_cpuinfo(text):
    """{name, cores, threads, avx512} from /proc/cpuinfo contents."""
    name, threads, flags = "", 0, ""
    phys_cores = set()
    cur_phys = cur_core = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = [s.strip() for s in line.split(":", 1)]
        if k == "processor":
            threads += 1
            cur_phys = cur_core = None
        elif k == "model name" and not name:
            name = v
        elif k == "physical id":
            cur_phys = v
        elif k == "core id":
            cur_core = v
        elif k == "flags" and not flags:
            flags = v
        if cur_phys is not None and cur_core is not None:
            phys_cores.add((cur_phys, cur_core))
    cores = len(phys_cores) or threads or None
    return {"name": name, "cores": cores, "threads": threads or None,
            "avx512": " avx512f" in " " + flags}


def linux_cpu():
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            return parse_proc_cpuinfo(f.read())
    except Exception:
        return {"name": "", "cores": None, "threads": None, "avx512": False}


# ---------- macOS: sysctl / vm_stat ----------

def mac_cpu():
    name = run_text(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    cores = run_text(["sysctl", "-n", "hw.physicalcpu"]).strip()
    threads = run_text(["sysctl", "-n", "hw.logicalcpu"]).strip()
    return {"name": name,
            "cores": int(cores) if cores.isdigit() else None,
            "threads": int(threads) if threads.isdigit() else None,
            "avx512": False}


def mac_mem_bytes():
    out = run_text(["sysctl", "-n", "hw.memsize"]).strip()
    return int(out) if out.isdigit() else 0


def parse_vm_stat(text):
    """Free+inactive bytes from `vm_stat` output (best-effort)."""
    m = re.search(r"page size of (\d+)", text)
    page = int(m.group(1)) if m else 16384
    free = 0
    for key in ("Pages free", "Pages inactive"):
        m = re.search(rf"{key}:\s+(\d+)", text)
        if m:
            free += int(m.group(1)) * page
    return free


def apple_silicon_gpu(mem_bytes, free_bytes=0):
    """Unified-memory pseudo-GPU entry shaped like a nvidia-smi row.
    `total` is the Metal-usable budget, not raw RAM, so VRAM-fit ratings in
    Discover stay honest on Apple Silicon."""
    total_mib = int(mem_bytes * METAL_BUDGET / (1024 * 1024))
    used_mib = max(0, int((mem_bytes - free_bytes) * METAL_BUDGET / (1024 * 1024)))
    return {"index": 0, "name": "Apple Silicon (unified memory)",
            "used": min(used_mib, total_mib), "total": total_mib,
            "util": 0, "temp": 0}


def mac_gpu_telemetry():
    mem = mac_mem_bytes()
    if not mem:
        return [{"error": "could not read hw.memsize"}]
    free = parse_vm_stat(run_text(["vm_stat"]))
    return [apple_silicon_gpu(mem, free)]


# ---------- POSIX: pid on port ----------

def parse_lsof_pids(text):
    """`lsof -ti :port` output -> [pid, ...]."""
    return [int(x) for x in text.split() if x.strip().isdigit()]


def pid_on_port_posix(port):
    out = run_text(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"])
    pids = parse_lsof_pids(out)
    return pids[0] if pids else None


# ---------- package managers ----------

def linux_pkg_manager():
    """First available of apt-get / dnf / pacman, or ""."""
    import shutil
    for pm in ("apt-get", "dnf", "pacman"):
        if shutil.which(pm):
            return pm
    return ""


def linux_install_hint(pm, package):
    """The exact command the user should run (we never sudo from the GUI)."""
    return {"apt-get": f"sudo apt-get install -y {package}",
            "dnf": f"sudo dnf install -y {package}",
            "pacman": f"sudo pacman -S --noconfirm {package}"}.get(pm, "")
