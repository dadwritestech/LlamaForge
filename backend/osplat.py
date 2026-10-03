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


def parse_meminfo(text):
    """MemTotal (bytes) from /proc/meminfo contents; 0 if absent."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024   # value is in kB
    return 0


def total_ram_bytes():
    """Total physical RAM in bytes across platforms; 0 if undetectable.
    Never raises."""
    try:
        if IS_MAC:
            return int(mac_mem_bytes() or 0)
        if IS_LINUX:
            with open("/proc/meminfo", encoding="utf-8") as f:
                return parse_meminfo(f.read())
        if IS_WIN:
            out = run_text(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
            digits = out.strip()
            return int(digits) if digits.isdigit() else 0
    except Exception:
        return 0
    return 0


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


def parse_ss_pids(text):
    """`ss -ltnpH sport = :port` output -> [pid, ...] from its users:((...)) column."""
    return [int(p) for p in re.findall(r"\bpid=(\d+)", text)]


def port_tool():
    """Which of lsof / ss / fuser this machine has, or None. lsof is missing on
    Arch, minimal Debian/Fedora and most containers; without any of them stop,
    restart and update silently found nothing (review 05 #3)."""
    import shutil
    for tool in ("lsof", "ss", "fuser"):
        if shutil.which(tool):
            return tool
    return None


def pid_on_port_posix(port, tool=None):
    tool = tool or port_tool()
    port = int(port)
    if tool == "lsof":
        pids = parse_lsof_pids(run_text(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"]))
    elif tool == "ss":
        pids = parse_ss_pids(run_text(["ss", "-ltnpH", f"sport = :{port}"]))
    elif tool == "fuser":     # pids on stdout, the "8080/tcp:" label on stderr
        pids = parse_lsof_pids(run_text(["fuser", f"{port}/tcp"]))
    else:
        pids = []
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


def refresh_path():
    """Re-read the machine PATH so a just-installed tool is visible right away.

    winget/choco write PATH into the registry and broadcast WM_SETTINGCHANGE,
    but this process inherited its copy of the environment at launch, so
    shutil.which() kept reporting a freshly installed ninja as MISSING until
    LlamaForge was restarted. Windows keeps the authoritative value in the
    registry: read it back and merge it in, keeping anything this process added
    itself (venv/activate prepends, test shims) ahead of the registry entries.

    POSIX package managers install into directories that are already on PATH, so
    there is nothing to refresh. Returns True if PATH gained any entries.
    """
    if not IS_WIN:
        return False
    try:
        import winreg
    except ImportError:                      # non-CPython / stripped install
        return False
    found = []
    for hive, sub in ((winreg.HKEY_LOCAL_MACHINE,
                       r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                      (winreg.HKEY_CURRENT_USER, "Environment")):
        try:
            with winreg.OpenKey(hive, sub) as key:
                raw, _ = winreg.QueryValueEx(key, "Path")
        except OSError:                      # key or value absent; the other may still work
            continue
        found += [os.path.expandvars(p) for p in str(raw).split(os.pathsep) if p.strip()]
    if not found:
        return False
    current_entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    seen = {p.lower().rstrip("\\/") for p in current_entries}
    added = []
    for p in found:
        key = p.lower().rstrip("\\/")
        if key not in seen:
            seen.add(key)
            added.append(p)
    if not added:
        return False
    os.environ["PATH"] = os.pathsep.join(current_entries + added)
    return True
