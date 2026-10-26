"""Load one candidate in a sandbox process and say whether it really fits.

"Fits" means the server came up AND no GPU is at the edge of its memory. On
Windows the CUDA driver quietly spills into shared system RAM instead of failing,
which loads fine and then decodes at a fraction of the speed - so a load that
leaves a device >98% full is a failure too.

The log is read while the process loads: with --split-mode tensor an OOM ends in
a GGML_ASSERT and the process can sit in "loading" without ever exiting.
"""
import re

HEADROOM = 0.98
POLL_S = 1.0

_OOM = re.compile(r"allocating ([\d.]+) MiB on device (\d+): cudaMalloc failed: out of memory")
_WHAT = re.compile(r"failed to allocate buffer for ([a-z ]+?)\s*$", re.M)
_BAD_ARG = re.compile(r"(?:invalid|unknown) argument:\s*(\S+)")
_ERR_LINE = re.compile(r"^\S*\s+E\s+(?:\w+\s+)?(.*)$")
_FATAL = ("exiting due to", "GGML_ASSERT", "failed to load model", "error loading model")


def _ignored(line):
    # printed on every load with a hand-set -ngl; not an error
    return "common_fit_params" in line


def this_run(log):
    """The part of a shared log since the last start: slotproc writes a
    "=== <time> <argv>" header each time, and an older process's error must not
    fail this one."""
    if not log:
        return log
    i = log.rfind("\n=== ")
    if i < 0:
        return log[4:] if log.startswith("=== ") else log
    return log[i + 1:]


def failure_reason(log):
    """None while the log shows nothing fatal, else one short reason."""
    if not log:
        return None
    lines = [ln for ln in log.splitlines() if not _ignored(ln)]
    text = "\n".join(lines)
    m = _OOM.search(text)
    if m:
        what = _WHAT.search(text)
        size = round(float(m.group(1)))
        return (f"out of memory on GPU{m.group(2)} allocating {size} MiB"
                + (f" for the {what.group(1)}" if what else ""))
    m = _BAD_ARG.search(text)
    if m:
        return f"this build rejects {m.group(1)}"
    if not any(f in text for f in _FATAL):
        return None
    errs = []
    for ln in lines:
        e = _ERR_LINE.match(ln)
        if e and "exiting due to" not in ln:
            errs.append(e.group(1).strip())
        elif "GGML_ASSERT" in ln:
            errs.append(ln[ln.index("GGML_ASSERT"):].strip())
    return (errs[-1] if errs else "the server exited while loading")[:200]


def spill(used, gpus, headroom=HEADROOM):
    """Devices whose memory is at the edge, as reasons. A missing or all-zero
    reading (WDDM can report 0 with memory in use) is no evidence either way."""
    if not used or not any(used):
        return []
    out = []
    for g, u in zip(gpus, used):
        total = g.get("vram_mib") or 0
        if total and u > headroom * total:
            out.append(f"GPU{g['index']} ({g['name']}) is {u / total:.0%} full - "
                       f"it would spill into system RAM")
    return out


class Prober:
    """start(settings) -> (ok, err); state() -> loading|ready|failed|None;
    stop(); vram() -> MiB used per GPU; log_tail() -> str. All injected."""

    def __init__(self, start, state, stop, vram, log_tail, gpus, clock, sleep,
                 headroom=HEADROOM):
        self._start, self._state, self._stop = start, state, stop
        self._vram, self._log, self._gpus = vram, log_tail, gpus
        self._clock, self._sleep, self.headroom = clock, sleep, headroom

    def _fail(self, t0, reason):
        self._stop()
        return {"ok": False, "reason": reason, "vram_mib": None,
                "load_s": round(self._clock() - t0, 1)}

    def load(self, settings, timeout=300):
        """{ok, reason, vram_mib, load_s}. On success the process stays up for the
        bench; on any failure it is stopped before returning."""
        t0 = self._clock()
        ok, err = self._start(settings)
        if not ok:
            return self._fail(t0, err or "couldn't start the server")
        while True:
            st = self._state()
            if st == "ready":
                break
            why = failure_reason(this_run(self._log()))
            if why:
                return self._fail(t0, why)
            if st in ("failed", None):
                return self._fail(t0, "the server exited while loading")
            if self._clock() - t0 >= timeout:
                return self._fail(t0, f"didn't load within {timeout} s")
            self._sleep(POLL_S)
        used = self._vram()
        over = spill(used, self._gpus, self.headroom)
        if over:
            return self._fail(t0, "; ".join(over))
        return {"ok": True, "reason": "", "vram_mib": list(used) if used else None,
                "load_s": round(self._clock() - t0, 1)}
