"""A/B a llama.cpp PR build against the user's own build with llama-bench.

Both runs use the same model file and the same GPU settings from models.ini,
llama-bench's standard pp512 / tg128 tests, three repetitions each, one after
the other with every other model unloaded (the caller's lease). Numbers are
llama-bench's own averages and standard deviations: nothing is estimated, and
a difference smaller than the two spreads added together is marked as noise.
The functions are pure; ABRunner does the I/O on a thread.
"""
import json
import os
import re
import subprocess
import threading
import time

PP, TG, REPS = 512, 128, 3
TIMEOUT = 900                   # one llama-bench run; a big model loads slowly

# models.ini key -> llama-bench flag; lists that the server separates with commas
# llama-bench separates with "/"
_FLAGS = (("n-gpu-layers", "-ngl"), ("gpu-layers", "-ngl"), ("n-cpu-moe", "-ncmoe"),
          ("split-mode", "-sm"), ("main-gpu", "-mg"), ("tensor-split", "-ts"),
          ("device", "-dev"), ("flash-attn", "-fa"), ("cache-type-k", "-ctk"),
          ("cache-type-v", "-ctv"))
_SLASHED = {"-ts", "-dev"}
_INT = re.compile(r"-?\d+")
_WORD = re.compile(r"[A-Za-z0-9_.,/-]+")
_FA = {"on": "on", "off": "off", "auto": "auto", "true": "on", "1": "on", "false": "off", "0": "off"}


def bench_bin(server_bin, isfile=os.path.isfile):
    """llama-bench next to this llama-server, or "" when the build has none."""
    if not server_bin:
        return ""
    d, name = os.path.split(server_bin)
    p = os.path.join(d, "llama-bench" + (".exe" if name.lower().endswith(".exe") else ""))
    return p if isfile(p) else ""


def argv(bench, settings):
    """llama-bench argv for one model's models.ini section."""
    s = settings or {}
    model = str(s.get("model") or "").strip()
    if not model:
        raise ValueError("that model has no model file in models.ini")
    out = [bench, "-m", model, "-p", str(PP), "-n", str(TG), "-r", str(REPS), "-o", "json"]
    seen = set()
    for key, flag in _FLAGS:
        v = str(s.get(key) or "").strip()
        if not v or flag in seen:
            continue
        if flag in ("-ngl", "-ncmoe", "-mg"):
            if not _INT.fullmatch(v):
                continue
        elif flag == "-fa":
            v = _FA.get(v.lower(), "")
            if not v:
                continue
        elif not _WORD.fullmatch(v):
            continue
        if flag in _SLASHED:
            v = v.replace(",", "/")
        out += [flag, v]
        seen.add(flag)
    return out


def parse(stdout):
    """llama-bench -o json -> {pp: {ts, sd}, tg: {ts, sd}, commit, model_type}."""
    text = stdout or ""
    i, j = text.find("["), text.rfind("]")
    try:
        rows = json.loads(text[i:j + 1]) if 0 <= i < j else None
    except ValueError:
        rows = None
    if not isinstance(rows, list):
        raise ValueError("llama-bench printed no results")
    out = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        kind = "pp" if r.get("n_prompt") and not r.get("n_gen") else (
            "tg" if r.get("n_gen") and not r.get("n_prompt") else None)
        if kind:
            out[kind] = {"ts": round(float(r.get("avg_ts") or 0), 2),
                         "sd": round(float(r.get("stddev_ts") or 0), 2)}
            out["commit"] = str(r.get("build_commit") or "")
            out["model_type"] = str(r.get("model_type") or "")
    if "pp" not in out or "tg" not in out:
        raise ValueError("llama-bench printed no pp/tg results")
    return out


def compare(base, pr):
    """{pp|tg: {pct, noise}}: PR vs base in percent; noise when the difference
    is inside the two runs' spreads."""
    out = {}
    for k in ("pp", "tg"):
        a, b = base[k], pr[k]
        diff = b["ts"] - a["ts"]
        out[k] = {"pct": round(diff / a["ts"] * 100, 1) if a["ts"] else 0.0,
                  "noise": abs(diff) <= a["sd"] + b["sd"]}
    return out


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                           encoding="utf-8", errors="replace")
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"timed out after {TIMEOUT // 60} minutes"
    except OSError as e:
        return 1, "", str(e)


class ABRunner:
    """One A/B at a time. run(argv) -> (returncode, stdout, stderr); lease() is a
    context manager that has the GPUs to itself. The last result is kept in
    `path` so the card still shows it after a restart."""

    def __init__(self, path, run=None, lease=None, clock=time.time):
        self.path, self._run, self._lease, self._clock = path, run or _run, lease, clock
        self._lock = threading.Lock()
        self._thread = None
        self._st = self._load() or {"running": False, "phase": "idle"}

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                st = json.load(f)
            return st if isinstance(st, dict) and not st.get("running") else None
        except (OSError, ValueError):
            return None

    def status(self):
        with self._lock:
            return dict(self._st)

    def start(self, what, base_argv, pr_argv):
        """what = {pr, install, model, label}. False if one is already running."""
        with self._lock:
            if self._st.get("running"):
                return False
            self._st = {"running": True, "phase": "base", "pr": what.get("pr"),
                        "install": what.get("install"), "model": what.get("model"),
                        "label": what.get("label") or (f"PR #{what['pr']}" if what.get("pr") else "the other build"),
                        "started": self._clock(), "error": ""}
        self._thread = threading.Thread(target=self._go, args=(base_argv, pr_argv), daemon=True)
        self._thread.start()
        return True

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def _set(self, **kw):
        with self._lock:
            self._st.update(kw)

    def _one(self, cmd, label):
        rc, out, err = self._run(cmd)
        try:
            if rc == 0:
                return parse(out)
        except ValueError:
            pass
        tail = [ln.strip() for ln in (err or out or "").splitlines() if ln.strip()][-3:]
        raise RuntimeError(f"llama-bench on {label} failed" + (": " + " / ".join(tail) if tail else ""))

    def _go(self, base_argv, pr_argv):
        try:
            with self._lease():
                base = self._one(base_argv, "your build")
                self._set(phase="pr", base=base)
                pr = self._one(pr_argv, self.status().get("label") or "the other build")
            self._set(running=False, phase="done", pr_result=pr, delta=compare(base, pr),
                      finished=self._clock())
        except Exception as e:  # noqa: BLE001 - shown in the card
            self._set(running=False, phase="failed", error=str(e), finished=self._clock())
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.status(), f)
        except OSError:
            pass
