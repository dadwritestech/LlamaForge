"""Usage statistics for LlamaForge.

The dashboard never sees inference traffic (clients hit the llama.cpp router
directly), and llama.cpp's own Prometheus counters reset on restart and keep no
per-model history. So this module runs a background poller that scrapes the
router's `/metrics`, diffs the token counters, attributes the delta to the
currently-loaded model (safe: the router runs with --models-max 1), and
persists per-model + daily totals to stats.json. Pure stdlib.
"""
import json, os, re, threading, time, urllib.request
from datetime import date

import config

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_FILE = os.path.join(ROOT, "stats.json")

# Prometheus metric names from `llama-server --metrics`. Centralized so a future
# llama.cpp rename is a one-line fix; any missing metric degrades to 0.
M_PROMPT_TOTAL   = "llamacpp:prompt_tokens_total"
M_GEN_TOTAL      = "llamacpp:tokens_predicted_total"
M_PROMPT_PER_SEC = "llamacpp:prompt_tokens_seconds"
M_GEN_PER_SEC    = "llamacpp:predicted_tokens_seconds"
M_REQ_PROCESSING = "llamacpp:requests_processing"

# vLLM Prometheus counters (different names than llama.cpp). Any missing -> 0.
VLLM_PROMPT_TOTAL = "vllm:prompt_tokens_total"
VLLM_GEN_TOTAL    = "vllm:generation_tokens_total"


def vllm_token_totals(metrics):
    """(prompt_total, gen_total) from parsed vLLM /metrics."""
    return (metrics.get(VLLM_PROMPT_TOTAL, 0.0),
            metrics.get(VLLM_GEN_TOTAL, 0.0))


POLL_SECS  = 5       # how often we scrape the router
FLUSH_SECS = 15      # min interval between stats.json writes
DAILY_KEEP = 30      # retain ~a month of daily buckets (UI shows last 14)

_METRIC_RE = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$")


def _parse_metrics(text):
    """Prometheus text -> {name: value}, summing across any label sets."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        try:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError:
            pass
    return out


def _empty():
    return {"models": {}, "daily": {}, "first_seen": time.time()}


class StatsTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = self._load()
        self._prev = None          # (prompt_total, gen_total) from last poll
        self._prev_model = None
        self._vprev = None         # (prompt, gen) from last vLLM poll
        self._vprev_model = None
        self._idle = True          # was generation idle last poll (for run count)
        self._dirty = False
        self._last_flush = 0.0
        self.live = {"prompt_per_sec": 0.0, "gen_per_sec": 0.0,
                     "requests_processing": 0, "loaded_model": None,
                     "router_up": False}

    # ---------- persistence ----------
    def _load(self):
        try:
            with open(STATS_FILE, encoding="utf-8") as f:
                d = json.load(f)
            d.setdefault("models", {})
            d.setdefault("daily", {})
            d.setdefault("first_seen", time.time())
            return d
        except Exception:
            return _empty()

    def _flush(self, force=False):
        now = time.time()
        if not force and (not self._dirty or now - self._last_flush < FLUSH_SECS):
            return
        try:
            tmp = STATS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f)
            os.replace(tmp, STATS_FILE)   # atomic on the same volume
            self._dirty = False
            self._last_flush = now
        except Exception:
            pass

    # ---------- router access ----------
    def _base(self):
        return f"http://127.0.0.1:{config.load()['router_port']}"

    def _get(self, path, timeout=4):
        with urllib.request.urlopen(self._base() + path, timeout=timeout) as r:
            return r.read().decode(errors="replace")

    def _loaded_model(self):
        try:
            data = json.loads(self._get("/models"))
        except Exception:
            return None
        for m in data.get("data", []):
            mid = m.get("id")
            if mid and mid != "default" and m.get("status", {}).get("value") == "loaded":
                return mid
        return None

    # ---------- accumulation (call under self.lock) ----------
    def _model(self, mid):
        return self.data["models"].setdefault(
            mid, {"prompt": 0, "generated": 0, "loaded_secs": 0, "runs": 0, "last_used": 0})

    def _record_tokens(self, mid, dp, dg):
        m = self._model(mid)
        m["prompt"] += int(dp)
        m["generated"] += int(dg)
        m["last_used"] = time.time()
        day = self.data["daily"].setdefault(date.today().isoformat(),
                                            {"prompt": 0, "generated": 0})
        day["prompt"] += int(dp)
        day["generated"] += int(dg)
        for d in sorted(self.data["daily"])[:-DAILY_KEEP]:
            self.data["daily"].pop(d, None)
        self._dirty = True

    # ---------- polling ----------
    def _poll_vllm(self):
        """Scrape vLLM's /metrics (if a model is loaded there) and attribute
        token deltas the same way as llama.cpp. Best-effort; silent on failure.
        Call under self.lock."""
        try:
            port = config.load().get("vllm_port", 8081)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3) as r:
                metrics = _parse_metrics(r.read().decode(errors="replace"))
        except Exception:
            self._vprev = None
            return
        p, g = vllm_token_totals(metrics)
        model = None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                data = json.loads(r.read().decode())
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            model = ids[0] if ids else None
        except Exception:
            model = None
        if self._vprev is not None and model and model == self._vprev_model:
            dp, dg = p - self._vprev[0], g - self._vprev[1]
            if dp < 0 or dg < 0:
                dp = dg = 0
            if dp or dg:
                self._record_tokens(model, dp, dg)
        self._vprev = (p, g)
        self._vprev_model = model

    def poll_once(self):
        try:
            metrics = _parse_metrics(self._get("/metrics"))
        except Exception:
            with self.lock:
                self.live.update(router_up=False, prompt_per_sec=0.0,
                                 gen_per_sec=0.0, requests_processing=0)
                self._prev = None          # re-baseline on next good poll
            return

        model = self._loaded_model()
        p = metrics.get(M_PROMPT_TOTAL, 0.0)
        g = metrics.get(M_GEN_TOTAL, 0.0)
        with self.lock:
            self.live.update(
                router_up=True,
                prompt_per_sec=metrics.get(M_PROMPT_PER_SEC, 0.0),
                gen_per_sec=metrics.get(M_GEN_PER_SEC, 0.0),
                requests_processing=int(metrics.get(M_REQ_PROCESSING, 0.0)),
                loaded_model=model,
            )
            if model:
                self._model(model)["loaded_secs"] += POLL_SECS
                self._dirty = True
            # attribute token deltas only when the same model stayed loaded
            if self._prev is not None and model and model == self._prev_model:
                dp = p - self._prev[0]
                dg = g - self._prev[1]
                if dp < 0 or dg < 0:       # counter reset (router restart)
                    dp = dg = 0
                if dp or dg:
                    self._record_tokens(model, dp, dg)
                if dg > 0 and self._idle:   # a fresh generation burst ~= one run
                    self._model(model)["runs"] += 1
                    self._dirty = True
                self._idle = (dg == 0)
            else:
                self._idle = True
            self._prev = (p, g)
            self._prev_model = model
            self._poll_vllm()
            self._flush()

    def run_forever(self):
        while True:
            try:
                self.poll_once()
            except Exception:
                pass
            time.sleep(POLL_SECS)

    def start(self):
        threading.Thread(target=self.run_forever, daemon=True, name="stats-poller").start()

    # ---------- read side (for the API) ----------
    def summary(self):
        with self.lock:
            models = self.data["models"]
            per_model = [{
                "id": mid,
                "prompt": m["prompt"], "generated": m["generated"],
                "tokens": m["prompt"] + m["generated"],
                "loaded_secs": m["loaded_secs"], "runs": m["runs"],
                "last_used": m["last_used"],
            } for mid, m in models.items()]
            per_model.sort(key=lambda x: x["tokens"], reverse=True)
            tot_p = sum(m["prompt"] for m in models.values())
            tot_g = sum(m["generated"] for m in models.values())
            tot_secs = sum(m["loaded_secs"] for m in models.values())
            most = per_model[0]["id"] if per_model and per_model[0]["tokens"] > 0 else None
            daily = [{"date": d, **v} for d, v in sorted(self.data["daily"].items())][-14:]
            return {
                "totals": {
                    "prompt": tot_p, "generated": tot_g, "tokens": tot_p + tot_g,
                    "loaded_hours": round(tot_secs / 3600, 1),
                    "models_used": sum(1 for m in models.values()
                                       if m["prompt"] + m["generated"] > 0),
                    "most_used": most,
                    "total_runs": sum(m["runs"] for m in models.values()),
                },
                "per_model": per_model,
                "daily": daily,
                "live": dict(self.live),
            }


TRACKER = StatsTracker()
