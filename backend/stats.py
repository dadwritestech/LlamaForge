"""Usage statistics for LlamaForge.

The dashboard never sees inference traffic (clients hit the llama.cpp router
directly), and llama.cpp's own Prometheus counters reset on restart and keep no
per-model history. So this module runs a background poller that scrapes the
router's `/metrics`, diffs the token counters of every loaded model (each has its own
`/metrics?model=`, so several loaded at once on a multi-model pool keep apart),
plus the bare `/metrics` of every process slot (a model pinned to another build
runs in its own llama-server, outside the router), and persists per-model +
daily totals to stats.json. Live numbers are reported per model too. Pure stdlib.
"""
import json, os, re, threading, time, urllib.request, urllib.parse
from datetime import date

import atomicio, config, network_policy

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
DAILY_KEEP = 30      # retain ~a month of daily buckets (UI toggles 14/30 days)

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
        self._prev = {}            # model -> (prompt_total, gen_total) from last poll
        self._vprev = None         # (prompt, gen) from last vLLM poll
        self._vprev_model = None
        self._idle = {}            # model -> was generation idle last poll (run count)
        self._dirty = False
        self._last_flush = 0.0
        self.live = {"prompt_per_sec": 0.0, "gen_per_sec": 0.0,
                     "requests_processing": 0, "loaded_model": None,
                     "loaded_models": [], "models": [], "router_up": False}
        # {model: endpoint} of the process slots that are up (models pinned to
        # another build run as their own llama-server, outside the router).
        # routes wires it to slotproc; stats can't import routes.
        self.proc_source = lambda: {}

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
            atomicio.write_json(STATS_FILE, self.data, indent=None)
            self._dirty = False
            self._last_flush = now
        except Exception:
            pass

    # ---------- router access ----------
    def _base(self):
        return f"http://127.0.0.1:{config.load()['router_port']}"

    def _fetch(self, url, c, timeout):
        headers = {}
        key = network_policy.effective_key(c)
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode(errors="replace")

    def _get(self, path, timeout=4):
        c = config.load()
        return self._fetch(f"http://127.0.0.1:{c['router_port']}" + path, c, timeout)

    def _get_proc(self, endpoint, path, timeout=4):
        """A process slot is started with the router's key (slotctl), so it
        takes the same bearer."""
        return self._fetch(endpoint + path, config.load(), timeout)

    def _procs(self):
        try:
            return dict(self.proc_source() or {})
        except Exception:
            return {}

    def _router_state(self):
        """(router_up, loaded model ids, the main first). One /models call
        decides both: the router is 'up' whenever /models answers, whether or
        not a model is loaded. (Its /metrics is per-model and 400s without a
        model name, so /metrics can't be used to judge liveness.)"""
        try:
            data = json.loads(self._get("/models"))
        except Exception:
            return (False, [])
        ids = [m.get("id") for m in data.get("data", [])
               if m.get("id") and m.get("id") != "default"
               and m.get("status", {}).get("value") == "loaded"]
        main = self._main()
        return (True, sorted(ids, key=lambda mid: mid != main))

    def _main(self):
        """The multi-model pool's main, if one is set."""
        try:
            s = config.load().get("slots")
            return (s.get("main") if isinstance(s, dict) else "") or ""
        except Exception:
            return ""

    # ---------- accumulation (call under self.lock) ----------
    def _model(self, mid):
        return self.data["models"].setdefault(
            mid, {"prompt": 0, "generated": 0, "loaded_secs": 0, "gen_secs": 0,
                  "runs": 0, "last_used": 0})

    def _record_tokens(self, mid, dp, dg):
        m = self._model(mid)
        m["prompt"] += int(dp)
        m["generated"] += int(dg)
        if dg > 0:   # generation was active this poll window -> feeds avg tok/s
            m["gen_secs"] = m.get("gen_secs", 0) + POLL_SECS
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
        # One /models call tells us both whether the router is up and which
        # models are loaded. The router's /metrics is per-model and 400s without
        # a model name, so we must know the models before scraping them -
        # scraping bare /metrics (the old bug) made every poll look like the
        # router down.
        # Process slots (models pinned to another build) run outside the router
        # with their own bare /metrics, and keep running when the router is
        # down. A model the router serves is scraped there, once. Models that
        # were loaded last poll but not now drop out of _prev, so they start
        # from a fresh baseline when they come back.
        up, loaded = self._router_state()
        procs = {m: ep for m, ep in self._procs().items() if m not in loaded}

        scraped, where = {}, {}
        for model in loaded:
            where[model] = "router"
            # autoload=false: a model unloaded since /models answered (by a
            # swap, an unload, another tool) must 400 here, not be reloaded.
            try:
                scraped[model] = _parse_metrics(
                    self._get("/metrics?model=" + urllib.parse.quote(model)
                              + "&autoload=false"))
            except Exception:
                scraped[model] = None      # unknown, not zero: keep its baseline
        for model, endpoint in procs.items():
            where[model] = "process"
            try:
                scraped[model] = _parse_metrics(self._get_proc(endpoint, "/metrics"))
            except Exception:
                scraped[model] = None
        names = list(scraped)
        per = [{"id": mid, "where": where[mid],
                "gen_per_sec": m.get(M_GEN_PER_SEC, 0.0),
                "prompt_per_sec": m.get(M_PROMPT_PER_SEC, 0.0),
                "requests_processing": int(m.get(M_REQ_PROCESSING, 0.0))}
               for mid, m in scraped.items() if m]
        with self.lock:
            self.live.update(
                router_up=up,
                prompt_per_sec=sum(p["prompt_per_sec"] for p in per),
                gen_per_sec=sum(p["gen_per_sec"] for p in per),
                requests_processing=sum(p["requests_processing"] for p in per),
                loaded_model=names[0] if names else None,
                loaded_models=names,
                models=per,
            )
            prev, idle = self._prev, self._idle
            self._prev, self._idle = {}, {}
            for model, metrics in scraped.items():
                self._model(model)["loaded_secs"] += POLL_SECS
                self._dirty = True
                if metrics is None:
                    if model in prev:
                        self._prev[model], self._idle[model] = prev[model], idle.get(model, True)
                    continue
                p = metrics.get(M_PROMPT_TOTAL, 0.0)
                g = metrics.get(M_GEN_TOTAL, 0.0)
                # attribute token deltas only when the model stayed loaded
                if model in prev:
                    dp = p - prev[model][0]
                    dg = g - prev[model][1]
                    if dp < 0 or dg < 0:       # counter reset (router restart)
                        dp = dg = 0
                    if dp or dg:
                        self._record_tokens(model, dp, dg)
                    if dg > 0 and idle.get(model, True):   # a fresh generation burst ~= one run
                        self._model(model)["runs"] += 1
                    self._idle[model] = (dg == 0)
                else:
                    self._idle[model] = True
                self._prev[model] = (p, g)
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
    def reset(self):
        """Zero the whole store (user-initiated from the Stats tab)."""
        with self.lock:
            self.data = _empty()
            self._prev, self._idle, self._vprev = {}, {}, None
            self._dirty = True
            self._flush(force=True)

    def summary(self):
        with self.lock:
            models = self.data["models"]
            per_model = [{
                "id": mid,
                "prompt": m["prompt"], "generated": m["generated"],
                "tokens": m["prompt"] + m["generated"],
                "loaded_secs": m["loaded_secs"], "runs": m["runs"],
                # avg generation speed over windows where generation was active;
                # gen_secs is missing in stats.json files written before v2
                "avg_tps": round(m["generated"] / m["gen_secs"], 1)
                           if m.get("gen_secs") else 0,
                "last_used": m["last_used"],
            } for mid, m in models.items()]
            per_model.sort(key=lambda x: x["tokens"], reverse=True)
            tot_p = sum(m["prompt"] for m in models.values())
            tot_g = sum(m["generated"] for m in models.values())
            tot_secs = sum(m["loaded_secs"] for m in models.values())
            most = per_model[0]["id"] if per_model and per_model[0]["tokens"] > 0 else None
            daily = [{"date": d, **v} for d, v in sorted(self.data["daily"].items())][-DAILY_KEEP:]
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
