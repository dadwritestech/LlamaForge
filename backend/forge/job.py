"""One Forge run: staged, resumable, cancellable, and it always gives the GPUs back.

Stages
  fit+A    every candidate (and the user's current setup) is loaded once: does it
           fit, a short speed round, and greedy samples for the sanity check -
           all while it's loaded, so nothing is loaded twice in this stage.
           The search is staged so it stays an hour, not a night: plain
           placements first, then drafters on the best few of them, then other
           draft windows on the best drafters only
  B        the finalists (top few, the best plain one, the current setup) get a
           longer round with different prompts; the ranking uses these numbers
  sanity   each drafting finalist vs the same setup without the drafter
  done     winner = best round-B row that passed sanity

State is written after every candidate; a run with the same request picks up
where a cancelled or crashed one stopped. Everything that touches a process
or the network is injected.
"""
import hashlib
import json
import os
import threading

from . import sanity, space

CURRENT = "current"
ROUND_A = {"code": 3, "prose": 3, "summary": 2}
ROUND_B = {"code": 8, "prose": 8, "summary": 3}
FINALISTS = {"quick": 3, "thorough": 5}
TOP_PLACEMENTS = {"quick": 2, "thorough": 3}   # plain placements that get drafters
TOP_DRAFTED = {"quick": 1, "thorough": 2}      # drafters that get other windows
SEARCH = 2                                     # bump when the plan changes shape
LOAD_TIMEOUT = 300

_RUNNING = threading.Lock()


class Busy(RuntimeError):
    pass


class Cancelled(Exception):
    pass


class _BenchFailed(Exception):
    """The server stopped answering mid-round: that candidate fails, the run goes on."""


def _fingerprint(inv, req):
    key = {"search": SEARCH, "req": req, "base": inv.get("base"), "gpus": inv.get("gpus"),
           "drafters": inv.get("drafters"), "caps": inv.get("caps")}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _plain_twin(settings):
    """The same setup without speculative decoding."""
    drop = ("spec-type", "spec-draft-model", "spec-draft-device", "spec-draft-ngl",
            "spec-draft-n-max", "model-draft", "device-draft", "draft-max")
    return {k: v for k, v in settings.items() if k not in drop}


def _pct(new, old):
    return round(100.0 * (new - old) / old, 1) if new and old else None


class Job:
    def __init__(self, path, inv, req, load, stop, bench, sample, lease, cancel=None,
                 clock=None, estimate=None):
        self.path, self.inv, self.req = path, inv, req
        self._load, self._stop, self._bench, self._sample = load, stop, bench, sample
        self._lease = lease
        self.cancel = cancel or threading.Event()
        self._clock = clock
        self._estimate = estimate or (lambda c: None)
        self.state = None

    # ---------------------------------------------------------- state
    def _now(self):
        return self._clock() if self._clock else None

    def _fresh(self, fp):
        return {"version": 1, "fingerprint": fp, "req": self.req, "stage": "fit",
                "phase": "plain", "cands": [], "rejected": [], "rows": {}, "failed": {}, "tokens": {},
                "finalists": [], "rows_b": {}, "sanity": {}, "result": None,
                "error": None, "current": None, "started_at": self._now(), "finished_at": None}

    def _open(self):
        fp = _fingerprint(self.inv, self.req)
        try:
            with open(self.path, encoding="utf-8") as f:
                st = json.load(f)
            if st.get("fingerprint") == fp and st.get("stage") not in ("done",):
                st["error"] = None
                return st
        except (OSError, ValueError):
            pass
        return self._fresh(fp)

    def save(self):
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)

    def snapshot(self):
        return json.loads(json.dumps(self.state)) if self.state else None

    # ---------------------------------------------------------- one candidate
    def _check_cancel(self):
        if self.cancel.is_set():
            raise Cancelled()

    def _with_loaded(self, cand, fn):
        """Load, run fn while it's up, always stop. -> (ok, reason|result)."""
        self.state["current"] = cand["id"]
        r = self._load(cand["settings"], LOAD_TIMEOUT)
        if not r["ok"]:
            return False, r["reason"], r
        try:
            return True, fn(), r
        except _BenchFailed as e:
            return False, str(e), r
        finally:
            self._stop()

    def _row(self, cand, rnd, load):
        vram = load.get("vram_mib") or []
        return {"id": cand["id"], "label": cand.get("label", ""),
                "code": rnd.get("code"), "prose": rnd.get("prose"),
                "summary": rnd.get("summary"), "prefill": rnd.get("prefill"),
                "vram_mib": sum(vram) if vram else None, "vram_per_gpu": vram,
                "parts": cand.get("parts", 1), "ctx": cand.get("ctx"),
                "lossy": cand.get("lossy", False), "spec": cand.get("spec"),
                "load_s": load.get("load_s")}

    def _stage_a(self, cand):
        def work():
            rnd = self._bench(ROUND_A, "A", 256)
            if rnd.get("error"):
                raise _BenchFailed(rnd["error"])
            return rnd, self._sample(sanity.PROMPTS)

        ok, out, load = self._with_loaded(cand, work)
        if not ok:
            self.state["failed"][cand["id"]] = out
        else:
            rnd, toks = out
            self.state["rows"][cand["id"]] = self._row(cand, rnd, load)
            self.state["tokens"][cand["id"]] = toks
        self.save()

    def _stage_b(self, cand):
        def work():
            rnd = self._bench(ROUND_B, "B", 512)
            if rnd.get("error"):
                raise _BenchFailed(rnd["error"])
            return rnd

        ok, out, load = self._with_loaded(cand, work)
        if ok:
            self.state["rows_b"][cand["id"]] = self._row(cand, out, load)
        else:
            self.state["failed"][cand["id"]] = f"round B: {out}"
        self.save()

    # ---------------------------------------------------------- stages
    def _add(self, cands):
        """Prune and queue candidates not seen before."""
        st = self.state
        seen = {c["id"] for c in st["cands"]} | {r["id"] for r in st["rejected"]}
        keep, rejected = space.prune([c for c in cands if c["id"] not in seen],
                                     self._estimate, self.inv.get("gpus") or [])
        st["cands"] += keep
        st["rejected"] += rejected

    def _plan(self):
        st = self.state
        if st["cands"]:
            return
        base = dict(self.inv.get("base") or {})
        st["cands"] = [{"id": CURRENT, "label": "your current setup", "settings": base,
                        "lossy": False, "parts": 1, "spec": None,
                        "ctx": int(base.get("ctx-size") or 0) or None}]
        self._add(space.candidates(self.inv, self.req, drafters=False))
        self.save()

    def _leaders(self, drafted, k):
        rows = [r for i, r in self.state["rows"].items()
                if i != CURRENT and bool(r.get("spec")) == drafted]
        return [r["id"] for r in space.rank(rows, self.req.get("intent", "coding"))[:k]]

    def _expand(self):
        """Next search phase: queue its candidates. False when the search is over."""
        st, by = self.state, self._by_id()
        t = self.req.get("time", "quick")
        phase = st.get("phase", "plain")
        if phase == "plain":
            new = [d for i in self._leaders(False, TOP_PLACEMENTS.get(t, 2))
                   for d in space.with_drafters(by[i], self.inv)]
            st["phase"] = "draft"
        elif phase == "draft":
            new = [v for i in self._leaders(True, TOP_DRAFTED.get(t, 1))
                   for v in space.spec_variants(by[i], t)]
            st["phase"] = "window"
        else:
            return False
        self._add(new)
        self.save()
        return True

    def _fit_pending(self):
        st = self.state
        for c in list(st["cands"]):
            if c["id"] in st["rows"] or c["id"] in st["failed"]:
                continue
            self._check_cancel()
            self._stage_a(c)

    def _by_id(self):
        return {c["id"]: c for c in self.state["cands"]}

    def _pick_finalists(self):
        st = self.state
        rows = [r for i, r in st["rows"].items() if i != CURRENT]
        ranked = space.rank(rows, self.req.get("intent", "coding"))
        k = FINALISTS.get(self.req.get("time"), 3)
        fin = [r["id"] for r in ranked[:k]]
        plain = next((r["id"] for r in ranked if not r.get("spec")), None)
        if plain and plain not in fin:
            fin.append(plain)             # a fallback if every drafter fails sanity
        if CURRENT in st["rows"]:
            fin.append(CURRENT)           # the baseline, measured the same way
        st["finalists"] = fin
        self.save()

    def _sanity(self):
        st, by = self.state, self._by_id()
        for fid in st["finalists"]:
            c = by.get(fid)
            if not c or not c.get("spec") or fid in st["sanity"]:
                continue
            twin = next((c2["id"] for c2 in st["cands"]
                         if not c2.get("spec") and c2["id"] != CURRENT
                         and _plain_twin(c2["settings"]) == _plain_twin(c["settings"])), None)
            if twin not in st["tokens"]:
                st["sanity"][fid] = {"ok": True, "checked": False,
                                     "reason": "no plain run to compare against"}
                continue
            r = sanity.check(st["tokens"][twin], st["tokens"].get(fid) or {})
            r["checked"] = True
            st["sanity"][fid] = r
        self.save()

    def _result(self):
        st = self.state
        rows = [r for i, r in st["rows_b"].items()
                if i != CURRENT and st["sanity"].get(i, {"ok": True})["ok"]]
        if not rows:            # every finalist failed round B or sanity: fall back to round A
            rows = [r for i, r in st["rows"].items()
                    if i != CURRENT and st["sanity"].get(i, {"ok": True})["ok"]]
        ranked = space.rank(rows, self.req.get("intent", "coding"))
        if not ranked:
            st["result"] = {"winner": None, "ranked": [], "vs_current": None,
                            "reason": "no candidate loaded on this machine"}
            return
        by = self._by_id()
        win = dict(ranked[0], settings=by[ranked[0]["id"]]["settings"])
        cur = st["rows_b"].get(CURRENT) or st["rows"].get(CURRENT)
        vs = None
        if cur:
            vs = {m: _pct(win.get(m), cur.get(m)) for m in ("code", "prose", "summary", "prefill")}
        st["result"] = {"winner": win, "ranked": ranked, "current": cur, "vs_current": vs,
                        "unverified_quality": bool(win.get("lossy"))}

    # ---------------------------------------------------------- run
    def run(self):
        if not _RUNNING.acquire(blocking=False):
            raise Busy("a Forge run is already going")
        try:
            self.state = self._open()
            st = self.state
            try:
                with self._lease():
                    self._plan()
                    st["stage"] = "fit"
                    self._fit_pending()
                    while self._expand():
                        self._fit_pending()
                    if not st["finalists"]:
                        self._pick_finalists()
                    st["stage"] = "speed"
                    by = self._by_id()
                    for fid in st["finalists"]:
                        if fid in st["rows_b"] or str(st["failed"].get(fid, "")).startswith("round B"):
                            continue
                        self._check_cancel()
                        self._stage_b(by[fid])
                    st["stage"] = "sanity"
                    self._sanity()
                    self._result()
                    st["stage"] = "done"
            except Cancelled:
                st["stage"] = "cancelled"
            except Exception as e:      # noqa: BLE001 - recorded, the lease already restored
                st["stage"], st["error"] = "error", f"{type(e).__name__}: {e}"
            st["current"] = None
            st["finished_at"] = self._now()
            self.save()
            return self.snapshot()
        finally:
            _RUNNING.release()
