"""Turn a job's state into what the panel shows, and the winner into models.ini edits."""
import os
import re

from . import space

REPO_URL = "https://github.com/dadwritestech/LlamaForge"

# keys a Forge winner decides; any of these the winner doesn't set is unset on apply
_OWNED = set(space._OWNED) | {"cache-type-k", "cache-type-v", "spec-draft-model",
                              "spec-draft-device", "spec-draft-ngl", "spec-draft-n-max",
                              "model-draft", "device-draft", "draft-max"}


def apply_plan(winner, current, base=None):
    """(updates, undo) for config.set_keys: the section becomes the winner for the
    keys Forge decides; everything else the user wrote is left alone.

    base = the settings Forge started from (the "current" candidate). With it,
    only what Forge changed from base is written: base has the [*] keys merged
    in and may hold keys the user has renamed or removed since the run, and
    neither belongs in this section."""
    updates = {}
    if base is not None:
        for k in set(base) | set(winner):
            if winner.get(k) != base.get(k) and current.get(k) != winner.get(k):
                updates[k] = winner.get(k)
        return updates, {k: current.get(k) for k in updates}
    for k, v in winner.items():
        if current.get(k) != v:
            updates[k] = v
    for k in current:
        if k in _OWNED and k not in winner:
            updates[k] = None
    undo = {k: current.get(k) for k in updates}
    return updates, undo


def summary(state, running):
    """{stage, running, progress, loading, leaderboard, failed, rejected, result, error}."""
    if not state:
        return {"stage": "idle", "running": running, "progress": None, "leaderboard": [],
                "failed": [], "rejected": [], "result": None, "error": None, "loading": None}
    labels = {c["id"]: c.get("label", c["id"]) for c in state.get("cands", [])}
    done = len(state.get("rows", {})) + len(state.get("failed", {}))
    rows = state.get("rows_b") or state.get("rows") or {}
    board = space.rank(list(rows.values()), (state.get("req") or {}).get("intent", "coding"))
    for r in board:
        r.setdefault("label", labels.get(r["id"], r["id"]))
        r["finalist"] = r["id"] in state.get("finalists", [])
        san = state.get("sanity", {}).get(r["id"])
        r["sanity"] = san
    return {"stage": state.get("stage"), "running": running,
            "progress": {"done": min(done, len(labels)), "total": len(labels)},
            "loading": labels.get(state.get("current")) if state.get("current") else None,
            "leaderboard": board,
            "failed": [{"id": i, "label": labels.get(i, i), "reason": why}
                       for i, why in state.get("failed", {}).items()],
            "rejected": list(state.get("rejected", [])),
            "result": state.get("result"), "error": state.get("error"),
            "applied": state.get("applied"), "req": state.get("req")}


# ---------------------------------------------------------------- sharing
# A result card people can post. It leaves the machine, so it carries only the
# flags Forge decides, with every file cut to its name: no paths, keys or hosts.
_FILE_KEYS = ("spec-draft-model", "model-draft")
_METRICS = ("code", "prose")


def _short_gpu(name):
    return re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", str(name or "")).strip() or "GPU"


def _basename(path):
    return os.path.basename(str(path or "").replace("\\", "/"))


def _md_safe(text):
    """One line with nothing that can open a table cell, code span or link."""
    return re.sub(r"[^\w .+()/:,-]", "_", str(text or "")).strip()


def _r1(v):
    return round(float(v), 1) if v is not None else None


def share(state, gpus, model_id):
    """{model, file, gpus, ctx, before, after, x, setup, settings, lossless,
    drafter_checked, drops_vision, markdown}, or None when there's no winner."""
    res = (state or {}).get("result") or {}
    win = res.get("winner")
    if not win:
        return None
    s = win.get("settings") or {}
    names = [_short_gpu(g.get("name")) for g in gpus or []]
    cur = res.get("current")
    after = {m: _r1(win.get(m)) for m in _METRICS}
    before = {m: _r1(cur.get(m)) for m in _METRICS} if cur else None
    x = ({m: round(after[m] / before[m], 1) if after[m] and before[m] else None for m in _METRICS}
         if before else None)
    settings = {k: _md_safe(_basename(v) if k in _FILE_KEYS else v)
                for k, v in sorted(s.items()) if k in _OWNED and str(v).strip()}
    setup = str(win.get("label") or win.get("id") or "")
    for g in gpus or []:
        setup = setup.replace(str(g.get("name") or "\0"), _short_gpu(g.get("name")))
    out = {"model": _md_safe(model_id), "file": _basename(s.get("model")), "gpus": names,
           "intent": ((state or {}).get("req") or {}).get("intent"),
           "ctx": win.get("ctx"), "before": before, "after": after, "x": x,
           "setup": _md_safe(setup), "settings": settings,
           "lossless": not res.get("unverified_quality"),
           "drafter_checked": bool(win.get("spec")), "drops_vision": bool(res.get("drops_vision"))}
    out["markdown"] = _markdown(out)
    return out


def _markdown(d):
    ctx = f", {d['ctx'] // 1024}k context" if d.get("ctx") else ""
    file = f" ({_md_safe(d['file'])})" if d.get("file") else ""
    lines = [f"**Auto-Forge** on `{d['model']}`{file}, {' + '.join(d['gpus']) or 'GPU'}{ctx}", ""]
    if d["before"]:
        lines += ["| tk/s | before | after | speedup |", "|---|---|---|---|"]
        lines += [f"| {m} | {d['before'][m]} | {d['after'][m]} | {d['x'][m]}x |" for m in _METRICS]
    else:
        lines += ["| tk/s | after |", "|---|---|"]
        lines += [f"| {m} | {d['after'][m]} |" for m in _METRICS]
    lines += ["", f"Winner: {d['setup']}", "", "```ini"]
    lines += [f"{k} = {v}" for k, v in d["settings"].items()]
    lines += ["```", ""]
    if not d["lossless"]:
        lines.append("Uses a lossy KV cache: answers can differ slightly from the plain setup.")
    elif d["drafter_checked"]:
        lines.append("Same answers: the drafter was checked token-for-token against the plain model.")
    else:
        lines.append("Same answers: lossless settings.")
    if d["drops_vision"]:
        lines.append("Runs with vision off (no projector loaded).")
    lines += ["", f"Measured on fresh prompts by [LlamaForge]({REPO_URL}) Auto-Forge."]
    return "\n".join(lines)
