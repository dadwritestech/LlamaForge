"""Turn a job's state into what the panel shows, and the winner into models.ini edits."""
from . import space

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
