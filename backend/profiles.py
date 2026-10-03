"""Launch profiles: model + preset + pinned engine build, in one click.

plan() turns a stored profile into the concrete steps routes.py executes
(switch engine? which knobs? which model?), checked against what is actually
installed right now. Pure, so the decisions are unit-tested without a router.
"""
import os


def _install_named(installs, name):
    return next((i for i in installs if os.path.basename(i["dir"]) == name), None)


def plan(prof, installs, presets):
    """{model, backend, switch_bin, settings}; ValueError when the profile can't
    launch as saved (its build was pruned, its preset deleted...)."""
    backend = prof.get("backend") or "llamacpp"
    switch_bin = None
    if prof.get("engine"):
        if backend == "vllm":
            raise ValueError("vLLM profiles can't pin a llama.cpp build")
        inst = _install_named(installs, prof["engine"])
        if not inst or not inst.get("server_bin"):
            raise ValueError(f"engine build {prof['engine']} is no longer installed - "
                             f"edit the profile or reinstall it from Build / Update")
        if not inst.get("active"):
            switch_bin = inst["server_bin"]
    settings = None
    if prof.get("preset"):
        if backend == "vllm":
            raise ValueError("presets are llama.cpp knobs; vLLM profiles can't use one")
        preset = presets.get(prof["preset"])
        if preset is None:
            raise ValueError(f"preset {prof['preset']} no longer exists")
        # same normalization as /api/save: blank means "unset this key"
        settings = {k: (None if str(v).strip() == "" else str(v).strip())
                    for k, v in preset.items()}
    return {"model": prof["model"], "backend": backend,
            "switch_bin": switch_bin, "settings": settings}


def pinned_dirs(profiles, installs):
    """Install dirs some profile pins, so pruning old builds never breaks one."""
    names = {p.get("engine") for p in profiles.values() if isinstance(p, dict)}
    return [i["dir"] for i in installs if os.path.basename(i["dir"]) in names]
