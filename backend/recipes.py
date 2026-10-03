"""Shareable recipes: a launch profile as readable JSON someone else can import.

A recipe names the GGUF file (and the Hugging Face repo it came from, when the
download folder says so), the tuning knobs, and the llama.cpp build it was
pinned to. Recipes come from strangers, so parse() is the trust boundary: only
plain tuning knobs survive - never paths, files, hosts, keys, or newlines that
could smuggle extra lines into models.ini.
"""
import json
import os
import re

import argspec

VERSION = 1
MAX_SETTINGS = 300
MAX_VALUE = 4000

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
# Flags that read/write files, reach other hosts, carry secrets, switch on
# server-side tools, loosen CORS, download models, or swap the chat template.
# Matched as exact names, prefixes, or suffixes on top of argspec.RESERVED
# (router-owned); on top of that, any knob the live schema types as "path".
# Audited against every flag in a live `llama-server --help` (2026-10).
_DENY = {"rpc", "lora", "lora-scaled", "slot-save-path", "media-path", "webui",
         "props", "metrics", "slots", "api-prefix", "static-path",
         "tools", "tools-runtime", "agent", "ui", "ui-config", "ui-mcp-proxy",
         "docker-repo", "offline", "list-devices", "chat-template", "reuse-port",
         "threads-http", "embedding", "rerank", "lookup-cache-static",
         "lookup-cache-dynamic"}
_DENY_PREFIX = ("model", "mmproj", "hf-", "lora", "control-vector", "ssl-",
                "log-", "api-key", "draft-model", "spec-draft-model", "vocoder",
                "tts-", "cors-", "mcp-")
_DENY_SUFFIX = ("-file", "-path", "-dir", "-url", "-host", "-model", "-config",
                "-default", "-spec")
# Right on the machine that made the recipe, wrong on anyone else's: GPU
# topology and CPU pinning. llama.cpp's own defaults are the better guess.
_MACHINE = {"main-gpu", "tensor-split", "split-mode", "device", "numa",
            "threads", "threads-batch", "prio", "prio-batch", "poll", "poll-batch"}
_MACHINE_PREFIX = ("cpu-", "spec-draft-cpu-", "spec-draft-threads",
                   "spec-draft-prio", "spec-draft-poll", "spec-draft-device")


def shareable(key, kind=None):
    """True for knobs safe and sensible to carry between machines (sampling,
    ctx, offload...). `kind` is the schema type when known."""
    return (isinstance(key, str) and bool(_KEY_RE.match(key)) and kind != "path"
            and key not in argspec.RESERVED and key not in _DENY
            and key not in _MACHINE and not key.startswith(_MACHINE_PREFIX)
            and not key.startswith(_DENY_PREFIX) and not key.endswith(_DENY_SUFFIX))


def origin(path):
    """'owner/repo' when the GGUF sits in a LlamaForge download folder
    (owner--repo/) or the Hugging Face cache (models--owner--repo/snapshots/..)."""
    parts = re.split(r"[\\/]+", path or "")
    if len(parts) < 2:
        return ""
    for i in range(len(parts) - 2, -1, -1):
        d = parts[i]
        if d.startswith("models--") and d.count("--") >= 2:
            owner, _, name = d[len("models--"):].partition("--")
            repo = f"{owner}/{name}"
            return repo if _REPO_RE.match(repo) else ""
    owner, sep, name = parts[-2].partition("--")
    repo = f"{owner}/{name}"
    return repo if sep and _REPO_RE.match(repo) else ""


def _install(installs, pred):
    return next((i for i in installs if pred(i)), None)


def export(name, prof, section, presets, installs):
    """Build the recipe for one saved profile. `section` is the model's
    models.ini section; the profile's preset is overlaid on it, the same order
    a launch applies them."""
    if (prof.get("backend") or "llamacpp") != "llamacpp":
        raise ValueError("only llama.cpp profiles can be shared as recipes")
    settings = {k: str(v) for k, v in section.items() if shareable(k) and str(v).strip()}
    for k, v in (presets.get(prof.get("preset") or "") or {}).items():
        if not shareable(k):
            continue
        if str(v).strip():
            settings[k] = str(v).strip()
        else:
            settings.pop(k, None)
    if prof.get("engine"):
        inst = _install(installs, lambda i: os.path.basename(i["dir"]) == prof["engine"])
    else:
        inst = _install(installs, lambda i: i.get("active"))
    path = section.get("model", "")
    return {"llamaforge_recipe": VERSION, "name": name,
            "model": {"id": prof["model"], "file": os.path.basename(re.sub(r"\\", "/", path)),
                      "hf_repo": origin(path)},
            "settings": dict(sorted(settings.items())),
            "engine": {"tag": inst.get("tag", ""), "variant": inst.get("variant", "")} if inst else None}


def _text(v, field, limit=200):
    v = "" if v is None else str(v).strip()
    if len(v) > limit or "\n" in v or "\r" in v:
        raise ValueError(f"recipe {field} is invalid")
    return v


def parse(src, known_keys=None):
    """Validate an untrusted recipe (dict or JSON text). Returns
    {name, model{id,file,hf_repo}, settings, engine, dropped}; ValueError if it
    isn't a usable recipe. Unsafe or unknown knobs are dropped and listed.
    `known_keys` is a set of schema keys, or a {key: type} dict so knobs the
    schema types as paths are dropped too."""
    if isinstance(src, str):
        try:
            src = json.loads(src)
        except ValueError:
            raise ValueError("that isn't recipe JSON") from None
    if not isinstance(src, dict) or src.get("llamaforge_recipe") != VERSION:
        raise ValueError("not a LlamaForge recipe (or made by a newer version)")
    name = _text(src.get("name"), "name", 40) or "imported"
    m = src.get("model") if isinstance(src.get("model"), dict) else {}
    file = _text(m.get("file"), "model file")
    if not file.lower().endswith(".gguf") or file != os.path.basename(file) \
            or "/" in file or "\\" in file or file.startswith("."):
        raise ValueError("recipe model file must be a bare .gguf file name")
    repo = _text(m.get("hf_repo"), "hf_repo")
    if repo and not _REPO_RE.match(repo):
        raise ValueError("recipe hf_repo must look like owner/name")
    raw = src.get("settings") or {}
    if not isinstance(raw, dict) or len(raw) > MAX_SETTINGS:
        raise ValueError("recipe settings are invalid")
    settings, dropped = {}, []
    for k, v in raw.items():
        v = _text(v, f"setting {k!r}", MAX_VALUE)
        kind = known_keys.get(k) if isinstance(known_keys, dict) else None
        if shareable(k, kind) and (known_keys is None or k in known_keys):
            if v:
                settings[k] = v
        else:
            dropped.append(str(k)[:60])
    eng = src.get("engine") if isinstance(src.get("engine"), dict) else None
    engine = {"tag": _text(eng.get("tag"), "engine tag", 40),
              "variant": _text(eng.get("variant"), "engine variant", 40)} if eng else None
    return {"name": name, "model": {"id": _text(m.get("id"), "model id") or file,
                                    "file": file, "hf_repo": repo},
            "settings": settings, "engine": engine, "dropped": dropped}


def match_local(model, sections):
    """The models.ini section holding this recipe's GGUF (by file name, then id)."""
    want = model["file"].lower()
    for sec, keys in sections.items():
        p = re.sub(r"\\", "/", keys.get("model", ""))
        if sec != "*" and p and os.path.basename(p).lower() == want:
            return sec
    return model["id"] if model["id"] in sections and model["id"] != "*" else None


def match_engine(engine, installs):
    """Install dir name of the recipe's build, or '' when not installed here."""
    if not engine:
        return ""
    inst = _install(installs, lambda i: i.get("tag") == engine["tag"]
                    and i.get("variant") == engine["variant"])
    return os.path.basename(inst["dir"]) if inst else ""


def unique_name(name, taken):
    if name not in taken:
        return name
    n = 2
    while f"{name} {n}" in taken:
        n += 1
    return f"{name} {n}"
