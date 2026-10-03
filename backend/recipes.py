"""Shareable recipes: a launch profile as readable JSON someone else can import.

A recipe names the GGUF file (and the Hugging Face repo it came from, when the
download folder says so), the tuning knobs, and the llama.cpp build it was
pinned to. Recipes come from strangers, so parse() is the trust boundary: only
allowlisted tuning knobs survive - never paths, files, hosts, keys, server
behaviour, or newlines that could smuggle extra lines into models.ini.
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


# What a recipe may carry at all, by canonical schema key. The denylist above
# stays as a second net, but a new upstream flag is unshareable until someone
# adds it here: tuning (context, memory, offload, sampling, speculative,
# chat/reasoning), never server behaviour like timeouts, logging or caches.
_ALLOW = {
    "ctx-size", "predict", "batch-size", "ubatch-size", "keep", "swa-full", "flash-attn",
    "rope-scaling", "rope-scale", "rope-freq-base", "rope-freq-scale", "kv-offload", "repack",
    "cache-type-k", "cache-type-v", "defrag-thold", "mlock", "mmap", "direct-io",
    "override-tensor", "n-cpu-moe", "gpu-layers", "fit", "fit-target", "fit-ctx", "op-offload",
    "no-mmproj", "mmproj-auto", "mmproj-offload", "override-kv",
    "samplers", "sampler-seq", "seed", "ignore-eos", "logit-bias", "temp", "top-k", "top-p",
    "min-p", "top-nsigma", "typical", "repeat-last-n", "repeat-penalty", "presence-penalty",
    "frequency-penalty", "adaptive-target", "adaptive-decay", "grammar", "json-schema",
    "backend-sampling",
    "spec-type", "draft", "draft-min", "ctx-checkpoints", "checkpoint-min-step",
    "spec-draft-type-k", "spec-draft-type-v", "spec-draft-override-tensor",
    "spec-draft-n-cpu-moe", "spec-draft-ngl", "spec-draft-backend-sampling",
    "kv-unified", "context-shift", "reverse-prompt", "special", "warmup", "pooling", "parallel",
    "cont-batching", "image-min-tokens", "image-max-tokens", "mtmd-batch-max-tokens",
    "embd-normalize",
    "chat-template-kwargs", "cache-prompt", "cache-reuse", "jinja", "reasoning-format",
    "reasoning", "reasoning-effort", "reasoning-budget", "reasoning-budget-message",
    "reasoning-preserve", "skip-chat-parsing", "prefill-assistant", "slot-prompt-similarity"}
_ALLOW_PREFIX = ("yarn-", "dry-", "xtc-", "mirostat", "dynatemp-", "spec-ngram-",
                 "spec-draft-n-", "spec-draft-p-")


def shareable(key, kind=None):
    """True for knobs safe and sensible to carry between machines (sampling,
    ctx, offload...). `key` is the canonical schema key; `kind` its schema
    type when known."""
    if not (isinstance(key, str) and _KEY_RE.match(key)) or kind == "path" \
            or key in argspec.RESERVED:
        return False
    if key in _ALLOW:              # audited by name; the patterns below are for prefixes
        return True
    return (key.startswith(_ALLOW_PREFIX) and key not in _DENY
            and key not in _MACHINE and not key.startswith(_MACHINE_PREFIX)
            and not key.startswith(_DENY_PREFIX) and not key.endswith(_DENY_SUFFIX))


def knob_index(schema):
    """{spelling: (canonical key, type)} for every long form the live schema
    knows (argspec aliases include negations like no-mmap). Short flags are
    never aliases, so they stay unknown and get dropped."""
    out = {}
    for g in (schema or {}).get("groups", []):
        for k in g.get("knobs", []):
            key = k.get("key")
            for a in [key] + list(k.get("aliases") or []):
                if a:
                    out[a] = (key, k.get("type"))
    return out


def clean(settings, known=None):
    """(kept, dropped) for a {knob: value} map. With a knob_index, spellings
    become the canonical key (a negation like no-mmap keeps its spelling, it
    is a different value) and the canonical key is what's checked; without
    one, only canonical allowlisted names pass."""
    kept, dropped = {}, []
    for k, v in settings.items():
        canon, kind = (known.get(k) or (None, None)) if known is not None else (k, None)
        if canon and isinstance(k, str) and shareable(canon, kind):
            negated = k.startswith("no-") and not canon.startswith("no-")
            kept[k if negated else canon] = v
        else:
            dropped.append(str(k)[:60])
    return kept, dropped


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


def export(name, prof, section, presets, installs, known=None):
    """Build the recipe for one saved profile. `section` is the model's
    models.ini section; the profile's preset is overlaid on it, the same order
    a launch applies them. `known` canonicalizes spellings (see clean())."""
    if (prof.get("backend") or "llamacpp") != "llamacpp":
        raise ValueError("only llama.cpp profiles can be shared as recipes")
    settings = {k: str(v) for k, v in clean(section, known)[0].items() if str(v).strip()}
    for k, v in clean(presets.get(prof.get("preset") or "") or {}, known)[0].items():
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


def parse(src, known=None):
    """Validate an untrusted recipe (dict or JSON text). Returns
    {name, model{id,file,hf_repo}, settings, engine, dropped}; ValueError if it
    isn't a usable recipe. Unsafe or unknown knobs are dropped and listed.
    `known` is knob_index(schema) - see clean()."""
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
    settings, dropped = clean({k: _text(v, f"setting {k!r}", MAX_VALUE) for k, v in raw.items()}, known)
    settings = {k: v for k, v in settings.items() if v}
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
