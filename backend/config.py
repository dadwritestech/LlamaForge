"""LlamaForge configuration + models.ini management.

All machine-specific paths live in config.json so the project is portable:
nothing is hardcoded. On a fresh machine, bootstrap writes config.json.
"""
import json, os, re

import gguf

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG    = os.path.join(ROOT, "config.json")

DEFAULTS = {
    "llama_src":   "",                       # git checkout of llama.cpp
    "build_dir":   "",                       # cmake build dir (usually <src>/build)
    "server_bin":  "",                       # path to llama-server(.exe)
    "models_ini":  os.path.join(ROOT, "models.ini"),
    "model_dirs":  [],                       # directories to scan for GGUFs
    "router_port": 8080,
    "panel_port":  8090,
    "router_host": "127.0.0.1",               # 127.0.0.1 = local only, 0.0.0.0 = reachable on the LAN
    "router_api_key": "",                     # required by clients when router_host != 127.0.0.1
    "cmake_flags": {},                       # persisted build flags (from hardware detect)
    "git_remote":  "https://github.com/ggml-org/llama.cpp",
}

def load():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG):
        try:
            with open(CONFIG, encoding="utf-8-sig") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def save(cfg):
    with open(CONFIG, "w", encoding="utf-8", newline="") as f:
        json.dump(cfg, f, indent=2)
    return cfg

# ---------------- models.ini (BOM-free, comment-preserving) ----------------

def ini_path():
    return load()["models_ini"]

def read_sections(path=None):
    """Return {section: {key: value}} for all sections including [*]."""
    path = path or ini_path()
    if not path or not os.path.exists(path):
        return {}
    out, cur = {}, None
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            m = re.match(r"^\[(.+?)\]", s)
            if m:
                cur = m.group(1); out.setdefault(cur, {}); continue
            if cur is None or not s or s.startswith(";"):
                continue
            if "=" in s:
                k, v = s.split("=", 1)
                v = v.split(";", 1)[0].strip() if ";" in v else v.strip()
                out[cur][k.strip()] = v
    return out

def set_keys(section, updates, path=None):
    """Set/remove keys within a section, preserving all other lines/comments.
    updates: {key: value or None(remove)}. Creates the section if missing.
    New keys are inserted right after the section's last existing key line
    (before any trailing blank/comment lines), so they stay visually grouped."""
    path = path or ini_path()
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            lines = f.read().split("\n")

    # locate the target section's [start, end) line range
    start = end = None
    for i, line in enumerate(lines):
        m = re.match(r"^\s*\[(.+?)\]", line)
        if m:
            if m.group(1) == section:
                start = i
            elif start is not None and end is None:
                end = i
                break
    if start is None:
        # create a fresh section at end of file
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append(f"[{section}]")
        for k, v in updates.items():
            if v is not None:
                lines.append(f"{k} = {v}")
        _write(path, lines); return
    if end is None:
        end = len(lines)

    seen = set()
    out = lines[:start + 1]                    # keep header
    last_key_local = 0                          # index within body of last key line
    body = lines[start + 1:end]
    for j, line in enumerate(body):
        km = re.match(r"^\s*([\w.\-]+)\s*=", line)
        if km:
            last_key_local = j + 1
            if km.group(1) in updates:
                k = km.group(1); seen.add(k)
                continue                        # drop; re-added in place below if not None
    # rebuild body inserting updated/new keys after last key line
    new_body, inserted = [], False
    for j, line in enumerate(body):
        km = re.match(r"^\s*([\w.\-]+)\s*=", line)
        if km and km.group(1) in updates:
            if updates[km.group(1)] is not None:
                new_body.append(f"{km.group(1)} = {updates[km.group(1)]}")
            continue
        new_body.append(line)
        if j + 1 == last_key_local:
            for k, v in updates.items():
                if v is not None and k not in seen:
                    new_body.append(f"{k} = {v}"); seen.add(k)
            inserted = True
    if not inserted:  # section had no keys; add after header
        adds = [f"{k} = {v}" for k, v in updates.items() if v is not None and k not in seen]
        new_body = adds + new_body

    _write(path, out + new_body + lines[end:])

def _write(path, lines):
    with open(path, "w", encoding="utf-8", newline="") as f:  # never write a BOM
        f.write("\n".join(lines))

def remove_section(section, path=None):
    """Delete an entire [section] block (header + body up to the next section),
    preserving everything else in the file. Returns True if it was removed."""
    path = path or ini_path()
    if not path or not os.path.exists(path):
        return False
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().split("\n")
    start = end = None
    for i, line in enumerate(lines):
        m = re.match(r"^\s*\[(.+?)\]", line)
        if m:
            if m.group(1) == section:
                start = i
            elif start is not None and end is None:
                end = i
                break
    if start is None:
        return False
    if end is None:
        end = len(lines)
    del lines[start:end]
    _write(path, lines)
    return True

# ---------------- automatic ctx-size defaults ----------------

CTX_GLOBAL_DEFAULT = str(gguf.CTX_FULL)   # "150000"

def apply_ctx_defaults(path=None):
    """Set sane ctx-size defaults across models.ini, idempotently.

    - global [*]: ctx-size = 150000 (the baseline for models that support it)
    - each model: an explicit ctx-size only when it can't reach the global -
      100000, capped at the model's GGUF-trained length (never over-extends).
      Models that support >= 150000 have their per-model override removed so they
      inherit the global; models whose trained length can't be read are left
      untouched. Only sections that actually change are rewritten.

    Returns {"changed": [section, ...]}.
    """
    path = path or ini_path()
    if not path or not os.path.exists(path):
        return {"changed": []}
    secs = read_sections(path)
    changed = []

    glob = secs.get("*", {})
    if glob.get("ctx-size") != CTX_GLOBAL_DEFAULT:
        set_keys("*", {"ctx-size": CTX_GLOBAL_DEFAULT}, path)
        changed.append("*")

    for sec, kv in secs.items():
        if sec == "*":
            continue
        mpath = kv.get("model")
        if not mpath:
            continue
        d = gguf.default_ctx(mpath)
        if d is None:                       # unknown trained length -> leave as-is
            continue
        cur = kv.get("ctx-size")
        if d == 0:                          # supports the global default
            if cur is not None:
                set_keys(sec, {"ctx-size": None}, path)
                changed.append(sec)
        elif cur != str(d):                 # needs an explicit smaller override
            set_keys(sec, {"ctx-size": str(d)}, path)
            changed.append(sec)
    return {"changed": changed}

def ensure_global(defaults, path=None):
    """Make sure a [*] global section exists with sane defaults (first run)."""
    path = path or ini_path()
    secs = read_sections(path)
    if "*" not in secs:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("version = 1\n")
        set_keys("*", defaults, path)
