"""LlamaForge configuration + models.ini management.

All machine-specific paths live in config.json so the project is portable:
nothing is hardcoded. On a fresh machine, bootstrap writes config.json.
"""
import copy, json, os, re, threading

import atomicio, gguf

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG    = os.path.join(ROOT, "config.json")

# The dashboard is a ThreadingHTTPServer with background workers (stats poller,
# build/download threads), and config.json is edited by load->mutate->save at
# many call sites. Without a lock those interleave and silently drop one side's
# change. RLock (not Lock) because update() below calls load() and save() while
# already holding it.
_LOCK = threading.RLock()

# Set when load() finds an unreadable config.json. The dashboard surfaces this
# instead of silently running on - and saving over - a corrupt file.
LOAD_ERROR = None

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
    "wsl_distro":  "",                        # WSL distro that runs vLLM ("" = auto-pick default)
    "vllm_port":   8081,                      # port vLLM serves on (WSL localhost-forwarded to Windows)
    "cmake_flags": {},                       # persisted build flags (from hardware detect)
    "git_remote":  "https://github.com/ggml-org/llama.cpp",
    # ik_llama mirrors the llama.cpp path trio and, like it, ships empty: these
    # belong to bootstrap and this machine, not to the defaults every install
    # inherits. An unset ik_llama_server_bin simply leaves the engine disabled.
    "ik_llama_src":    "",                    # git checkout of ik_llama.cpp
    "ik_llama_build_dir": "",                 # cmake build dir (usually <src>/build)
    "ik_llama_server_bin": "",                # path to ik_llama's llama-server(.exe)
    "ik_llama_models_ini": "",                # its own models.ini ("" -> sibling of models_ini)
    "ik_llama_git_remote": "https://github.com/ikawrakow/ik_llama.cpp",
    "ik_llama_cmake_flags": {},               # separate build flags for ik_llama
    "active_engine": "llamacpp",              # which binary the router uses: llamacpp | ikllama
    "auto_load_model": "",                    # model id to load automatically on launch ("" = none)
    "presets":     {},                       # named knob sets: {name: {knob: value}}
    "preset_bindings": {},                    # {model_id: preset_name} auto-applied on bind/edit
    "mtp_auto_owned": {},                     # {engine: {model_id: {key: value}}}
    "ui_mode":     "lite",                    # "lite" (curated knobs) or "advanced" (all ~220)
    "onboarded":   False,                     # first-run wizard shown once, then True
    "anthropic_default_model": "",           # fallback local model id for the Anthropic shim
    "anthropic_shim_enabled":  True,          # serve /v1/messages (Anthropic-compatible)
    "wiki_dir":      "",                       # context-doc directory ("" -> <ROOT>/wiki)
    "wiki_profiles": {},                       # {name: {"docs":[...], "description":str}}
    "wiki_active":   {},                       # {model_id: profile_name}
    "theme":         "",                       # "" = follow OS/localStorage; "light"|"dark"
    "cvd":           False,                     # colorblind-safe palette + non-color cues
    "vram_bandwidths":      {},   # optional {vram_bw,ram_bw,disk_bw} GB/s overrides (empty = presets/defaults)
    "vram_predict_enabled": True, # compute vramwise placement/tok-s estimates (offline; Discover only on expand)
    "docs_dir":      "",                        # "" = <ROOT>/docs/content
}

def load():
    global LOAD_ERROR
    cfg = copy.deepcopy(DEFAULTS)   # deep so mutable defaults (presets, lists) never alias
    with _LOCK:
        if not os.path.exists(CONFIG):
            LOAD_ERROR = None
            return cfg
        try:
            with open(CONFIG, encoding="utf-8-sig") as f:
                cfg.update(json.load(f))
            LOAD_ERROR = None
        except Exception as e:
            # An unreadable config.json used to look exactly like a fresh
            # install, and the next save() wrote defaults over it - silent total
            # loss of the user's settings. Quarantine the bytes before anything
            # can overwrite them, and remember why so the UI can say so.
            LOAD_ERROR = f"config.json could not be read ({e}); running on defaults"
            _quarantine(CONFIG)
        return cfg

def _quarantine(path):
    """Copy an unreadable file aside, write-once, so the original survives the
    next save(). Best-effort: never let this break startup."""
    bak = path + ".corrupt"
    try:
        if not os.path.exists(bak):
            import shutil
            shutil.copy2(path, bak)
    except OSError:
        pass
    return bak

def save(cfg):
    """Write config.json atomically: a crash or power loss mid-write leaves the
    previous file intact rather than a truncated one."""
    with _LOCK:
        atomicio.write_json(CONFIG, cfg)
    return cfg

def update(changes):
    """Atomic read-modify-write of config.json.

    Every caller that used to do `c = load(); c[k] = v; save(c)` should use this
    instead: those sequences interleave across request/worker threads and the
    last writer silently wins. Returns the full saved config.
    """
    with _LOCK:
        cfg = load()
        cfg.update(changes or {})
        return save(cfg)

def mutate(fn):
    """Atomic read-modify-write where the new value depends on the old one
    (nested dicts like presets / wiki_profiles / wiki_active). `fn` receives the
    loaded config and edits it in place; the result is saved under the lock.
    Returns fn's return value, so callers can hand back the sub-dict they built.
    """
    with _LOCK:
        cfg = load()
        out = fn(cfg)
        save(cfg)
        return out

def migrate():
    """One-time upgrade of an on-disk config.json for the Lite/Advanced feature.

    Runs at server startup. A config that predates this feature has no `ui_mode`
    key: classify it so returning users see no change. An install that already
    built llama.cpp (server_bin set) is treated as existing -> Advanced +
    onboarded; a fresh checkout -> Lite + not onboarded (so the wizard shows).
    Idempotent: a config already carrying `ui_mode` is returned unchanged.
    """
    with _LOCK:
        if not os.path.exists(CONFIG):
            return load()
        cfg = load()
        raw = {}
        try:
            with open(CONFIG, encoding="utf-8-sig") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
        if "ui_mode" in raw:
            return cfg
        existing = bool(cfg.get("server_bin"))
        cfg["ui_mode"] = "advanced" if existing else "lite"
        cfg["onboarded"] = existing
        save(cfg)
        return cfg

# ---------------- models.ini (BOM-free, comment-preserving) ----------------

# models.ini is edited the same read-modify-write way as config.json (set_keys,
# remove_section, apply_ctx_defaults) from request threads and from autotune's
# refine loop. Separate lock from _LOCK: the two files are independent, and
# apply_ctx_defaults holds this one across many set_keys calls.
_INI_LOCK = threading.RLock()

# A Windows drive-letter (C:\ or C:/) or UNC (\\host) path. os.path.isabs() only
# recognizes these when running ON Windows; a config.json written on a Windows
# box is still absolute when its paths are read anywhere else (e.g. CI), so match
# them directly rather than trust the host's path flavour.
_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]|^\\\\|^//")

def _abs(p):
    """Anchor a configured path to the repo root, unless it is already absolute.

    The router is spawned detached and is handed this path as an argument, so it
    resolves any relative value against *its* CWD - whatever directory the user
    launched the panel from. config.example.json ships "./models.ini", so
    starting from anywhere but the repo root pointed llama-server at a different,
    usually nonexistent registry: it came up with 0 models and the dashboard
    looked like it had lost every model on restart.

    "Absolute" spans both path flavours on purpose: re-rooting a Windows user's
    "D:/models.ini" because the host happens to be POSIX would corrupt it.
    """
    if not p:
        return p
    if os.path.isabs(p) or _WIN_ABS.match(p):
        return p
    return os.path.normpath(os.path.join(ROOT, p))

def ini_path():
    """The models.ini the active engine reads, always as an absolute path.

    ik_llama gets its own registry because the two binaries accept different
    knobs; when the user has not named one, derive a sibling of the llama.cpp
    file. Split on the extension rather than str.replace(".ini", ...), which is
    a global replace and rewrites any directory that happens to contain ".ini"."""
    c = load()
    if c.get("active_engine") == "ikllama":
        p = c.get("ik_llama_models_ini")
        if p:
            return _abs(p)
        stem, ext = os.path.splitext(_abs(c["models_ini"]))
        return stem + "-ikllama" + (ext or ".ini")
    return _abs(c["models_ini"])

def read_sections(path=None):
    """Return {section: {key: value}} for all sections including [*]."""
    path = path or ini_path()
    if not path or not os.path.exists(path):
        return {}
    out, cur = {}, None
    with _INI_LOCK, open(path, encoding="utf-8-sig") as f:
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
    with _INI_LOCK:
        return _set_keys_locked(section, updates, path)

def _set_keys_locked(section, updates, path):
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
    """Atomic, BOM-free rewrite. models.ini is the router's source of truth for
    every model; a truncated one means the router loses them all."""
    atomicio.write_text(path, "\n".join(lines))

def remove_section(section, path=None):
    """Delete an entire [section] block (header + body up to the next section),
    preserving everything else in the file. Returns True if it was removed."""
    path = path or ini_path()
    with _INI_LOCK:
        return _remove_section_locked(section, path)

# ---------------- automatic MTP wiring ----------------

def reconcile_mtp_autowire(model_id, current, desired):
    """Return safe models.ini updates and record the values LlamaForge owns.

    ``desired`` contains speculative keys inferred by a scan. Previously
    auto-written values may be replaced or removed while unchanged; a missing,
    different, or manually deleted value is treated as a user override.
    Ownership is engine-scoped because each llama family has its own registry.
    """
    current = current or {}
    desired = desired or {}

    with _LOCK:
        cfg = load()
        engine = cfg.get("active_engine", "llamacpp")
        all_owned = cfg.get("mtp_auto_owned")
        if not isinstance(all_owned, dict):
            all_owned = {}
        before_owned = copy.deepcopy(all_owned)
        engine_owned = all_owned.get(engine)
        if not isinstance(engine_owned, dict):
            engine_owned = {}
        prior = engine_owned.get(model_id)
        if not isinstance(prior, dict) or "model" not in current:
            prior = {}

        updates, next_owned = {}, {}
        for key in ("spec-draft-model", "spec-type"):
            want = desired.get(key)
            before = prior.get(key)
            actual = current.get(key)
            if key in prior:
                if actual == before:
                    updates[key] = want
                    if want is not None:
                        next_owned[key] = str(want)
            elif want is not None and not actual:
                updates[key] = want
                next_owned[key] = str(want)

        if next_owned:
            engine_owned[model_id] = next_owned
        else:
            engine_owned.pop(model_id, None)
        if engine_owned:
            all_owned[engine] = engine_owned
        else:
            all_owned.pop(engine, None)
        if all_owned != before_owned:
            cfg["mtp_auto_owned"] = all_owned
            save(cfg)
        return updates

def _remove_section_locked(section, path):
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

# ---------------- knob presets ----------------

def get_presets():
    """Named knob sets from config.json, e.g. {"coding": {"temp": "0.2"}}."""
    p = load().get("presets")
    return p if isinstance(p, dict) else {}

def save_preset(name, settings):
    """Store a named preset. `settings` is {knob: value}; blank values are
    dropped so a preset only carries the knobs it actually pins. Returns the
    full preset map."""
    name = (name or "").strip()
    if not name:
        raise ValueError("preset name is required")
    clean = {k: str(v).strip() for k, v in (settings or {}).items()
             if str(v).strip() != ""}
    with _LOCK:
        cfg = load()
        presets = cfg.get("presets")
        if not isinstance(presets, dict):
            presets = {}
        presets[name] = clean
        cfg["presets"] = presets
        save(cfg)
        return presets

def delete_preset(name):
    """Remove a named preset and any model bindings that pointed at it. Returns
    True if the preset existed."""
    with _LOCK:
        cfg = load()
        presets = cfg.get("presets")
        if not (isinstance(presets, dict) and name in presets):
            return False
        del presets[name]
        cfg["presets"] = presets
        binds = cfg.get("preset_bindings")
        if isinstance(binds, dict):
            cfg["preset_bindings"] = {m: n for m, n in binds.items() if n != name}
        save(cfg)
        return True

# ---------------- preset bindings ----------------
# A binding names the preset a model defaults to. The knobs themselves are
# materialized into models.ini by the caller (routes) at bind/edit time - config
# only owns the mapping, because reloading the router is not config's job.

def get_bindings():
    b = load().get("preset_bindings")
    return b if isinstance(b, dict) else {}

def bind_preset(model_id, name):
    """Bind model_id to preset `name` (or unbind when name is ""). Raises if the
    preset does not exist. Returns the full bindings map."""
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("model id is required")
    name = (name or "").strip()
    with _LOCK:
        cfg = load()
        binds = cfg.get("preset_bindings")
        if not isinstance(binds, dict):
            binds = {}
        if name == "":
            binds.pop(model_id, None)
        else:
            if name not in (cfg.get("presets") or {}):
                raise ValueError(f"unknown preset: {name}")
            binds[model_id] = name
        cfg["preset_bindings"] = binds
        save(cfg)
        return binds

def unbind_preset(model_id):
    """Remove a model's binding. Returns True if it had one."""
    with _LOCK:
        cfg = load()
        binds = cfg.get("preset_bindings")
        if isinstance(binds, dict) and model_id in binds:
            del binds[model_id]
            cfg["preset_bindings"] = binds
            save(cfg)
            return True
        return False

def prune_binding(model_id):
    """Drop a deleted model's binding. Alias of unbind_preset for call-site
    clarity."""
    return unbind_preset(model_id)

def bindings_for_preset(name):
    """Model ids bound to preset `name`."""
    return [m for m, n in get_bindings().items() if n == name]

# ---------------- automatic ctx-size defaults ----------------

CTX_GLOBAL_DEFAULT = str(gguf.CTX_FULL)   # "150000"

def apply_ctx_defaults(path=None):
    """Set sane ctx-size defaults across models.ini, idempotently.

    - global [*]: ctx-size = 150000 (the baseline for models that support it)
    - each model with no ctx-size of its own: get one when it can't reach the
      global - 100000, capped at the model's GGUF-trained length.
    - each model that already has one: keep it, except to clamp a value that
      over-extends past the trained length.

    This runs on every panel startup, so it only ever fills a gap or clamps an
    impossible value; it never overrules a ctx-size that is already there.
    A trained length is not a VRAM budget: a 27B Q6_K trained to 262144 still
    OOMs at 150000 on a 32 GB box, and the explicit 65536 sitting in the file is
    how the user encoded that. Deleting it (the old behaviour, on the theory
    that the model "supports the global") silently reimposed a config that could
    not load. Models whose trained length can't be read are left untouched, and
    only sections that actually change are rewritten.

    Returns {"changed": [section, ...]}.
    """
    path = path or ini_path()
    if not path or not os.path.exists(path):
        return {"changed": []}
    # Held across the whole scan+rewrite: the decision to drop or set each
    # section's ctx-size is made from `secs`, so a concurrent set_keys between
    # the read and the writes would be silently reverted.
    with _INI_LOCK:
        secs = read_sections(path)
        changed = []

        glob = secs.get("*", {})
        if glob.get("ctx-size") != CTX_GLOBAL_DEFAULT:
            _set_keys_locked("*", {"ctx-size": CTX_GLOBAL_DEFAULT}, path)
            changed.append("*")

        for sec, kv in secs.items():
            if sec == "*":
                continue
            mpath = kv.get("model")
            if not mpath:
                continue
            d = gguf.default_ctx(mpath)
            if d is None:                   # unknown trained length -> leave as-is
                continue
            cur = kv.get("ctx-size")
            if d == 0:                      # can reach the global; nothing to add
                continue
            if cur is None:                 # gap -> fill it
                _set_keys_locked(sec, {"ctx-size": str(d)}, path)
                changed.append(sec)
                continue
            try:                            # present -> only clamp over-extension
                over = int(cur) > d
            except ValueError:              # unparseable: the user's problem, not ours
                continue
            if over:
                _set_keys_locked(sec, {"ctx-size": str(d)}, path)
                changed.append(sec)
        return {"changed": changed}

def ensure_models_ini(path=None, defaults=None):
    """Create models.ini with a [*] global section if it isn't there yet.

    llama-server refuses to start without this file, and on a fresh checkout
    nothing created it: the repo ships no models.ini (it's per-machine), so the
    very first launch failed with a bind-less router and an empty dashboard.
    Runs on every startup and is idempotent - an existing file, even one with no
    [*] section, is left exactly as the user wrote it.

    Returns True if the file was created.
    """
    path = path or ini_path()
    with _INI_LOCK:
        if os.path.exists(path):
            return False
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("; LlamaForge model registry - read by llama-server's router.\n"
                    "; Sections are model ids; keys are llama-server flags.\n"
                    "version = 1\n")
        _set_keys_locked("*", defaults or {"ctx-size": CTX_GLOBAL_DEFAULT}, path)
        return True
