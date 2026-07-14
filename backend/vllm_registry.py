"""Registry for vLLM models — the vLLM counterpart of llama.cpp's models.ini.

Kept as its own JSON file because models.ini is owned by llama.cpp's router,
which treats every section as a GGUF preset. Shape:

    {"*": {global default knobs},
     "<model-id>": {repo, wsl_path, size_bytes, quant, settings{}}}

Model id defaults to the HF repo id (org/name), which is also the name vLLM
reports on its OpenAI API. Pure stdlib.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "vllm_models.json")


def _path(path):
    return path or DEFAULT_PATH


def load(path=None):
    p = _path(path)
    if not os.path.exists(p):
        return {"*": {}}
    try:
        with open(p, encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception:
        return {"*": {}}
    data.setdefault("*", {})
    return data


def save(data, path=None):
    with open(_path(path), "w", encoding="utf-8", newline="") as f:
        json.dump(data, f, indent=2)
    return data


def upsert(model_id, fields, path=None):
    """Create or update a model's metadata (repo/wsl_path/size_bytes/quant).
    Preserves any existing settings; ensures a settings dict exists."""
    data = load(path)
    entry = data.get(model_id, {})
    entry.update(fields)
    entry.setdefault("settings", {})
    data[model_id] = entry
    return save(data, path)


def set_settings(model_id, updates, path=None):
    """Merge knob updates into a model's settings (or the global '*').
    A blank/None value removes the key."""
    data = load(path)
    if model_id == "*":
        target = data["*"]
    else:
        target = data.setdefault(model_id, {"repo": model_id}).setdefault("settings", {})
    for k, v in updates.items():
        v = ("" if v is None else str(v)).strip()
        if v == "":
            target.pop(k, None)
        else:
            target[k] = v
    return save(data, path)


def remove(model_id, path=None):
    data = load(path)
    data.pop(model_id, None)
    return save(data, path)


def effective_settings(model_id, path=None):
    """Global '*' defaults overlaid with the model's own settings."""
    data = load(path)
    eff = dict(data.get("*", {}))
    eff.update(data.get(model_id, {}).get("settings", {}))
    return eff


def models(path=None):
    """All model ids (excludes the '*' globals bucket)."""
    return [k for k in load(path) if k != "*"]
