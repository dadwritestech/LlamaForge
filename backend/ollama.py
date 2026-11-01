"""Find the models Ollama already downloaded, so llama.cpp can run the same files.

Ollama keeps each model as an extension-less blob (blobs/sha256-<hex>) named by
a manifest (manifests/<host>/<namespace>/<name>/<tag>, OCI-style JSON). The
blob behind the "image.model" layer is a plain GGUF; an "image.projector"
layer is the vision projector. A drive scan never sees them (no .gguf suffix),
so this reads the manifests instead and points models.ini at the blobs in
place: nothing is copied, and Ollama keeps working.

Models Ollama converted for its own engine can hold tensors upstream llama.cpp
doesn't know; those fail at load with the router's own error, like any GGUF.
"""
import json, os, re

import gguf

MODEL = "application/vnd.ollama.image.model"
PROJECTOR = "application/vnd.ollama.image.projector"
DEFAULT_HOST, DEFAULT_NS = "registry.ollama.ai", "library"


def store_dirs(env, home):
    """Where Ollama may keep models, most specific first. Pure for tests."""
    dirs = [env["OLLAMA_MODELS"]] if env.get("OLLAMA_MODELS") else []
    dirs.append(os.path.join(home, ".ollama", "models"))
    if os.name != "nt":          # the Linux service runs as its own user
        dirs += ["/usr/share/ollama/.ollama/models", "/var/lib/ollama/.ollama/models"]
    return dirs


def _slug(s):
    s = s.lower().replace("_", "-").replace(" ", "-").replace("/", "-")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9.\-]", "", s)).strip("-")


def _model_id(host, ns, name, tag):
    parts = [] if host == DEFAULT_HOST else [host]
    if not (host == DEFAULT_HOST and ns == DEFAULT_NS):
        parts.append(ns)
    parts.append(name)
    if tag != "latest":
        parts.append(tag)
    return _slug("-".join(parts))


def _blob(root, digest):
    """Path of a GGUF blob for a 'sha256:<hex>' digest, or None."""
    m = re.fullmatch(r"sha256[:\-]([0-9a-f]{64})", digest or "")
    if not m:
        return None
    path = os.path.join(root, "blobs", "sha256-" + m.group(1))
    try:
        with open(path, "rb") as f:
            return path if f.read(4) == b"GGUF" else None
    except OSError:
        return None


def _manifests(root):
    """(host, namespace, name, tag, path) for every manifest under root, sorted."""
    base = os.path.join(root, "manifests")
    out = []
    for dirpath, dirnames, files in os.walk(base):
        dirnames.sort()
        rel = os.path.relpath(dirpath, base).replace("\\", "/").split("/")
        if len(rel) != 3:
            continue
        for tag in sorted(files):
            out.append((*rel, tag, os.path.join(dirpath, tag)))
    return out


def entries(root=None):
    """Scanner-shaped entries ({id, model, mmproj?, embeddings?, gib, ollama})."""
    roots = [root] if root else store_dirs(os.environ, os.path.expanduser("~"))
    out, seen = [], set()
    for r in roots:
        if not os.path.isdir(os.path.join(r, "manifests")):
            continue
        for host, ns, name, tag, path in _manifests(r):
            try:
                with open(path, encoding="utf-8") as f:
                    layers = json.load(f).get("layers") or []
            except (OSError, ValueError, AttributeError):
                continue
            digest = {l.get("mediaType"): l.get("digest") for l in layers if isinstance(l, dict)}
            model = _blob(r, digest.get(MODEL))
            if not model or model in seen:
                continue
            seen.add(model)
            mid = _model_id(host, ns, name, tag)
            short = name if (host, ns) == (DEFAULT_HOST, DEFAULT_NS) else f"{host}/{ns}/{name}"
            e = {"id": mid, "model": model.replace("\\", "/"),
                 "gib": round(os.path.getsize(model) / 1024**3, 2),
                 "ollama": f"{short}:{tag}"}
            proj = _blob(r, digest.get(PROJECTOR))
            if proj:
                e["mmproj"] = proj.replace("\\", "/")
            kind = gguf.retrieval(model)
            if kind == "rerank":
                e["reranking"] = True
            elif kind == "embed" or "embed" in mid:
                e["embeddings"] = True
            out.append(e)
    return out
