"""Introspect vLLM's server knobs from `vllm serve --help` (run inside WSL).

Same idea as argspec.py for llama.cpp: parse argparse --help into a grouped,
typed schema the existing UI renders, so vLLM's flags stay correct across
upgrades. vLLM's help is argparse-standard: choices as {a,b,c}, defaults as
(default: X), store-true flags have no metavar.
"""
import re

import wsl

RESERVED = {"help", "host", "port", "model", "served-model-name", "api-key",
            "download-dir", "disable-log-requests"}

_FLAG_RE = re.compile(r"^\s{2,}(--[\w\-]+)(.*)$")
# A real metavar is ALL-CAPS (HOST, MAX_MODEL_LEN) or an argparse choice set
# ({auto,half,...}). Description text that follows a bare flag (store_true,
# e.g. "Always use eager-mode PyTorch.") is Titlecase/lowercase and must NOT
# be mistaken for a metavar, or bool flags misclassify as str.
_METAVAR_TOKEN = r"(?:\{[^}]*\}|[A-Z][A-Z0-9_]*)"
_HEAD_RE = re.compile(
    r"^\s+(" + _METAVAR_TOKEN + r")((?:,\s+-[\w\-]+\s+" + _METAVAR_TOKEN + r")*)"
    r"(?:\s{2,}(.*))?$"
)


def _classify(metavar, default):
    """(type, options) from a metavar like MAX_LEN / {a,b,c} / '' (store-true)."""
    m = (metavar or "").strip()
    if not m:
        return "bool", None
    ch = re.match(r"^\{([^}]*)\}$", m)
    if ch:
        opts = [o.strip() for o in ch.group(1).split(",") if o.strip()]
        return "enum", opts
    if re.fullmatch(r"-?\d+", (default or "").strip()):
        return "int", None
    if re.fullmatch(r"-?\d*\.\d+", (default or "").strip()):
        return "float", None
    return "str", None


def parse_help(text):
    items, pending, section = [], None, "general"

    def flush():
        nonlocal pending
        if pending:
            items.append(pending); pending = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if re.match(r"^[A-Za-z].*:\s*$", line) and not line.startswith(" "):
            flush(); section = line.strip().rstrip(":"); continue
        if pending and raw.startswith("      ") and not raw.lstrip().startswith("-"):
            extra = raw.strip()
            pending["desc"] = (pending["desc"] + " " + extra).strip()
            d = re.search(r"\(default:\s*(.*?)\)", pending["desc"])
            if d and not pending.get("default"):
                pending["default"] = d.group(1).strip()
            typ, opts = _classify(pending["_metavar"], pending.get("default", ""))
            pending["type"], pending["options"] = typ, opts
            continue
        m = _FLAG_RE.match(line)
        if not m:
            continue
        flush()
        flag, rest = m.group(1), m.group(2)
        hm = _HEAD_RE.match(rest)
        if hm:
            metavar, desc = hm.group(1).strip(), (hm.group(3) or "").strip()
        else:
            metavar, desc = "", rest.strip()
        dflt = ""
        dm = re.search(r"\(default:\s*(.*?)\)", desc)
        if dm:
            dflt = dm.group(1).strip()
        key = flag[2:]
        typ, opts = _classify(metavar, dflt)
        pending = {
            "key": key, "flags": [flag], "aliases": [key], "section": section,
            "type": typ, "options": opts, "placeholder": metavar,
            "desc": re.sub(r"\s*\(default:.*", "", desc).strip(),
            "default": dflt, "env": "",
            "reserved": key in RESERVED,
            "_metavar": metavar,
        }
    flush()
    for it in items:
        it.pop("_metavar", None)
    return items


def build_schema(distro, venv):
    """Run `vllm serve --help` inside WSL and return grouped editable knobs."""
    code, out, err = wsl.run(f"{venv}/bin/vllm serve --help", distro=distro, timeout=40)
    if code != 0 or not out.strip():
        return {"error": (err or "vllm --help failed").strip()[:300], "groups": [], "count": 0}
    items = [i for i in parse_help(out) if not i["reserved"]]
    groups = {}
    for it in items:
        groups.setdefault(it["section"], []).append(it)
    ordered = [{"name": k, "knobs": v} for k, v in groups.items()]
    return {"groups": ordered, "count": len(items)}
