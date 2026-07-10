"""Introspect every llama.cpp server argument from `llama-server --help`.

Produces a categorized, typed schema that the UI renders as "all knobs".
Because it is generated from the binary, it stays correct across llama.cpp
versions automatically.
"""
import re, subprocess

# args the router owns / that must not be set per-model in the panel
RESERVED = {
    "help", "usage", "version", "completion-bash", "cache-list",
    "host", "port", "api-key", "api-key-file", "alias", "model", "mmproj",
    "hf-repo", "hf-repo-draft", "hf-repo-v", "hf-file", "hf-token",
    "models-dir", "models-preset", "models-max", "models-autoload",
    "no-models-autoload", "ssl-key-file", "ssl-cert-file", "path",
}

SECTION_RE = re.compile(r"^-+\s*(.+?)\s*-+\s*$")

def _balance_parens(s):
    """Drop orphan ')' left over after trimming a '(default:/env:...)' tail.
    Upstream --help text (and our own truncation) can leave a dangling ')'
    with no matching '(' - it shows up as a stray ')' in the UI, so strip it."""
    out, depth = [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue            # orphan close paren -> drop
            depth -= 1
        out.append(ch)
    return "".join(out).strip()

# curated types/options for common knobs (help text lacks enum values for these)
OVERRIDES = {
    "cache-type-k": ("enum", ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]),
    "cache-type-v": ("enum", ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]),
    "spec-type":    ("enum", ["none", "draft-simple", "draft-eagle3", "draft-mtp",
                               "ngram-simple", "ngram-map-k", "ngram-map-k4v",
                               "ngram-mod", "ngram-cache"]),
    "tensor-split": ("str", None),
    "override-tensor": ("str", None),
    "cpu-range":    ("str", None),
    "cpu-range-batch": ("str", None),
}

def _classify(placeholder, default):
    """Return (type, options) for a value placeholder."""
    p = (placeholder or "").strip()
    if not p:
        return "bool", None
    # enum: bracketed [a|b|c] / <0|1> / {none,mean,cls}  OR bare word list a,b,c
    m = re.search(r"[\[<{]([^\]>}]*[|,][^\]>}]*)[\]>}]", p)
    body = m.group(1) if m else (p if ("," in p or "|" in p) else "")
    if body and "..." not in body:
        opts = [o.strip() for o in re.split(r"[|,]", body) if o.strip()]
        # numeric placeholder list (N0,N1,...) is a free string, not an enum
        if opts and not any(re.match(r"^[NM]\d*$", o) for o in opts):
            return "enum", opts
    if re.fullmatch(r"[NM]", p) or re.search(r"<[\d.\s\-]+\.\.\.?[\d.\s]*>", p):
        return ("float" if re.search(r"\d\.\d", default or "") else "int"), None
    if p in ("FNAME", "PATH", "FILE"):
        return "path", None
    return "str", None

def parse_help(text):
    section = "general"
    items, pending = [], None

    def flush():
        nonlocal pending
        if pending:
            items.append(pending); pending = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        sm = SECTION_RE.match(line.strip())
        if sm and "params" in sm.group(1).lower() or (sm and len(sm.group(1)) < 40 and not line.startswith(" ")):
            flush(); section = sm.group(1).strip(); continue

        # continuation line (indented, no flag) -> append to current desc
        if pending and (raw.startswith("   ") and not raw.lstrip().startswith("-")):
            extra = raw.strip()
            env = re.search(r"\(env:\s*([A-Z0-9_]+)\)", extra)
            if env: pending["env"] = env.group(1)
            dflt = re.search(r"\(default:\s*(.*?)\)", extra)
            if dflt and not pending.get("default"): pending["default"] = dflt.group(1)
            continue

        if not line.lstrip().startswith("-"):
            continue
        # column-aligned help: split on runs of 2+ spaces
        parts = re.split(r"\s{2,}", line.strip())
        flag_parts, desc_parts = [], []
        for p in parts:
            if p.startswith("-") and not desc_parts:
                flag_parts.append(p)
            else:
                desc_parts.append(p)
        if not flag_parts:
            continue
        flush()
        flags, placeholder = [], ""
        for fp in flag_parts:
            toks = fp.split()
            j = 0
            while j < len(toks) and toks[j].rstrip(",").startswith("-"):
                flags.append(toks[j].rstrip(",")); j += 1
            if j < len(toks):
                placeholder = " ".join(toks[j:])
        longs = [f[2:] for f in flags if f.startswith("--")]
        if not longs:
            continue
        key = longs[0]                        # ini key = first long flag w/o --
        desc = "  ".join(desc_parts)
        env = re.search(r"\(env:\s*([A-Z0-9_]+)\)", desc)
        dflt = re.search(r"\(default:\s*(.*?)\)", desc)
        default_val = dflt.group(1) if dflt else ""
        typ, opts = OVERRIDES.get(key, _classify(placeholder, default_val))
        # canonical default: the value before any ", explanation" tail
        clean_default = re.split(r",\s", default_val)[0].strip() if default_val else ""
        pending = {
            "key": key, "flags": flags, "aliases": longs, "section": section,
            "type": typ, "options": opts,
            "placeholder": placeholder,
            "desc": _balance_parens(re.sub(r"\s*\((env|default):.*", "", desc)),
            "default": clean_default,
            "env": env.group(1) if env else "",
            "reserved": any(l in RESERVED for l in longs),
        }
    flush()
    return items

def build_schema(server_bin):
    """Run the server's --help and return grouped, editable knobs."""
    try:
        out = subprocess.run([server_bin, "--help"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception as e:
        return {"error": str(e), "groups": []}
    items = [i for i in parse_help(out) if not i["reserved"]]
    groups = {}
    for it in items:
        groups.setdefault(it["section"], []).append(it)
    ordered = [{"name": k, "knobs": v} for k, v in groups.items()]
    return {"groups": ordered, "count": len(items)}

if __name__ == "__main__":
    import json, sys
    print(json.dumps(build_schema(sys.argv[1]), indent=2)[:3000])
