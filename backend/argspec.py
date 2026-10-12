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
# a values line under an option: ik's "types: none, draft, mtp", mainline's
# "allowed values: f32, f16, q8_0". Only a bare list: "types: int, str. example: ..." is prose
VALUES_RE = re.compile(r"(?:types|allowed values):\s*([\w.-]+(?:,\s*[\w.-]+)+)")
VALUES_LEAD_RE = re.compile(r"(?:types|allowed values):")
CONT_INDENT = 20        # flags start by column 9 (ik) or 0 (mainline); descriptions at 34 / 40

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

# curated types/options for knobs whose help can lack them. An enum here yields
# to one the build prints itself: these lists are mainline's, and ik spells some
# values differently (spec-type mtp, not draft-mtp).
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
    # enum: bracketed [a|b|c] / <0|1> / {none,mean,cls} / (auto|on|off)  OR bare word list a,b,c
    m = re.search(r"[\[<{(]([^\]>})]*[|,][^\]>})]*)[\]>})]", p)
    body = m.group(1) if m else (p if ("," in p or "|" in p) else "")
    if body and ".." not in body:
        opts = [o.strip() for o in re.split(r"[|,]", body) if o.strip()]
        # a numbered list's shape (N0,N1 / dev1,dev2) is a free string, not an enum
        numbered = all(re.search(r"[A-Za-z]\d+$", o) for o in opts) and \
            len({re.sub(r"\d+$", "", o) for o in opts}) == 1
        if opts and not numbered and not any(re.match(r"^[NM]\d*$", o) for o in opts):
            return "enum", opts
    if re.fullmatch(r"[NM]", p) or re.search(r"<[\d.\s\-]+\.\.\.?[\d.\s]*>", p):
        return ("float" if re.search(r"\d\.\d", default or "") else "int"), None
    if p in ("FNAME", "PATH", "FILE"):
        return "path", None
    return "str", None

def parse_help(text):
    section = "general"
    items, pending = [], None

    values_open = None  # an "allowed values:" list that continues on the next line

    def flush():
        nonlocal pending, values_open
        values_open = None
        if pending:
            items.append(pending); pending = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        sm = SECTION_RE.match(line.strip())
        if sm and "params" in sm.group(1).lower() or (sm and len(sm.group(1)) < 40 and not line.startswith(" ")):
            flush(); section = sm.group(1).strip(); continue

        # continuation line (indented, no flag) -> append to current desc. Deep in
        # the description column it's one even when it starts with a flag (ik's
        # "examples: --spec-type mtp:..." lines, mainline's wrapped "--hf-repo")
        indent = len(raw) - len(raw.lstrip())
        if pending and raw.startswith("   ") and (indent >= CONT_INDENT or not raw.lstrip().startswith("-")):
            extra = raw.strip()
            env = re.search(r"\(env:\s*([A-Z0-9_]+)\)", extra)
            if env: pending["env"] = env.group(1)
            dflt = re.search(r"\(default:\s*(.*?)\)", extra)
            if dflt and not pending.get("default"): pending["default"] = dflt.group(1)
            # A long list wraps: "allowed values: a, b," then "c, d" on the next line.
            text = extra if values_open is None else values_open + " " + extra
            if VALUES_LEAD_RE.match(text) and text.endswith(","):
                values_open = text
                continue
            values_open = None
            vals = VALUES_RE.fullmatch(text)
            if vals and pending["values"] is None and pending["type"] != "bool":
                pending["values"] = [v.strip() for v in vals.group(1).split(",")]
                if pending["curated"]:            # the build's own list beats ours
                    pending.update(options=pending["values"], curated=False)
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
                flag = toks[j].rstrip(",")
                plural = re.fullmatch(r"(--?[\w-]+)\((\w+)\)", flag)
                # ik: --embedding(s) is --embedding and --embeddings
                flags += [plural.group(1), plural.group(1) + plural.group(2)] if plural else [flag]
                j += 1
            if j < len(toks):
                placeholder = " ".join(toks[j:])
        longs = [f[2:] for f in flags if f.startswith("--")]
        if not longs:
            continue
        key = longs[0]                        # ini key = first long flag w/o --
        desc = "  ".join(desc_parts)
        glued = not placeholder and re.match(r"([\[<{][^\]>}\s]*[|,][^\]>}\s]*[\]>}])\s*(.*)", desc)
        if glued:                             # ik: `--reasoning   [on|off|auto]Use reasoning...`
            placeholder, desc = glued.group(1), glued.group(2)
        env = re.search(r"\(env:\s*([A-Z0-9_]+)\)", desc)
        dflt = re.search(r"\(default:\s*(.*?)\)", desc)
        default_val = dflt.group(1) if dflt else ""
        typ, opts = _classify(placeholder, default_val)
        cur = OVERRIDES.get(key)
        # ...nor to a value with a payload (ik: SPEC[:k=v,...]), which no select can hold
        curated = bool(cur) and cur[0] == "enum" and typ != "enum" and not re.search(r"[:=]", placeholder)
        if cur and (curated or cur[0] != "enum"):
            typ, opts = cur
        # canonical default: the value before any ", explanation" tail
        clean_default = re.split(r",\s", default_val)[0].strip() if default_val else ""
        pending = {
            "key": key, "flags": flags, "aliases": longs, "section": section,
            "type": typ, "options": opts,
            # what the build itself says it takes (a continuation line can add
            # them); None when it doesn't say, `curated` when options are ours
            "values": opts if typ == "enum" and not curated else None,
            "curated": curated,
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
    if not server_bin:
        return {"error": "server_bin is not set in config.json", "groups": []}
    try:
        # utf-8 explicitly: text=True alone decodes with the locale codepage on
        # Windows (cp1252), which can blow up or mangle upstream help text.
        r = subprocess.run([server_bin, "--help"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=20)
    except Exception as e:
        return {"error": str(e), "groups": []}
    # some builds/forks route usage through the log system -> stderr
    out = r.stdout if r.stdout.strip() else r.stderr
    if not out.strip():
        return {"error": f"`{server_bin} --help` produced no output "
                         f"(exit code {r.returncode}) - missing DLLs or wrong binary?",
                "groups": []}
    items = [i for i in parse_help(out) if not i["reserved"]]
    if not items:
        return {"error": "could not parse any arguments from --help output "
                         "(unrecognized help format?)", "groups": []}
    groups = {}
    for it in items:
        groups.setdefault(it["section"], []).append(it)
    ordered = [{"name": k, "knobs": v} for k, v in groups.items()]
    return {"groups": ordered, "count": len(items)}

if __name__ == "__main__":
    import json, sys
    print(json.dumps(build_schema(sys.argv[1]), indent=2)[:3000])
