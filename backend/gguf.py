"""Minimal GGUF metadata reader (pure stdlib).

We only need one number per model: the trained context length, stored in the
GGUF header as `<arch>.context_length` (e.g. `qwen2.context_length`). That lives
in the metadata block at the very start of the file, so we parse just the header
and seek past every value we don't care about - never reading the multi-GB
tensor payload, and never loading giant arrays (like the tokenizer vocab) into
memory. Any malformed/unreadable file degrades to None; this module never raises.
"""
import os
import re
import struct

_SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)


def total_size(path):
    """Bytes on disk for the whole model. A split GGUF is registered by its
    first shard (foo-00001-of-00003.gguf), so sum every sibling shard that is
    present. The one function here that raises: OSError if `path` is gone,
    same as os.path.getsize, which it replaces."""
    size = os.path.getsize(path)
    m = _SHARD.match(os.path.basename(path))
    if not m:
        return size
    stem, n = m.group(1), int(m.group(3))
    d = os.path.dirname(path)
    for i in range(1, n + 1):
        sib = os.path.join(d, f"{stem}-{i:05d}-of-{n:05d}.gguf")
        if os.path.normcase(sib) != os.path.normcase(path) and os.path.exists(sib):
            size += os.path.getsize(sib)
    return size


# Context-size policy tiers (see default_ctx).
CTX_FULL     = 150000   # baseline default for models that support it
CTX_FALLBACK = 100000   # cap for models trained below CTX_FULL

# GGUF metadata value types (ggml gguf spec).
_SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_SCALAR_SZ  = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_INT_TYPES  = (0, 1, 2, 3, 4, 5, 10, 11)   # types context_length may be stored as
_STRING = 8
_ARRAY  = 9


def _rd(f, n):
    b = f.read(n)
    if len(b) < n:
        raise EOFError
    return b

def _u32(f): return struct.unpack("<I", _rd(f, 4))[0]
def _u64(f): return struct.unpack("<Q", _rd(f, 8))[0]

def _read_str(f):
    n = _u64(f)
    if n > (1 << 20):            # 1 MiB key/string ceiling = corrupt/hostile file
        raise ValueError("string too long")
    return _rd(f, n).decode("utf-8", "replace")

def _skip_str(f):
    f.seek(_u64(f), 1)

def _read_scalar(f, t):
    return struct.unpack(_SCALAR_FMT[t], _rd(f, _SCALAR_SZ[t]))[0]

def _skip_value(f, t):
    """Advance the cursor past a value of type t without materializing it."""
    if t in _SCALAR_SZ:
        f.seek(_SCALAR_SZ[t], 1)
    elif t == _STRING:
        _skip_str(f)
    elif t == _ARRAY:
        et, cnt = _u32(f), _u64(f)
        if et == _STRING:
            for _ in range(cnt):
                _skip_str(f)
        elif et in _SCALAR_SZ:
            f.seek(_SCALAR_SZ[et] * cnt, 1)
        else:                    # arrays of arrays aren't used in practice
            raise ValueError("unsupported array element type %d" % et)
    else:
        raise ValueError("unsupported value type %d" % t)


def context_length(path):
    """Trained context length from a GGUF file, or None if unreadable/unknown."""
    try:
        with open(path, "rb") as f:
            if _rd(f, 4) != b"GGUF":
                return None
            if _u32(f) < 2:              # v1 used 32-bit counts; too old to bother
                return None
            _u64(f)                       # tensor count (unused)
            n_kv = _u64(f)
            if n_kv > 1_000_000:          # sanity guard against a corrupt count
                return None
            arch, ctx = None, {}
            for _ in range(n_kv):
                key = _read_str(f)
                vt  = _u32(f)
                if key == "general.architecture" and vt == _STRING:
                    arch = _read_str(f)
                elif key.endswith(".context_length") and vt in _INT_TYPES:
                    ctx[key] = int(_read_scalar(f, vt))
                else:
                    _skip_value(f, vt)
            if arch and f"{arch}.context_length" in ctx:
                return ctx[f"{arch}.context_length"]
            return next(iter(ctx.values())) if ctx else None
    except Exception:
        return None


def has_nextn(path):
    """True if this GGUF declares MTP/NextN layers (`*.nextn_predict_layers` > 0).

    llama.cpp gates draft-mtp speculative decoding on NextN layers being present
    in the model, not on the arch name, so this file-intrinsic signal is what
    decides whether an mtp-* sidecar can actually be enabled. The curated
    metadata() drops this key, hence a dedicated raw-KV walk. Never raises.
    """
    try:
        with open(path, "rb") as f:
            if _rd(f, 4) != b"GGUF":
                return False
            if _u32(f) < 2:
                return False
            _u64(f)                       # tensor count (unused)
            n_kv = _u64(f)
            if n_kv > 1_000_000:
                return False
            for _ in range(n_kv):
                key = _read_str(f)
                vt  = _u32(f)
                if key.endswith(".nextn_predict_layers") and vt in _INT_TYPES:
                    return int(_read_scalar(f, vt)) > 0
                _skip_value(f, vt)
            return False
    except Exception:
        return False


# general.file_type is an LLAMA_FTYPE enum; map the common quant tiers to labels.
_FTYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}

# Strings worth decoding by exact key; every other string (chat_template, the
# tokenizer vocab, etc.) is skipped so we never pull KBs/MBs off disk.
_META_STR_KEYS = {"general.architecture", "general.name", "general.size_label",
                  "general.basename", "general.finetune"}
_META_STR_SUFFIX = (".rope.scaling.type",)   # e.g. llama.rope.scaling.type = "yarn"


def _read_header_kv(f, n_kv, arrays=(), strings=()):
    """Collect scalar KVs (all cheap) + a small allowlist of strings into a dict.
    Numeric arrays whose key ends with one of `arrays` (per-layer facts, one entry
    per block) are kept; every other array and unlisted string is seeked past."""
    kv = {}
    for _ in range(n_kv):
        key = _read_str(f)
        vt = _u32(f)
        if vt in _SCALAR_SZ:                 # int/float/bool: 1-8 bytes, always keep
            kv[key] = _read_scalar(f, vt)
        elif vt == _STRING:
            if key in _META_STR_KEYS or key in strings or key.endswith(_META_STR_SUFFIX):
                kv[key] = _read_str(f)
            else:
                _skip_str(f)
        elif vt == _ARRAY and arrays and key.endswith(arrays):
            et, cnt = _u32(f), _u64(f)
            if et in _SCALAR_SZ and cnt <= 4096:
                kv[key] = [_read_scalar(f, et) for _ in range(cnt)]
            else:
                f.seek(-12, 1)
                _skip_value(f, vt)
        else:
            _skip_value(f, vt)
    return kv


def metadata(path):
    """Human-facing GGUF header facts for the model card, or None if unreadable.
    Only keys actually present are returned (missing fields are omitted)."""
    try:
        with open(path, "rb") as f:
            if _rd(f, 4) != b"GGUF":
                return None
            if _u32(f) < 2:
                return None
            _u64(f)                            # tensor count (unused)
            n_kv = _u64(f)
            if n_kv > 1_000_000:
                return None
            kv = _read_header_kv(f, n_kv, strings=("tokenizer.chat_template",))
    except Exception:
        return None
    arch = kv.get("general.architecture")
    def a(suffix):
        return kv.get(f"{arch}.{suffix}") if arch else None
    ft = kv.get("general.file_type")
    out = {
        "architecture":     arch,
        "name":             kv.get("general.name"),
        "size_label":       kv.get("general.size_label"),
        "quantization":     _FTYPE.get(int(ft)) if isinstance(ft, (int, float)) else None,
        "context_length":   a("context_length"),
        "embedding_length": a("embedding_length"),
        "block_count":      a("block_count"),
        "head_count":       a("attention.head_count"),
        "vocab_size":       a("vocab_size"),
        "expert_count":     a("expert_count"),
        "expert_used_count": a("expert_used_count"),
        "rope_freq_base":   a("rope.freq_base"),
        "rope_scaling":     a("rope.scaling.type"),
        "thinking":         thinking_kind(kv.get("tokenizer.chat_template")),
        "effort_levels":    effort_levels(kv.get("tokenizer.chat_template")),
        "retrieval":        _retrieval_kind(kv),
    }
    return {k: v for k, v in out.items() if v is not None}


# Architectures that only ever encode (no generation head): always embedders.
_ENCODER_ARCHS = {"bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3",
                  "neo-bert", "modern-bert", "eurobert", "t5encoder", "gemma-embedding"}


def _retrieval_kind(kv):
    """'rerank', 'embed' or None from {arch}.pooling_type (llama.h: 1 mean, 2 cls,
    3 last, 4 rank) and the architecture. Chat models carry no pooling type."""
    arch = kv.get("general.architecture")
    pt = kv.get(f"{arch}.pooling_type") if arch else None
    if pt == 4:
        return "rerank"
    if pt in (1, 2, 3) or arch in _ENCODER_ARCHS:
        return "embed"
    return None


def _has_cls_head(f, n_tensors):
    """Tensor infos follow the KVs: a classifier head (cls.weight /
    cls.output.weight) is how llama.cpp tells a reranker from an embedder in
    conversions that predate pooling_type (gpustack's jina-reranker GGUFs)."""
    for _ in range(min(n_tensors, 100_000)):
        name = _read_str(f)
        n_dims = _u32(f)
        f.seek(8 * n_dims + 4 + 8, 1)       # dims, type, offset
        if name in ("cls.weight", "cls.output.weight"):
            return True
    return False


def retrieval(path):
    """_retrieval_kind for a file, reading only scalars (no chat template)."""
    try:
        with open(path, "rb") as f:
            if _rd(f, 4) != b"GGUF" or _u32(f) < 2:
                return None
            n_tensors = _u64(f)
            n_kv = _u64(f)
            if n_kv > 1_000_000:
                return None
            kv = _read_header_kv(f, n_kv)
            kind = _retrieval_kind(kv)
            if (kind == "embed" and kv.get("general.architecture") in _ENCODER_ARCHS
                    and _has_cls_head(f, n_tensors)):
                return "rerank"
            return kind
    except Exception:
        return None


# Template variables llama.cpp's --reasoning / --reasoning-effort feed: a template
# that reads one can really turn thinking off (or down).
_THINK_SWITCHES = ("enable_thinking", "reasoning_effort", "thinking_mode", "reasoning_mode")


def thinking_kind(template):
    """'switch' when the chat template has a thinking toggle, 'tags' when it only
    handles <think> blocks (Off may not stop a model trained to always think),
    None for a template with no sign of reasoning. Advice, not a guarantee."""
    if not isinstance(template, str) or not template:
        return None
    if any(w in template for w in _THINK_SWITCHES):
        return "switch"
    if any(t in template for t in ("<think>", "[THINK]", "reasoning_content")):
        return "tags"
    return None


# llama.cpp's --reasoning-effort levels, lowest to highest (its capability probe's order).
EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")
_EFFORT_ALIAS = re.compile(r"""==\s*['"](\w+)['"]\s*-?%\}\s*\{%-?\s*set\s+\w+\s*=\s*['"](\w+)['"]""")


def effort_levels(template):
    """The --reasoning-effort levels a chat template names as quoted literals,
    lowest to highest; every level when it names fewer than two (the value is
    inserted as is, like gpt-oss's "Reasoning: {{ reasoning_effort }}"). None
    when the template never reads reasoning_effort. llama.cpp renders the
    template to know for sure; this reads the text, so it is advice."""
    if not isinstance(template, str) or "reasoning_effort" not in template:
        return None
    named = [lv for lv in EFFORT_LEVELS if f"'{lv}'" in template or f'"{lv}"' in template]
    # `== 'high' %}{% set e = 'xhigh'` renames a level: the alias isn't a level of its own
    aliases = {a for a, b in _EFFORT_ALIAS.findall(template) if a != b and b in EFFORT_LEVELS}
    named = [lv for lv in named if lv not in aliases]
    return named if len(named) >= 2 else list(EFFORT_LEVELS)


def projector_info(path):
    """{"projection_dim": n} for an mmproj (clip) file, or None if the file is
    unreadable or not a projector. projection_dim is the text-model width the
    projector feeds; llama.cpp refuses a pair whose widths differ. Vision wins
    over audio when both are present (they're equal in practice)."""
    try:
        with open(path, "rb") as f:
            if _rd(f, 4) != b"GGUF" or _u32(f) < 2:
                return None
            _u64(f)
            n_kv = _u64(f)
            if n_kv > 1_000_000:
                return None
            kv = _read_header_kv(f, n_kv)
    except Exception:
        return None
    if kv.get("general.architecture") != "clip":
        return None
    dim = kv.get("clip.vision.projection_dim") or kv.get("clip.audio.projection_dim")
    return {"projection_dim": int(dim)} if dim else {}


# Per-architecture header facts the memory footprint depends on (see footprint.py).
_LAYOUT_KEYS = ("block_count", "embedding_length", "context_length", "expert_count",
                "attention.head_count", "attention.head_count_kv", "attention.key_length",
                "attention.value_length", "attention.key_length_swa",
                "attention.value_length_swa", "attention.sliding_window",
                "attention.sliding_window_pattern", "attention.shared_kv_layers",
                "full_attention_interval", "nextn_predict_layers")
_LAYOUT_ARRAYS = (".attention.head_count_kv", ".attention.sliding_window_pattern")


def _layout_one(path):
    """(arch, kv, [(name, type_id, nbytes)]) for one GGUF file. A tensor's size is
    the gap to the next tensor's offset (the last one runs to the end of the
    file), so no per-type block-size table is needed and quant types this module
    has never heard of (ik_llama's IQ*_K) size correctly."""
    with open(path, "rb") as f:
        return _layout_file(f, os.fstat(f.fileno()).st_size)


def _layout_file(f, size):
    """_layout_one over an open stream holding at least the header, with the
    whole file's size given (a remote quant read through an HTTP Range)."""
    if _rd(f, 4) != b"GGUF" or _u32(f) < 2:
        return None
    n_t, n_kv = _u64(f), _u64(f)
    if n_kv > 1_000_000 or n_t > 1_000_000:
        return None
    kv = _read_header_kv(f, n_kv, _LAYOUT_ARRAYS)
    infos = []
    for _ in range(n_t):
        name = _read_str(f)
        n_dims = _u32(f)
        if n_dims > 8:
            return None
        f.seek(8 * n_dims, 1)
        infos.append((name, _u32(f), _u64(f)))
    align = kv.get("general.alignment") or 32
    data = f.tell() + (-f.tell()) % align
    end = size - data
    order = sorted(range(len(infos)), key=lambda i: infos[i][2])
    size = {}
    for j, i in enumerate(order):
        nxt = infos[order[j + 1]][2] if j + 1 < len(order) else end
        size[i] = max(nxt - infos[i][2], 0)
    return (kv.get("general.architecture"), kv,
            [(n, t, size[i]) for i, (n, t, _) in enumerate(infos)])


def layout(path):
    """Every tensor (name, type id, bytes) across all shards of a model, plus the
    header facts the memory footprint depends on (keys without the arch prefix):
    {"arch", "kv", "tensors"}. None if any shard is unreadable."""
    try:
        paths = [path]
        m = _SHARD.match(os.path.basename(path))
        if m:
            stem, n = m.group(1), int(m.group(3))
            d = os.path.dirname(path)
            paths = [os.path.join(d, f"{stem}-{i:05d}-of-{n:05d}.gguf")
                     for i in range(1, n + 1)]
        return _merge([_layout_one(p) for p in paths])
    except Exception:
        return None


def layout_streams(parts):
    """layout() over [(stream, file size)], one per shard in order: the headers
    of a quant that isn't on this disk (tools/canirun_fetch.py)."""
    try:
        return _merge([_layout_file(f, size) for f, size in parts])
    except Exception:
        return None


def _merge(got_all):
    if not got_all or any(g is None for g in got_all):
        return None
    arch, kv, tensors = None, {}, []
    for got in got_all:
        arch = arch or got[0]
        kv = kv or got[1]
        tensors += got[2]
    facts = {}
    for k in _LAYOUT_KEYS:
        v = kv.get(f"{arch}.{k}") if arch else None
        if v is not None:
            facts[k.split(".")[-1]] = v
    return {"arch": arch, "kv": facts, "tensors": tensors}


def default_ctx(path):
    """Per-model ctx-size override for a GGUF at `path`.

    Returns:
        int   - write this exact value as the model's ctx-size
        0     - no override needed; model supports the CTX_FULL global default
        None  - trained length unknown (unreadable/missing); leave the model as-is
    """
    n = context_length(path)
    if not n or n <= 0:
        return None
    if n >= CTX_FULL:
        return 0
    return min(CTX_FALLBACK, n)   # cap at trained length; never over-extend
