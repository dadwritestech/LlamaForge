"""Minimal GGUF metadata reader (pure stdlib).

We only need one number per model: the trained context length, stored in the
GGUF header as `<arch>.context_length` (e.g. `qwen2.context_length`). That lives
in the metadata block at the very start of the file, so we parse just the header
and seek past every value we don't care about - never reading the multi-GB
tensor payload, and never loading giant arrays (like the tokenizer vocab) into
memory. Any malformed/unreadable file degrades to None; this module never raises.
"""
import struct

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
