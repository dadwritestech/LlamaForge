"""Recommended samplers: the temperature / top-p / top-k a model's makers
publish in generation_config.json, mapped onto llama-server's knobs.

GGUF repos rarely carry that file, so the GGUF repo is tried first and then the
base model its card names. Model-card prose ("we recommend temperature=0.6")
is not parsed: too fuzzy to apply with one click. Pure stdlib; reuses the
template doctor's HF helpers."""
import tmpldoctor

# generation_config.json key -> llama-server knob
KEYS = {"temperature": "temp", "top_p": "top-p", "top_k": "top-k", "min_p": "min-p",
        "repetition_penalty": "repeat-penalty", "presence_penalty": "presence-penalty"}
KNOBS = tuple(KEYS.values())


def _fmt(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return f"{round(float(v), 6):g}"


def from_config(cfg):
    """{knob: "value"} from a generation_config dict. do_sample false means
    greedy, which llama.cpp spells temp 0."""
    if not isinstance(cfg, dict):
        return {}
    if cfg.get("do_sample") is False:
        return {"temp": "0"}
    out = {}
    for k, knob in KEYS.items():
        s = _fmt(cfg.get(k))
        if s is not None:
            out[knob] = s
    return out


def recommend(path):
    """{source, url, values, checked, error} for the GGUF at `path`."""
    repo, _ = tmpldoctor.repo_path(path)
    out = {"source": "", "url": "", "values": {}, "checked": [], "error": ""}
    if not repo:
        out["error"] = "not downloaded from a Hugging Face repo LlamaForge can name"
        return out
    for r in (repo, None):
        r = r or tmpldoctor.base_repo(repo)
        if not r or r in out["checked"]:
            continue
        out["checked"].append(r)
        try:
            vals = from_config(tmpldoctor._json(tmpldoctor._url(r, "generation_config.json")))
        except Exception:
            continue
        if vals:
            out.update(source=r, values=vals,
                       url=f"{tmpldoctor.HF}/{r}/blob/main/generation_config.json")
            return out
    return out
