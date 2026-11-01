"""Rig card: one postable summary of this machine and what it runs.

GPUs, RAM, OS, LlamaForge version, and the most-used models with the speed
they actually ran at here (llama.cpp's own decode timing, token-weighted by
stats.py), plus how many tokens the box has produced. People post their rigs; this gives them
the numbers and a link, with no hosted database behind it.

Model ids lose any folder part and anything that could open markdown or HTML,
so a card never carries a home path or a username. Pure: routes passes in the
GPU telemetry, RAM, platform, version and stats summary.
"""
import os
import re

TOP = 5
LINK = "github.com/dadwritestech/LlamaForge"
_OS = {"windows": "Windows", "linux": "Linux", "macos": "macOS"}


def _short_gpu(name):
    return re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", str(name or "")).strip() or "GPU"


def _safe(text):
    """The file name only, and nothing that can open a cell, link, code span or tag."""
    base = os.path.basename(str(text or "").replace("\\", "/"))
    return re.sub(r"[^\w .+()-]", "_", base).strip()


def human(n):
    n = int(n or 0)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            return f"{n / div:.1f}{suf}"
    return str(n)


def build(gpus, ram, platform, version, stats):
    cards = []
    for g in gpus or []:
        if not isinstance(g, dict) or g.get("error") or not g.get("name"):
            continue
        gb = round((g.get("total") or 0) / 1024)
        cards.append(f"{_short_gpu(g['name'])} {gb} GB" if gb else _short_gpu(g["name"]))
    stats = stats or {}
    tot = stats.get("totals") or {}
    models = [{"name": _safe(m.get("id")), "tps": round(float(m.get("decode_tps") or 0), 1),
               "generated": int(m.get("generated") or 0)}
              for m in stats.get("per_model") or [] if (m.get("generated") or 0) > 0]
    models.sort(key=lambda m: m["generated"], reverse=True)
    out = {"gpus": cards,
           "ram_gb": round(ram["total"] / 1024) if ram and ram.get("total") else None,
           "os": _OS.get(platform, str(platform or "")),
           "version": str(version or ""),
           "models": models[:TOP],
           "generated": int(tot.get("generated") or 0),
           "hours": int(round(float(tot.get("loaded_hours") or 0))),
           "link": LINK}
    out["markdown"] = _markdown(out)
    return out


def hardware_line(d):
    parts = [" + ".join(d["gpus"]) or "CPU only"]
    if d.get("ram_gb"):
        parts.append(f"{d['ram_gb']} GB RAM")
    if d.get("os"):
        parts.append(d["os"])
    return " · ".join(parts)


def _markdown(d):
    lines = [f"**My local AI rig**: {hardware_line(d)}", ""]
    if d["models"]:
        lines += ["| model | tok/s | tokens generated |", "|---|---|---|"]
        lines += [f"| {m['name']} | {m['tps'] or '-'} | {human(m['generated'])} |" for m in d["models"]]
        lines.append("")
    if d["generated"]:
        lines.append(f"{human(d['generated'])} tokens generated over {d['hours']} hours of loaded models.")
        lines.append("")
    lines.append(f"Run with LlamaForge {d['version']} on llama.cpp: https://{LINK}")
    return "\n".join(lines)
