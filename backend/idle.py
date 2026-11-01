"""Sleep idle models: llama.cpp's --sleep-idle-seconds, set once for every model.

A model that hasn't served a request for that long frees its VRAM and RAM but
stays in the router's list as "sleeping"; the next request wakes it (a 3B
model came back in 0.65 s here). It lives in models.ini's [*] section, so the
router passes it to each model it starts, and it reaches a model at its next
load. The stats poller only scrapes "loaded" models, so it doesn't keep a
sleeping one awake.
"""
KEY = "sleep-idle-seconds"
MAX_MINUTES = 24 * 60


def minutes(sections):
    """The idle timeout in [*], in whole minutes; 0 when off or unreadable."""
    raw = ((sections or {}).get("*") or {}).get(KEY)
    try:
        s = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return max(1, round(s / 60)) if s > 0 else 0


def updates(mins):
    """The [*] write for `mins` minutes (0 removes the key). ValueError if bad."""
    if isinstance(mins, bool) or not isinstance(mins, int) or not 0 <= mins <= MAX_MINUTES:
        raise ValueError(f"minutes must be a whole number from 0 to {MAX_MINUTES}")
    return {KEY: str(mins * 60) if mins else None}
