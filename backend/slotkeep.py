"""Keep a model's conversations across a reload: save its slots through the
router before, restore them after. llama.cpp writes the files under the
router's --slot-save-path; we only name them - lf-<sha1(model)[:12]>-<slot>.bin,
which its fs_validate_filename accepts - and delete them once restored. They
hold the whole conversation as KV state, so they never outlive the swap."""
import hashlib
import os
import shutil
import time
import urllib.parse

PREFIX = "lf-"
RESTART_HINT = ("conversation not kept: the router was started without a slot save "
                "folder (it gets one only while Elastic VRAM is on) - restart the router "
                "once from LlamaForge (Setup > Network Access > Apply & Restart Router) "
                "to keep conversations")


def filename(mid, sid):
    return f"{PREFIX}{hashlib.sha1(mid.encode()).hexdigest()[:12]}-{int(sid)}.bin"


def slots(router, mid):
    # autoload=false: reading the slots of a model that's gone must not load it.
    code, data = router("/slots?model=" + urllib.parse.quote(mid, safe="") + "&autoload=false")
    return data if code == 200 and isinstance(data, list) else []


def wait_idle(router, mid, timeout=30, sleep=time.sleep):
    """True once no slot of mid is generating; False after ~timeout seconds."""
    for i in range(max(1, int(timeout))):
        if not any(s.get("is_processing") for s in slots(router, mid)):
            return True
        if i < int(timeout) - 1:
            sleep(1)
    return False


def _err(res, default):
    e = res.get("error") if isinstance(res, dict) else None
    if isinstance(e, dict):
        e = e.get("message")
    return e or default


def save(router, mid, folder, need_mib=0, disk_free=None):
    """([(slot, file)] saved, why the conversation wasn't kept or "")."""
    try:
        free = (disk_free or (lambda p: shutil.disk_usage(p).free))(folder) // 2**20
    except OSError:
        free = None
    if free is not None and free < 2 * need_mib:
        return [], f"conversation not kept: under {2 * need_mib} MiB free on disk"
    out, why = [], ""
    for s in slots(router, mid):
        if not s.get("n_prompt_tokens", 1):
            continue                      # nothing in it
        sid = int(s.get("id", 0))
        fn = filename(mid, sid)
        code, res = router(f"/slots/{sid}?action=save", "POST", {"model": mid, "filename": fn})
        if code == 200 and isinstance(res, dict) and res.get("n_saved"):
            out.append((sid, fn))
        elif code == 200:
            why = "conversation not kept: the router saved nothing"
        elif code == 501:
            why = RESTART_HINT
        elif code != 200:
            why = "conversation not kept: " + _err(res, "the router refused the save")
    return out, why


def restore(router, mid, files):
    """Restore each saved slot that's still empty and idle - a request that
    landed during the swap owns its slot. How many were restored."""
    now = {int(s.get("id", -1)): s for s in slots(router, mid)}
    n = 0
    for sid, fn in files:
        s = now.get(sid)
        if s is None or s.get("is_processing") or s.get("n_prompt_tokens"):
            continue
        code, res = router(f"/slots/{sid}?action=restore", "POST", {"model": mid, "filename": fn})
        if code == 200 and isinstance(res, dict) and res.get("n_restored"):
            n += 1
    return n


def drop(folder, files):
    for _, fn in files:
        try:
            os.remove(os.path.join(folder, fn))
        except OSError:
            pass


def sweep(folder):
    """At startup: delete what a crash mid-swap left behind."""
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    n = 0
    for f in names:
        if f.startswith(PREFIX) and f.endswith(".bin"):
            try:
                os.remove(os.path.join(folder, f))
                n += 1
            except OSError:
                pass
    return n
