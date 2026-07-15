"""Optional system-tray icon for LlamaForge.

This is the one piece of LlamaForge that isn't pure stdlib: it needs `pystray`
and `Pillow`. Both are imported lazily and every entry point degrades to a
harmless no-op when they're absent, so the default install stays stdlib-only and
nothing here can break the dashboard. Enable it with `pip install pystray pillow`.

The tray shows the loaded-model count in its tooltip, refreshes it on a timer,
and offers Open dashboard / Quit from the menu.
"""
import threading
import webbrowser

try:                                    # optional deps - absence is fine
    import pystray
    from PIL import Image, ImageDraw
    _DEPS = True
except Exception:                       # ImportError, or a broken partial install
    _DEPS = False


def available():
    return _DEPS


def _icon_image(loaded):
    """A 64x64 amber-on-dark forge glyph; a green dot when a model is loaded."""
    img = Image.new("RGBA", (64, 64), (8, 10, 11, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([7, 7, 57, 57], outline=(255, 176, 0, 255), width=3)
    for y in (20, 30, 40):              # three "forge bars"
        d.rectangle([18, y, 46, y + 4], fill=(255, 176, 0, 255))
    if loaded:
        d.ellipse([44, 44, 56, 56], fill=(57, 217, 138, 255))
    return img


def start(panel_port, counts_fn, refresh_secs=5):
    """Start the tray in a background thread. `counts_fn` returns (loaded,total).
    Returns the pystray Icon, or None when deps are missing / startup fails."""
    if not _DEPS:
        return None
    url = f"http://127.0.0.1:{panel_port}/"

    def _title(loaded, total):
        return f"LlamaForge - {loaded}/{total} loaded"

    try:
        loaded, total = counts_fn()
    except Exception:
        loaded, total = 0, 0

    icon = pystray.Icon(
        "llamaforge", _icon_image(loaded), _title(loaded, total),
        menu=pystray.Menu(
            pystray.MenuItem("Open dashboard", lambda *_: webbrowser.open(url),
                             default=True),
            pystray.MenuItem("Quit", lambda ic, *_: ic.stop()),
        ))

    def _refresh():
        while True:
            threading.Event().wait(refresh_secs)
            try:
                loaded, total = counts_fn()
                icon.icon = _icon_image(loaded)
                icon.title = _title(loaded, total)
            except Exception:
                pass

    try:
        threading.Thread(target=_refresh, daemon=True, name="tray-refresh").start()
        threading.Thread(target=icon.run, daemon=True, name="tray").start()
        return icon
    except Exception:
        return None
