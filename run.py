"""LlamaForge TUI launcher.

Shows llama-server and server.py output in a split-pane terminal interface.
Auto-installs textual if not present. Press 'q' to quit (kills both processes).
"""
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser

# ── auto-install textual ──────────────────────────────────────────────────────
def _ensure_textual():
    try:
        import textual  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "textual"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

_ensure_textual()

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, RichLog, Static

# ── config ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

def _load_config():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8-sig") as f:
        return json.load(f)

CFG = _load_config()

SERVER_BIN = CFG.get("server_bin", "")
MODELS_INI = CFG.get("models_ini", "")
ROUTER_PORT = str(CFG.get("router_port", 8080))
PANEL_PORT = str(CFG.get("panel_port", 8090))
ROUTER_HOST = CFG.get("router_host", "127.0.0.1") or "127.0.0.1"
BACKEND_DIR = os.path.join(ROOT, "backend")

# ── port helpers ──────────────────────────────────────────────────────────────

def _port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def _wait_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.3)
    return False

# ── stream reader thread ─────────────────────────────────────────────────────

def _reader(pipe, widget: RichLog, prefix: str):
    """Read lines from a subprocess pipe and write them to a RichLog widget."""
    try:
        for raw in iter(pipe.readline, ""):
            line = raw.rstrip("\n")
            if line:
                widget.write(f"{prefix}{line}")
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass

# ── status bar ────────────────────────────────────────────────────────────────

class StatusBar(Static):
    """Bottom bar showing port status and shortcuts."""
    router_up = reactive(False)
    panel_up = reactive(False)

    def render(self) -> str:
        r = "[green]UP[/green]" if self.router_up else "[red]DOWN[/red]"
        p = "[green]UP[/green]" if self.panel_up else "[red]DOWN[/red]"
        return (
            f"  router:{ROUTER_PORT} {r}  |  dashboard:{PANEL_PORT} {p}"
            f"  |  [bold]q[/bold] quit  [bold]b[/bold] open browser"
        )

# ── TUI app ───────────────────────────────────────────────────────────────────

class LlamaForgeApp(App):
    TITLE = "LlamaForge"
    CSS = """
    Screen {
        layout: vertical;
    }
    #top-pane {
        height: 1fr;
        border: solid $accent;
    }
    #bottom-pane {
        height: 1fr;
        border: solid $secondary;
    }
    #status {
        height: 1;
        dock: bottom;
        background: $surface;
    }
    .pane-title {
        height: 1;
        dock: top;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("b", "open_browser", "Open Browser"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._router_proc: subprocess.Popen | None = None
        self._panel_proc: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="top-pane"):
            yield Static("  [bold]llama-server[/bold]  (router)", classes="pane-title")
            yield RichLog(id="router-log", highlight=True, markup=True, wrap=False)
        with Vertical(id="bottom-pane"):
            yield Static("  [bold]server.py[/bold]  (dashboard)", classes="pane-title")
            yield RichLog(id="panel-log", highlight=True, markup=True, wrap=False)
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(2.0, self._refresh_status)
        # Start processes asynchronously so TUI renders immediately
        self.set_timer(0.1, self._start_router)
        self.set_timer(0.3, self._start_panel)

    # ── process management ────────────────────────────────────────────────────

    def _start_router(self):
        log = self.query_one("#router-log", RichLog)

        if _port_open(int(ROUTER_PORT)):
            log.write(f"[yellow]llama-server already listening on port {ROUTER_PORT}[/yellow]")
            return

        if not SERVER_BIN or not os.path.isfile(SERVER_BIN):
            log.write(f"[red]server_bin not found: {SERVER_BIN}[/red]")
            log.write("[yellow]Open the Build tab in the dashboard to compile llama.cpp first.[/yellow]")
            return

        args = [
            SERVER_BIN,
            "--models-preset", MODELS_INI,
            "--models-max", "1",
            "--offline",
            "--host", ROUTER_HOST,
            "--port", ROUTER_PORT,
            "--metrics",
        ]

        log.write(f"[dim]$ {' '.join(args)}[/dim]")
        try:
            self._router_proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            t = threading.Thread(target=_reader, args=(self._router_proc.stdout, log, ""), daemon=True)
            t.start()
            self._threads.append(t)
            log.write("[green]llama-server started[/green]")
        except Exception as e:
            log.write(f"[red]Failed to start llama-server: {e}[/red]")

    def _start_panel(self):
        log = self.query_one("#panel-log", RichLog)

        if _port_open(int(PANEL_PORT)):
            log.write(f"[yellow]server.py already listening on port {PANEL_PORT}[/yellow]")
            return

        args = [sys.executable, os.path.join(BACKEND_DIR, "server.py")]
        log.write(f"[dim]$ {' '.join(args)}[/dim]")
        try:
            self._panel_proc = subprocess.Popen(
                args,
                cwd=BACKEND_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            t = threading.Thread(target=_reader, args=(self._panel_proc.stdout, log, ""), daemon=True)
            t.start()
            self._threads.append(t)
            log.write("[green]server.py started[/green]")
        except Exception as e:
            log.write(f"[red]Failed to start server.py: {e}[/red]")

        # open browser after a short delay
        def _open():
            time.sleep(3)
            webbrowser.open(f"http://127.0.0.1:{PANEL_PORT}/")

        bt = threading.Thread(target=_open, daemon=True)
        bt.start()

    # ── status refresh ────────────────────────────────────────────────────────

    def _refresh_status(self):
        bar = self.query_one("#status", StatusBar)
        bar.router_up = _port_open(int(ROUTER_PORT))
        bar.panel_up = _port_open(int(PANEL_PORT))

    # ── actions ───────────────────────────────────────────────────────────────

    def action_open_browser(self) -> None:
        webbrowser.open(f"http://127.0.0.1:{PANEL_PORT}/")

    def action_quit(self) -> None:
        self._cleanup()
        self.exit()

    def _cleanup(self):
        """Terminate both subprocesses cleanly."""
        for proc in (self._router_proc, self._panel_proc):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    app = LlamaForgeApp()
    app.run()