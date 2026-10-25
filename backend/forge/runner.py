"""Runs one Forge job on a background thread; the panel polls snapshot()."""
import threading


class Runner:
    def __init__(self):
        self._lock = threading.Lock()
        self._job = None
        self._thread = None
        self._crash = None

    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def start(self, job):
        """False when a job is already running."""
        with self._lock:
            if self.running():
                return False
            self._job, self._crash = job, None

            def go():
                try:
                    job.run()
                except Exception as e:      # noqa: BLE001 - Busy, or a bug: shown, not lost
                    self._crash = {"stage": "error", "error": f"{type(e).__name__}: {e}"}

            self._thread = threading.Thread(target=go, name="forge", daemon=True)
            self._thread.start()
            return True

    def cancel(self):
        if not self.running():
            return False
        self._job.cancel.set()
        return True

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def snapshot(self):
        if self._crash and not self.running():
            return self._crash
        return self._job.snapshot() if self._job else None
