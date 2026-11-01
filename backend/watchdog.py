"""Bring the llama.cpp router back when it crashes.

Clients talk to the router directly, so when it dies every app pointed at
:8080 just gets "connection refused" until someone notices. The watchdog
probes it every few seconds and restarts it, backing off when it keeps dying,
and stops after a few quick deaths with the reason read from the router log
(diag.py) instead of crash-looping in the background.

It only acts on a router it has seen running (no llama.cpp yet, or a router
the user never started, is left alone), only after several dead probes in a
row (a restart someone else is doing looks dead for a moment), and the
restart callback skips while another lifecycle change holds the lock.

Pure logic: the clock, probe, restart and verdict are passed in, so the tests
drive it without a router.
"""
import socket
import threading
import time


def tcp_probe(port, timeout=5):
    """"up" if something is listening on the router port, else "down". A TCP
    connect, not an HTTP request: the kernel accepts it even while the router
    is busy, and there's no auth to get through. The timeout has to be longer
    than Windows' ~2 s retry before it reports a refused localhost connect."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout).close()
        return "up"
    except OSError:
        return "down"


class Watchdog:
    def __init__(self, enabled, probe, restart, verdict, now=time.monotonic,
                 dead_ticks=3, max_quick=5, stable_s=60, backoff=(0, 4, 8, 16, 32)):
        """enabled() -> bool; probe() -> "up"|"down"; restart() -> (ok, err), or
        None when another router change is under way; verdict() -> {error,
        suggestion} for giving up."""
        self._enabled, self._probe, self._restart, self._verdict = enabled, probe, restart, verdict
        self._now = now
        self.dead_ticks, self.max_quick, self.stable_s, self.backoff = (
            dead_ticks, max_quick, stable_s, tuple(backoff) or (0,))
        self._lock = threading.Lock()
        self._thread = None
        self.armed = False
        self._dead = 0
        self._quick = 0                 # restarts not yet followed by stable_s of uptime
        self._last = None               # monotonic time of the last restart
        self.restarts = 0
        self.last_restart = None        # wall clock, for the UI
        self.last_error = None          # a restart that failed to start
        self.gave_up = False
        self.error = self.suggestion = None

    def tick(self):
        with self._lock:
            if not self._enabled():
                self._dead = 0
                return
            t = self._now()
            if self._probe() == "up":
                self.armed, self._dead = True, 0
                if self.gave_up:            # someone fixed it and started it again
                    self.gave_up, self._quick = False, 0
                    self.error = self.suggestion = None
                if self._quick and self._last is not None and t - self._last >= self.stable_s:
                    self._quick = 0
                return
            if not self.armed or self.gave_up:
                return
            self._dead += 1
            if self._dead < self.dead_ticks:
                return
            if self._quick >= self.max_quick:
                v = self._verdict() or {}
                self.gave_up = True
                self.error = v.get("error") or "The router keeps exiting right after it starts."
                self.suggestion = v.get("suggestion") or (
                    "Its log is at the bottom of Models. Fix the cause, then restart LlamaForge.")
                return
            wait = self.backoff[min(self._quick, len(self.backoff) - 1)]
            if self._last is not None and t - self._last < wait:
                return
            r = self._restart()
            self._dead = 0
            if r is None:                   # another lifecycle change is restarting it
                return
            ok, err = r
            self._quick += 1
            self.restarts += 1
            self._last = t
            self.last_restart = time.time()
            self.last_error = None if ok else (err or "restart failed")

    def safe_tick(self):
        try:
            self.tick()
        except Exception:
            pass

    def status(self):
        with self._lock:
            return {"armed": self.armed, "restarts": self.restarts,
                    "last_restart": self.last_restart, "last_error": self.last_error,
                    "gave_up": self.gave_up, "error": self.error, "suggestion": self.suggestion}

    def start(self, interval=5.0):
        if self._thread:
            return

        def loop():
            while True:
                self.safe_tick()
                time.sleep(interval)
        self._thread = threading.Thread(target=loop, daemon=True, name="router-watchdog")
        self._thread.start()
