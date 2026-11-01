import conftest_paths  # noqa: F401
import unittest

import watchdog


class Rig:
    """A fake router: `up` decides the probe, restart() records the call and
    by default brings it back."""
    def __init__(self, up=True, revive=True, busy=False, ok=True):
        self.up, self.revive, self.busy, self.ok = up, revive, busy, ok
        self.t = 0.0
        self.calls = 0
        self.enabled = True

    def probe(self):
        return "up" if self.up else "down"

    def restart(self):
        if self.busy:
            return None
        self.calls += 1
        if self.revive:
            self.up = True
        return (True, "") if self.ok else (False, "server_bin not found")


def make(rig, **kw):
    return watchdog.Watchdog(lambda: rig.enabled, rig.probe, rig.restart,
                             lambda: {"error": "boom", "suggestion": "fix it"},
                             now=lambda: rig.t, **kw)


def run(w, rig, ticks, step=5.0):
    for _ in range(ticks):
        w.tick()
        rig.t += step


class WatchdogTest(unittest.TestCase):
    def test_never_seen_up_is_not_armed(self):
        # no llama.cpp yet, or the user never started it: nothing to bring back
        rig = Rig(up=False)
        w = make(rig)
        run(w, rig, 10)
        self.assertEqual(rig.calls, 0)
        self.assertFalse(w.status()["armed"])

    def test_restarts_a_crashed_router_after_a_few_dead_probes(self):
        rig = Rig()
        w = make(rig, dead_ticks=3)
        run(w, rig, 2)
        rig.up = False
        run(w, rig, 2)
        self.assertEqual(rig.calls, 0)          # one missed probe isn't a crash
        run(w, rig, 1)
        self.assertEqual(rig.calls, 1)
        s = w.status()
        self.assertEqual(s["restarts"], 1)
        self.assertFalse(s["gave_up"])
        self.assertIsNotNone(s["last_restart"])

    def test_a_blip_does_not_restart(self):
        rig = Rig()
        w = make(rig, dead_ticks=3)
        run(w, rig, 1)
        for up in (False, False, True, False, False, True):
            rig.up = up
            run(w, rig, 1)
        self.assertEqual(rig.calls, 0)

    def test_busy_lifecycle_lock_means_someone_else_is_restarting(self):
        rig = Rig(busy=True)
        w = make(rig, dead_ticks=1)
        run(w, rig, 1)
        rig.up = False
        run(w, rig, 5)
        self.assertEqual(w.status()["restarts"], 0)

    def test_backs_off_then_gives_up_with_the_verdict(self):
        rig = Rig(revive=False)                 # it dies right away every time
        w = make(rig, dead_ticks=1, max_quick=3, backoff=(0, 10, 20))
        run(w, rig, 1)
        rig.up = False
        run(w, rig, 1, step=1)
        self.assertEqual(rig.calls, 1)
        run(w, rig, 9, step=1)                  # 9 s after: still inside the 10 s backoff
        self.assertEqual(rig.calls, 1)
        run(w, rig, 2, step=1)
        self.assertEqual(rig.calls, 2)
        run(w, rig, 30, step=1)
        self.assertEqual(rig.calls, 3)
        run(w, rig, 100, step=1)
        self.assertEqual(rig.calls, 3)          # gave up, no more tries
        s = w.status()
        self.assertTrue(s["gave_up"])
        self.assertEqual((s["error"], s["suggestion"]), ("boom", "fix it"))

    def test_a_failed_start_counts_and_is_reported(self):
        rig = Rig(revive=False, ok=False)
        w = make(rig, dead_ticks=1, max_quick=1)
        run(w, rig, 1)
        rig.up = False
        run(w, rig, 5)
        s = w.status()
        self.assertTrue(s["gave_up"])
        self.assertEqual(s["error"], "boom")
        self.assertEqual(s["last_error"], "server_bin not found")

    def test_staying_up_resets_the_quick_death_count(self):
        rig = Rig()
        w = make(rig, dead_ticks=1, max_quick=2, stable_s=60, backoff=(0,))
        run(w, rig, 1)
        for _ in range(4):                      # crash, come back, stay up 2 min, repeat
            rig.up = False
            run(w, rig, 1)
            run(w, rig, 24)
        self.assertEqual(rig.calls, 4)
        self.assertFalse(w.status()["gave_up"])
        self.assertEqual(w.status()["restarts"], 4)

    def test_coming_back_by_hand_clears_give_up(self):
        rig = Rig(revive=False)
        w = make(rig, dead_ticks=1, max_quick=1)
        run(w, rig, 1)
        rig.up = False
        run(w, rig, 3)
        self.assertTrue(w.status()["gave_up"])
        rig.up = True                           # the user fixed it and restarted
        run(w, rig, 1)
        s = w.status()
        self.assertFalse(s["gave_up"])
        self.assertIsNone(s["error"])

    def test_disabled_does_nothing(self):
        rig = Rig()
        w = make(rig, dead_ticks=1)
        run(w, rig, 1)
        rig.enabled, rig.up = False, False
        run(w, rig, 5)
        self.assertEqual(rig.calls, 0)

    def test_a_probe_or_restart_that_raises_does_not_kill_the_loop(self):
        rig = Rig()
        w = make(rig, dead_ticks=1)
        w._probe = lambda: (_ for _ in ()).throw(OSError("x"))
        w.safe_tick()                           # must not raise


class ProbeTest(unittest.TestCase):
    def test_nothing_listening_is_down(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.assertEqual(watchdog.tcp_probe(port), "down")

    def test_any_http_answer_is_up(self):
        import http.server, threading
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(401); self.end_headers()
            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.handle_request, daemon=True).start()
        try:
            self.assertEqual(watchdog.tcp_probe(srv.server_address[1], timeout=3), "up")
        finally:
            srv.server_close()


class WiringTest(unittest.TestCase):
    def test_restart_skips_while_another_change_holds_the_lock(self):
        import threading
        from unittest import mock
        import routes, router_ctl
        held, release = threading.Event(), threading.Event()
        def hold():
            with routes._ROUTER_LIFECYCLE_LOCK:
                held.set(); release.wait(5)
        th = threading.Thread(target=hold); th.start(); held.wait(5)
        try:
            with mock.patch.object(router_ctl, "start") as start:
                self.assertIsNone(routes._watchdog_restart())
                start.assert_not_called()
        finally:
            release.set(); th.join()

    def test_restart_starts_the_router_with_the_configured_pool(self):
        from unittest import mock
        import routes, router_ctl
        with mock.patch.object(router_ctl, "is_running", return_value=False),              mock.patch.object(router_ctl, "start", return_value=(True, "")) as start,              mock.patch.object(routes, "_router_pool", return_value=None),              mock.patch.object(routes, "cfg", return_value={"router_port": 8197, "server_bin": "x"}):
            self.assertEqual(routes._watchdog_restart(), (True, ""))
        self.assertEqual(start.call_args[0][2], 8197)

    def test_listening_after_all_is_a_skip(self):
        from unittest import mock
        import routes, router_ctl
        with mock.patch.object(router_ctl, "is_running", return_value=True),              mock.patch.object(router_ctl, "start") as start:
            self.assertIsNone(routes._watchdog_restart())
        start.assert_not_called()

    def test_state_reports_it(self):
        import routes
        s = routes.WATCHDOG.status()
        self.assertEqual(set(s), {"armed", "restarts", "last_restart", "last_error",
                                  "gave_up", "error", "suggestion"})

    def test_on_by_default(self):
        import config
        self.assertTrue(config.DEFAULTS["router_watchdog"])


if __name__ == "__main__":
    unittest.main()
