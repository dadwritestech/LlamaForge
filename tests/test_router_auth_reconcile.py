"""Upgrading from a build that ran the router unkeyed leaves that router
running: run.ps1/run.sh only start one when the port is free. The backend
checks on startup and restarts a router that answers without the key (or
rejects the one LlamaForge would send)."""
import conftest_paths  # noqa: F401
import io, unittest, urllib.error
from unittest import mock

import router_ctl, routes


def _http_error(code):
    return urllib.error.HTTPError("http://x/props", code, "x", {}, io.BytesIO(b""))


class AuthStateTest(unittest.TestCase):
    def _state(self, outcomes, key="k" * 43):
        calls = []

        def fake_urlopen(req, timeout=0):
            calls.append(req.get_header("Authorization"))
            out = outcomes[len(calls) - 1]
            if isinstance(out, Exception):
                raise out
            resp = mock.MagicMock()
            resp.__enter__.return_value.status = out
            return resp

        with mock.patch.object(router_ctl.urllib.request, "urlopen", fake_urlopen):
            return router_ctl.auth_state(8080, key), calls

    def test_open_router(self):
        state, calls = self._state([200])
        self.assertEqual(state, "open")
        self.assertEqual(calls, [None])

    def test_keyed_router_accepting_our_key(self):
        state, calls = self._state([_http_error(401), 200])
        self.assertEqual(state, "ok")
        self.assertEqual(calls[1], "Bearer " + "k" * 43)

    def test_keyed_router_rejecting_our_key(self):
        state, _ = self._state([_http_error(401), _http_error(401)])
        self.assertEqual(state, "mismatch")

    def test_unreachable_or_odd_router_is_unknown(self):
        self.assertEqual(self._state([OSError("down")])[0], "unknown")
        self.assertEqual(self._state([_http_error(501)])[0], "unknown")


class ReconcileTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {"router_port": 8080, "router_host": "127.0.0.1",
                    "router_api_key": "", "router_local_key": "L" * 43,
                    "server_bin": "/bin/llama-server", "active_engine": "llamacpp"}
        mock.patch.object(routes, "cfg", side_effect=lambda: dict(self.cfg)).start()
        mock.patch.object(routes.config, "ini_path", return_value="/tmp/models.ini").start()
        mock.patch.object(routes.os.path, "exists", return_value=True).start()
        self.restart = mock.patch.object(routes.router_ctl, "restart",
                                         return_value=(True, "")).start()
        self.addCleanup(mock.patch.stopall)

    def _run(self, state):
        with mock.patch.object(routes.router_ctl, "auth_state", return_value=state) as probe:
            out = routes.reconcile_router_auth()
        return out, probe

    def test_open_router_is_restarted_keyed(self):
        out, probe = self._run("open")
        probe.assert_called_once_with(8080, "L" * 43)
        self.restart.assert_called_once()
        args = self.restart.call_args.args
        self.assertEqual(args[4], "")          # user key decides policy
        self.assertEqual(args[6], "L" * 43)    # local key reaches argv
        self.assertTrue(out)

    def test_mismatched_router_is_restarted(self):
        self._run("mismatch")
        self.restart.assert_called_once()

    def test_healthy_or_unknown_router_is_left_alone(self):
        for state in ("ok", "unknown"):
            out, _ = self._run(state)
            self.assertFalse(out)
        self.restart.assert_not_called()

    def test_no_binary_means_no_restart(self):
        self.cfg["server_bin"] = ""
        self._run("open")
        self.restart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
