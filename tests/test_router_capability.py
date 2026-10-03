"""Whether a server binary can actually be the router.

LlamaForge drives the router as `<server_bin> --models-preset <ini> --models-max 1`.
ik_llama.cpp is a fork of pre-router llama.cpp and rejects that outright:

    error: unknown argument: --models-preset

Switching to it therefore persisted active_engine and then left the machine
with no router at all. The switch has to ask the binary first.
"""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import router_ctl, routes
from routes import Req

ROUTER_HELP = "usage: llama-server\n  --models-preset PATH  ini\n  --models-max N  max\n"
PLAIN_HELP  = "usage: llama-server\n  -m, --model FNAME  path\n  --port N  port\n"


class SupportsCorsOriginsTest(unittest.TestCase):
    def setUp(self):
        router_ctl.clear_router_mode_cache()
        self.addCleanup(router_ctl.clear_router_mode_cache)

    def test_detects_flag_and_shares_one_probe_with_router_mode(self):
        text = ROUTER_HELP + "  --cors-origins ORIGINS  allowed origins\n"
        with mock.patch.object(router_ctl.subprocess, "check_output",
                               return_value=text) as probe, \
             mock.patch.object(router_ctl.os.path, "getmtime", return_value=1.0):
            self.assertTrue(router_ctl.supports_cors_origins("/bin/x"))
            self.assertTrue(router_ctl.supports_router_mode("/bin/x"))
        probe.assert_called_once()

    def test_false_for_builds_without_the_flag(self):
        with mock.patch.object(router_ctl.subprocess, "check_output",
                               return_value=ROUTER_HELP), \
             mock.patch.object(router_ctl.os.path, "getmtime", return_value=1.0):
            self.assertFalse(router_ctl.supports_cors_origins("/bin/x"))


class SupportsRouterModeTest(unittest.TestCase):
    def setUp(self):
        router_ctl.clear_router_mode_cache()
        self.addCleanup(router_ctl.clear_router_mode_cache)

    def _probe(self, help_text, mtime=1.0):
        with mock.patch.object(router_ctl.subprocess, "check_output", return_value=help_text), \
             mock.patch.object(router_ctl.os.path, "getmtime", return_value=mtime):
            return router_ctl.supports_router_mode("/bin/llama-server")

    def test_true_when_the_binary_advertises_models_preset(self):
        self.assertTrue(self._probe(ROUTER_HELP))

    def test_false_for_a_binary_without_router_mode(self):
        self.assertFalse(self._probe(PLAIN_HELP))

    def test_false_when_the_binary_cannot_be_run(self):
        with mock.patch.object(router_ctl.subprocess, "check_output",
                               side_effect=OSError("nope")), \
             mock.patch.object(router_ctl.os.path, "getmtime", return_value=1.0):
            self.assertFalse(router_ctl.supports_router_mode("/bin/missing"))

    def test_a_failed_probe_is_not_cached(self):
        """Same rule the knob schema follows: never cache a failure, so fixing
        the binary takes effect without restarting the backend."""
        with mock.patch.object(router_ctl.os.path, "getmtime", return_value=1.0):
            with mock.patch.object(router_ctl.subprocess, "check_output",
                                   side_effect=OSError("nope")):
                self.assertFalse(router_ctl.supports_router_mode("/bin/x"))
            with mock.patch.object(router_ctl.subprocess, "check_output",
                                   return_value=ROUTER_HELP) as ok:
                self.assertTrue(router_ctl.supports_router_mode("/bin/x"))
                ok.assert_called_once()

    def test_a_success_is_cached_per_binary_mtime(self):
        with mock.patch.object(router_ctl.os.path, "getmtime", return_value=1.0):
            with mock.patch.object(router_ctl.subprocess, "check_output",
                                   return_value=ROUTER_HELP) as first:
                self.assertTrue(router_ctl.supports_router_mode("/bin/x"))
                self.assertTrue(router_ctl.supports_router_mode("/bin/x"))
                first.assert_called_once()


class EngineSwitchRefusesNonRouterBinaryTest(unittest.TestCase):
    def setUp(self):
        self.saved = {}
        self.base = {"router_port": 8080, "router_host": "127.0.0.1",
                     "router_api_key": "", "models_ini": "/tmp/models.ini",
                     "server_bin": "/bin/llama-server", "active_engine": "llamacpp",
                     "ik_llama_server_bin": "/bin/ik/llama-server"}
        mock.patch.object(routes.config, "load",
                          side_effect=lambda: dict(self.base, **self.saved)).start()
        mock.patch.object(routes.config, "update",
                          side_effect=lambda ch: (self.saved.update(ch),
                                                  dict(self.base, **self.saved))[1]).start()
        mock.patch.object(routes.config, "ini_path", return_value="/tmp/models.ini").start()
        self.restart = mock.patch.object(routes.router_ctl, "restart",
                                         return_value=(True, "")).start()
        mock.patch.object(routes.os.path, "exists", return_value=True).start()
        self.addCleanup(mock.patch.stopall)

    def test_refuses_and_keeps_the_current_engine(self):
        with mock.patch.object(routes.router_ctl, "supports_router_mode", return_value=False):
            status, out = routes.post_engine_switch(Req(body={"engine": "ikllama"}))
        self.assertEqual(status, 200)
        self.assertFalse(out["ok"])
        self.assertIn("router mode", out["error"])
        self.assertNotIn("active_engine", self.saved)
        self.restart.assert_not_called()

    def test_allows_a_binary_that_does_support_router_mode(self):
        with mock.patch.object(routes.router_ctl, "supports_router_mode", return_value=True):
            status, out = routes.post_engine_switch(Req(body={"engine": "ikllama"}))
        self.assertTrue(out["ok"])
        self.assertEqual(self.saved["active_engine"], "ikllama")
        self.restart.assert_called_once()
