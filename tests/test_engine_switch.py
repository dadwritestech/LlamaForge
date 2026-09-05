"""Switching the active engine, and deriving the per-engine models.ini path.

Both are new with the ik_llama work. The switch route is the risky one: it
writes `active_engine` into config.json and then restarts the router, so a
write that happens before the binary is validated leaves the whole panel
pointed at an engine that cannot start.
"""
import conftest_paths  # noqa: F401
import os
import threading
import unittest
from unittest import mock

import config, routes
from routes import Req


class EngineSwitchRouteTest(unittest.TestCase):
    def setUp(self):
        self.saved = {}
        self.base = {"router_port": 8080, "router_host": "127.0.0.1",
                     "router_api_key": "", "models_ini": "/tmp/models.ini",
                     "server_bin": "/bin/llama-server", "active_engine": "llamacpp"}

        def fake_update(changes):
            self.saved.update(changes)
            return dict(self.base, **self.saved)

        for target, name, new in (
                (config, "update", fake_update),
                (config, "load", lambda: dict(self.base, **self.saved)),
                (config, "ini_path", lambda: "/tmp/models.ini")):
            p = mock.patch.object(target, name, side_effect=new) \
                if callable(new) else None
            p.start(); self.addCleanup(p.stop)

        self.restart = mock.patch.object(
            routes.router_ctl, "restart", return_value=(True, "")).start()
        self.addCleanup(mock.patch.stopall)

    def test_rejects_an_unknown_engine(self):
        with self.assertRaises(routes.ApiError) as cm:
            routes.post_engine_switch(Req(body={"engine": "vllm"}))
        self.assertEqual(cm.exception.status, 400)
        self.assertEqual(self.saved, {}, "config was written for a bad engine")

    def test_does_not_persist_the_switch_when_the_binary_is_missing(self):
        """The regression: config.update ran before the existence check, so a
        failed switch still left active_engine=ikllama behind."""
        self.base["ik_llama_server_bin"] = "/nope/llama-server"
        with mock.patch.object(os.path, "exists", return_value=False):
            status, out = routes.post_engine_switch(Req(body={"engine": "ikllama"}))
        self.assertEqual(status, 200)
        self.assertFalse(out["ok"])
        self.assertNotIn("active_engine", self.saved)
        self.restart.assert_not_called()

    def test_persists_and_restarts_when_the_binary_is_present(self):
        """Present AND router-capable: see test_router_capability.py for the
        binary that exists but cannot be the router."""
        self.base["ik_llama_server_bin"] = "/opt/ik/llama-server"
        with mock.patch.object(os.path, "exists", return_value=True), \
             mock.patch.object(routes.router_ctl, "supports_router_mode", return_value=True):
            status, out = routes.post_engine_switch(Req(body={"engine": "ikllama"}))
        self.assertEqual(status, 200)
        self.assertTrue(out["ok"])
        self.assertEqual(self.saved["active_engine"], "ikllama")
        self.restart.assert_called_once()
        self.assertEqual(self.restart.call_args[0][0], "/opt/ik/llama-server")

    def test_engine_switch_uses_shared_router_transaction_lock(self):
        """Engine and network changes must not cross their router restarts."""
        entered = threading.Event()
        result = []

        def load():
            entered.set()
            return dict(self.base, **self.saved)

        def switch():
            result.append(routes.post_engine_switch(Req(body={"engine": "llamacpp"})))

        with mock.patch.object(routes, "cfg", side_effect=load), \
             mock.patch.object(os.path, "exists", return_value=True), \
             mock.patch.object(routes.router_ctl, "supports_router_mode", return_value=True):
            with routes._ROUTER_LIFECYCLE_LOCK:
                worker = threading.Thread(target=switch)
                worker.start()
                self.assertFalse(entered.wait(0.25))
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(entered.is_set())
        self.assertEqual(result[0][0], 200)


class IniPathTest(unittest.TestCase):
    """`ini_path()` derives a sibling models.ini for ik_llama when none is set."""

    def _path(self, cfg):
        with mock.patch.object(config, "load", return_value=cfg):
            return config.ini_path()

    def test_llamacpp_uses_the_configured_ini(self):
        self.assertEqual(
            self._path({"models_ini": "D:/x/models.ini", "active_engine": "llamacpp"}),
            "D:/x/models.ini")

    def test_ikllama_prefers_its_own_explicit_ini(self):
        self.assertEqual(
            self._path({"models_ini": "D:/x/models.ini",
                        "ik_llama_models_ini": "D:/x/ik.ini",
                        "active_engine": "ikllama"}),
            "D:/x/ik.ini")

    def test_ikllama_derives_a_sibling_when_unset(self):
        self.assertEqual(
            self._path({"models_ini": "D:/x/models.ini", "ik_llama_models_ini": "",
                        "active_engine": "ikllama"}),
            os.path.join("D:/x", "models-ikllama.ini").replace("\\", "/"))

    def test_derivation_does_not_maul_a_directory_named_ini(self):
        """str.replace('.ini', ...) is global, so a directory containing '.ini'
        got rewritten too. Only the file's extension may change."""
        got = self._path({"models_ini": "D:/conf.ini.d/models.ini",
                          "ik_llama_models_ini": "", "active_engine": "ikllama"})
        self.assertEqual(got.replace("\\", "/"), "D:/conf.ini.d/models-ikllama.ini")
