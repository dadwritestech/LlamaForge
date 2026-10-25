import conftest_paths  # noqa: F401
import os
import tempfile
import unittest
from unittest import mock

import config
import routes
from routes import ApiError, Req


CFG = {"active_engine": "llamacpp", "router_host": "127.0.0.1", "router_port": 8080,
       "panel_port": 8090, "router_api_key": "k" * 40, "anthropic_shim_enabled": True,
       "claude_bin": "", "claude_recent_dirs": ["/old"]}
ROWS = [
    {"id": "big", "backend": "llamacpp", "status": "loaded", "file_gib": 20,
     "settings": {"ctx-size": "131072"}},
    {"id": "tiny", "backend": "llamacpp", "status": "loaded", "file_gib": 1, "settings": {}},
    {"id": "off", "backend": "llamacpp", "status": "unloaded", "file_gib": 5,
     "settings": {"ctx-size": "8192"}},
    {"id": "vm", "backend": "vllm", "status": "loaded", "settings": {}},
]


class FakeRegistry:
    def __init__(self, rows=ROWS):
        self.rows = rows

    def state(self):
        return {"models": [dict(r) for r in self.rows], "global": {}}

    def active_engine(self):
        return "llamacpp"


def props(n_ctx):
    return lambda path, **kw: (200, {"default_generation_settings": {"n_ctx": n_ctx}})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        fd, self.cfgfile = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.cfgfile)
        self._cfg_was = config.CONFIG
        config.CONFIG = self.cfgfile
        config.save(dict(CFG))
        self.patches = [
            mock.patch.object(routes, "REGISTRY", FakeRegistry()),
            mock.patch.object(routes, "cfg", side_effect=lambda: config.load()),
            mock.patch.object(routes.claudelaunch, "locate", return_value="/bin/claude"),
            mock.patch.object(routes.claudelaunch, "version", return_value="2.1.0"),
            mock.patch.object(routes, "router", side_effect=props(131072)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        config.CONFIG = self._cfg_was
        if os.path.exists(self.cfgfile):
            os.remove(self.cfgfile)


class CheckRouteTest(Base):
    def test_ready_with_worker_as_small(self):
        st, out = routes.get_claude_check(Req(qs={"model": "big"}))
        self.assertEqual(st, 200)
        self.assertTrue(out["ready"])
        self.assertEqual(out["small"], "tiny")
        self.assertEqual(out["cli"], {"path": "/bin/claude", "version": "2.1.0"})
        self.assertEqual(out["recent"], ["/old"])
        self.assertTrue(out["install_hint"])

    def test_live_ctx_only_without_autoload(self):
        calls = []
        with mock.patch.object(routes, "router",
                               side_effect=lambda path, **kw: calls.append(path) or (200, {})):
            routes.get_claude_check(Req(qs={"model": "big"}))
        self.assertTrue(calls and all("autoload=false" in c for c in calls))

    def test_unloaded_model_uses_configured_ctx_and_never_asks_router(self):
        with mock.patch.object(routes, "router", side_effect=AssertionError("no router")):
            st, out = routes.get_claude_check(Req(qs={"model": "off"}))
        ctx = next(i for i in out["checks"] if i["id"] == "ctx")
        self.assertEqual(ctx["level"], "error")
        self.assertFalse(out["ready"])

    def test_vllm_and_unknown_rejected(self):
        for mid in ("vm", "nope", ""):
            with self.subTest(mid=mid), self.assertRaises(ApiError):
                routes.get_claude_check(Req(qs={"model": mid}))

    def test_claude_bin_override(self):
        config.update({"claude_bin": "/custom/claude"})
        with mock.patch.object(routes.os.path, "isfile", return_value=True):
            st, out = routes.get_claude_check(Req(qs={"model": "big"}))
        self.assertEqual(out["cli"]["path"], "/custom/claude")


class LaunchRouteTest(Base):
    def test_launches_and_remembers_folder(self):
        with mock.patch.object(routes.claudelaunch, "launch",
                               return_value={"kind": "spawn", "command": "cmd"}) as launch:
            st, out = routes.post_claude_launch(Req(body={"model": "big", "folder": self.tmp}))
        self.assertEqual(st, 200)
        self.assertEqual(out, {"launched": True, "command": "cmd"})
        args = launch.call_args
        self.assertEqual(args.args[0], "/bin/claude")
        self.assertEqual(args.args[1], self.tmp)
        self.assertEqual(args.args[3], "http://127.0.0.1:8090")
        self.assertEqual(args.args[5:7], ("big", "tiny"))
        self.assertEqual(config.load()["claude_recent_dirs"], [self.tmp, "/old"])

    def test_manual_plan_is_not_launched(self):
        with mock.patch.object(routes.claudelaunch, "launch",
                               return_value={"kind": "manual", "command": "cmd"}):
            st, out = routes.post_claude_launch(Req(body={"model": "big", "folder": self.tmp}))
        self.assertEqual(out, {"launched": False, "command": "cmd"})

    def test_rejects_bad_input(self):
        bad = [{"model": "big", "folder": os.path.join(self.tmp, "missing")},
               {"model": "big", "folder": ""}, {"model": "big", "folder": 5},
               {"model": "vm", "folder": self.tmp}, {"model": "big", "folder": self.tmp, "x": 1}]
        with mock.patch.object(routes.claudelaunch, "launch") as launch:
            for body in bad:
                with self.subTest(body=body), self.assertRaises(ApiError):
                    routes.post_claude_launch(Req(body=body))
        launch.assert_not_called()

    def test_blocking_check_refuses(self):
        with mock.patch.object(routes.claudelaunch, "locate", return_value=""), \
             mock.patch.object(routes.claudelaunch, "launch") as launch:
            with self.assertRaises(ApiError) as e:
                routes.post_claude_launch(Req(body={"model": "big", "folder": self.tmp}))
        self.assertEqual(e.exception.status, 409)
        launch.assert_not_called()

    def test_spawn_failure_is_reported(self):
        with mock.patch.object(routes.claudelaunch, "launch", side_effect=OSError("nope")):
            with self.assertRaises(ApiError) as e:
                routes.post_claude_launch(Req(body={"model": "big", "folder": self.tmp}))
        self.assertEqual(e.exception.status, 500)


class PickFolderRouteTest(Base):
    def test_returns_choice(self):
        with mock.patch.object(routes.claudelaunch, "pick_folder_argv", return_value=["x"]), \
             mock.patch.object(routes.claudelaunch, "pick_folder", return_value="/w"):
            self.assertEqual(routes.post_claude_pick_folder(Req()), (200, {"path": "/w"}))

    def test_unsupported(self):
        with mock.patch.object(routes.claudelaunch, "pick_folder_argv", return_value=None):
            st, out = routes.post_claude_pick_folder(Req())
        self.assertEqual(out, {"path": "", "unsupported": True})


class WiredTest(unittest.TestCase):
    def test_routes_registered(self):
        self.assertIs(routes.GET_ROUTES["/api/claude/check"], routes.get_claude_check)
        self.assertIs(routes.POST_ROUTES["/api/claude/launch"], routes.post_claude_launch)
        self.assertIs(routes.POST_ROUTES["/api/claude/pick-folder"], routes.post_claude_pick_folder)


if __name__ == "__main__":
    unittest.main()
