import conftest_paths  # noqa: F401
import json
import os
import tempfile
import unittest
from unittest import mock

import config
import idle


class IdleKeysTest(unittest.TestCase):
    def test_minutes_from_the_global_section(self):
        self.assertEqual(idle.minutes({"*": {"sleep-idle-seconds": "900"}}), 15)
        self.assertEqual(idle.minutes({"*": {"sleep-idle-seconds": "30"}}), 1)   # never shows 0 when on
        for off in ({}, {"*": {}}, {"*": {"sleep-idle-seconds": "-1"}},
                    {"*": {"sleep-idle-seconds": "soon"}}, {"*": {"sleep-idle-seconds": "0"}}):
            self.assertEqual(idle.minutes(off), 0)

    def test_updates(self):
        self.assertEqual(idle.updates(15), {"sleep-idle-seconds": "900"})
        self.assertEqual(idle.updates(0), {"sleep-idle-seconds": None})
        for bad in (-1, 1441, "15", 2.5, True, None):
            with self.assertRaises(ValueError):
                idle.updates(bad)


class IdleRouteTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.ini = os.path.join(self.d, "models.ini")
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write("version = 1\n\n[*]\njinja = true ; keep me\n\n[m]\nmodel = x.gguf\n")
        self.cfg = os.path.join(self.d, "config.json")
        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump({"models_ini": self.ini, "server_bin": os.path.join(self.d, "llama-server")}, f)
        self._orig = config.CONFIG
        config.CONFIG = self.cfg
        import routes
        self.routes = routes
        p1 = mock.patch.object(routes.router_ctl, "supports_sleep_idle", return_value=True)
        p2 = mock.patch.object(routes, "reload_presets", return_value=(200, {"success": True}))
        self.reload = p2.start(); p1.start()
        self.addCleanup(mock.patch.stopall)

    def tearDown(self):
        config.CONFIG = self._orig

    def req(self, body=None):
        return mock.Mock(body=body, query={})

    def test_registered(self):
        self.assertIn("/api/idle", self.routes.GET_ROUTES)
        self.assertIn("/api/idle", self.routes.POST_ROUTES)

    def test_set_then_read_then_turn_off(self):
        st, out = self.routes.post_idle(self.req({"minutes": 10}))
        self.assertEqual((st, out["ok"], out["minutes"]), (200, True, 10))
        self.assertEqual(config.read_sections(self.ini)["*"]["sleep-idle-seconds"], "600")
        with open(self.ini, encoding="utf-8") as f:
            self.assertIn("; keep me", f.read())
        self.reload.assert_called_once()
        st, out = self.routes.get_idle(self.req())
        self.assertEqual(out, {"supported": True, "minutes": 10})
        self.routes.post_idle(self.req({"minutes": 0}))
        self.assertNotIn("sleep-idle-seconds", config.read_sections(self.ini)["*"])

    def test_bad_minutes_and_unsupported_build_are_refused(self):
        st, out = self.routes.post_idle(self.req({"minutes": -5}))
        self.assertFalse(out["ok"])
        with mock.patch.object(self.routes.router_ctl, "supports_sleep_idle", return_value=False):
            st, out = self.routes.post_idle(self.req({"minutes": 10}))
            self.assertFalse(out["ok"])
            self.assertEqual(self.routes.get_idle(self.req())[1]["supported"], False)
        self.assertNotIn("sleep-idle-seconds", config.read_sections(self.ini)["*"])


if __name__ == "__main__":
    unittest.main()
