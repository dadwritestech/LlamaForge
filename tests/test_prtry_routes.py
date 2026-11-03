"""/api/prtry*: try a llama.cpp PR from the panel. git, cmake and GitHub are
faked; config.json and the engines dir are temp."""
import conftest_paths  # noqa: F401
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import config
import prbench
import prtry
import routes
from routes import ApiError, Req

SHA = "0123456789abcdef0123456789abcdef01234567"


def api(sha=SHA):
    return {"number": 1234, "title": "CUDA: faster MMQ", "state": "open", "merged_at": None,
            "user": {"login": "someone"}, "head": {"sha": sha}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        old = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")
        self.addCleanup(setattr, config, "CONFIG", old)
        with open(config.CONFIG, "w", encoding="utf-8") as f:
            json.dump({"server_bin": os.path.join(self.tmp, "router", "llama-server.exe"),
                       "cmake_flags": {"GGML_CUDA": "ON"},
                       "model_builds": {"qwen": "pr-1234-0123456", "gemma": "b7000-cuda"}}, f)
        self.engines = os.path.join(self.tmp, "engines")
        self.remote = api()

        def run(cmd, cwd=None):
            if cmd[:2] == ["cmake", "--build"]:
                out = os.path.join(cmd[2], "bin")
                os.makedirs(out, exist_ok=True)
                open(os.path.join(out, "llama-server.exe"), "w").close()
            return 0
        self.pr = prtry.PrBuilder(self.engines, os.path.join(self.tmp, "work"), self.tmp,
                                  run=run, capture=lambda cmd: SHA,
                                  fetch_json=lambda url: self.remote)
        mock.patch.object(routes, "PRTRY", self.pr).start()
        mock.patch.object(routes, "ENGINES_DIR", self.engines).start()
        self.status = {}
        mock.patch.object(routes, "_model_status",
                          side_effect=lambda mid: {"value": self.status.get(mid, "unloaded")}).start()
        self.unloaded = []
        mock.patch.object(routes, "_unload_and_wait",
                          side_effect=lambda mid: (self.unloaded.append(mid), (200, {}))[1]).start()
        self.addCleanup(mock.patch.stopall)

    def install(self, name, channel):
        d = os.path.join(self.engines, name)
        os.makedirs(d)
        open(os.path.join(d, "llama-server.exe"), "w").close()
        with open(os.path.join(d, "lf-install.json"), "w") as f:
            json.dump({"tag": name, "channel": channel, "installed_at": 1}, f)
        return d


class Look(Base):
    def test_look_shows_what_will_be_built(self):
        code, out = routes.post_prtry_look(Req(body={"pr": "#1234"}))
        self.assertEqual(code, 200)
        self.assertEqual((out["number"], out["sha"], out["author"]), (1234, SHA, "someone"))

    def test_bad_input_is_a_400(self):
        for body in ({"pr": "abc"}, {"pr": "https://github.com/x/llama.cpp/pull/1"}, {}):
            with self.assertRaises(ApiError) as e:
                routes.post_prtry_look(Req(body=body))
            self.assertEqual(e.exception.status, 400)


class Build(Base):
    def test_builds_the_commit_that_was_shown(self):
        code, out = routes.post_prtry_build(Req(body={"pr": 1234, "sha": SHA}))
        self.assertEqual(code, 200)
        self.assertTrue(out["started"])
        self.pr.join(10)
        st = routes.get_prtry(Req())[1]
        self.assertEqual(st["status"]["phase"], "done", st)
        self.assertEqual([i["name"] for i in st["installs"]], ["pr-1234-0123456"])
        self.assertEqual(st["installs"][0]["models"], ["qwen"])
        with open(os.path.join(self.engines, "pr-1234-0123456", "lf-install.json")) as f:
            self.assertEqual(json.load(f)["flags"], {"GGML_CUDA": "ON"})

    def test_a_pr_that_moved_since_the_look_is_refused(self):
        self.remote = api("f" * 40)
        with self.assertRaises(ApiError) as e:
            routes.post_prtry_build(Req(body={"pr": 1234, "sha": SHA}))
        self.assertEqual(e.exception.status, 409)

    def test_not_beside_a_main_build(self):
        with mock.patch.dict(routes.BUILDER_LLAMA.state, running=True):
            with self.assertRaises(ApiError) as e:
                routes.post_prtry_build(Req(body={"pr": 1234, "sha": SHA}))
        self.assertEqual(e.exception.status, 409)

    def test_get_lists_only_pr_builds(self):
        self.install("b7000-cuda", "nightly")
        self.install("pr-99-abcdef0", "pr")
        st = routes.get_prtry(Req())[1]
        self.assertEqual([i["name"] for i in st["installs"]], ["pr-99-abcdef0"])
        self.assertIn("log", st)


class Remove(Base):
    def test_remove_unloads_and_unpins_its_models(self):
        d = self.install("pr-1234-0123456", "pr")
        self.status["qwen"] = "loaded"
        code, out = routes.post_prtry_remove(Req(body={"name": "pr-1234-0123456"}))
        self.assertEqual(code, 200)
        self.assertEqual(out["unpinned"], ["qwen"])
        self.assertEqual(self.unloaded, ["qwen"])
        self.assertFalse(os.path.exists(d))
        self.assertEqual(config.load()["model_builds"], {"gemma": "b7000-cuda"})

    def test_released_builds_are_not_removable_here(self):
        self.install("b7000-cuda", "nightly")
        with self.assertRaises(ApiError) as e:
            routes.post_prtry_remove(Req(body={"name": "b7000-cuda"}))
        self.assertEqual(e.exception.status, 404)

    def test_the_router_never_switches_to_a_pr_build(self):
        d = self.install("pr-1234-0123456", "pr")
        with mock.patch.object(routes, "_activate_prebuilt") as act:
            with self.assertRaises(ApiError) as e:
                routes.post_engine_prebuilt_use(Req(body={"dir": d}))
        self.assertEqual(e.exception.status, 400)
        act.assert_not_called()


class AB(Base):
    """POST /api/prtry/ab: llama-bench on your build vs a PR build, one model."""

    def setUp(self):
        super().setUp()
        self.pr_dir = self.install("pr-1234-0123456", "pr")
        with open(os.path.join(self.pr_dir, "lf-install.json"), "w") as f:
            json.dump({"tag": "PR #1234", "channel": "pr", "pr": 1234, "installed_at": 1}, f)
        open(os.path.join(self.pr_dir, "llama-bench.exe"), "w").close()
        router = os.path.join(self.tmp, "router")
        os.makedirs(router)
        open(os.path.join(router, "llama-bench.exe"), "w").close()
        self.sections = {"*": {"flash-attn": "on"},
                         "qwen": {"model": "Q.gguf", "n-gpu-layers": "99"}}
        mock.patch.object(config, "read_sections", lambda: self.sections).start()
        self.ran, self.events = [], []
        self.ab = prbench.ABRunner(os.path.join(self.tmp, "ab.json"), run=self.bench,
                                   lease=routes._ab_lease)
        mock.patch.object(routes, "PRBENCH", self.ab).start()
        mock.patch.object(routes, "_game_loaded", lambda: ["gemma"]).start()
        mock.patch.object(routes, "post_unload_all",
                          lambda req: (self.events.append("unload"), (200, {}))[1]).start()
        mock.patch.object(routes, "_game_load",
                          lambda mid: (self.events.append("load " + mid), True)[1]).start()

    def bench(self, argv):
        self.ran.append(argv)
        self.events.append("bench")
        ts = 100.0 if "router" in argv[0] else 110.0
        return 0, json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 5000.0, "stddev_ts": 10.0},
                              {"n_prompt": 0, "n_gen": 128, "avg_ts": ts, "stddev_ts": 0.5}]), ""

    def test_both_builds_same_model_same_flags(self):
        code, out = routes.post_prtry_ab(Req(body={"model": "qwen", "install": "pr-1234-0123456"}))
        self.assertEqual(code, 200)
        self.ab.join(5)
        st = routes.get_prtry(Req())[1]["ab"]
        self.assertEqual(st["phase"], "done", st)
        self.assertEqual(st["delta"]["tg"]["pct"], 10.0)
        self.assertEqual((st["pr"], st["model"]), (1234, "qwen"))
        base, pr = self.ran
        self.assertEqual(base[0], os.path.join(self.tmp, "router", "llama-bench.exe"))
        self.assertEqual(pr[0], os.path.join(self.pr_dir, "llama-bench.exe"))
        self.assertEqual(base[1:], pr[1:])
        self.assertIn("-fa", base)          # the "*" defaults apply too
        # the GPUs are cleared first, and what was loaded comes back after
        self.assertEqual(self.events, ["unload", "bench", "bench", "load gemma"])

    def test_refusals(self):
        def refused(body, code):
            with self.assertRaises(ApiError, msg=body) as e:
                routes.post_prtry_ab(Req(body=body))
            self.assertEqual(e.exception.status, code, body)
        refused({"model": "qwen", "install": "nope"}, 404)
        refused({"model": "ghost", "install": "pr-1234-0123456"}, 404)
        os.remove(os.path.join(self.pr_dir, "llama-bench.exe"))
        refused({"model": "qwen", "install": "pr-1234-0123456"}, 400)
        open(os.path.join(self.pr_dir, "llama-bench.exe"), "w").close()
        with mock.patch.object(self.pr, "status", lambda: {"running": True}):
            refused({"model": "qwen", "install": "pr-1234-0123456"}, 409)
        with mock.patch.object(routes, "_active_server_bin", lambda c=None: ""):
            refused({"model": "qwen", "install": "pr-1234-0123456"}, 400)
        with mock.patch.object(self.ab, "status", lambda: {"running": True}):
            refused({"model": "qwen", "install": "pr-1234-0123456"}, 409)
        self.assertEqual(self.ran, [])


if __name__ == "__main__":
    unittest.main()
