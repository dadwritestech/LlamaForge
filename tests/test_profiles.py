import conftest_paths  # noqa: F401
import json, os, shutil, tempfile, unittest

import config
import prebuilt
import profiles


class ProfileStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._orig = config.CONFIG
        config.CONFIG = os.path.join(self.dir, "config.json")

    def tearDown(self):
        config.CONFIG = self._orig
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_save_read_delete(self):
        config.save_profile("coder", {"model": "qwen", "preset": "coding", "engine": "b11300-cuda"})
        self.assertEqual(config.get_profiles()["coder"],
                         {"model": "qwen", "backend": "llamacpp", "preset": "coding", "engine": "b11300-cuda"})
        self.assertTrue(config.delete_profile("coder"))
        self.assertFalse(config.delete_profile("coder"))
        self.assertEqual(config.get_profiles(), {})

    def test_optional_fields_default_blank(self):
        config.save_profile("plain", {"model": "gemma"})
        self.assertEqual(config.get_profiles()["plain"],
                         {"model": "gemma", "backend": "llamacpp", "preset": "", "engine": ""})

    def test_name_and_model_required(self):
        with self.assertRaises(ValueError):
            config.save_profile(" ", {"model": "x"})
        with self.assertRaises(ValueError):
            config.save_profile("x", {"model": ""})

    def test_engine_must_be_a_plain_dir_name(self):
        for bad in ["../evil", "C:\\engines\\b1", "a/b"]:
            with self.assertRaises(ValueError, msg=bad):
                config.save_profile("x", {"model": "m", "engine": bad})

    def test_deleting_a_preset_clears_it_from_profiles(self):
        config.save_preset("coding", {"temp": "0.2"})
        config.save_profile("coder", {"model": "qwen", "preset": "coding"})
        config.delete_preset("coding")
        self.assertEqual(config.get_profiles()["coder"]["preset"], "")


class Plan(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.old = self._install("b11000-cuda", 1)
        self.new = self._install("b11300-cuda", 2)
        self.installs = prebuilt.list_installs(self.root, active_bin=self._bin(self.new))
        self.presets = {"coding": {"temp": "0.2", "top-k": ""}}

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _bin(self, d):
        return os.path.join(d, "llama-server.exe")

    def _install(self, name, at):
        d = os.path.join(self.root, name)
        os.makedirs(d)
        open(self._bin(d), "w").close()
        with open(os.path.join(d, prebuilt.MANIFEST), "w") as f:
            json.dump({"tag": name.split("-")[0], "installed_at": at}, f)
        return d

    def plan(self, **p):
        prof = dict({"model": "qwen", "backend": "llamacpp", "preset": "", "engine": ""}, **p)
        return profiles.plan(prof, self.installs, self.presets)

    def test_plain_profile_just_loads(self):
        self.assertEqual(self.plan(), {"model": "qwen", "backend": "llamacpp",
                                       "switch_bin": None, "settings": None})

    def test_pinned_older_build_switches(self):
        self.assertEqual(self.plan(engine="b11000-cuda")["switch_bin"], self._bin(self.old))

    def test_pinned_active_build_does_not_restart(self):
        self.assertIsNone(self.plan(engine="b11300-cuda")["switch_bin"])

    def test_removed_build_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "b10000-cuda"):
            self.plan(engine="b10000-cuda")

    def test_preset_is_cleaned_like_save(self):
        self.assertEqual(self.plan(preset="coding")["settings"], {"temp": "0.2", "top-k": None})

    def test_missing_preset_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "nope"):
            self.plan(preset="nope")

    def test_vllm_profiles_cannot_pin_a_llama_build(self):
        with self.assertRaises(ValueError):
            self.plan(backend="vllm", engine="b11000-cuda")

    def test_vllm_profiles_cannot_use_presets(self):
        with self.assertRaises(ValueError):
            self.plan(backend="vllm", preset="coding")

    def test_pinned_dirs(self):
        profs = {"a": {"engine": "b11000-cuda"}, "b": {"engine": ""}}
        self.assertEqual(profiles.pinned_dirs(profs, self.installs), [self.old])

    def test_prune_spares_pinned_builds(self):
        extra = [self._install(f"b{11400 + i}-cuda", 10 + i) for i in range(3)]
        removed = prebuilt.prune_installs(self.root, keep=2, protect=[self.old])
        self.assertNotIn(self.old, removed)
        self.assertIn(self.new, removed)
        self.assertTrue(os.path.isdir(self.old))
        self.assertTrue(all(os.path.isdir(d) for d in extra[1:]))


class LaunchRoute(unittest.TestCase):
    """post_profiles_launch runs plan() steps in order: engine, knobs, load."""

    def setUp(self):
        import routes
        from unittest import mock
        self.routes, self.calls = routes, []
        prof = {"model": "qwen", "backend": "llamacpp", "preset": "coding", "engine": "b1-cuda"}
        steps = {"model": "qwen", "backend": "llamacpp",
                 "switch_bin": "/e/b1-cuda/llama-server", "settings": {"temp": "0.2"}}
        backend = mock.Mock()
        backend.load.side_effect = lambda mid: (self.calls.append(("load", mid)), (True, ""))[1]
        for target, attr, kw in [
            (routes.config, "get_profiles", {"return_value": {"p": prof}}),
            (routes.config, "get_presets", {"return_value": {}}),
            (routes.profiles, "plan", {"return_value": steps}),
            (routes.prebuilt, "list_installs", {"return_value": []}),
            (routes, "_activate_prebuilt",
             {"side_effect": lambda b: (self.calls.append(("engine", b)), (True, ""))[1]}),
            (routes, "_wait_router", {"side_effect": lambda: (self.calls.append(("wait",)), True)[1]}),
            (routes, "_apply_knobs_and_reload",
             {"side_effect": lambda m, k: self.calls.append(("knobs", m, k))}),
        ]:
            pt = mock.patch.object(target, attr, **kw)
            pt.start(); self.addCleanup(pt.stop)
        pt = mock.patch.object(routes.REGISTRY, "for_model", return_value=backend)
        pt.start(); self.addCleanup(pt.stop)

    def test_switch_then_knobs_then_load(self):
        status, out = self.routes.post_profiles_launch(self.routes.Req(body={"name": "p"}))
        self.assertTrue(out["ok"], out)
        self.assertEqual(self.calls, [("engine", "/e/b1-cuda/llama-server"), ("wait",),
                                      ("knobs", "qwen", {"temp": "0.2"}), ("load", "qwen")])

    def test_unknown_profile_is_404(self):
        with self.assertRaises(self.routes.ApiError):
            self.routes.post_profiles_launch(self.routes.Req(body={"name": "nope"}))

    def test_plan_errors_stop_before_anything_runs(self):
        self.routes.profiles.plan.side_effect = ValueError("engine build b1-cuda is no longer installed")
        status, out = self.routes.post_profiles_launch(self.routes.Req(body={"name": "p"}))
        self.assertEqual((out["ok"], out["step"]), (False, "plan"))
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
