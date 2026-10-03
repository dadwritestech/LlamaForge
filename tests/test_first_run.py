"""First-run blockers reported in issue #5.

A fresh Windows install hit a chain of failures that all shared one shape: a
path or a port was wrong, and the code found out by handing the bad value to an
external process (cmake, llama-server) whose error the user never saw. These
tests pin the checks that happen *before* we shell out.
"""
import conftest_paths  # noqa: F401
import json, os, tempfile, unittest

import builder, config, osplat, router_ctl
from builder import BuildManager


class _ConfigTempCase(unittest.TestCase):
    """config-touching tests must never read or write the real config.json."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")

    def tearDown(self):
        config.CONFIG = self._saved

    def _write_cfg(self, **keys):
        with open(config.CONFIG, "w", encoding="utf-8") as f:
            json.dump(keys, f)


class IniPathIsAbsoluteTest(_ConfigTempCase):
    """The router is launched detached and does not inherit the panel's CWD, so
    a relative models_ini (config.example.json ships './models.ini') resolved
    against whatever directory the user happened to launch from - it read a
    different, usually empty, registry and loaded 0 models."""

    def test_relative_models_ini_resolves_against_repo_root(self):
        self._write_cfg(models_ini="./models.ini")
        p = config.ini_path()
        self.assertTrue(os.path.isabs(p), f"expected absolute, got {p!r}")
        self.assertEqual(p, os.path.join(config.ROOT, "models.ini"))

    def test_nested_relative_path_resolves(self):
        self._write_cfg(models_ini="reg/models.ini")
        self.assertEqual(config.ini_path(),
                         os.path.join(config.ROOT, "reg", "models.ini"))

    def test_absolute_models_ini_is_left_alone(self):
        absolute = os.path.join(self.tmp, "elsewhere.ini")
        self._write_cfg(models_ini=absolute)
        self.assertEqual(config.ini_path(), absolute)

    def test_ikllama_sibling_is_also_absolute(self):
        self._write_cfg(models_ini="./models.ini", active_engine="ikllama")
        p = config.ini_path()
        self.assertTrue(os.path.isabs(p), f"expected absolute, got {p!r}")
        self.assertEqual(p, os.path.join(config.ROOT, "models-ikllama.ini"))

    def test_explicit_ikllama_relative_path_resolves(self):
        self._write_cfg(models_ini="./models.ini", active_engine="ikllama",
                        ik_llama_models_ini="./ik.ini")
        self.assertEqual(config.ini_path(), os.path.join(config.ROOT, "ik.ini"))


class EnsureModelsIniTest(_ConfigTempCase):
    """The router refuses to start when models.ini is absent, and nothing ever
    created it: ensure_global() existed but had no callers at all."""

    def test_creates_the_file_when_missing(self):
        path = os.path.join(self.tmp, "models.ini")
        created = config.ensure_models_ini(path)
        self.assertTrue(created)
        self.assertTrue(os.path.exists(path))

    def test_created_file_has_a_global_section(self):
        path = os.path.join(self.tmp, "models.ini")
        config.ensure_models_ini(path)
        self.assertIn("*", config.read_sections(path))

    def test_existing_file_is_never_touched(self):
        path = os.path.join(self.tmp, "models.ini")
        original = "[*]\nctx-size = 4096\n\n[mymodel]\nmodel = /m.gguf\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(original)
        self.assertFalse(config.ensure_models_ini(path))
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), original)

    def test_creates_missing_parent_directory(self):
        path = os.path.join(self.tmp, "nested", "dir", "models.ini")
        self.assertTrue(config.ensure_models_ini(path))
        self.assertTrue(os.path.exists(path))


class RouterPortConflictTest(unittest.TestCase):
    """Port 8080 is a popular default (XAMPP, Apache). llama-server logged
    'couldn't bind HTTP server socket' and died, but start() had already
    returned success, so the dashboard just showed every model offline."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fake_bin = os.path.join(self.tmp, "llama-server")
        with open(self.fake_bin, "w") as f:      # must exist to reach the port check
            f.write("")
        self._saved = router_ctl.is_running
        self.spawned = []
        self._saved_popen = router_ctl.subprocess.Popen
        router_ctl.subprocess.Popen = lambda *a, **k: self.spawned.append(a)
        self._saved_cors = router_ctl.supports_cors_origins   # its --help probe also Popens
        router_ctl.supports_cors_origins = lambda server_bin: False

    def tearDown(self):
        router_ctl.is_running = self._saved
        router_ctl.subprocess.Popen = self._saved_popen
        router_ctl.supports_cors_origins = self._saved_cors

    def test_refuses_to_start_when_port_is_taken(self):
        router_ctl.is_running = lambda port: True
        ok, err = router_ctl.start(self.fake_bin, "m.ini", 8080, "127.0.0.1", "", self.tmp)
        self.assertFalse(ok)
        self.assertIn("8080", err)
        self.assertIn("in use", err.lower())

    def test_does_not_spawn_a_doomed_process(self):
        router_ctl.is_running = lambda port: True
        router_ctl.start(self.fake_bin, "m.ini", 8080, "127.0.0.1", "", self.tmp)
        self.assertEqual(self.spawned, [], "spawned llama-server onto a taken port")

    def test_starts_normally_when_the_port_is_free(self):
        router_ctl.is_running = lambda port: False
        ok, err = router_ctl.start(self.fake_bin, "m.ini", 8080, "127.0.0.1", "", self.tmp)
        self.assertTrue(ok, err)
        self.assertEqual(len(self.spawned), 1)

    def test_missing_binary_still_reported_first(self):
        router_ctl.is_running = lambda port: True
        ok, err = router_ctl.start("", "m.ini", 8080, "127.0.0.1", "", self.tmp)
        self.assertFalse(ok)
        self.assertIn("server_bin", err)


class BuildPathValidationTest(unittest.TestCase):
    """With server_bin blank (the intended first-run state) Rebuild ran
    `cmake -B  -S ` and surfaced CMake's own 'No build directory specified'."""

    def test_empty_source_is_rejected(self):
        err = BuildManager.validate_paths("", "/b", isdir=lambda p: True)
        self.assertTrue(err)
        self.assertIn("source", err.lower())

    def test_empty_build_dir_is_rejected(self):
        err = BuildManager.validate_paths("/s", "", isdir=lambda p: True)
        self.assertTrue(err)
        self.assertIn("build", err.lower())

    def test_nonexistent_source_is_rejected(self):
        err = BuildManager.validate_paths("/nope", "/b", isdir=lambda p: False)
        self.assertTrue(err)
        self.assertIn("/nope", err)

    def test_valid_paths_pass(self):
        self.assertEqual(BuildManager.validate_paths("/s", "/b", isdir=lambda p: True), "")

    def test_build_dir_need_not_exist_yet(self):
        """cmake creates -B itself; only the source tree must already be there."""
        self.assertEqual(
            BuildManager.validate_paths("/s", "/b", isdir=lambda p: p == "/s"), "")

    def test_run_build_fails_fast_without_invoking_cmake(self):
        bm = BuildManager(tempfile.mkdtemp())
        calls = []
        bm._stream = lambda cmd, cwd=None: calls.append(cmd) or 0
        bm.run_build("", "", {}, pull=False)
        self.assertEqual(calls, [], "invoked cmake despite unset paths")
        self.assertEqual(bm.state["phase"], "failed")
        self.assertIn("source", bm.tail().lower())


class PathRefreshTest(unittest.TestCase):
    """winget/choco edit the machine PATH, but shutil.which() reads the copy
    this process inherited at launch - so a freshly installed ninja stayed
    MISSING until LlamaForge was restarted."""

    def test_refresh_path_is_safe_to_call(self):
        before = os.environ.get("PATH", "")
        try:
            osplat.refresh_path()
        finally:
            if not os.environ.get("PATH"):
                os.environ["PATH"] = before
        self.assertTrue(os.environ.get("PATH"), "refresh_path() emptied PATH")

    def test_refresh_path_reports_whether_it_did_anything(self):
        self.assertIsInstance(osplat.refresh_path(), bool)

    @unittest.skipUnless(osplat.IS_WIN, "registry PATH is Windows-only")
    def test_windows_refresh_keeps_existing_entries(self):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + "C:\\lf-sentinel"
        osplat.refresh_path()
        self.assertIn("C:\\lf-sentinel", os.environ["PATH"],
                      "refresh_path() dropped process-local PATH entries")


if __name__ == "__main__":
    unittest.main()
