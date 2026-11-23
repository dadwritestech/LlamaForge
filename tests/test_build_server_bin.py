"""Record where a finished build actually put llama-server (issue #6).

bootstrap guesses `<src>/build/bin/llama-server`, which is right for Ninja/Make
but wrong for MSVC's multi-config generator (`bin/Release/llama-server.exe`).
The reporter had to hand-edit config.json after every build. The build already
knows where the binary landed, so it should say so.
"""
import conftest_paths  # noqa: F401
import json, os, tempfile, unittest

import config, routes
from builder import BuildManager


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("")
    return path


class LocateServerBinTest(unittest.TestCase):
    """Pure lookup over the build tree - no config, no side effects."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_none_without_a_build_dir(self):
        self.assertIsNone(BuildManager.locate_server_bin(""))

    def test_none_when_nothing_was_built(self):
        self.assertIsNone(BuildManager.locate_server_bin(self.tmp))

    def test_finds_msvc_release_exe(self):
        want = _touch(os.path.join(self.tmp, "bin", "Release", "llama-server.exe"))
        self.assertEqual(BuildManager.locate_server_bin(self.tmp), want)

    def test_finds_flat_bin_binary(self):
        want = _touch(os.path.join(self.tmp, "bin", "llama-server"))
        self.assertEqual(BuildManager.locate_server_bin(self.tmp), want)

    def test_prefers_release_over_flat_bin(self):
        _touch(os.path.join(self.tmp, "bin", "llama-server"))
        want = _touch(os.path.join(self.tmp, "bin", "Release", "llama-server.exe"))
        self.assertEqual(BuildManager.locate_server_bin(self.tmp), want)

    def test_ignores_a_bin_dir_without_the_server(self):
        _touch(os.path.join(self.tmp, "bin", "llama-cli"))
        self.assertIsNone(BuildManager.locate_server_bin(self.tmp))


class BuildRecordsBinaryTest(unittest.TestCase):
    """A successful build reports the binary; a failed one must not."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "src")
        self.build = os.path.join(self.tmp, "build")
        os.makedirs(self.src)
        self.seen = []
        self.bm = BuildManager(os.path.join(self.tmp, "logs"),
                               on_built=self.seen.append)
        self.bm._stream = lambda cmd, cwd=None, env=None: 0      # never invoke cmake

    def test_success_reports_the_located_binary(self):
        want = _touch(os.path.join(self.build, "bin", "llama-server"))
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(self.bm.state["phase"], "done")
        self.assertEqual(self.seen, [want])
        self.assertEqual(self.bm.state.get("server_bin"), want)

    def test_success_without_a_binary_reports_nothing(self):
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(self.bm.state["phase"], "done")
        self.assertEqual(self.seen, [])

    def test_failed_build_reports_nothing(self):
        _touch(os.path.join(self.build, "bin", "llama-server"))
        self.bm._stream = lambda cmd, cwd=None, env=None: 1      # cmake configure fails
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(self.bm.state["phase"], "failed")
        self.assertEqual(self.seen, [])

    def test_a_raising_callback_never_fails_the_build(self):
        _touch(os.path.join(self.build, "bin", "llama-server"))
        def boom(_):
            raise OSError("config.json is read-only")
        bm = BuildManager(os.path.join(self.tmp, "logs2"), on_built=boom)
        bm._stream = lambda cmd, cwd=None, env=None: 0
        bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(bm.state["phase"], "done")
        self.assertIn("read-only", bm.tail())


class RecordServerBinPolicyTest(unittest.TestCase):
    """Fill in a missing or stale path, but never overrule a working one the
    user chose deliberately (they may have moved the binary themselves)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")
        self.built = _touch(os.path.join(self.tmp, "build", "bin", "llama-server"))

    def tearDown(self):
        config.CONFIG = self._saved

    def _write_cfg(self, **keys):
        with open(config.CONFIG, "w", encoding="utf-8") as f:
            json.dump(keys, f)

    def test_sets_an_unconfigured_path(self):
        self._write_cfg(server_bin="")
        self.assertTrue(routes._record_server_bin("server_bin", self.built))
        self.assertEqual(config.load()["server_bin"], self.built)

    def test_replaces_a_path_that_does_not_exist(self):
        self._write_cfg(server_bin=os.path.join(self.tmp, "guessed", "llama-server"))
        self.assertTrue(routes._record_server_bin("server_bin", self.built))
        self.assertEqual(config.load()["server_bin"], self.built)

    def test_leaves_an_existing_working_path_alone(self):
        chosen = _touch(os.path.join(self.tmp, "chosen", "llama-server"))
        self._write_cfg(server_bin=chosen)
        self.assertFalse(routes._record_server_bin("server_bin", self.built))
        self.assertEqual(config.load()["server_bin"], chosen)

    def test_writes_the_ik_llama_key_for_that_engine(self):
        self._write_cfg(ik_llama_server_bin="")
        self.assertTrue(routes._record_server_bin("ik_llama_server_bin", self.built))
        c = config.load()
        self.assertEqual(c["ik_llama_server_bin"], self.built)
        self.assertEqual(c["server_bin"], "", "wrote the wrong engine's key")


if __name__ == "__main__":
    unittest.main()
