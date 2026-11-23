"""Partial build success (issue #5 item 5).

The llama.cpp UI-asset step (npm/sharp) can fail while llama-server itself
builds. cmake returns one exit code for the whole build, so LlamaForge reported
BUILD FAILED even though the binary was there. Report "built, with warnings" -
but only when the binary is genuinely from THIS build, never a stale one from a
previous success (which would hide a real compile failure).
"""
import conftest_paths  # noqa: F401
import os, tempfile, time, unittest

from builder import BuildManager


def _touch(path, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class PartialBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "src")
        self.build = os.path.join(self.tmp, "build")
        os.makedirs(self.src)
        self.built = []
        self.bm = BuildManager(os.path.join(self.tmp, "logs"),
                               on_built=self.built.append)

    def _stub(self, build_rc, on_build_step=None):
        """Make _stream succeed for every phase except the build step, which
        returns build_rc. on_build_step fires when the build command runs."""
        def stream(cmd, cwd=None, env=None):
            is_build = "--build" in cmd
            if is_build:
                if on_build_step:
                    on_build_step()
                return build_rc
            return 0
        self.bm._stream = stream

    def test_fresh_binary_after_failed_step_is_warnings_not_failure(self):
        bin_path = os.path.join(self.build, "bin", "llama-server")
        # binary appears DURING the build step -> mtime is after `started`
        self._stub(build_rc=1, on_build_step=lambda: _touch(bin_path))
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(self.bm.state["phase"], "done_warnings")
        self.assertEqual(self.bm.state["returncode"], 0)
        self.assertTrue(self.bm.state.get("warning"))
        self.assertEqual(self.built, [bin_path])

    def test_stale_binary_after_failed_step_is_a_real_failure(self):
        # a binary from a PREVIOUS build: mtime well before this build starts
        bin_path = os.path.join(self.build, "bin", "llama-server")
        _touch(bin_path, mtime=time.time() - 3600)
        self._stub(build_rc=1)                     # build fails, touches nothing
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(self.bm.state["phase"], "failed")
        self.assertEqual(self.built, [], "recorded a stale binary as this build's output")

    def test_no_binary_after_failed_step_is_a_real_failure(self):
        self._stub(build_rc=1)
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(self.bm.state["phase"], "failed")
        self.assertEqual(self.built, [])

    def test_clean_build_is_plain_done(self):
        bin_path = os.path.join(self.build, "bin", "llama-server")
        self._stub(build_rc=0, on_build_step=lambda: _touch(bin_path))
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(self.bm.state["phase"], "done")
        self.assertIsNone(self.bm.state.get("warning"))
        self.assertEqual(self.built, [bin_path])

    def test_configure_failure_is_failure_even_with_a_binary(self):
        _touch(os.path.join(self.build, "bin", "llama-server"))
        # fail the configure step (the first non-git _stream), not the build step
        def stream(cmd, cwd=None, env=None):
            if cmd[:1] == ["cmake"] and "--build" not in cmd:
                return 1                            # configure fails
            return 0
        self.bm._stream = stream
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(self.bm.state["phase"], "failed")
        self.assertEqual(self.built, [])

    def test_warning_is_cleared_on_the_next_run(self):
        bin_path = os.path.join(self.build, "bin", "llama-server")
        self._stub(build_rc=1, on_build_step=lambda: _touch(bin_path))
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertTrue(self.bm.state.get("warning"))
        # a clean re-run must not leave the old warning hanging around
        self._stub(build_rc=0, on_build_step=lambda: _touch(bin_path))
        self.bm.run_build(self.src, self.build, {}, pull=False)
        self.assertEqual(self.bm.state["phase"], "done")
        self.assertIsNone(self.bm.state.get("warning"))


if __name__ == "__main__":
    unittest.main()
