import conftest_paths  # noqa: F401
import json
import os
import shutil
import tempfile
import unittest

import prtry


SHA = "0123456789abcdef0123456789abcdef01234567"


def api(**kw):
    d = {"number": 1234, "title": "CUDA: faster MMQ for Blackwell", "state": "open",
         "merged_at": None, "draft": False, "user": {"login": "someone"},
         "head": {"sha": SHA}, "html_url": "https://github.com/ggml-org/llama.cpp/pull/1234"}
    d.update(kw)
    return d


class ParsePr(unittest.TestCase):
    def test_number_hash_and_url(self):
        for s in ["1234", " #1234 ", "https://github.com/ggml-org/llama.cpp/pull/1234",
                  "github.com/ggml-org/llama.cpp/pull/1234/files",
                  "https://github.com/ggerganov/llama.cpp/pull/1234#issuecomment-9"]:
            self.assertEqual(prtry.parse_pr(s), 1234, s)

    def test_rejects_other_repos_and_junk(self):
        for s in ["", "abc", "0", "-3", "12345678", None, 12,
                  "https://github.com/someone/llama.cpp/pull/1234",
                  "https://github.com/ggml-org/whisper.cpp/pull/1234",
                  "https://evil.example/ggml-org/llama.cpp/pull/1234"]:
            with self.assertRaises(ValueError, msg=repr(s)):
                prtry.parse_pr(s)


class PrInfo(unittest.TestCase):
    def test_open_pr(self):
        i = prtry.pr_info(api())
        self.assertEqual(i, {"number": 1234, "title": "CUDA: faster MMQ for Blackwell",
                             "author": "someone", "sha": SHA, "state": "open", "draft": False,
                             "url": "https://github.com/ggml-org/llama.cpp/pull/1234"})

    def test_merged_and_closed(self):
        self.assertEqual(prtry.pr_info(api(state="closed", merged_at="2026-11-01T00:00:00Z"))["state"], "merged")
        self.assertEqual(prtry.pr_info(api(state="closed"))["state"], "closed")

    def test_bad_sha_is_refused(self):
        with self.assertRaises(ValueError):
            prtry.pr_info(api(head={"sha": "nope; rm -rf"}))
        with self.assertRaises(ValueError):
            prtry.pr_info({"message": "Not Found"})

    def test_title_is_trimmed_and_capped(self):
        self.assertEqual(len(prtry.pr_info(api(title="x" * 500))["title"]), 200)


class Plan(unittest.TestCase):
    def test_install_name(self):
        self.assertEqual(prtry.install_name(1234, SHA), "pr-1234-0123456")

    def test_commands_pin_the_sha_and_build_two_targets(self):
        cmds = prtry.commands(1234, SHA, "W", "W/build", {"GGML_CUDA": "ON"}, jobs=8)
        flat = [" ".join(c) for c in cmds]
        self.assertEqual(flat[0], "git init -q W")
        self.assertEqual(flat[1], "git -C W fetch -q --depth 1 "
                                  "https://github.com/ggml-org/llama.cpp pull/1234/head")
        self.assertEqual(flat[2], "git -C W checkout -q --detach FETCH_HEAD")
        self.assertTrue(flat[3].startswith("cmake -B W/build -S W -DCMAKE_BUILD_TYPE=Release"))
        self.assertIn("-DGGML_CUDA=ON", flat[3])
        self.assertIn("-DLLAMA_CURL=OFF", flat[3])
        self.assertEqual(flat[4], "cmake --build W/build --config Release --parallel 8 "
                                  "--target llama-server llama-bench")
        # every command is a list of plain strings (no shell)
        self.assertTrue(all(isinstance(a, str) for c in cmds for a in c))

    def test_cuda_toolset_is_pinned_for_visual_studio(self):
        # two CUDA toolkits: nvcc from PATH (12.9) vs MSBuild's (13.2) mixed headers
        # and failed on VS 2026; -T cuda=<CUDA_PATH> pins nvcc, CUDAToolkit_ROOT the headers
        cuda = {"GGML_CUDA": "ON"}
        env = {"CUDA_PATH": r"C:\CUDA\v13.2"}
        isdir = lambda p: True
        pinned = ["-T", r"cuda=C:\CUDA\v13.2", r"-DCUDAToolkit_ROOT=C:\CUDA\v13.2"]
        self.assertEqual(prtry.toolset_args(cuda, env, "nt", isdir), pinned)
        self.assertEqual(prtry.toolset_args(cuda, dict(env, CMAKE_GENERATOR="Visual Studio 17 2022"), "nt", isdir),
                         pinned)
        # Ninja takes no toolset; no CUDA, no CUDA_PATH, a missing dir, or not Windows: nothing
        self.assertEqual(prtry.toolset_args(cuda, dict(env, CMAKE_GENERATOR="Ninja"), "nt", isdir), [])
        self.assertEqual(prtry.toolset_args({"GGML_VULKAN": "ON"}, env, "nt", isdir), [])
        self.assertEqual(prtry.toolset_args(cuda, {}, "nt", isdir), [])
        self.assertEqual(prtry.toolset_args(cuda, env, "nt", lambda p: False), [])
        self.assertEqual(prtry.toolset_args(cuda, env, "posix", isdir), [])
        cmds = prtry.commands(1, SHA, "W", "W/build", cuda, toolset=["-T", "cuda=X"])
        self.assertEqual(cmds[3][-2:], ["-T", "cuda=X"])

    def test_head_must_match(self):
        self.assertEqual(prtry.check_head(SHA + "\n", SHA), "")
        self.assertIn("changed", prtry.check_head("f" * 40, SHA))

    def test_manifest(self):
        m = prtry.manifest(prtry.pr_info(api()), {"GGML_CUDA": "ON"}, now=5.0)
        self.assertEqual(m["tag"], "PR #1234")
        self.assertEqual(m["label"], "PR #1234: CUDA: faster MMQ for Blackwell")
        self.assertEqual(m["channel"], "pr")
        self.assertEqual(m["pr"], 1234)
        self.assertEqual(m["sha"], SHA)
        self.assertEqual(m["installed_at"], 5.0)
        self.assertTrue(prtry.is_pr({"channel": "pr"}))
        self.assertFalse(prtry.is_pr({"channel": "nightly"}))


class Builder(unittest.TestCase):
    """PrBuilder with git/cmake faked: the build command drops binaries."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.engines = os.path.join(self.tmp, "engines")
        self.work = os.path.join(self.tmp, "work")
        self.ran, self.head, self.fail_at, self.hook = [], SHA, None, None
        self.b = prtry.PrBuilder(self.engines, self.work, self.tmp,
                                 run=self._run, capture=self._capture,
                                 fetch_json=lambda url: api())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, cmd, cwd=None):
        self.ran.append(cmd)
        if self.hook:
            self.hook(cmd)
        if self.fail_at and cmd[0] == self.fail_at[0] and self.fail_at[1] in cmd:
            return 1
        if cmd[:2] == ["cmake", "--build"]:
            out = os.path.join(cmd[2], "bin", "Release")
            os.makedirs(out, exist_ok=True)
            for n in ("llama-server.exe", "llama-bench.exe", "ggml-cuda.dll"):
                open(os.path.join(out, n), "w").close()
        return 0

    def _capture(self, cmd):
        return self.head + "\n"

    def build(self):
        info = self.b.look("1234")
        self.assertTrue(self.b.start(info, {"GGML_CUDA": "ON"}))
        self.b.join(10)
        return self.b.status()

    def test_look(self):
        self.assertEqual(self.b.look("#1234")["sha"], SHA)
        bad = prtry.PrBuilder(self.engines, self.work, self.tmp,
                              fetch_json=lambda url: {"message": "Not Found"})
        with self.assertRaises(ValueError):
            bad.look("99")
        def down(url):
            raise OSError("no network")
        with self.assertRaisesRegex(ValueError, "GitHub"):
            prtry.PrBuilder(self.engines, self.work, self.tmp, fetch_json=down).look("1")

    def test_build_installs_beside_the_prebuilt_ones(self):
        st = self.build()
        self.assertEqual(st["phase"], "done", st)
        d = os.path.join(self.engines, "pr-1234-0123456")
        self.assertEqual(st["install"], "pr-1234-0123456")
        self.assertEqual(sorted(os.listdir(d)),
                         ["ggml-cuda.dll", "lf-install.json", "llama-bench.exe", "llama-server.exe"])
        with open(os.path.join(d, "lf-install.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["channel"], "pr")
        self.assertFalse(os.path.exists(os.path.join(self.work, "pr-1234")))   # sources gone
        self.assertEqual(self.ran[0][:2], ["git", "init"])

    def test_a_changed_head_stops_before_cmake(self):
        self.head = "f" * 40
        st = self.build()
        self.assertEqual(st["phase"], "failed")
        self.assertIn("changed", st["error"])
        self.assertFalse(any(c[0] == "cmake" for c in self.ran))
        self.assertFalse(os.path.exists(self.engines) and os.listdir(self.engines))

    def test_a_failing_step_says_which(self):
        self.fail_at = ("cmake", "-S")
        st = self.build()
        self.assertEqual(st["phase"], "failed")
        self.assertEqual(st["error"], "cmake configure failed")
        self.fail_at = ("git", "fetch")
        self.assertIn("fetch", self.build()["error"])

    def test_cancel_stops_between_steps(self):
        self.hook = lambda cmd: cmd[:2] == ["git", "init"] and self.b.cancel()
        st = self.build()
        self.assertEqual(st["phase"], "cancelled")
        self.assertEqual(len(self.ran), 1)

    def test_one_build_at_a_time(self):
        import threading
        gate = threading.Event()
        self.hook = lambda cmd: gate.wait(5)
        info = self.b.look("1234")
        self.assertTrue(self.b.start(info, {}))
        self.assertFalse(self.b.start(info, {}))
        gate.set()
        self.b.join(10)

    def test_remove_only_pr_builds(self):
        self.build()
        os.makedirs(os.path.join(self.engines, "b7000-cuda"))
        with open(os.path.join(self.engines, "b7000-cuda", "lf-install.json"), "w") as f:
            json.dump({"tag": "b7000", "channel": "nightly"}, f)
        with self.assertRaises(ValueError):
            self.b.remove("b7000-cuda")
        with self.assertRaises(ValueError):
            self.b.remove("../etc")
        self.b.remove("pr-1234-0123456")
        self.assertEqual(os.listdir(self.engines), ["b7000-cuda"])


if __name__ == "__main__":
    unittest.main()
