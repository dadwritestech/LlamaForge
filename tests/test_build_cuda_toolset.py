"""Build from source into a fresh build dir on Windows with two CUDA toolkits.

The PR builds (prtry) found it first: with no CMake cache, nvcc came from PATH
(12.9) but MSBuild's CUDA targets from CUDA_PATH (13.2), the headers mixed and
the build failed on VS 2026. A fresh dir gets the toolset pinned. A dir that
already has a cache keeps it: CMake refuses a toolset that differs from the
cached one, and that build already works.
"""
import conftest_paths  # noqa: F401
import os, shutil, tempfile, unittest

import builder
from builder import BuildManager


class FreshDirToolset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, "src")
        self.build = os.path.join(self.tmp, "build")
        os.makedirs(self.src)
        self.ran = []
        self.bm = BuildManager(os.path.join(self.tmp, "logs"))
        self.bm._stream = lambda cmd, cwd=None: self.ran.append(cmd) or 0
        self.seen_flags = []
        real = builder._toolset

        def fake(flags):
            self.seen_flags.append(flags)
            return ["-T", "cuda=C:/CUDA/v13.2", "-DCUDAToolkit_ROOT=C:/CUDA/v13.2"]
        builder._toolset = fake
        self.addCleanup(setattr, builder, "_toolset", real)

    def configure(self):
        return next(c for c in self.ran if c[:2] == ["cmake", "-B"])

    def test_a_fresh_build_dir_is_pinned(self):
        self.bm.run_build(self.src, self.build, {"GGML_CUDA": "ON"}, pull=False)
        cfg = self.configure()
        self.assertEqual(cfg[-3:], ["-T", "cuda=C:/CUDA/v13.2", "-DCUDAToolkit_ROOT=C:/CUDA/v13.2"])
        self.assertEqual(self.seen_flags, [{"GGML_CUDA": "ON"}])

    def test_an_existing_cache_is_left_alone(self):
        os.makedirs(self.build)
        open(os.path.join(self.build, "CMakeCache.txt"), "w").close()
        self.bm.run_build(self.src, self.build, {"GGML_CUDA": "ON"}, pull=False)
        self.assertNotIn("-T", self.configure())


class ToolsetArgs(unittest.TestCase):
    def test_same_rule_as_pr_builds(self):
        import prtry
        self.assertIs(prtry.toolset_args, builder.toolset_args)
        env = {"CUDA_PATH": "C:/CUDA/v13.2"}
        yes = lambda p: True
        self.assertEqual(builder.toolset_args({"GGML_CUDA": "ON"}, env, "nt", yes),
                         ["-T", "cuda=C:/CUDA/v13.2", "-DCUDAToolkit_ROOT=C:/CUDA/v13.2"])
        self.assertEqual(builder.toolset_args({"GGML_CUDA": "ON"}, env, "posix", yes), [])


if __name__ == "__main__":
    unittest.main()
