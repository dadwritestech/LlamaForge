import conftest_paths  # noqa: F401
import unittest
import scanner
from builder import BuildManager


class TestPosixRoots(unittest.TestCase):
    def test_home_plus_existing_mounts(self):
        exists = {"/mnt", "/data"}
        roots = scanner.posix_roots("/home/u", ["/mnt", "/media", "/srv", "/data"],
                                    isdir=lambda p: p in exists)
        self.assertEqual(roots, ["/home/u", "/mnt", "/data"])

    def test_home_only_when_no_mounts(self):
        roots = scanner.posix_roots("/Users/u", ["/Volumes"], isdir=lambda p: False)
        self.assertEqual(roots, ["/Users/u"])


class TestBinariesDir(unittest.TestCase):
    def test_prefers_msvc_release_dir(self):
        exists = {"b/bin/Release".replace("/", __import__("os").sep),
                  "b/bin".replace("/", __import__("os").sep)}
        import os
        d = BuildManager.binaries_dir("b", isdir=lambda p: p in exists)
        self.assertEqual(d, os.path.join("b", "bin", "Release"))

    def test_falls_back_to_flat_bin(self):
        import os
        flat = os.path.join("b", "bin")
        d = BuildManager.binaries_dir("b", isdir=lambda p: p == flat)
        self.assertEqual(d, flat)

    def test_none_when_no_build_yet(self):
        self.assertIsNone(BuildManager.binaries_dir("b", isdir=lambda p: False))


if __name__ == "__main__":
    unittest.main()
