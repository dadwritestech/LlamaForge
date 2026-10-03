import conftest_paths  # noqa: F401
import os, shutil, tempfile, unittest

import gguf, routes, scanner


def _touch(d, name, n):
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(b"\0" * n)
    return p


class TotalSizeTest(unittest.TestCase):
    """A split GGUF is registered by its first shard; its size is all shards
    (review 03 H5: a 3-shard 70B read as one third of its real size)."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_single_file(self):
        self.assertEqual(gguf.total_size(_touch(self.d, "m.gguf", 100)), 100)

    def test_shards_are_summed(self):
        first = _touch(self.d, "m-00001-of-00003.gguf", 100)
        _touch(self.d, "m-00002-of-00003.gguf", 200)
        _touch(self.d, "m-00003-of-00003.gguf", 50)
        _touch(self.d, "other-00002-of-00003.gguf", 999)
        self.assertEqual(gguf.total_size(first), 350)

    def test_missing_shard_counts_what_is_there(self):
        first = _touch(self.d, "m-00001-of-00002.gguf", 100)
        self.assertEqual(gguf.total_size(first), 100)

    def test_missing_file_raises_oserror(self):
        with self.assertRaises(OSError):
            gguf.total_size(os.path.join(self.d, "gone.gguf"))

    def test_registry_and_scan_report_the_whole_model(self):
        first = _touch(self.d, "m-00001-of-00002.gguf", 1024**3)
        _touch(self.d, "m-00002-of-00002.gguf", 1024**3)
        self.assertEqual(routes._file_gib(first), 2.0)
        self.assertEqual(scanner.build_entries([first])[0]["gib"], 2.0)


if __name__ == "__main__":
    unittest.main()
