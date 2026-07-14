import conftest_paths  # noqa: F401  (puts backend/ on sys.path)
import unittest
from unittest import mock
import wsl


class TestWinToWsl(unittest.TestCase):
    def test_drive_path(self):
        self.assertEqual(wsl.win_to_wsl(r"D:\LlamaForge\x"), "/mnt/d/LlamaForge/x")

    def test_forward_slash_input(self):
        self.assertEqual(wsl.win_to_wsl("C:/Users/a/b"), "/mnt/c/Users/a/b")

    def test_already_posix(self):
        self.assertEqual(wsl.win_to_wsl("/home/me/x"), "/home/me/x")


class TestListDistros(unittest.TestCase):
    def test_parses_utf16_verbose_output(self):
        raw = "  NAME      STATE    VERSION\n* Ubuntu    Running  2\n  Debian    Stopped  2\n"
        with mock.patch.object(wsl, "_run_text", return_value=raw):
            distros = wsl.list_distros()
        self.assertEqual(distros, [
            {"name": "Ubuntu", "state": "Running", "version": "2", "default": True},
            {"name": "Debian", "state": "Stopped", "version": "2", "default": False},
        ])

    def test_no_wsl_returns_empty(self):
        with mock.patch.object(wsl, "_run_text", side_effect=FileNotFoundError()):
            self.assertEqual(wsl.list_distros(), [])


class TestRun(unittest.TestCase):
    def test_run_builds_bash_lc_invocation(self):
        with mock.patch("subprocess.run") as sr:
            sr.return_value = mock.Mock(returncode=0, stdout="hi\n", stderr="")
            code, out, err = wsl.run("echo hi", distro="Ubuntu")
        args = sr.call_args[0][0]
        self.assertEqual(args[:5], ["wsl.exe", "-d", "Ubuntu", "--", "bash"])
        self.assertIn("echo hi", args[-1])
        self.assertEqual((code, out.strip()), (0, "hi"))


if __name__ == "__main__":
    unittest.main()
