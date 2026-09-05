import conftest_paths  # noqa: F401
import json
import os
import pathlib
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for(path, seconds=5):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()


class RunnerSourceContractTest(unittest.TestCase):
    def test_windows_prefers_py_falls_back_and_reuses_resolution(self):
        text = (ROOT / "run.ps1").read_text(encoding="utf-8-sig")
        self.assertLess(text.index("Get-Command py"), text.index("Get-Command python"))
        self.assertIn(r"backend\network_policy.py", text)
        self.assertIn("Test-LlamaForgePython", text)
        self.assertIn("& $pythonFile @preflightArgs", text)
        self.assertIn("-FilePath $pythonFile", text)
        self.assertIn("LLAMAFORGE_NO_BROWSER", text)

    def test_posix_preflights_with_python3_and_keeps_dashboard_after_guard(self):
        text = (ROOT / "run.sh").read_text(encoding="utf-8")
        preflight = 'python3 "$here/backend/network_policy.py" --preflight "$cfg"'
        panel = '(cd "$here/backend" && nohup python3 server.py'
        self.assertIn(preflight, text)
        self.assertIn(panel, text)
        self.assertLess(text.index(preflight), text.index(panel))
        self.assertIn("LLAMAFORGE_NO_BROWSER", text)


class RunnerIntegrationMixin:
    runner_name = ""

    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self.tmp_obj.name)
        self.backend = self.tmp / "backend"
        self.backend.mkdir()
        shutil.copy2(ROOT / "backend" / "network_policy.py",
                     self.backend / "network_policy.py")
        shutil.copy2(ROOT / self.runner_name, self.tmp / self.runner_name)
        self.router_marker = self.tmp / "router-started"
        self.panel_marker = self.tmp / "panel-started"
        self.bad_py_marker = self.tmp / "bad-py-probed"
        (self.backend / "server.py").write_text(
            "import os\n"
            "open(os.environ['RUNNER_PANEL_MARKER'], 'w').write('started')\n",
            encoding="utf-8",
        )
        self.router_port = _free_port()
        self.panel_port = _free_port()
        self.models_ini = self.tmp / "models.ini"
        if os.name == "nt":
            self.server_bin = self.tmp / "fake-router.cmd"
            self.server_bin.write_text(
                "@echo off\r\n"
                "@echo started>\"%RUNNER_ROUTER_MARKER%\"\r\n",
                encoding="utf-8",
            )
        else:
            self.server_bin = self.tmp / "fake-router"
            self.server_bin.write_text(
                "#!/bin/sh\nprintf started > \"$RUNNER_ROUTER_MARKER\"\n",
                encoding="utf-8",
            )
            self.server_bin.chmod(
                self.server_bin.stat().st_mode | stat.S_IXUSR)
        config = {
            "router_port": self.router_port,
            "panel_port": self.panel_port,
            "router_host": "0.0.0.0",
            "router_api_key": "",
            "server_bin": str(self.server_bin),
            "models_ini": str(self.models_ini),
            "active_engine": "llamacpp",
        }
        (self.tmp / "config.json").write_text(
            json.dumps(config), encoding="utf-8")

    def tearDown(self):
        for _ in range(20):
            try:
                self.tmp_obj.cleanup()
                return
            except PermissionError:
                time.sleep(0.05)
        self.tmp_obj.cleanup()

    def _environment(self, reported_listener=-1):
        env = os.environ.copy()
        env.update({
            "RUNNER_ROUTER_MARKER": str(self.router_marker),
            "RUNNER_PANEL_MARKER": str(self.panel_marker),
            "RUNNER_LISTEN_PORT": str(reported_listener),
            "LLAMAFORGE_NO_BROWSER": "1",
        })
        if os.name == "nt":
            tools = self.tmp / "test-bin"
            tools.mkdir(exist_ok=True)
            (tools / "py.cmd").write_text(
                "@echo off\r\n"
                "@echo probed>\"%RUNNER_BAD_PY_MARKER%\"\r\n"
                "@exit /b 9\r\n",
                encoding="utf-8",
            )
            (tools / "python.cmd").write_text(
                f"@echo off\r\n@\"{sys.executable}\" %*\r\n",
                encoding="utf-8",
            )
            env["RUNNER_BAD_PY_MARKER"] = str(self.bad_py_marker)
            env["PATH"] = str(tools) + os.pathsep + env.get("PATH", "")
            env["PATHEXT"] = ".CMD;" + env.get("PATHEXT", "")
        else:
            tools = self.tmp / "test-bin"
            tools.mkdir(exist_ok=True)
            lsof = tools / "lsof"
            lsof.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in *\"tcp:$RUNNER_LISTEN_PORT\"*) exit 0;; "
                "*) exit 1;; esac\n",
                encoding="utf-8",
            )
            lsof.chmod(lsof.stat().st_mode | stat.S_IXUSR)
            env["PATH"] = str(tools) + os.pathsep + env.get("PATH", "")
        return env

    def _run(self, reported_listener=-1):
        env = self._environment(reported_listener)
        if os.name == "nt":
            cmd = [POWERSHELL, "-NoProfile", "-File",
                   str(self.tmp / self.runner_name)]
        else:
            cmd = ["bash", str(self.tmp / self.runner_name)]
        return subprocess.run(
            cmd, cwd=self.tmp, env=env, text=True,
            capture_output=True, timeout=20, check=False)

    def test_unsafe_new_router_is_skipped_but_dashboard_starts(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.router_marker.exists())
        self.assertTrue(_wait_for(self.panel_marker), result.stdout + result.stderr)
        self.assertIn("repair Network Access", result.stdout + result.stderr)

    def test_existing_listener_is_left_reachable(self):
        held = socket.socket()
        held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        held.bind(("127.0.0.1", self.router_port))
        held.listen()
        self.addCleanup(held.close)
        result = self._run(reported_listener=self.router_port)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.router_marker.exists())
        with socket.create_connection(("127.0.0.1", self.router_port), timeout=2):
            pass

    @unittest.skipUnless(os.name == "nt", "Windows fallback only")
    def test_unusable_py_candidate_falls_back_to_working_python(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.bad_py_marker.exists())
        self.assertTrue(_wait_for(self.panel_marker), result.stdout + result.stderr)


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Windows runner only")
class RunPs1PreflightTest(RunnerIntegrationMixin, unittest.TestCase):
    runner_name = "run.ps1"


@unittest.skipUnless(os.name != "nt", "POSIX runner only")
class RunShPreflightTest(RunnerIntegrationMixin, unittest.TestCase):
    runner_name = "run.sh"


if __name__ == "__main__":
    unittest.main()
