"""stop.ps1 / stop.sh stop only what this LlamaForge copy started (review 05 #1).

They used to kill every llama-server on the machine and whatever held the
panel/router ports - a user's own llama-server, or XAMPP on 8080.
"""
import conftest_paths  # noqa: F401
import json, os, shutil, tempfile, unittest
from unittest import mock

import procs

CFG = {"router_port": 8080, "panel_port": 8090}
ROUTER, CHILD, GRANDCHILD, PANEL, STRANGER = 100, 101, 102, 200, 300


def table(**over):
    t = {
        ROUTER: (1, r"C:\lf\engines\b1\llama-server.exe"),
        CHILD: (ROUTER, r"C:\lf\engines\b1\llama-server.exe"),
        GRANDCHILD: (CHILD, r"C:\lf\engines\b1\llama-server.exe"),
        PANEL: (1, r"C:\lf\python\python.exe"),
        STRANGER: (1, r"C:\other\llama-server.exe"),
    }
    t.update(over)
    return t


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.logdir, True)

    def owners(self, router=ROUTER, panel=PANEL):
        return lambda port: {8080: router, 8090: panel}.get(port)

    def plan(self, procs_table=None, **owners):
        return procs.plan(CFG, self.logdir, procs_table or table(), self.owners(**owners))

    def test_recorded_router_its_children_and_the_panel(self):
        procs.write_pid(self.logdir, "router", ROUTER)
        procs.write_pid(self.logdir, "panel", PANEL)
        self.assertEqual(self.plan(), [("model instance", GRANDCHILD),
                                       ("model instance", CHILD),
                                       ("llama.cpp router", ROUTER),
                                       ("LlamaForge dashboard", PANEL)])

    def test_never_sweeps_an_unrelated_llama_server(self):
        procs.write_pid(self.logdir, "router", ROUTER)
        self.assertNotIn(STRANGER, [pid for _l, pid in self.plan()])

    def test_skips_a_port_owner_that_is_not_the_recorded_pid(self):
        """The router died and the user's own llama-server took the port."""
        procs.write_pid(self.logdir, "router", ROUTER)
        self.assertEqual(self.plan(router=STRANGER, panel=None), [])

    def test_skips_a_port_owner_that_is_not_a_llama_server(self):
        """XAMPP on 8080 - even when a reused PID matches the pidfile."""
        procs.write_pid(self.logdir, "router", ROUTER)
        t = table()
        t[ROUTER] = (1, r"C:\xampp\apache\bin\httpd.exe")
        self.assertEqual(self.plan(t, panel=None), [])

    def test_skips_a_panel_port_owner_that_is_not_python(self):
        t = table()
        t[PANEL] = (1, r"C:\Program Files\nodejs\node.exe")
        self.assertEqual(self.plan(t, router=None), [])

    def test_older_copy_without_pidfiles_still_stops_by_port_and_image(self):
        self.assertEqual([pid for _l, pid in self.plan()],
                         [GRANDCHILD, CHILD, ROUTER, PANEL])

    def test_garbage_pidfile_trusts_nothing(self):
        with open(procs.pidfile(self.logdir, "router"), "w") as f:
            f.write("not a pid")
        self.assertEqual(self.plan(panel=None), [])

    def test_nothing_on_the_ports(self):
        self.assertEqual(self.plan(router=None, panel=None), [])

    def test_never_stops_itself(self):
        with mock.patch.object(procs.os, "getpid", return_value=PANEL):
            self.assertEqual([pid for _l, pid in self.plan(router=None)], [])


class ImageTest(unittest.TestCase):
    def test_names(self):
        self.assertTrue(procs.is_llama("/opt/lf/engines/b/llama-server"))
        self.assertTrue(procs.is_llama(r"C:\x\LLAMA-SERVER.EXE"))
        self.assertFalse(procs.is_llama("httpd"))
        self.assertTrue(procs.is_python("/Library/Frameworks/Python.framework/x/Python"))
        self.assertTrue(procs.is_python(r"C:\lf\python\pythonw.exe"))
        self.assertFalse(procs.is_python("node"))


class ParseTest(unittest.TestCase):
    def test_ps(self):
        text = "  100     1 /opt/lf/llama-server\n  101   100 llama-server\nbad line\n"
        self.assertEqual(procs.parse_ps(text), {100: (1, "/opt/lf/llama-server"),
                                                101: (100, "llama-server")})

    def test_cim(self):
        rows = [{"ProcessId": 4, "ParentProcessId": 0, "ExecutablePath": None, "Name": "System"},
                {"ProcessId": 100, "ParentProcessId": 4,
                 "ExecutablePath": r"C:\lf\llama-server.exe", "Name": "llama-server.exe"}]
        self.assertEqual(procs.parse_cim(json.dumps(rows)),
                         {4: (0, "System"), 100: (4, r"C:\lf\llama-server.exe")})
        self.assertEqual(procs.parse_cim(json.dumps(rows[1])), {100: (4, r"C:\lf\llama-server.exe")})
        self.assertEqual(procs.parse_cim("garbage"), {})


class VllmTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_not_set_up_means_wsl_is_left_alone(self):
        with mock.patch.object(procs, "IS_WIN", True):
            self.assertIsNone(procs.vllm_command({"vllm_port": 8081}, self.root))

    def test_only_the_server_on_our_port(self):
        open(os.path.join(self.root, "vllm_models.json"), "w").close()
        with mock.patch.object(procs, "IS_WIN", True):
            cmd = procs.vllm_command({"vllm_port": 8123, "wsl_distro": "Ubuntu"}, self.root)
        self.assertEqual(cmd[:3], ["wsl.exe", "-d", "Ubuntu"])
        self.assertIn("--port 8123( |$)", cmd[-1])


class StopTest(unittest.TestCase):
    def test_removes_the_pidfiles_of_what_it_stopped(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        logdir = os.path.join(root, "logs")
        procs.write_pid(logdir, "router", ROUTER)
        procs.write_pid(logdir, "panel", PANEL)
        with open(os.path.join(root, "config.json"), "w") as f:
            json.dump(CFG, f)
        killed = []
        with mock.patch.object(procs, "table", return_value=table()), \
             mock.patch.object(procs, "pid_on_port",
                               side_effect=lambda p: {8080: ROUTER, 8090: PANEL}[p]), \
             mock.patch.object(procs, "kill", side_effect=killed.append), \
             mock.patch("builtins.print"):
            procs.stop(root)
        self.assertEqual(killed, [GRANDCHILD, CHILD, ROUTER, PANEL])
        self.assertIsNone(procs.read_pid(logdir, "router"))
        self.assertIsNone(procs.read_pid(logdir, "panel"))


class ScriptsTest(unittest.TestCase):
    """The shell side stays thin: no machine-wide sweeps."""
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def read(self, name):
        with open(os.path.join(self.ROOT, name), encoding="utf-8-sig") as f:
            return f.read()

    def test_stop_scripts_delegate_and_never_sweep(self):
        for name in ("stop.ps1", "stop.sh"):
            text = self.read(name)
            self.assertIn("procs.py", text, name)
            self.assertNotIn("Get-Process llama-server", text, name)
            self.assertNotIn("pkill", text, name)

    def test_launchers_record_the_router_pid(self):
        self.assertIn("router.pid", self.read("run.ps1"))
        self.assertIn("router.pid", self.read("run.sh"))

    def test_launchers_no_longer_pin_context(self):
        for name in ("run.ps1", "run.sh"):
            self.assertNotIn("ctx-size", self.read(name), name)


if __name__ == "__main__":
    unittest.main()
