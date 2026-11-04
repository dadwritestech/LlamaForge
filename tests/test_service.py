import conftest_paths  # noqa: F401
import os
import plistlib
import tempfile
import unittest
from unittest import mock

import service

_REAL_FILE = service._file


class ServiceFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = self.tmp.name
        self.here = "/opt/llama forge"
        self.calls = []
        run = mock.patch.object(service, "_run", side_effect=lambda argv: self.calls.append(argv) or 0)
        run.start()
        self.addCleanup(run.stop)

    def test_systemd_unit_runs_run_sh_without_a_browser_and_stops_with_stop_sh(self):
        text = service.systemd_unit(self.here, "/usr/bin:/bin")
        self.assertIn('ExecStart=/bin/bash "/opt/llama forge/run.sh"', text)
        self.assertIn('ExecStop=/bin/bash "/opt/llama forge/stop.sh"', text)
        self.assertIn("Environment=LLAMAFORGE_NO_BROWSER=1", text)
        self.assertIn("Type=oneshot", text)
        self.assertIn("RemainAfterExit=yes", text)
        self.assertIn("WantedBy=default.target", text)

    def test_systemd_escapes_percent_and_dollar(self):
        text = service.systemd_unit("/srv/100%$x", "/usr/bin")
        self.assertIn('"/srv/100%%$$x/run.sh"', text)

    def test_unsafe_paths_are_refused(self):
        for bad in ('/a"b', "/a\\b", "/a\nb", "relative/dir"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                service.systemd_unit(bad, "/usr/bin")

    def test_launchd_plist_is_valid_and_abandons_the_process_group(self):
        data = plistlib.loads(service.launchd_plist(self.here, "/usr/bin:/bin"))
        self.assertEqual(data["Label"], service.LABEL)
        self.assertEqual(data["ProgramArguments"], ["/bin/bash", "/opt/llama forge/run.sh"])
        self.assertTrue(data["RunAtLoad"])
        self.assertTrue(data["AbandonProcessGroup"])   # run.sh exits; its children must live
        self.assertEqual(data["EnvironmentVariables"]["LLAMAFORGE_NO_BROWSER"], "1")
        self.assertIn("/usr/bin", data["EnvironmentVariables"]["PATH"])

    def test_enable_on_linux_writes_the_user_unit_and_enables_it(self):
        msg = service.enable(self.here, home=self.home, system="Linux", path="/usr/bin", uid=1000)
        unit = os.path.join(self.home, ".config", "systemd", "user", "llamaforge.service")
        self.assertTrue(os.path.isfile(unit))
        self.assertEqual(self.calls, [["systemctl", "--user", "daemon-reload"],
                                      ["systemctl", "--user", "enable", "llamaforge.service"]])
        self.assertIn("loginctl enable-linger", msg)

    def test_enable_on_macos_writes_the_agent_and_bootstraps_it(self):
        service.enable(self.here, home=self.home, system="Darwin", path="/usr/bin", uid=501)
        plist = os.path.join(self.home, "Library", "LaunchAgents", service.LABEL + ".plist")
        self.assertTrue(os.path.isfile(plist))
        self.assertEqual(self.calls[-1], ["launchctl", "bootstrap", "gui/501", plist])

    def test_disable_removes_what_enable_wrote(self):
        for system in ("Linux", "Darwin"):
            with self.subTest(system=system):
                service.enable(self.here, home=self.home, system=system, path="/usr/bin", uid=1)
                self.assertTrue(service.status(home=self.home, system=system))
                service.disable(home=self.home, system=system, uid=1)
                self.assertFalse(service.status(home=self.home, system=system))

    def test_disable_when_nothing_is_installed_is_quiet(self):
        msg = service.disable(home=self.home, system="Linux", uid=1)
        self.assertEqual(self.calls, [])
        self.assertIn("not set up", msg)

    def test_windows_vbs_runs_run_ps1_hidden_without_a_browser(self):
        text = service.windows_vbs(r"C:\Users\me\Llama Forge")
        self.assertIn('sh.Environment("Process")("LLAMAFORGE_NO_BROWSER") = "1"', text)
        self.assertIn("-WindowStyle Hidden", text)
        self.assertIn(r'-File ""C:\Users\me\Llama Forge\run.ps1""', text)
        self.assertIn(", 0, False", text)   # hidden window, don't wait

    def test_windows_unsafe_paths_are_refused(self):
        for bad in ('C:\\a"b', "C:\\a\nb", "relative\\dir", "/opt/x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                service.windows_vbs(bad)

    def test_enable_on_windows_writes_the_startup_script_without_getuid(self):
        with mock.patch.object(service.os, "getuid", create=True, side_effect=AttributeError):
            msg = service.enable(r"C:\LlamaForge", home=self.home, system="Windows", path="")
        vbs = os.path.join(self.home, "AppData", "Roaming", "Microsoft", "Windows",
                           "Start Menu", "Programs", "Startup", "LlamaForge.vbs")
        self.assertTrue(os.path.isfile(vbs))
        with open(vbs, encoding="utf-8") as f:
            self.assertIn("run.ps1", f.read())
        self.assertEqual(self.calls, [])
        self.assertIn("starts when you log in", msg)

    def test_disable_on_windows_removes_the_startup_script(self):
        with mock.patch.object(service.os, "getuid", create=True, side_effect=AttributeError):
            service.enable(r"C:\LlamaForge", home=self.home, system="Windows", path="")
            self.assertTrue(service.status(home=self.home, system="Windows"))
            service.disable(home=self.home, system="Windows")
        self.assertFalse(service.status(home=self.home, system="Windows"))

    def test_windows_uses_appdata_when_no_home_is_given(self):
        with mock.patch.dict(service.os.environ, {"APPDATA": self.home}):
            self.assertEqual(service._file(None, "Windows"),
                             os.path.join(self.home, "Microsoft", "Windows", "Start Menu",
                                          "Programs", "Startup", "LlamaForge.vbs"))

    def test_other_systems_are_not_supported(self):
        with self.assertRaises(service.Unsupported):
            service.enable(self.here, home=self.home, system="FreeBSD", path="", uid=0)



class ServiceRouteTest(unittest.TestCase):
    """Setup > Startup > start at login: the only way in on Windows, which has
    no llamaforge command."""
    def setUp(self):
        import routes
        self.routes = routes
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = os.path.join(self.tmp.name, "Startup", "LlamaForge.vbs")
        for p in (mock.patch.object(service, "_file", return_value=self.target),
                  mock.patch.object(service, "_run", return_value=0)):
            p.start()
        self.addCleanup(mock.patch.stopall)

    def req(self, body=None):
        return mock.Mock(body=body, query={})

    def test_registered(self):
        self.assertIn("/api/service", self.routes.GET_ROUTES)
        self.assertIn("/api/service", self.routes.POST_ROUTES)

    def test_get_reports_off_then_on(self):
        st, out = self.routes.get_service(self.req())
        self.assertEqual((st, out["on"]), (200, False))
        with mock.patch.object(service.platform, "system", return_value="Windows"):
            st, out = self.routes.post_service(self.req({"on": True}))
            self.assertTrue(out["ok"])
            self.assertTrue(os.path.isfile(self.target))
            with open(self.target, encoding="utf-8") as f:
                self.assertIn(os.path.join(self.routes.ROOT, "run.ps1"), f.read())
            self.assertTrue(self.routes.get_service(self.req())[1]["on"])
            st, out = self.routes.post_service(self.req({"on": False}))
        self.assertTrue(out["ok"])
        self.assertFalse(os.path.exists(self.target))

    def test_unsupported_system_says_so(self):
        with mock.patch.object(service.platform, "system", return_value="FreeBSD"),              mock.patch.object(service, "_file", _REAL_FILE):
            st, out = self.routes.get_service(self.req())
            self.assertFalse(out["supported"])
            st, out = self.routes.post_service(self.req({"on": True}))
        self.assertFalse(out["ok"])
        self.assertIn("only for", out["error"])

    def test_on_must_be_a_bool(self):
        st, out = self.routes.post_service(self.req({"on": "yes"}))
        self.assertFalse(out["ok"])


if __name__ == "__main__":
    unittest.main()
