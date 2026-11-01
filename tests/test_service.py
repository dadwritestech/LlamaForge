import conftest_paths  # noqa: F401
import os
import plistlib
import tempfile
import unittest
from unittest import mock

import service


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

    def test_windows_is_not_supported_yet(self):
        with self.assertRaises(service.Unsupported):
            service.enable(self.here, home=self.home, system="Windows", path="", uid=0)


if __name__ == "__main__":
    unittest.main()
