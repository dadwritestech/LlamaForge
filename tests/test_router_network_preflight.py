import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import router_ctl


class RouterStartPreflightTest(unittest.TestCase):
    def test_unsafe_start_touches_no_binary_port_log_or_process(self):
        with mock.patch.object(router_ctl.os.path, "exists") as exists, \
             mock.patch.object(router_ctl, "is_running") as running, \
             mock.patch.object(router_ctl.os, "makedirs") as makedirs, \
             mock.patch("builtins.open") as open_file, \
             mock.patch.object(router_ctl.subprocess, "Popen") as popen:
            ok, error = router_ctl.start(
                "server", "models.ini", 8080, "0.0.0.0", "", "logs")
        self.assertFalse(ok)
        self.assertIn("API key", error)
        exists.assert_not_called()
        running.assert_not_called()
        makedirs.assert_not_called()
        open_file.assert_not_called()
        popen.assert_not_called()

    def test_valid_lan_start_reaches_spawn_and_passes_key(self):
        key = "k" * 32
        handles = mock.mock_open()
        with mock.patch.object(router_ctl.os.path, "exists", return_value=True), \
             mock.patch.object(router_ctl, "is_running", return_value=False), \
             mock.patch.object(router_ctl.os, "makedirs"), \
             mock.patch("builtins.open", handles), \
             mock.patch.object(router_ctl.subprocess, "Popen") as popen:
            ok, error = router_ctl.start(
                "server", "models.ini", 8080, "0.0.0.0", key, "logs")
        self.assertTrue(ok, error)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[argv.index("--host") + 1], "0.0.0.0")
        self.assertEqual(argv[argv.index("--api-key") + 1], key)


class RouterRestartPreflightTest(unittest.TestCase):
    def test_unsafe_restart_does_not_lookup_stop_kill_or_spawn(self):
        with mock.patch.object(router_ctl, "_pid_on_port") as pid, \
             mock.patch.object(router_ctl, "_kill") as kill, \
             mock.patch.object(router_ctl, "stop") as stop, \
             mock.patch.object(router_ctl.os.path, "exists") as exists, \
             mock.patch.object(router_ctl.subprocess, "Popen") as popen:
            ok, error = router_ctl.restart(
                "server", "models.ini", 8080, "0.0.0.0", "", "logs")
        self.assertFalse(ok)
        self.assertIn("API key", error)
        pid.assert_not_called()
        kill.assert_not_called()
        stop.assert_not_called()
        exists.assert_not_called()
        popen.assert_not_called()

    def test_valid_restart_stops_only_after_preflight_then_calls_start(self):
        key = "s" * 32
        events = []
        with mock.patch.object(
                router_ctl.network_policy, "start_error",
                side_effect=lambda host, api_key: events.append("preflight") or ""), \
             mock.patch.object(
                router_ctl, "stop",
                side_effect=lambda port: events.append("stop") or True), \
             mock.patch.object(
                router_ctl, "start",
                side_effect=lambda *args: events.append("start") or (True, "")):
            ok, error = router_ctl.restart(
                "server", "models.ini", 8080, "0.0.0.0", key, "logs")
        self.assertTrue(ok, error)
        self.assertEqual(events, ["preflight", "stop", "start"])


if __name__ == "__main__":
    unittest.main()
