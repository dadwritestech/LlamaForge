import conftest_paths  # noqa: F401
import json
import threading
import unittest
from unittest import mock

import config
import routes
from routes import ApiError, Req


SECRET = "route-secret-" + "r" * 32


class GetNetworkRouteTest(unittest.TestCase):
    def test_supported_stored_conditions_map_to_redacted_status(self):
        cases = [
            ("127.0.0.1", "", "local", "local"),
            ("127.0.0.1", "legacy-route-key", "local", "local_keyed"),
            ("127.0.0.1", "s" * 32, "local", "local_keyed"),
            ("0.0.0.0", "legacy-route-key", "lan", "protected_legacy"),
            ("0.0.0.0", "s" * 32, "lan", "protected"),
        ]
        for host, key, scope, configured in cases:
            stored = {"router_host": host, "router_api_key": key,
                      "router_port": 8080}
            with self.subTest(host=host, configured=configured), \
                 mock.patch.object(routes, "cfg", return_value=stored), \
                 mock.patch.object(routes.router_ctl, "lan_ip",
                                   return_value="192.168.1.9"), \
                 mock.patch.object(routes.router_ctl, "is_running",
                                   return_value=False):
                status, out = routes.get_network(Req())
            self.assertEqual(status, 200)
            self.assertEqual(out["access_scope"], scope)
            self.assertEqual(out["configured_security_status"], configured)
            if key:
                self.assertNotIn(key, json.dumps(out))

    def test_unsafe_legacy_is_redacted_and_observation_is_qualified(self):
        stored = {
            "router_host": "192.168.1.50",
            "router_api_key": SECRET,
            "router_port": 8080,
        }
        with mock.patch.object(routes, "cfg", return_value=stored), \
             mock.patch.object(routes.router_ctl, "lan_ip", return_value="192.168.1.9"), \
             mock.patch.object(routes.router_ctl, "is_running", return_value=True), \
             mock.patch.object(config, "update") as update:
            status, out = routes.get_network(Req())
        self.assertEqual(status, 200)
        self.assertEqual(out["access_scope"], "legacy")
        self.assertEqual(out["configured_security_status"], "unsafe_legacy")
        self.assertEqual(out["listener_status"], "listening")
        self.assertTrue(out["router_running"])
        self.assertTrue(out["remediation_required"])
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertNotIn("router_api_key", out)
        update.assert_not_called()


class PostNetworkRouteTest(unittest.TestCase):
    def _call(self, current, body, restart=(True, ""), running=False):
        saved = {}
        updated = dict(current)

        def persist(changes):
            saved.update(changes)
            updated.update(changes)
            return dict(updated)

        with mock.patch.object(routes, "cfg", return_value=current), \
             mock.patch.object(config, "update", side_effect=persist) as update, \
             mock.patch.object(routes.router_ctl, "restart",
                               return_value=restart) as restart_call, \
             mock.patch.object(routes.router_ctl, "is_running",
                               return_value=running), \
             mock.patch.object(routes.router_ctl, "lan_ip",
                               return_value="192.168.1.9"):
            result = routes.post_network(Req(body=body))
        return result, saved, update, restart_call

    def test_invalid_transition_neither_saves_nor_restarts(self):
        current = {"router_host": "127.0.0.1", "router_api_key": "",
                   "router_port": 8080, "server_bin": "server"}
        with mock.patch.object(routes, "cfg", return_value=current), \
             mock.patch.object(config, "update") as update, \
             mock.patch.object(routes.router_ctl, "restart") as restart:
            with self.assertRaises(ApiError) as cm:
                routes.post_network(Req(body={
                    "access_scope": "lan", "key_action": "keep"}))
        self.assertEqual(cm.exception.status, 400)
        update.assert_not_called()
        restart.assert_not_called()

    def test_generate_saves_before_restart_and_returns_key_once(self):
        current = {"router_host": "127.0.0.1", "router_api_key": "",
                   "router_port": 8080, "server_bin": "server"}
        events = []

        def persist(changes):
            events.append("save")
            return dict(current, **changes)

        def restart(*args):
            events.append("restart")
            return True, ""

        with mock.patch.object(routes, "cfg", return_value=current), \
             mock.patch.object(routes.network_policy, "generate_key",
                               return_value=SECRET), \
             mock.patch.object(config, "update", side_effect=persist), \
             mock.patch.object(routes.router_ctl, "restart", side_effect=restart), \
             mock.patch.object(routes.router_ctl, "is_running", return_value=False), \
             mock.patch.object(routes.router_ctl, "lan_ip", return_value="192.168.1.9"):
            status, out = routes.post_network(Req(body={
                "access_scope": "lan", "key_action": "generate"}))
        self.assertEqual(status, 200)
        self.assertEqual(events, ["save", "restart"])
        self.assertEqual(out["generated_api_key"], SECRET)
        self.assertEqual(out["restart_status"], "starting")
        self.assertFalse(out["router_running"])
        redacted = dict(out)
        redacted.pop("generated_api_key")
        self.assertNotIn(SECRET, json.dumps(redacted))

    def test_replace_restarts_with_exact_validated_settings(self):
        current = {"router_host": "127.0.0.1", "router_api_key": "",
                   "router_port": 8080, "server_bin": "server",
                   "models_ini": "models.ini"}
        (status, out), saved, _, restart = self._call(
            current,
            {"access_scope": "lan", "key_action": "replace",
             "api_key": SECRET},
            running=True,
        )
        self.assertEqual(status, 200)
        self.assertEqual(saved, {
            "router_host": "0.0.0.0", "router_api_key": SECRET})
        self.assertEqual(restart.call_args.args[3:5], ("0.0.0.0", SECRET))
        self.assertEqual(out["restart_status"], "running")
        self.assertTrue(out["ok"])
        self.assertNotIn("generated_api_key", out)
        self.assertNotIn(SECRET, json.dumps(out))

    def test_failed_restart_is_saved_not_running_and_scrubs_error(self):
        current = {"router_host": "127.0.0.1", "router_api_key": SECRET,
                   "router_port": 8080, "server_bin": "server"}
        (status, out), saved, _, _ = self._call(
            current,
            {"access_scope": "lan", "key_action": "keep"},
            restart=(False, "could not launch with " + SECRET),
            running=False,
        )
        self.assertEqual(status, 500)
        self.assertEqual(saved["router_host"], "0.0.0.0")
        self.assertFalse(out["ok"])
        self.assertTrue(out["saved"])
        self.assertEqual(out["restart_status"], "failed")
        self.assertEqual(out["configured_security_status"], "protected")
        self.assertNotIn(SECRET, json.dumps(out))

    def test_generation_failure_is_generic_and_never_mutates(self):
        current = {"router_host": "127.0.0.1", "router_api_key": "",
                   "router_port": 8080, "server_bin": "server"}
        with mock.patch.object(routes, "cfg", return_value=current), \
             mock.patch.object(
                 routes.network_policy, "generate_key",
                 side_effect=RuntimeError("generation failed with " + SECRET)), \
             mock.patch.object(config, "update") as update, \
             mock.patch.object(routes.router_ctl, "restart") as restart, \
             self.assertRaises(ApiError) as cm:
            routes.post_network(Req(body={
                "access_scope": "lan", "key_action": "generate"}))
        self.assertEqual(cm.exception.status, 500)
        self.assertNotIn(SECRET, str(cm.exception))
        update.assert_not_called()
        restart.assert_not_called()

    def test_save_failure_is_generic_and_stops_before_lifecycle_work(self):
        current = {"router_host": "127.0.0.1", "router_api_key": "",
                   "router_port": 8080, "server_bin": "server"}
        with mock.patch.object(routes, "cfg", return_value=current), \
             mock.patch.object(
                 config, "update",
                 side_effect=OSError("save refused with " + SECRET)) as update, \
             mock.patch.object(routes.router_ctl, "restart") as restart, \
             mock.patch.object(routes.router_ctl, "is_running") as running, \
             mock.patch.object(routes.router_ctl, "lan_ip") as lan_ip, \
             self.assertRaises(ApiError) as cm:
            routes.post_network(Req(body={
                "access_scope": "lan", "key_action": "replace",
                "api_key": SECRET}))
        self.assertEqual(cm.exception.status, 500)
        self.assertEqual(str(cm.exception), "network settings could not be saved")
        self.assertNotIn(SECRET, str(cm.exception))
        update.assert_called_once_with({
            "router_host": "0.0.0.0", "router_api_key": SECRET})
        restart.assert_not_called()
        running.assert_not_called()
        lan_ip.assert_not_called()

    def test_restart_exception_still_returns_saved_generated_key_once(self):
        current = {"router_host": "127.0.0.1", "router_api_key": "",
                   "router_port": 8080, "server_bin": "server"}
        updated = dict(current)

        def persist(changes):
            updated.update(changes)
            return dict(updated)

        with mock.patch.object(routes, "cfg", return_value=current), \
             mock.patch.object(routes.network_policy, "generate_key",
                               return_value=SECRET), \
             mock.patch.object(config, "update", side_effect=persist), \
             mock.patch.object(
                 routes.router_ctl, "restart",
                 side_effect=PermissionError("launch refused with " + SECRET)), \
             mock.patch.object(routes.router_ctl, "is_running", return_value=False), \
             mock.patch.object(routes.router_ctl, "lan_ip", return_value="192.168.1.9"):
            status, out = routes.post_network(Req(body={
                "access_scope": "lan", "key_action": "generate"}))
        self.assertEqual(status, 500)
        self.assertTrue(out["saved"])
        self.assertEqual(out["restart_status"], "failed")
        self.assertEqual(out["generated_api_key"], SECRET)
        redacted = dict(out)
        redacted.pop("generated_api_key")
        self.assertNotIn(SECRET, json.dumps(redacted))

    def test_network_changes_are_one_serialized_router_transaction(self):
        """A second tab must not read or restart during the first change."""
        original = "original-route-key-" + "o" * 20
        generated = "generated-route-key-" + "g" * 20
        state = {
            "router_host": "0.0.0.0", "router_api_key": original,
            "router_port": 8080, "server_bin": "server",
        }
        first_restart_entered = threading.Event()
        release_first_restart = threading.Event()
        second_cfg_read = threading.Event()
        cfg_reads = 0
        restart_keys = []
        results = []

        def load_current():
            nonlocal cfg_reads
            cfg_reads += 1
            if cfg_reads == 2:
                second_cfg_read.set()
            return dict(state)

        def persist(changes):
            state.update(changes)
            return dict(state)

        def restart(*args):
            restart_keys.append(args[4])
            if len(restart_keys) == 1:
                first_restart_entered.set()
                self.assertTrue(release_first_restart.wait(2))
            return True, ""

        def change(body):
            try:
                results.append(routes.post_network(Req(body=body)))
            except Exception as exc:  # make worker failures visible to the test
                results.append(exc)

        with mock.patch.object(routes, "cfg", side_effect=load_current), \
             mock.patch.object(routes.network_policy, "generate_key",
                               return_value=generated), \
             mock.patch.object(config, "update", side_effect=persist), \
             mock.patch.object(routes.router_ctl, "restart", side_effect=restart), \
             mock.patch.object(routes.router_ctl, "is_running", return_value=True), \
             mock.patch.object(routes.router_ctl, "lan_ip", return_value="192.168.1.9"):
            first = threading.Thread(target=change, args=({
                "access_scope": "lan", "key_action": "generate"},))
            second = threading.Thread(target=change, args=({
                "access_scope": "lan", "key_action": "keep"},))
            first.start()
            self.assertTrue(first_restart_entered.wait(2))
            second.start()
            try:
                self.assertFalse(
                    second_cfg_read.wait(0.25),
                    "second request read config while first restart was in flight")
            finally:
                release_first_restart.set()
            first.join(2)
            second.join(2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not isinstance(result, Exception)
                            for result in results), results)
        self.assertEqual(restart_keys, [generated, generated])
        self.assertEqual(state["router_api_key"], generated)

    def test_socket_construction_failure_does_not_lose_generated_key(self):
        current = {"router_host": "127.0.0.1", "router_api_key": "",
                   "router_port": 8080, "server_bin": "server"}

        with mock.patch.object(routes, "cfg", return_value=current), \
             mock.patch.object(routes.network_policy, "generate_key",
                               return_value=SECRET), \
             mock.patch.object(config, "update",
                               return_value=dict(current, router_host="0.0.0.0",
                                                 router_api_key=SECRET)), \
             mock.patch.object(routes.router_ctl, "restart", return_value=(True, "")), \
             mock.patch.object(routes.router_ctl, "is_running", return_value=True), \
             mock.patch.object(routes.router_ctl.socket, "socket",
                               side_effect=OSError("socket table exhausted")):
            status, out = routes.post_network(Req(body={
                "access_scope": "lan", "key_action": "generate"}))

        self.assertEqual(status, 200)
        self.assertEqual(out["generated_api_key"], SECRET)
        self.assertIsNone(out["lan_ip"])

    def test_old_shape_is_narrow_and_still_fails_closed(self):
        current = {"router_host": "127.0.0.1", "router_api_key": SECRET,
                   "router_port": 8080, "server_bin": "server"}
        (status, _), saved, _, _ = self._call(
            current, {"host": "0.0.0.0"}, running=True)
        self.assertEqual(status, 200)
        self.assertEqual(saved["router_api_key"], SECRET)
        for body in ({"host": "192.168.1.9"},
                     {"host": "0.0.0.0", "api_key": ""}):
            with self.subTest(body=body), \
                 mock.patch.object(routes, "cfg",
                                   return_value=dict(current, router_api_key="")), \
                 mock.patch.object(config, "update") as update, \
                 mock.patch.object(routes.router_ctl, "restart") as restart, \
                 self.assertRaises(ApiError):
                routes.post_network(Req(body=body))
            update.assert_not_called()
            restart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
