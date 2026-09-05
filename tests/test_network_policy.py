import conftest_paths  # noqa: F401
import copy, json, os, re, tempfile, unittest
from pathlib import Path
from unittest import mock

import network_policy as np


class KeyStatusTest(unittest.TestCase):
    def test_key_classes_are_deterministic(self):
        cases = [
            (None, "absent"), ("", "absent"),
            ("a" * 32, "strong"), ("A0._~-" * 6, "strong"),
            ("secret", "legacy"), ('quote"key', "legacy"),
            (r"back\slash", "legacy"),
            ("has space", "invalid"), ("tab\tkey", "invalid"),
            ("line\nkey", "invalid"), ("nul\0key", "invalid"),
            ("del\x7fkey", "invalid"), ("unicode-π", "invalid"),
            ("x" * 257, "invalid"), (123, "invalid"),
        ]
        for value, expected in cases:
            with self.subTest(value=repr(value)):
                self.assertEqual(np.key_status(value), expected)

    def test_generated_key_is_strong(self):
        generated = "g" * 43
        with mock.patch.object(np.secrets, "token_urlsafe", return_value=generated) as token:
            key = np.generate_key()
        token.assert_called_once_with(32)
        self.assertIsNotNone(re.fullmatch(r"[A-Za-z0-9._~-]{32,256}", key))
        self.assertGreaterEqual(len(key), 32)
        self.assertLessEqual(len(key), 256)


class AssessmentTest(unittest.TestCase):
    def test_supported_states(self):
        cases = [
            ("127.0.0.1", "", "local", "local", True),
            ("127.0.0.1", "secret", "local", "local_keyed", True),
            ("127.0.0.1", "s" * 32, "local", "local_keyed", True),
            ("0.0.0.0", "s" * 32, "lan", "protected", True),
            ("0.0.0.0", "secret", "lan", "protected_legacy", True),
        ]
        for host, key, scope, status, allowed in cases:
            with self.subTest(host=host, key_status=np.key_status(key)):
                out = np.assess(host, key)
                self.assertEqual(out.access_scope, scope)
                self.assertEqual(out.configured_security_status, status)
                self.assertEqual(out.start_allowed, allowed)
                self.assertNotIn(key, repr(out)) if key else None

    def test_unsafe_states_never_start(self):
        invalid_lan_keys = [
            "", "bad key", "tab\tkey", "line\nkey", "nul\0key",
            "del\x7fkey", "unicode-π", "x" * 257, 123,
        ]
        for host, key in [
            *(("0.0.0.0", key) for key in invalid_lan_keys),
            ("192.168.1.20", "s" * 32), ("localhost", ""),
            ("::", "s" * 32), (None, ""),
        ]:
            with self.subTest(host=host):
                out = np.assess(host, key)
                self.assertEqual(out.access_scope, "legacy")
                self.assertEqual(out.configured_security_status, "unsafe_legacy")
                self.assertTrue(out.remediation_required)
                self.assertFalse(out.start_allowed)
                self.assertTrue(np.start_error(host, key))


class MutationTest(unittest.TestCase):
    def test_keep_preserves_a_legacy_lan_key(self):
        out = np.apply_request(
            {"router_host": "0.0.0.0", "router_api_key": "secret"},
            {"access_scope": "lan", "key_action": "keep"})
        self.assertEqual(out.router_host, "0.0.0.0")
        self.assertEqual(out.router_api_key, "secret")
        self.assertIsNone(out.generated_api_key)

    def test_generate_returns_the_same_strong_key_it_selects(self):
        generated = "g" * 43
        current = {"router_host": "127.0.0.1", "router_api_key": ""}
        body = {"access_scope": "lan", "key_action": "generate"}
        current_before, body_before = copy.deepcopy(current), copy.deepcopy(body)
        with mock.patch.object(np.secrets, "token_urlsafe", return_value=generated) as token:
            out = np.apply_request(
                current, body)
        token.assert_called_once_with(32)
        self.assertEqual(out.router_api_key, generated)
        self.assertEqual(out.generated_api_key, generated)
        self.assertEqual(out.assessment.configured_security_status, "protected")
        self.assertEqual(current, current_before)
        self.assertEqual(body, body_before)

    def test_replace_requires_a_new_strong_key(self):
        current = {"router_host": "127.0.0.1", "router_api_key": ""}
        with self.assertRaisesRegex(ValueError, "strong"):
            np.apply_request(current, {
                "access_scope": "lan", "key_action": "replace",
                "api_key": "short"})
        out = np.apply_request(current, {
            "access_scope": "lan", "key_action": "replace",
            "api_key": "r" * 32})
        self.assertEqual(out.router_api_key, "r" * 32)

    def test_clear_is_local_only(self):
        current = {"router_host": "0.0.0.0", "router_api_key": "s" * 32}
        with self.assertRaisesRegex(ValueError, "local"):
            np.apply_request(current, {
                "access_scope": "lan", "key_action": "clear"})
        out = np.apply_request(current, {
            "access_scope": "local", "key_action": "clear"})
        self.assertEqual((out.router_host, out.router_api_key),
                         ("127.0.0.1", ""))

    def test_contradictory_and_unknown_fields_are_rejected(self):
        current = {"router_host": "127.0.0.1", "router_api_key": ""}
        bad = [
            {"access_scope": "lan", "key_action": "generate", "api_key": "x" * 32},
            {"access_scope": "lan", "key_action": "replace"},
            {"access_scope": "public", "key_action": "keep"},
            {"access_scope": "local", "key_action": "erase"},
            {"access_scope": "local", "key_action": "keep", "extra": True},
            {"access_scope": "local", "host": "127.0.0.1"},
        ]
        for body in bad:
            with self.subTest(body=body), self.assertRaises(ValueError):
                np.apply_request(current, body)

    def test_old_shape_maps_only_the_two_canonical_hosts(self):
        current = {"router_host": "127.0.0.1", "router_api_key": "k" * 32}
        kept = np.apply_request(current, {"host": "0.0.0.0"})
        self.assertEqual((kept.access_scope, kept.key_action), ("lan", "keep"))
        cleared = np.apply_request(current, {"host": "127.0.0.1", "api_key": ""})
        self.assertEqual(cleared.router_api_key, "")
        with self.assertRaises(ValueError):
            np.apply_request(current, {"host": "192.168.1.20"})
        with self.assertRaises(ValueError):
            np.apply_request({"router_api_key": ""}, {"host": "0.0.0.0"})

    def test_mutation_requests_do_not_modify_inputs(self):
        cases = [
            ({"router_host": "0.0.0.0", "router_api_key": "secret"},
             {"access_scope": "lan", "key_action": "keep"}),
            ({"router_host": "127.0.0.1", "router_api_key": ""},
             {"access_scope": "lan", "key_action": "replace", "api_key": "r" * 32}),
            ({"router_host": "0.0.0.0", "router_api_key": "s" * 32},
             {"access_scope": "local", "key_action": "clear"}),
        ]
        for current, body in cases:
            with self.subTest(body=body):
                current_before, body_before = copy.deepcopy(current), copy.deepcopy(body)
                np.apply_request(current, body)
                self.assertEqual(current, current_before)
                self.assertEqual(body, body_before)


class ProjectionAndCliTest(unittest.TestCase):
    def test_public_projection_is_allowlisted_and_redacted(self):
        cfg = {
            "theme": "dark", "cvd": True, "auto_load_model": "m",
            "vram_bandwidths": {"vram_bw": 900}, "presets": {"fast": {}},
            "preset_bindings": {"llamacpp": {"m": "fast"}},
            "active_engine": "llamacpp", "router_api_key": "do-not-return",
            "server_bin": "C:/private/server.exe", "future_secret": "hidden",
        }
        out = np.public_config(cfg)
        self.assertEqual(set(out), {
            "theme", "cvd", "auto_load_model", "vram_bandwidths", "presets",
            "preset_bindings", "active_engine", "router_api_key_configured"})
        self.assertTrue(out["router_api_key_configured"])
        self.assertNotIn("do-not-return", repr(out))
        out["vram_bandwidths"]["vram_bw"] = 1
        out["presets"]["fast"]["changed"] = True
        out["preset_bindings"]["llamacpp"]["m"] = "changed"
        self.assertEqual(cfg["vram_bandwidths"]["vram_bw"], 900)
        self.assertEqual(cfg["presets"], {"fast": {}})
        self.assertEqual(cfg["preset_bindings"]["llamacpp"]["m"], "fast")

    def test_preflight_never_rewrites_and_cli_status_is_truthful(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            unsafe = {"router_host": "0.0.0.0", "router_api_key": ""}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(unsafe, f)
            before = Path(path).read_bytes()
            ok, message = np.preflight_config_file(path)
            self.assertFalse(ok)
            self.assertIn("API key", message)
            self.assertEqual(Path(path).read_bytes(), before)
            self.assertNotEqual(np.main(["--preflight", path]), 0)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"router_host": "127.0.0.1", "router_api_key": ""}, f)
            self.assertEqual(np.main(["--preflight", path]), 0)


if __name__ == "__main__":
    unittest.main()
