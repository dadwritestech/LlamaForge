import conftest_paths  # noqa: F401
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import clientsetup
import routes
from routes import ApiError, Req


SECRET = "client-secret-" + "x" * 32
BASE_CFG = {
    "active_engine": "llamacpp",
    "router_host": "127.0.0.1",
    "router_port": 8080,
    "router_api_key": SECRET,
}


class FakeRegistry:
    def __init__(self, rows, active="llamacpp"):
        self.rows = rows
        self.active = active

    def state(self):
        return {"models": [dict(row) for row in self.rows], "global": {}}

    def active_engine(self):
        return self.active


class ClientSetupGeneratorTest(unittest.TestCase):
    def test_exact_existing_modal_snippets_are_generated(self):
        out = clientsetup.generate(
            "http://127.0.0.1:8080", SECRET, "qwen", "llamacpp", "posix")
        self.assertEqual(out["endpoint"], "http://127.0.0.1:8080")
        self.assertTrue(out["auth_required"])
        self.assertIn("/v1/chat/completions", out["curl"])
        self.assertIn("Authorization: Bearer " + SECRET, out["curl"])
        self.assertIn("OPENAI_BASE_URL=http://127.0.0.1:8080/v1", out["environment"])
        self.assertIn("OPENAI_API_KEY=" + SECRET, out["environment"])
        self.assertEqual(json.loads(out["payload"])["model"], "qwen")

    def test_unauthenticated_target_uses_not_required_marker(self):
        out = clientsetup.generate(
            "http://127.0.0.1:8081/", "", "v", "vllm", "posix")
        self.assertFalse(out["auth_required"])
        self.assertNotIn("Authorization", out["curl"])
        self.assertIn("OPENAI_API_KEY=not-required", out["environment"])

    def test_posix_curl_quotes_model_and_retained_legacy_key_as_arguments(self):
        model = "model'; touch should-not-run"
        key = "legacy'\"$key"
        out = clientsetup.generate(
            "http://127.0.0.1:8080", key, model, "llamacpp", "posix")
        self.assertIn(" \\\n  ", out["curl"])
        argv = shlex.split(out["curl"].replace("\\\n", ""))
        self.assertEqual(argv[0:2], ["curl", "http://127.0.0.1:8080/v1/chat/completions"])
        self.assertIn("Authorization: Bearer " + key, argv)
        self.assertEqual(json.loads(argv[-1])["model"], model)

    @unittest.skipUnless(
        os.name == "nt" and (shutil.which("powershell") or shutil.which("pwsh")),
        "PowerShell is required for argument-preservation coverage")
    def test_powershell_executes_adversarial_values_as_literal_arguments(self):
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        model = "model'; Write-Error injected; # \"quoted\""
        key = "legacy'\"$key"
        out = clientsetup.generate(
            "http://127.0.0.1:8080", key, model, "llamacpp", "powershell")
        self.assertIn("`\n", out["curl"])
        with tempfile.TemporaryDirectory() as tmp:
            recorded = os.path.join(tmp, "curl-argv.json")
            script = os.path.join(tmp, "snippet.ps1")
            with open(script, "w", encoding="utf-8") as f:
                f.write(
                    "function curl_stub {\n"
                    "  @($args) | ConvertTo-Json -Compress | "
                    "Set-Content -LiteralPath $env:LF_CAPTURE -NoNewline\n"
                    "}\n"
                    "Set-Alias curl curl_stub -Scope Global -Option AllScope -Force\n"
                    + out["curl"] + "\n")
            env = dict(os.environ, LF_CAPTURE=recorded)
            result = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-File", script],
                capture_output=True, text=True, check=False, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(recorded), result.stderr + result.stdout)
            with open(recorded, encoding="utf-8") as f:
                argv = json.load(f)
        self.assertEqual(argv[0:2], ["http://127.0.0.1:8080/v1/chat/completions", "-H"])
        self.assertIn("Authorization: Bearer " + key, argv)
        self.assertEqual(json.loads(argv[-1])["model"], model)


class ClientConfigRouteTest(unittest.TestCase):
    def _call(self, body, rows, cfg=None, active="llamacpp", vllm_status=()):
        with mock.patch.object(routes, "REGISTRY", FakeRegistry(rows, active)), \
             mock.patch.object(routes, "cfg", return_value=cfg or BASE_CFG), \
             mock.patch.object(routes, "vllm_mgr") as manager:
            manager.return_value.status.return_value = list(vllm_status)
            return routes.post_client_config(Req(body=body))

    def test_llama_target_includes_router_credential_only_on_explicit_post(self):
        status, out = self._call(
            {"model": "qwen", "backend": "llamacpp"},
            [{"id": "qwen", "backend": "llamacpp", "status": "loaded"}],
        )
        self.assertEqual(status, 200)
        self.assertEqual(out["backend"], "llamacpp")
        self.assertIn(SECRET, out["curl"] + out["environment"])
        self.assertTrue(out["model_loaded"])

    def test_stale_llama_hint_snaps_to_active_family_engine(self):
        status, out = self._call(
            {"model": "qwen", "backend": "llamacpp"},
            [{"id": "qwen", "backend": "ikllama", "status": "unloaded"}],
            active="ikllama",
        )
        self.assertEqual(status, 200)
        self.assertEqual(out["backend"], "ikllama")
        self.assertFalse(out["model_loaded"])

    def test_unsafe_legacy_lan_config_is_rejected(self):
        config = dict(BASE_CFG, router_host="0.0.0.0", router_api_key="")
        with self.assertRaises(ApiError) as cm:
            self._call(
                {"model": "qwen", "backend": "llamacpp"},
                [{"id": "qwen", "backend": "llamacpp", "status": "loaded"}],
                cfg=config,
            )
        self.assertEqual(cm.exception.status, 409)

    def test_protected_lan_uses_canonical_lan_ip(self):
        config = dict(BASE_CFG, router_host="0.0.0.0")
        with mock.patch.object(routes.router_ctl, "lan_ip", return_value="192.0.2.10"):
            status, out = self._call(
                {"model": "qwen", "backend": "llamacpp"},
                [{"id": "qwen", "backend": "llamacpp", "status": "loaded"}],
                cfg=config,
            )
        self.assertEqual(status, 200)
        self.assertEqual(out["endpoint"], "http://192.0.2.10:8080")

    def test_protected_lan_without_an_ip_is_unavailable(self):
        config = dict(BASE_CFG, router_host="0.0.0.0")
        with mock.patch.object(routes.router_ctl, "lan_ip", return_value=""):
            with self.assertRaises(ApiError) as cm:
                self._call(
                    {"model": "qwen", "backend": "llamacpp"},
                    [{"id": "qwen", "backend": "llamacpp", "status": "loaded"}],
                    cfg=config,
                )
        self.assertEqual(cm.exception.status, 503)

    def test_vllm_ready_target_uses_manager_endpoint_and_never_router_key(self):
        status, out = self._call(
            {"model": "vmodel", "backend": "vllm"},
            [{"id": "vmodel", "backend": "vllm", "status": "loaded",
              "endpoint": "http://caller-controlled.invalid"}],
            vllm_status=[{"model_id": "vmodel", "state": "ready",
                          "endpoint": "http://127.0.0.1:8081"}],
        )
        self.assertEqual(status, 200)
        rendered = json.dumps(out)
        self.assertEqual(out["endpoint"], "http://127.0.0.1:8081")
        self.assertNotIn(SECRET, rendered)
        self.assertNotIn("caller-controlled", rendered)

    def test_vllm_unloaded_is_rejected_without_llama_fallback(self):
        with self.assertRaises(ApiError) as cm:
            self._call(
                {"model": "vmodel", "backend": "vllm"},
                [{"id": "vmodel", "backend": "vllm", "status": "offline"}],
            )
        self.assertEqual(cm.exception.status, 400)
        self.assertIn("load it first", str(cm.exception))

    def test_unknown_and_ambiguous_ownership_are_rejected(self):
        with self.assertRaises(ApiError) as unknown:
            self._call({"model": "missing"}, [])
        self.assertEqual(unknown.exception.status, 400)
        rows = [
            {"id": "same", "backend": "llamacpp", "status": "loaded"},
            {"id": "same", "backend": "vllm", "status": "loaded"},
        ]
        with self.assertRaises(ApiError) as ambiguous:
            self._call({"model": "same"}, rows)
        self.assertEqual(ambiguous.exception.status, 409)

    def test_arbitrary_endpoint_and_unknown_backend_are_rejected(self):
        rows = [{"id": "qwen", "backend": "llamacpp", "status": "loaded"}]
        for body in (
            {"model": "qwen", "backend": "llamacpp", "endpoint": "http://evil"},
            {"model": "qwen", "backend": "other"},
        ):
            with self.subTest(body=body), self.assertRaises(ApiError) as cm:
                self._call(body, rows)
            self.assertEqual(cm.exception.status, 400)

    def test_generation_failure_never_echoes_the_router_key(self):
        rows = [{"id": "qwen", "backend": "llamacpp", "status": "loaded"}]
        with mock.patch.object(
                clientsetup, "generate",
                side_effect=RuntimeError("generator failed with " + SECRET)), \
             self.assertRaises(ApiError) as cm:
            self._call({"model": "qwen", "backend": "llamacpp"}, rows)
        self.assertEqual(cm.exception.status, 500)
        self.assertNotIn(SECRET, str(cm.exception))

    def test_route_is_post_only(self):
        self.assertIs(routes.POST_ROUTES["/api/client/config"], routes.post_client_config)
        self.assertNotIn("/api/client/config", routes.GET_ROUTES)


if __name__ == "__main__":
    unittest.main()
