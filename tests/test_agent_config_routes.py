import conftest_paths  # noqa: F401
import json
import unittest
from unittest import mock

import routes
from routes import ApiError, Req


SECRET = "agent-secret-" + "z" * 32
CFG = {
    "active_engine": "llamacpp",
    "router_host": "127.0.0.1",
    "router_port": 8080,
    "panel_port": 8090,
    "router_api_key": SECRET,
}
LLAMA_ROWS = [
    {"id": "main", "backend": "llamacpp", "status": "loaded"},
    {"id": "small", "backend": "llamacpp", "status": "unloaded"},
]


class FakeRegistry:
    def __init__(self, rows=LLAMA_ROWS, active="llamacpp"):
        self.rows = rows
        self.active = active

    def state(self):
        return {"models": [dict(row) for row in self.rows], "global": {}}

    def active_engine(self):
        return self.active


class AgentEndpointMatrixTest(unittest.TestCase):
    def test_claude_is_always_panel_loopback(self):
        self.assertEqual(
            routes._agent_endpoint_for("claude-code", False, CFG),
            "http://127.0.0.1:8090",
        )

    def test_codex_and_pi_injected_are_panel_loopback_v1(self):
        for agent in ("codex", "pi"):
            with self.subTest(agent=agent):
                self.assertEqual(
                    routes._agent_endpoint_for(agent, True, CFG),
                    "http://127.0.0.1:8090/v1",
                )

    def test_direct_agent_uses_lan_router_not_lan_panel(self):
        lan = dict(CFG, router_host="0.0.0.0")
        with mock.patch.object(routes.router_ctl, "lan_ip", return_value="192.168.1.8"):
            self.assertEqual(
                routes._agent_endpoint_for("pi", False, lan),
                "http://192.168.1.8:8080/v1",
            )


class AgentConfigRouteTest(unittest.TestCase):
    def _patch(self, rows=LLAMA_ROWS, cfg=CFG, active="llamacpp"):
        return (
            mock.patch.object(routes, "REGISTRY", FakeRegistry(rows, active)),
            mock.patch.object(routes, "cfg", return_value=cfg),
        )

    def test_preview_and_apply_use_the_same_resolved_endpoint(self):
        body = {"agent": "pi", "model": "main", "backend": "llamacpp",
                "small": "", "inject": True}
        patches = self._patch()
        with patches[0], patches[1], \
             mock.patch.object(routes.agentsetup, "generate",
                               return_value={"content": SECRET, "endpoint": "x"}) as generate:
            status, preview = routes.post_agent_config(Req(body=body))
        self.assertEqual(status, 200)
        self.assertEqual(generate.call_args.args[1], "http://127.0.0.1:8090/v1")
        self.assertIn(SECRET, json.dumps(preview))

        patches = self._patch()
        with patches[0], patches[1], \
             mock.patch.object(routes.agentsetup, "apply",
                               return_value={"ok": True, "path": "p",
                                             "backup": None, "action": "created"}) as apply:
            status, written = routes.post_agent_apply(Req(body=body))
        self.assertEqual(status, 200)
        self.assertEqual(apply.call_args.args[2], "http://127.0.0.1:8090/v1")
        self.assertNotIn(SECRET, json.dumps(written))

    def test_claude_small_model_is_validated_on_same_backend(self):
        body = {"agent": "claude-code", "model": "main", "backend": "llamacpp",
                "small": "small", "inject": False}
        patches = self._patch()
        with patches[0], patches[1], \
             mock.patch.object(routes.agentsetup, "generate",
                               return_value={"content": "ok"}) as generate:
            routes.post_agent_config(Req(body=body))
        self.assertEqual(generate.call_args.args[4], "small")

    def test_vllm_unknown_and_stale_backends_are_rejected(self):
        vrows = [{"id": "v", "backend": "vllm", "status": "loaded"}]
        bodies = [
            {"agent": "pi", "model": "v", "backend": "vllm",
             "small": "", "inject": False},
            {"agent": "pi", "model": "main", "backend": "unknown",
             "small": "", "inject": False},
            {"agent": "pi", "model": "main", "backend": "ikllama",
             "small": "", "inject": False},
        ]
        for body in bodies:
            rows = vrows if body["backend"] == "vllm" else LLAMA_ROWS
            patches = self._patch(rows=rows)
            with self.subTest(body=body), patches[0], patches[1], \
                 self.assertRaises(ApiError) as cm:
                routes.post_agent_config(Req(body=body))
            self.assertEqual(cm.exception.status, 400)

    def test_bad_inject_or_extra_fields_fail_before_generate_or_write(self):
        bad = [
            {"agent": "pi", "model": "main", "backend": "llamacpp",
             "small": "", "inject": "true"},
            {"agent": "claude-code", "model": "main", "backend": "llamacpp",
             "small": "", "inject": True},
            {"agent": "pi", "model": "main", "backend": "llamacpp",
             "small": "", "inject": False, "endpoint": "http://evil"},
            {"agent": "claude-code", "model": "main", "backend": "llamacpp",
             "small": [], "inject": False},
        ]
        for body in bad:
            patches = self._patch()
            with self.subTest(body=body), patches[0], patches[1], \
                 mock.patch.object(routes.agentsetup, "generate") as generate, \
                 mock.patch.object(routes.agentsetup, "apply") as apply, \
                 self.assertRaises(ApiError):
                routes.post_agent_config(Req(body=body))
            generate.assert_not_called()
            apply.assert_not_called()

    def test_apply_validates_before_touching_home(self):
        body = {"agent": "pi", "model": "missing", "backend": "llamacpp",
                "small": "", "inject": False}
        patches = self._patch()
        with patches[0], patches[1], \
             mock.patch.object(routes.os.path, "expanduser") as expanduser, \
             mock.patch.object(routes.agentsetup, "apply") as apply, \
             self.assertRaises(ApiError):
            routes.post_agent_apply(Req(body=body))
        expanduser.assert_not_called()
        apply.assert_not_called()

    def test_temporary_apply_adapter_preserves_the_current_frontend_shape(self):
        old_body = {"agent": "pi", "model": "main", "small": ""}
        patches = self._patch()
        with patches[0], patches[1], \
             mock.patch.object(
                 routes.agentsetup, "apply",
                 return_value={"ok": True, "path": "p", "backup": None,
                               "action": "created"}) as apply:
            status, _ = routes.post_agent_apply(Req(body=old_body))
        self.assertEqual(status, 200)
        self.assertEqual(apply.call_args.args[2], "http://127.0.0.1:8080/v1")

    def test_preview_and_apply_failures_never_echo_the_router_key(self):
        body = {"agent": "pi", "model": "main", "backend": "llamacpp",
                "small": "", "inject": False}
        for handler, dependency in (
            (routes.post_agent_config, "generate"),
            (routes.post_agent_apply, "apply"),
        ):
            patches = self._patch()
            with self.subTest(handler=handler.__name__), patches[0], patches[1], \
                 mock.patch.object(
                     routes.agentsetup, dependency,
                     side_effect=RuntimeError("operation failed with " + SECRET)), \
                 self.assertRaises(ApiError) as cm:
                handler(Req(body=body))
            self.assertEqual(cm.exception.status, 500)
            self.assertNotIn(SECRET, str(cm.exception))

    def test_post_preview_is_registered(self):
        self.assertIs(routes.POST_ROUTES["/api/agent/config"], routes.post_agent_config)


if __name__ == "__main__":
    unittest.main()
