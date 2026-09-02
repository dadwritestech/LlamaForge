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

    def test_direct_codex_and_pi_use_the_resolved_router_v1(self):
        lan = dict(CFG, router_host="0.0.0.0")
        with mock.patch.object(routes.router_ctl, "lan_ip", return_value="192.168.1.8"):
            for agent in ("codex", "pi"):
                with self.subTest(agent=agent, scope="local"):
                    self.assertEqual(
                        routes._agent_endpoint_for(agent, False, CFG),
                        "http://127.0.0.1:8080/v1",
                    )
                with self.subTest(agent=agent, scope="lan"):
                    self.assertEqual(
                        routes._agent_endpoint_for(agent, False, lan),
                        "http://192.168.1.8:8080/v1",
                    )


class AgentConfigRouteTest(unittest.TestCase):
    def _patch(self, rows=LLAMA_ROWS, cfg=CFG, active="llamacpp"):
        return (
            mock.patch.object(routes, "REGISTRY", FakeRegistry(rows, active)),
            mock.patch.object(routes, "cfg", return_value=cfg),
        )

    def _assert_rejected_by_both(self, body, rows=LLAMA_ROWS, active="llamacpp",
                                 message=None):
        for handler in (routes.post_agent_config, routes.post_agent_apply):
            patches = self._patch(rows=rows, active=active)
            req = Req()
            req.body = body
            with self.subTest(handler=handler.__name__), patches[0], patches[1], \
                 mock.patch.object(routes.agentsetup, "generate") as generate, \
                 mock.patch.object(routes.agentsetup, "apply") as apply, \
                 mock.patch.object(routes.os.path, "expanduser") as expanduser, \
                 self.assertRaises(ApiError) as cm:
                handler(req)
            self.assertEqual(cm.exception.status, 400)
            if message:
                self.assertEqual(cm.exception.message, message)
            generate.assert_not_called()
            apply.assert_not_called()
            expanduser.assert_not_called()

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

    def test_malformed_body_and_agent_are_rejected(self):
        self._assert_rejected_by_both([[]])
        self._assert_rejected_by_both({
            "agent": [], "model": "main", "backend": "llamacpp",
            "small": "", "inject": False,
        })

    def test_rejection_matrix_validates_before_dependencies(self):
        cases = [
            ("empty array", [], LLAMA_ROWS, "llamacpp"),
            ("extra field", {"agent": "pi", "model": "main", "backend": "llamacpp",
                             "small": "", "inject": False, "endpoint": "http://evil"},
             LLAMA_ROWS, "llamacpp"),
            ("bad inject", {"agent": "pi", "model": "main", "backend": "llamacpp",
                            "small": "", "inject": "true"}, LLAMA_ROWS, "llamacpp"),
            ("claude inject", {"agent": "claude-code", "model": "main",
                               "backend": "llamacpp", "small": "", "inject": True},
             LLAMA_ROWS, "llamacpp"),
            ("malformed small", {"agent": "claude-code", "model": "main",
                                 "backend": "llamacpp", "small": [], "inject": False},
             LLAMA_ROWS, "llamacpp"),
            ("contradictory ownership", {"agent": "pi", "model": "main",
                                         "backend": "llamacpp", "small": "", "inject": False},
             [{"id": "main", "backend": "ikllama", "status": "loaded"}], "llamacpp"),
            ("active vllm", {"agent": "pi", "model": "v", "backend": "vllm",
                             "small": "", "inject": False},
             [{"id": "v", "backend": "vllm", "status": "loaded"}], "vllm"),
            ("non-claude small", {"agent": "pi", "model": "main", "backend": "llamacpp",
                                  "small": "small", "inject": False}, LLAMA_ROWS, "llamacpp"),
            ("unknown backend", {"agent": "pi", "model": "main", "backend": "unknown",
                                 "small": "", "inject": False}, LLAMA_ROWS, "llamacpp"),
            ("stale backend", {"agent": "pi", "model": "main", "backend": "ikllama",
                               "small": "", "inject": False}, LLAMA_ROWS, "llamacpp"),
            ("missing model", {"agent": "pi", "model": "missing", "backend": "llamacpp",
                              "small": "", "inject": False}, LLAMA_ROWS, "llamacpp"),
        ]
        for name, body, rows, active in cases:
            with self.subTest(case=name):
                self._assert_rejected_by_both(
                    body, rows, active,
                    "agent configuration must be an object" if name == "empty array" else None)

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
