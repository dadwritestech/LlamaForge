"""Route handlers that the old if-chain made unreachable from a test.

Each handler is now a plain function of Req -> (status, payload), so these run
with no socket and no live router.
"""
import conftest_paths  # noqa: F401
import json, os, tempfile, unittest
from unittest import mock

import config, routes
from routes import Req, ApiError


class RouterAuthenticationTest(unittest.TestCase):
    def test_internal_router_requests_include_configured_bearer_key(self):
        seen = {}

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return json.dumps({"data": []}).encode()

        def open_request(request, timeout=30):
            seen["authorization"] = request.get_header("Authorization")
            return Response()

        with mock.patch.object(routes, "cfg", return_value={
                "router_port": 8080, "router_api_key": "secret-key"}), \
             mock.patch.object(routes.urllib.request, "urlopen",
                               side_effect=open_request):
            status, _ = routes.router("/models")

        self.assertEqual(status, 200)
        self.assertEqual(seen["authorization"], "Bearer secret-key")


class ConfigAllowlistTest(unittest.TestCase):
    """/api/config used to be `cfg.update(body)` - an unfiltered merge that let
    a request set server_bin, which /api/schema then executes."""

    def setUp(self):
        self.saved = {}
        self.patch = mock.patch.object(
            config, "update", side_effect=lambda ch: (self.saved.update(ch),
                                                      dict(self.saved))[1])
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_accepts_the_keys_the_ui_sets(self):
        status, out = routes.post_config(Req(body={
            "theme": "dark", "cvd": True, "ui_mode": "advanced",
            "onboarded": True, "auto_load_model": "qwen", "wsl_distro": "Ubuntu"}))
        self.assertEqual(status, 200)
        self.assertEqual(self.saved["theme"], "dark")
        self.assertEqual(self.saved["ui_mode"], "advanced")
        self.assertNotIn("rejected", out)

    def test_refuses_executable_path_keys(self):
        """The RCE path: set server_bin, then GET /api/schema runs it."""
        for key in ("server_bin", "llama_src", "build_dir", "models_ini",
                    "wiki_dir", "docs_dir"):
            with self.assertRaises(ApiError, msg=key) as cm:
                routes.post_config(Req(body={key: "/tmp/evil"}))
            self.assertEqual(cm.exception.status, 400)
            self.assertEqual(self.saved, {}, f"{key} was written")

    def test_refuses_keys_owned_by_other_routes(self):
        for key in ("router_api_key", "router_host", "presets", "cmake_flags"):
            with self.assertRaises(ApiError, msg=key):
                routes.post_config(Req(body={key: "x"}))
        self.assertEqual(self.saved, {})

    def test_rejects_ill_typed_values(self):
        for body in ({"ui_mode": "root"}, {"theme": "neon"}, {"cvd": "yes"},
                     {"vllm_port": 99999}, {"vllm_port": "8081"},
                     {"model_dirs": "not-a-list"}, {"onboarded": 1}):
            with self.assertRaises(ApiError, msg=str(body)):
                routes.post_config(Req(body=body))
        self.assertEqual(self.saved, {})

    def test_mixed_body_applies_good_keys_and_names_the_bad(self):
        status, out = routes.post_config(
            Req(body={"theme": "light", "server_bin": "/tmp/evil"}))
        self.assertEqual(status, 200)
        self.assertEqual(self.saved, {"theme": "light"})
        self.assertEqual(out["rejected"], ["server_bin"])

    def test_valid_ports_and_dirs_pass(self):
        routes.post_config(Req(body={"vllm_port": 8081,
                                     "model_dirs": ["/models", "/mnt/d"]}))
        self.assertEqual(self.saved["vllm_port"], 8081)
        self.assertEqual(self.saved["model_dirs"], ["/models", "/mnt/d"])


class SaveAndPresetTest(unittest.TestCase):
    """/api/save and /api/presets/apply share the write-then-reload sequence
    that was duplicated inline in two branches of the old if-chain."""

    def setUp(self):
        self.calls = []
        self.set_keys = mock.patch.object(config, "set_keys",
                                          side_effect=lambda *a, **k: self.calls.append(("set", a)))
        self.set_keys.start()
        self.addCleanup(self.set_keys.stop)

    def _router(self, loaded_ids=()):
        def fake(path, method="GET", body=None, timeout=30):
            self.calls.append(("router", path, method, body))
            if path == "/models":
                return 200, {"data": [{"id": i, "status": {"value": "loaded"}}
                                      for i in loaded_ids]}
            return 200, {}
        return fake

    def test_save_blank_value_clears_the_key(self):
        with mock.patch.object(routes, "router", self._router()):
            status, out = routes.post_save(Req(body={
                "model": "m", "settings": {"ctx-size": "4096", "temp": "  "}}))
        self.assertEqual(status, 200)
        written = [c for c in self.calls if c[0] == "set"][0][1][1]
        self.assertEqual(written, {"ctx-size": "4096", "temp": None})

    def test_save_unloads_a_running_model_before_reload(self):
        with mock.patch.object(routes, "router", self._router(loaded_ids=["m"])):
            status, out = routes.post_save(
                Req(body={"model": "m", "settings": {"ctx-size": "8192"}}))
        self.assertTrue(out["was_running"])
        paths = [c[1] for c in self.calls if c[0] == "router"]
        self.assertIn("/models/unload", paths)
        self.assertIn("/models?reload=1", paths)

    def test_save_does_not_unload_an_idle_model(self):
        with mock.patch.object(routes, "router", self._router(loaded_ids=[])):
            status, out = routes.post_save(
                Req(body={"model": "m", "settings": {"ctx-size": "8192"}}))
        self.assertFalse(out["was_running"])
        paths = [c[1] for c in self.calls if c[0] == "router"]
        self.assertNotIn("/models/unload", paths)

    def test_apply_unknown_preset_is_a_400(self):
        with mock.patch.object(config, "get_presets", return_value={}):
            with self.assertRaises(ApiError) as cm:
                routes.post_presets_apply(Req(body={"model": "m", "name": "nope"}))
        self.assertEqual(cm.exception.status, 400)

    def test_apply_preset_uses_the_same_reload_sequence_as_save(self):
        presets = {"coding": {"temp": "0.2", "ctx-size": ""}}
        with mock.patch.object(config, "get_presets", return_value=presets), \
             mock.patch.object(routes, "router", self._router(loaded_ids=["m"])):
            status, out = routes.post_presets_apply(
                Req(body={"model": "m", "name": "coding"}))
        self.assertEqual(status, 200)
        self.assertTrue(out["was_running"])
        written = [c for c in self.calls if c[0] == "set"][0][1][1]
        self.assertEqual(written, {"temp": "0.2", "ctx-size": None})


class ScanPruneTest(unittest.TestCase):
    """/api/scan/prune deletes models.ini sections - it must never remove one
    whose file is actually present."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old_config = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")
        self.removed = []
        mock.patch.object(config, "remove_section",
                          side_effect=lambda s: (self.removed.append(s), True)[1]).start()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        mock.patch.stopall()
        config.CONFIG = self.old_config

    def _no_router(self, *a, **k):
        return 599, {}

    def test_keeps_a_section_whose_file_reappeared(self):
        sections = {"gone": {"model": "/nope/a.gguf"},
                    "back": {"model": "/real/b.gguf"}}
        with mock.patch.object(config, "read_sections", return_value=sections), \
             mock.patch.object(routes, "router", self._no_router), \
             mock.patch.object(os.path, "exists",
                               side_effect=lambda p: p == "/real/b.gguf"):
            status, out = routes.post_scan_prune(Req(body={"ids": ["gone", "back"]}))
        self.assertEqual(out["removed"], ["gone"])
        self.assertEqual(self.removed, ["gone"])

    def test_ignores_unknown_ids(self):
        with mock.patch.object(config, "read_sections", return_value={}), \
             mock.patch.object(routes, "router", self._no_router):
            status, out = routes.post_scan_prune(Req(body={"ids": ["ghost"]}))
        self.assertEqual(out["removed"], [])

    def test_pruning_a_removed_section_drops_its_preset_binding(self):
        config.save_preset("coding", {"temp": "0.2"})
        config.bind_preset("gone", "coding")
        with mock.patch.object(config, "read_sections",
                               return_value={"gone": {"model": "/nope/a.gguf"}}), \
             mock.patch.object(routes, "router", self._no_router):
            status, out = routes.post_scan_prune(Req(body={"ids": ["gone"]}))

        self.assertEqual(out["removed"], ["gone"])
        self.assertEqual(config.get_bindings(), {})


class ScanRootsTest(unittest.TestCase):
    def test_explicit_roots_override_saved_directories(self):
        with mock.patch.object(routes, "cfg", return_value={"model_dirs": ["/saved"]}), \
             mock.patch.object(routes.scanner, "scan", return_value=[]) as scan:
            routes.post_scan(Req(body={"roots": ["/chosen"]}))
        scan.assert_called_once_with(["/chosen"])

    def test_saved_directories_are_used_when_request_has_no_roots(self):
        with mock.patch.object(routes, "cfg", return_value={"model_dirs": ["/saved"]}), \
             mock.patch.object(routes.scanner, "scan", return_value=[]) as scan:
            routes.post_scan(Req(body={}))
        scan.assert_called_once_with(["/saved"])

    def test_explicit_empty_roots_requests_the_platform_defaults(self):
        with mock.patch.object(routes, "cfg", return_value={"model_dirs": ["/saved"]}), \
             mock.patch.object(routes.scanner, "scan", return_value=[]) as scan:
            routes.post_scan(Req(body={"roots": []}))
        scan.assert_called_once_with(None)


class ModelUnregisterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ini = os.path.join(self.tmp, "models.ini")
        self.model_file = os.path.join(self.tmp, "bogus.gguf")
        with open(self.model_file, "wb") as f:
            f.write(b"GGUF")
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write(f"[*]\nctx-size = 8192\n\n[bogus]\nmodel = {self.model_file}\n")

    def _backend(self, name="llamacpp"):
        return type("Backend", (), {"name": name})()

    def test_unregister_removes_registry_entry_without_touching_model_file(self):
        calls = []
        with mock.patch.object(routes, "_backend_for", return_value=("bogus", self._backend())), \
             mock.patch.object(config, "ini_path", return_value=self.ini), \
             mock.patch.object(routes, "router", side_effect=lambda *a, **k: (calls.append(a), (200, {"data": []}))[1]), \
             mock.patch.object(routes, "_reconcile_preset_binding"), \
             mock.patch.object(config, "prune_binding"):
            status, out = routes.post_model_unregister(Req(body={"model": "bogus"}))
        self.assertEqual(status, 200)
        self.assertTrue(out["ok"])
        self.assertNotIn("bogus", config.read_sections(self.ini))
        self.assertIn("*", config.read_sections(self.ini))
        self.assertTrue(os.path.isfile(self.model_file))

    def test_unregister_unloads_loaded_model_before_registry_reload(self):
        calls = []
        def router(path, method="GET", body=None, timeout=30):
            calls.append((path, method, body))
            if path == "/models":
                return 200, {"data": [{"id": "bogus", "status": {"value": "loaded"}}]}
            return 200, {}
        with mock.patch.object(routes, "_backend_for", return_value=("bogus", self._backend())), \
             mock.patch.object(config, "ini_path", return_value=self.ini), \
             mock.patch.object(routes, "router", side_effect=router), \
             mock.patch.object(routes, "_reconcile_preset_binding"), \
             mock.patch.object(config, "prune_binding"):
            routes.post_model_unregister(Req(body={"model": "bogus"}))
        paths = [c[0] for c in calls]
        self.assertLess(paths.index("/models/unload"), paths.index("/models?reload=1"))

    def test_unregister_keeps_entry_when_loaded_model_cannot_unload(self):
        def router(path, method="GET", body=None, timeout=30):
            if path == "/models":
                return 200, {"data": [{"id": "bogus", "status": {"value": "loaded"}}]}
            if path == "/models/unload":
                return 500, {"error": "busy"}
            return 200, {}
        with mock.patch.object(routes, "_backend_for", return_value=("bogus", self._backend())), \
             mock.patch.object(config, "ini_path", return_value=self.ini), \
             mock.patch.object(routes, "router", side_effect=router):
            with self.assertRaises(ApiError) as cm:
                routes.post_model_unregister(Req(body={"model": "bogus"}))
        self.assertEqual(cm.exception.status, 409)
        self.assertIn("bogus", config.read_sections(self.ini))

    def test_unregister_rejects_global_section(self):
        with mock.patch.object(routes, "_backend_for", return_value=("*", self._backend())):
            with self.assertRaises(ApiError) as cm:
                routes.post_model_unregister(Req(body={"model": "*"}))
        self.assertEqual(cm.exception.status, 400)

    def test_unregister_rejects_vllm_models(self):
        with mock.patch.object(routes, "_backend_for", return_value=("m", self._backend("vllm"))):
            with self.assertRaises(ApiError) as cm:
                routes.post_model_unregister(Req(body={"model": "m"}))
        self.assertEqual(cm.exception.status, 400)


class HubAddTest(unittest.TestCase):
    def test_missing_file_is_a_400(self):
        with self.assertRaises(ApiError) as cm:
            routes.post_hub_add(Req(body={"path": "/definitely/not/here.gguf"}))
        self.assertEqual(cm.exception.status, 400)

    def test_registers_every_gguf_in_the_folder(self):
        tmp = tempfile.mkdtemp()
        for n in ("a.gguf", "b.gguf", "notes.txt"):
            open(os.path.join(tmp, n), "w").close()
        seen = {}
        with mock.patch.object(routes.scanner, "build_entries",
                               side_effect=lambda paths: (seen.setdefault("paths", paths),
                                                          [{"id": "a", "model": paths[0]}])[1]), \
             mock.patch.object(config, "set_keys"), \
             mock.patch.object(config, "apply_ctx_defaults"), \
             mock.patch.object(routes, "router", lambda *a, **k: (200, {})):
            status, out = routes.post_hub_add(
                Req(body={"path": os.path.join(tmp, "a.gguf")}))
        self.assertEqual(status, 200)
        self.assertEqual(sorted(os.path.basename(p) for p in seen["paths"]),
                         ["a.gguf", "b.gguf"])       # .txt not registered


class PublicConfigProjectionTest(unittest.TestCase):
    def setUp(self):
        self.secret = "projection-secret-" + "p" * 32
        self.stored = {
            "theme": "dark",
            "cvd": True,
            "auto_load_model": "llama-fixture",
            "vram_bandwidths": {"vram_bw": 900.0},
            "presets": {"fast": {"temp": "0.2"}},
            "preset_bindings": {"llamacpp": {"llama-fixture": "fast"}},
            "active_engine": "llamacpp",
            "router_api_key": self.secret,
            "router_host": "0.0.0.0",
            "router_port": 8080,
            "panel_port": 8090,
            "server_bin": "private-server",
            "future_secret": "private-future-value",
        }

    def test_projection_is_exact_allowlist_and_never_returns_key(self):
        out = routes._public_config(self.stored)
        self.assertEqual(set(out), {
            "theme", "cvd", "auto_load_model", "vram_bandwidths",
            "presets", "preset_bindings", "active_engine",
            "router_api_key_configured",
        })
        self.assertTrue(out["router_api_key_configured"])
        self.assertNotIn(self.secret, repr(out))
        self.assertNotIn("router_api_key", out)

    def test_state_uses_the_same_projection(self):
        registry = mock.Mock()
        registry.state.return_value = {"models": [], "global": {}}
        registry.enabled.return_value = []
        with mock.patch.object(routes, "cfg", return_value=self.stored), \
             mock.patch.object(routes, "REGISTRY", registry), \
             mock.patch.object(routes, "_gpu_telemetry", return_value=[]), \
             mock.patch.object(routes.os.path, "exists", return_value=True):
            status, out = routes.get_state(Req())
        self.assertEqual(status, 200)
        self.assertEqual(out["config"], routes._public_config(self.stored))
        self.assertNotIn(self.secret, repr(out))

    def test_successful_config_post_uses_the_same_projection(self):
        updated = dict(self.stored, theme="light")
        with mock.patch.object(config, "update", return_value=updated):
            status, out = routes.post_config(Req(body={"theme": "light"}))
        self.assertEqual(status, 200)
        self.assertEqual(out["config"], routes._public_config(updated))
        self.assertNotIn(self.secret, repr(out))


class ExplicitSecretRouteTableTest(unittest.TestCase):
    def test_agent_preview_get_is_removed_and_posts_are_present(self):
        self.assertNotIn("/api/agent/config", routes.GET_ROUTES)
        self.assertIs(routes.POST_ROUTES["/api/agent/config"],
                      routes.post_agent_config)
        self.assertIs(routes.POST_ROUTES["/api/client/config"],
                      routes.post_client_config)


class RouteTableTest(unittest.TestCase):
    def test_every_route_maps_to_a_callable(self):
        for table in (routes.GET_ROUTES, routes.POST_ROUTES):
            for path, handler in table.items():
                self.assertTrue(callable(handler), path)
                self.assertTrue(path.startswith("/"), path)

    def test_no_path_is_registered_twice_in_one_table(self):
        for table in (routes.GET_ROUTES, routes.POST_ROUTES):
            self.assertEqual(len(table), len(set(table)))

    def test_vllm_routes_are_all_under_the_gated_prefix(self):
        """server._vllm_gate short-circuits on /api/vllm/ - a vLLM route named
        anything else would run on Linux and fail confusingly."""
        for table in (routes.GET_ROUTES, routes.POST_ROUTES):
            for path, handler in table.items():
                if handler.__name__.startswith(("get_vllm", "post_vllm")):
                    self.assertTrue(path.startswith("/api/vllm/"), path)


if __name__ == "__main__":
    unittest.main()
