"""The MCP server: both protocol eras, the tools over a fake panel, model choice,
cancel and progress. Nothing here talks to a real panel, router or pi."""
import conftest_paths  # noqa: F401
import copy
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest

import mcp_server as M

V_MODERN = "2026-07-28"
META = {"io.modelcontextprotocol/protocolVersion": V_MODERN,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"}}

STATE = {
    "version": "0.15.0", "active_engine": "llamacpp", "config_error": None,
    "models": [
        {"id": "big-27b", "status": "loaded", "failed": False, "backend": "llamacpp",
         "eff_ctx": "32768", "file_gib": 16.1, "endpoint": "http://127.0.0.1:8080",
         "modalities": ["text"], "settings": {"ctx-size": "32768"}},
        {"id": "small-9b", "status": "loaded", "failed": False, "backend": "llamacpp",
         "eff_ctx": "16384", "file_gib": 5.2, "endpoint": "http://127.0.0.1:8080",
         "modalities": ["text", "image"], "build": "b1234", "settings": {}},
        {"id": "cold-4b", "status": "unloaded", "failed": False, "backend": "llamacpp",
         "eff_ctx": "?", "file_gib": 2.5, "modalities": ["text"], "settings": {}},
    ],
    "gpus": [{"index": 0, "name": "RTX 5080", "used": 12000, "total": 16303, "util": 5, "temp": 40}],
    "slots": {"enabled": True, "main": "big-27b", "devices": {}, "settings": {}},
}


def state_with(**over):
    s = copy.deepcopy(STATE)
    s.update(over)
    return s


class FakePanel:
    def __init__(self, routes=None):
        self.routes = {("GET", "/api/state"): STATE}
        self.routes.update(routes or {})
        self.calls = []

    def _answer(self, key, arg):
        self.calls.append((key[0], key[1], arg))
        r = self.routes[key]
        if isinstance(r, Exception):
            raise r
        return r(arg) if callable(r) else copy.deepcopy(r)

    def get(self, path, params=None, timeout=None):
        return self._answer(("GET", path), params)

    def post(self, path, body=None, timeout=None):
        return self._answer(("POST", path), body)


CFG = {"panel_port": 8090, "router_api_key": "", "router_local_key": "loc-key", "pi_bin": "/x/pi.js"}


def make(routes=None, **kw):
    panel = FakePanel(routes)
    kw.setdefault("cfg", lambda: dict(CFG))
    sent = []
    srv = M.Server(panel=panel, send=sent.append, threaded=False, poll_s=0, **kw)
    return srv, panel, sent


def req(method, params=None, id_=1, meta=None):
    p = dict(params or {})
    if meta:
        p["_meta"] = dict(meta)
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if p or meta:
        msg["params"] = p
    return msg


def call(srv, name, args=None, meta=None, id_=1):
    return srv.handle(req("tools/call", {"name": name, "arguments": args or {}}, id_, meta))


def text_of(res):
    return "\n".join(c["text"] for c in res["result"]["content"] if c["type"] == "text")


class Legacy(unittest.TestCase):
    def test_initialize_echoes_a_supported_version(self):
        srv, _, _ = make()
        r = srv.handle(req("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                          "clientInfo": {"name": "c", "version": "1"}}))
        res = r["result"]
        self.assertEqual(res["protocolVersion"], "2025-06-18")
        self.assertEqual(res["serverInfo"]["name"], "llamaforge")
        self.assertIn("tools", res["capabilities"])
        self.assertIn("pi", res["instructions"])
        self.assertNotIn("resultType", res)

    def test_initialize_with_an_unknown_version_offers_the_newest_legacy(self):
        srv, _, _ = make()
        r = srv.handle(req("initialize", {"protocolVersion": "1999-01-01"}))
        self.assertEqual(r["result"]["protocolVersion"], "2025-11-25")

    def test_ping_and_notifications(self):
        srv, _, _ = make()
        self.assertEqual(srv.handle(req("ping"))["result"], {})
        self.assertIsNone(srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        # a response from the client (we never ask) is ignored
        self.assertIsNone(srv.handle({"jsonrpc": "2.0", "id": 9, "result": {}}))

    def test_tools_list_in_a_fixed_order(self):
        srv, _, _ = make()
        tools = srv.handle(req("tools/list"))["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], [
            "status", "list_models", "load_model", "unload_model", "fit_check", "stats",
            "diagnose", "search_models", "list_files", "download_model",
            "download_progress", "ask", "pi_run", "forge_start", "forge_status",
            "forge_apply", "forge_undo"])
        for t in tools:
            self.assertEqual(t["inputSchema"]["type"], "object", t["name"])
            self.assertFalse(t["inputSchema"].get("additionalProperties", False), t["name"])
            self.assertTrue(t["description"], t["name"])
        pi = next(t for t in tools if t["name"] == "pi_run")
        self.assertIn("Mario Zechner", pi["description"])
        self.assertIn("github.com/earendil-works/pi", pi["description"])

    def test_json_rpc_errors(self):
        srv, _, _ = make()
        self.assertEqual(srv.handle(req("nope"))["error"]["code"], -32601)
        r = srv.handle([req("ping")])
        self.assertEqual((r["error"]["code"], r["id"]), (-32600, None))
        self.assertEqual(srv.handle({"jsonrpc": "1.0", "id": 1, "method": "ping"})["error"]["code"], -32600)
        self.assertEqual(srv.handle({"jsonrpc": "2.0", "id": True, "method": "ping"})["error"]["code"], -32600)
        self.assertEqual(srv.handle({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": [1]})
                         ["error"]["code"], -32602)
        r = srv.handle_line("{not json")
        self.assertEqual((r["error"]["code"], r["id"]), (-32700, None))
        self.assertIsNone(srv.handle_line("   "))


class Modern(unittest.TestCase):
    def test_results_are_marked_complete_and_signed(self):
        srv, _, _ = make()
        res = srv.handle(req("tools/list", meta=META))["result"]
        self.assertEqual(res["resultType"], "complete")
        self.assertEqual(res["_meta"]["io.modelcontextprotocol/serverInfo"]["name"], "llamaforge")
        self.assertGreater(res["ttlMs"], 0)
        self.assertIn(res["cacheScope"], ("public", "private"))

    def test_unsupported_version(self):
        srv, _, _ = make()
        meta = dict(META, **{"io.modelcontextprotocol/protocolVersion": "2099-01-01"})
        err = srv.handle(req("tools/list", meta=meta))["error"]
        self.assertEqual(err["code"], -32022)
        self.assertIn(V_MODERN, err["data"]["supported"])
        self.assertEqual(err["data"]["requested"], "2099-01-01")

    def test_client_capabilities_are_required(self):
        srv, _, _ = make()
        meta = {"io.modelcontextprotocol/protocolVersion": V_MODERN}
        self.assertEqual(srv.handle(req("tools/list", meta=meta))["error"]["code"], -32602)

    def test_discover(self):
        srv, _, _ = make()
        res = srv.handle(req("server/discover", meta=META))["result"]
        self.assertIn(V_MODERN, res["supportedVersions"])
        self.assertIn("2025-11-25", res["supportedVersions"])
        self.assertIn("tools", res["capabilities"])
        self.assertEqual(res["resultType"], "complete")
        self.assertTrue(res["instructions"])
        self.assertEqual(srv.handle(req("server/discover"))["error"]["code"], -32602)

    def test_ping_is_gone(self):
        srv, _, _ = make()
        self.assertEqual(srv.handle(req("ping", meta=META))["error"]["code"], -32601)

    def test_tool_results_are_complete_too(self):
        srv, _, _ = make()
        self.assertEqual(call(srv, "status", meta=META)["result"]["resultType"], "complete")


class Arguments(unittest.TestCase):
    def test_unknown_tool_is_a_protocol_error(self):
        srv, _, _ = make()
        self.assertEqual(call(srv, "rm_rf")["error"]["code"], -32602)

    def test_bad_arguments_are_tool_errors_the_model_can_fix(self):
        srv, _, _ = make()
        for args, word in (({}, "model"), ({"model": 3}, "model"),
                           ({"model": "cold-4b", "role": "boss"}, "role"),
                           ({"model": "cold-4b", "colour": "red"}, "colour"),
                           ({"model": "cold-4b", "wait_s": 0}, "wait_s"),
                           ({"model": "cold-4b", "evict": "yes"}, "evict")):
            r = call(srv, "load_model", args)
            self.assertTrue(r["result"]["isError"], args)
            self.assertIn(word, text_of(r), args)

    def test_panel_down(self):
        srv, _, _ = make({("GET", "/api/state"): M.PanelError("the LlamaForge panel isn't running")})
        r = call(srv, "status")
        self.assertTrue(r["result"]["isError"])
        self.assertIn("panel isn't running", text_of(r))


class Tools(unittest.TestCase):
    def test_status(self):
        srv, _, _ = make()
        r = call(srv, "status")
        self.assertFalse(r["result"]["isError"])
        s = r["result"]["structuredContent"]
        self.assertEqual(s["version"], "0.15.0")
        self.assertTrue(s["multi_model"])
        self.assertEqual([(m["id"], m["role"]) for m in s["loaded"]],
                         [("big-27b", "main"), ("small-9b", "worker")])
        self.assertEqual(s["loaded"][0]["endpoint"], "http://127.0.0.1:8080/v1")
        self.assertEqual(s["gpus"][0]["name"], "RTX 5080")
        # the same JSON as text, for clients that only read text
        self.assertEqual(json.loads(text_of(r)), s)

    def test_list_models(self):
        srv, _, _ = make()
        ms = call(srv, "list_models")["result"]["structuredContent"]["models"]
        self.assertEqual([m["id"] for m in ms], ["big-27b", "small-9b", "cold-4b"])
        cold = ms[2]
        self.assertEqual((cold["status"], cold["size_gib"]), ("unloaded", 2.5))
        self.assertNotIn("endpoint", cold)
        self.assertEqual(ms[1]["build"], "b1234")

    def test_load_waits_until_loaded(self):
        seq = [state_with(), state_with(), state_with()]
        seq[1]["models"][2]["status"] = "loading"
        seq[2]["models"][2].update(status="loaded", endpoint="http://127.0.0.1:8080")
        states = iter(seq)
        srv, panel, sent = make({
            ("POST", "/api/models/load"): {"ok": True, "error": "", "backend": "llamacpp"},
            ("GET", "/api/state"): lambda _: next(states)})
        r = call(srv, "load_model", {"model": "cold-4b", "role": "worker"})
        self.assertFalse(r["result"]["isError"], text_of(r))
        self.assertEqual(r["result"]["structuredContent"]["status"], "loaded")
        self.assertEqual(r["result"]["structuredContent"]["endpoint"], "http://127.0.0.1:8080/v1")
        post = [c for c in panel.calls if c[0] == "POST"][0]
        self.assertEqual(post[2], {"model": "cold-4b", "backend": "llamacpp",
                                   "role": "worker", "evict": False})

    def test_load_of_a_loaded_model_is_a_no_op(self):
        srv, panel, _ = make()
        r = call(srv, "load_model", {"model": "small-9b"})
        self.assertFalse(r["result"]["isError"])
        self.assertFalse([c for c in panel.calls if c[0] == "POST"])

    def test_load_failure_brings_the_diagnosis(self):
        failed = state_with()
        failed["models"][2].update(status="unloaded", failed=True)
        srv, _, _ = make({
            ("POST", "/api/models/load"): {"ok": True, "error": "", "backend": "llamacpp"},
            ("GET", "/api/state"): lambda _: copy.deepcopy(failed),
            ("GET", "/api/model/diag"): {"diag": {"error": "CUDA out of memory",
                                                  "suggestion": "lower ctx-size"}}})
        # the first poll sees the stale failed flag from before? no: rows start clean
        r = call(srv, "load_model", {"model": "cold-4b"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("CUDA out of memory", text_of(r))
        self.assertIn("lower ctx-size", text_of(r))

    def test_load_refused_by_the_planner(self):
        srv, _, _ = make({("POST", "/api/models/load"): {
            "ok": False, "error": "won't fit beside big-27b", "reason": "fit",
            "backend": "llamacpp"}})
        r = call(srv, "load_model", {"model": "cold-4b"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("won't fit", text_of(r))

    def test_load_of_an_unknown_model(self):
        srv, _, _ = make()
        r = call(srv, "load_model", {"model": "ghost"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("list_models", text_of(r))

    def test_load_gives_up_waiting(self):
        loading = state_with()
        loading["models"][2]["status"] = "loading"
        srv, _, _ = make({("POST", "/api/models/load"): {"ok": True, "backend": "llamacpp"},
                          ("GET", "/api/state"): lambda _: copy.deepcopy(loading)})
        srv.poll_s = 0.05
        r = call(srv, "load_model", {"model": "cold-4b", "wait_s": 1})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("still loading", text_of(r))

    def test_unload(self):
        srv, panel, _ = make({("POST", "/api/models/unload"): {"ok": True, "error": ""}})
        r = call(srv, "unload_model", {"model": "small-9b"})
        self.assertFalse(r["result"]["isError"])
        self.assertEqual(panel.calls[-1], ("POST", "/api/models/unload",
                                           {"model": "small-9b", "backend": "llamacpp"}))

    def test_fit_check(self):
        plan = {"fits": True, "devices": [0], "reason": ""}
        srv, panel, _ = make({("GET", "/api/slots/plan"): plan})
        r = call(srv, "fit_check", {"model": "cold-4b"})
        self.assertEqual(r["result"]["structuredContent"], plan)
        self.assertEqual(panel.calls[-1][2], {"model": "cold-4b", "role": "worker"})

    def test_diagnose(self):
        log = "\n".join(f"line {i}" for i in range(100))
        srv, _, _ = make({("GET", "/api/model/diag"): {"diag": None},
                          ("GET", "/api/router/log"): {"log": log}})
        s = call(srv, "diagnose", {"model": "cold-4b", "log_lines": 3})["result"]["structuredContent"]
        self.assertIsNone(s["diag"])
        self.assertEqual(s["log_tail"], "line 97\nline 98\nline 99")

    def test_search_trims_and_reports_hub_errors(self):
        res = [{"repo": f"u/m{i}-GGUF", "downloads": i, "likes": 1, "updated": "2026-10-01",
                "created": "2026-09-01", "gated": False, "platforms": ["x"]} for i in range(40)]
        srv, panel, _ = make({("POST", "/api/hub/search"): {"results": res, "vram_mib": 32000,
                                                            "installed": ["u/m3-GGUF"]}})
        s = call(srv, "search_models", {"query": "qwen", "sort": "likes"})["result"]["structuredContent"]
        self.assertEqual(len(s["results"]), 20)
        self.assertNotIn("platforms", s["results"][0])
        self.assertTrue(s["results"][3]["installed"])
        self.assertEqual(panel.calls[-1][2], {"query": "qwen", "sort": "likes"})
        srv, _, _ = make({("POST", "/api/hub/search"): {"error": "HTTP 503", "results": []}})
        r = call(srv, "search_models", {"query": "qwen"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("503", text_of(r))

    def test_list_files_drops_the_bulky_prediction(self):
        listing = {"files": [{"path": "m-Q4_K_M.gguf", "size": 5 * 1024**3, "shards": 1,
                              "fit": "fits", "predict": {"big": "blob"}}],
                   "mmproj": [{"path": "mmproj-f16.gguf", "size": 1}], "mtp": []}
        srv, _, _ = make({("POST", "/api/hub/files"): listing})
        s = call(srv, "list_files", {"repo": "u/m"})["result"]["structuredContent"]
        self.assertEqual(s["files"], [{"path": "m-Q4_K_M.gguf", "size_gib": 5.0, "shards": 1,
                                       "fit": "fits"}])
        self.assertEqual(s["mmproj"], ["mmproj-f16.gguf"])

    def test_download(self):
        srv, panel, _ = make({("POST", "/api/hub/download"): {"started": True, "dest": "D:/m"}})
        r = call(srv, "download_model", {"repo": "u/m", "path": "a-00001-of-00002.gguf", "shards": 2})
        self.assertFalse(r["result"]["isError"])
        self.assertEqual(panel.calls[-1][2], {"repo": "u/m", "path": "a-00001-of-00002.gguf",
                                              "shards": 2})
        srv, _, _ = make({("POST", "/api/hub/download"): {"started": False, "dest": "D:/m"}})
        r = call(srv, "download_model", {"repo": "u/m", "path": "a.gguf"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("download_progress", text_of(r))


class Forge(unittest.TestCase):
    BOARD = [{"id": f"c{i}", "label": f"setup {i}", "score": 1 - i / 10, "code": 100 - i,
              "prose": 50, "vram_mib": 9000, "finalist": True, "sanity": None,
              "settings": {"model": "D:/secret/path.gguf"}} for i in range(8)]
    SHARE = {"before": {"code": 26.8}, "after": {"code": 126.0}, "x": {"code": 4.7},
             "markdown": "**Auto-Forge** ...", "settings": {"n-gpu-layers": "99"}}

    def test_start_passes_the_choices_and_warns_about_the_gpus(self):
        srv, panel, _ = make({("POST", "/api/forge/start"): {"ok": True, "model": "big-27b",
                                                             "req": {"intent": "coding"}}})
        tools = {t["name"]: t for t in srv.handle(req("tools/list"))["result"]["tools"]}
        self.assertIn("unload", tools["forge_start"]["description"])
        r = call(srv, "forge_start", {"model": "big-27b", "intent": "coding", "ctx": 32768,
                                      "thorough": True, "allow_lossy_kv": True})
        self.assertFalse(r["result"]["isError"])
        self.assertEqual(panel.calls[-1], ("POST", "/api/forge/start", {
            "model": "big-27b", "intent": "coding", "ctx": 32768, "time": "thorough",
            "allow_loss": True, "drop_vision": False}))

    def test_status_while_running_keeps_the_top_five_without_paths(self):
        st = {"stage": "round_a", "running": True, "progress": {"done": 3, "total": 20},
              "loading": "setup 3", "leaderboard": self.BOARD, "failed": [], "rejected": [],
              "result": None, "error": None, "applied": None, "req": {"intent": "coding"},
              "model": "big-27b"}
        srv, panel, _ = make({("GET", "/api/forge/status"): st})
        s = call(srv, "forge_status", {"model": "big-27b"})["result"]["structuredContent"]
        self.assertTrue(s["running"])
        self.assertEqual(len(s["leaderboard"]), 5)
        self.assertNotIn("settings", s["leaderboard"][0])
        self.assertNotIn("secret", json.dumps(s))
        self.assertNotIn("share", s)

    def test_status_when_done_brings_the_before_and_after(self):
        st = {"stage": "done", "running": False, "progress": {"done": 20, "total": 20},
              "loading": None, "leaderboard": self.BOARD[:2], "failed": [], "rejected": [],
              "result": {"winner": {"id": "c0"}}, "error": None, "applied": None,
              "req": {"intent": "coding"}, "model": "big-27b"}
        srv, _, _ = make({("GET", "/api/forge/status"): st, ("GET", "/api/forge/share"): self.SHARE})
        s = call(srv, "forge_status", {"model": "big-27b"})["result"]["structuredContent"]
        self.assertEqual(s["share"]["x"], {"code": 4.7})
        self.assertIn("Auto-Forge", s["share"]["markdown"])

    def test_apply_and_undo(self):
        srv, panel, _ = make({("POST", "/api/forge/apply"): {"ok": True, "updates": {"n-gpu-layers": "99"}},
                              ("POST", "/api/forge/undo"): {"ok": True}})
        tools = {t["name"]: t for t in srv.handle(req("tools/list"))["result"]["tools"]}
        self.assertIn("asked", tools["forge_apply"]["description"])
        self.assertFalse(call(srv, "forge_apply", {"model": "big-27b"})["result"]["isError"])
        self.assertEqual(panel.calls[-1], ("POST", "/api/forge/apply", {"model": "big-27b"}))
        self.assertFalse(call(srv, "forge_undo", {"model": "big-27b"})["result"]["isError"])
        self.assertEqual(panel.calls[-1], ("POST", "/api/forge/undo", {"model": "big-27b"}))

    def test_a_refused_start_is_a_tool_error(self):
        srv, _, _ = make({("POST", "/api/forge/start"): M.PanelError("HTTP 409 from the LlamaForge panel: a Forge run is already going")})
        r = call(srv, "forge_start", {"model": "big-27b"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("already going", text_of(r))


class PickModel(unittest.TestCase):
    def test_a_worker_before_the_main(self):
        self.assertEqual(M.pick_model(STATE)["id"], "small-9b")

    def test_single_model_mode_takes_the_loaded_one(self):
        s = state_with(slots={"enabled": False, "main": ""})
        self.assertEqual(M.pick_model(s)["id"], "big-27b")

    def test_only_the_main_loaded(self):
        s = state_with()
        s["models"][1]["status"] = "unloaded"
        self.assertEqual(M.pick_model(s)["id"], "big-27b")

    def test_named(self):
        self.assertEqual(M.pick_model(STATE, "big-27b")["id"], "big-27b")
        with self.assertRaisesRegex(M.ToolError, "load_model"):
            M.pick_model(STATE, "cold-4b")
        with self.assertRaisesRegex(M.ToolError, "list_models"):
            M.pick_model(STATE, "ghost")

    def test_nothing_loaded(self):
        s = state_with()
        for m in s["models"]:
            m["status"] = "unloaded"
        with self.assertRaisesRegex(M.ToolError, "load_model"):
            M.pick_model(s)


class Ask(unittest.TestCase):
    def test_one_completion_from_the_default_worker(self):
        seen = {}

        def chat(url, body, key, timeout):
            seen.update(url=url, body=body, key=key)
            return {"choices": [{"message": {"content": "  4  ", "reasoning_content": "2+2"}}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 1}}
        srv, _, _ = make(chat=chat)
        r = call(srv, "ask", {"prompt": "2+2?", "system": "terse", "max_tokens": 16})
        self.assertFalse(r["result"]["isError"], text_of(r))
        self.assertEqual(r["result"]["content"][0]["text"], "4")
        self.assertEqual(seen["url"], "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(seen["key"], "loc-key")
        self.assertEqual(seen["body"]["model"], "small-9b")
        self.assertEqual(seen["body"]["messages"], [{"role": "system", "content": "terse"},
                                                    {"role": "user", "content": "2+2?"}])
        self.assertEqual(seen["body"]["max_tokens"], 16)
        self.assertFalse(seen["body"]["stream"])
        self.assertEqual(r["result"]["structuredContent"]["model"], "small-9b")

    def test_thinking_ate_the_budget(self):
        chat = lambda *a: {"choices": [{"message": {"content": "", "reasoning_content": "hmm"},
                                        "finish_reason": "length"}]}
        srv, _, _ = make(chat=chat)
        r = call(srv, "ask", {"prompt": "x"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("max_tokens", text_of(r))

    def test_endpoint_errors(self):
        def chat(*a):
            raise M.PanelError("HTTP 400: model not found")
        srv, _, _ = make(chat=chat)
        r = call(srv, "ask", {"prompt": "x", "model": "big-27b"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("model not found", text_of(r))


class PiRun(unittest.TestCase):
    def fake(self, result=None):
        self.kw = {}

        def runner(**kw):
            self.kw = kw
            out = {"ok": True, "text": "Refactored 3 files.", "error": "", "model": kw["model"],
                   "endpoint": kw["endpoint"], "cwd": kw["cwd"], "tools": kw["tools"],
                   "tool_calls": [{"tool": "edit", "error": False}], "turns": 4,
                   "usage": {"input": 900, "output": 120}, "stop": "stop", "elapsed_s": 12.5,
                   "timed_out": False, "cancelled": False}
            out.update(result or {})
            return out
        return runner

    def test_runs_on_the_default_worker(self):
        srv, _, _ = make(runner=self.fake())
        r = call(srv, "pi_run", {"task": "refactor utils", "cwd": "D:/proj", "tools": "edit",
                                 "timeout_s": 120})
        self.assertFalse(r["result"]["isError"], text_of(r))
        self.assertEqual(r["result"]["content"][0]["text"], "Refactored 3 files.")
        kw = self.kw
        self.assertEqual((kw["task"], kw["model"], kw["endpoint"], kw["key"]),
                         ("refactor utils", "small-9b", "http://127.0.0.1:8080", "loc-key"))
        self.assertEqual((kw["cwd"], kw["tools"], kw["timeout"], kw["ctx"], kw["pi_bin"]),
                         ("D:/proj", "edit", 120, "16384", "/x/pi.js"))
        s = r["result"]["structuredContent"]
        self.assertEqual((s["turns"], s["tool_calls"][0]["tool"]), (4, "edit"))
        self.assertNotIn("text", s)

    def test_the_user_key_wins_over_the_local_one(self):
        srv, _, _ = make(runner=self.fake(),
                         cfg=lambda: dict(CFG, router_api_key="user-key"))
        call(srv, "pi_run", {"task": "t"})
        self.assertEqual(self.kw["key"], "user-key")

    def test_failure_keeps_partial_output(self):
        srv, _, _ = make(runner=self.fake({"ok": False, "error": "pi hit the 60s timeout",
                                           "text": "half done", "timed_out": True}))
        r = call(srv, "pi_run", {"task": "t"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("timeout", text_of(r))
        self.assertIn("half done", text_of(r))

    def test_bad_tool_set(self):
        srv, _, _ = make(runner=self.fake())
        r = call(srv, "pi_run", {"task": "t", "tools": "root"})
        self.assertTrue(r["result"]["isError"])

    def test_progress_notifications(self):
        def runner(**kw):
            kw["progress"]("pi is using read")
            kw["progress"]("pi finished a turn")
            return self.fake()(**kw)
        srv, _, sent = make(runner=runner)
        srv.handle(req("tools/call", {"name": "pi_run", "arguments": {"task": "t"},
                                      "_meta": {"progressToken": "tok-1"}}))
        notes = [m for m in sent if m.get("method") == "notifications/progress"]
        self.assertGreaterEqual(len(notes), 2)
        self.assertTrue(all(n["params"]["progressToken"] == "tok-1" for n in notes))
        prog = [n["params"]["progress"] for n in notes]
        self.assertEqual(prog, sorted(set(prog)))   # strictly increasing
        self.assertIn("read", " ".join(n["params"].get("message", "") for n in notes))


class Threads(unittest.TestCase):
    def test_cancel_kills_the_run_and_sends_nothing(self):
        started, finished = threading.Event(), threading.Event()

        def runner(**kw):
            started.set()
            kw["cancel"].wait(10)
            finished.set()
            return {"ok": False, "error": "cancelled", "text": "", "cancelled": True}
        sent = []
        srv = M.Server(panel=FakePanel(), cfg=lambda: dict(CFG), runner=runner,
                       send=sent.append, threaded=True, poll_s=0)
        self.assertIsNone(srv.handle(req("tools/call", {"name": "pi_run",
                                                        "arguments": {"task": "t"}}, id_=7)))
        self.assertTrue(started.wait(5))
        srv.handle({"jsonrpc": "2.0", "method": "notifications/cancelled",
                    "params": {"requestId": 7, "reason": "user"}})
        self.assertTrue(finished.wait(5))
        time.sleep(0.2)
        self.assertFalse([m for m in sent if m.get("id") == 7])

    def test_a_threaded_call_answers_through_send(self):
        sent = []
        srv = M.Server(panel=FakePanel(), cfg=lambda: dict(CFG), send=sent.append,
                       threaded=True, poll_s=0)
        srv.handle(req("tools/call", {"name": "status", "arguments": {}}, id_="a"))
        for _ in range(50):
            if sent:
                break
            time.sleep(0.05)
        self.assertEqual(sent[0]["id"], "a")
        self.assertFalse(sent[0]["result"]["isError"])


class Stdio(unittest.TestCase):
    def test_line_framing_and_eof(self):
        lines = [json.dumps(req("initialize", {"protocolVersion": "2025-11-25"}, 1)),
                 json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                 "", json.dumps(req("tools/list", None, 2)),
                 json.dumps(req("tools/call", {"name": "status", "arguments": {}}, 3))]
        inp = io.StringIO("\n".join(lines) + "\n")
        out = io.StringIO()
        srv = M.Server(panel=FakePanel(), cfg=lambda: dict(CFG), threaded=True, poll_s=0)
        M.serve(inp, out, srv)
        got = out.getvalue().split("\n")
        self.assertEqual(got[-1], "")       # every message ends its line
        msgs = [json.loads(x) for x in got[:-1]]
        self.assertEqual(sorted(m["id"] for m in msgs), [1, 2, 3])


class Http(unittest.TestCase):
    def test_a_slow_answer_is_a_timeout_not_a_dead_panel(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)                      # accepts the connection, never answers
        try:
            port = s.getsockname()[1]
            panel = M.Panel(cfg=lambda: {"panel_port": port})
            with self.assertRaises(M.PanelError) as cm:
                panel.get("/api/state", timeout=0.3)
            self.assertIn("did not answer within", str(cm.exception))
            self.assertNotIn("isn't running", str(cm.exception))
        finally:
            s.close()

    def test_a_closed_port_says_start_llamaforge(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        with self.assertRaises(M.PanelError) as cm:
            # Windows retries a refused loopback SYN for ~2s before it says so
            M.Panel(cfg=lambda: {"panel_port": port}).get("/api/state", timeout=10)
        self.assertIn("isn't running", str(cm.exception))


class Setup(unittest.TestCase):
    PY = r"C:\Program Files\Python314\python.exe"
    SCRIPT = r"D:\Llama Forge\backend\mcp_server.py"

    def test_snippets_quote_paths_with_spaces(self):
        s = M.setup_info(self.PY, self.SCRIPT)
        self.assertEqual(s["claude"], 'claude mcp add --scope user llamaforge -- '
                         f'"{self.PY}" "{self.SCRIPT}"')
        cfg = json.loads(s["json"])["mcpServers"]["llamaforge"]
        self.assertEqual(cfg, {"command": self.PY, "args": [self.SCRIPT]})

    def test_codex_toml_escapes_backslashes_and_raises_the_timeout(self):
        toml = M.setup_info(self.PY, self.SCRIPT)["codex_toml"]
        self.assertIn("[mcp_servers.llamaforge]", toml)
        self.assertIn(r'command = "C:\\Program Files\\Python314\\python.exe"', toml)
        self.assertIn(r'args = ["D:\\Llama Forge\\backend\\mcp_server.py"]', toml)
        self.assertIn("tool_timeout_sec = 900", toml)

    def test_no_secrets_and_lists_the_tools(self):
        s = M.setup_info(self.PY, self.SCRIPT)
        self.assertNotIn("key", json.dumps(s).lower())
        self.assertEqual(s["tools"][-1], "forge_undo")

    def test_pythonw_is_swapped_for_the_console_build(self):
        d = tempfile.mkdtemp()
        try:
            for n in ("python.exe", "pythonw.exe"):
                open(os.path.join(d, n), "w").close()
            s = M.setup_info(os.path.join(d, "pythonw.exe"), self.SCRIPT)
            self.assertEqual(s["python"], os.path.join(d, "python.exe"))
        finally:
            shutil.rmtree(d)

    def test_the_route_serves_it(self):
        import routes
        status, out = routes.GET_ROUTES["/api/mcp/setup"](routes.Req())
        self.assertEqual(status, 200)
        self.assertTrue(out["script"].endswith("mcp_server.py"))
        self.assertTrue(os.path.isfile(out["script"]))


if __name__ == "__main__":
    unittest.main()
