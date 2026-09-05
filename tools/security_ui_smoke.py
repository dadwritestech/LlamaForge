#!/usr/bin/env python3
"""Local real-frontend smoke test for explicit credential handling.

This fixture deliberately lives outside unittest discovery and CI.  It serves
the repository's real browser application while replacing only the API surface
needed by the security acceptance scenario.
"""
from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
FIXTURE_JS = Path(__file__).with_name("security_ui_smoke.js")
SENTINEL = "fixture-secret-must-not-leak-" + "f" * 16

PUBLIC_CONFIG = {
    "theme": "dark",
    "cvd": False,
    "auto_load_model": "",
    "vram_bandwidths": {},
    "presets": {},
    "preset_bindings": {},
    "active_engine": "llamacpp",
    "router_api_key_configured": False,
}
DEFAULT_MODELS = [
    {"id": "llama-fixture", "backend": "llamacpp", "status": "loaded",
     "settings": {}, "modalities": ["text"], "in_ini": True,
     "endpoint": "http://127.0.0.1:8080", "eff_ctx": 4096},
    {"id": "vllm-fixture", "backend": "vllm", "status": "loaded",
     "settings": {}, "modalities": ["text"], "in_ini": True,
     "endpoint": "http://127.0.0.1:8081", "eff_ctx": 4096},
]
MODEL_SCENARIOS = {
    "default": (DEFAULT_MODELS, "llamacpp"),
    # Deliberately put vLLM first. State rows are not specified to be ordered
    # by backend, and this catches DOM reconciliation that keys only on id.
    "shared-llamacpp": ([
        {"id": "shared-fixture", "backend": "vllm", "status": "loaded",
         "settings": {}, "modalities": ["text"], "in_ini": True,
         "endpoint": "http://127.0.0.1:8081", "eff_ctx": 4096},
        {"id": "shared-fixture", "backend": "llamacpp", "status": "loaded",
         "settings": {}, "modalities": ["text"], "in_ini": True,
         "endpoint": "http://127.0.0.1:8080", "eff_ctx": 4096},
    ], "llamacpp"),
    "shared-ikllama": ([
        {"id": "shared-fixture", "backend": "vllm", "status": "loaded",
         "settings": {}, "modalities": ["text"], "in_ini": True,
         "endpoint": "http://127.0.0.1:8081", "eff_ctx": 4096},
        {"id": "shared-fixture", "backend": "ikllama", "status": "loaded",
         "settings": {}, "modalities": ["text"], "in_ini": True,
         "endpoint": "http://127.0.0.1:8080", "eff_ctx": 4096},
    ], "ikllama"),
}
LOCAL_NETWORK = {
    "access_scope": "local", "host": "127.0.0.1", "port": 8080,
    "lan_ip": "192.168.1.44", "router_running": True,
    "configured_security_status": "local", "listener_status": "listening",
    "key_status": "absent", "has_api_key": False,
    "remediation_required": False, "message": "",
}

SCENARIOS = {
    "local-no-key": LOCAL_NETWORK,
    "local-keyed": {
        **LOCAL_NETWORK,
        "configured_security_status": "local_keyed",
        "key_status": "strong", "has_api_key": True,
    },
    "lan-protected": {
        **LOCAL_NETWORK,
        "access_scope": "lan", "host": "0.0.0.0",
        "configured_security_status": "protected",
        "key_status": "strong", "has_api_key": True,
    },
    "protected-legacy": {
        **LOCAL_NETWORK,
        "access_scope": "lan", "host": "0.0.0.0",
        "configured_security_status": "protected_legacy",
        "key_status": "legacy", "has_api_key": True,
        "message": "Rotate this legacy API key when convenient.",
    },
    "unsafe-legacy": {
        **LOCAL_NETWORK,
        "access_scope": "legacy", "host": "192.168.1.55",
        "configured_security_status": "unsafe_legacy",
        "key_status": "absent", "has_api_key": False,
        "remediation_required": True,
        "message": "Stored LAN policy is unsafe and the router start is refused.",
    },
    "forced-restart-failure": {
        **LOCAL_NETWORK,
        "configured_security_status": "local_keyed",
        "key_status": "strong", "has_api_key": True,
    },
}

SCHEMA = {
    "count": 1,
    "groups": [{
        "name": "Fixture",
        "knobs": [{
            "key": "fixture-note", "aliases": ["fixture-note"],
            "type": "string", "default": "", "desc": "Harmless fixture knob",
        }],
    }],
}
SETUP = {
    "prereqs": {
        "tools": {
            "git": {"present": True, "version": "fixture"},
            "cmake": {"present": True, "version": "fixture"},
        },
        "msvc": {"label": "C++ compiler", "present": True, "url": ""},
        "cuda": {"applicable": False, "present": False, "version": ""},
        "installers": {},
    },
    "hardware": {
        "cpu": {"name": "Fixture CPU", "cores": 8, "threads": 16},
        "gpus": [], "cmake_flags": {}, "notes": [],
    },
}


class FixtureState:
    DELAY_KINDS = frozenset(("client", "agent", "network"))

    def __init__(self):
        self.lock = threading.Lock()
        self.network = copy.deepcopy(LOCAL_NETWORK)
        self.models = copy.deepcopy(DEFAULT_MODELS)
        self.active_engine = "llamacpp"
        self.fail_next_restart = False
        self.requests = []
        self.delays = {}
        self.delay_completed = {kind: 0 for kind in self.DELAY_KINDS}

    def record(self, method, path, body=None):
        with self.lock:
            self.requests.append({
                "method": method, "path": path, "body": copy.deepcopy(body),
            })

    def snapshot_requests(self):
        with self.lock:
            return copy.deepcopy(self.requests)

    def select(self, name):
        if name not in SCENARIOS:
            return False
        with self.lock:
            self.network = copy.deepcopy(SCENARIOS[name])
            self.fail_next_restart = name == "forced-restart-failure"
        return True

    def select_models(self, name):
        scenario = MODEL_SCENARIOS.get(name)
        if scenario is None:
            return False
        rows, active_engine = scenario
        with self.lock:
            self.models = copy.deepcopy(rows)
            self.active_engine = active_engine
        return True

    def snapshot_models(self):
        with self.lock:
            return copy.deepcopy(self.models), self.active_engine

    def arm_delay(self, kind, delay_ms=750):
        if (kind not in self.DELAY_KINDS or not isinstance(delay_ms, int) or
                delay_ms < 100 or delay_ms > 5000):
            return False
        event = threading.Event()
        with self.lock:
            previous = self.delays.get(kind)
            self.delays[kind] = event
        if previous is not None:
            previous.set()
        timer = threading.Timer(delay_ms / 1000, event.set)
        timer.daemon = True
        timer.start()
        return True

    def release_delay(self, kind):
        if kind not in self.DELAY_KINDS:
            return False
        with self.lock:
            event = self.delays.get(kind)
        if event is None:
            return False
        event.set()
        return True

    def wait_delay(self, kind):
        with self.lock:
            event = self.delays.get(kind)
        if event is not None:
            event.wait(timeout=15)
        return event

    def complete_delay(self, kind, event):
        if event is None:
            return
        with self.lock:
            if self.delays.get(kind) is event:
                del self.delays[kind]
            self.delay_completed[kind] += 1

    def snapshot_delays(self):
        with self.lock:
            return {
                "pending": sorted(self.delays),
                "completed": copy.deepcopy(self.delay_completed),
            }


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, FixtureHandler)
        self.fixture = FixtureState()


class FixtureHandler(BaseHTTPRequestHandler):
    server: FixtureServer

    def log_message(self, _format, *args):
        pass

    def _send(self, status, payload, content_type="application/json; charset=utf-8"):
        if isinstance(payload, str):
            raw = payload.encode("utf-8")
        else:
            raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": "invalid fixture request"})
            return None
        return body

    def _send_delayed(self, kind, status, payload):
        event = self.server.fixture.wait_delay(kind)
        try:
            self._send(status, payload)
        finally:
            self.server.fixture.complete_delay(kind, event)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self.server.fixture.record("GET", path, None)
        if path in ("/api/schema", "/api/vllm/schema"):
            self._send(200, SCHEMA)
        elif path == "/api/state":
            models, active_engine = self.server.fixture.snapshot_models()
            public_config = copy.deepcopy(PUBLIC_CONFIG)
            public_config["active_engine"] = active_engine
            self._send(200, {
                "models": models,
                "global": {},
                "gpus": [],
                "config": public_config,
                "platform": "windows",
                "vllm_supported": True,
                "backends": ["llamacpp", "vllm"],
                "active_engine": active_engine,
                "config_error": None,
                "onboarding": {
                    "server_bin_ok": True, "model_count": len(models),
                    "ui_mode": "advanced", "onboarded": True,
                },
            })
        elif path in ("/api/router/log", "/api/vllm/log"):
            self._send(200, {"log": "fixture idle"})
        elif path == "/api/setup":
            self._send(200, SETUP)
        elif path == "/api/vllm/setup":
            self._send(200, {"supported": False})
        elif path == "/api/network":
            with self.server.fixture.lock:
                network = copy.deepcopy(self.server.fixture.network)
            self._send(200, network)
        elif path == "/api/agent/config":
            self._send(404, {"error": "not found"})
        elif path == "/_fixture/requests":
            self._send(200, {"requests": self.server.fixture.snapshot_requests()})
        elif path == "/_fixture/delays":
            self._send(200, self.server.fixture.snapshot_delays())
        elif path == "/_fixture.js":
            self._send(200, FIXTURE_JS.read_text(encoding="utf-8"),
                       "text/javascript; charset=utf-8")
        else:
            self._serve_static(path)

    def do_POST(self):
        path = urlsplit(self.path).path
        body = self._body()
        if body is None:
            return
        if path.startswith("/api/"):
            self.server.fixture.record("POST", path, body)
        if path == "/_fixture/scenario":
            if not isinstance(body, dict) or not self.server.fixture.select(body.get("name")):
                self._send(400, {"ok": False, "error": "unknown fixture scenario"})
            else:
                self._send(200, {"ok": True})
        elif path == "/_fixture/models":
            if (not isinstance(body, dict) or
                    not self.server.fixture.select_models(body.get("name"))):
                self._send(400, {"ok": False, "error": "unknown model scenario"})
            else:
                self._send(200, {"ok": True})
        elif path == "/_fixture/delay":
            if (not isinstance(body, dict) or
                    not self.server.fixture.arm_delay(
                        body.get("kind"), body.get("delay_ms", 750))):
                self._send(400, {"ok": False, "error": "unknown delay kind"})
            else:
                self._send(200, {"ok": True})
        elif path == "/_fixture/release":
            if (not isinstance(body, dict) or
                    not self.server.fixture.release_delay(body.get("kind"))):
                self._send(400, {"ok": False, "error": "delay not armed"})
            else:
                self._send(200, {"ok": True})
        elif path == "/api/network":
            self._post_network(body)
        elif path == "/api/client/config":
            self._post_client(body)
        elif path == "/api/agent/config":
            self._post_agent_config(body)
        elif path == "/api/agent/apply":
            self._send(200, {
                "ok": True, "path": "fixture-agent.json", "backup": None,
                "action": "updated",
            })
        else:
            self._send(404, {"error": "not found"})

    def _post_network(self, body):
        allowed = {"access_scope", "key_action", "api_key"}
        if (not isinstance(body, dict) or set(body) - allowed or
                body.get("access_scope") not in ("local", "lan") or
                body.get("key_action") not in ("keep", "generate", "replace", "clear")):
            self._send(400, {"ok": False, "error": "invalid canonical network request"})
            return
        scope = body["access_scope"]
        action = body["key_action"]
        supplied = body.get("api_key")
        if action == "replace":
            if (not isinstance(supplied, str) or
                    not re.fullmatch(r"[A-Za-z0-9._~-]{32,256}", supplied)):
                self._send(400, {"ok": False, "error": "invalid replacement key"})
                return
        elif "api_key" in body:
            self._send(400, {"ok": False, "error": "unexpected API key field"})
            return

        with self.server.fixture.lock:
            current = self.server.fixture.network
            has_key = bool(current.get("has_api_key"))
            if action in ("generate", "replace"):
                has_key = True
            elif action == "clear":
                has_key = False
            if scope == "lan" and not has_key:
                self._send(400, {"ok": False, "error": "LAN requires a key"})
                return
            if action == "clear" and scope != "local":
                self._send(400, {"ok": False, "error": "LAN key cannot be removed"})
                return

            fail = self.server.fixture.fail_next_restart
            self.server.fixture.fail_next_restart = False
            configured = ("protected" if scope == "lan" else
                          ("local_keyed" if has_key else "local"))
            network = {
                "access_scope": scope,
                "host": "0.0.0.0" if scope == "lan" else "127.0.0.1",
                "port": 8080, "lan_ip": "192.168.1.44",
                "router_running": True,
                "configured_security_status": configured,
                "listener_status": "listening",
                "key_status": "strong" if has_key else "absent",
                "has_api_key": has_key,
                "remediation_required": False, "message": "",
            }
            self.server.fixture.network = network

        out = {
            **copy.deepcopy(network),
            "ok": not fail,
            "saved": True,
            "restart_status": "failed" if fail else "running",
        }
        if fail:
            out["error"] = "Router restart failed; inspect Router Log."
        if action == "generate":
            out["generated_api_key"] = SENTINEL
        self._send_delayed("network", 500 if fail else 200, out)

    def _post_client(self, body):
        if not isinstance(body, dict) or set(body) != {"model", "backend"}:
            self._send(400, {"error": "invalid client request"})
            return
        backend = body.get("backend")
        model = body.get("model")
        if ((backend, model) == ("llamacpp", "llama-fixture") or
                (backend in ("llamacpp", "ikllama") and
                 model == "shared-fixture")):
            endpoint = "http://127.0.0.1:8080"
            payload = {
                "backend": backend, "endpoint": endpoint,
                "auth_required": True, "model_loaded": True,
                "curl": ("curl http://127.0.0.1:8080/v1/chat/completions "
                         "-H 'Authorization: Bearer " + SENTINEL + "'"),
                "environment": ("OPENAI_BASE_URL=http://127.0.0.1:8080/v1\n"
                                "OPENAI_API_KEY=" + SENTINEL),
                "payload": json.dumps({"model": model, "messages": []}),
            }
        elif (backend == "vllm" and
              model in ("vllm-fixture", "shared-fixture")):
            endpoint = "http://127.0.0.1:8081"
            payload = {
                "backend": backend, "endpoint": endpoint,
                "auth_required": False, "model_loaded": True,
                "curl": "curl http://127.0.0.1:8081/v1/chat/completions",
                "environment": ("OPENAI_BASE_URL=http://127.0.0.1:8081/v1\n"
                                "OPENAI_API_KEY=not-required"),
                "payload": json.dumps({"model": model, "messages": []}),
            }
        else:
            self._send(400, {"error": "unknown client target"})
            return
        self._send_delayed("client", 200, payload)

    def _post_agent_config(self, body):
        expected = {"agent", "model", "backend", "small", "inject"}
        if (not isinstance(body, dict) or set(body) != expected or
                body.get("model") != "llama-fixture" or
                body.get("backend") != "llamacpp"):
            self._send(400, {"error": "invalid agent request"})
            return
        self._send_delayed("agent", 200, {
            "target_path": "fixture-agent.json",
            "endpoint": "http://127.0.0.1:8090/v1",
            "instructions": "Use the generated local fixture configuration.",
            "content": "fixture-agent-key=" + SENTINEL,
        })

    def _serve_static(self, path):
        if path in ("", "/", "/index.html"):
            target = WEB / "index.html"
        elif path.startswith("/web/"):
            relative = Path(unquote(path[len("/web/"):]))
            target = WEB / relative
        else:
            self._send(404, {"error": "not found"})
            return
        try:
            resolved = target.resolve(strict=True)
            resolved.relative_to(WEB.resolve())
        except (OSError, ValueError):
            self._send(404, {"error": "not found"})
            return
        if resolved == (WEB / "index.html").resolve():
            text = resolved.read_text(encoding="utf-8")
            marker = '<script type="module" src="/_fixture.js"></script>'
            text = text.replace("</body>", marker + "\n</body>", 1)
            self._send(200, text, "text/html; charset=utf-8")
            return
        mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        raw = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class ResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.capture = False
        self.status = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "pre" and values.get("id") == "lf-security-result":
            self.capture = True
            self.status = values.get("data-status")

    def handle_endtag(self, tag):
        if tag == "pre" and self.capture:
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.text.append(data)


def _chrome_path(explicit):
    candidates = [
        explicit,
        os.environ.get("LF_CHROME"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        shutil.which("chrome"),
        shutil.which("google-chrome"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


def _redact(text):
    return (text or "").replace(SENTINEL, "[fixture secret redacted]")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome")
    args = parser.parse_args(argv)
    chrome = _chrome_path(args.chrome)
    if not chrome:
        print("FAIL: chrome-unavailable")
        return 2

    httpd = FixtureServer(("127.0.0.1", 0))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="llamaforge-security-ui-") as profile:
            url = "http://127.0.0.1:%d/#setup" % httpd.server_address[1]
            command = [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--dump-dom",
                "--virtual-time-budget=60000",
                "--user-data-dir=" + profile,
                url,
            ]
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=90,
                    encoding="utf-8", errors="replace", check=False)
            except (OSError, subprocess.TimeoutExpired):
                print("FAIL: chrome-execution-failed")
                return 2
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    stdout = _redact(result.stdout)
    stderr = _redact(result.stderr)
    parsed = ResultParser()
    parsed.feed(stdout)
    message = "".join(parsed.text).strip()
    if result.returncode == 0 and parsed.status == "pass" and message == "PASS":
        print("PASS")
        return 0
    if parsed.status == "fail" and message:
        print("FAIL: " + re.sub(r"[^a-z0-9-]", "-", message.lower()).strip("-"))
    elif result.returncode:
        diagnostic = next((line.strip() for line in stderr.splitlines() if line.strip()),
                          "chrome returned a nonzero status")
        print("FAIL: " + _redact(diagnostic))
    else:
        print("FAIL: missing-final-pass-marker")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
