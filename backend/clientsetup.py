"""Generate deliberate OpenAI-compatible client snippets from a resolved target."""
import json, shlex


def _quote(value, shell):
    if shell == "posix":
        return shlex.quote(value)
    if shell == "powershell":
        return "'" + value.replace("'", "''") + "'"
    raise ValueError("shell must be posix or powershell")


def _apps(v1, key, model, shell):
    """Paste-ready settings for the apps people point at a local server. Values
    are JSON-escaped in YAML (a valid double-quoted scalar) and shell-quoted in
    commands, so a model id can never break out of its field."""
    j = lambda s: json.dumps(s, ensure_ascii=True)
    docker = v1.replace("://127.0.0.1:", "://host.docker.internal:").replace("://localhost:", "://host.docker.internal:")
    webui = ("Admin Panel > Settings > Connections > OpenAI API > +\n"
             f"  URL: {v1}\n  Key: {key}\n"
             + (f"Open WebUI in Docker: use {docker} (127.0.0.1 is the container itself),\nor "
                if docker != v1 else "In Docker: ")
             + f"start it with -e OPENAI_API_BASE_URL={docker} -e OPENAI_API_KEY={key}")
    cont = ("name: LlamaForge\nversion: 0.0.1\nschema: v1\nmodels:\n"
            f"  - name: {j(model)}\n    provider: openai\n    model: {j(model)}\n"
            f"    apiBase: {j(v1)}\n    apiKey: {j(key)}\n    roles: [chat, edit, apply]")
    cline = (f"API Provider: OpenAI Compatible\nBase URL: {v1}\nAPI Key: {key}\nModel ID: {model}")
    cont_line = " `\n  " if shell == "powershell" else " \\\n  "
    aider = cont_line.join(["aider --model " + _quote("openai/" + model, shell),
                            "--openai-api-base " + _quote(v1, shell),
                            "--openai-api-key " + _quote(key, shell)])
    return [{"name": "Open WebUI", "text": webui},
            {"name": "Continue (config.yaml)", "text": cont},
            {"name": "Cline", "text": cline},
            {"name": "aider", "text": aider}]


def generate(endpoint, api_key, model, backend, shell):
    base = endpoint.rstrip("/")
    payload_obj = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    compact = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False)
    args = [
        "curl", _quote(base + "/v1/chat/completions", shell),
        "-H", _quote("Content-Type: application/json", shell),
    ]
    if api_key:
        args += ["-H", _quote("Authorization: Bearer " + api_key, shell)]
    args += ["-d", _quote(compact, shell)]
    continuation = " `\n  " if shell == "powershell" else " \\\n  "
    curl = continuation.join(args)
    shown_key = api_key or "not-required"
    if shell == "powershell":
        environment = (
            "$env:OPENAI_BASE_URL = " + _quote(base + "/v1", shell) + "\n"
            "$env:OPENAI_API_KEY = " + _quote(shown_key, shell) + "\n")
    else:
        environment = (
            "export OPENAI_BASE_URL=" + _quote(base + "/v1", shell) + "\n"
            "export OPENAI_API_KEY=" + _quote(shown_key, shell) + "\n")
    environment += "# model id: " + json.dumps(model, ensure_ascii=True)
    v1 = base + "/v1"
    return {
        "apps": _apps(v1, shown_key, model, shell),
        "backend": backend,
        "endpoint": base,
        "auth_required": bool(api_key),
        "curl": curl,
        "environment": environment,
        "payload": json.dumps(payload_obj, indent=2, ensure_ascii=False),
    }
