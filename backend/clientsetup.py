"""Generate deliberate OpenAI-compatible client snippets from a resolved target."""
import json, shlex


def _quote(value, shell):
    if shell == "posix":
        return shlex.quote(value)
    if shell == "powershell":
        return "'" + value.replace("'", "''") + "'"
    raise ValueError("shell must be posix or powershell")


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
    continuation = " `\n  " if shell == "powershell" else " "
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
    return {
        "backend": backend,
        "endpoint": base,
        "auth_required": bool(api_key),
        "curl": curl,
        "environment": environment,
        "payload": json.dumps(payload_obj, indent=2, ensure_ascii=False),
    }
