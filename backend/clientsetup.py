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


def _retrieval_apps(v1, key, model, kind):
    """Embedding and rerank models feed RAG, not chat: point Open WebUI's
    Documents settings (and Continue's embed role) at them."""
    j = lambda s: json.dumps(s, ensure_ascii=True)
    if kind == "embed":
        webui = ("Admin Panel > Settings > Documents > Embedding\n"
                 f"  Engine: OpenAI\n  URL: {v1}\n  Key: {key}\n  Model: {model}")
        cont = ("name: LlamaForge\nversion: 0.0.1\nschema: v1\nmodels:\n"
                f"  - name: {j(model)}\n    provider: openai\n    model: {j(model)}\n"
                f"    apiBase: {j(v1)}\n    apiKey: {j(key)}\n    roles: [embed]")
        return [{"name": "Open WebUI (Documents)", "text": webui},
                {"name": "Continue (config.yaml)", "text": cont}]
    webui = ("Admin Panel > Settings > Documents > Reranking (turn on Hybrid Search)\n"
             f"  Engine: External\n  URL: {v1}/rerank\n  Key: {key}\n  Model: {model}")
    return [{"name": "Open WebUI (Documents)", "text": webui}]


_CALLS = {
    "chat": ("/v1/chat/completions",
             lambda m: {"model": m, "messages": [{"role": "user", "content": "Hello"}],
                        "stream": False}),
    "embed": ("/v1/embeddings", lambda m: {"model": m, "input": "Hello"}),
    "rerank": ("/v1/rerank",
               lambda m: {"model": m, "query": "What is a llama?",
                          "documents": ["A llama is a South American camelid.",
                                        "llama.cpp runs language models locally."]}),
}


def generate(endpoint, api_key, model, backend, shell, kind="chat"):
    kind = kind if kind in _CALLS else "chat"
    base = endpoint.rstrip("/")
    path, make = _CALLS[kind]
    payload_obj = make(model)
    compact = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False)
    args = [
        "curl", _quote(base + path, shell),
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
        "apps": (_apps(v1, shown_key, model, shell) if kind == "chat"
                 else _retrieval_apps(v1, shown_key, model, kind)),
        "kind": kind,
        "backend": backend,
        "endpoint": base,
        "auth_required": bool(api_key),
        "curl": curl,
        "environment": environment,
        "payload": json.dumps(payload_obj, indent=2, ensure_ascii=False),
    }
