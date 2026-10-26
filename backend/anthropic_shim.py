"""Anthropic <-> OpenAI translation for LlamaForge's /v1/messages shim.

Pure-stdlib, pure functions over dicts (no I/O): backend/server.py orchestrates
the actual router calls. Translates Anthropic Messages API requests to the
router's OpenAI chat-completions shape and back, including streaming SSE and
tool use, so Claude Code can drive local llama.cpp models.
"""
import json
import uuid


# ---------------- request: Anthropic -> OpenAI ----------------

def _collapse(parts):
    """A list of OpenAI content parts -> a plain string when all are text,
    else the parts list (needed for image content)."""
    if all(p.get("type") == "text" for p in parts):
        return "".join(p.get("text", "") for p in parts)
    return parts


def content_blocks_to_openai(role, content):
    """Translate one Anthropic message's content into OpenAI messages.
    Usually one message, but tool_result blocks become separate role=tool
    messages, and an assistant turn with tool_use becomes one assistant message
    carrying tool_calls."""
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    text_parts, tool_calls, tool_msgs = [], [], []
    for b in content or []:
        t = b.get("type")
        if t == "text":
            text_parts.append({"type": "text", "text": b.get("text", "")})
        elif t == "image":
            src = b.get("source", {})
            if src.get("type") == "base64":
                url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
                text_parts.append({"type": "image_url", "image_url": {"url": url}})
        elif t == "tool_use":
            tool_calls.append({"id": b.get("id"), "type": "function", "function": {
                "name": b.get("name"), "arguments": json.dumps(b.get("input", {}))}})
        elif t == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
            tool_msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                              "content": c if isinstance(c, str) else json.dumps(c)})
    out = []
    if role == "assistant" and tool_calls:
        msg = {"role": "assistant", "content": _collapse(text_parts) if text_parts else None,
               "tool_calls": tool_calls}
        out.append(msg)
    elif text_parts:
        out.append({"role": role, "content": _collapse(text_parts)})
    out.extend(tool_msgs)
    return out


def _fold_system(msgs, content):
    """Claude Code sends role=system messages mid-conversation (hook context).
    Strict chat templates (Qwen's) raise unless every system message leads, so
    before the first turn it joins the system prompt; after that it becomes a
    <system-reminder> on the latest user turn (or a new one after a tool result)."""
    if isinstance(content, str):
        text = content
    else:
        text = "".join(b.get("text", "") for b in content or []
                       if isinstance(b, dict) and b.get("type") == "text")
    if not text:
        return
    if all(m["role"] == "system" for m in msgs):
        if msgs:
            msgs[0]["content"] += "\n\n" + text
        else:
            msgs.append({"role": "system", "content": text})
        return
    note = f"<system-reminder>\n{text}\n</system-reminder>"
    last = msgs[-1]
    if last["role"] != "user":
        msgs.append({"role": "user", "content": note})
    elif isinstance(last["content"], str):
        last["content"] += "\n\n" + note
    else:
        last["content"].append({"type": "text", "text": "\n\n" + note})


def map_tools(tools):
    out = []
    for t in tools or []:
        out.append({"type": "function", "function": {
            "name": t.get("name"), "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})
    return out


def map_tool_choice(tc):
    if isinstance(tc, dict):
        ty = tc.get("type")
        if ty == "auto":
            return "auto"
        if ty == "any":
            return "required"
        if ty == "tool" and tc.get("name"):
            return {"type": "function", "function": {"name": tc["name"]}}
    return "auto"


def to_openai_request(a):
    out = {"model": a.get("model", "")}
    msgs = []
    system = a.get("system")
    if isinstance(system, str) and system:
        msgs.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "".join(b.get("text", "") for b in system if b.get("type") == "text")
        if text:
            msgs.append({"role": "system", "content": text})
    for m in a.get("messages", []):
        if m.get("role") == "system":
            _fold_system(msgs, m.get("content"))
        else:
            msgs.extend(content_blocks_to_openai(m.get("role"), m.get("content")))
    out["messages"] = msgs
    if "max_tokens" in a:
        out["max_tokens"] = a["max_tokens"]
    if "temperature" in a:
        out["temperature"] = a["temperature"]
    if "top_p" in a:
        out["top_p"] = a["top_p"]
    if a.get("stop_sequences"):
        out["stop"] = a["stop_sequences"]
    if a.get("stream"):
        out["stream"] = True
    if a.get("tools"):
        out["tools"] = map_tools(a["tools"])
    if a.get("tool_choice") is not None:
        out["tool_choice"] = map_tool_choice(a["tool_choice"])
    return out


# ---------------- response: OpenAI -> Anthropic (non-streaming) ----------------

_STOP = {"stop": "end_turn", "length": "max_tokens",
         "tool_calls": "tool_use", "function_call": "tool_use"}


def map_stop_reason(fr):
    return _STOP.get(fr, "end_turn")


def to_anthropic_response(o, model):
    choice = (o.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except Exception:
            inp = {}
        content.append({"type": "tool_use", "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:20]), "name": fn.get("name"), "input": inp})
    usage = o.get("usage") or {}
    return {
        "id": o.get("id") or ("msg_" + uuid.uuid4().hex[:24]),
        "type": "message", "role": "assistant", "model": model,
        "content": content,
        "stop_reason": map_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


def count_tokens_estimate(a):
    """Advisory only: ~4 chars/token over the translated prompt text."""
    o = to_openai_request(a)
    chars = 0
    for m in o.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    chars += len(part.get("text", ""))
    return max(1, chars // 4)


# ---------------- errors ----------------

def anthropic_error(status, err_type, message):
    return status, {"type": "error", "error": {"type": err_type, "message": message}}


def error_type_for_status(status):
    if status == 404:
        return "not_found_error"
    if status in (529, 599):
        return "overloaded_error"
    if 400 <= status < 500:
        return "invalid_request_error"
    return "api_error"


# ---------------- streaming: OpenAI SSE -> Anthropic SSE ----------------

def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _iter_openai_chunks(lines):
    """Parse OpenAI 'data: {json}' SSE lines into dicts; stop at [DONE]."""
    for raw in lines:
        line = raw.decode() if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except Exception:
            continue


def stream_anthropic_events(openai_lines, model):
    """Stateful translation of the router's OpenAI chat SSE into the Anthropic
    Messages streaming event sequence. Yields framed SSE byte strings."""
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    started = False
    cur_index = -1
    cur_kind = None          # "text" | "tool" | None
    tool_slots = {}          # openai tool_calls index -> anthropic block index
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"

    def close_block():
        return _sse("content_block_stop", {"type": "content_block_stop", "index": cur_index})

    for chunk in _iter_openai_chunks(openai_lines):
        if not started:
            started = True
            yield _sse("message_start", {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}})
        usage = chunk.get("usage")
        if usage:
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", output_tokens)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        ch = choices[0]
        delta = ch.get("delta") or {}

        text = delta.get("content")
        if text:
            if cur_kind != "text":
                if cur_kind is not None:
                    yield close_block()
                cur_index += 1
                cur_kind = "text"
                yield _sse("content_block_start", {"type": "content_block_start",
                    "index": cur_index, "content_block": {"type": "text", "text": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta",
                "index": cur_index, "delta": {"type": "text_delta", "text": text}})

        for tc in delta.get("tool_calls") or []:
            oi = tc.get("index", 0)
            fn = tc.get("function") or {}
            if oi not in tool_slots:
                if cur_kind is not None:
                    yield close_block()
                cur_index += 1
                cur_kind = "tool"
                tool_slots[oi] = cur_index
                yield _sse("content_block_start", {"type": "content_block_start",
                    "index": cur_index, "content_block": {"type": "tool_use",
                    "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:20]),
                    "name": fn.get("name") or "", "input": {}}})
            args = fn.get("arguments")
            if args:
                yield _sse("content_block_delta", {"type": "content_block_delta",
                    "index": tool_slots[oi], "delta": {"type": "input_json_delta",
                    "partial_json": args}})

        if ch.get("finish_reason"):
            stop_reason = map_stop_reason(ch["finish_reason"])

    if not started:
        yield _sse("message_start", {"type": "message_start", "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}}})
    if cur_kind is not None:
        yield close_block()
    yield _sse("message_delta", {"type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens}})
    yield _sse("message_stop", {"type": "message_stop"})
