"""Local Batch API, pure parts (no I/O): OpenAI-shaped ids, the SDK's multipart
upload, input-file checks, output/error lines, file and batch objects, and the
batch status machine. The store and the runner build on these.

Formats follow OpenAI's Batch API so an unmodified SDK works against the panel.
Nothing here records who sent a request: no IPs, no client ids."""
import json
import os
import re
import secrets
import time

ENDPOINTS = ("/v1/chat/completions",)
WINDOW = "24h"
WINDOW_SECS = 24 * 3600
MAX_LINES = 50_000
DONE = ("completed", "failed", "cancelled", "expired")

_PREFIX = {"file": "file-", "batch": "batch_", "batch_req": "batch_req_"}
_MOVES = {
    "validating": ("in_progress", "failed", "cancelling", "expired"),
    "in_progress": ("finalizing", "failed", "cancelling", "expired"),
    "finalizing": ("completed", "failed"),
    "cancelling": ("cancelled",),
}


def new_id(kind):
    return _PREFIX[kind] + secrets.token_hex(12)


def valid_id(s, kind):
    """Ids end up in file names, so only our own shape passes."""
    return isinstance(s, str) and re.fullmatch(re.escape(_PREFIX[kind]) + "[0-9a-f]{8,32}", s) is not None


# ---------------------------------------------------------------- upload
def _params(header):
    """'form-data; name="file"; filename="a.jsonl"' -> {'name': 'file', ...}"""
    out = {}
    for part in header.split(";")[1:]:
        k, _, v = part.strip().partition("=")
        out[k.strip().lower()] = v.strip().strip('"')
    return out


def parse_multipart(ctype, raw):
    """The fields of an SDK upload -> {purpose, filename, content bytes}."""
    fields = multipart_fields(ctype, raw)
    if "file" not in fields:
        raise ValueError("the upload has no 'file' field")
    content, filename = fields["file"]
    purpose = fields.get("purpose", (b"", None))[0].decode("utf-8", "replace").strip()
    name = os.path.basename((filename or "upload.jsonl").replace("\\", "/")) or "upload.jsonl"
    return {"purpose": purpose, "filename": name, "content": content}


def multipart_fields(ctype, raw):
    """A multipart/form-data body -> {name: (value bytes, filename or None)}."""
    boundary = _params(ctype or "").get("boundary")
    if not boundary:
        raise ValueError("multipart upload without a boundary")
    fields = {}
    for chunk in raw.split(b"--" + boundary.encode())[1:]:
        if chunk.startswith(b"--"):
            break
        head, sep, value = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if value.endswith(b"\r\n"):
            value = value[:-2]
        disp = ""
        for h in head.decode("utf-8", "replace").split("\r\n"):
            if h.lower().startswith("content-disposition:"):
                disp = h.split(":", 1)[1]
        p = _params(disp)
        if p.get("name"):
            fields[p["name"]] = (value, p.get("filename"))
    return fields


# ---------------------------------------------------------------- input file
def parse_input(data, endpoint, max_lines=MAX_LINES):
    """Check a batch input file. -> (requests, model). Raises ValueError that
    names the first bad line. One model per batch: one load, no thrashing."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {', '.join(ENDPOINTS)}")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("the file is not UTF-8 text") from None
    reqs, seen, model = [], set(), None
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        if len(reqs) >= max_lines:
            raise ValueError(f"more than {max_lines} lines; split the file")
        try:
            d = json.loads(raw)
        except ValueError:
            raise ValueError(f"line {n}: not JSON") from None
        if not isinstance(d, dict):
            raise ValueError(f"line {n}: not a JSON object")
        cid = d.get("custom_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"line {n}: custom_id must be a non-empty string")
        if cid in seen:
            raise ValueError(f"line {n}: duplicate custom_id '{cid}'")
        if d.get("method", "POST") != "POST":
            raise ValueError(f"line {n}: method must be POST")
        if d.get("url") != endpoint:
            raise ValueError(f"line {n}: url must be {endpoint}")
        body = d.get("body")
        if not isinstance(body, dict):
            raise ValueError(f"line {n}: body must be an object")
        m = body.get("model") or ""
        if model is None:
            model = m
        elif m != model:
            raise ValueError(f"line {n}: model '{m}' differs from '{model}'; "
                             "one model per batch, split the file")
        seen.add(cid)
        reqs.append({"custom_id": cid, "body": {**body, "stream": False}})
    if not reqs:
        raise ValueError("the file has no requests")
    return reqs, model


# ---------------------------------------------------------------- output file
def ok_status(code):
    return 200 <= int(code) < 300


def result_line(custom_id, status, body):
    return {"id": new_id("batch_req"), "custom_id": custom_id,
            "response": {"status_code": status, "request_id": new_id("batch_req"), "body": body},
            "error": None}


def error_line(custom_id, code, message):
    return {"id": new_id("batch_req"), "custom_id": custom_id, "response": None,
            "error": {"code": code, "message": message}}


# ---------------------------------------------------------------- objects
def file_object(meta):
    return {"id": meta["id"], "object": "file", "bytes": meta["bytes"],
            "created_at": meta["created_at"], "filename": meta["filename"],
            "purpose": meta["purpose"], "status": "processed"}


_PUBLIC = ("id", "endpoint", "errors", "input_file_id", "completion_window", "status",
           "output_file_id", "error_file_id", "created_at", "in_progress_at", "expires_at",
           "finalizing_at", "completed_at", "failed_at", "expired_at", "cancelling_at",
           "cancelled_at", "request_counts", "metadata", "model")


def new_batch(input_file_id, endpoint, window, total, model, metadata=None, when="now", now=None):
    if window != WINDOW:
        raise ValueError(f"completion_window must be {WINDOW}")
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {', '.join(ENDPOINTS)}")
    md = metadata or {}
    if (not isinstance(md, dict) or len(md) > 16
            or any(not isinstance(v, str) or len(str(k)) > 64 or len(v) > 512 for k, v in md.items())):
        raise ValueError("metadata: at most 16 string pairs (keys 64, values 512 chars)")
    t = int(now if now is not None else time.time())
    b = {k: None for k in _PUBLIC}
    b.update(id=new_id("batch"), endpoint=endpoint, input_file_id=input_file_id,
             completion_window=window, status="validating", created_at=t,
             expires_at=t + WINDOW_SECS, metadata=md, model=model,
             request_counts={"total": total, "completed": 0, "failed": 0},
             when=when, next_line=0)
    return b


def batch_object(b):
    o = {k: b.get(k) for k in _PUBLIC}
    o["object"] = "batch"
    return o


def move(b, status, now=None):
    if status not in _MOVES.get(b["status"], ()):
        raise ValueError(f"can't go {b['status']} -> {status}")
    b["status"] = status
    b[f"{status}_at"] = int(now if now is not None else time.time())
    return b


def is_done(b):
    return b.get("status") in DONE
