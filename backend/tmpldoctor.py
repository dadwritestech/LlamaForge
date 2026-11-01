"""Template doctor: is this GGUF's embedded chat template the one its makers
ship today? A stale template is the usual cause of broken tool calls and
thinking that won't switch off.

Two references, both fetched on demand:
  repo  - the same file in the repo it was downloaded from. Only the GGUF
          header is streamed (a Range request, a few MB), never the weights.
          Different = the quantizer re-uploaded with a fixed template.
  base  - the base model's chat_template.jinja / tokenizer_config.json, from
          the GGUF repo's card. Different is a hint, not a fault: quantizers
          sometimes fix templates on purpose.
The fix is llama.cpp's --chat-template-file pointing at a saved copy; the GGUF
is never edited. Pure stdlib."""
import difflib
import json
import os
import re
import urllib.parse
import urllib.request

import gguf

HF = "https://huggingface.co"
UA = {"User-Agent": "LlamaForge/1.0 (+local model manager)"}
CAP = 64 * 1024 * 1024        # most header bytes streamed; vocab+merges are a few MB
DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat-templates")
_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
LAST = {}                     # normcase(path) -> {"repo": text, "base": text} from the last check


class Forward:
    """A forward-only reader over a stream: seek(n, 1) reads and drops n bytes.
    Reading past `cap` raises EOFError, so a wrong guess never pulls a 20 GB file."""

    def __init__(self, raw, cap=CAP):
        self.raw, self.cap, self.pos = raw, cap, 0

    def read(self, n):
        if self.pos + n > self.cap:
            raise EOFError("GGUF header is larger than the read cap")
        out = b""
        while len(out) < n:
            b = self.raw.read(min(n - len(out), 1 << 20))
            if not b:
                break
            out += b
        self.pos += len(out)
        return out

    def seek(self, n, whence=0):
        if whence != 1 or n < 0:
            raise OSError("forward-only stream")
        while n > 0:
            got = len(self.read(min(n, 1 << 20)))
            if not got:
                raise EOFError
            n -= got


def repo_path(path):
    """('owner/repo', 'path/in/repo.gguf') for a file in a LlamaForge download
    folder (owner--repo/...) or the HF cache (models--owner--repo/snapshots/sha/...)."""
    parts = re.split(r"[\\/]+", path or "")
    for i in range(len(parts) - 1, -1, -1):
        d = parts[i]
        if d.startswith("models--") and d.count("--") >= 2 and len(parts) > i + 3 and parts[i + 1] == "snapshots":
            owner, _, name = d[len("models--"):].partition("--")
            repo, rel = f"{owner}/{name}", "/".join(parts[i + 3:])
            return (repo, rel) if _REPO_RE.match(repo) else ("", "")
    for i in range(len(parts) - 2, -1, -1):
        owner, sep, name = parts[i].partition("--")
        repo = f"{owner}/{name}"
        if sep and _REPO_RE.match(repo):
            return repo, "/".join(parts[i + 1:])
    return "", ""


def _open(url, headers=None, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, headers=dict(UA, **(headers or {}))),
                                  timeout=timeout)


def _text(url):
    with _open(url) as r:
        return r.read(4 * 1024 * 1024).decode("utf-8", "replace")


def _json(url):
    return json.loads(_text(url))


def _url(repo, rel):
    return f"{HF}/{repo}/resolve/main/" + urllib.parse.quote(rel)


def remote_template(repo, rel):
    """The chat template in the repo's current copy of `rel`, header only."""
    raw = _open(_url(repo, rel), headers={"Range": f"bytes=0-{CAP - 1}"})
    try:
        return gguf.chat_template_of(Forward(raw))
    finally:
        close = getattr(raw, "close", None)
        if close:
            close()


def base_repo(repo):
    """The base model a GGUF repo's card names ('' when none or offline)."""
    try:
        info = _json(f"{HF}/api/models/{repo}")
    except Exception:
        return ""
    b = (info.get("cardData") or {}).get("base_model")
    if isinstance(b, list):
        b = b[0] if b else ""
    return b if isinstance(b, str) and _REPO_RE.match(b) and b != repo else ""


def base_template(repo):
    """The base model's default chat template: chat_template.jinja (newer
    transformers), else tokenizer_config.json's chat_template. None if neither."""
    try:
        return _text(_url(repo, "chat_template.jinja"))
    except Exception:
        pass
    try:
        t = _json(_url(repo, "tokenizer_config.json")).get("chat_template")
    except Exception:
        return None
    if isinstance(t, list):
        t = next((x.get("template") for x in t if isinstance(x, dict) and x.get("name") == "default"), None)
    return t if isinstance(t, str) else None


def norm(t):
    return (t or "").replace("\r\n", "\n").strip()


def diff(old, new, limit=80):
    """A short unified diff, mine -> theirs."""
    lines = list(difflib.unified_diff(norm(old).split("\n"), norm(new).split("\n"),
                                      "your file", "current", n=1, lineterm=""))[2:]
    return "\n".join(lines[:limit] + ([f"... {len(lines) - limit} more lines"] if len(lines) > limit else []))


def check(path):
    """Compare `path`'s embedded template with the repo's and the base model's.
    verdict: repo_newer | base_differs | same | unknown. Template texts stay
    here (LAST) for apply(); the answer carries only diffs."""
    mine = gguf.chat_template(path)
    repo, rel = repo_path(path)
    out = {"embedded": mine is not None, "repo": repo, "file": rel, "base": "",
           "verdict": "unknown", "repo_diff": "", "base_diff": "", "error": ""}
    texts = {}
    if not repo:
        out["error"] = "not downloaded from a Hugging Face repo LlamaForge can name"
        return out
    try:
        theirs = remote_template(repo, rel)
        if theirs is not None:
            texts["repo"] = theirs
            if norm(theirs) != norm(mine):
                out["repo_diff"] = diff(mine, theirs)
    except Exception as e:
        out["error"] = f"could not read {repo}: {e}"
    out["base"] = base_repo(repo)
    if out["base"]:
        b = base_template(out["base"])
        if b is not None:
            texts["base"] = b
            if norm(b) != norm(mine):
                out["base_diff"] = diff(mine, b)
    if out["repo_diff"]:
        out["verdict"] = "repo_newer"
    elif out["base_diff"]:
        out["verdict"] = "base_differs"
    elif texts:
        out["verdict"] = "same"
    LAST[os.path.normcase(path)] = texts
    return out


def save(mid, path, source):
    """Write the `source` ('repo' or 'base') template from the last check of
    `path` to chat-templates/<model>.jinja; its path. ValueError when there's none."""
    t = LAST.get(os.path.normcase(path), {}).get(source)
    if not t:
        raise ValueError("check the template first")
    os.makedirs(DIR, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", mid).strip("._") or "model"
    dest = os.path.join(DIR, f"{safe}.{source}.jinja")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(t)
    return dest
