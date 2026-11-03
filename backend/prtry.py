"""Try a llama.cpp pull request: build ggml-org/llama.cpp PR #N beside the
user's own build and install it as a pin target.

The build lands in ENGINES_DIR like a prebuilt install (lf-install.json with
channel "pr"), so builds.options / pinned_bin already let a model run on it in
its own process. It never touches llama_src, build_dir or the router's binary:
the work dir is a fresh shallow clone of just that PR's head, and the build is
pinned to the SHA the user was shown (a force-push in between aborts). Pure:
the runner does the I/O.
"""
import re
import time

REPO = "https://github.com/ggml-org/llama.cpp"
API = "https://api.github.com/repos/ggml-org/llama.cpp/pulls/"
CHANNEL = "pr"
TARGETS = ("llama-server", "llama-bench")

_NUM = re.compile(r"#?\s*(\d{1,7})")
# ggerganov/llama.cpp redirects to ggml-org/llama.cpp, so old links still work
_URL = re.compile(r"(?:https?://)?(?:www\.)?github\.com/(?:ggml-org|ggerganov)/llama\.cpp/pull/(\d{1,7})(?:[/?#].*)?",
                  re.IGNORECASE)
_SHA = re.compile(r"[0-9a-f]{40}")


def parse_pr(text):
    """The PR number in "1234", "#1234" or a ggml-org/llama.cpp pull URL.
    ValueError for anything else, including other repos."""
    if not isinstance(text, str):
        raise ValueError("type a PR number or a github.com/ggml-org/llama.cpp/pull/... link")
    s = text.strip()
    m = _URL.fullmatch(s) or _NUM.fullmatch(s)
    if not m or int(m.group(1)) <= 0:
        raise ValueError("type a PR number or a github.com/ggml-org/llama.cpp/pull/... link")
    return int(m.group(1))


def pr_info(d):
    """GitHub's pull JSON -> the fields the panel shows. ValueError when it
    isn't a PR (e.g. {"message": "Not Found"}) or the SHA looks wrong."""
    try:
        n, sha = int(d["number"]), d["head"]["sha"]
    except (KeyError, TypeError, ValueError):
        raise ValueError(d.get("message") if isinstance(d, dict) and d.get("message")
                         else "that isn't a llama.cpp pull request")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise ValueError("GitHub sent an odd commit id for that PR")
    state = "merged" if d.get("merged_at") else ("open" if d.get("state") == "open" else "closed")
    return {"number": n, "title": str(d.get("title") or "").strip()[:200],
            "author": str((d.get("user") or {}).get("login") or ""),
            "sha": sha, "state": state, "draft": bool(d.get("draft")),
            "url": f"{REPO}/pull/{n}"}


def install_name(n, sha):
    return f"pr-{int(n)}-{sha[:7]}"


def commands(n, sha, work, build_dir, flags, jobs=8):
    """argv lists, run in order (never through a shell). check_head() goes
    between the checkout and the configure."""
    cfg = ["cmake", "-B", build_dir, "-S", work, "-DCMAKE_BUILD_TYPE=Release",
           "-DLLAMA_CURL=OFF"]
    cfg += [f"-D{k}={v}" for k, v in (flags or {}).items()]
    return [
        ["git", "init", "-q", work],
        ["git", "-C", work, "fetch", "-q", "--depth", "1", REPO, f"pull/{int(n)}/head"],
        ["git", "-C", work, "checkout", "-q", "--detach", "FETCH_HEAD"],
        cfg,
        ["cmake", "--build", build_dir, "--config", "Release", "--parallel", str(int(jobs)),
         "--target", *TARGETS],
    ]


def check_head(rev_parse_out, sha):
    """'' when the fetched head is the commit the user looked at."""
    got = (rev_parse_out or "").strip()
    return "" if got == sha else (f"the PR changed since you looked (now {got[:7] or '?'}, "
                                  f"was {sha[:7]}); look it up again")


def manifest(info, flags, now=None):
    n = info["number"]
    return {"tag": f"PR #{n}", "label": f"PR #{n}: {info['title']}".rstrip(": "),
            "channel": CHANNEL, "pr": n, "sha": info["sha"], "author": info["author"],
            "url": info["url"], "flags": dict(flags or {}),
            "installed_at": time.time() if now is None else now}


def is_pr(install):
    return (install or {}).get("channel") == CHANNEL
