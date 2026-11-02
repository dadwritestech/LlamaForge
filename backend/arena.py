"""Blind local arena: two of your models answer the same prompt, unlabeled and
in random order, you pick the better one, and an Elo board kept in arena.json
on this PC ranks them. LMArena ranks models on everyone's prompts; this ranks
yours, on your prompts, on your GPUs. Pure stdlib.

The routes in routes.py do the I/O (loading models, calling the router); this
module is the parts worth testing on their own: Elo, the board file, the
request body and reading the reply."""
import json, os, threading

import atomicio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARENA_FILE = os.path.join(ROOT, "arena.json")

START = 1000.0
K = 32
MAX_PROMPT = 8000       # characters
MAX_TOKENS = 1024       # per answer: long enough to judge, short enough to wait for
_LOCK = threading.Lock()


def elo(ra, rb, score_a, k=K):
    """New (ra, rb) after one game; score_a is 1 (a won), 0.5 (tie) or 0."""
    expected = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    delta = k * (score_a - expected)
    return ra + delta, rb - delta


def _empty():
    return {"models": {}, "games": 0}


def load(path=None):
    try:
        with open(path or ARENA_FILE, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict) or not isinstance(d.get("models"), dict):
            return _empty()
        d.setdefault("games", 0)
        return d
    except Exception:
        return _empty()


def _row(board, mid):
    return board["models"].setdefault(mid, {"elo": START, "wins": 0, "losses": 0, "ties": 0})


def vote(path, a, b, winner):
    """Record one blind game. winner is "a", "b" or "tie". Returns the board."""
    if not (isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()) or a == b:
        raise ValueError("two different models are needed")
    if winner not in ("a", "b", "tie"):
        raise ValueError('winner must be "a", "b" or "tie"')
    with _LOCK:
        board = load(path)
        ra, rb = _row(board, a), _row(board, b)
        score = {"a": 1.0, "b": 0.0, "tie": 0.5}[winner]
        ra["elo"], rb["elo"] = elo(ra["elo"], rb["elo"], score)
        if winner == "tie":
            ra["ties"] += 1
            rb["ties"] += 1
        else:
            w, l = (ra, rb) if winner == "a" else (rb, ra)
            w["wins"] += 1
            l["losses"] += 1
        board["games"] = int(board.get("games", 0)) + 1
        atomicio.write_json(path or ARENA_FILE, board)
        return board


def reset(path=None):
    with _LOCK:
        p = path or ARENA_FILE
        if os.path.exists(p):
            os.remove(p)


def leaderboard(board):
    rows = []
    for mid, r in (board.get("models") or {}).items():
        games = r.get("wins", 0) + r.get("losses", 0) + r.get("ties", 0)
        rows.append({"model": mid, "elo": int(round(r.get("elo", START))), "games": games,
                     "wins": r.get("wins", 0), "losses": r.get("losses", 0), "ties": r.get("ties", 0)})
    rows.sort(key=lambda r: (-r["elo"], -r["games"], r["model"]))
    return rows


def markdown(board):
    """The board as a postable markdown table."""
    lines = ["My local model arena (blind votes, my prompts, my GPUs)", "",
             "| # | Model | Elo | W-L-T |", "|---|---|---|---|"]
    for i, r in enumerate(leaderboard(board), 1):
        lines.append(f"| {i} | {r['model']} | {r['elo']} | {r['wins']}-{r['losses']}-{r['ties']} |")
    lines += ["", f"{board.get('games', 0)} games. Made with LlamaForge: "
              "https://github.com/dadwritestech/LlamaForge"]
    return "\n".join(lines)


def request_body(model, prompt):
    """The router body for one arena answer. Same prompt, same settings for both
    sides: only the model differs."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("type a prompt first")
    if len(prompt) > MAX_PROMPT:
        raise ValueError(f"keep the prompt under {MAX_PROMPT} characters")
    return {"model": model, "messages": [{"role": "user", "content": prompt.strip()}],
            "max_tokens": MAX_TOKENS, "stream": False}


def reply(res):
    """{text, thinking, tokens, tok_s, cut} from a router chat completion."""
    ch = ((res or {}).get("choices") or [{}])[0] or {}
    msg = ch.get("message") or {}
    t = (res or {}).get("timings") or {}
    tps = t.get("predicted_per_second")
    return {"text": msg.get("content") or "",
            "thinking": msg.get("reasoning_content") or "",
            "tokens": t.get("predicted_n"),
            "tok_s": round(tps, 1) if isinstance(tps, (int, float)) else None,
            "cut": ch.get("finish_reason") == "length"}


def pick_two(models, rng):
    """Two different models at random, for "surprise me" rounds."""
    pool = sorted(set(models))
    if len(pool) < 2:
        raise ValueError("the arena needs at least two chat models")
    a, b = rng.sample(pool, 2)
    return a, b
