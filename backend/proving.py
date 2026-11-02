"""Proving ground: your models on tasks a script can mark right or wrong.

The arena is taste (you vote); this is correctness. Each task asks for a
short answer in a fixed form, and a checker compares the reply with the known
answer. Model output is only ever compared, never run. Fifteen tasks is a
smoke test, not a benchmark: it says which of *your* models gets *these* right
on *your* GPUs, and how fast.
"""
import json
import re
import statistics

CATEGORIES = ("extract", "math", "json", "follow", "reason")
CHECKS = ("exact", "exact_case", "number", "json", "lines_set", "words", "csv")

TASKS = [
    {"id": "extract-invoice", "cat": "extract", "check": "exact", "expect": "INV-20931",
     "prompt": "Reply with only the new invoice number from this message:\n"
               "\"Hi team, attached is invoice INV-20931 for March. It replaces INV-20391, which had a typo.\""},
    {"id": "extract-email", "cat": "extract", "check": "exact", "expect": "priya.n@example.org",
     "prompt": "Reply with only the email address of the person who is on call this week:\n"
               "\"Escalations: Priya (priya.n@example.org) is on call this week; Tom (tom.b@example.org) is back Monday.\""},
    {"id": "extract-year", "cat": "extract", "check": "number", "expect": 1937,
     "prompt": "In what year did the bridge open to traffic? Reply with only the year.\n"
               "\"Work on the bridge began in 1927, stopped from 1931 to 1933, and it opened to traffic in 1937.\""},
    {"id": "math-kv", "cat": "math", "check": "number", "expect": 90112,
     "prompt": "A GPU has 16 GB. A model's weights take 9.5 GB and 1 GB must stay free. The KV cache needs "
               "0.25 GB per 4096 tokens of context. What is the largest context, in tokens, that fits? "
               "Reply with only the number."},
    {"id": "math-pct", "cat": "math", "check": "number", "expect": 397.8,
     "prompt": "What is 17% of 2340? Reply with only the number."},
    {"id": "math-time", "cat": "math", "check": "exact", "expect": "12:25",
     "prompt": "A train leaves at 09:47 and the trip takes 2 hours 38 minutes. When does it arrive? "
               "Reply with only the time as HH:MM (24-hour)."},
    {"id": "json-person", "cat": "json", "check": "json", "expect": {"name": "Rin", "gpus": 3},
     "prompt": "Reply with only a JSON object with the keys \"name\" (a string) and \"gpus\" (an integer) "
               "for this sentence: \"Rin's workstation has three GPUs.\""},
    {"id": "json-sort", "cat": "json", "check": "json", "expect": [3, 7, 19, 25, 42],
     "prompt": "Reply with only a JSON array of these numbers sorted from smallest to largest: 42, 7, 19, 3, 25"},
    {"id": "json-prime", "cat": "json", "check": "json", "expect": {"ok": False},
     "prompt": "Is 221 a prime number? Reply with only the JSON object {\"ok\": true} or {\"ok\": false}."},
    {"id": "follow-colors", "cat": "follow", "check": "lines_set", "expect": ["red", "green", "blue"],
     "prompt": "Name the three primary colors of light, one per line, with no numbering and no other text."},
    {"id": "follow-reverse", "cat": "follow", "check": "exact_case", "expect": "AMALL",
     "prompt": "Write the word \"llama\" backwards in uppercase letters. Reply with only that word."},
    {"id": "follow-words", "cat": "follow", "check": "words", "expect": 7,
     "prompt": "Write one sentence of exactly 7 words about the sea. Reply with only the sentence."},
    {"id": "reason-boxes", "cat": "reason", "check": "number", "expect": 7,
     "prompt": "Box A holds 2 more apples than box B. Box C holds twice as many as box A. Together the "
               "three boxes hold 34 apples. How many apples are in box B? Reply with only the number."},
    {"id": "reason-sort", "cat": "reason", "check": "csv", "expect": ["apple", "banana", "cherry", "mango", "pear"],
     "prompt": "Sort these words alphabetically and reply with only the words, comma-separated: "
               "pear, apple, mango, banana, cherry"},
    {"id": "reason-days", "cat": "reason", "check": "exact", "expect": "friday",
     "prompt": "Today is Wednesday. What day of the week is it 100 days from today? Reply with only the day."},
]

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_EDGE = " \t\r\n`*\"'.!"


def _clean(text):
    return _THINK.sub("", text).strip() if isinstance(text, str) else ""


def _first_json(s):
    s = re.sub(r"```(?:json)?", "", s)
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "[{":
            try:
                return True, dec.raw_decode(s, i)[0]
            except ValueError:
                continue
    return False, None


def _canon(v):
    return json.dumps(v, sort_keys=True)


def check(task, text):
    """True when the reply is the task's answer in the asked-for form."""
    s = _clean(text)
    if not s:
        return False
    kind, want = task["check"], task["expect"]
    if kind == "exact":
        return s.strip(_EDGE).lower() == str(want).lower()
    if kind == "exact_case":
        return s.strip(_EDGE) == want
    if kind == "number":
        nums = _NUM.findall(s)
        if not nums:
            return False
        try:
            got = float(nums[-1].replace(",", ""))
        except ValueError:
            return False
        return abs(got - want) <= 1e-6 * max(1.0, abs(want))
    if kind == "json":
        ok, val = _first_json(s)
        return ok and _canon(val) == _canon(want)
    if kind == "lines_set":
        lines = [ln.strip(_EDGE + ",;").lower() for ln in s.splitlines() if ln.strip()]
        return len(lines) == len(want) and set(lines) == set(want)
    if kind == "words":
        return len(re.findall(r"[A-Za-z0-9']+", s)) == want
    if kind == "csv":
        return [w.strip(_EDGE).lower() for w in s.split(",")] == want
    return False


def sample_answer(task):
    """A reply that should pass: lets tests prove every task is answerable."""
    kind, want = task["check"], task["expect"]
    if kind == "number":
        return f"{want:g}"
    if kind == "json":
        return json.dumps(want)
    if kind == "lines_set":
        return "\n".join(want)
    if kind == "words":
        return " ".join(["word"] * want)
    if kind == "csv":
        return ", ".join(want)
    return str(want)


def score(results):
    """One model's results {task_id: {pass, tok_s, cut}} -> totals and per-category passes."""
    cats = {c: [0, 0] for c in CATEGORIES}
    by_id = {t["id"]: t for t in TASKS}
    speeds = []
    passed = total = 0
    for tid, r in (results or {}).items():
        t = by_id.get(tid)
        if not t or not isinstance(r, dict):
            continue
        total += 1
        cats[t["cat"]][1] += 1
        if r.get("pass"):
            passed += 1
            cats[t["cat"]][0] += 1
        if isinstance(r.get("tok_s"), (int, float)) and r["tok_s"] > 0:
            speeds.append(float(r["tok_s"]))
    return {"passed": passed, "total": total, "cats": cats,
            "tok_s": round(statistics.median(speeds), 1) if speeds else None}


def verdict(all_results, model, driver):
    """The morning sentence: how `model` does against your daily `driver`."""
    if model == driver or model not in (all_results or {}) or driver not in all_results:
        return None
    m, d = score(all_results[model]), score(all_results[driver])
    wins = sum(1 for c in CATEGORIES if m["cats"][c][0] > d["cats"][c][0])
    ties = sum(1 for c in CATEGORIES if m["cats"][c][0] == d["cats"][c][0])
    tie = f" ({ties} tie{'' if ties == 1 else 's'})" if ties else ""
    if m["tok_s"] and d["tok_s"]:
        ratio = m["tok_s"] / d["tok_s"]
        speed = (f"{ratio:.1f}x faster" if ratio >= 1.1 else
                 f"{round((1 - ratio) * 100)}% slower" if ratio <= 0.9 else "about as fast")
    else:
        speed = "speed unknown"
    return f"{model} beats {driver} on {wins} of {len(CATEGORIES)} categories{tie}, {speed}."
