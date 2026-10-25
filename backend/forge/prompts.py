"""A deterministic prompt pool for benchmarking: code, prose, editing, summary.

Built from templates x subjects, so it is large, unique and the same on every
machine (results are comparable between users). No prompt is ever repeated
inside one server process - a repeated prompt hits the slot's prefix cache and
the drafter's n-gram memory and reports speeds nobody gets in real use.

Every candidate in a round gets the SAME prompts (a fair comparison); different
rounds get different ones.
"""
import hashlib
import random

_LANGS = ("Python", "TypeScript", "Go", "Rust", "C#", "Java")
_CODE_TASKS = (
    "parses an ISO-8601 duration like 'P3DT4H12M' into total seconds",
    "merges k sorted lists into one sorted list",
    "validates and normalises email addresses",
    "computes the longest common subsequence of two strings",
    "renders a list of rows as an aligned plain-text table",
    "rate-limits calls with a token bucket",
    "finds the strongly connected components of a directed graph",
    "diffs two lists of lines and prints a unified diff",
    "parses a simple INI file into nested maps",
    "implements an LRU cache with O(1) get and put",
    "converts Roman numerals to integers and back",
    "tokenises arithmetic expressions and evaluates them with operator precedence",
)
_CODE_TEMPLATES = (
    "Write a {lang} function that {task}. Include edge cases and unit tests.",
    "Implement code in {lang} that {task}, as a small module with a command-line entry point.",
    "Write {lang} code that {task}, then briefly explain its time and space complexity.",
    "Show an idiomatic {lang} solution that {task}, with clear names and short comments.",
)

_PROSE_TOPICS = (
    "why the sky is blue but sunsets are red", "the four-day work week",
    "how vaccines train the immune system", "the life cycle of a star like the Sun",
    "urban beekeeping", "the invention of the printing press",
    "remote work and city centres", "how tides work", "the economics of streaming music",
    "night trains in Europe", "coral reef bleaching", "learning a second language as an adult",
    "the history of the bicycle", "public libraries in the digital age",
)
_PROSE_TEMPLATES = (
    "Explain {t} to a curious 12-year-old in about 300 words.",
    "Write a balanced overview of the arguments around {t}, with concrete examples.",
    "Write a short story of about 350 words in which {t} matters to the plot.",
    "Draft a clear one-page briefing note for a busy manager on {t}.",
    "Describe the history of {t} for a general audience.",
    "Write a persuasive op-ed of about 400 words about {t}.",
)

_SNIPPETS = (
    ("inventory.py", '''def load(path):
    items = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, qty, price = line.split(",")
        items[name] = (int(qty), float(price))
    return items


def total(items):
    t = 0
    for name in items:
        qty, price = items[name]
        t += qty * price
    return t


def low_stock(items, limit):
    out = []
    for name in items:
        if items[name][0] < limit:
            out.append(name)
    return sorted(out)
'''),
    ("router.js", '''function route(table, path) {
  const parts = path.split("/").filter(Boolean);
  for (const entry of table) {
    const pat = entry.pattern.split("/").filter(Boolean);
    if (pat.length !== parts.length) continue;
    const params = {};
    let ok = true;
    for (let i = 0; i < pat.length; i++) {
      if (pat[i].startsWith(":")) params[pat[i].slice(1)] = parts[i];
      else if (pat[i] !== parts[i]) { ok = false; break; }
    }
    if (ok) return { handler: entry.handler, params };
  }
  return null;
}

function addRoute(table, pattern, handler) {
  table.push({ pattern, handler });
  return table;
}
'''),
    ("queue.go", '''package queue

type Queue struct {
	items []int
}

func (q *Queue) Push(v int) {
	q.items = append(q.items, v)
}

func (q *Queue) Pop() int {
	v := q.items[0]
	q.items = q.items[1:]
	return v
}

func (q *Queue) Len() int {
	return len(q.items)
}

func (q *Queue) Peek() int {
	return q.items[0]
}
'''),
    ("report.py", '''import csv


def read_sales(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def by_region(rows):
    out = {}
    for r in rows:
        out.setdefault(r["region"], 0)
        out[r["region"]] += float(r["amount"])
    return out


def print_report(rows):
    totals = by_region(rows)
    for region in sorted(totals):
        print(region, round(totals[region], 2))
'''),
    ("Temperature.java", '''public class Temperature {
    private double celsius;

    public Temperature(double celsius) {
        this.celsius = celsius;
    }

    public double getCelsius() {
        return celsius;
    }

    public double getFahrenheit() {
        return celsius * 9 / 5 + 32;
    }

    public boolean isFreezing() {
        return celsius <= 0;
    }

    public String describe() {
        if (celsius < 10) return "cold";
        if (celsius < 25) return "mild";
        return "hot";
    }
}
'''),
)
_EDITS = (
    "Add input validation that raises a clear error on bad input. Show the full updated file.",
    "Add docstrings or doc comments to every function and type. Show the full updated file.",
    "Rename identifiers to be more descriptive, changing nothing else. Show the full updated file.",
    "Add logging at the start and end of each function. Show the full updated file.",
    "Fix any edge-case bugs you can find and mark each fix with a short comment. Show the full updated file.",
)

_NAMES = ("Asha", "Bruno", "Chen", "Dana", "Emeka", "Freya", "Goran", "Hana", "Ivan", "Jules",
          "Kofi", "Lena", "Mateo", "Nadia", "Omar", "Priya")
_TEAMS = ("payments", "search", "mobile", "data platform", "support tooling", "billing",
          "onboarding", "infrastructure")
_EVENTS = (
    "{who} reported that the {team} migration finished {n} days {early}.",
    "The {team} team closed {n} tickets this sprint; {who} flagged {m} as needing follow-up.",
    "{who} raised a risk: the {team} vendor contract renews in {n} weeks and pricing may rise {m}%.",
    "Latency for {team} rose to {n} ms at peak; {who} traced it to a cache with a {m}% miss rate.",
    "{who} proposed moving the {team} release from week {n} to week {m} to absorb QA findings.",
    "Hiring for {team}: {n} offers out, {m} accepted, according to {who}.",
    "{who} noted that customer complaints about {team} fell by {m}% after the {n}-step fix.",
    "Budget review: {team} is {m}% over plan, mostly cloud spend; {who} owns a {n}-point savings plan.",
)


def _report(i):
    rnd = random.Random(f"forge-summary-{i}")
    paras = []
    for _ in range(14):
        sents = []
        for _ in range(6):
            sents.append(rnd.choice(_EVENTS).format(
                who=rnd.choice(_NAMES), team=rnd.choice(_TEAMS), n=rnd.randint(2, 40),
                m=rnd.randint(3, 60), early=rnd.choice(("early", "late"))))
        paras.append(" ".join(sents))
    return f"Weekly status report #{i + 1}\n\n" + "\n\n".join(paras)


def pool():
    """{category: [prompt, ...]}, identical on every call and every machine."""
    code = [t.format(lang=lang, task=task)
            for t in _CODE_TEMPLATES for task in _CODE_TASKS for lang in _LANGS]
    prose = [t.format(t=topic) for t in _PROSE_TEMPLATES for topic in _PROSE_TOPICS]
    editing = [f"Here is {name}:\n\n```\n{src}```\n\n{edit}"
               for name, src in _SNIPPETS for edit in _EDITS]
    summary = ["Summarise the following report in five bullet points, then list the "
               "three biggest risks.\n\n" + _report(i) for i in range(24)]
    return {"code": code, "prose": prose, "editing": editing, "summary": summary}


_POOL = None


def select(category, n, round_id):
    """n distinct prompts of a category for one round. Never wraps around."""
    global _POOL
    if _POOL is None:
        _POOL = pool()
    items = _POOL[category]
    if n > len(items):
        raise ValueError(f"only {len(items)} {category} prompts, asked for {n}")
    seed = int(hashlib.sha1(f"{category}:{round_id}".encode()).hexdigest()[:8], 16)
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    return [items[i] for i in order[:n]]
