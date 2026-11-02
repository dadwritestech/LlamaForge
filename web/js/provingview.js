// Proving ground view helpers (pure, no DOM): the board rows, the progress
// line and a model's misses, from GET /api/proving. Rendering is proving.js.
export const CAT_LABELS = {extract: "Extract", math: "Math", json: "JSON", follow: "Follow", reason: "Reason"};

export function boardRows(d) {
  const scores = (d && d.scores) || {};
  const cats = (d && d.categories) || Object.keys(CAT_LABELS);
  const rows = [];
  for (const [model, s] of Object.entries(scores)) {
    if (!s || typeof s !== "object") continue;
    rows.push({
      model, passed: s.passed || 0, total: s.total || 0,
      tok_s: typeof s.tok_s === "number" ? s.tok_s : null,
      cats: cats.map(c => (s.cats && Array.isArray(s.cats[c])) ? s.cats[c] : [0, 0]),
      driver: model === d.driver,
      verdict: (d.verdicts && d.verdicts[model]) || "",
    });
  }
  return rows.sort((a, b) => b.passed - a.passed || (b.tok_s ?? -1) - (a.tok_s ?? -1) || a.model.localeCompare(b.model));
}

export function progressLine(st) {
  if (!st) return "";
  if (st.running) return `${st.model ? `Checking ${st.model}` : "Starting"}: ${st.done || 0} of ${st.total || 0} answers`;
  return st.cancelled ? "Cancelled. Models that finished were kept." : "";
}

export function failures(d, model) {
  const res = d && d.results && d.results[model];
  if (!res) return [];
  return ((d && d.tasks) || [])
    .filter(t => res[t.id] && !res[t.id].pass)
    .map(t => ({id: t.id, cat: t.cat, answer: res[t.id].answer || "", cut: !!res[t.id].cut}));
}
