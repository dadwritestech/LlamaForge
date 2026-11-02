// Arena recent prompts: the last few prompts, newest first, so a board can be
// built from many rounds without retyping. Kept in localStorage on this
// browser only; labels go through esc() where they are shown.
export const MAX_RECENT = 10;
const KEY = "lf_arena_recent";

/** The list with prompt moved (or added) to the front, deduped and capped. */
export function pushRecent(list, prompt) {
  const p = String(prompt || "").trim();
  const old = (Array.isArray(list) ? list : []).filter(x => typeof x === "string" && x && x !== p);
  return (p ? [p, ...old] : old).slice(0, MAX_RECENT);
}

/** One short line for a picker. */
export function recentLabel(prompt) {
  const s = String(prompt || "").replace(/\s+/g, " ").trim();
  return s.length > 70 ? s.slice(0, 69) + "…" : s;
}

export function loadRecent() {
  try { return pushRecent(JSON.parse(localStorage.getItem(KEY) || "[]"), ""); } catch { return []; }
}

export function saveRecent(prompt) {
  const l = pushRecent(loadRecent(), prompt);
  try { localStorage.setItem(KEY, JSON.stringify(l)); } catch { /* private mode: just don't remember */ }
  return l;
}
