// URL-hash deep links. Imports nothing (no DOM), so the node tests load it directly.
//   #models                                   -> a tab
//   #discover?repo=owner/name&file=x.gguf     -> Discover opened on that repo, that quant
// The link only shows the quant; downloading still takes a click, so a page that
// links here can't start a multi-GB download on its own.

const SEG = /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/;

function cleanRepo(v) {
  if (!v) return null;
  v = v.trim().replace(/^https?:\/\//, "").replace(/^(huggingface\.co|hf\.co)\//, "");
  const parts = v.split("/");
  return parts.length === 2 && parts.every(p => SEG.test(p) && !/^\.+$/.test(p)) ? v : null;
}

function cleanFile(v) {
  if (!v || !/\.gguf$/i.test(v)) return null;
  const parts = v.split("/");
  return parts.every(p => p && p !== "." && p !== ".." && !/[\\<>"'?#]/.test(p)) ? v : null;
}

/** "#discover?repo=a/b&file=x.gguf" -> {tab, repo, file}; bad values come back null. */
export function parseHash(hash) {
  const h = (hash || "").replace(/^#/, "");
  const q = h.indexOf("?");
  const tab = q < 0 ? h : h.slice(0, q);
  const p = new URLSearchParams(q < 0 ? "" : h.slice(q + 1));
  const repo = cleanRepo(p.get("repo"));
  return {tab, repo, file: repo ? cleanFile(p.get("file")) : null};
}

/** The hash that opens Discover on a repo (and optionally one quant). */
export function discoverLink(repo, file) {
  return `#discover?repo=${repo}` + (file ? `&file=${encodeURIComponent(file)}` : "");
}
