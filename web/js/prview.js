// "Try a llama.cpp PR" view helpers (pure, no DOM): the look card, the build's
// progress line, which models can be pinned. Rendering is prtry.js.

export function lookCard(info) {
  const i = info || {};
  let warn = "";
  if (i.state === "merged") warn = "This PR is already merged: the next nightly build has it.";
  else if (i.state === "closed") warn = "This PR was closed without merging.";
  else if (i.draft) warn = "This PR is a draft: expect rough edges.";
  return {
    heading: `PR #${i.number}${i.title ? ": " + i.title : ""}`,
    byline: `by ${i.author || "?"} at ${String(i.sha || "").slice(0, 7)}`,
    warn,
  };
}

const RUNNING = {
  fetch: n => `Fetching PR #${n} from GitHub...`,
  configure: () => "Configuring with your build flags...",
  build: () => "Compiling llama-server and llama-bench. This takes several minutes.",
  install: () => "Installing...",
};

export function phaseLine(st) {
  if (!st) return "";
  const n = st.pr;
  if (st.running) return (RUNNING[st.phase] || RUNNING.fetch)(n);
  if (st.phase === "done") return `Built PR #${n}. Pin a model to it to try it; the router keeps your build.`;
  if (st.phase === "failed") return `PR #${n} didn't build: ${st.error || "unknown error"}. The log below says why.`;
  if (st.phase === "cancelled") return "Cancelled.";
  return "";
}

export function pinChoices(models, inst) {
  const on = new Set((inst && inst.models) || []);
  return (models || [])
    .map(m => m && m.id)
    .filter(id => typeof id === "string" && id && id !== "*")
    .map(id => ({id, pinned: on.has(id)}))
    .sort((a, b) => a.id.localeCompare(b.id));
}
