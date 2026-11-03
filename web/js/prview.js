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

// The A/B card: llama-bench pp512 / tg128 on your build vs the PR build.
const fmtTs = r => r ? `${Number(r.ts).toFixed(1)} ± ${Number(r.sd).toFixed(1)}` : "";
const fmtPct = d => d ? `${d.pct > 0 ? "+" : ""}${Number(d.pct).toFixed(1)}%` : "";

export function abView(ab) {
  const a = ab || {};
  const out = {phase: "", heading: "", rows: [], markdown: ""};
  if (a.running) {
    out.phase = `Benchmarking ${a.model} on ${a.phase === "pr" ? `PR #${a.pr}` : "your build"} `
      + "(pp512 and tg128, 3 runs). Your models come back when it's done.";
    return out;
  }
  if (a.phase === "failed") {
    out.phase = `The comparison failed: ${a.error || "unknown error"}.`;
    return out;
  }
  if (a.phase !== "done" || !a.base || !a.pr_result || !a.delta) return out;
  out.heading = `PR #${a.pr} vs your build, ${a.model}`;
  const tests = [["pp", "Prompt (pp512)", "pp512"], ["tg", "Generation (tg128)", "tg128"]];
  out.rows = tests.map(([k, label]) => ({label, base: fmtTs(a.base[k]), pr: fmtTs(a.pr_result[k]),
                                        change: fmtPct(a.delta[k]), noise: !!(a.delta[k] || {}).noise}));
  out.markdown = [`**llama.cpp PR #${a.pr} vs my build**, ${a.model}, llama-bench pp512 / tg128, 3 runs each (tok/s)`,
    "", `| test | my build | PR #${a.pr} | change |`, "|---|---|---|---|",
    ...tests.map(([k, , short], i) => `| ${short} | ${out.rows[i].base} | ${out.rows[i].pr} | `
      + `${out.rows[i].change}${out.rows[i].noise ? " (within noise)" : ""} |`)].join("\n");
  return out;
}
