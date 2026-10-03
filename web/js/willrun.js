// "Will it run?" panel — a standalone estimator over /api/vram/predict.
// Mirrors vramwise's CLI: pick a repo + quant, see placement regime, a tok/s
// estimate, and the per-token time budget. Registered as a tab loader in main.js;
// loadWillRun() re-renders idempotently each time the tab is shown.
import { $, esc, setHTML, api } from "./core.js";

const REGIME = {
  "gpu-resident": ["FITS IN VRAM", "var(--green)"],
  "hybrid":       ["HYBRID (VRAM+RAM)", "var(--amber)"],
  "streaming":    ["STREAMS FROM DISK", "var(--red)"],
};

export function loadWillRun() {
  setHTML($("#view-willrun"), `
    <div class="card">
      <div class="row" style="gap:8px">
        <input id="wr-repo" placeholder="HuggingFace repo, e.g. unsloth/GLM-4.5-Air-GGUF" style="flex:1">
        <input id="wr-quant" placeholder="quant" value="q4_k_m" style="width:150px">
        <button id="wr-go" class="primary">Estimate</button>
      </div>
      <div id="wr-out" class="note" style="margin-top:10px">Enter a HuggingFace repo to estimate placement and speed on this machine. Numbers are approximate (&plusmn;order-of-magnitude).</div>
    </div>`);
  const go = $("#wr-go");
  if (go) go.onclick = runEstimate;
  const repo = $("#wr-repo");
  if (repo) repo.onkeydown = e => { if (e.key === "Enter") runEstimate(); };
}

async function runEstimate() {
  const repo = $("#wr-repo").value.trim();
  if (!repo) return;
  setHTML($("#wr-out"), `<div class="note">estimating&hellip;</div>`);
  const p = await api("/api/vram/predict", { repo, quant: ($("#wr-quant").value.trim() || "q4_k_m") });
  if (!p || p.error) {
    setHTML($("#wr-out"), `<div class="note" style="color:var(--red)">${esc((p && p.error) || "request failed")}</div>`);
    return;
  }
  if (p.confidence === "unknown" || !p.regime) {
    setHTML($("#wr-out"), `<div class="note">Couldn't estimate: ${esc(p.note || "unknown")}</div>`);
    return;
  }
  const [label, col] = REGIME[p.regime] || ["?", "var(--dim)"];
  const tb = p.time_budget_ms || {};
  setHTML($("#wr-out"), `
    <div class="kv"><span class="k">regime</span><span class="v"><span class="tag" style="color:${col};border-color:${col}">${esc(label)}</span></span></div>
    <div class="kv"><span class="k">speed</span><span class="v">~${esc(String(p.tok_s))} tok/s <span style="color:var(--dim)">(rough estimate)</span> <span class="tag">${esc(p.usability || "")}</span></span></div>
    <div class="kv"><span class="k">time / token</span><span class="v">disk ${esc(String(tb.disk))}ms &middot; weights ${esc(String(tb.weight_read))}ms &middot; compute ${esc(String(tb.compute))}ms</span></div>
    <div class="note" style="margin-top:8px">${esc(p.note || "")} ${p.confidence !== "high" ? "(estimate)" : ""}</div>`);
}
