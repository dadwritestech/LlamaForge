// Stats tab: totals, live throughput, a daily activity chart, per-model usage.
import { $, esc, setHTML, api, toast, fmtNum, fmtDur, fmtAgo, meter } from "./core.js";
import { createFactRotator, normalizeVram } from "./stats-facts.js";

let statsSort = "tokens", statsRange = 14, statsRequest = 0;
let gpuRequest = null, lastGpuPayload, hasGpuPayload = false;
const tokenFact = createFactRotator();
const SORT_COLS = {tokens:"Total", prompt:"Prompt", generated:"Gen",
                   avg_tps:"Tok/s", runs:"Runs", loaded_secs:"Loaded"};

function setStatsRange(n) { statsRange = n; loadStats(true); }
function sortStats(c) { statsSort = c; loadStats(true); }
async function resetStats() {
  if (!confirm("Reset ALL usage statistics? Per-model and daily history will be zeroed. This cannot be undone.")) return;
  await api("/api/stats/reset", {});
  toast("Stats reset", "ok");
  loadStats(true);
}

// The stats view is fully re-rendered on each load, so its controls are wired
// once by delegation on the container rather than per-render.
export function initStats() {
  const view = $("#view-stats");
  if (!view) return;
  view.addEventListener("click", e => {
    const range = e.target.closest("[data-range]");
    if (range) { setStatsRange(+range.dataset.range); return; }
    const sort = e.target.closest("[data-sort]");
    if (sort) { sortStats(sort.dataset.sort); return; }
    if (e.target.closest("[data-statsreset]")) resetStats();
  });
}

function statCard(label, val) {
  return `<div class="gpu"><div class="stats" style="margin:0"><span>${esc(label)}</span></div><div style="font-family:var(--disp);font-weight:600;color:var(--ink-strong);font-size:22px;margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(val)}</div></div>`;
}

export function renderStatsVram(payload) {
  const gpus = normalizeVram(payload);
  if (!gpus.length) return `<div class="stats-vram-empty">VRAM TELEMETRY UNAVAILABLE</div>`;
  return gpus.map(gpu => { const usedGb = (gpu.used/1024).toFixed(1), totalGb = (gpu.total/1024).toFixed(1);
    return `<div class="stats-vram-card">
    <div class="stats-vram-head"><span>${esc(gpu.name)}</span><span>GPU ${esc(gpu.index)}</span></div>
    <div class="meter" role="progressbar" aria-label="${esc(gpu.name)} VRAM used" aria-valuemin="0" aria-valuenow="${esc(gpu.used)}" aria-valuemax="${esc(gpu.total)}" aria-valuetext="${esc(usedGb)} of ${esc(totalGb)} GB used">${meter(gpu.used, gpu.total)}</div>
    <div class="stats"><span><b>${esc(usedGb)}</b>/${esc(totalGb)} GB</span><span>FREE <b>${esc((gpu.free/1024).toFixed(1))}</b> GB</span></div>
  </div>`; }).join("");
}

export function renderTokenScale(generated, fact, open = false) {
  const total = `${fmtNum(generated)} GENERATED`;
  const accessible = `${total} is approximately ${fact.text}. ${fact.assumption}`;
  return `<div class="token-scale">
    <span class="token-scale-label">TOKEN SCALE //</span>
    <span class="token-scale-total">${esc(total)}</span>
    <span class="token-scale-mark" aria-hidden="true">≈</span>
    <span class="token-scale-fact">${esc(fact.text)}</span>
    <details class="token-scale-details"${open ? " open" : ""}>
      <summary aria-label="${esc(accessible)}">EST.</summary>
      <div class="token-scale-note"><b>${esc(total)} ≈ ${esc(fact.text)}</b><span>${esc(fact.assumption)}</span></div>
    </details>
  </div>`;
}

function refreshStatsVram() {
  if (gpuRequest) return;
  gpuRequest = api("/api/gpus").catch(() => null).then(payload => {
    lastGpuPayload = payload;
    hasGpuPayload = true;
    setHTML($("#stats-vram"), renderStatsVram(payload));
  }).finally(() => { gpuRequest = null; });
}

export async function loadStats(silent) {
  const request = ++statsRequest;
  const v = $("#view-stats");
  if (!silent) setHTML(v, `<div class="skel">LOADING STATS...</div>`);
  refreshStatsVram();
  const s = await api("/api/stats").catch(() => null);
  if (request !== statsRequest) return;
  // fetch() doesn't reject on HTTP errors, so a 404/500 arrives as a parsed
  // error body, not an exception - guard on shape, not just the catch.
  if (!s || s.error || !Array.isArray(s.per_model)) {
    if (!silent) setHTML(v, `<div class="skel" style="color:var(--red)">BACKEND UNREACHABLE</div>`);
    return;
  }
  const t = s.totals, live = s.live;
  const rows = [...s.per_model].sort((a,b) => (b[statsSort]||0) - (a[statsSort]||0));
  const daily = s.daily.slice(-statsRange);
  const maxDaily = Math.max(1, ...daily.map(d => d.prompt + d.generated));
  const fact = tokenFact(t.generated);
  const oldSummary = $(".token-scale-details summary", v);
  const restoreDetailsFocus = !!oldSummary && document.activeElement === oldSummary;
  const detailsOpen = !!$(".token-scale-details", v)?.open;
  setHTML(v, `
    ${renderTokenScale(t.generated, fact, detailsOpen)}
    <div class="stats-vram" id="stats-vram" aria-label="GPU VRAM usage">${hasGpuPayload ? renderStatsVram(lastGpuPayload) : `<div class="stats-vram-empty">VRAM TELEMETRY LOADING...</div>`}</div>
    <div class="gpus" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">
      ${statCard("Tokens processed", fmtNum(t.tokens))}
      ${statCard("Generated", fmtNum(t.generated))}
      ${statCard("Inference time", fmtDur(t.loaded_hours*3600))}
      ${statCard("Models used", t.models_used)}
      ${statCard("Runs (approx)", fmtNum(t.total_runs))}
      ${statCard("Most used", t.most_used||"-")}
    </div>
    <div class="card"><h3>Live Throughput${live.router_up?"":` <span style="color:var(--red);font-size:10px">(router offline)</span>`}</h3>
      <div class="kv"><span class="k">loaded model</span><span class="v ${live.loaded_model?"ok":""}">${esc(live.loaded_model||"none")}</span></div>
      <div class="kv"><span class="k">generation</span><span class="v">${(live.gen_per_sec||0).toFixed(1)} tok/s</span></div>
      <div class="kv"><span class="k">prompt eval</span><span class="v">${(live.prompt_per_sec||0).toFixed(1)} tok/s</span></div>
      <div class="kv"><span class="k">active requests</span><span class="v">${esc(live.requests_processing)}</span></div>
    </div>
    <div class="card"><h3>Activity${daily.length?` (last ${daily.length} days)`:""}
        <span style="float:right">
          <span class="chip ${statsRange===14?"on":""}" data-range="14">14d</span>
          <span class="chip ${statsRange===30?"on":""}" data-range="30">30d</span>
        </span></h3>
      ${daily.length?`<div style="display:flex;align-items:flex-end;gap:4px;height:120px;margin-top:10px">
        ${daily.map(d=>{const hp=Math.round(100*d.prompt/maxDaily),hg=Math.round(100*d.generated/maxDaily);
          return `<div title="${esc(d.date)} &middot; ${fmtNum(d.generated)} generated + ${fmtNum(d.prompt)} prompt" style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%">
            <div style="height:${hg}%;min-height:${d.generated?2:0}px;background:var(--amber);box-shadow:0 0 6px var(--amber-dim)"></div>
            <div style="height:${hp}%;min-height:${d.prompt?2:0}px;background:var(--cyan);opacity:.55"></div></div>`;}).join("")}
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;color:var(--dim);font-size:9px">
        <span>${esc(daily[0].date)}</span>
        <span><span style="color:var(--amber)">&#9632;</span> generated &nbsp;<span style="color:var(--cyan)">&#9632;</span> prompt</span>
        <span>${esc(daily[daily.length-1].date)}</span></div>`
      :`<div class="note">No usage recorded yet - load a model and run some inference.</div>`}
    </div>
    <div class="card"><h3>Per-model Usage</h3>
      <div class="note" style="margin:0 0 6px">Usage is scraped from the router's own metrics and totalled per model across all clients. Per-client / per-IP breakdown isn't available: clients hit the llama.cpp router directly, so the dashboard never sees individual request origins.</div>
      ${rows.length?`<div class="toolbar" style="margin:6px 0 0">
        ${Object.keys(SORT_COLS).map(c=>`<span class="chip ${statsSort===c?"on":""}" data-sort="${c}">${SORT_COLS[c]}</span>`).join("")}
        <span class="chip" data-statsreset style="margin-left:auto;color:var(--red);border-color:var(--red)" title="zero all usage statistics">Reset stats</span>
      </div>
      <div class="list" style="margin-top:12px">${rows.map(m=>`
        <div class="row"><div class="rhead" style="cursor:default;grid-template-columns:9px 1fr auto auto auto auto auto">
          <span class="led ${live.loaded_model===m.id?"loaded":""}"></span>
          <span class="mid">${esc(m.id)}</span>
          <span class="ctxpill" title="prompt ${fmtNum(m.prompt)} + generated ${fmtNum(m.generated)}">${fmtNum(m.tokens)} tok</span>
          <span class="stat" title="average generation speed while active">${m.avg_tps?m.avg_tps+" tok/s":"-"}</span>
          <span class="stat">${fmtNum(m.runs)} runs</span>
          <span class="stat">${fmtDur(m.loaded_secs)}</span>
          <span class="stat">${fmtAgo(m.last_used)}</span>
        </div></div>`).join("")}</div>`
      :`<div class="note">No models have logged usage yet.</div>`}
    </div>`);
  if (restoreDetailsFocus) $(".token-scale-details summary", v)?.focus({preventScroll: true});
}
