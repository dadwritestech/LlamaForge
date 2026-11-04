// Stats tab: totals, live throughput, a daily activity chart, per-model usage.
import { $, $$, esc, setHTML, api, toast, fmtNum, fmtDur, fmtAgo, meter, motionOK, askYes } from "./core.js";
import { createFactRotator, normalizeVram } from "./stats-facts.js";
import { bayPlans, bayInputs, cargoHue } from "./bays.js";
import { initBatches, refreshBatches, batchesCached } from "./batches.js";

let statsSort = "tokens", statsRange = 14, statsRequest = 0;
let gpuRequest = null, lastGpuPayload, hasGpuPayload = false;
const seenBoxes = new Map();      // bays.js: when each box was first drawn on this tab
const stowage = () => document.documentElement?.dataset.skin === "stowage";
// Live Throughput matches the bay plan's colours once /api/state has arrived.
const liveHue = () => {
  const sl = stowage() && lastGpuPayload && lastGpuPayload.slots;
  return sl ? cargoHue(sl.footprints, sl.main) : undefined;
};
const tokenFact = createFactRotator();
const SORT_COLS = {tokens:"Total", prompt:"Prompt", generated:"Gen",
                   decode_tps:"Tok/s", runs:"Runs", loaded_secs:"Loaded"};

function setStatsRange(n) { statsRange = n; loadStats(true); }
function sortStats(c) { statsSort = c; loadStats(true); }
async function resetStats() {
  if (!(await askYes("Per-model and daily history will be zeroed. This cannot be undone.",
      {title: "Reset all usage statistics", ok: "Reset", danger: true}))) return;
  await api("/api/stats/reset", {});
  toast("Stats reset", "ok");
  loadStats(true);
}

// The stats view is fully re-rendered on each load, so its controls are wired
// once by delegation on the container rather than per-render.
export function initStats(openRig = () => {}) {
  const view = $("#view-stats");
  if (!view) return;
  view.addEventListener("click", e => {
    const range = e.target.closest("[data-range]");
    if (range) { setStatsRange(+range.dataset.range); return; }
    const sort = e.target.closest("[data-sort]");
    if (sort) { sortStats(sort.dataset.sort); return; }
    if (e.target.closest("[data-statsreset]")) resetStats();
    if (e.target.closest("[data-rigshare]")) openRig();
  });
  initBatches(view);
  // a skin switch redraws from what we have, unless Stowage needs a fuller payload
  document.addEventListener("lf-skin", () => {
    if (!$("#stats-vram") || !hasGpuPayload) return;
    if (stowage() && !(lastGpuPayload && lastGpuPayload.slots)) refreshStatsVram();
    else setHTML($("#stats-vram"), vramView(lastGpuPayload));
  });
}

const FMT = { num: fmtNum, dur: fmtDur, int: n => String(Math.round(n)) };

// `count` (a raw number and a FMT key) lets the first render count up to it.
function statCard(label, val, count) {
  const c = count ? ` data-count="${esc(count[0])}" data-fmt="${esc(count[1])}"` : "";
  return `<div class="gpu"><div class="stats" style="margin:0"><span>${esc(label)}</span></div><div class="statnum"${c} title="${esc(val)}" style="font-family:var(--disp);font-weight:600;color:var(--ink-strong);font-size:22px;margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(val)}</div></div>`;
}

// Odometer on the first paint of the tab only; the 4s refresh replaces these
// nodes, which simply strands any frame still in flight on a detached element.
function countUp(root) {
  if (!motionOK()) return;
  const t0 = performance.now(), ms = 750;
  const els = $$("[data-count]", root).map(el => [el, +el.dataset.count || 0, FMT[el.dataset.fmt] || fmtNum]);
  const step = now => {
    const k = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    for (const [el, to, f] of els) el.textContent = f(to * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// One line per loaded model with its own live speed. live.models comes from a
// backend that scrapes each model (router pool + process slots); an older
// backend only sends the names, so those rows show no numbers.
// hue (Stowage only) gives each LED its model's container colour from bays.js.
export function liveModels(loaded, live, hue = () => "") {
  if (!loaded.length)
    return `<div class="kv"><span class="k">loaded model</span><span class="v">none</span></div>`;
  const by = Object.fromEntries((live.models || []).map(m => [m.id, m]));
  return loaded.map(id => {
    const m = by[id];
    const own = m && m.where === "process"
      ? ` <span class="tag" title="Pinned to another build, so it runs in its own llama-server outside the router">own process</span>` : "";
    const nums = m
      ? `${(m.gen_per_sec||0).toFixed(1)} tok/s <span style="color:var(--dim)">&middot; prompt ${(m.prompt_per_sec||0).toFixed(1)} &middot; ${esc(m.requests_processing)} active</span>`
      : `<span style="color:var(--dim)">-</span>`;
    return `<div class="kv"><span class="k" style="color:var(--ink);display:flex;align-items:center;gap:8px;min-width:0"><span class="led loaded ${hue(id)}" style="flex:0 0 auto"></span><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(id)}</span>${own}</span><span class="v" style="white-space:nowrap">${nums}</span></div>`;
  }).join("");
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

// System RAM beside the GPUs: what's left for offloaded layers and the OS.
// used = total - available, so reclaimable cache doesn't count (osplat.ram_usage).
export function renderStatsRam(payload) {
  const ram = payload && payload.ram;
  const total = Number(ram && ram.total);
  if (!Number.isFinite(total) || total <= 0) return "";
  const used = Math.min(total, Math.max(0, Number(ram.used) || 0));
  const usedGb = (used/1024).toFixed(1), totalGb = (total/1024).toFixed(1);
  return `<div class="stats-vram-card stats-ram-card">
    <div class="stats-vram-head"><span>System RAM</span><span>HOST</span></div>
    <div class="meter" role="progressbar" aria-label="System RAM used" aria-valuemin="0" aria-valuenow="${esc(used)}" aria-valuemax="${esc(total)}" aria-valuetext="${esc(usedGb)} of ${esc(totalGb)} GB used">${meter(used, total)}</div>
    <div class="stats"><span><b>${esc(usedGb)}</b>/${esc(totalGb)} GB</span><span>FREE <b>${esc(((total-used)/1024).toFixed(1))}</b> GB</span></div>
  </div>`;
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

// Stowage draws the same bay plan as the Models tab (without a booking: no row
// is open here), so it needs the footprints in /api/state; the other skins
// only need the GPU counters. /api/state carries those too.
function vramView(payload) {
  return gpuView(payload) + renderStatsRam(payload);
}
function gpuView(payload) {
  if (!stowage()) return renderStatsVram(payload);
  if (!payload || !Array.isArray(payload.gpus)) return `<div class="stats-vram-empty">VRAM TELEMETRY UNAVAILABLE</div>`;
  return bayPlans(payload.gpus, {...bayInputs(payload), seen: seenBoxes});
}

function refreshStatsVram() {
  if (gpuRequest) return;
  gpuRequest = api(stowage() ? "/api/state" : "/api/gpus").catch(() => null).then(payload => {
    lastGpuPayload = payload;
    hasGpuPayload = true;
    setHTML($("#stats-vram"), vramView(payload));
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
  // a multi-model pool has several; an older backend only names one
  const loaded = live.loaded_models || (live.loaded_model ? [live.loaded_model] : []);
  const rows = [...s.per_model].sort((a,b) => (b[statsSort]||0) - (a[statsSort]||0));
  const daily = s.daily.slice(-statsRange);
  const maxDaily = Math.max(1, ...daily.map(d => d.prompt + d.generated));
  const fact = tokenFact(t.generated);
  const oldSummary = $(".token-scale-details summary", v);
  const restoreDetailsFocus = !!oldSummary && document.activeElement === oldSummary;
  const detailsOpen = !!$(".token-scale-details", v)?.open;
  const batchPyOpen = !!$(".batch-py", v)?.open;
  setHTML(v, `
    <div class="stats-body${silent ? "" : " intro"}">
    ${renderTokenScale(t.generated, fact, detailsOpen)}
    <div class="stats-vram" id="stats-vram" aria-label="GPU VRAM and system RAM usage">${hasGpuPayload ? vramView(lastGpuPayload) : `<div class="stats-vram-empty">VRAM TELEMETRY LOADING...</div>`}</div>
    <div class="gpus" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">
      ${statCard("Tokens processed", fmtNum(t.tokens), [t.tokens, "num"])}
      ${statCard("Generated", fmtNum(t.generated), [t.generated, "num"])}
      ${statCard("Inference time", fmtDur(t.loaded_hours*3600), [t.loaded_hours*3600, "dur"])}
      ${statCard("Models used", t.models_used, [t.models_used, "int"])}
      ${statCard("Runs (approx)", fmtNum(t.total_runs), [t.total_runs, "num"])}
      ${statCard("Most used", t.most_used||"-")}
    </div>
    <div class="card"><h3>Live Throughput${live.router_up?"":` <span style="color:var(--red);font-size:12px">(router offline)</span>`}</h3>
      ${liveModels(loaded, live, liveHue())}
      <div class="kv"><span class="k">generation${loaded.length>1?" (all models)":""}</span><span class="v">${(live.gen_per_sec||0).toFixed(1)} tok/s</span></div>
      <div class="kv"><span class="k">prompt eval${loaded.length>1?" (all models)":""}</span><span class="v">${(live.prompt_per_sec||0).toFixed(1)} tok/s</span></div>
      <div class="kv"><span class="k">active requests</span><span class="v">${esc(live.requests_processing)}</span></div>
    </div>
    <div class="card"><h3>Activity${daily.length?` (last ${daily.length} days)`:""}
        <span style="float:right">
          <span class="chip ${statsRange===14?"on":""}" data-range="14">14d</span>
          <span class="chip ${statsRange===30?"on":""}" data-range="30">30d</span>
        </span></h3>
      ${daily.length?`<div style="display:flex;align-items:flex-end;gap:4px;height:120px;margin-top:10px">
        ${daily.map((d,i)=>{const hp=Math.round(100*d.prompt/maxDaily),hg=Math.round(100*d.generated/maxDaily);
          return `<div class="bar" title="${esc(d.date)} &middot; ${fmtNum(d.generated)} generated + ${fmtNum(d.prompt)} prompt" style="--i:${i};flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%">
            <div style="height:${hg}%;min-height:${d.generated?2:0}px;background:var(--amber);box-shadow:0 0 6px var(--amber-dim)"></div>
            <div style="height:${hp}%;min-height:${d.prompt?2:0}px;background:var(--cyan);opacity:.55"></div></div>`;}).join("")}
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;color:var(--dim);font-size:11px">
        <span>${esc(daily[0].date)}</span>
        <span><span style="color:var(--amber)">&#9632;</span> generated &nbsp;<span style="color:var(--cyan)">&#9632;</span> prompt</span>
        <span>${esc(daily[daily.length-1].date)}</span></div>`
      :`<div class="note">No usage recorded yet - load a model and run some inference.</div>`}
    </div>
    <div class="card"><h3>Per-model Usage</h3>
      <div class="note" style="margin:0 0 6px">Usage is scraped from the router's own metrics and totalled per model across all clients. Per-client / per-IP breakdown isn't available: clients hit the llama.cpp router directly, so the dashboard never sees individual request origins.</div>
      ${rows.length?`<div class="toolbar" style="margin:6px 0 0">
        ${Object.keys(SORT_COLS).map(c=>`<span class="chip ${statsSort===c?"on":""}" data-sort="${c}">${SORT_COLS[c]}</span>`).join("")}
      </div>
      <div class="list" style="margin-top:12px">${rows.map(m=>`
        <div class="row"><div class="rhead" style="cursor:default;grid-template-columns:9px 1fr auto auto auto auto auto">
          <span class="led ${loaded.includes(m.id)?"loaded":""}"></span>
          <span class="mid">${esc(m.id)}</span>
          <span class="ctxpill" title="prompt ${fmtNum(m.prompt)} + generated ${fmtNum(m.generated)}">${fmtNum(m.tokens)} tok</span>
          <span class="stat" title="generation speed as llama.cpp timed it, averaged over every token">${m.decode_tps?m.decode_tps+" tok/s":"-"}</span>
          <span class="stat">${fmtNum(m.runs)} runs</span>
          <span class="stat">${fmtDur(m.loaded_secs)}</span>
          <span class="stat">${fmtAgo(m.last_used)}</span>
        </div></div>`).join("")}</div>
      <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:18px">
        <button data-rigshare title="an image of your GPUs, top models and their measured speed, to post">Share my rig&hellip;</button>
        <button class="danger" data-statsreset title="zero all usage statistics">Reset stats&hellip;</button></div>`
      :`<div class="note">No models have logged usage yet.</div>`}
    </div>
    ${batchesCached(batchPyOpen)}</div>`);
  refreshBatches();
  if (!silent) countUp(v);
  if (restoreDetailsFocus) $(".token-scale-details summary", v)?.focus({preventScroll: true});
}
