// The Models tab: GPU telemetry, the model list, the per-model knob editor,
// load/unload, compare, presets, and the router/vLLM log panels.
//
// Rendering contract - the thing to preserve when editing this file:
// renderModels() reconciles rows against the DOM instead of rebuilding it.
// A row whose data is unchanged is left alone, so focus, caret position,
// scroll and half-typed knob values survive the 4-second poll. See syncEditor()
// for how the editor separates what the server owns from what the user is
// typing. Nothing here may re-render a knob input the user might be editing.
import { $, $$, esc, setHTML, api, toast, meter, motionOK, askYes, askText } from "./core.js";
import { S, models as modelRows, config as cfgOf } from "./state.js";
import { on, emit } from "./bus.js";
import { activeTab } from "./ui.js";
import * as slots from "./slots.js";
import * as elastic from "./elastic.js";
import * as autoforge from "./autoforge.js";
import { bayPlans, bayInputs } from "./bays.js";

const LITE_KNOBS = new Set(["n-gpu-layers","ctx-size","cache-type-k","cache-type-v",
  "flash-attn","batch-size","ubatch-size","threads","tensor-split","temp","top-p"]);

/* ---------- view-local state ---------- */
let openId = localStorage.getItem("lf_openid") || null;   // expanded row, persisted
let selId = null;                 // keyboard-selected row
let compareMode = false;
let onlySet = false, kquery = "", mquery = "", favOnly = false;
let vllmSchemaPending = false;
let loadBusy = false;
let knobEpoch = 0;
const cmpSet = new Set();         // model ids picked for compare
const diagCache = {};             // failure diagnosis per model id
const metaCache = {};             // GGUF metadata per model id
const compatCache = {};           // which llama.cpp family can load it (compat.py), per model id
const buildsCache = {};           // {options, pinned}: the builds a model can be pinned to (builds.py)
const loadQ = [];                 // sequential load queue
const favs = new Set(JSON.parse(localStorage.getItem("lf_favs") || "[]"));
const loadingSince = {};

function setOpenId(id) {
  openId = id;
  if (id) localStorage.setItem("lf_openid", id);
  else localStorage.removeItem("lf_openid");
}
function saveFavs() { localStorage.setItem("lf_favs", JSON.stringify([...favs])); }

function toggleFav(id) {
  favs.has(id) ? favs.delete(id) : favs.add(id);
  saveFavs(); renderModels();
}
function filterModels(inp) { mquery = inp.value.trim().toLowerCase(); renderModels(); }
function toggleFavOnly(el) {
  favOnly = !favOnly; el.classList.toggle("on", favOnly); renderModels();
}

/* ---------- GPU telemetry ----------
   Classic and Hearth draw a tile per GPU with a segment meter; Stowage draws
   a bay plan (bays.js) with the open row's planner verdict as a booked box.
   The HTML is cached so an unchanged poll touches nothing. */
let lastGpus = null, lastBays = "";
const seenBoxes = new Map();      // bays.js: when each box was first drawn here

export function renderGpus(g) { lastGpus = g; drawGpus(); }
function drawGpus() {
  const host = $("#gpus"), g = lastGpus;
  if (!host || !g) return;
  const html = document.documentElement.dataset.skin === "stowage"
    ? bayPlans(g, {...bayInputs(S.STATE), book: slots.booked(openId), bookId: openId, seen: seenBoxes})
    : gpuTiles(g);
  if (html === lastBays) return;
  lastBays = html;
  setHTML(host, html);
}
document.addEventListener("lf-skin", drawGpus);

function gpuTiles(g) {
  if (!g.length || g[0].error) return `<div class="gpu"><div class="stats">GPU telemetry unavailable</div></div>`;
  return g.map(x => `<div class="gpu"><div class="top"><span class="name">${esc(x.name)}</span><span class="idx">CUDA${esc(x.index)}</span></div>
    <div class="meter">${meter(x.used,x.total)}</div>
    <div class="stats"><span><b>${esc((x.used/1024).toFixed(1))}</b>/${esc((x.total/1024).toFixed(1))} GB</span><span>FREE <b>${esc(((x.total-x.used)/1024).toFixed(1))}</b> GB</span><span>UTIL <b>${esc(x.util)}%</b></span><span>TEMP <b>${esc(x.temp)}&deg;C</b></span></div></div>`).join("");
}

/* ---------- knob fields ---------- */
function curVal(m, knob) {
  for (const a of knob.aliases) if (m.settings[a] != null) return m.settings[a];
  return "";
}
function knobField(m, k) {
  const v = curVal(m, k), isSet = v !== "";
  const ph = k.default ? `inherit (${k.default})` : "inherit";
  const id = "k-" + esc(k.key);  // unique id for label/control association
  let ctrl;
  if (k.type === "enum") {
    const opts = [""].concat(k.options||[]).map(o => `<option value="${esc(o)}" ${String(o)===String(v)?"selected":""}>${o===""?"(inherit)":esc(o)}</option>`).join("");
    ctrl = `<select id="${id}" data-k="${esc(k.key)}">${opts}</select>`;
  } else if (k.type === "bool") {
    const opts = ["","true","false"].map(o => `<option value="${o}" ${String(o)===String(v)?"selected":""}>${o===""?"(inherit)":o}</option>`).join("");
    ctrl = `<select id="${id}" data-k="${esc(k.key)}">${opts}</select>`;
  } else {
    ctrl = `<input id="${id}" data-k="${esc(k.key)}" value="${esc(v)}" placeholder="${esc(ph)}" ${(k.type==="int"||k.type==="float")?'inputmode="numeric"':''}>`;
  }
  return `<div class="fld ${isSet?"set":""}${LITE_KNOBS.has(k.key)?"":" advanced-only"}" data-desc="${esc((k.key+' '+k.desc).toLowerCase())}">
    <label for="${id}" title="${esc(k.desc)}">${esc(k.key)}</label>${ctrl}
    ${k.desc?`<div class="hint" title="${esc(k.desc)}">${esc(k.desc)}</div>`:""}</div>`;
}
function modelMeta(m) {
  const mp = m.settings && m.settings.model;
  if (!mp && !m.file_gib) return "";
  return `<div class="note" style="margin-bottom:10px">
    ${mp?`<span class="tag ep" data-copy="${esc(mp)}" title="click to copy the file path">copy path</span> ${esc(mp)}`:""}
    ${m.file_gib?`<span style="color:var(--cyan)"> &middot; ${esc(m.file_gib)} GiB on disk</span>`:""}</div>`;
}

/* ---------- editor regions ----------------------------------------------
   .ed-live    diagnosis / GGUF card / path / presets   - server state
   .ed-tools   knob filter box                          - user state
   .ed-knobs   the knob fields themselves               - user state
   .ed-btns    action buttons (depend on load status)   - server state
   [data-msg]  status line                              - written by handlers

   Only .ed-live, .ed-btns and .ed-note are refreshed on a poll. The knob grid
   is built when the row opens and then left alone; rebuilding it discards
   whatever the user has typed, so it happens only via invalidateKnobs(). */
function knobGroups(m, schema) {
  return schema.groups.map((g, gi) => {
    const flds = g.knobs.map(k => knobField(m, k)).join("");
    return `<details class="kgroup" ${gi===0?"open":""}><summary>${esc(g.name)} &middot; ${g.knobs.length}</summary><div class="kgrid">${flds}</div></details>`;
  }).join("");
}
function editorLive(m) {
  return m.backend === "vllm"
    ? `${diagBlock(m)}${modelMeta(m)}`
    : `${diagBlock(m)}${metaBlock(m)}${modelMeta(m)}${buildBar(m)}${thinkBar(m)}${visionBar(m)}${presetBar(m)}${autoTuneBar(m)}`;
}
function editorButtons(m) {
  if (m.backend === "vllm") {
    return `<button class="primary" data-act="vsave">Save${m.status==="loaded"?" + Restart":""}</button>
      ${m.status==="loaded"||m.status==="loading"?`<button class="ghost" data-act="vunload">${m.status==="loading"?"Cancel / Stop":"Stop"}</button>`:`<button data-act="vload">Load</button>`}
      <button class="ghost" data-act="client">Client config</button>
      <button class="ghost" data-act="profile" title="save a one-click launch for this model">Save as profile</button>
      <button class="danger" data-act="vdelete" title="remove model + delete its files from WSL">Delete</button>`;
  }
  return `<button class="primary" data-act="save">Save + Reload</button>
      ${m.status==="loaded"||m.status==="loading"?`<button class="ghost" data-act="unload">${m.status==="loading"?"Cancel / Unload":"Unload"}</button>`:`<button data-act="load"${slots.slotsOn()?' title="load as the main model: it gets the fastest GPU that fits"':""}>Load</button>`}
      ${slots.canLoadWorker(m)?`<button class="ghost" data-act="loadw" title="load beside the main model, on the GPUs it doesn't use; never unloads anything">Load as worker</button>`:""}
      ${m.status==="loaded"?`<button data-act="chat">Chat</button>`:""}
      <button class="ghost" data-act="client">Client config</button>
      <button class="ghost" data-act="profile" title="save a one-click launch: model + preset + engine build">Save as profile</button>
      <button class="danger" data-act="unregister" title="remove from models.ini; does not delete the GGUF">Unregister</button>`;
}
function editorNote(m) {
  if (m.backend === "vllm")
    return `<div class="note">vLLM runs one model at a time inside WSL. Saving knobs on a loaded model restarts it (vLLM has no hot reload). Startup can take 1&ndash;5 minutes; watch the vLLM Log panel below.</div>`;
  return m.status === "loading"
    ? `<div class="note">Still loading? Check the Router Log panel below the model list for the real llama.cpp output (crashes, out-of-memory, etc. show up there).</div>` : "";
}
// A plain message (no knob grid) when the editor can't be built.
function editorBlocked(m) {
  if (m.backend === "vllm") {
    if (!S.VLLM_SCHEMA) {
      if (!vllmSchemaPending) {
        vllmSchemaPending = true;
        api("/api/vllm/schema").then(s => {
          S.VLLM_SCHEMA = s; vllmSchemaPending = false; invalidateKnobs(); renderModels();
        });
      }
      return `<div class="note">Loading vLLM knob schema...</div>`;
    }
    if (S.VLLM_SCHEMA.error) return `<div class="note" style="color:var(--red)">vLLM knobs unavailable: ${esc(S.VLLM_SCHEMA.error)} &mdash; install vLLM from the Setup tab.</div>`;
    return null;
  }
  if (!m.in_ini) return `<div class="note">Auto-discovered (not in models.ini) &mdash; add it via Setup &rarr; Scan Drives to tune it here.</div>`;
  if (!S.SCHEMA) return `<div class="note">Loading knob schema...</div>`;
  if (S.SCHEMA.error) return `<div class="note" style="color:var(--red)">Could not read knobs from <code>llama-server --help</code>: ${esc(S.SCHEMA.error)}<br>Check <code>server_bin</code> in config.json - the schema is retried automatically once it's fixed.</div>`;
  if (!S.SCHEMA.groups || !S.SCHEMA.groups.length) return `<div class="note">llama-server --help returned no tunable arguments.</div>`;
  return null;
}
function editor(m) {
  const blocked = editorBlocked(m);
  if (blocked !== null) return blocked;
  const schema = m.backend === "vllm" ? S.VLLM_SCHEMA : S.SCHEMA;
  const placeholder = m.backend === "vllm"
    ? "filter knobs (e.g. tensor, memory, quant)..."
    : "filter knobs (e.g. cache, rope, temp)...";
  return `<div class="ed-slot">${slots.block(m, renderModels)}</div>
    <div class="ed-live">${editorLive(m)}</div>
    <div class="toolbar ed-tools">
      <input class="search" data-knobfilter placeholder="${esc(placeholder)}">
      <button type="button" class="chip ${onlySet?"on":""}" data-onlyset>Only set</button>
    </div>
    <div class="ed-knobs">${knobGroups(m,schema)}</div>
    <div class="actions">
      <span class="ed-btns">${editorButtons(m)}</span>
      <span class="msg" data-msg></span>
    </div>
    <div class="ed-note">${editorNote(m)}</div>
    ${forgeable(m) ? `<div class="ed-forge">${autoforge.markup(m.id)}</div>` : ""}`;
}
// Auto-Forge drives llama-server with the model's models.ini section, so only those
const forgeable = m => (m.backend || "llamacpp") === "llamacpp" && m.in_ini;
function applyKnobFilter(root) {
  $$(".fld", root).forEach(f => {
    const hitQ = !kquery || f.dataset.desc.includes(kquery);
    const hitSet = !onlySet || f.classList.contains("set");
    f.style.display = (hitQ && hitSet) ? "" : "none";
  });
}
function filterKnobs(inp) {
  kquery = inp.value.trim().toLowerCase();
  applyKnobFilter(inp.closest(".edit"));
}
function toggleOnlySet(el) {
  onlySet = !onlySet;
  el.classList.toggle("on", onlySet);
  applyKnobFilter(el.closest(".edit"));
}
function invalidateKnobs() { knobEpoch++; }

/* ---------- the list ---------- */
function loadingSecs(m) {
  if (m.status !== "loading") { delete loadingSince[m.id]; return 0; }
  if (!loadingSince[m.id]) loadingSince[m.id] = Date.now();
  return Math.round((Date.now() - loadingSince[m.id]) / 1000);
}
function shownModels() {
  return modelRows()
    .filter(m => (!mquery || m.id.toLowerCase().includes(mquery)) && (!favOnly || favs.has(m.id)))
    .sort((a, b) => (favs.has(b.id)?1:0) - (favs.has(a.id)?1:0));
}
const BACKEND_LABEL = { vllm: "vLLM", ikllama: "ik_llama", llamacpp: "llama.cpp" };

// The engine tag only carries information when more than one engine is serving
// models. On a llama.cpp-only install it was 15 identical LLAMA.CPP tags - pure
// noise competing with the id for attention.
function backendTagNeeded() {
  return new Set(modelRows().map(m => m.backend || "llamacpp")).size > 1;
}

function rowHead(m, showBackend) {
  const vis = m.modalities.includes("image"), loaded = m.status === "loaded", isFav = favs.has(m.id);
  const stuckSecs = loadingSecs(m);
  const be = m.backend || "llamacpp";
  const beTag = showBackend
    ? `<span class="tag be-${esc(be)}">${esc(BACKEND_LABEL[be] || be)}</span>` : "";
  return `${compareMode?`<input type="checkbox" class="cmp" data-cmp="${esc(m.id)}" ${cmpSet.has(m.id)?"checked":""} title="pick to compare">`:""}
        <span class="led ${loaded?"loaded":""} ${m.status==="loading"?"loading":""} ${m.failed?"failed":""}"></span>
        <button type="button" class="fav ${isFav?"on":""}" data-fav="${esc(m.id)}" aria-label="${isFav?"unfavorite":"favorite"}" aria-pressed="${isFav}">&starf;</button>
        <span class="mid" title="${esc(m.id)}">${esc(m.id)}${beTag}${slots.chip(m)}${elastic.chip(S.STATE, m)}${buildTag(m)}${vis?'<span class="tag vis">vision</span>':''}${!m.in_ini?'<span class="tag">auto</span>':''}${m.endpoint?`<span class="tag ep" data-ep="${esc(m.endpoint)}" title="click to copy endpoint">${esc(m.endpoint.replace('http://',''))}</span>`:''}</span>
        <span class="ctxpill"><span class="k">CTX</span> ${esc(m.eff_ctx)}</span>
        ${statCell(m, loaded, stuckSecs)}
        <span class="qbtns">${quickBtn(m)}</span>
        <span class="chev">&#9654;</span>`;
}
// A loading row shows the router's real progress when it streams it (/models/sse);
// older routers send none, so the stopwatch ("check log" after 20 s) stays.
const STAGE_LABEL = { text_model: "weights", spec_model: "draft", mmproj_model: "vision" };
function loadPct(m) {
  const p = m.status === "loading" && !m.failed && m.load_progress;
  return p && Number.isFinite(p.pct) ? Math.max(0, Math.min(100, p.pct)) : null;
}
function statCell(m, loaded, stuckSecs) {
  const pct = loadPct(m);
  if (pct !== null) {
    const stage = STAGE_LABEL[m.load_progress.stage] || "model";
    return `<span class="stat" title="loading ${esc(stage)}: ${pct}%">loading ${pct}%`
      + `<span class="lbar" role="progressbar" aria-label="load progress" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><i style="width:${pct}%"></i></span></span>`;
  }
  return `<span class="stat ${loaded?"loaded":""}" style="${stuckSecs>=20?"color:var(--red)":""}">${m.failed?"FAILED":esc(m.status)}${stuckSecs>=20?` (${stuckSecs}s, check log)`:""}</span>`;
}
// Everything rowHead() reads. Compared as a string so an unchanged row is left
// in the DOM untouched - which is what keeps focus, selection and scroll alive.
function headSig(m, cols, showBackend) {
  return JSON.stringify([m.id, m.status, m.failed, m.backend, m.endpoint, m.build, m.eff_ctx,
    m.modalities, m.in_ini, favs.has(m.id), compareMode, cmpSet.has(m.id),
    loadQ.findIndex(j => j.id === m.id), loadingSecs(m) >= 20, loadPct(m), cols, showBackend,
    slots.chipSig(m), elastic.chipSig(S.STATE, m)]);
}
// Keyed so only a different model, backend or schema rebuilds the knob grid.
function knobSig(m) {
  return JSON.stringify([m.id, m.backend, m.in_ini,
    m.backend === "vllm" ? (S.VLLM_SCHEMA ? S.VLLM_SCHEMA.count||0 : -1)
                         : (S.SCHEMA ? S.SCHEMA.count||0 : -1), knobEpoch]);
}

function rowIdentity(backend, id) {
  return JSON.stringify([backend || "llamacpp", id]);
}

export function renderModels() {
  if (!S.STATE) return;
  const all = modelRows();
  const ms = shownModels();
  const nLoaded = all.filter(m => m.status === "loaded").length;
  const count = $("#count");
  if (count) count.textContent = `${nLoaded} LOADED / ${all.length} TOTAL` +
    (ms.length !== all.length ? ` · ${ms.length} shown` : "");
  document.title = nLoaded ? `▸${nLoaded} LLAMAFORGE` : "LLAMAFORGE";
  slots.renderBanner();
  drawGpus();                       // the bay plan shows the open row's booking
  const cols = compareMode ? "16px 14px 18px 1fr auto auto auto auto"
                           : "14px 18px 1fr auto auto auto auto";
  const showBackend = backendTagNeeded();
  const list = $("#list");
  if (!list) return;
  if (!ms.length) { setHTML(list, `<div class="skel empty">NO MODELS MATCH</div>`); return; }
  if (list.firstElementChild && list.firstElementChild.classList.contains("skel")) setHTML(list, "");

  const existing = new Map($$(".row", list).map(r => [
    rowIdentity(r.dataset.backend, r.dataset.id), r,
  ]));
  let prev = null;
  for (const m of ms) {
    const backend = m.backend || "llamacpp";
    const identity = rowIdentity(backend, m.id);
    let row = existing.get(identity);
    if (!row) {
      row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<div class="rhead"></div><div class="edit"></div>`;
    } else existing.delete(identity);
    row.dataset.id = m.id;
    row.dataset.backend = backend;

    const hs = headSig(m, cols, showBackend);
    if (row._hs !== hs) {
      const head = row.firstElementChild;
      head.style.gridTemplateColumns = cols;
      setHTML(head, rowHead(m, showBackend));
      row._hs = hs;
    }
    // a model that just finished loading lights its row once (Hearth styles .ignite)
    if (row._st && row._st !== "loaded" && m.status === "loaded") ignite(row);
    row._st = m.status;
    row.classList.toggle("open", m.id === openId);
    row.classList.toggle("sel", m.id === selId);
    syncEditor(row, m);

    // place the row without moving one already in the right slot
    const want = prev ? prev.nextElementSibling : list.firstElementChild;
    if (want !== row) list.insertBefore(row, want);
    prev = row;
  }
  for (const stale of existing.values()) stale.remove();
}
function ignite(row) {
  if (!motionOK()) return;
  row.classList.remove("ignite");
  void row.offsetWidth;              // restart if it is still glowing from a moment ago
  row.classList.add("ignite");
  row.addEventListener("animationend", function done(e) {
    if (e.animationName !== "hearth-ignite") return;
    row.classList.remove("ignite");
    row.removeEventListener("animationend", done);
  });
}
/* Bring one row's editor in line with the model, preserving user input. */
function syncEditor(row, m) {
  const edit = row.lastElementChild;
  if (m.id !== openId) {
    if (edit.firstChild) { setHTML(edit, ""); row._ks = null; }
    return;
  }
  const ks = knobSig(m);
  if (row._ks !== ks) {          // first open, or the schema/model actually changed
    setHTML(edit, editor(m));
    row._ks = ks;
    row._slot = row._live = row._btns = row._note = row._forge = null;
  }
  const regions = [[".ed-slot", x => slots.block(x, renderModels), "_slot"],
                   [".ed-live", editorLive, "_live"],
                   [".ed-btns", editorButtons, "_btns"],
                   [".ed-note", editorNote, "_note"],
                   [".ed-forge", x => autoforge.markup(x.id), "_forge"]];
  for (const [sel, build, key] of regions) {
    const el = $(sel, edit);
    if (!el) continue;
    const sig = build(m);
    if (row[key] !== sig) { setHTML(el, sig); row[key] = sig; }
  }
}

/* ---------- quick-load + sequential queue ---------- */
function quickBtn(m) {
  const q = loadQ.findIndex(j => j.id === m.id);
  if (q >= 0) return `<span class="qbadge">QUEUED #${q+1}</span>`;
  if (m.status === "loading") return `<button class="qbtn stop" data-quick="stop" data-qid="${esc(m.id)}">Cancel</button>`;
  if (m.status === "loaded") return `${(m.backend || "llamacpp") === "llamacpp"
    ? `<button class="qbtn load" data-quick="chat" data-qid="${esc(m.id)}">Chat</button>` : ""}<button class="qbtn stop" data-quick="unload" data-qid="${esc(m.id)}">Unload</button>`;
  return `<button class="qbtn load" data-quick="load" data-qid="${esc(m.id)}">Load</button>`;
}
function beOf(id) {
  const m = modelRows().find(x => x.id === id);
  return m ? (m.backend || "llamacpp") : "llamacpp";
}
function enqueueLoad(id) {
  if (loadQ.some(j => j.id === id) || (loadBusy && loadQ[0] && loadQ[0].id === id)) return;
  delete diagCache[id];               // a retry should re-diagnose, not show stale error
  slots.forget(id);
  loadQ.push({id});
  toast(loadBusy ? `Queued #${loadQ.length}` : "Loading...", "ok");
  renderModels(); processQ();
}
async function processQ() {
  if (loadBusy || !loadQ.length) return;
  loadBusy = true;
  const job = loadQ[0];
  let r = null;
  try {
    r = beOf(job.id) === "vllm" ? await api("/api/vllm/load", {model: job.id})
                                : await slots.load(job.id);
  } catch (e) {}
  // a planner refusal is an answer, not a crash: say it (the open row shows the details)
  if (r && r.success === false && typeof r.reason === "string")
    toast(`${job.id}: ${slots.errText(r)}`.slice(0, 220), "err");
  loadQ.shift(); loadBusy = false;
  await refresh(true);
  if (chatAfterLoad.delete(job.id)) {
    const m = modelRows().find(x => x.id === job.id);
    if (m && m.status === "loaded") {
      if ((m.backend || "llamacpp") === "llamacpp") emit("chat", job.id);
      else toast(`${job.id} is loaded`, "ok");
    } else {
      toast(`${job.id} did not load - open it in My Models for the reason`, "err");
    }
  }
  processQ();
}
// Discover's "Load & Chat" after a download: refresh first so a just-registered
// model (and whether it is vLLM) is known, then load it and open Chat.
const chatAfterLoad = new Set();
on("load-chat", async id => {
  await refresh(true);
  chatAfterLoad.add(id);
  const m = modelRows().find(x => x.id === id);
  if (m && m.status === "loaded") { chatAfterLoad.delete(id); if ((m.backend || "llamacpp") === "llamacpp") emit("chat", id); return; }
  enqueueLoad(id);
});
async function quickAction(act, id) {
  const be = beOf(id);
  if (act === "load") { enqueueLoad(id); return; }
  if (act === "chat") { emit("chat", id); return; }
  if (act === "unload") {
    await api(be === "vllm" ? "/api/vllm/unload" : "/api/unload", {model: id});
    toast("Unloaded", "ok"); await refresh(true); return;
  }
  if (act === "stop") {
    const qi = loadQ.findIndex(j => j.id === id);     // still queued -> just drop it
    if (qi > 0) { loadQ.splice(qi, 1); renderModels(); return; }
    await api(be === "vllm" ? "/api/vllm/unload" : "/api/unload", {model: id});
    toast("Cancelled", "ok"); await refresh(true);
  }
}
async function unloadAll() {
  loadQ.length = 0;
  const r = await api("/api/unload_all", {});
  toast(`Unloaded ${(r.unloaded||[]).length}`, "ok");
  await refresh(true);
}

/* ---------- compare ---------- */
function toggleCompare(el) {
  compareMode = !compareMode;
  el.classList.toggle("on", compareMode);
  if (!compareMode) cmpSet.clear();
  updateCmpRun(); renderModels();
}
function updateCmpRun() {
  const run = $("#cmp-run"); if (!run) return;
  const n = $("#cmp-n"); if (n) n.textContent = cmpSet.size;
  run.style.display = (compareMode && cmpSet.size >= 2) ? "" : "none";
}
function openCompare() {
  const ms = [...cmpSet].map(id => modelRows().find(x => x.id === id)).filter(Boolean);
  if (ms.length < 2) { toast("Pick at least 2 models", "err"); return; }
  const keys = [...new Set(ms.flatMap(m => Object.keys(m.settings||{})))].sort();
  const head = `<tr><th>knob</th>${ms.map(m => `<th>${esc(m.id)}</th>`).join("")}</tr>`;
  const rows = keys.map(k => {
    const vals = ms.map(m => (m.settings||{})[k]);
    const diff = new Set(vals.map(v => v==null?"":String(v))).size > 1;
    return `<tr><td class="kname">${esc(k)}</td>${vals.map(v => `<td class="${diff?"diff":""}">${v==null?`<span style="color:var(--dim)">inherit</span>`:`<span class="mono">${esc(v)}</span>`}</td>`).join("")}</tr>`;
  }).join("");
  showModal("Compare settings", keys.length ? `<table class="cmptbl">${head}${rows}</table>
    <div class="note">Highlighted cells differ across the selected models. "inherit" = not set for that model (falls back to the global [*] default).</div>`
    : `<div class="note">The selected models have no per-model knobs set - they all inherit the global defaults.</div>`);
}

/* ---------- modal ---------- */
function closeModal() {
  const dialog = $("#modal-root dialog");
  if (dialog && dialog.open) dialog.close();
  else setHTML($("#modal-root"), "");
}

export function showModal(title, inner, privateCopies = [], returnTo = document.activeElement) {
  const root = $("#modal-root");
  setHTML(root, `<dialog class="modal-dialog" aria-labelledby="modal-title">
    <div class="modal">
      <button type="button" class="mclose" data-mclose aria-label="Close dialog">&times;</button>
      <h3 id="modal-title">${esc(title)}</h3>${inner}
    </div>
  </dialog>`);
  const dialog = $("dialog", root);
  const finish = () => {
    // "close" fires async: if another modal replaced this one meanwhile, keep it
    if (!dialog.isConnected) return;
    setHTML(root, "");
    if (returnTo && returnTo.isConnected) returnTo.focus();
  };
  dialog.addEventListener("close", finish, {once: true});
  dialog.addEventListener("cancel", e => {
    e.preventDefault();
    dialog.close();
  });
  dialog.addEventListener("click", e => {
    if (e.target === dialog) dialog.close();
  });
  $$("[data-private-copy-index]", dialog).forEach(button => {
    const value = privateCopies[Number(button.dataset.privateCopyIndex)];
    button.removeAttribute("data-private-copy-index");
    button.onclick = e => {
      e.stopPropagation();
      navigator.clipboard.writeText(value).then(
        () => toast("Copied to clipboard", "ok"));
    };
  });
  dialog.showModal();
  $("[data-mclose]", dialog).focus();
  return {
    returnTo,
    isOpen: () => dialog.isConnected && dialog.open,
    close: () => { if (dialog.open) dialog.close(); },
  };
}

/* ---------- client config ---------- */
async function openClientConfig(id, backend) {
  const m = modelRows().find(
    row => row.id === id && (row.backend || "llamacpp") === backend);
  if (!m) return;
  const pending = showModal(
    "Client config - " + id,
    `<div class="note" role="status">Generating configuration...</div>`);
  let r;
  try {
    r = await api("/api/client/config", {model: m.id, backend});
  } catch (error) {
    if (pending.isOpen()) {
      showModal("Client config - " + id,
        `<div class="note" role="alert">Client configuration is unavailable.</div>`,
        [], pending.returnTo);
    }
    return;
  }
  if (!pending.isOpen()) return;
  if (r.error) {
    showModal("Client config - " + id,
      `<div class="note" role="alert">${esc(r.error)}</div>`,
      [], pending.returnTo);
    return;
  }
  const values = [r.curl, r.environment, r.payload];
  const snip = (label, text, index) =>
    `<div class="slabel">${esc(label)}</div><div class="snip">
      <button type="button" class="qbtn scopy"
              data-private-copy-index="${index}"
              aria-label="Copy ${esc(label)}">Copy</button>${esc(text)}</div>`;
  showModal("Client config - " + id,
    `<div class="note">Endpoint: <b>${esc(r.endpoint)}</b>. ${
      r.auth_required ? "The configured router credential is included." :
                        "No API key is required for this target."}${
      r.model_loaded ? "" :
        ` <b style="color:var(--amber)">Load the model before sending requests.</b>`
    }</div>` +
    snip("curl", r.curl, 0) +
    snip("OpenAI client (environment)", r.environment, 1) +
    snip("Test JSON payload", r.payload, 2),
    values, pending.returnTo);
}

/* ---------- presets ---------- */
function presetBar(m) {
  const c = cfgOf(), P = c.presets || {};
  const bound = ((c.preset_bindings || {})[m.backend || c.active_engine || "llamacpp"] || {})[m.id] || "";
  const chips = Object.keys(P).map(n => {
    const isBound = n === bound;
    return `<button type="button" class="pchip${isBound ? " bound" : ""}" data-preset-apply="${esc(n)}" data-preset-model="${esc(m.id)}" title="apply preset to this model">`
      + `<span class="pbind" data-preset-bind="${esc(n)}" data-preset-bind-model="${esc(m.id)}" title="${isBound ? "bound as default - click to unbind" : "bind as this model's default"}">${isBound ? "◉" : "○"}</span>`
      + `${esc(n)}<span class="px" data-preset-del="${esc(n)}" title="delete preset">&times;</span></button>`;
  }).join("");
  return `<div class="presetbar">
    <span style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)">Presets</span>
    ${chips||'<span class="note" style="margin:0">none saved yet</span>'}
    <button class="qbtn" data-preset-save="${esc(m.id)}" title="save this model's set knobs as a named preset">Save current +</button>
  </div>`;
}
async function applyPreset(model, name) {
  const r = await api("/api/presets/apply", {model, name});
  if (r.ok) { toast(`Applied "${name}"`, "ok"); delete diagCache[model]; invalidateKnobs(); await refresh(true); }
  else toast(r.error || "apply failed", "err");
}
async function bindPreset(model, name) {
  // toggle: clicking the dot of an already-bound preset unbinds it
  const c = cfgOf(), engine = c.active_engine || "llamacpp";
  const cur = ((c.preset_bindings || {})[engine] || {})[model] || "";
  const next = (cur === name) ? "" : name;
  const r = await api("/api/presets/bind", {model, name: next});
  if (r.ok) {
    toast(next ? `Bound "${name}" as default` : `Unbound "${name}"`, "ok");
    delete diagCache[model]; invalidateKnobs(); await refresh(true);
  } else toast(r.error || "bind failed", "err");
}
async function savePresetFrom(model) {
  const row = $(`.row[data-id="${CSS.escape(model)}"]`); if (!row) return;
  const settings = {};
  $$("[data-k]", row).forEach(el => { const v = el.value.trim(); if (v !== "") settings[el.dataset.k] = v; });
  if (!Object.keys(settings).length) { toast("No knobs set to save", "err"); return; }
  const name = await askText("Preset name", {title: "Save as preset", placeholder: "coding, creative, fast"});
  if (!name) return;
  const r = await api("/api/presets/save", {name: name.trim(), settings});
  if (r.ok) { toast(`Saved preset "${name.trim()}"`, "ok"); await refresh(true); }
  else toast(r.error || "save failed", "err");
}

/* ---------- GGUF metadata card ---------- */
function metaBlock(m) {
  if (m.backend === "vllm" || !m.in_ini) return "";
  const meta = metaCache[m.id];
  if (meta === undefined) { setTimeout(() => fetchMeta(m.id), 0); return `<div class="metacard"><div class="m"><span class="mv">reading GGUF header...</span></div></div>`; }
  if (!meta || !Object.keys(meta).length) return compatNote(m.id);
  const row = (k, v) => v == null ? "" : `<div class="m"><div class="mk">${esc(k)}</div><div class="mv">${esc(v)}</div></div>`;
  const runsOn = RUNS_ON[(compatCache[m.id] || {}).class];
  return `<div class="metacard">${row("architecture",meta.architecture)}${row("parameters",meta.size_label)}${row("quantization",meta.quantization)}${row("trained ctx",meta.context_length)}${row("embedding",meta.embedding_length)}${row("layers",meta.block_count)}${row("attn heads",meta.head_count)}${row("vocab",meta.vocab_size)}${row("experts",meta.expert_count)}${row("rope base",meta.rope_freq_base)}${row("rope scaling",meta.rope_scaling)}${row("runs on",runsOn)}</div>${compatNote(m.id)}`;
}
// From the GGUF's tensor types; advice only, since a third fork can reuse an id.
const RUNS_ON = {"any": "llama.cpp + ik_llama.cpp", "ik-only": "ik_llama.cpp, not mainline", "mainline-only": "llama.cpp, not ik_llama"};
function compatNote(id) {
  const c = compatCache[id];
  if (!c) return "";
  return (c.arch_advice ? `<div class="slotnote warn"><b>Build:</b> ${esc(c.arch_advice)}</div>` : "")
       + (c.advice ? `<div class="slotnote warn"><b>Quant:</b> ${esc(c.advice)}</div>` : "");
}
async function fetchMeta(id) {
  try {
    const r = await api("/api/model/metadata?model=" + encodeURIComponent(id));
    metaCache[id] = r.metadata || {}; compatCache[id] = r.compat || null; buildsCache[id] = r.builds || null;
  } catch (e) { metaCache[id] = {}; }
  if (openId === id) renderModels();
}

/* ---------- per-model build ----------------------------------------------
   A model pinned to a build other than the router's runs in its own
   llama-server process (slotproc.py), on its own port. That is how a model
   whose quant only ik_llama.cpp has can run at all: ik has no router mode. */
function buildTag(m) {
  if (!m.build) return "";
  const tip = m.process
    ? `runs on build ${m.build}, in its own llama-server process`
    : `pinned to build ${m.build}`;
  return `<span class="tag build" title="${esc(tip)}">${esc(m.build)}</span>`;
}
function buildBar(m) {
  const b = buildsCache[m.id];
  if (!b || !Array.isArray(b.options)) return "";
  const pinned = b.pinned || "", opts = b.options;
  const cur = opts.find(o => o.ref === pinned);
  // nothing to choose when every build is the router's own; one other build
  // (ik beside a router on the user's own binary) is already a choice
  if (!pinned && !opts.some(o => !o.router)) return "";
  const router = opts.find(o => o.router);
  const opt = (v, label) => `<option value="${esc(v)}"${v === pinned ? " selected" : ""}>${esc(label)}</option>`;
  const items = [opt("", `follow the router${router ? " (" + router.label + ")" : ""}`)]
    .concat(opts.map(o => opt(o.ref, o.label + (o.router ? " · the router's build now" : ""))));
  if (pinned && !cur) items.push(opt(pinned, `${pinned} (not installed)`));
  let note = "";
  if (pinned && !cur)
    note = `<div class="slotnote warn"><b>Build gone:</b> ${esc(pinned)} isn't installed any more, so this model won't load. Pick another build, or reinstall it from Build / Update.</div>`;
  else if (cur && cur.router)
    note = `<div class="slotnote dim">Pinned: it stays on ${esc(cur.label)} when the router moves to another build.</div>`;
  else if (cur)
    note = `<div class="slotnote dim">Runs in its own llama-server${m.process ? " on port " + esc(m.process.port) : ""}, not the router: clients reach it at its own endpoint, and Embers can't use it.</div>`;
  return `<div class="tunebar">
    <span class="tunebar-label" title="Which llama.cpp build runs this model. Follow the router, or keep it on one build whatever the router runs. A build other than the router's runs the model in its own process.">Build</span>
    <select data-build-pick="${esc(m.id)}">${items.join("")}</select>
  </div>${note}${slots.droppedNote(m.id)}`;
}
async function setBuild(sel) {
  const id = sel.dataset.buildPick, ref = sel.value;
  const row = sel.closest(".row"), msg = row && $("[data-msg]", row);
  const m = modelRows().find(x => x.id === id);
  const up = m && (m.status === "loaded" || m.status === "loading" || m.status === "sleeping");
  if (up && !(await askYes(`Switching the build unloads ${id}.`, {title: "Switch build", ok: "Switch and unload", danger: true}))) {
    if (row) row._live = null;                         // put the select back
    renderModels(); return;
  }
  sel.disabled = true;
  try {
    const r = await api("/api/model/build", {model: id, build: ref});
    if (r && r.ok) {
      if (buildsCache[id]) buildsCache[id].pinned = r.build;
      delete diagCache[id]; slots.forget(id);
      if (msg) { msg.className = "msg ok"; msg.textContent = r.was_running ? "build saved - unloaded to apply" : "build saved - used on the next load"; }
      toast(ref ? `${id} now runs on ${ref}` : `${id} follows the router's build`, "ok");
    } else {
      toast((r && r.error) || "could not change the build", "err");
      if (row) row._live = null;
    }
  } catch (e) { toast("could not change the build: " + e, "err"); if (row) row._live = null; }
  await refresh(true);
}

/* ---------- thinking + vision -----------------------------------------------
   One-click switches over knobs that already exist: Thinking writes llama.cpp's
   reasoning / reasoning-budget (Auto and Unlimited remove the key, so the
   router's default applies); Vision parks the mmproj line server-side. Both
   apply at once - a loaded model is unloaded so its next load picks it up. */
const hasKnob = key => !!(S.SCHEMA && (S.SCHEMA.groups || []).some(g => g.knobs.some(k => k.key === key)));
const THINK_MODES = [["", "Auto"], ["on", "On"], ["off", "Off"]];
const THINK_BUDGETS = [["", "unlimited"], ["512", "512 tokens"], ["2048", "2k tokens"], ["8192", "8k tokens"]];
function thinkBar(m) {
  if (!m.in_ini || !hasKnob("reasoning")) return "";
  const s = m.settings || {}, mode = s.reasoning || "", budget = s["reasoning-budget"] || "";
  const meta = metaCache[m.id], kind = meta && meta.thinking;
  // a template with no reasoning markers has nothing to switch; still shown if
  // something was set, so it can be put back
  if (meta && Object.keys(meta).length && !kind && !mode && !budget) return "";
  const segs = THINK_MODES.map(([v, label]) =>
    `<button class="qbtn" data-think-mode="${v}" data-think="${esc(m.id)}" aria-pressed="${mode === v}">${label}</button>`).join("");
  let budgetSel = "";
  if (hasKnob("reasoning-budget")) {
    const opts = THINK_BUDGETS.slice();
    const odd = {"0": "0 · no thinking", "-1": "unlimited (-1)"};   // llama.cpp's two special values
    if (budget && !opts.some(([v]) => v === budget)) opts.push([budget, odd[budget] || `${budget} tokens`]);
    budgetSel = `<select data-think-budget="${esc(m.id)}" title="most tokens it may think before answering"${mode === "off" ? " disabled" : ""}>`
      + opts.map(([v, label]) => `<option value="${esc(v)}"${v === budget ? " selected" : ""}>${esc(label)}</option>`).join("") + `</select>`;
  }
  const hint = kind === "switch" ? "Its chat template has a thinking switch, so Off really turns it off."
    : kind === "tags" ? "Its template shows thinking but has no switch: Off may not stop a model trained to always think."
    : "";
  return `<div class="tunebar">
    <span class="tunebar-label" title="llama.cpp's --reasoning and --reasoning-budget. Auto lets the chat template decide.">Thinking</span>
    <span class="tseg" role="group" aria-label="thinking">${segs}</span>${budgetSel}
  </div>${hint ? `<div class="slotnote dim">${esc(hint)}</div>` : ""}`;
}
function visionBar(m) {
  if (!m.in_ini) return "";
  const on = !!(m.settings || {}).mmproj;
  if (!on && !m.vision_off) return "";
  const seg = (v, label) => `<button class="qbtn" data-vision="${esc(m.id)}" data-vision-on="${v}" aria-pressed="${on === v}">${label}</button>`;
  return `<div class="tunebar">
    <span class="tunebar-label" title="Loads the model's vision projector (mmproj). Off keeps the file on disk and frees its VRAM.">Vision</span>
    <span class="tseg" role="group" aria-label="vision">${seg(true, "On")}${seg(false, "Off")}</span>
  </div>${on ? "" : `<div class="slotnote dim">Off: text only. The projector file is kept; turn it back on any time.</div>`}`;
}
async function applyToggle(row, id, label, call) {
  const m = modelRows().find(x => x.id === id);
  const up = m && (m.status === "loaded" || m.status === "loading" || m.status === "sleeping");
  if (up && !(await askYes(`Changing this unloads ${id}; its next load uses the new setting.`, {title: label, ok: "Change and unload"}))) {
    if (row) row._live = null;                         // put the controls back
    renderModels(); return;
  }
  const msg = row && $("[data-msg]", row);
  if (msg) { msg.className = "msg work"; msg.textContent = "writing models.ini..."; }
  let r;
  try { r = await call(); } catch (e) { r = {error: String(e)}; }
  if (r && r.ok) {
    if (msg) { msg.className = "msg ok"; msg.textContent = r.was_running ? `${label.toLowerCase()} saved - unloaded to apply` : `${label.toLowerCase()} saved - used on the next load`; }
    delete diagCache[id]; invalidateKnobs(); slots.forget(id);
    toast(`${label} saved`, "ok");
  } else {
    if (msg) { msg.className = "msg err"; msg.textContent = (r && r.error) || "failed"; }
    toast((r && r.error) || `could not change ${label.toLowerCase()}`, "err");
  }
  if (row) row._live = null;
  await refresh(true);
}
const saveThinking = (row, id, settings) =>
  applyToggle(row, id, "Thinking", () => api("/api/save", {model: id, settings}));
const saveVision = (row, id, on) =>
  applyToggle(row, id, "Vision", () => api("/api/model/vision", {model: id, on}));

/* ---------- autotune bar ---------- */
function autoTuneBar(m) {
  return `<div class="tunebar">
    <span class="tunebar-label" title="Refine tries a few variants with a short generation test and keeps the best. A quick check, not llama-bench. Each candidate runs one real ~200-token completion, so results move with system load.">⚙ Refine</span>
    <select data-tune-intent>
      <option value="balanced">Balanced</option>
      <option value="speed">Max speed</option>
      <option value="context">Max context</option>
      <option value="coding">Coding</option>
    </select>
    <button class="qbtn" data-tune-refine="${esc(m.id)}">Run (~1 min)</button>
  </div>
  <div class="tunebar-results" data-tune-results hidden></div>`;
}
function applyTuneResult(row, rec) {
  const knobs = rec.knobs || {}, changed = [];
  $$("[data-k]", row).forEach(el => {
    const k = el.dataset.k;
    if (knobs[k] != null && knobs[k] !== el.value) {
      el.value = knobs[k];
      if (el.value.trim()) el.classList.add("set");
      changed.push(k);
    }
  });
  const msg = $("[data-msg]", row);
  if (msg && changed.length) {
    msg.className = "msg work";
    msg.textContent = `${changed.length} knobs updated — unsaved changes`;
  }
  const refineBtn = row.querySelector("[data-tune-refine]");
  if (refineBtn) refineBtn.hidden = false;
  row._tuneRec = rec;
}
function renderTuneResults(row, measurements) {
  const el = $("[data-tune-results]", row);
  if (!el) return;
  const cands = measurements?.candidates || [];
  if (!cands.length) { el.hidden = true; return; }
  const bestTok = measurements?.chosen_tok_s || 0;
  const rows = cands.map(c => {
    const tok = (c.tok_s || 0).toFixed(1);
    const isBest = Math.abs(c.tok_s - bestTok) < 0.01;
    const diff = Object.entries(c.knobs).filter(([k,v]) => {
      const base = cands[0]?.knobs?.[k];
      return base != null && base !== v;
    }).map(([k,v]) => `${k}=${v}`).join(", ");
    const label = diff ? diff : "base";
    return `<div class="tunebar-cand${isBest?" best":""}"><span class="tunebar-cand-label">${esc(label)}</span><span class="tunebar-cand-tok">${tok} tok/s</span>${isBest?'<span class="tunebar-cand-best">← chosen</span>':''}</div>`;
  }).join("");
  setHTML(el, `<div class="tunebar-cand-header"><span>candidate</span><span>speed</span></div>${rows}`);
  el.hidden = false;
}
async function handleTuneRefine(modelId) {
  const row = $(`.row[data-id="${CSS.escape(modelId)}"]`); if (!row) return;
  const intent = $("[data-tune-intent]", row)?.value || "balanced";
  const btn = $("[data-tune-refine]", row);
  btn.disabled = true; btn.textContent = "benchmarking...";
  try {
    const r = await api("/api/autotune/refine", {model: modelId, intent});
    if (r.error) { toast(r.error, "err"); return; }
    const tok = (r.measurements?.chosen_tok_s || 0).toFixed(1);
    applyTuneResult(row, {knobs: r.knobs, intent});
    renderTuneResults(row, r.measurements);
    if (r.restore_error) {
      const msg = $("[data-msg]", row);
      if (msg) { msg.className = "msg err"; msg.textContent = `benchmarked, but ${modelId} didn't load back: ${r.restore_error}`; }
      toast(`Refined, but ${modelId} didn't load back`, "err");
    } else toast(`Refined — ${tok} tok/s`, "ok");
  } catch (e) { toast("Refine failed: " + e, "err"); }
  btn.disabled = false; btn.textContent = "Run (~1 min)";
}

/* ---------- inline load-failure diagnosis ---------- */
function diagBlock(m) {
  if (!m.failed) return "";
  const d = diagCache[m.id];
  if (d === undefined) { setTimeout(() => fetchDiag(m.id), 0); return `<div class="faildiag"><div class="ferr">reading the router log...</div></div>`; }
  if (!d) return `<div class="faildiag"><div class="ffix">Load failed, but no specific cause was found in the router log - see the Router Log panel below.</div></div>`;
  return `<div class="faildiag"><div class="ferr">${esc(d.error)}</div><div class="ffix"><b>Suggested fix:</b> ${esc(d.suggestion)}</div></div>`;
}
async function fetchDiag(id) {
  try { const r = await api("/api/model/diag?model=" + encodeURIComponent(id)); diagCache[id] = r.diag || null; }
  catch (e) { diagCache[id] = null; }
  if (openId === id) renderModels();
}

/* ---------- keyboard navigation ---------- */
function moveSel(delta) {
  const ms = shownModels(); if (!ms.length) return;
  let i = ms.findIndex(m => m.id === selId);
  i = i < 0 ? (delta>0?0:ms.length-1) : Math.min(ms.length-1, Math.max(0, i+delta));
  selId = ms[i].id; renderModels();
  const row = $(`.row[data-id="${CSS.escape(selId)}"]`);
  if (row) row.scrollIntoView({block: "nearest"});
}

/* ---------- refresh ---------- */
// The tab polls every 4 s; while the router streams a load's progress, a 1 s
// follow-up keeps the bar moving. One timer at a time; it stops with the load.
let fastPoll = null;
function pollFasterWhileLoading() {
  if (fastPoll || !modelRows().some(m => loadPct(m) !== null)) return;
  fastPoll = setTimeout(() => { fastPoll = null; if (activeTab() === "models") refresh(true); }, 1000);
}
export async function refresh(silent) {
  try {
    // recover the knob schema without a reload once config.json is fixed
    if (!S.SCHEMA || S.SCHEMA.error || !(S.SCHEMA.groups||[]).length) S.SCHEMA = await api("/api/schema");
    const s = await api("/api/state");
    S.STATE = s;
    lastGpus = s.gpus;
    renderModels();
    pollFasterWhileLoading();
    updateCmpRun();
    emit("state", s);
    const vlog = $("#vllm-log-details");
    if (vlog && s.vllm_supported === false) vlog.style.display = "none";
  } catch (e) {
    if (!silent) setHTML($("#list"), `<div class="skel" style="color:var(--red)">BACKEND UNREACHABLE</div>`);
  }
}
on("refresh", silent => refresh(silent));

/* ---------- log panels ---------- */
export async function refreshRouterLog() {
  const el = $("#router-log"); if (!el) return;
  const s = await api("/api/router/log");
  const wasBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
  el.textContent = s.log || "idle";
  if (wasBottom) el.scrollTop = el.scrollHeight;
}
export async function refreshVllmLog() {
  const el = $("#vllm-log"); if (!el) return;
  const s = await api("/api/vllm/log");
  const wasBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
  el.textContent = s.log || "idle";
  if (wasBottom) el.scrollTop = el.scrollHeight;
}

/* ---------- event wiring ---------- */
export function initModels() {
  autoforge.initAutoForge(renderModels);
  // The Models toolbar lives in index.html with stable ids, so it is wired
  // directly rather than through the delegated handlers the dynamic rows use.
  const ms = $("#model-search"); if (ms) ms.oninput = () => filterModels(ms);
  const fc = $("#fav-chip"); if (fc) fc.onclick = () => toggleFavOnly(fc);
  const cc = $("#cmp-chip"); if (cc) cc.onclick = () => toggleCompare(cc);
  const cr = $("#cmp-run"); if (cr) cr.onclick = () => openCompare();
  const ua = $("#unload-all"); if (ua) ua.onclick = () => unloadAll();
  const routerLog = $("#router-log-details");
  if (routerLog) routerLog.addEventListener("toggle", () => {
    if (routerLog.open) refreshRouterLog();
  });
  const vllmLog = $("#vllm-log-details");
  if (vllmLog) vllmLog.addEventListener("toggle", () => {
    if (vllmLog.open) refreshVllmLog();
  });

  document.addEventListener("change", e => {
    const pick = e.target.closest("#view-models [data-build-pick]");
    if (pick) setBuild(pick);
    const tb = e.target.closest("#view-models [data-think-budget]");
    if (tb) saveThinking(tb.closest(".row"), tb.dataset.thinkBudget, {"reasoning-budget": tb.value});
  });

  document.addEventListener("input", e => {
    // knob-filter box (dynamic, inside an open editor)
    if (e.target.matches("[data-knobfilter]")) { filterKnobs(e.target); return; }
    // flag knob edits so the user knows a Save is pending
    const row = e.target.closest("#view-models .row.open");
    if (!row || e.target.dataset.k == null) return;
    const msg = $("[data-msg]", row);
    if (msg) { msg.className = "msg work"; msg.textContent = "unsaved changes"; }
  });

  // keyboard map: 1-9 tabs (sidebar order), / search, j/k or arrows navigate, Enter expand,
  // L load, U unload, S save the open model. Esc closes a modal / clears search.
  document.addEventListener("keydown", e => {
    const tag = (document.activeElement || {}).tagName || "";
    const typing = /INPUT|SELECT|TEXTAREA/.test(tag);
    if (typing) {
      if (e.key === "Escape" && document.activeElement === $("#model-search")) {
        const inp = $("#model-search"); inp.value = ""; mquery = ""; renderModels(); inp.blur();
      }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (/^[1-9]$/.test(e.key)) {
      // follow the sidebar, so a new tab can't silently shift the numbering
      const el = document.querySelectorAll(".navitem[data-tab]")[+e.key-1]; if (el) el.click();
      return;
    }
    if (activeTab() !== "models" || !S.STATE) return;
    if (e.key === "/") { e.preventDefault(); const inp = $("#model-search"); if (inp) inp.focus(); return; }
    if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); moveSel(1); return; }
    if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); moveSel(-1); return; }
    if (!selId) return;
    const m = modelRows().find(x => x.id === selId); if (!m) return;
    if (e.key === "Enter") { e.preventDefault(); setOpenId(openId===selId?null:selId); renderModels(); return; }
    if (e.key === "l" || e.key === "L") { if (m.status !== "loaded") quickAction("load", selId); return; }
    if (e.key === "u" || e.key === "U") { if (m.status === "loaded" || m.status === "loading") quickAction("unload", selId); return; }
    if (e.key === "s" || e.key === "S") {
      if (openId === selId) {
        const b = $(`.row[data-id="${CSS.escape(selId)}"] button[data-act="save"]`)
               || $(`.row[data-id="${CSS.escape(selId)}"] button[data-act="vsave"]`);
        if (b) b.click();
      }
      return;
    }
  });

  document.addEventListener("click", async e => {
    const cpChip = e.target.closest("#view-models [data-copy]");
    if (cpChip) { e.stopPropagation(); navigator.clipboard.writeText(cpChip.dataset.copy).then(() => toast("Path copied","ok")); return; }
    const epChip = e.target.closest("#view-models [data-ep]");
    if (epChip) { e.stopPropagation(); navigator.clipboard.writeText(epChip.dataset.ep).then(() => toast("Endpoint copied","ok")); return; }
    const favBtn = e.target.closest("#view-models [data-fav]");
    if (favBtn) { e.stopPropagation(); toggleFav(favBtn.dataset.fav); return; }
    const elGrow = e.target.closest("#view-models [data-elastic-grow]");
    if (elGrow) {
      e.stopPropagation(); elGrow.disabled = true;
      const id = elGrow.dataset.elasticGrow;
      const r = await api("/api/elastic/grow", {model: id}).catch(() => null);
      elGrow.disabled = false;
      toast(r && r.ok ? `Growing ${id} back if it fits; the Setup tab's Elastic VRAM log shows how it went`
                      : (r && r.error) || "grow failed", r && r.ok ? "ok" : "err");
      return;
    }
    const onlySetChip = e.target.closest("#view-models [data-onlyset]");
    if (onlySetChip) { e.stopPropagation(); toggleOnlySet(onlySetChip); return; }
    if (e.target.closest("[data-mclose]")) { closeModal(); return; }
    // compare-pick checkbox
    const cmpBox = e.target.closest("[data-cmp]");
    if (cmpBox) {
      const id = cmpBox.dataset.cmp;
      if (cmpBox.checked) {
        if (cmpSet.size >= 3 && !cmpSet.has(id)) { cmpBox.checked = false; toast("Compare up to 3 at once","err"); return; }
        cmpSet.add(id);
      } else cmpSet.delete(id);
      updateCmpRun(); return;
    }
    // quick load/unload in the row header
    const quick = e.target.closest("#view-models [data-quick]");
    if (quick) { e.stopPropagation(); quickAction(quick.dataset.quick, quick.dataset.qid); return; }
    // multi-model slots: make main, unload-the-workers-and-load, restart with the pool
    const sMain = e.target.closest("#view-models [data-slot-main]");
    if (sMain) {
      e.stopPropagation(); sMain.disabled = true;
      await slots.makeMain(sMain.dataset.slotMain); await refresh(true); return;
    }
    const sEvict = e.target.closest("#view-models [data-slot-evict]");
    if (sEvict) {
      e.stopPropagation(); sEvict.disabled = true;
      const id = sEvict.dataset.slotEvict;
      toast(`Unloading workers, then loading ${id}...`, "ok");
      const r = await slots.load(id, "main", true);
      toast(r.success ? `${id} is loaded` : `${id}: ${slots.errText(r)}`.slice(0, 220), r.success ? "ok" : "err");
      await refresh(true); return;
    }
    if (e.target.closest("#view-models [data-slot-apply]")) {
      e.stopPropagation();
      if (await slots.applyPool()) await refresh(true);
      return;
    }
    // presets
    const pApply = e.target.closest("[data-preset-apply]");
    if (pApply) {
      e.stopPropagation();
      const pbind = e.target.closest("[data-preset-bind]");
      if (pbind) { await bindPreset(pbind.dataset.presetBindModel, pbind.dataset.presetBind); return; }
      const pdel = e.target.closest("[data-preset-del]");
      if (pdel) {
        if (!(await askYes("This can't be undone.", {title: `Delete preset "${pdel.dataset.presetDel}"`, ok: "Delete", danger: true}))) return;
        await api("/api/presets/delete", {name: pdel.dataset.presetDel}); toast("Preset deleted","ok"); await refresh(true); return;
      }
      await applyPreset(pApply.dataset.presetModel, pApply.dataset.presetApply); return;
    }
    const pSave = e.target.closest("[data-preset-save]");
    if (pSave) { e.stopPropagation(); await savePresetFrom(pSave.dataset.presetSave); return; }
    // thinking / vision switches
    const tMode = e.target.closest("#view-models [data-think-mode]");
    if (tMode) {
      e.stopPropagation();
      if (tMode.getAttribute("aria-pressed") !== "true")
        await saveThinking(tMode.closest(".row"), tMode.dataset.think, {reasoning: tMode.dataset.thinkMode});
      return;
    }
    const vis = e.target.closest("#view-models [data-vision]");
    if (vis) {
      e.stopPropagation();
      if (vis.getAttribute("aria-pressed") !== "true")
        await saveVision(vis.closest(".row"), vis.dataset.vision, vis.dataset.visionOn === "true");
      return;
    }
    if (autoforge.handleClick(e)) return;
    // autotune
    const tRef = e.target.closest("[data-tune-refine]");
    if (tRef) { e.stopPropagation(); await handleTuneRefine(tRef.dataset.tuneRefine); return; }
    const head = e.target.closest("#view-models .rhead");
    if (head && !e.target.closest("button,input")) {
      const id = head.parentElement.dataset.id;
      setOpenId(openId === id ? null : id); selId = id; kquery = ""; renderModels(); return;
    }
    const btn = e.target.closest("#view-models button[data-act]");
    if (!btn) return;
    const row = btn.closest(".row"), id = row.dataset.id, msg = $("[data-msg]", row), act = btn.dataset.act;
    if (act === "chat") { emit("chat", id); return; }
    if (act === "profile") { emit("profile-save", {id, backend: row.dataset.backend}); return; }
    const clientOpening = act === "client"
      ? openClientConfig(id, row.dataset.backend) : null;
    btn.disabled = true;
    try {
      if (act === "save") {
        const settings = {}; $$("[data-k]", row).forEach(el => settings[el.dataset.k] = el.value.trim());
        msg.className = "msg work"; msg.textContent = "writing models.ini...";
        const r = await api("/api/save", {model: id, settings});
        if (r.ok) {
          msg.className = "msg ok";
          msg.textContent = r.was_running ? "saved - unloaded to apply" : "saved + reloaded";
          toast("Saved & reloaded", "ok");
          invalidateKnobs();   // server now matches the inputs; refresh "set" marks
          slots.forget(id);    // new knobs, new footprint: ask the planner again
        } else { msg.className = "msg err"; msg.textContent = r.error || "failed"; }
      } else if (act === "load") {
        msg.className = "msg work"; msg.textContent = "loading (may take seconds)...";
        const r = await slots.load(id);
        r.success ? toast("Loaded","ok") : (msg.className="msg err", msg.textContent=slots.errText(r));
      } else if (act === "loadw") {
        msg.className = "msg work"; msg.textContent = "loading beside the main model...";
        const r = await slots.load(id, "worker");
        r.success ? toast("Loaded as a worker","ok") : (msg.className="msg err", msg.textContent=slots.errText(r));
      } else if (act === "unload") {
        msg.className = "msg work"; msg.textContent = "unloading...";
        await api("/api/unload", {model: id}); toast("Unloaded", "ok");
      } else if (act === "client") {
        await clientOpening; return;
      } else if (act === "unregister") {
        if (!(await askYes("Its GGUF file stays on disk.", {title: `Unregister ${id}`, ok: "Unregister", danger: true}))) { btn.disabled = false; return; }
        msg.className = "msg work"; msg.textContent = "removing from models.ini...";
        const r = await api("/api/models/unregister", {model: id, backend: beOf(id)});
        if (r.ok) { toast("Unregistered — file kept on disk", "ok"); setOpenId(null); }
      } else if (act === "vsave") {
        const settings = {}; $$("[data-k]", row).forEach(el => settings[el.dataset.k] = el.value.trim());
        msg.className = "msg work"; msg.textContent = "saving vLLM knobs...";
        const r = await api("/api/vllm/save", {model: id, settings});
        msg.className = "msg ok"; msg.textContent = r.restarted ? "saved - restarting" : "saved";
        toast(r.restarted ? "Saved & restarting" : "Saved", "ok");
        invalidateKnobs();
      } else if (act === "vload") {
        msg.className = "msg work"; msg.textContent = "starting vLLM (1-5 min)...";
        const r = await api("/api/vllm/load", {model: id});
        r.ok ? toast("vLLM starting","ok") : (msg.className="msg err", msg.textContent=r.error||"load failed");
      } else if (act === "vunload") {
        msg.className = "msg work"; msg.textContent = "stopping vLLM...";
        await api("/api/vllm/unload", {model: id}); toast("vLLM stopped", "ok");
      } else if (act === "vdelete") {
        if (!(await askYes("Its files are deleted from WSL. This cannot be undone.", {title: `Delete ${id}`, ok: "Delete", danger: true}))) return;
        msg.className = "msg work"; msg.textContent = "deleting from WSL...";
        const r = await api("/api/vllm/delete", {model: id});
        r.ok ? toast("Deleted","ok") : (msg.className="msg err", msg.textContent=r.error||"delete failed");
        setOpenId(null);
      }
      await refresh(true);
    } catch (err) { msg.className = "msg err"; msg.textContent = String(err); }
    finally { if (btn.isConnected) btn.disabled = false; }
  });
}
