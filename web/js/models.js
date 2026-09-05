// The Models tab: GPU telemetry, the model list, the per-model knob editor,
// load/unload, compare, presets, and the router/vLLM log panels.
//
// Rendering contract - the thing to preserve when editing this file:
// renderModels() reconciles rows against the DOM instead of rebuilding it.
// A row whose data is unchanged is left alone, so focus, caret position,
// scroll and half-typed knob values survive the 4-second poll. See syncEditor()
// for how the editor separates what the server owns from what the user is
// typing. Nothing here may re-render a knob input the user might be editing.
import { $, $$, esc, setHTML, api, toast, meter } from "./core.js";
import { S, models as modelRows, config as cfgOf } from "./state.js";
import { on, emit } from "./bus.js";
import { activeTab } from "./ui.js";

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

/* ---------- GPU telemetry ---------- */
export function renderGpus(g) {
  if (!g || !g.length || g[0].error) {
    setHTML($("#gpus"), `<div class="gpu"><div class="stats">GPU telemetry unavailable</div></div>`);
    return;
  }
  setHTML($("#gpus"), g.map(x => `<div class="gpu"><div class="top"><span class="name">${esc(x.name)}</span><span class="idx">CUDA${esc(x.index)}</span></div>
    <div class="meter">${meter(x.used,x.total)}</div>
    <div class="stats"><span><b>${esc((x.used/1024).toFixed(1))}</b>/${esc((x.total/1024).toFixed(1))} GB</span><span>FREE <b>${esc(((x.total-x.used)/1024).toFixed(1))}</b> GB</span><span>UTIL <b>${esc(x.util)}%</b></span><span>TEMP <b>${esc(x.temp)}&deg;C</b></span></div></div>`).join(""));
}

/* ---------- knob fields ---------- */
function curVal(m, knob) {
  for (const a of knob.aliases) if (m.settings[a] != null) return m.settings[a];
  return "";
}
function knobField(m, k) {
  const v = curVal(m, k), isSet = v !== "";
  const ph = k.default ? `inherit (${k.default})` : "inherit";
  let ctrl;
  if (k.type === "enum") {
    const opts = [""].concat(k.options||[]).map(o => `<option value="${esc(o)}" ${String(o)===String(v)?"selected":""}>${o===""?"(inherit)":esc(o)}</option>`).join("");
    ctrl = `<select data-k="${esc(k.key)}">${opts}</select>`;
  } else if (k.type === "bool") {
    const opts = ["","true","false"].map(o => `<option value="${o}" ${String(o)===String(v)?"selected":""}>${o===""?"(inherit)":o}</option>`).join("");
    ctrl = `<select data-k="${esc(k.key)}">${opts}</select>`;
  } else {
    ctrl = `<input data-k="${esc(k.key)}" value="${esc(v)}" placeholder="${esc(ph)}" ${(k.type==="int"||k.type==="float")?'inputmode="numeric"':''}>`;
  }
  return `<div class="fld ${isSet?"set":""}${LITE_KNOBS.has(k.key)?"":" advanced-only"}" data-desc="${esc((k.key+' '+k.desc).toLowerCase())}">
    <label title="${esc(k.desc)}">${esc(k.key)}</label>${ctrl}
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
    : `${diagBlock(m)}${metaBlock(m)}${modelMeta(m)}${presetBar(m)}${autoTuneBar(m)}`;
}
function editorButtons(m) {
  if (m.backend === "vllm") {
    return `<button class="primary" data-act="vsave">Save${m.status==="loaded"?" + Restart":""}</button>
      ${m.status==="loaded"||m.status==="loading"?`<button class="ghost" data-act="vunload">${m.status==="loading"?"Cancel / Stop":"Stop"}</button>`:`<button data-act="vload">Load</button>`}
      <button class="ghost" data-act="client">Client config</button>
      <button class="ghost" data-act="vdelete" title="remove model + delete its files from WSL">Delete</button>`;
  }
  return `<button class="primary" data-act="save">Save + Reload</button>
      ${m.status==="loaded"||m.status==="loading"?`<button class="ghost" data-act="unload">${m.status==="loading"?"Cancel / Unload":"Unload"}</button>`:`<button data-act="load">Load</button>`}
      <button class="ghost" data-act="client">Client config</button>
      <button class="ghost" data-act="unregister" title="remove from models.ini; does not delete the GGUF">Unregister</button>`;
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
  return `<div class="ed-live">${editorLive(m)}</div>
    <div class="toolbar ed-tools">
      <input class="search" data-knobfilter placeholder="${esc(placeholder)}">
      <span class="chip ${onlySet?"on":""}" data-onlyset>Only set</span>
    </div>
    <div class="ed-knobs">${knobGroups(m,schema)}</div>
    <div class="actions">
      <span class="ed-btns">${editorButtons(m)}</span>
      <span class="msg" data-msg></span>
    </div>
    <div class="ed-note">${editorNote(m)}</div>`;
}
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
        <span class="led ${loaded?"loaded":""} ${m.failed?"failed":""}"></span>
        <span class="fav ${isFav?"on":""}" data-fav="${esc(m.id)}" title="${isFav?"unfavorite":"favorite"}">&starf;</span>
        <span class="mid" title="${esc(m.id)}">${esc(m.id)}${beTag}${vis?'<span class="tag vis">vision</span>':''}${!m.in_ini?'<span class="tag">auto</span>':''}${m.endpoint?`<span class="tag ep" data-ep="${esc(m.endpoint)}" title="click to copy endpoint">${esc(m.endpoint.replace('http://',''))}</span>`:''}</span>
        <span class="ctxpill"><span class="k">CTX</span> ${esc(m.eff_ctx)}</span>
        <span class="stat ${loaded?"loaded":""}" style="${stuckSecs>=20?"color:var(--red)":""}">${m.failed?"FAILED":esc(m.status)}${stuckSecs>=20?` (${stuckSecs}s, check log)`:""}</span>
        ${quickBtn(m)}
        <span class="chev">&#9654;</span>`;
}
// Everything rowHead() reads. Compared as a string so an unchanged row is left
// in the DOM untouched - which is what keeps focus, selection and scroll alive.
function headSig(m, cols, showBackend) {
  return JSON.stringify([m.id, m.status, m.failed, m.backend, m.endpoint, m.eff_ctx,
    m.modalities, m.in_ini, favs.has(m.id), compareMode, cmpSet.has(m.id),
    loadQ.findIndex(j => j.id === m.id), loadingSecs(m) >= 20, cols, showBackend]);
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
  const cols = compareMode ? "16px 14px 18px 1fr auto auto auto auto"
                           : "14px 18px 1fr auto auto auto auto";
  const showBackend = backendTagNeeded();
  const list = $("#list");
  if (!list) return;
  if (!ms.length) { setHTML(list, `<div class="skel">NO MODELS MATCH</div>`); return; }
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
    row._live = row._btns = row._note = null;
  }
  const regions = [[".ed-live", editorLive, "_live"],
                   [".ed-btns", editorButtons, "_btns"],
                   [".ed-note", editorNote, "_note"]];
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
  if (m.status === "loaded") return `<button class="qbtn stop" data-quick="unload" data-qid="${esc(m.id)}">Unload</button>`;
  return `<button class="qbtn load" data-quick="load" data-qid="${esc(m.id)}">Load</button>`;
}
function beOf(id) {
  const m = modelRows().find(x => x.id === id);
  return m ? (m.backend || "llamacpp") : "llamacpp";
}
function enqueueLoad(id) {
  if (loadQ.some(j => j.id === id) || (loadBusy && loadQ[0] && loadQ[0].id === id)) return;
  delete diagCache[id];               // a retry should re-diagnose, not show stale error
  loadQ.push({id});
  toast(loadBusy ? `Queued #${loadQ.length}` : "Loading...", "ok");
  renderModels(); processQ();
}
async function processQ() {
  if (loadBusy || !loadQ.length) return;
  loadBusy = true;
  const job = loadQ[0];
  try { await api(beOf(job.id) === "vllm" ? "/api/vllm/load" : "/api/load", {model: job.id}); } catch (e) {}
  loadQ.shift(); loadBusy = false;
  await refresh(true);
  processQ();
}
async function quickAction(act, id) {
  const be = beOf(id);
  if (act === "load") { enqueueLoad(id); return; }
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

function showModal(title, inner, privateCopies = [], returnTo = document.activeElement) {
  const root = $("#modal-root");
  setHTML(root, `<dialog class="modal-dialog" aria-labelledby="modal-title">
    <div class="modal">
      <button type="button" class="mclose" data-mclose aria-label="Close dialog">&times;</button>
      <h3 id="modal-title">${esc(title)}</h3>${inner}
    </div>
  </dialog>`);
  const dialog = $("dialog", root);
  const finish = () => {
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
    return `<span class="pchip${isBound ? " bound" : ""}" data-preset-apply="${esc(n)}" data-preset-model="${esc(m.id)}" title="apply preset to this model">`
      + `<span class="pbind" data-preset-bind="${esc(n)}" data-preset-bind-model="${esc(m.id)}" title="${isBound ? "bound as default - click to unbind" : "bind as this model's default"}">${isBound ? "◉" : "○"}</span>`
      + `${esc(n)}<span class="px" data-preset-del="${esc(n)}" title="delete preset">&times;</span></span>`;
  }).join("");
  return `<div class="presetbar">
    <span style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)">Presets</span>
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
  const name = prompt("Preset name (e.g. coding, creative, fast):");
  if (!name || !name.trim()) return;
  const r = await api("/api/presets/save", {name: name.trim(), settings});
  if (r.ok) { toast(`Saved preset "${name.trim()}"`, "ok"); await refresh(true); }
  else toast(r.error || "save failed", "err");
}

/* ---------- GGUF metadata card ---------- */
function metaBlock(m) {
  if (m.backend === "vllm" || !m.in_ini) return "";
  const meta = metaCache[m.id];
  if (meta === undefined) { setTimeout(() => fetchMeta(m.id), 0); return `<div class="metacard"><div class="m"><span class="mv">reading GGUF header...</span></div></div>`; }
  if (!meta || !Object.keys(meta).length) return "";
  const row = (k, v) => v == null ? "" : `<div class="m"><div class="mk">${esc(k)}</div><div class="mv">${esc(v)}</div></div>`;
  return `<div class="metacard">${row("architecture",meta.architecture)}${row("parameters",meta.size_label)}${row("quantization",meta.quantization)}${row("trained ctx",meta.context_length)}${row("embedding",meta.embedding_length)}${row("layers",meta.block_count)}${row("attn heads",meta.head_count)}${row("vocab",meta.vocab_size)}${row("experts",meta.expert_count)}${row("rope base",meta.rope_freq_base)}${row("rope scaling",meta.rope_scaling)}</div>`;
}
async function fetchMeta(id) {
  try { const r = await api("/api/model/metadata?model=" + encodeURIComponent(id)); metaCache[id] = r.metadata || {}; }
  catch (e) { metaCache[id] = {}; }
  if (openId === id) renderModels();
}

/* ---------- autotune bar ---------- */
function autoTuneBar(m) {
  return `<div class="tunebar">
    <span class="tunebar-label" title="⚠ Benchmarks run a real completion request (~200 tokens) per candidate. Results depend on your hardware and current system load.">⚙ Refine</span>
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
  el.innerHTML = `<div class="tunebar-cand-header"><span>candidate</span><span>speed</span></div>${rows}`;
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
    toast(`Refined — ${tok} tok/s`, "ok");
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
export async function refresh(silent) {
  try {
    // recover the knob schema without a reload once config.json is fixed
    if (!S.SCHEMA || S.SCHEMA.error || !(S.SCHEMA.groups||[]).length) S.SCHEMA = await api("/api/schema");
    const s = await api("/api/state");
    S.STATE = s;
    renderGpus(s.gpus);
    renderModels();
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

  document.addEventListener("input", e => {
    // knob-filter box (dynamic, inside an open editor)
    if (e.target.matches("[data-knobfilter]")) { filterKnobs(e.target); return; }
    // flag knob edits so the user knows a Save is pending
    const row = e.target.closest("#view-models .row.open");
    if (!row || e.target.dataset.k == null) return;
    const msg = $("[data-msg]", row);
    if (msg) { msg.className = "msg work"; msg.textContent = "unsaved changes"; }
  });

  // keyboard map: 1-7 tabs, / search, j/k or arrows navigate, Enter expand,
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
    if (/^[1-7]$/.test(e.key)) {
      const t = ["models","stats","discover","build","setup","context","help"][+e.key-1];
      const el = $(`.tab[data-tab="${t}"]`); if (el) el.click();
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
    // presets
    const pApply = e.target.closest("[data-preset-apply]");
    if (pApply) {
      e.stopPropagation();
      const pbind = e.target.closest("[data-preset-bind]");
      if (pbind) { await bindPreset(pbind.dataset.presetBindModel, pbind.dataset.presetBind); return; }
      const pdel = e.target.closest("[data-preset-del]");
      if (pdel) { await api("/api/presets/delete", {name: pdel.dataset.presetDel}); toast("Preset deleted","ok"); await refresh(true); return; }
      await applyPreset(pApply.dataset.presetModel, pApply.dataset.presetApply); return;
    }
    const pSave = e.target.closest("[data-preset-save]");
    if (pSave) { e.stopPropagation(); await savePresetFrom(pSave.dataset.presetSave); return; }
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
        } else { msg.className = "msg err"; msg.textContent = r.error || "failed"; }
      } else if (act === "load") {
        msg.className = "msg work"; msg.textContent = "loading (may take seconds)...";
        const r = await api("/api/load", {model: id});
        r.success ? toast("Loaded","ok") : (msg.className="msg err", msg.textContent=(r.error&&r.error.message)||"load failed");
      } else if (act === "unload") {
        msg.className = "msg work"; msg.textContent = "unloading...";
        await api("/api/unload", {model: id}); toast("Unloaded", "ok");
      } else if (act === "client") {
        await clientOpening; return;
      } else if (act === "unregister") {
        if (!confirm(`Unregister ${id}? Its GGUF file will remain on disk.`)) { btn.disabled = false; return; }
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
        if (!confirm(`Delete ${id} and its files from WSL? This cannot be undone.`)) return;
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
