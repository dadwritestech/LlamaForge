// Auto-Forge: try placements, KV types and drafters for one model on this
// machine, measure each honestly, and offer the winner as a models.ini edit.
//
// Lives in the open model row (models.js puts markup(id) in .ed-forge). The
// backend does the work (/api/forge/*); this file draws its status and polls
// while a run is going. Every dynamic value goes through esc().
import { esc, api, toast, askYes } from "./core.js";

const POLL_MS = 2000;
const STAGE = { fit: "trying every candidate", speed: "re-measuring the finalists on fresh prompts",
  sanity: "checking drafters give the same answers", done: "done", cancelled: "cancelled",
  error: "stopped on an error", idle: "" };
const CTX = [8192, 16384, 32768, 65536, 131072];

const cache = new Map();            // model id -> last /api/forge/status
const inflight = new Set();
let timer = null;
let redraw = () => {};

/** cb(): status changed, re-render the open row (models.js passes renderModels). */
export function initAutoForge(cb) { redraw = cb; }

const num = v => (v == null || !Number.isFinite(+v)) ? "–" : (+v).toFixed(1);
const int = v => (v == null || !Number.isFinite(+v)) ? "–" : String(Math.round(+v));
const gb = mib => (mib == null) ? "–" : (mib / 1024).toFixed(1);
export function pct(v) {
  if (v == null || !Number.isFinite(+v)) return "";
  const n = Math.round(+v);
  return n === 0 ? "same" : (n > 0 ? `+${n}%` : `${n}%`);
}

export function setupMarkup(id, req = {}) {
  const intent = req.intent || "coding", ctx = +req.ctx || 32768;
  const opt = (v, label, cur) => `<option value="${esc(v)}" ${String(v) === String(cur) ? "selected" : ""}>${esc(label)}</option>`;
  return `<div class="af-setup">
    <span class="af-label" title="Loads this model in a private sandbox once per candidate placement, KV type and drafter, measures code and prose speed on fresh prompts, and keeps your answers unchanged.">Auto-Forge</span>
    <select data-af-intent aria-label="what you use this model for">
      ${opt("coding", "Coding", intent)}${opt("chat", "Chat", intent)}${opt("long", "Long documents", intent)}${opt("agent", "Agents", intent)}
    </select>
    <select data-af-ctx aria-label="context to keep">
      ${CTX.map(c => opt(c, `${c / 1024}k context`, ctx)).join("")}
    </select>
    <select data-af-time aria-label="how long to search">
      ${opt("quick", "Quick", req.time)}${opt("thorough", "Thorough", req.time)}
    </select>
    <label class="af-loss" title="Also try 8-bit / 4-bit KV cache. Faster or longer context, answers can change slightly."><input type="checkbox" data-af-loss ${req.allow_loss ? "checked" : ""}> allow lossy KV</label>
    <label class="af-loss" title="Also try without the vision projector (mmproj). Frees about 1 GB on the first GPU, which can make room for a drafter; the model can no longer read images. Applying such a result turns vision off with the usual switch, so you can turn it back on."><input type="checkbox" data-af-novision ${req.drop_vision ? "checked" : ""}> try without vision</label>
    <button type="button" class="qbtn" data-af-start="${esc(id)}">Forge</button>
  </div>`;
}

function rowMarkup(r) {
  const tags = [r.finalist && r.id !== "current" ? `<span class="tag">finalist</span>` : "",
    r.lossy ? `<span class="tag warn" title="lossy KV cache: answers can differ slightly">lossy</span>` : "",
    r.novision ? `<span class="tag warn" title="without the vision projector: can't read images">no vision</span>` : "",
    r.sanity && r.sanity.ok === false ? `<span class="tag bad" title="${esc(r.sanity.reason || "")}">drafter diverged</span>` : ""].join("");
  const per = (r.vram_per_gpu || []).map(gb).join(" + ");
  return `<tr${r.id === "current" ? ` class="af-cur"` : ""}>
    <td>${esc(r.label || r.id)}${tags}</td>
    <td class="n">${num(r.code)}</td><td class="n">${num(r.prose)}</td><td class="n">${int(r.prefill)}</td>
    <td class="n" title="${esc(per)} GB">${gb(r.vram_mib)}</td></tr>`;
}

export function boardMarkup(st) {
  const rows = st.leaderboard || [];
  const out = st.failed || [], rej = st.rejected || [];
  const table = rows.length ? `<table class="af-board">
    <thead><tr><th>setup</th><th class="n">code tk/s</th><th class="n">prose tk/s</th><th class="n">prefill tk/s</th><th class="n">VRAM GB</th></tr></thead>
    <tbody>${rows.map(r => rowMarkup(r)).join("")}</tbody></table>` : "";
  const why = list => list.map(f => `<li>${esc(f.label || f.id)}: ${esc(f.reason || "")}</li>`).join("");
  const miss = (out.length || rej.length) ? `<details class="af-miss"><summary>${out.length + rej.length} tried or skipped, didn't make it</summary>
    <ul>${why(out)}${why(rej)}</ul></details>` : "";
  return table + miss;
}

export function resultMarkup(id, st) {
  const res = st.result;
  if (!res) return "";
  if (!res.winner) return `<div class="af-result bad">No candidate ran on this machine${res.reason ? `: ${esc(res.reason)}` : ""}.</div>`;
  const w = res.winner, vs = res.vs_current || {}, cur = res.current;
  const gain = cur ? `<div class="af-vs">Your current setup: ${num(cur.code)} code / ${num(cur.prose)} prose tk/s.
      Forged: <b>${num(w.code)} code</b> (${esc(pct(vs.code))}) / <b>${num(w.prose)} prose</b> (${esc(pct(vs.prose))}).</div>`
    : `<div class="af-vs">Forged: <b>${num(w.code)} code</b> / <b>${num(w.prose)} prose</b> tk/s (your current setup didn't load to compare).</div>`;
  const quality = res.unverified_quality
    ? `<div class="af-q warn">Uses a lossy KV cache: faster, but answers can differ slightly from your current setup.</div>`
    : `<div class="af-q">Same answers: lossless settings${w.spec ? ", drafter checked token-for-token against the plain model" : ""}.</div>`;
  const act = st.applied
    ? `<button type="button" class="qbtn" data-af-undo="${esc(id)}">Undo</button><span class="af-note">applied to models.ini</span>`
    : (w.id === "current" ? `<span class="af-note">your current setup is already the best</span>`
      : `<button type="button" class="qbtn load" data-af-apply="${esc(id)}">Apply</button>`);
  const vision = res.drops_vision
    ? `<div class="af-q warn">Turns vision off: the model can't read images until you turn it back on (Undo or the vision switch).</div>` : "";
  return `<div class="af-result"><div class="af-win">${esc(w.label || w.id)}</div>${gain}${quality}${vision}<div class="af-act">${act}</div></div>`;
}

export function liveMarkup(id, st) {
  const p = st.progress || {};
  const stage = STAGE[st.stage] ?? st.stage;
  const bar = p.total ? Math.round(100 * p.done / p.total) : 0;
  const head = st.running
    ? `<div class="af-live"><span class="af-stage">${esc(stage)}${st.loading ? ` · loading ${esc(st.loading)}` : ""}</span>
        <span class="lbar" role="progressbar" aria-label="Auto-Forge progress" aria-valuenow="${bar}" aria-valuemin="0" aria-valuemax="100"><i style="width:${bar}%"></i></span>
        <span class="af-count">${esc(p.done ?? 0)}/${esc(p.total ?? 0)}</span>
        <button type="button" class="qbtn stop" data-af-cancel="${esc(id)}">Cancel</button></div>`
    : (st.error ? `<div class="af-result bad">${esc(stage)}: ${esc(st.error)}</div>` : "");
  return head + resultMarkup(id, st) + boardMarkup(st);
}

/** The whole region for one model, from the cached status. */
export function markup(id) {
  const st = cache.get(id);
  if (!st) { refresh(id); return `<div class="af-setup"><span class="af-label">Auto-Forge</span> …</div>`; }
  const mine = !st.model || st.model === id;
  const busyElsewhere = st.running && !mine;
  const setup = st.running ? "" : setupMarkup(id, st.req || {});
  if (busyElsewhere) return `<div class="af-setup"><span class="af-label">Auto-Forge</span> busy with ${esc(st.model)}</div>`;
  return `<div class="af">${setup}${st.stage && st.stage !== "idle" ? liveMarkup(id, st) : ""}</div>`;
}

export async function refresh(id) {
  if (inflight.has(id)) return;
  inflight.add(id);
  try {
    const st = await api(`/api/forge/status?model=${encodeURIComponent(id)}`);
    if (st && !st.error) { cache.set(id, st); redraw(); }
    schedule(id, st && st.running);
  } catch (e) {
    /* router or panel restarting: try again on the next open */
  } finally { inflight.delete(id); }
}

function schedule(id, running) {
  clearTimeout(timer);
  timer = running ? setTimeout(() => refresh(id), POLL_MS) : null;
}

async function start(row, id) {
  const req = {model: id, intent: row.querySelector("[data-af-intent]")?.value,
    ctx: +(row.querySelector("[data-af-ctx]")?.value || 32768),
    time: row.querySelector("[data-af-time]")?.value,
    allow_loss: !!row.querySelector("[data-af-loss]")?.checked,
    drop_vision: !!row.querySelector("[data-af-novision]")?.checked};
  const ok = await askYes(`Auto-Forge unloads every model and uses all GPUs until it finishes `
    + `(${req.time === "thorough" ? "about an hour" : "15-30 minutes"}). Your models load back afterwards. Start?`);
  if (!ok) return;
  const r = await api("/api/forge/start", req).catch(e => ({error: String(e.message || e)}));
  if (!r || r.error) { toast((r && r.error) || "Auto-Forge didn't start", "err"); return; }
  cache.set(id, {...(cache.get(id) || {}), model: id, running: true, stage: "fit", req});
  redraw(); refresh(id);
}

async function post(path, id, okMsg) {
  const r = await api(path, {model: id}).catch(e => ({error: String(e.message || e)}));
  if (!r || r.error) toast((r && r.error) || "failed", "err");
  else if (okMsg) toast(okMsg, "ok");
  refresh(id);
}

/** Click handling for the region; models.js calls this from its delegated listener.
 *  True when the click was ours. */
export function handleClick(e) {
  const t = e.target.closest("[data-af-start],[data-af-cancel],[data-af-apply],[data-af-undo]");
  if (!t) return false;
  e.stopPropagation();
  const row = t.closest(".row");
  if (t.dataset.afStart) start(row, t.dataset.afStart);
  else if (t.dataset.afCancel) post("/api/forge/cancel", t.dataset.afCancel, "Cancelling after this candidate");
  else if (t.dataset.afApply) post("/api/forge/apply", t.dataset.afApply, "Applied; the model reloads with it");
  else if (t.dataset.afUndo) post("/api/forge/undo", t.dataset.afUndo, "Restored your previous settings");
  return true;
}
