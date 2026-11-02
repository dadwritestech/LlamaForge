// Proving ground, in the Arena tab: the arena is taste (you vote), this is
// right or wrong. Fifteen short tasks with known answers (backend/proving.py);
// a script marks every reply. Model answers are plain text through esc().
import { $, esc, setHTML, api, toast } from "./core.js";
import { boardRows, progressLine, failures, CAT_LABELS } from "./provingview.js";
import { showModal } from "./models.js";
import { drawProving } from "./provingcard.js";

let timer = null;
let picked = null;      // models ticked for the next run; null = all

export async function loadProving() {
  const el = $("#proving");
  if (!el) return;
  clearTimeout(timer);
  const d = await api("/api/proving").catch(e => ({error: String(e.message || e)}));
  if (!el.isConnected) return;
  if (!d || d.error) { setHTML(el, `<div class="card"><h3>Checked tasks</h3><div class="note">${esc((d && d.error) || "couldn't reach the panel")}</div></div>`); return; }
  render(el, d);
  if (d.status && d.status.running) timer = setTimeout(loadProving, 2500);
}

function render(el, d) {
  const st = d.status || {};
  const models = d.models || [];
  const on = m => picked === null || picked.includes(m);
  const pick = models.length ? `<div class="pv-pick">${models.map(m => `<label><input type="checkbox" value="${esc(m)}"${on(m) ? " checked" : ""}${st.running ? " disabled" : ""}> ${esc(m)}</label>`).join("")}</div>`
    : `<div class="note">No chat models yet. Find one in Discover.</div>`;
  const prog = progressLine(st);
  const bar = st.running ? `<progress max="${esc(st.total || 1)}" value="${esc(st.done || 0)}" aria-label="Proving progress"></progress>` : "";
  const errs = Object.entries(st.errors || {}).map(([m, e]) => `<div class="note">${esc(m)} didn't load: ${esc(e)}</div>`).join("");
  setHTML(el, `<div class="card"><h3>Checked tasks</h3>
      <p class="note" style="margin-top:0">${esc(d.tasks ? d.tasks.length : 15)} short tasks with one right answer each:
        pulling facts out of text, arithmetic, JSON, following instructions, simple reasoning. A script marks every reply,
        one try each, so read it as a smoke test, not a benchmark. Models load one at a time and the ones you had up stay up.
        A run takes a few minutes per model; it's a good thing to start before bed.</p>
      ${pick}
      <div class="actions">${st.running ? `<button type="button" class="ghost" id="pv-cancel">Cancel</button>`
        : `<button type="button" id="pv-run"${models.length ? "" : " disabled"}>Run checks</button>`}
        ${Object.keys(d.scores || {}).length ? `<button type="button" class="ghost" id="pv-card">Share image</button>` : ""}
        <span class="note" style="margin:0" aria-live="polite">${esc(prog)}</span></div>
      ${bar}${errs}
      ${table(d)}
    </div>`);
  el.querySelectorAll(".pv-pick input").forEach(cb => cb.onchange = () => {
    picked = [...el.querySelectorAll(".pv-pick input:checked")].map(x => x.value);
  });
  const run = $("#pv-run", el);
  if (run) run.onclick = start;
  const card = $("#pv-card", el);
  if (card) card.onclick = () => openCard(d);
  const cancel = $("#pv-cancel", el);
  if (cancel) cancel.onclick = async () => {
    cancel.disabled = true;
    await api("/api/proving/cancel", {}).catch(() => null);
    loadProving();
  };
}

function table(d) {
  const rows = boardRows(d);
  if (!rows.length) return `<div class="note">No results yet.</div>`;
  const cats = d.categories || Object.keys(CAT_LABELS);
  const cell = ([p, t]) => t ? `<td class="${p === t ? "pv-full" : p ? "" : "pv-zero"}">${esc(p)}/${esc(t)}</td>` : `<td>-</td>`;
  const verdicts = rows.filter(r => r.verdict).map(r => `<li>${esc(r.verdict)}</li>`).join("");
  const misses = rows.map(r => {
    const f = failures(d, r.model);
    if (!f.length) return "";
    return `<details class="pv-miss"><summary>What ${esc(r.model)} got wrong (${esc(f.length)})</summary>
      ${f.map(x => `<div class="pv-row"><b>${esc(x.id)}</b>${x.cut ? ` <span class="note">cut off at the token limit</span>` : ""}
        <pre>${esc(x.answer || "(empty answer)")}</pre></div>`).join("")}</details>`;
  }).join("");
  return `<table class="arena-board pv-board"><thead><tr><th>Model</th><th>Passed</th>
      ${cats.map(c => `<th>${esc(CAT_LABELS[c] || c)}</th>`).join("")}<th>tok/s</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td>${esc(r.model)}${r.driver ? ` <span class="note">your main model</span>` : ""}</td>
        <td><b>${esc(r.passed)}/${esc(r.total)}</b></td>${r.cats.map(cell).join("")}<td>${r.tok_s == null ? "-" : esc(r.tok_s)}</td></tr>`).join("")}
    </tbody></table>
    ${verdicts ? `<ul class="pv-verdicts">${verdicts}</ul>` : d.driver ? "" : `<div class="note">Load a model in the main slot to compare the others against it.</div>`}
    ${misses}`;
}

async function start() {
  const el = $("#proving");
  const models = [...el.querySelectorAll(".pv-pick input:checked")].map(x => x.value);
  if (!models.length) { toast("Tick at least one model", "err"); return; }
  const d = await api("/api/proving/run", {models}).catch(e => ({error: String(e.message || e)}));
  if (!d || d.error) { toast((d && d.error) || "couldn't start", "err"); return; }
  toast(`Checking ${models.length} model${models.length === 1 ? "" : "s"}. You can leave this tab; it keeps going.`, "ok");
  loadProving();
}

// The board as a 1200x630 image (provingcard.js). Same dialog as the arena card.
async function openCard(d) {
  showModal("Share your checked tasks", `<canvas class="af-card" width="1200" height="630" aria-label="Checked tasks card"></canvas>
    <p class="af-q">Holds model names, which tasks each got right, the passed count, tok/s and your GPU names. No answers, prompts, paths or hosts.</p>
    <div class="af-act"><button type="button" class="qbtn load" data-pc-png>Download image</button>
      <button type="button" class="qbtn" data-pc-copyimg>Copy image</button></div>`);
  const dlg = document.querySelector("#modal-root dialog");
  const canvas = dlg && dlg.querySelector("canvas.af-card");
  if (!canvas) return;
  await drawProving(canvas, d);
  const blob = () => new Promise(r => canvas.toBlob(r, "image/png"));
  dlg.querySelector("[data-pc-png]").onclick = async () => {
    const url = URL.createObjectURL(await blob());
    const a = Object.assign(document.createElement("a"), {href: url, download: "llamaforge-checked-tasks.png"});
    a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  dlg.querySelector("[data-pc-copyimg]").onclick = async () => {
    try { await navigator.clipboard.write([new ClipboardItem({"image/png": blob()})]); toast("Image copied", "ok"); }
    catch { toast("This browser can't copy images; use Download", "err"); }
  };
}
