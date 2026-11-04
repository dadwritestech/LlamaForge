// Batches card on Stats: the local OpenAI Batch API (batchrun.py). Hand it a
// .jsonl of chat requests and it works through them on your GPU, now or once
// the router has gone quiet. Scripts use /v1/files + /v1/batches directly;
// this card uses /api/batch/* so it needs no router key.
import { $, esc, setHTML, api, toast, fmtAgo, askYes } from "./core.js";

const OPEN = new Set(["validating", "in_progress", "finalizing", "cancelling"]);

export function progress(b) {
  const c = (b && b.request_counts) || {};
  const total = c.total || 0, done = (c.completed || 0) + (c.failed || 0);
  return {done, total, failed: c.failed || 0};
}

/** Answered and failed shares of the bar, in whole percent. */
export function barWidths(b) {
  const {done, total, failed} = progress(b);
  if (!total) return [0, 0];
  const bad = Math.round(100 * failed / total);
  return [Math.min(100 - bad, Math.round(100 * (done - failed) / total)), bad];
}

/** What the batch is doing, in words. */
export function statusText(b, quietMinutes = 20) {
  const {done, total, failed} = progress(b);
  const fails = failed ? `, ${failed} failed` : "";
  switch (b.status) {
    case "validating":
      return b.when === "idle" ? `waiting for ${quietMinutes} quiet minutes` : "starting";
    case "in_progress": return `${done} of ${total}${fails}`;
    case "finalizing": return "writing results";
    case "cancelling": return "cancelling after this request";
    case "completed": return `done: ${total - failed} answered${fails}`;
    case "failed": {
      const e = b.errors && b.errors.data && b.errors.data[0];
      return `failed${e ? ": " + e.message : ""}`;
    }
    case "expired": return `expired after 24 h at ${done} of ${total}`;
    case "cancelled": return `cancelled at ${done} of ${total}`;
    default: return String(b.status || "");
  }
}

export function pySnippet(origin) {
  return `from openai import OpenAI
client = OpenAI(base_url="${origin}/v1", api_key="none")   # or your router key

f = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                          completion_window="24h")
# later
b = client.batches.retrieve(b.id)
if b.status == "completed":
    print(client.files.content(b.output_file_id).text)`;
}

function row(b, quiet) {
  const {done, total} = progress(b);
  const [ok, bad] = barWidths(b);
  const open = OPEN.has(b.status);
  const led = b.status === "in_progress" ? "loaded" : b.status === "failed" ? "failed" : open ? "sleeping" : "";
  const dl = (id, label, name) => id
    ? `<a class="chip" href="/api/batch/file?id=${encodeURIComponent(id)}" download="${esc(name)}">${label}</a>` : "";
  const stem = (b.input_filename || b.id).replace(/\.jsonl$/i, "");
  return `<div class="row" data-batch="${esc(b.id)}"><div class="rhead" style="cursor:default;grid-template-columns:9px 1fr auto auto">
      <span class="led ${led}"></span>
      <span class="mid" title="${esc(b.id)}">${esc(b.input_filename || b.id)}</span>
      <span class="ctxpill">${esc(b.model || "")}</span>
      <span class="stat">${esc(fmtAgo(b.created_at))}</span>
    </div>
    <div class="batch-bar" role="progressbar" aria-valuemin="0" aria-valuemax="${esc(total)}" aria-valuenow="${esc(done)}" aria-label="${esc(done)} of ${esc(total)} requests"><i style="width:${ok}%"></i><b style="width:${bad}%"></b></div>
    <div class="batch-foot"><span class="batch-status">${esc(statusText(b, quiet))}</span>
      <span class="batch-acts">
        ${dl(b.output_file_id, "Results", stem + "_results.jsonl")}
        ${dl(b.error_file_id, "Errors", stem + "_errors.jsonl")}
        ${open && b.status !== "cancelling" ? `<button data-batchcancel="${esc(b.id)}">Cancel</button>` : ""}
        ${open ? "" : `<button class="danger" data-batchremove="${esc(b.id)}" title="delete this batch and its files">Remove</button>`}
      </span></div>
  </div>`;
}

export function batchesMarkup(st, origin) {
  if (!st || !Array.isArray(st.batches)) return "";
  const quiet = st.quiet_minutes || 20;
  return `<div class="card" id="stats-batches"><h3>Batches
      <span class="batch-new"><button data-batchrun="idle" title="starts once the router has been quiet ${esc(quiet)} minutes">Run a file tonight&hellip;</button>
      <button data-batchrun="now">Run now&hellip;</button></span></h3>
    <div class="note" style="margin-top:0">OpenAI's Batch API, on your own GPU. Give it a <code>.jsonl</code> of chat requests (one model per file) and it answers them one by one into a results file. <b>Tonight</b> waits until nobody has used the router for ${esc(quiet)} minutes; anything that needs the GPUs (a speed test, Auto-Forge) pauses it.</div>
    ${st.batches.length ? `<div class="list" style="margin-top:12px">${st.batches.map(b => row(b, quiet)).join("")}</div>`
      : `<div class="note">No batches yet.</div>`}
    <details class="batch-py"><summary>Use it from a script</summary>
      <pre>${esc(pySnippet(origin))}</pre>
      <div class="note">Each line of <code>requests.jsonl</code>: <code>${esc('{"custom_id": "q1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "…", "messages": […]}}')}</code></div>
    </details>
  </div>`;
}

let last = null, picking = "now", input = null, busy = false;

export async function refreshBatches() {
  const st = await api("/api/batch/state").catch(() => null);
  if (!st || st.error) return;
  last = st;
  const box = $("#stats-batches-wrap");
  if (box) {
    const openPy = !!$(".batch-py", box)?.open;
    setHTML(box, batchesMarkup(st, location.origin));
    if (openPy) $(".batch-py", box).open = true;
  }
}

/** For loadStats: the card's wrapper, drawn from the last fetch so a re-render doesn't flicker. */
export function batchesCached(openPy = false) {
  const card = last ? batchesMarkup(last, location.origin)
    : `<div class="card" id="stats-batches"><h3>Batches</h3><div class="note">Loading&hellip;</div></div>`;
  return `<div id="stats-batches-wrap">${openPy ? card.replace('<details class="batch-py">', '<details class="batch-py" open>') : card}</div>`;
}

async function runFile(file) {
  if (!file || busy) return;
  busy = true;
  try {
    const content = await file.text();
    const r = await api("/api/batch/run", {filename: file.name, content, when: picking});
    if (r.error) { toast(r.error, "err"); return; }
    toast(picking === "idle" ? `Queued ${r.request_counts.total} requests for tonight` : `Running ${r.request_counts.total} requests`, "ok");
    refreshBatches();
  } finally { busy = false; }
}

export function initBatches(view) {
  // The file picker lives outside the Stats view, which re-renders every few seconds.
  input = document.createElement("input");
  Object.assign(input, {type: "file", accept: ".jsonl,application/jsonl,text/plain", hidden: true});
  input.addEventListener("change", () => { runFile(input.files[0]); input.value = ""; });
  document.body.appendChild(input);
  view.addEventListener("click", async e => {
    const run = e.target.closest("[data-batchrun]");
    if (run) { picking = run.dataset.batchrun; input.click(); return; }
    const c = e.target.closest("[data-batchcancel]");
    if (c) {
      const r = await api("/api/batch/cancel", {id: c.dataset.batchcancel});
      if (r.error) toast(r.error, "err");
      refreshBatches();
      return;
    }
    const rm = e.target.closest("[data-batchremove]");
    if (rm) {
      if (!(await askYes("The batch, its input file and its results are deleted from this machine.",
          {title: "Remove this batch", ok: "Remove", danger: true}))) return;
      const r = await api("/api/batch/remove", {id: rm.dataset.batchremove});
      if (r.error) toast(r.error, "err");
      refreshBatches();
    }
  });
}
