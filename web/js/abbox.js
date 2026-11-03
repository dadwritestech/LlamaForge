// The "Compare speed" result box, shared by the Try-a-PR card and the Engine
// card: llama-bench on your build vs another build, plus Copy as markdown.
// One comparison runs at a time (GET /api/prtry "ab"); each card shows only its
// own kind, a PR build's (ab.pr set) or an installed release's.
import { $, esc, toast } from "./core.js";
import { abView } from "./prview.js";

export const abIsPr = ab => !!(ab && ab.pr);

export function abHTML(ab) {
  const v = abView(ab);
  if (!v.phase && !v.rows.length) return "";
  const cls = ab.running ? "work" : ab.phase === "failed" ? "err" : "";
  const other = ab.label || (ab.pr ? `PR #${ab.pr}` : "other build");
  const row = r => `<tr><td>${esc(r.label)}</td><td>${esc(r.base)}</td><td>${esc(r.pr)}</td>
    <td>${esc(r.change)}${r.noise ? ` <span class="dim">(within noise)</span>` : ""}</td></tr>`;
  return `<div class="pr-ab">
    ${v.heading ? `<div class="pr-look-h">${esc(v.heading)}</div>` : ""}
    ${v.phase ? `<span class="msg ${cls}" data-ab-phase>${esc(v.phase)}</span>` : ""}
    ${v.rows.length ? `<table class="pr-ab-t"><thead><tr><th>llama-bench, tok/s</th><th>your build</th>
      <th>${esc(other)}</th><th>change</th></tr></thead><tbody>${v.rows.map(row).join("")}</tbody></table>
      <div class="actions" style="margin-top:4px"><button class="ghost" data-ab-md>Copy as markdown</button>
      <span class="note" style="margin:0">Same model file and GPU settings on both builds, every other model unloaded.</span></div>` : ""}
  </div>`;
}

export function abTick(el, ab) {
  const m = $("[data-ab-phase]", el);
  if (m) m.textContent = abView(ab).phase;
}

export function wireAb(el, ab) {
  const md = $("[data-ab-md]", el);
  if (md) md.onclick = () => navigator.clipboard.writeText(abView(ab).markdown)
    .then(() => toast("Markdown copied", "ok"), () => toast("Couldn't copy", "err"));
}

export const abAsk = (model, other) => `Run llama-bench on ${model} with your build, then with ${other}? `
  + "Every loaded model is unloaded for a few minutes and loaded back afterwards.";
