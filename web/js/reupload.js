// Re-upload watch (reupload.py): has the HF repo a model came from replaced
// its file since you downloaded it? Markup only; models.js loads /api/reupload
// (refreshing it when over a day old) and wires Re-download and Check now.
import { esc, fmtAgo } from "./core.js";

export const S = { data: null, busy: false };

const rec = id => (S.data && S.data.models && S.data.models[id]) || null;
const day = iso => String(iso || "").slice(0, 10);
const said = r => r.title ? ` ("${esc(r.title)}")` : "";

export function chip(m) {
  const r = rec(m.id);
  if (!r || r.verdict !== "reuploaded") return "";
  return `<span class="tag reup" title="${esc(r.repo)} replaced this file on ${esc(day(r.date))}. Open the model to re-download it.">re-uploaded</span>`;
}
export const chipSig = m => (rec(m.id) || {}).verdict || "";

export function noteMarkup(m) {
  const r = rec(m.id);
  if (!r) return "";
  const when = S.data.checked_at ? `checked ${esc(fmtAgo(S.data.checked_at))}` : "";
  const again = `<button class="qbtn" data-reup-check ${S.busy ? "disabled" : ""}>${S.busy ? "checking..." : "Check now"}</button>`;
  const repo = `<a href="https://huggingface.co/${esc(r.repo)}" target="_blank" rel="noopener">${esc(r.repo)}</a>`;
  if (r.verdict === "reuploaded") {
    const fix = r.hf_cache
      ? ` Update it with <code>hf download ${esc(r.repo)} ${esc(r.path)}</code>.`
      : ` <button class="qbtn" data-reup-fetch="${esc(m.id)}">Re-download</button>`;
    return `<div class="slotnote warn">${repo} replaced this file on ${esc(day(r.date))}${said(r)}; your copy is the old one. Re-uploads usually fix templates or broken quants.${fix} <span class="dim">${when}</span></div>`;
  }
  if (r.verdict === "changed")
    return `<div class="slotnote dim">${repo} changed this file on ${esc(day(r.date))}${said(r)}, after your copy, at the same size: often metadata only. The Template row's Check shows whether the chat template changed. <button class="qbtn" data-reup-fetch="${esc(m.id)}">Re-download anyway</button> ${when}</div>`;
  if (r.verdict === "gone")
    return `<div class="slotnote dim">${repo} no longer has this file (renamed or removed). ${when} ${again}</div>`;
  return "";
}
