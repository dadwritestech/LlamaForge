// Template doctor (tmpldoctor.py): the GGUF's embedded chat template against
// its repo's current upload and its base model. Markup only; models.js wires
// the buttons (Check -> /api/template/check, apply -> /api/template/apply).
import { esc } from "./core.js";

export const results = {};            // model id -> last check (survives re-renders)

const fileName = p => String(p || "").split(/[\\/]/).pop();
const btn = (id, source, label) =>
  `<button class="qbtn" data-tmpl-apply="${esc(id)}" data-tmpl-source="${esc(source)}">${esc(label)}</button>`;
const changes = d => d ? `<details class="tmpl-diff"><summary>What changed</summary><pre>${esc(d)}</pre></details>` : "";

export function templateBar(m) {
  if (!m || !m.in_ini) return "";
  const over = (m.settings || {})["chat-template-file"];
  return `<div class="tunebar">
    <span class="tunebar-label" title="Compares the chat template inside the GGUF with its repo's current upload and the base model's. Stale templates break tool calls and thinking.">Template</span>
    <button class="qbtn" data-tmpl-check="${esc(m.id)}">Check</button>
    ${over ? `<span class="ctxpill" title="${esc(over)}">using ${esc(fileName(over))}</span>${btn(m.id, "", "Use the file's own")}` : ""}
  </div><div data-tmpl-out>${templateMarkup(m.id, results[m.id])}</div>`;
}

export function templateMarkup(id, r) {
  if (!r) return "";
  if (r.checking) return `<div class="slotnote dim">checking ${esc(r.repo || "the repo")}...</div>`;
  const own = r.embedded ? "" : `<div class="slotnote dim">This GGUF has no chat template of its own; llama.cpp falls back to a generic one.</div>`;
  let body;
  if (r.verdict === "repo_newer")
    body = `<div class="slotnote warn">${esc(r.repo)} has a newer chat template than your file: the GGUF was re-uploaded with template fixes. Using it changes only the template, not the weights. ${btn(id, "repo", "Use the repo's template")}</div>${changes(r.repo_diff)}`;
  else if (r.verdict === "base_differs")
    body = `<div class="slotnote dim">Matches ${esc(r.repo)}, but differs from ${esc(r.base)}'s current template. GGUF makers often change templates on purpose, so try it only if tool calls or thinking misbehave. ${btn(id, "base", "Try the base template")}</div>${changes(r.base_diff)}`;
  else if (r.verdict === "same")
    body = `<div class="slotnote dim">Up to date: matches ${esc(r.repo)}${r.base ? ` and ${esc(r.base)}` : ""}.</div>`;
  else
    body = `<div class="slotnote dim">Nothing to compare against: ${esc(r.error || "no template found upstream")}.</div>`;
  return own + body;
}
