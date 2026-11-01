// Recommended samplers (samplers.py): the temperature / top-p / top-k the
// model's makers publish in generation_config.json, next to what models.ini
// sets now. Markup only; models.js wires Check (/api/samplers) and Apply (/api/save).
import { esc } from "./core.js";

export const results = {};            // model id -> last check (survives re-renders)

export function samplerBar(m) {
  if (!m || !m.in_ini) return "";
  return `<div class="tunebar">
    <span class="tunebar-label" title="Reads the sampler settings the model's makers publish (generation_config.json, from the GGUF repo or its base model) and compares them with yours.">Samplers</span>
    <button class="qbtn" data-smp-check="${esc(m.id)}">Recommended</button>
  </div><div data-smp-out>${samplerMarkup(m.id, results[m.id])}</div>`;
}

// knobs whose recommended value differs from what the model runs with now
// (as numbers: models.ini's "1.0" is the same as "1")
const same = (a, b) => a != null && a !== "" && (String(a) === b || Number(a) === Number(b));
export function changed(r) {
  const cur = (r && r.current) || {};
  return Object.keys((r && r.values) || {}).filter(k => !same(cur[k], r.values[k]));
}

export function samplerMarkup(id, r) {
  if (!r) return "";
  if (r.checking) return `<div class="slotnote dim">reading generation_config.json...</div>`;
  const vals = r.values || {};
  if (!Object.keys(vals).length) {
    const where = (r.checked || []).length ? ` (checked ${r.checked.map(esc).join(", ")})` : "";
    return `<div class="slotnote dim">No published samplers: ${esc(r.error || "the repo has no generation_config.json with sampler values")}${where}. llama.cpp's defaults apply.</div>`;
  }
  const cur = r.current || {}, inh = r.inherited || [];
  const diff = changed(r);
  const cells = Object.entries(vals).map(([k, v]) => {
    const now = cur[k] == null ? "default" : `${cur[k]}${inh.includes(k) ? " (from [*])" : ""}`;
    return `<span class="ctxpill${diff.includes(k) ? "" : " same"}" title="yours: ${esc(now)}">${esc(k)} ${esc(v)}</span>`;
  }).join(" ");
  const src = `<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.source)}</a>`;
  const act = diff.length
    ? ` ${diff.length} differ from yours. <button class="qbtn" data-smp-apply="${esc(id)}">Use these</button>`
    : " Yours already match.";
  return `<div class="slotnote dim">From ${src}: ${cells}.${act}</div>`;
}
