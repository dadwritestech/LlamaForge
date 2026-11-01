// Free Disk Space (reclaim.py): models not used in N days and byte-identical
// copies. Every Delete goes through askYes in setup.js; nothing here deletes.
import { esc, fmtAgo } from "./core.js";

export const gb = b => ((b || 0) / 1024 ** 3).toFixed(1) + " GB";

const row = (id, path, right) => `<div class="row"><div class="rhead" style="cursor:default;grid-template-columns:1fr auto auto">
      <span class="mid" title="${esc(path)}">${esc(id)}</span>${right}
      <button class="ghost" data-del="${esc(id)}" data-path="${esc(path)}">Delete</button></div></div>`;

export function reclaimMarkup(r) {
  if (!r) return "";
  const n = r.stale.length, parts = [];
  if (n) {
    const when = r.tracked_long_enough ? `not used in ${esc(r.days)} days`
      : `not used since usage tracking began (${esc(new Date(r.since * 1000).toISOString().slice(0, 10))})`;
    parts.push(`<div class="note">${esc(n)} model${n > 1 ? "s" : ""} ${when}: ${esc(gb(r.stale_bytes))}</div>
      <div class="list" style="margin-top:10px">${r.stale.map(m =>
        row(m.id, m.path, `<span class="ctxpill">${esc(gb(m.bytes))} · ${esc(fmtAgo(m.last_used))}</span>`)).join("")}</div>`);
  }
  for (const d of r.dups)
    parts.push(`<div class="note" style="margin-top:12px">${esc(d.copies)} copies of the same file: ${esc(gb(d.bytes * (d.copies - 1)))} to free</div>
      <div class="list" style="margin-top:6px">${d.files.map(f =>
        row(f.ids[0], f.path, `<span class="ctxpill" title="${esc(f.path)}">${esc(f.ids.length > 1 ? `also ${f.ids.slice(1).join(", ")} · ` : "")}${esc(gb(d.bytes))}</span>`)).join("")}</div>`);
  if (!parts.length)
    return `<div class="note">Every model was used in the last ${esc(r.days)} days, and there are no duplicate files.</div>`;
  return parts.join("");
}
