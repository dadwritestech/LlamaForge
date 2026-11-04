// Model X-ray: the GGUF taken apart, one bar per layer (backend/xray.py).
// Each bar is split into the kinds of tensors in that layer; what stays in
// system RAM under the model's settings is hatched. The GPU-layers slider asks
// the backend "what if" with ?ngl=, so the placement rule lives in one place
// (footprint.placer) and nothing is written to models.ini.
import { $, esc, setHTML, api, toast } from "./core.js";
import { showModal } from "./models.js";

export const KINDS = [
  ["embedding", "embedding"], ["attention", "attention"], ["ssm", "SSM / linear attn"],
  ["ffn", "FFN"], ["experts", "MoE experts"], ["shared", "shared expert"],
  ["per-layer", "per-layer input"], ["mtp", "MTP head"], ["norm", "norms"],
  ["output", "output"], ["other", "other"],
];

export const gb = b => b >= 1024 ** 3 ? (b / 1024 ** 3).toFixed(2) + " GB"
                                       : Math.max(0, Math.round(b / 1024 ** 2)) + " MB";

/** The slider's range: 0 .. every block plus the output layer. A configured
 *  99 / 999 / "all" means the top. */
export const clampNgl = (n, nLayer) => Math.max(0, Math.min(Number(n) || 0, nLayer + 1));

/** [{kind, pct, ram}] for one row, pct of the widest row, GPU part then RAM
 *  part of each kind, in KINDS order; slivers under 0.05% are dropped. */
export function segments(row, widest) {
  const out = [];
  for (const [k] of KINDS) {
    const all = (row.parts || {})[k] || 0;
    if (!all || !widest) continue;
    const ram = (row.ram_parts || {})[k] || 0;
    for (const [bytes, isRam] of [[all - ram, false], [ram, true]]) {
      const pct = bytes / widest * 100;
      if (pct >= 0.05) out.push({kind: k, pct: +pct.toFixed(2), ram: isRam});
    }
  }
  return out;
}

export function nglText(ngl, nLayer) {
  if (ngl <= 0) return "none: every layer runs from system RAM";
  if (ngl > nLayer) return `all ${nLayer} blocks + output`;
  return `last ${ngl} of ${nLayer} blocks`;
}

export function body(v) {
  const widest = Math.max(1, ...v.rows.map(r => r.gpu + r.ram));
  const used = new Set(v.rows.flatMap(r => Object.keys(r.parts)));
  const legend = KINDS.filter(([k]) => used.has(k))
    .map(([k, label]) => `<span><i class="xr-k-${esc(k)}"></i>${esc(label)}</span>`).join("")
    + `<span><i class="xr-k-ram"></i>in system RAM</span>`;
  const types = v.types.slice(0, 8).map(t =>
    `<span class="xr-type" title="${esc(gb(t.bytes))}">${esc(t.type)} <b>${esc(Math.round(t.share * 100))}%</b></span>`).join("");
  const rows = v.rows.map(r => {
    const tip = `${r.label} · ${r.type} · ${gb(r.gpu + r.ram)}` + (r.ram ? ` · ${gb(r.ram)} in RAM` : " · on GPU");
    const segs = segments(r, widest).map(s =>
      `<i class="xr-k-${esc(s.kind)}${s.ram ? " ram" : ""}" style="width:${s.pct}%"></i>`).join("");
    const n = /^blk (\d+)$/.exec(r.label);
    const tick = !n || Number(n[1]) % 5 === 0 ? " tick" : "";
    return `<div class="xr-row${tick}" title="${esc(tip)}"><span class="xr-l">${esc(r.label)}</span><span class="xr-bar">${segs}</span></div>`;
  }).join("");
  return `<div class="xr-sum"><b>${esc(gb(v.total))}</b> of weights · <span class="xr-gpu">${esc(gb(v.gpu))} on GPU</span>
      · <span class="xr-ram">${esc(gb(v.ram))} in system RAM</span>${v.tied_copy
        ? ` <span class="xr-tied" title="The output layer is tied to the embedding, so llama.cpp keeps a GPU copy of it">+ ${esc(gb(v.tied_copy))} GPU copy of the embedding (tied output)</span>` : ""}</div>
    <div class="xr-types">${types}</div>
    <div class="xr-legend">${legend}</div>
    <div class="xr-stack" data-xr-stack>${rows}</div>
    ${v.confident ? "" : `<p class="slotnote warn">An override-tensor pattern didn't parse, so the RAM / GPU split is a guess.</p>`}`;
}

let seq = 0;

export async function openXray(id) {
  let v;
  try { v = await api("/api/model/xray?model=" + encodeURIComponent(id)); }
  catch (e) { toast("Couldn't X-ray " + id + ": " + (e.message || e), "err"); return; }
  const top = v.n_layer + 1, conf = clampNgl(v.configured_ngl, v.n_layer);
  showModal(`X-ray: ${id}`, `
    <p class="af-q">${esc(v.arch)} · ${esc(v.n_layer)} blocks, read from the GGUF header. Nothing is loaded.</p>
    <div class="xr-ngl"><label for="xr-ngl">GPU layers</label>
      <input type="range" id="xr-ngl" min="0" max="${esc(top)}" step="1" value="${esc(conf)}">
      <output for="xr-ngl" data-xr-ngl>${esc(nglText(conf, v.n_layer))}</output>
      <button type="button" class="qbtn" data-xr-reset hidden>Back to ${esc(conf)}</button></div>
    <p class="af-q dim xr-what">Drag to see what moves. A what-if only: nothing is saved.</p>
    <div data-xr-body>${body(v)}</div>`);
  const range = $("#xr-ngl"), out = $("[data-xr-ngl]"), reset = $("[data-xr-reset]"), box = $("[data-xr-body]");
  if (!range || !box) return;
  let t = 0;
  const update = () => {
    const n = clampNgl(range.value, v.n_layer);
    out.textContent = nglText(n, v.n_layer);
    reset.hidden = n === conf;
    clearTimeout(t);
    t = setTimeout(async () => {
      const mine = ++seq;
      try {
        const w = await api(`/api/model/xray?model=${encodeURIComponent(id)}&ngl=${n}`);
        if (mine === seq && box.isConnected) setHTML(box, body(w));
      } catch (e) { toast(String(e.message || e), "err"); }
    }, 90);
  };
  range.oninput = update;
  reset.onclick = () => { range.value = String(conf); update(); };
}
