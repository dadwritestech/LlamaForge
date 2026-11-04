// Model X-ray: the GGUF taken apart, one bar per layer (backend/xray.py).
// Each bar is split into the kinds of tensors in that layer; what stays in
// system RAM under the model's settings is hatched. The GPU-layers slider asks
// the backend "what if" with ?ngl=, so the placement rule lives in one place
// (footprint.placer) and nothing is written to models.ini.
import { $, esc, setHTML, api, toast, askYes } from "./core.js";
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

/** The --n-cpu-moe slider's label: the first k blocks keep their experts in RAM. */
export function moeText(k, nLayer) {
  if (k <= 0) return "none: every expert on the GPU";
  if (k >= nLayer) return `all ${nLayer} blocks`;
  return `first ${k} of ${nLayer} blocks`;
}

function stack(v, widest) {
  return v.rows.map(r => {
    const tip = `${r.label} · ${r.type} · ${gb(r.gpu + r.ram)}` + (r.ram ? ` · ${gb(r.ram)} in RAM` : " · on GPU");
    const segs = segments(r, widest).map(s =>
      `<i class="xr-k-${esc(s.kind)}${s.ram ? " ram" : ""}" style="width:${s.pct}%"></i>`).join("");
    const n = /^blk (\d+)$/.exec(r.label);
    const tick = !n || Number(n[1]) % 5 === 0 ? " tick" : "";
    return `<div class="xr-row${tick}" title="${esc(tip)}"><span class="xr-l">${esc(r.label)}</span><span class="xr-bar">${segs}</span></div>`;
  }).join("");
}

const widestOf = (...vs) => Math.max(1, ...vs.flatMap(v => v.rows.map(r => r.gpu + r.ram)));

function legendHTML(...vs) {
  const used = new Set(vs.flatMap(v => v.rows.flatMap(r => Object.keys(r.parts))));
  return `<div class="xr-legend">${KINDS.filter(([k]) => used.has(k))
    .map(([k, label]) => `<span><i class="xr-k-${esc(k)}"></i>${esc(label)}</span>`).join("")}<span><i class="xr-k-ram"></i>in system RAM</span></div>`;
}

const typesHTML = (v, n = 8) => `<div class="xr-types">${v.types.slice(0, n).map(t =>
  `<span class="xr-type" title="${esc(gb(t.bytes))}">${esc(t.type)} <b>${esc(Math.round(t.share * 100))}%</b></span>`).join("")}</div>`;

const guess = v => v.confident ? "" : `<p class="slotnote warn">An override-tensor pattern didn't parse, so the RAM / GPU split is a guess.</p>`;

/** For a file you don't have yet (Discover): does it fit your VRAM as shown?
 *  "" for a model you have, whose view carries no vram. */
export function fitText(v) {
  if (!("vram" in v)) return "";
  if (!v.vram) return v.model ? "" : "No GPU detected, so this shows every layer on the GPU: compare the GPU figure with your card's VRAM.";
  const used = v.gpu + (v.tied_copy || 0);
  return used <= v.budget
    ? `Fits your ${gb(v.vram)} of VRAM: ${gb(used)} of weights on the GPU, ${gb(v.vram - used)} left for the KV cache and buffers.`
    : `Too much for your ${gb(v.vram)} of VRAM: ${gb(used)} of weights on the GPU, ${gb(used - v.budget)} over what leaves room for the KV cache.`;
}

const fitHTML = v => {
  const t = fitText(v);
  return t ? `<p class="xr-fit ${v.vram && v.gpu + (v.tied_copy || 0) > v.budget ? "over" : ""}">${esc(t)}</p>` : "";
};

export function body(v) {
  return `<div class="xr-sum"><b>${esc(gb(v.total))}</b> of weights · <span class="xr-gpu">${esc(gb(v.gpu))} on GPU</span>
      · <span class="xr-ram">${esc(gb(v.ram))} in system RAM</span>${v.tied_copy
        ? ` <span class="xr-tied" title="The output layer is tied to the embedding, so llama.cpp keeps a GPU copy of it">+ ${esc(gb(v.tied_copy))} GPU copy of the embedding (tied output)</span>` : ""}</div>
    ${fitHTML(v)}
    ${typesHTML(v)}
    ${legendHTML(v)}
    <div class="xr-stack" data-xr-stack>${stack(v, widestOf(v))}</div>
    ${guess(v)}`;
}

/** "Q4_K_M" from "Q4_K_M/gemma-4-26b-Q4_K_M-00001-of-00002.gguf"; the file stem when no quant tag is in it. */
export function quantName(path) {
  const stem = String(path || "").split("/").pop().replace(/(-\d{5}-of-\d{5})?\.gguf$/i, "");
  const m = /(?:^|[-_.])((?:UD-)?(?:[IPT]{0,2}Q\d\w*|MXFP4\w*|BF16|F16|F32|FP8\w*))$/i.exec(stem);
  return m ? m[1] : stem;
}

const signed = b => (b < 0 ? "-" : "+") + gb(Math.abs(b));

/** What the second quant costs against the first, under the same settings. */
export function compareText(a, b) {
  return `${quantName(b.file)} vs ${quantName(a.file)}: ${signed(b.total - a.total)} of weights, `
    + `${signed(b.gpu - a.gpu)} on GPU, ${signed(b.ram - a.ram)} in system RAM`;
}

/** Both quants side by side, bars on one scale so the bigger file looks bigger. */
export function compareBody(v) {
  const w = v.vs, widest = widestOf(v, w);
  const half = (x, tag) => `<div class="xr-half"><div class="xr-file" title="${esc(x.file)}">${esc(tag)} <b>${esc(quantName(x.file))}</b></div>
      <div class="xr-sum"><b>${esc(gb(x.total))}</b> · <span class="xr-gpu">${esc(gb(x.gpu))} GPU</span>
        · <span class="xr-ram">${esc(gb(x.ram))} RAM</span></div>
      ${typesHTML(x, 4)}
      <div class="xr-stack">${stack(x, widest)}</div></div>`;
  return `<div class="xr-delta">${esc(compareText(v, w))}</div>
    ${legendHTML(v, w)}
    <div class="xr-two">${half(v, "this file")}${half(w, "vs")}</div>
    <p class="af-q dim">Same settings and GPU layers for both. The other file's header was read from Hugging Face; none of its weights were downloaded.</p>
    ${guess(v)}`;
}

let seq = 0;

/** The question before Apply writes a placement into a model's settings. */
export function applyAsk(id, flags) {
  return `Save ${flags} as the GPU layers${/n-cpu-moe/.test(flags) ? " and experts on CPU" : ""} of ${id}? `
    + "It replaces any GPU layers or CPU-experts setting in its models.ini section; the rest stays as it is."
    + " If it's loaded, it is unloaded so the next load uses them.";
}

const withQ = (base, n, vs, k) => base + (n == null ? "" : "&ngl=" + n) + (k == null ? "" : "&ncmoe=" + k)
  + (vs ? "&vs=" + encodeURIComponent(vs) : "");

/** X-ray of a model you have, under its own settings. */
export const openXray = id => xrayModal({
  id, title: `X-ray: ${id}`, from: "read from the GGUF header. Nothing is loaded.",
  what: "Drag to see what moves. Nothing is saved until you press Apply to this model.",
  url: (n, vs, k) => withQ("/api/model/xray?model=" + encodeURIComponent(id), n, vs, k),
  quants: () => api("/api/model/xray/quants?model=" + encodeURIComponent(id)),
  apply: (n, k) => api("/api/model/xray/apply", {model: id, ngl: n, ncmoe: k || 0}),
});

/** X-ray of a Hugging Face file before you download it (Discover). files is the
 *  repo listing already on screen, for Compare with. */
export const openHubXray = (repo, path, files = []) => xrayModal({
  id: String(path).split("/").pop().replace(/(-\d{5}-of-\d{5})?\.gguf$/i, ""),
  title: `X-ray before download: ${quantName(path)}`,
  from: "read from the file's header on Hugging Face. Nothing is downloaded.",
  what: "Starts where the most weight fits your VRAM: fewer GPU layers, or for a MoE its experts in RAM. Drag to see what moves.",
  url: (n, vs, k) => withQ(`/api/hub/xray?repo=${encodeURIComponent(repo)}&path=${encodeURIComponent(path)}`, n, vs, k),
  quants: async () => ({repo, current: path, files, reason: ""}),
});

async function xrayModal(src) {
  const id = src.id;
  let v;
  try { v = await api(src.url(null, "")); }
  catch (e) { toast("Couldn't X-ray " + id + ": " + (e.message || e), "err"); return; }
  const top = v.n_layer + 1, conf = clampNgl(v.configured_ngl, v.n_layer);
  // a MoE you don't have yet: its experts can go to RAM block by block (--n-cpu-moe)
  const moe = !!v.experts, confK = Math.max(0, Math.min(Number(v.configured_ncmoe) || 0, v.n_layer));
  showModal(src.title, `
    <p class="af-q">${esc(v.arch)} · ${esc(v.n_layer)} blocks, ${esc(src.from)}</p>
    <div class="xr-ngl"><label for="xr-ngl">GPU layers</label>
      <input type="range" id="xr-ngl" min="0" max="${esc(top)}" step="1" value="${esc(conf)}">
      <output for="xr-ngl" data-xr-ngl>${esc(nglText(conf, v.n_layer))}</output>
      <button type="button" class="qbtn" data-xr-reset hidden>Back to ${esc(conf)}</button>
      <button type="button" class="qbtn" data-xr-share title="This view as a 1200x630 image">Share image</button></div>
    ${moe ? `<div class="xr-ngl"><label for="xr-moe">Experts on CPU</label>
      <input type="range" id="xr-moe" min="0" max="${esc(v.n_layer)}" step="1" value="${esc(confK)}">
      <output for="xr-moe" data-xr-moe>${esc(moeText(confK, v.n_layer))}</output></div>` : ""}
    ${v.flags ? `<p class="xr-flags">Load it with <code data-xr-flags>${esc(v.flags)}</code>
      <button type="button" class="qbtn" data-xr-copy title="Copy these llama-server flags">Copy</button>
      ${v.best ? `<button type="button" class="qbtn" data-xr-fit title="Move the sliders to where the most weight fits your VRAM">Fit my VRAM</button>` : ""}
      ${src.apply ? `<button type="button" class="qbtn load" data-xr-apply title="Write these flags into this model's settings">Apply to this model</button>` : ""}</p>` : ""}
    <div class="xr-ngl"><label for="xr-vs">Compare with</label>
      <select id="xr-vs" data-xr-vs disabled><option value="">looking up the other quants…</option></select>
      <span class="af-q dim" data-xr-vsnote></span></div>
    <p class="af-q dim xr-what">${esc(src.what || "Drag to see what moves. A what-if only: nothing is saved.")}</p>
    <div data-xr-body>${body(v)}</div>`);
  const range = $("#xr-ngl"), out = $("[data-xr-ngl]"), reset = $("[data-xr-reset]"), box = $("[data-xr-body]");
  const pick = $("[data-xr-vs]"), note = $("[data-xr-vsnote]");
  const moeRange = $("#xr-moe"), moeOut = $("[data-xr-moe]"), flagsEl = $("[data-xr-flags]");
  const kNow = () => moeRange ? Math.max(0, Math.min(Number(moeRange.value) || 0, v.n_layer)) : null;
  if (!range || !box) return;
  let t = 0, cur = v;
  const fetchView = async () => {
    const n = clampNgl(range.value, v.n_layer), vs = pick.value;
    const mine = ++seq;
    if (vs) note.textContent = "reading its header…";
    try {
      const w = await api(src.url(n, vs, kNow()));
      if (mine === seq && box.isConnected) {
        cur = w; note.textContent = "";
        if (flagsEl && w.flags) flagsEl.textContent = w.flags;
        setHTML(box, w.vs ? compareBody(w) : body(w));
      }
    } catch (e) {
      if (mine === seq) note.textContent = "";
      toast(String(e.message || e), "err");
    }
  };
  const update = () => {
    const n = clampNgl(range.value, v.n_layer);
    out.textContent = nglText(n, v.n_layer);
    if (moeOut) moeOut.textContent = moeText(kNow(), v.n_layer);
    reset.hidden = n === conf && (kNow() ?? confK) === confK;
    clearTimeout(t);
    t = setTimeout(fetchView, 90);
  };
  range.oninput = update;
  if (moeRange) moeRange.oninput = update;
  reset.onclick = () => { range.value = String(conf); if (moeRange) moeRange.value = String(confK); update(); };
  const fitBtn = $("[data-xr-fit]");
  if (fitBtn) fitBtn.onclick = () => {
    range.value = String(clampNgl(v.best.ngl, v.n_layer));
    if (moeRange) moeRange.value = String(v.best.ncmoe || 0);
    update();
  };
  const applyBtn = $("[data-xr-apply]");
  if (applyBtn) applyBtn.onclick = async () => {
    const n = clampNgl(range.value, v.n_layer), k = kNow() || 0;
    const shown = flagsEl.textContent;
    if (!await askYes(applyAsk(id, shown),
                      {title: "Apply to " + id, ok: "Apply"})) return;
    const r = await src.apply(n, k).catch(e => ({error: e.message || String(e)}));
    if (!r || !r.ok) { toast("Couldn't apply: " + ((r && r.error) || "unknown error"), "err"); return; }
    toast(r.was_running ? `Saved ${r.flags}; ${id} was unloaded, load it again to use them`
                        : `Saved ${r.flags} to ${id}`, "ok");
  };
  const copy = $("[data-xr-copy]");
  if (copy) copy.onclick = () => navigator.clipboard.writeText(flagsEl.textContent)
    .then(() => toast("Flags copied", "ok"), () => toast("Couldn't copy", "err"));
  pick.onchange = fetchView;
  $("[data-xr-share]").onclick = () => shareXray(cur, id);
  try {
    const q = await src.quants();
    const others = q.files.filter(f => f.path !== q.current);
    if (!pick.isConnected) return;
    if (!others.length) {
      pick.hidden = true;
      note.textContent = q.reason || "This repo has no other quant to compare with.";
      return;
    }
    setHTML(pick, `<option value="">another quant of ${esc(q.repo)}…</option>` + others.map(f =>
      `<option value="${esc(f.path)}">${esc(quantName(f.path))} · ${esc(gb(f.size))}</option>`).join(""));
    pick.disabled = false;
  } catch {
    if (pick.isConnected) { pick.hidden = true; note.textContent = "Couldn't list the other quants."; }
  }
}

// The view on screen (slider included) as a 1200x630 image (xraycard.js), loaded on
// first use. Same dialog as the other share cards.
async function shareXray(v, id) {
  const { drawXray, drawCompare } = await import("./xraycard.js");
  showModal(v.vs ? `Share quant compare: ${id}` : `Share X-ray: ${id}`, `<canvas class="af-card" width="1200" height="630" aria-label="Model X-ray card"></canvas>
    <p class="af-q">Holds the model's name, architecture, layer sizes, quant types and the GPU / RAM split. No paths.</p>
    <div class="af-act"><button type="button" class="qbtn load" data-xc-png>Download image</button>
      <button type="button" class="qbtn" data-xc-copyimg>Copy image</button></div>`);
  const dlg = document.querySelector("#modal-root dialog");
  const canvas = dlg && dlg.querySelector("canvas.af-card");
  if (!canvas) return;
  await (v.vs ? drawCompare : drawXray)(canvas, v, id);
  const blob = () => new Promise(r => canvas.toBlob(r, "image/png"));
  dlg.querySelector("[data-xc-png]").onclick = async () => {
    const url = URL.createObjectURL(await blob());
    const name = "llamaforge-xray-" + String(id).replace(/[^\w.-]+/g, "_")
      + (v.vs ? "-vs-" + quantName(v.vs.file).replace(/[^\w.-]+/g, "_") : "") + ".png";
    const a = Object.assign(document.createElement("a"), {href: url, download: name});
    a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  dlg.querySelector("[data-xc-copyimg]").onclick = async () => {
    try { await navigator.clipboard.write([new ClipboardItem({"image/png": blob()})]); toast("Image copied", "ok"); }
    catch { toast("This browser can't copy images; use Download", "err"); }
  };
}
