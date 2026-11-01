// The bay plan: Stowage's GPU view, drawn on the Models and Stats tabs.
//
// Each GPU is a hold ruled in 1 GiB cells, on one scale shared by every GPU,
// with each model it holds stowed as a container at its real size. Per-model
// sizes come from the pool's footprints (models this panel loaded in
// multi-model mode); whatever the card uses beyond them is one honest
// remainder block. A caller with an open row passes its planner verdict as
// `book`, drawn as a booked box; a booking bigger than the free space hangs
// past the hold's end as hatched deck cargo, and anything the planner would
// unload for it is marked. Sizes are GiB, as the planner reports them.
//
// `seen` (box id -> {t, i}) is the caller's record of when each box was first
// drawn: a box keeps .stow for as long as the drop-in motion runs, so a
// second render inside one poll can't cut it short. Each view owns its own,
// so switching tabs never replays the motion for boxes that view has shown.
import { esc } from "./core.js";

export const tenth = mib => Math.round(mib / 102.4) / 10;
export const gb = mib => tenth(mib).toFixed(1);

/** What a bay plan reads from an /api/state payload. */
export function bayInputs(st) {
  const sl = (st && st.slots) || {};
  const up = ((st && st.models) || []).filter(m => (m.backend || "llamacpp") === "llamacpp" &&
    (m.status === "loaded" || m.status === "sleeping")).map(m => m.id);
  return {fps: sl.footprints || {}, main: sl.main || "", up};
}

// The colour a model's container wears: orange for the main, then blue and
// green alternating over the workers in name order, so a worker keeps its
// colour on every GPU it spans and on every view that shows it. "" means the
// model has no footprint, so no container to match.
export function cargoHue(fps = {}, main = "") {
  const workers = Object.keys(fps).filter(id => id !== main).sort();
  return id => !fps[id] ? "" : id === main ? "main" : (workers.indexOf(id) % 2 ? "w2" : "w1");
}

// util/temp come from nvidia-smi; an AMD/Intel card read through the engine's
// --list-devices (or amdgpu sysfs) has neither, so say where its memory came from.
export function gpuTele(x) {
  if (x.via) return `<span>VIA <b>${esc(x.via)}</b></span>`;
  return `<span>UTIL <b>${esc(x.util)}%</b></span><span>TEMP <b>${esc(x.temp)}&deg;C</b></span>`;
}

export function bayPlans(g, {fps = {}, main = "", up = [], book = null, bookId = "", seen = new Map()} = {}) {
  if (!g.length || g[0].error) return `<div class="bay"><div class="bay-off">GPU telemetry unavailable: neither nvidia-smi nor llama.cpp reported a GPU.</div></div>`;
  const unknown = up.filter(id => !fps[id]);           // loaded, but no footprint we can stow
  const workers = Object.keys(fps).filter(id => id !== main).sort();
  const hue = cargoHue(fps, main);
  const order = [...(fps[main] ? [main] : []), ...workers];
  const want = x => book && book.footprint ? +(book.footprint[String(x.index)] || 0) : 0;
  const evict = new Set(book && book.evict || []);
  // one scale for every hold, wide enough for any booking that overhangs
  const maxTotal = Math.max(...g.map(x => Math.max(x.total || 0, (x.used || 0) + want(x)))) || 1;
  const fresh = new Set(), now = Date.now();
  let n = [...seen.values()].filter(v => now - v.t < 1500).length;

  const out = g.map(x => {
    const key = String(x.index), total = Math.max(x.total || 0, 1), used = Math.min(Math.max(x.used || 0, 0), total);
    const cells = total / 1024, boxes = [];
    let at = 0;
    // g: the size printed on the box (a booking shows all it needs, not just what fits)
    const put = (cls, id, label, mib, title, g = mib) => {
      if (mib <= 0) return;
      const bid = key + ":" + (id || cls);
      if (!seen.has(bid)) seen.set(bid, {t: now, i: n++});
      const rec = seen.get(bid), stow = cls !== "sys" && now - rec.t < 1500;
      fresh.add(bid);
      if (id && evict.has(id)) { cls += " out"; title += ", unloaded to make room"; }
      // too narrow to carry its name: the key under the hold names it instead
      const narrow = mib / maxTotal < 0.16;
      boxes.push({cls: cls + (narrow ? " nar" : ""), stow, label, mib, g, x: at, title, i: stow ? rec.i : 0});
      at += mib;
    };
    for (const id of order) {
      const mib = Math.min((fps[id] || {})[key] || 0, used - at);
      put(hue(id), id, id, mib, `${id}: ${gb(mib)} GiB on GPU ${key}`);
    }
    const rest = used - at;
    if (rest >= 51) {
      // memory no footprint accounts for: say plainly what it is, or that we can't split it
      let cls = "sys", label = "In use";
      if (!unknown.length) label = order.length ? "System + other apps" : "System";
      else if (unknown.length === 1 && rest >= 1024) { cls = "main"; label = unknown[0] + " + system"; }
      put(cls, cls === "main" ? unknown[0] : "", label, rest,
        `${label}: ${gb(rest)} GiB` + (cls === "main" ? " (the model's share is not measured separately here)" : ""));
    }
    const free = total - used, need = want(x);
    let bookMib = 0, shortMib = 0, deck = "";
    if (need > 0) {
      bookMib = Math.min(need, free); shortMib = need - bookMib;
      const title = shortMib > 0 ? `${bookId} needs ${gb(need)} GiB here, ${gb(shortMib)} GiB more than is free`
                                 : `${bookId} would take ${gb(need)} GiB here`;
      put(shortMib > 0 ? "booked over" : "booked", "book:" + bookId, bookId, bookMib, title, need);
      // the part that doesn't fit hangs past the end of the hold as deck cargo
      if (shortMib > 0) deck = `<div class="deck" title="${esc(title)}" style="--w:${(shortMib / 1024).toFixed(3)}">
          <span class="box-n">Short</span><span class="box-g">${esc(gb(shortMib))}</span></div>`;
    }
    const usedT = tenth(used), bookT = tenth(bookMib), totT = tenth(total);
    // a booking that fits comes out of free; one that doesn't is never stowed
    const freeT = Math.max(0, Math.round((totT - usedT - (shortMib > 0 ? 0 : bookT)) * 10) / 10);
    const step = cells > 24 ? 8 : 4, ticks = [];
    for (let v = 0; v <= cells - step / 2; v += step) ticks.push(`<span style="left:${(v / cells * 100).toFixed(3)}%">${v}</span>`);
    ticks.push(`<span class="end" style="left:100%">${esc(totT.toFixed(totT % 1 ? 1 : 0))} GiB</span>`);
    const aria = `GPU ${key}: ` + boxes.map(b => `${b.label} ${gb(b.g)} GiB`).join(", ") +
      (boxes.length ? ", " : "") + `${freeT.toFixed(1)} GiB free of ${totT.toFixed(1)} GiB` +
      (shortMib > 0 ? `, ${gb(shortMib)} GiB short` : "");
    const boxHTML = boxes.map(b => `<div class="box ${esc(b.cls + (b.stow ? " stow" : ""))}" title="${esc(b.title)}" style="--x:${(b.x / 1024).toFixed(3)};--w:${(b.mib / 1024).toFixed(3)};--i:${b.i}">
        <span class="box-n">${esc(b.label)}</span><span class="box-g">${esc(gb(b.g))}</span>${b.cls.split(" ").includes("out") ? `<span class="box-x">Unload</span>` : ""}</div>`).join("");
    // the key names every box; CSS shows it for narrow boxes only, and for all on a phone
    const keyHTML = boxes.map(b => `<span class="k ${esc(b.cls)}"><i></i>${esc(b.label)} <b>${esc(gb(b.g))}</b>${b.cls.split(" ").includes("out") ? " <em>unload</em>" : ""}</span>`).join("");
    return `<section class="bay" aria-label="${esc(x.name)}, GPU ${esc(key)}">
      <div class="bay-head"><span class="bay-name">${esc(x.name)}</span><span class="bay-idx">GPU ${esc(key)}</span>
        <span class="bay-tele">${gpuTele(x)}</span></div>
      <div class="bay-body">
        <div class="hold-col" style="--span:${(total / maxTotal).toFixed(4)}">
          <div class="hold" role="img" aria-label="${esc(aria)}" style="--cells:${cells.toFixed(3)}">${boxHTML}${deck}</div>
          <div class="hold-scale" aria-hidden="true">${ticks.join("")}</div>
          ${boxes.length ? `<div class="hold-key" aria-hidden="true">${keyHTML}</div>` : ""}
        </div>
        <dl class="ledger">
          <dd class="unit">GiB</dd>
          <dt>Used</dt><dd>${usedT.toFixed(1)}</dd>
          ${need > 0 && !shortMib ? `<dt>Booked</dt><dd>${bookT.toFixed(1)}</dd>` : ""}
          <dt>Free</dt><dd>${freeT.toFixed(1)}</dd>
          ${shortMib > 0 ? `<dt>Needs</dt><dd>${gb(need)}</dd><dt class="short">Short</dt><dd class="short">${gb(shortMib)}</dd>` : ""}
          <dt class="tot">Total</dt><dd class="tot">${totT.toFixed(1)}</dd>
        </dl>
      </div></section>`;
  }).join("");
  // forget boxes that left, so a model loaded again is stowed again
  for (const b of [...seen.keys()]) if (!fresh.has(b)) seen.delete(b);
  return out;
}
