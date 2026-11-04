// Model X-ray share card: the layer stack as a 1200x630 image, same look and
// size as the other cards. Columns run left to right (embedding, every block,
// output), each split by tensor kind, the system-RAM part hatched. Data is the
// /api/model/xray view: model id, arch, sizes and quant types. No paths.
// xrayCardText() decides every string and fraction (pure, tested); drawXray() only paints.
import { W, H, C, COND, SANS, fit } from "./forgecard.js";
import { KINDS, gb, nglText, quantName, compareText } from "./xray.js";

export const KIND_ORDER = KINDS;
// the panel's X-ray colours, fixed so the image looks the same on every skin
export const KIND_COLORS = {
  embedding: "#5d6d7e", attention: "#3fd7e6", ssm: "#9d8cd6", ffn: "#ffc21a", experts: "#c7782a",
  shared: "#39d98a", "per-layer": "#6f9fd8", mtp: "#d26b9a", norm: "#8a8f98", output: "#7b8a99",
  other: "#6b6f76",
};

const r4 = x => Math.round(x * 10000) / 10000;

const sizeOf = r => r.gpu + r.ram;

// one column per row, height relative to `tallest`, kinds bottom-up with the RAM part hatched
function columns(rows, tallest) {
  return rows.map(r => {
    const all = sizeOf(r), segs = [];
    for (const [k] of KINDS) {
      const bytes = (r.parts || {})[k] || 0, ram = (r.ram_parts || {})[k] || 0;
      if (!bytes || !all) continue;
      for (const [b, isRam] of [[bytes - ram, false], [ram, true]])
        if (b / all >= 0.002) segs.push({kind: k, frac: r4(b / all), ram: isRam});
    }
    return {label: r.label, h: r4(all / tallest), segs};
  });
}

// a kind earns a legend entry only if it's big enough to see (norms usually aren't)
function legendOf(...vs) {
  const bytes = {};
  for (const v of vs) for (const r of v.rows || []) for (const [k, b] of Object.entries(r.parts || {})) bytes[k] = (bytes[k] || 0) + b;
  const total = Object.values(bytes).reduce((a, b) => a + b, 0);
  const legend = KINDS.filter(([k]) => bytes[k] && bytes[k] / total >= 0.005).map(([kind, label]) => ({kind, label}));
  if (vs.some(v => v.ram > 0 && (v.rows || []).some(r => Object.keys(r.ram_parts || {}).length)))
    legend.push({kind: "ram", label: "in system RAM"});
  return legend;
}

const typeMix = (v, n = 4) => (v.types || []).filter(t => Math.round(t.share * 100) >= 1).slice(0, n)
  .map(t => `${t.type} ${Math.round(t.share * 100)}%`).join("  ·  ");

const NOTE_GUESS = "An override-tensor pattern didn't parse: the split is a guess.";

export function xrayCardText(v, id) {
  const rows = v.rows || [];
  return {
    kicker: "MODEL X-RAY",
    brand: "LLAMAFORGE",
    title: id,
    sub: [v.arch, `${v.n_layer} blocks`, `${gb(v.total)} of weights`].filter(Boolean).join("  ·  "),
    gpu: `${gb(v.gpu)} on GPU` + (v.tied_copy ? ` + ${gb(v.tied_copy)} tied copy` : ""),
    ram: `${gb(v.ram)} in system RAM`,
    layers: "GPU layers: " + nglText(v.ngl, v.n_layer),
    types: typeMix(v),
    cols: columns(rows, Math.max(1, ...rows.map(sizeOf))),
    legend: legendOf(v),
    note: v.confident === false ? NOTE_GUESS : "Read from the GGUF header. Nothing loaded.",
    foot: "github.com/dadwritestech/LlamaForge",
  };
}

/** The two-up card: this file and the compared quant (v.vs), columns on one scale. */
export function compareCardText(v, id) {
  const w = v.vs;
  const tallest = Math.max(1, ...[v, w].flatMap(x => (x.rows || []).map(sizeOf)));
  const half = x => ({
    name: quantName(x.file),
    line: `${gb(x.total)}  ·  ${gb(x.gpu)} GPU  ·  ${gb(x.ram)} RAM`,
    types: typeMix(x, 3),
    cols: columns(x.rows || [], tallest),
  });
  return {
    kicker: "WHICH QUANT FITS?",
    brand: "LLAMAFORGE",
    title: id,
    sub: [v.arch, `${v.n_layer} blocks`, "GPU layers: " + nglText(v.ngl, v.n_layer)].filter(Boolean).join("  ·  "),
    delta: compareText(v, w),
    halves: [half(v), half(w)],
    legend: legendOf(v, w),
    note: v.confident === false || w.confident === false ? NOTE_GUESS
      : "Same settings for both. Headers only, nothing downloaded.",
    foot: "github.com/dadwritestech/LlamaForge",
  };
}

function hatch(g) {
  const p = document.createElement("canvas");
  p.width = p.height = 8;
  const q = p.getContext("2d");
  q.strokeStyle = "rgba(0,0,0,.6)"; q.lineWidth = 2.5;
  q.beginPath(); q.moveTo(-2, 10); q.lineTo(10, -2); q.moveTo(-2, 2); q.lineTo(2, -2);
  q.moveTo(6, 10); q.lineTo(10, 6); q.stroke();
  return g.createPattern(p, "repeat");
}

async function fonts() {
  try {
    await Promise.all([`600 30px ${COND}`, `600 40px ${SANS}`, `500 24px ${SANS}`].map(f => document.fonts.load(f)));
  } catch { /* system fallbacks are fine */ }
}

function frame(canvas, t) {
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext("2d");
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  g.fillStyle = C.amber; g.fillRect(0, 0, 14, H);
  g.textBaseline = "alphabetic";
  g.fillStyle = C.amber; g.font = `600 30px ${COND}`; g.fillText(t.kicker, 48, 64);
  g.fillStyle = C.dim; g.textAlign = "right"; g.fillText(t.brand, W - 48, 64); g.textAlign = "left";
  g.fillStyle = C.ink; g.fillText(fit(g, t.title, `600 40px ${SANS}`, W - 96), 48, 114);
  return g;
}

function paintCols(g, cols, X0, XN, TOP, BASE, pat) {
  const n = cols.length || 1;
  const step = (XN - X0) / n, gap = step >= 5 ? 1 : 0, cw = Math.max(1, step - gap);
  cols.forEach((c, i) => {
    const cx = X0 + i * step, colH = c.h * (BASE - TOP);
    let y = BASE;
    for (const s of c.segs) {
      const sh = s.frac * colH;
      g.fillStyle = KIND_COLORS[s.kind] || KIND_COLORS.other;
      g.fillRect(cx, y - sh, cw, sh);
      if (s.ram) { g.fillStyle = pat; g.fillRect(cx, y - sh, cw, sh); }
      y -= sh;
    }
  });
  g.strokeStyle = C.rule; g.lineWidth = 1;
  g.beginPath(); g.moveTo(X0, BASE + 0.5); g.lineTo(XN, BASE + 0.5); g.stroke();
}

function paintLegend(g, legend, y, pat) {
  let x = 48;
  g.font = `500 20px ${SANS}`;
  for (const l of legend) {
    g.fillStyle = l.kind === "ram" ? C.dim : KIND_COLORS[l.kind];
    g.fillRect(x, y, 16, 16);
    if (l.kind === "ram") { g.fillStyle = pat; g.fillRect(x, y, 16, 16); }
    g.fillStyle = C.ink; g.fillText(l.label, x + 22, y + 15);
    x += 22 + g.measureText(l.label).width + 22;
  }
}

function paintFoot(g, t, warn) {
  g.strokeStyle = C.rule; g.beginPath(); g.moveTo(48, 560.5); g.lineTo(W - 48, 560.5); g.stroke();
  g.font = `500 24px ${SANS}`;
  const footW = g.measureText(t.foot).width;
  g.fillStyle = C.dim; g.textAlign = "right"; g.fillText(t.foot, W - 48, 600);
  g.textAlign = "left"; g.fillStyle = warn ? C.warn : C.green;
  g.fillText(fit(g, t.note, `500 24px ${SANS}`, W - 96 - footW - 32), 48, 600);
}

/** Paint the card for view v of model id onto canvas. Waits for the panel fonts. */
export async function drawXray(canvas, v, id) {
  const t = xrayCardText(v, id);
  await fonts();
  const g = frame(canvas, t);
  g.fillStyle = C.dim; g.font = `500 24px ${SANS}`;
  const subW = Math.min(g.measureText(t.sub).width, W - 96);
  g.fillText(fit(g, t.sub, `500 24px ${SANS}`, W - 96), 48, 150);
  // the quant mix, right-aligned on the same line when there's room
  if (t.types && W - 96 - subW > 160) {
    g.textAlign = "right"; g.fillStyle = C.ink;
    g.fillText(fit(g, t.types, `600 24px ${SANS}`, W - 96 - subW - 40), W - 48, 150);
    g.textAlign = "left";
  }
  // the split: GPU in green, RAM in amber, and the slider setting it came from
  g.font = `600 30px ${COND}`;
  g.fillStyle = C.green; g.fillText(t.gpu, 48, 194);
  const x = 48 + g.measureText(t.gpu).width + 28;
  g.fillStyle = C.amber; g.fillText(t.ram, x, 194);
  g.fillStyle = C.dim; g.textAlign = "right";
  g.fillText(fit(g, t.layers, `500 22px ${SANS}`, W - 96 - x - g.measureText(t.ram).width), W - 48, 194);
  g.textAlign = "left";

  // the stack: one column per layer, bottom-aligned, kinds stacked bottom-up
  const X0 = 48, XN = W - 48, BASE = 478, pat = hatch(g);
  paintCols(g, t.cols, X0, XN, 222, BASE, pat);
  g.fillStyle = C.dim; g.font = `500 18px ${SANS}`;
  g.fillText("embedding", X0, BASE + 24);
  g.textAlign = "center"; g.fillText(`blocks 0 to ${Math.max(0, v.n_layer - 1)}`, (X0 + XN) / 2, BASE + 24);
  g.textAlign = "right"; g.fillText("output", XN, BASE + 24); g.textAlign = "left";

  paintLegend(g, t.legend, 518, pat);
  paintFoot(g, t, v.confident === false);
  return canvas;
}

/** Paint the two-up card (v with v.vs) onto canvas. */
export async function drawCompare(canvas, v, id) {
  const t = compareCardText(v, id);
  await fonts();
  const g = frame(canvas, t);
  g.fillStyle = C.dim; g.font = `500 24px ${SANS}`;
  g.fillText(fit(g, t.sub, `500 24px ${SANS}`, W - 96), 48, 150);
  // the headline: what the other quant costs
  g.fillStyle = C.ink; g.font = `600 30px ${COND}`;
  g.fillText(fit(g, t.delta, `600 30px ${COND}`, W - 96), 48, 194);

  const pat = hatch(g), GAP = 48, half = (W - 96 - GAP) / 2, TOP = 300, BASE = 482;
  t.halves.forEach((h, i) => {
    const X0 = 48 + i * (half + GAP), XN = X0 + half;
    g.textAlign = "left"; g.fillStyle = i ? C.amber : C.green; g.font = `600 30px ${COND}`;
    g.fillText(fit(g, h.name, `600 30px ${COND}`, half), X0, 240);
    g.fillStyle = C.ink; g.font = `500 22px ${SANS}`;
    const lineW = Math.min(g.measureText(h.line).width, half);
    g.fillText(fit(g, h.line, `500 22px ${SANS}`, half), X0, 272);
    // the quant mix, right-aligned on the same row when there's room
    if (h.types && half - lineW > 120) {
      g.fillStyle = C.dim; g.textAlign = "right";
      g.fillText(fit(g, h.types, `500 18px ${SANS}`, half - lineW - 24), XN, 272);
    }
    paintCols(g, h.cols, X0, XN, TOP, BASE, pat);
  });
  g.textAlign = "left";
  paintLegend(g, t.legend, 512, pat);
  paintFoot(g, t, /guess/.test(t.note));
  return canvas;
}
