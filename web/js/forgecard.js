// Auto-Forge share card: a 1200x630 image of one result (the size Reddit, X
// and GitHub previews use), drawn on a canvas in the Stowage look.
//
// The data comes from /api/forge/share, which already keeps paths, keys and
// hosts out. cardText() decides every string (pure, tested); drawCard() only
// paints them.

export const W = 1200, H = 630;
export const C = {bg: "#0f1d33", hold: "#0b172a", rule: "#2a4466", ink: "#eaf0f6", dim: "#a4b6c8",
  amber: "#ffc21a", onAmber: "#14110a", green: "#5cc595", warn: "#ffcf4d"};
export const COND = "'Barlow Condensed','Arial Narrow',sans-serif", SANS = "'Barlow','Segoe UI',sans-serif";

const f1 = v => (v == null || !Number.isFinite(+v)) ? "–" : (+v).toFixed(1);

/** Every string on the card. The headline is prose for chat, code otherwise. */
export function cardText(d) {
  const lead = d.intent === "chat" ? "prose" : "code";
  const other = lead === "code" ? "prose" : "code";
  const line = m => ({metric: m, before: d.before ? f1(d.before[m]) : null, after: f1(d.after?.[m]),
    x: d.x && d.x[m] ? `${(+d.x[m]).toFixed(1)}×` : null});
  const ctx = d.ctx ? `${Math.floor(d.ctx / 1024)}k context` : "";
  const o = line(other);
  const note = !d.lossless ? {text: "Lossy KV cache: answers can differ slightly", warn: true}
    : {text: d.drafter_checked ? "Same answers: drafter checked token-for-token" : "Same answers: lossless settings", warn: false};
  return {
    kicker: "AUTO-FORGE",
    model: d.file || d.model || "",
    sub: [(d.gpus || []).join(" + "), ctx].filter(Boolean).join("  ·  "),
    head: line(lead),
    other: o.before ? `${other} ${o.before} → ${o.after} tok/s${o.x ? ` (${o.x})` : ""}` : `${other} ${o.after} tok/s`,
    note: d.drops_vision ? {text: note.text + ", vision off", warn: true} : note,
    foot: "github.com/dadwritestech/LlamaForge",
  };
}

export function fit(ctx, text, font, max) {
  text = String(text).replace(/--/g, "-\u200C-");    // a font ligature would draw "--" as one dash
  ctx.font = font;
  if (ctx.measureText(text).width <= max) return text;
  let t = text;
  while (t.length > 1 && ctx.measureText(t + "…").width > max) t = t.slice(0, -1);
  return t + "…";
}

// numbers are never cut: step the size down until the line fits
function shrink(ctx, text, weight, size, family, max, min = 64) {
  for (; size > min; size -= 4) {
    ctx.font = `${weight} ${size}px ${family}`;
    if (ctx.measureText(text).width <= max) return size;
  }
  ctx.font = `${weight} ${min}px ${family}`;
  return min;
}

/** Paint the card for d onto canvas (sized W x H). Waits for the panel fonts. */
export async function drawCard(canvas, d) {
  const t = cardText(d);
  try {
    await Promise.all([`700 150px ${COND}`, `600 30px ${COND}`, `600 44px ${SANS}`, `500 26px ${SANS}`]
      .map(f => document.fonts.load(f)));
  } catch { /* system fallbacks are fine */ }
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext("2d");
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  // the hold: 1-unit cells, like the bay plan
  g.fillStyle = C.hold; g.fillRect(48, 196, W - 96, 250);
  g.strokeStyle = "rgba(164,182,200,.10)"; g.lineWidth = 1;
  for (let x = 48; x <= W - 48; x += 25) { g.beginPath(); g.moveTo(x + .5, 196); g.lineTo(x + .5, 446); g.stroke(); }
  g.fillStyle = C.amber; g.fillRect(0, 0, 14, H);
  g.textBaseline = "alphabetic";

  g.fillStyle = C.amber; g.font = `600 30px ${COND}`;
  g.fillText(t.kicker, 48, 70);
  g.fillStyle = C.dim; g.textAlign = "right"; g.fillText("LLAMAFORGE", W - 48, 70);
  g.textAlign = "left";
  g.fillStyle = C.ink;
  g.fillText(fit(g, t.model, `600 44px ${SANS}`, W - 96), 48, 128);
  g.fillStyle = C.dim;
  g.fillText(fit(g, t.sub, `500 26px ${SANS}`, W - 96), 48, 170);

  const h = t.head;
  const xw = h.x ? 300 : 0;
  g.fillStyle = C.ink;
  const big = h.before ? `${h.before} → ${h.after}` : h.after;
  shrink(g, big, 700, 150, COND, W - 156 - xw);
  g.fillText(big, 76, 370);
  g.fillStyle = C.dim; g.font = `600 30px ${COND}`;
  g.fillText(`TOK/S ON ${h.metric.toUpperCase()}`, 80, 420);
  if (h.x) {
    g.fillStyle = C.amber; g.fillRect(W - 48 - xw, 196, xw, 250);
    g.fillStyle = C.onAmber; g.textAlign = "center";
    g.font = `700 130px ${COND}`; g.fillText(h.x, W - 48 - xw / 2, 360);
    g.font = `600 30px ${COND}`; g.fillText("FASTER", W - 48 - xw / 2, 412);
    g.textAlign = "left";
  }

  g.fillStyle = C.ink;
  g.fillText(fit(g, t.other, `500 30px ${SANS}`, W - 96), 48, 500);
  g.fillStyle = t.note.warn ? C.warn : C.green;
  g.fillText(fit(g, t.note.text, `500 26px ${SANS}`, 700), 48, 572);
  g.fillStyle = C.dim; g.textAlign = "right";
  g.font = `500 26px ${SANS}`; g.fillText(t.foot, W - 48, 572);
  g.textAlign = "left";
  g.strokeStyle = C.rule; g.beginPath(); g.moveTo(48, 532.5); g.lineTo(W - 48, 532.5); g.stroke();
  return canvas;
}
