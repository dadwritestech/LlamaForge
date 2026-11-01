// Rig card: a 1200x630 image of this machine (GPUs, RAM, OS), the models it
// runs most with their measured speed, and the tokens it has produced. Same
// look and size as the Auto-Forge share card.
//
// The data comes from /api/rig, which already keeps paths and markup out.
// rigText() decides every string (pure, tested); drawRig() only paints them.
import { W, H, C, COND, SANS, fit } from "./forgecard.js";

export function human(n) {
  n = Math.trunc(+n || 0);
  for (const [div, suf] of [[1e9, "B"], [1e6, "M"], [1e3, "k"]])
    if (n >= div) return `${(n / div).toFixed(1)}${suf}`;
  return String(n);
}

export function rigText(d) {
  const hw = [(d.gpus || []).join(" + ") || "CPU only"];
  if (d.ram_gb) hw.push(`${d.ram_gb} GB RAM`);
  if (d.os) hw.push(d.os);
  return {
    kicker: "MY LOCAL AI RIG",
    brand: `LLAMAFORGE ${d.version || ""}`.trim(),
    hw: hw.join(" · "),
    big: human(d.generated),
    label: "TOKENS GENERATED HERE",
    sub: d.hours ? `over ${d.hours} hours of loaded models` : "",
    rows: (d.models || []).map(m => ({name: String(m.name || ""), tps: m.tps ? (+m.tps).toFixed(1) : "–",
                                      gen: human(m.generated)})),
    empty: "No model usage logged yet",
    foot: d.link || "github.com/dadwritestech/LlamaForge",
  };
}

/** Paint the card for d onto canvas (sized W x H). Waits for the panel fonts. */
export async function drawRig(canvas, d) {
  const t = rigText(d);
  try {
    await Promise.all([`700 120px ${COND}`, `600 30px ${COND}`, `600 34px ${SANS}`, `500 26px ${SANS}`]
      .map(f => document.fonts.load(f)));
  } catch { /* system fallbacks are fine */ }
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext("2d");
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  g.fillStyle = C.amber; g.fillRect(0, 0, 14, H);
  g.textBaseline = "alphabetic";

  g.fillStyle = C.amber; g.font = `600 30px ${COND}`;
  g.fillText(t.kicker, 48, 70);
  g.fillStyle = C.dim; g.textAlign = "right"; g.fillText(t.brand, W - 48, 70);
  g.textAlign = "left";
  g.fillStyle = C.ink;
  g.fillText(fit(g, t.hw, `600 34px ${SANS}`, W - 96), 48, 126);

  // left: the hold, with the token count stowed in it
  const LW = 400;
  g.fillStyle = C.hold; g.fillRect(48, 166, LW, 330);
  g.strokeStyle = "rgba(164,182,200,.10)"; g.lineWidth = 1;
  for (let x = 48; x <= 48 + LW; x += 25) { g.beginPath(); g.moveTo(x + .5, 166); g.lineTo(x + .5, 496); g.stroke(); }
  g.fillStyle = C.ink;
  g.fillText(fit(g, t.big, `700 120px ${COND}`, LW - 48), 72, 350);
  g.fillStyle = C.amber; g.font = `600 30px ${COND}`;
  g.fillText(t.label, 74, 400);
  g.fillStyle = C.dim;
  g.fillText(fit(g, t.sub, `500 24px ${SANS}`, LW - 52), 74, 440);

  // right: the models it runs most
  const X0 = 48 + LW + 40, XT = W - 220, XG = W - 48;
  g.fillStyle = C.dim; g.font = `600 26px ${COND}`;
  g.fillText("MOST USED", X0, 196);
  g.textAlign = "right"; g.fillText("TOK/S", XT, 196); g.fillText("TOKENS", XG, 196);
  g.textAlign = "left";
  g.strokeStyle = C.rule; g.beginPath(); g.moveTo(X0, 214.5); g.lineTo(XG, 214.5); g.stroke();
  if (!t.rows.length) {
    g.fillStyle = C.dim; g.fillText(fit(g, t.empty, `500 26px ${SANS}`, XG - X0), X0, 262);
  }
  t.rows.forEach((r, i) => {
    const y = 258 + i * 54;
    g.fillStyle = C.ink;
    g.fillText(fit(g, r.name, `500 28px ${SANS}`, XT - X0 - 130), X0, y);
    g.textAlign = "right";
    g.fillStyle = r.tps === "–" ? C.dim : C.amber; g.font = `700 34px ${COND}`; g.fillText(r.tps, XT, y);
    g.fillStyle = C.dim; g.font = `600 30px ${COND}`; g.fillText(r.gen, XG, y);
    g.textAlign = "left";
  });

  g.strokeStyle = C.rule; g.beginPath(); g.moveTo(48, 532.5); g.lineTo(W - 48, 532.5); g.stroke();
  g.fillStyle = C.green; g.font = `500 26px ${SANS}`;
  g.fillText("Measured on this machine by llama.cpp", 48, 572);
  g.fillStyle = C.dim; g.textAlign = "right"; g.fillText(t.foot, W - 48, 572);
  g.textAlign = "left";
  return canvas;
}
