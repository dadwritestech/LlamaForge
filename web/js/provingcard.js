// Checked-tasks share card: the proving board as a 1200x630 image, same look and
// size as the arena card. Data is /api/proving: model names, passed counts, tok/s,
// pass/fail per task and the GPU names. No answers, prompts, paths or hosts.
// provingText() decides every string (pure, tested); drawProving() only paints.
import { W, H, C, COND, SANS, fit } from "./forgecard.js";
import { boardRows } from "./provingview.js";

const TOP = 6;

export function provingText(d) {
  d = d || {};
  const tasks = Array.isArray(d.tasks) ? d.tasks : [];
  const n = tasks.length || 15;
  const sub = [(d.gpus || []).join(" + "), `${n} tasks, one try each`, "scored by a script"]
    .filter(Boolean).join("  ·  ");
  const rows = boardRows(d);
  const res = d.results || {};
  // a gap before each new category, so the pips read as five groups
  const breaks = tasks.map((t, i) => i > 0 && t.cat !== tasks[i - 1].cat);
  // the hook: when the fastest model isn't the one that gets most right, say both
  const fast = rows.filter(r => r.tok_s != null).sort((a, b) => b.tok_s - a.tok_s)[0];
  const best = rows[0];
  const tps = v => `${Math.round(v)} tok/s`;
  const contrast = fast && best && fast.model !== best.model && best.tok_s != null
    ? [`Fastest: ${fast.model}, ${tps(fast.tok_s)}, ${fast.passed} of ${fast.total} right.`,
       `Most right: ${best.model}, ${best.passed} of ${best.total} at ${tps(best.tok_s)}.`] : [];
  return {
    kicker: "MY CHECKED TASKS",
    brand: "LLAMAFORGE",
    sub,
    rows: rows.slice(0, TOP).map((r, i) => ({
      rank: String(i + 1), name: r.model, passed: `${r.passed}/${r.total}`,
      speed: r.tok_s == null ? "–" : tps(r.tok_s),
      pips: res[r.model] ? tasks.map(t => !!(res[r.model][t.id] && res[r.model][t.id].pass)) : [],
    })),
    breaks,
    contrast,
    groups: breaks.filter(Boolean).length + (tasks.length ? 1 : 0),
    more: rows.length > TOP ? `+${rows.length - TOP} more` : "",
    empty: "No checked runs yet",
    note: "Checked by a script. A smoke test, not a benchmark.",
    foot: "github.com/dadwritestech/LlamaForge",
  };
}

/** Paint the card for d onto canvas (sized W x H). Waits for the panel fonts. */
export async function drawProving(canvas, d) {
  const t = provingText(d);
  try {
    await Promise.all([`700 40px ${COND}`, `600 30px ${COND}`, `600 34px ${SANS}`, `500 26px ${SANS}`]
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
  g.fillText(fit(g, t.sub, `600 34px ${SANS}`, W - 96), 48, 126);

  // the board: rank, model, one pip per task (green right, dark wrong), passed, speed
  const X0 = 48, XN = 110, XP = 570, XS = W - 200, XW = W - 48, PIP = 16, GAP = 3, GROUP = 10;
  g.fillStyle = C.dim; g.font = `600 26px ${COND}`;
  g.fillText("#", X0, 186); g.fillText("MODEL", XN, 186); g.fillText("EACH TASK", XP, 186);
  g.textAlign = "right"; g.fillText("PASSED", XS, 186); g.fillText("SPEED", XW, 186);
  g.textAlign = "left";
  g.strokeStyle = C.rule; g.lineWidth = 1;
  g.beginPath(); g.moveTo(X0, 204.5); g.lineTo(XW, 204.5); g.stroke();
  if (!t.rows.length) {
    g.fillStyle = C.dim; g.fillText(fit(g, t.empty, `500 28px ${SANS}`, XW - X0), X0, 256);
  }
  t.rows.forEach((r, i) => {
    const y = 250 + i * 50;
    if (i === 0) { g.fillStyle = C.hold; g.fillRect(X0 - 12, y - 36, XW - X0 + 24, 50); }
    g.fillStyle = i === 0 ? C.amber : C.dim; g.font = `700 34px ${COND}`; g.fillText(r.rank, X0, y);
    g.fillStyle = C.ink;
    g.fillText(fit(g, r.name, `500 28px ${SANS}`, XP - XN - 24), XN, y);
    let x = XP;
    r.pips.forEach((ok, k) => {
      if (t.breaks[k]) x += GROUP;
      g.fillStyle = ok ? C.green : C.rule;
      g.fillRect(x, y - 20, PIP, PIP);
      x += PIP + GAP;
    });
    g.textAlign = "right";
    g.fillStyle = i === 0 ? C.amber : C.ink; g.font = `700 34px ${COND}`; g.fillText(r.passed, XS, y);
    g.fillStyle = C.dim; g.font = `600 30px ${COND}`; g.fillText(r.speed, XW, y);
    g.textAlign = "left";
  });
  if (t.more) { g.fillStyle = C.dim; g.font = `500 22px ${SANS}`; g.fillText(t.more, XN, 250 + TOP * 50 - 14); }
  // under a short board there's room for the hook
  if (t.contrast.length && t.rows.length <= 4) {
    const y0 = 250 + t.rows.length * 50 + 40;
    g.fillStyle = C.amber; g.fillText(fit(g, t.contrast[0], `600 30px ${SANS}`, XW - X0), X0, y0);
    g.fillStyle = C.ink; g.fillText(fit(g, t.contrast[1], `500 28px ${SANS}`, XW - X0), X0, y0 + 44);
  }

  g.strokeStyle = C.rule; g.beginPath(); g.moveTo(48, 560.5); g.lineTo(W - 48, 560.5); g.stroke();
  g.fillStyle = C.green; g.font = `500 26px ${SANS}`;
  g.font = `500 24px ${SANS}`;
  const footW = g.measureText(t.foot).width;
  g.fillStyle = C.dim; g.textAlign = "right"; g.fillText(t.foot, W - 48, 600);
  g.textAlign = "left"; g.fillStyle = C.green;
  g.fillText(fit(g, t.note, `500 24px ${SANS}`, W - 96 - footW - 32), 48, 600);
  g.textAlign = "left";
  return canvas;
}
