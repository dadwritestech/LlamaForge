// Arena share card: the blind-vote leaderboard as a 1200x630 image, the same
// look and size as the Auto-Forge and rig cards. The data is /api/arena:
// model names, Elo, W-L-T, the vote count and the GPU names. No paths or hosts.
// arenaText() decides every string (pure, tested); drawArena() only paints them.
import { W, H, C, COND, SANS, fit } from "./forgecard.js";

const TOP = 6;

export function arenaText(d) {
  const games = Math.trunc(+(d && d.games) || 0);
  const board = (d && d.board) || [];
  const sub = [((d && d.gpus) || []).join(" + "), `${games} blind vote${games === 1 ? "" : "s"}`, "my prompts"]
    .filter(Boolean).join("  ·  ");
  return {
    kicker: "MY BLIND ARENA",
    brand: "LLAMAFORGE",
    sub,
    rows: board.slice(0, TOP).map((r, i) => ({rank: String(i + 1), name: String(r.model || ""),
      elo: String(Math.round(+r.elo || 0)), wlt: `${r.wins || 0}-${r.losses || 0}-${r.ties || 0}`})),
    more: board.length > TOP ? `+${board.length - TOP} more` : "",
    empty: "No votes yet",
    note: "Names hidden until the vote",
    foot: "github.com/dadwritestech/LlamaForge",
  };
}

/** Paint the card for d onto canvas (sized W x H). Waits for the panel fonts. */
export async function drawArena(canvas, d) {
  const t = arenaText(d);
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

  // the board: rank, model, Elo, W-L-T
  const X0 = 48, XN = 120, XE = W - 260, XW = W - 48;
  g.fillStyle = C.dim; g.font = `600 26px ${COND}`;
  g.fillText("#", X0, 186); g.fillText("MODEL", XN, 186);
  g.textAlign = "right"; g.fillText("ELO", XE, 186); g.fillText("W-L-T", XW, 186);
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
    g.fillText(fit(g, r.name, `500 28px ${SANS}`, XE - XN - 120), XN, y);
    g.textAlign = "right";
    g.fillStyle = i === 0 ? C.amber : C.ink; g.font = `700 34px ${COND}`; g.fillText(r.elo, XE, y);
    g.fillStyle = C.dim; g.font = `600 30px ${COND}`; g.fillText(r.wlt, XW, y);
    g.textAlign = "left";
  });
  if (t.more) { g.fillStyle = C.dim; g.font = `500 22px ${SANS}`; g.fillText(t.more, XN, 250 + TOP * 50 - 14); }

  g.strokeStyle = C.rule; g.beginPath(); g.moveTo(48, 560.5); g.lineTo(W - 48, 560.5); g.stroke();
  g.fillStyle = C.green; g.font = `500 26px ${SANS}`;
  g.fillText(t.note, 48, 600);
  g.fillStyle = C.dim; g.textAlign = "right"; g.fillText(t.foot, W - 48, 600);
  g.textAlign = "left";
  return canvas;
}
