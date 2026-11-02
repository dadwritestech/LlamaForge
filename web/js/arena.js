// Arena tab: two of your models answer one prompt, unlabeled and in random
// order; you pick the better one and a local Elo board (backend/arena.py,
// arena.json) ranks them. Names and speeds stay hidden until the vote.
// Model output is plain text through esc(): it never becomes markup.
import { $, esc, setHTML, api, toast, askYes } from "./core.js";
import { showModal } from "./models.js";
import { drawArena } from "./arenacard.js";
import { loadRecent, saveRecent, recentLabel } from "./arenarecent.js";

let round = null;       // {left, right, prompt, answers: {left, right}, voted}
let models = [];
let busy = false;

export async function loadArena() {
  const view = $("#view-arena");
  if (!view) return;
  const d = await api("/api/arena").catch(e => ({error: String(e.message || e)}));
  if (!d || d.error) { setHTML(view, `<div class="card"><h3>Arena</h3><div class="note">${esc((d && d.error) || "couldn't reach the panel")}</div></div>`); return; }
  models = d.models || [];
  if (!$("#arena-ask", view)) render(view);
  renderBoard(d);
}

function render(view) {
  const opts = sel => models.map(m => `<option value="${esc(m)}"${m === sel ? " selected" : ""}>${esc(m)}</option>`).join("");
  if (models.length < 2) {
    setHTML(view, `<div class="card"><h3>Blind arena</h3><div class="note">The arena needs at least two chat models.
      Find another one in Discover.</div></div><div id="arena-board"></div>`);
    return;
  }
  setHTML(view, `<div class="card"><h3>Blind arena</h3>
      <p class="note" style="margin-top:0">Two of your models answer the same prompt. You see them as Left and Right
        in random order and vote before the names and speeds show. Votes build an Elo board on this PC:
        your prompts, your GPUs. One model loads at a time, so a round may swap models.</p>
      <div class="arena-pick">
        <select id="arena-a" aria-label="First model">${opts(models[0])}</select>
        <span class="note" style="margin:0">vs</span>
        <select id="arena-b" aria-label="Second model">${opts(models[1])}</select>
        <button type="button" class="ghost" id="arena-surprise">Surprise me</button>
      </div>
      <select id="arena-recent" aria-label="Recent prompts" hidden></select>
      <textarea id="arena-prompt" placeholder="Ask something you actually care about: a bug, an email, a tricky question" style="min-height:90px"></textarea>
      <div class="actions"><button type="button" id="arena-ask">Ask both</button>
        <span class="note" id="arena-status" style="margin:0" aria-live="polite"></span></div>
    </div>
    <div id="arena-round"></div>
    <div id="arena-board"></div>`);
  $("#arena-surprise", view).onclick = surprise;
  $("#arena-ask", view).onclick = ask;
  $("#arena-recent", view).onchange = e => {
    const p = recent[+e.target.value];
    if (p) $("#arena-prompt").value = p;
    e.target.value = "";
  };
  fillRecent(loadRecent());
}

let recent = [];
function fillRecent(list) {
  recent = list;
  const el = $("#arena-recent");
  if (!el) return;
  el.hidden = !list.length;
  setHTML(el, `<option value="">Recent prompts (${list.length})</option>` +
    list.map((p, i) => `<option value="${i}">${esc(recentLabel(p))}</option>`).join(""));
}

async function surprise(quiet) {
  const d = await api("/api/arena/pair", {}).catch(() => null);
  if (!d || d.error) { toast((d && d.error) || "couldn't pick", "err"); return false; }
  $("#arena-a").value = d.a;
  $("#arena-b").value = d.b;
  if (quiet !== true) toast("Two models picked. Their names stay up there, but Left and Right are shuffled.", "ok");
  return true;
}

// After a vote: the same prompt again, either on a fresh random pair (builds
// the board fastest) or on the same two models, shuffled again.
async function again(how) {
  if (busy || !round) return;
  $("#arena-prompt").value = round.prompt;
  if (how === "pair") { if (!(await surprise(true))) return; }
  else { $("#arena-a").value = round.left; $("#arena-b").value = round.right; }
  ask();
}

function status(t) { const s = $("#arena-status"); if (s) s.textContent = t; }

async function ask() {
  if (busy) return;
  const a = $("#arena-a").value, b = $("#arena-b").value, prompt = $("#arena-prompt").value.trim();
  if (a === b) { toast("Pick two different models", "err"); return; }
  if (!prompt) { toast("Type a prompt first", "err"); return; }
  const flip = crypto.getRandomValues(new Uint8Array(1))[0] & 1;
  round = {left: flip ? b : a, right: flip ? a : b, prompt, answers: {}, voted: false};
  fillRecent(saveRecent(prompt));
  busy = true;
  $("#arena-ask").disabled = true;
  renderRound();
  try {
    for (const side of ["left", "right"]) {
      status(`${side === "left" ? "Left" : "Right"} is answering (loading it first if it isn't up)...`);
      const d = await api("/api/arena/answer", {model: round[side], prompt}).catch(e => ({error: String(e.message || e)}));
      round.answers[side] = d && !d.error ? d : {error: (d && d.error) || "no answer"};
      renderRound();
    }
    status("");
  } finally {
    busy = false;
    const btn = $("#arena-ask");
    if (btn) btn.disabled = false;
  }
}

function side(name) {
  const r = round.answers[name];
  const label = name === "left" ? "Left" : "Right";
  const who = round.voted ? `<span class="arena-who">${esc(round[name])}${r && r.tok_s ? ` &middot; ${esc(r.tok_s)} tok/s` : ""}</span>` : "";
  let body;
  if (!r) body = `<div class="skel">WAITING...</div>`;
  else if (r.error) body = `<div class="note">${esc(r.error)}</div>`;
  else body = `${r.thinking ? `<details class="arena-think"><summary>thinking</summary><pre>${esc(r.thinking)}</pre></details>` : ""}
      <pre class="arena-ans">${esc(r.text || "(empty answer)")}</pre>${r.cut ? `<div class="note">cut off at the token limit</div>` : ""}`;
  return `<div class="arena-side"><div class="arena-head"><b>${label}</b>${who}</div>${body}</div>`;
}

function renderRound() {
  const el = $("#arena-round");
  if (!el || !round) return;
  const done = round.answers.left && round.answers.right && !round.answers.left.error && !round.answers.right.error;
  const votes = done && !round.voted ? `<div class="actions arena-vote">
      <button type="button" data-vote="left">Left is better</button>
      <button type="button" class="ghost" data-vote="tie">Tie</button>
      <button type="button" data-vote="right">Right is better</button></div>` : "";
  setHTML(el, `<div class="card"><h3>This round</h3><div class="arena-cols">${side("left")}${side("right")}</div>${votes}
    ${round.voted ? `<div class="note">${esc(round.voted)}</div>
      <div class="actions"><button type="button" data-again="pair">Same prompt, new pair</button>
        <button type="button" class="ghost" data-again="rematch">Rematch these two</button></div>` : ""}</div>`);
  el.querySelectorAll("[data-vote]").forEach(btn => btn.onclick = () => vote(btn.dataset.vote));
  el.querySelectorAll("[data-again]").forEach(btn => btn.onclick = () => again(btn.dataset.again));
}

async function vote(pick) {
  if (!round || round.voted) return;
  const winner = pick === "tie" ? "tie" : "a";
  const a = pick === "right" ? round.right : round.left;
  const b = pick === "right" ? round.left : round.right;
  const d = await api("/api/arena/vote", {a, b, winner}).catch(e => ({error: String(e.message || e)}));
  if (!d || d.error) { toast((d && d.error) || "vote not saved", "err"); return; }
  round.voted = pick === "tie" ? "Tie recorded." : `You picked ${pick === "left" ? round.left : round.right}.`;
  renderRound();
  renderBoard(d);
}

function renderBoard(d) {
  const el = $("#arena-board");
  if (!el) return;
  const rows = d.board || [];
  if (!rows.length) {
    setHTML(el, `<div class="card"><h3>Your board</h3><div class="note">No votes yet. Your first round puts two models on it.</div></div>`);
    return;
  }
  setHTML(el, `<div class="card"><h3>Your board</h3>
    <table class="arena-board"><thead><tr><th>#</th><th>Model</th><th>Elo</th><th>W-L-T</th></tr></thead><tbody>
    ${rows.map((r, i) => `<tr><td>${i + 1}</td><td>${esc(r.model)}</td><td>${esc(r.elo)}</td><td>${esc(r.wins)}-${esc(r.losses)}-${esc(r.ties)}</td></tr>`).join("")}
    </tbody></table>
    <div class="note">${esc(d.games)} games. Elo starts at 1000; a few dozen votes before it means much.</div>
    <div class="actions"><button type="button" class="ghost" id="arena-card">Share image</button>
      <button type="button" class="ghost" id="arena-md">Copy as markdown</button>
      <button type="button" class="ghost" id="arena-reset">Reset board</button></div></div>`);
  $("#arena-md", el).onclick = () => navigator.clipboard.writeText(d.markdown || "")
    .then(() => toast("Board copied as markdown", "ok"), () => toast("Couldn't copy", "err"));
  $("#arena-card", el).onclick = openArenaCard;
  $("#arena-reset", el).onclick = async () => {
    if (!(await askYes("Every vote and rating goes. This can't be undone.", {title: "Reset the board?", ok: "Reset", danger: true}))) return;
    renderBoard(await api("/api/arena/reset", {}));
  };
}

// The board as a 1200x630 image (arenacard.js), fetched fresh so it carries
// the GPU names. Same dialog as the rig and Auto-Forge cards.
async function openArenaCard() {
  const d = await api("/api/arena").catch(e => ({error: String(e.message || e)}));
  if (!d || d.error) { toast((d && d.error) || "couldn't build the card", "err"); return; }
  showModal("Share your board", `<canvas class="af-card" width="1200" height="630" aria-label="Arena board card"></canvas>
    <p class="af-q">Holds model names, Elo, wins, losses and ties, the vote count and your GPU names. No prompts, paths or hosts.</p>
    <div class="af-act"><button type="button" class="qbtn load" data-ac-png>Download image</button>
      <button type="button" class="qbtn" data-ac-copyimg>Copy image</button></div>`);
  const dlg = document.querySelector("#modal-root dialog");
  const canvas = dlg && dlg.querySelector("canvas.af-card");
  if (!canvas) return;
  await drawArena(canvas, d);
  const blob = () => new Promise(r => canvas.toBlob(r, "image/png"));
  dlg.querySelector("[data-ac-png]").onclick = async () => {
    const url = URL.createObjectURL(await blob());
    const a = Object.assign(document.createElement("a"), {href: url, download: "llamaforge-arena.png"});
    a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  dlg.querySelector("[data-ac-copyimg]").onclick = async () => {
    try { await navigator.clipboard.write([new ClipboardItem({"image/png": blob()})]); toast("Image copied", "ok"); }
    catch { toast("This browser can't copy images; use Download", "err"); }
  };
}
