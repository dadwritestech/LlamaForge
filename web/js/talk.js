// Voice tab, "Talk" card: talk to a model out loud. Press Talk, speak, and the
// recording goes to /api/stt/talk, where one model that hears (an audio
// mmproj) writes down what you said and answers it; the answer is then spoken
// by the Speak card's engine and played. Hearing and answering on one model
// means models-max 1 never swaps between them.
//
// Hands-free: after each answer the mic opens again and stops by itself once
// you have spoken and then paused (an energy gate on the live mic, no model).
// The conversation (the last few turns) lives here, in the page, and goes up
// with each turn. Every value interpolated goes through esc().
import { $, api, esc, setHTML, toast } from "./core.js";
import { startMic, toSamples, wavB64, RATE } from "./listen.js";

const TURNS = 6;                 // exchanges sent back as context
const QUIET_S = 1.2;             // hands-free: this much quiet after speech ends a turn
const LOUD = 0.02;               // RMS that counts as speech
const T = { mic: null, turns: [], busy: false, audio: null, speak: null, canSpeak: false,
            model: globalThis.localStorage?.getItem("lf_talk_model") || "" };

const s1 = x => (Math.round(x * 10) / 10).toFixed(1) + " s";

export function pickTalkModel(models, saved) {
  return models.includes(saved) ? saved : (models[0] || "");
}

export function timings(t) {
  const parts = [`hear ${s1(t.hear_s)}`, `think ${s1(t.think_s)}`];
  if (t.speak_s != null) parts.push(`speak ${s1(t.speak_s)}`);
  return parts.join(" · ");
}

/** The last n answered turns as chat messages for /api/stt/talk. */
export function historyOf(turns, n) {
  return turns.filter(t => t.heard && t.reply).slice(-n)
    .flatMap(t => [{ role: "user", content: t.heard }, { role: "assistant", content: t.reply }]);
}

export function turnMarkup(t) {
  return `<div class="talk-turn">
    <div class="kv"><span class="k">You</span><span class="v">${esc(t.heard)}</span></div>
    <div class="kv"><span class="k">${esc(t.model)}</span><span class="v">${esc(t.reply)}</span></div>
    <div class="note" style="margin:0">${esc(timings(t))}</div></div>`;
}

/** The card for /api/stt/status; "" until a model here can hear. */
export function talkMarkup(st, canSpeak) {
  if (!st || !st.models || !st.models.length) return "";
  const m = pickTalkModel(st.models, T.model);
  const pick = st.models.length > 1
    ? `<div class="f"><span class="lbl">Model</span><select id="talk-model">${st.models.map(x =>
        `<option value="${esc(x)}"${x === m ? " selected" : ""}>${esc(x)}</option>`).join("")}</select></div>` : "";
  return `<div class="card" id="talk-card"><h3>Talk</h3>
    <div class="note">Talk to a model out loud. It hears you, answers, and the answer is spoken back,
      all on this machine. The same model hears and answers, so nothing swaps in between.</div>
    ${canSpeak ? "" : `<div class="note">Set up Speak above to hear the answers; until then they show as text.</div>`}
    <div class="formrow">${pick}
      <label class="f"><span class="lbl">Hands-free</span><span><input type="checkbox" id="talk-hands">
        listen again after each answer, stop when I pause</span></label></div>
    <div class="actions">
      <button type="button" class="primary" id="talk-go">Talk</button>
      <button type="button" class="ghost" id="talk-new">New conversation</button>
      <span class="note" id="talk-state" style="margin:0"></span></div>
    <div id="talk-log">${T.turns.map(turnMarkup).join("")}</div></div>`;
}

/** Wire the card. speak(text) -> WAV bytes (throws); canSpeak says whether it can. */
export function wireTalk(host, st, { speak, canSpeak }) {
  T.speak = speak; T.canSpeak = canSpeak;
  const go = $("#talk-go", host);
  if (!go) return;
  go.onclick = () => (T.mic ? T.mic.stop() : T.audio ? hush() : listen());
  $("#talk-new", host).onclick = () => { T.turns = []; if (T.mic) T.mic.stop(); hush(); drawLog(); state(""); };
  const sel = $("#talk-model", host);
  if (sel) sel.onchange = () => { T.model = sel.value; localStorage.setItem("lf_talk_model", sel.value); };
  T.model = pickTalkModel(st.models, T.model);
}

function state(text) { const el = $("#talk-state"); if (el) el.textContent = text; }
function button(text) { const b = $("#talk-go"); if (b) b.textContent = text; }
const handsFree = () => !!$("#talk-hands")?.checked;

function drawLog() {
  const log = $("#talk-log");
  if (!log) return;
  setHTML(log, T.turns.map(turnMarkup).join(""));
  log.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hush() {
  if (T.audio) { T.audio.pause(); T.audio = null; }
  button("Talk");
}

async function listen() {
  if (T.busy) return;
  try { T.mic = await startMic(s => state(`Listening... ${s.toFixed(0)} s`), 60); }
  catch (e) { toast("No microphone access: " + (e.message || e), "err"); return; }
  button("Stop");
  state("Listening...");
  const gate = handsFree() ? quietGate(T.mic) : null;
  const blob = await T.mic.done;
  if (gate) gate();
  T.mic = null;
  button("Talk");
  turn(blob);
}

/** Stop the mic once there was speech and then QUIET_S of quiet. Returns a cleanup. */
function quietGate(mic) {
  const ctx = new AudioContext(), an = ctx.createAnalyser();
  ctx.createMediaStreamSource(mic.stream).connect(an);
  const buf = new Float32Array(an.fftSize);
  let spoke = false, quiet = 0, last = performance.now();
  const iv = setInterval(() => {
    an.getFloatTimeDomainData(buf);
    let sum = 0;
    for (const v of buf) sum += v * v;
    const now = performance.now();
    if (Math.sqrt(sum / buf.length) > LOUD) { spoke = true; quiet = 0; }
    else if (spoke) quiet += (now - last) / 1000;
    last = now;
    if (spoke && quiet >= QUIET_S) mic.stop();
  }, 100);
  return () => { clearInterval(iv); ctx.close(); };
}

async function turn(blob) {
  let samples;
  try { samples = await toSamples(blob); }
  catch (e) { state(""); toast("Could not read the recording: " + (e.message || e), "err"); return; }
  if (samples.length < RATE / 2) { state("Didn't catch anything. Try again, a little closer to the mic."); return; }
  busy(true);
  state("Thinking...");
  let r;
  try {
    r = await api("/api/stt/talk", { audio: wavB64(samples), history: historyOf(T.turns, TURNS),
                                     model: T.model });
  } catch (e) { r = { error: e.message || String(e) }; }
  if (r.error) { busy(false); state(""); toast(r.error, "err"); return; }
  if (!r.heard) { busy(false); state("Didn't catch any words. Try again."); return; }
  const t = { heard: r.heard, reply: r.reply, model: r.model, hear_s: r.hear_s, think_s: r.think_s, speak_s: null };
  T.turns.push(t);
  drawLog();
  if (!T.canSpeak || !r.reply) { busy(false); state(""); return; }
  state("Speaking...");                     // the button stays off until the answer plays
  const t0 = performance.now();
  let bytes;
  try { bytes = await T.speak(r.reply); }
  catch (e) { busy(false); state(""); toast("Could not speak the answer: " + (e.message || e), "err"); return; }
  busy(false);
  t.speak_s = (performance.now() - t0) / 1000;
  drawLog();
  state("");
  play(bytes);
}

function busy(on) {
  T.busy = on;
  const b = $("#talk-go");
  if (b) b.disabled = on;
}

function play(bytes) {
  const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  const a = new Audio(url);
  T.audio = a;
  button("Stop");
  a.onended = () => {
    URL.revokeObjectURL(url);
    if (T.audio !== a) return;                 // hushed or replaced
    T.audio = null;
    button("Talk");
    if (handsFree() && $("#talk-card")?.offsetParent) listen();
  };
  a.play().catch(() => { T.audio = null; button("Talk"); });
}
