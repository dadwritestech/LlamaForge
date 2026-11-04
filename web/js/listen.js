// Voice tab, "Speech to text" card: talk into the mic and get the words back,
// transcribed on this machine by a model whose mmproj hears audio (Gemma 4
// and friends, through backend/stt.py). Offers Gemma 4 E4B when nothing here
// can hear yet; the download goes through the Discover downloader, which
// registers it in models.ini with its mmproj.
//
// The mic records with MediaRecorder (webm/ogg), Web Audio decodes it to
// 16 kHz mono and wav.js wraps it as WAV, since llama.cpp decodes only
// WAV/MP3/FLAC. Every value interpolated goes through esc().
import { $, api, esc, setHTML, toast } from "./core.js";
import { encodeWav, toBase64, trimSilence } from "./wav.js";

const RATE = 16000;             // what speech encoders listen at; smaller uploads too
const MAX_REC_S = 120;
const L = { rec: null, poll: null, onSpeak: null };

const gib = b => (b / 1073741824).toFixed(1) + " GiB";

/** One line for the shared downloader's progress. */
export function dlLine(d) {
  if (d.phase === "registering") return "Adding it to your models...";
  const pct = d.total ? Math.floor(100 * d.downloaded / d.total) : 0;
  return `Downloading ${d.file || "..."} (${(d.done_files || 0) + 1} of ${d.total_files || 2}) ${pct}%`;
}

export function curlSnippet(origin) {
  return `curl ${origin}/v1/audio/transcriptions \\
  -F file=@speech.wav -F model=whisper-1`;
}

/** The card for /api/stt/status ({models, offer, download}); "" without one. */
export function listenMarkup(st, origin) {
  if (!st) return "";
  const o = st.offer || {}, d = st.download || {};
  const busy = !!d.running;
  const ours = d.repo && d.repo === o.repo;
  let body;
  if (st.models && st.models.length) {
    body = `<div class="note">Press <b>Record</b>, talk, press <b>Stop</b>. The words come back
        from <b>${esc(st.models[0])}</b> on your own GPU; the first time, the model loads first.</div>
      <div class="actions">
        <button type="button" class="primary" id="stt-rec">Record</button>
        <span class="note" id="stt-state" style="margin:0"></span></div>
      <div id="stt-out"></div>
      <div class="slabel">From code (OpenAI&rsquo;s <code>audio.transcriptions</code>)</div>
      <div class="snip"><button type="button" class="qbtn scopy" id="stt-copy"
        aria-label="Copy curl">Copy</button>${esc(curlSnippet(origin))}</div>`;
  } else if (busy && ours) {
    body = `<div class="msg work" id="stt-dl">${esc(dlLine(d))}</div>`;
  } else {
    body = `<div class="note">Nothing here can hear yet. Speech to text needs a model whose
        mmproj has an audio encoder.</div>
      <div class="kv"><span class="k">${esc(o.label)}</span><span class="v">${esc(o.note)}
        <span class="note">${esc(o.repo)}, ${esc(o.license)}.</span>
        <button type="button" class="primary" data-stt-get="1" ${busy ? "disabled" : ""}>Get (${esc(gib(o.bytes || 0))})</button></span></div>
      ${busy ? `<div class="note">Another download is running; this one can start when it ends.</div>` : ""}
      ${!busy && ours && d.phase === "failed" ? `<div class="msg err">Download failed: ${esc(d.error)}</div>` : ""}`;
  }
  return `<div class="card" id="stt-card"><h3>Speech to text</h3>${body}</div>`;
}

/** Wire the card rendered into `host`. onSpeak(text) hands a transcript to Speak. */
export function wireListen(host, st, { onSpeak, reload }) {
  L.onSpeak = onSpeak;
  const get = $("[data-stt-get]", host);
  if (get) get.onclick = async () => {
    get.disabled = true;
    const o = st.offer;
    const r = await api("/api/hub/download", { repo: o.repo, path: o.path, mmproj: o.mmproj });
    if (r.error || !r.started) {
      toast(r.error || "Another download is running", "err"); get.disabled = false; return;
    }
    reload();
  };
  const rec = $("#stt-rec", host);
  if (rec) rec.onclick = toggle;
  const copy = $("#stt-copy", host);
  if (copy) copy.onclick = () => navigator.clipboard.writeText(copy.nextSibling.textContent)
    .then(() => toast("Copied to clipboard", "ok"));
  clearTimeout(L.poll);
  const d = st.download || {};
  if (d.running && d.repo === st.offer?.repo) L.poll = setTimeout(async () => {
    if (!$("#stt-card")?.offsetParent) return;          // tab hidden: the next load re-arms it
    let s;
    try { s = await api("/api/stt/status"); } catch (e) { return; }
    const el = $("#stt-dl");
    if (s.download?.running && el) { el.textContent = dlLine(s.download); wireListen(host, s, { onSpeak, reload }); }
    else reload();
  }, 1000);
}

function state(text) { const el = $("#stt-state"); if (el) el.textContent = text; }

/** Start the mic. Returns {stop, done, stream}: done resolves to the recording
 *  (a Blob) when stop() is called or maxS runs out; onTick(seconds) runs 4x a second. */
export async function startMic(onTick, maxS = MAX_REC_S) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const chunks = [], rec = new MediaRecorder(stream), started = Date.now();
  rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  let tick;
  const done = new Promise(res => {
    rec.onstop = () => {
      clearInterval(tick);
      stream.getTracks().forEach(t => t.stop());
      res(new Blob(chunks, { type: rec.mimeType }));
    };
  });
  const stop = () => { if (rec.state === "recording") rec.stop(); return done; };
  tick = setInterval(() => {
    const s = (Date.now() - started) / 1000;
    if (onTick) onTick(s);
    if (s >= maxS) stop();
  }, 250);
  rec.start();
  return { stop, done, stream };
}

/** A recording -> 16 kHz mono samples with the silent ends trimmed. */
export async function toSamples(blob) {
  const ctx = new AudioContext();
  const src = await ctx.decodeAudioData(await blob.arrayBuffer());
  ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(src.duration * RATE), RATE);
  const node = off.createBufferSource();
  node.buffer = src; node.connect(off.destination); node.start();
  return trimSilence((await off.startRendering()).getChannelData(0), RATE);
}

/** Samples -> the base64 WAV the /api/stt routes take. */
export const wavB64 = samples => toBase64(encodeWav(samples, RATE));
export { RATE };

async function toggle() {
  const btn = $("#stt-rec");
  if (L.rec) { L.rec.stop(); return; }
  try { L.rec = await startMic(s => state(`Listening... ${s.toFixed(0)} s`)); }
  catch (e) { toast("No microphone access: " + (e.message || e), "err"); return; }
  btn.textContent = "Stop";
  state("Listening...");
  const blob = await L.rec.done;
  L.rec = null;
  btn.textContent = "Record";
  transcribe(blob);
}

async function transcribe(blob) {
  const btn = $("#stt-rec"), out = $("#stt-out");
  let samples;
  try { samples = await toSamples(blob); }
  catch (e) {
    state(""); toast("Could not read the recording: " + (e.message || e), "err"); return;
  }
  if (samples.length < RATE / 2) { state("That was silent. Try again, a little closer to the mic."); return; }
  btn.disabled = true;
  state("Transcribing...");
  const t0 = performance.now();
  let r;
  try { r = await api("/api/stt/transcribe", { audio: wavB64(samples) }); }
  catch (e) { r = { error: e.message || String(e) }; }
  btn.disabled = false;
  if (r.error) { state(""); setHTML(out, `<div class="msg err">${esc(r.error)}</div>`); return; }
  const took = ((performance.now() - t0) / 1000).toFixed(1);
  state(`${(samples.length / RATE).toFixed(1)} s of speech, words back in ${took} s.`);
  setHTML(out, `<textarea id="stt-text" rows="3" readonly>${esc(r.text)}</textarea>
    <div class="actions">
      <button type="button" class="ghost" id="stt-copytext">Copy</button>
      <button type="button" class="ghost" id="stt-speak">Speak it back</button></div>`);
  $("#stt-copytext").onclick = () => navigator.clipboard.writeText(r.text).then(() => toast("Copied", "ok"));
  $("#stt-speak").onclick = () => L.onSpeak && L.onSpeak(r.text);
}
