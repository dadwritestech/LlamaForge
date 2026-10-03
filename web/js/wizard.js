// First-run wizard: engine -> hardware -> model -> tune -> load.
// Talks to the model list only through the bus, so neither imports the other.
import { $, esc, setHTML, api, toast, meter } from "./core.js";
import { S, models } from "./state.js";
import { on, emit } from "./bus.js";
import { switchTab, applyMode } from "./ui.js";
import { mountEngineCard } from "./engine.js";

const WIZ = {step:0, engine:null, model:null, intent:"balanced", rec:null,
  steps:["engine","hardware","model","tune","load"]};

function wizShow() { $("#wizard").hidden = false; wizRender(); }
function wizHide() { $("#wizard").hidden = true; }

function wizRender() {
  const kind = WIZ.steps[WIZ.step];
  const body = $("#wizard-body");
  ({engine:wizEngine, hardware:wizHardware, model:wizModel,
    tune:wizTune, load:wizLoad}[kind])(body);
  $("#wiz-back").disabled = WIZ.step === 0;
}

function wizEngine(body) {
  setHTML(body, `<div class="wizard-step"><h2>llama.cpp engine</h2>
    <p>LlamaForge drives llama.cpp. How do you want to get it?</p>
    <label><input type="radio" name="eng" value="prebuilt" checked> Download the official build for this PC
      <span class="note" style="display:inline;margin:0">(recommended, about a minute, no compiler)</span></label><br>
    <label><input type="radio" name="eng" value="have"> I already have llama-server built</label><br>
    <label><input type="radio" name="eng" value="source"> Build from source (Build tab, needs CMake + a compiler)</label>
    <div id="wiz-engine-card" style="margin-top:10px"></div></div>`);
  const card = $("#wiz-engine-card");
  const sync = () => {
    const v = (document.querySelector('input[name="eng"]:checked') || {}).value;
    card.style.display = v === "prebuilt" ? "" : "none";
    if (v === "prebuilt" && !card.dataset.mounted) {
      card.dataset.mounted = "1";
      mountEngineCard(card, {compact: true, onDone: () => emit("refresh", true)});
    }
  };
  body.querySelectorAll('input[name="eng"]').forEach(r => r.onchange = sync);
  sync();
}

function wizHardware(body) {
  const g = (S.STATE && S.STATE.gpus) || [];
  const rows = (g.length && !g[0].error)
    ? g.map(x => `<li>${esc(x.name||"GPU")} — ${esc(x.total?(x.total/1024).toFixed(1):"?")} GB</li>`).join("")
    : "";
  setHTML(body, `<div class="wizard-step"><h2>Your hardware</h2>
    <ul>${rows||"<li>No GPU detected — CPU mode.</li>"}</ul></div>`);
}

function wizModel(body) {
  const ms = models().filter(m => m.backend !== "vllm");
  if (!ms.length) { wizStarters(body); return; }
  const opts = ms.map(m => `<option value="${esc(m.id)}"${m.id === WIZ.model ? " selected" : ""}>${esc(m.id)}</option>`).join("");
  setHTML(body, `<div class="wizard-step"><h2>Pick a model</h2>
    <select id="wiz-model">${opts}</select></div>`);
  const sel = $("#wiz-model"); if (sel) WIZ.model = sel.value;
}

// Nothing downloaded yet: three starters sized to this GPU, downloaded right
// here. Sending a newcomer off to a 50-result Hub search was where the first
// run ended (review 01 #2). The backend registers the file when it finishes.
let wizDlPoll = null;
async function wizStarters(body) {
  setHTML(body, `<div class="wizard-step"><h2>Get your first model</h2>
    <p class="note" style="margin:0">Checking what fits your GPU...</p></div>`);
  let r;
  try { r = await api("/api/starters"); } catch (e) { r = {starters: []}; }
  if (WIZ.steps[WIZ.step] !== "model") return;      // user moved on meanwhile
  const gb = b => (b / 1e9).toFixed(1) + " GB";
  const vram = r.vram_mib ? `${(r.vram_mib / 1024).toFixed(0)} GB of VRAM` : "no GPU (these run on the CPU, slowly)";
  const cards = (r.starters || []).map((m, i) => `
    <div class="starter${m.recommended ? " rec" : ""}">
      <div><div class="t">${esc(m.title)} ${m.recommended ? '<span class="tag" style="color:var(--amber);border-color:var(--amber)">best fit</span>' : ""}</div>
        <div class="b">${esc(m.blurb)} &middot; ${esc(gb(m.size))}</div></div>
      <button ${m.recommended ? 'class="primary"' : ""} data-starter="${i}">Download</button>
    </div>`).join("");
  setHTML(body, `<div class="wizard-step"><h2>Get your first model</h2>
    <p>Picked for ${esc(vram)}. One click downloads it and adds it to your models.</p>
    ${cards}
    <div id="wiz-dl" hidden><div class="meter" id="wiz-dl-meter"></div><div class="note" id="wiz-dl-msg"></div></div>
    <p class="note">Want something else? <a href="#" id="wiz-discover">Browse Hugging Face in Discover</a>.</p></div>`);
  const d = $("#wiz-discover");
  if (d) d.onclick = e => { e.preventDefault(); wizHide(); switchTab("discover"); };
  body.querySelectorAll("[data-starter]").forEach(b => b.onclick = () => wizDownload(r.starters[+b.dataset.starter]));
}

async function wizDownload(m) {
  const res = await api("/api/hub/download", {repo: m.repo, path: m.path, shards: m.shards || 1, mmproj: m.mmproj || ""});
  if (!res.started) { toast("A download is already running - see Discover", "err"); return; }
  document.querySelectorAll("[data-starter]").forEach(b => b.disabled = true);
  $("#wiz-dl").hidden = false;
  clearInterval(wizDlPoll);
  wizDlPoll = setInterval(async () => {
    let s;
    try { s = await api("/api/hub/progress"); } catch (e) { return; }
    const box = $("#wiz-dl-msg");
    if (!box) { clearInterval(wizDlPoll); return; }   // wizard closed; the download carries on
    setHTML($("#wiz-dl-meter"), meter(s.downloaded, Math.max(s.total, 1)));
    box.textContent = s.phase === "registering" ? "Downloaded - adding it to your models..."
      : s.phase === "done" ? "" : s.phase === "failed" ? "Download failed: " + (s.error || "").slice(0, 100)
      : `${(s.downloaded / 1e9).toFixed(2)} / ${(s.total / 1e9).toFixed(2)} GB`;
    if (["done", "failed", "cancelled", "paused"].includes(s.phase)) clearInterval(wizDlPoll);
    if (s.phase === "done") {
      if ((s.added || []).length) {
        WIZ.model = s.added[0];
        box.textContent = `${s.added[0]} is ready. Press Next.`;
        emit("refresh", true);
      } else {
        box.textContent = "Downloaded, but it could not be added: " + (s.register_error || "unknown error").slice(0, 100);
      }
    }
    if (s.phase === "failed" || s.phase === "cancelled")
      document.querySelectorAll("[data-starter]").forEach(b => b.disabled = false);
  }, 1000);
}

function wizTune(body) {
  setHTML(body, `<div class="wizard-step"><h2>Tune for your goal</h2>
    <select id="wiz-intent">
      <option value="balanced">Balanced</option>
      <option value="speed">Max speed</option>
      <option value="context">Max context</option>
      <option value="coding">Coding</option>
    </select>
    <button id="wiz-tune-run">Auto-tune</button>
    <button id="wiz-tune-refine" hidden>Refine by benchmarking (~1 min)</button>
    <div id="wiz-tune-out"></div></div>`);
  $("#wiz-tune-run").onclick = async () => {
    WIZ.intent = $("#wiz-intent").value;
    const r = await api("/api/autotune/recommend", {model: WIZ.model, intent: WIZ.intent});
    WIZ.rec = r; wizRenderRec(r); $("#wiz-tune-refine").hidden = false;
  };
  $("#wiz-tune-refine").onclick = async () => {
    const r = await api("/api/autotune/refine",
      {model: WIZ.model, intent: WIZ.intent, knobs: WIZ.rec.knobs});
    WIZ.rec = {...WIZ.rec, knobs: r.knobs}; wizRenderRec(WIZ.rec);
  };
}

function wizRenderRec(r) {
  const rows = Object.entries(r.knobs||{}).map(([k,v]) =>
    `<tr><td>${esc(k)}</td><td>${esc(v)}</td><td class="wizard-rationale">${esc((r.rationale||{})[k]||"")}</td></tr>`).join("");
  setHTML($("#wiz-tune-out"), `<table>${rows}</table>`);
}

function wizLoad(body) {
  setHTML(body, `<div class="wizard-step"><h2>Ready</h2>
    <p>Apply these settings to <b>${esc(WIZ.model)}</b> and load it now.</p></div>`);
}

async function wizNext() {
  const kind = WIZ.steps[WIZ.step];
  if (kind === "engine") {
    const sel = document.querySelector('input[name="eng"]:checked');
    WIZ.engine = sel ? sel.value : "prebuilt";
    // a prebuilt install keeps running in the background while the wizard moves on
  }
  if (kind === "model") {
    const sel = $("#wiz-model"); if (sel) WIZ.model = sel.value;
    if (!WIZ.model) return;                    // require a model
  }
  if (kind === "load") {
    try {
      if (WIZ.rec) await api("/api/save", {model: WIZ.model, settings: WIZ.rec.knobs});
      // detect both thrown errors and failed responses
      let loadErr = false;
      try {
        const r = await api("/api/load", {model: WIZ.model});
        if (!r.success) loadErr = true;
      } catch (e) { loadErr = true; }
      // always persist onboarding and close, regardless of load outcome
      await api("/api/config", {onboarded: true, ui_mode: "lite"});
      applyMode("lite"); wizHide(); emit("refresh", true);
      toast(loadErr ? "Setup done — model failed to load; load it from the Models tab"
                    : "Setup complete", loadErr ? "err" : "ok");
      if (!loadErr) emit("chat", WIZ.model);   // the point of setup: talking to it
    } catch (e) { toast("Setup failed", "err"); }
    return;
  }
  WIZ.step = Math.min(WIZ.step + 1, WIZ.steps.length - 1);
  wizRender();
}

export function initWizard() {
  $("#wiz-next").onclick = wizNext;
  $("#wiz-back").onclick = () => { WIZ.step = Math.max(0, WIZ.step-1); wizRender(); };
  // Skipping is "not now", not "I'm an expert": stay in the simple Lite view.
  $("#wiz-skip").onclick = async () => {
    await api("/api/config", {onboarded: true, ui_mode: "lite"});
    applyMode("lite"); wizHide();
  };
  // show it the first time state says onboarding hasn't happened
  on("state", s => {
    const el = $("#wizard");
    if (!(s.onboarding||{}).onboarded && el && el.hidden) wizShow();
  });
}
