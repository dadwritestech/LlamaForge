// First-run wizard: engine -> hardware -> model -> tune -> load.
// Talks to the model list only through the bus, so neither imports the other.
import { $, esc, setHTML, api, toast } from "./core.js";
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
  if (!ms.length) {
    setHTML(body, `<div class="wizard-step"><h2>Pick a model</h2>
      <p>No models found. <a href="#" id="wiz-discover">Find one in Discover</a>.</p></div>`);
    const d = $("#wiz-discover");
    if (d) d.onclick = e => { e.preventDefault(); wizHide(); switchTab("discover"); };
    return;
  }
  const opts = ms.map(m => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join("");
  setHTML(body, `<div class="wizard-step"><h2>Pick a model</h2>
    <select id="wiz-model">${opts}</select></div>`);
  const sel = $("#wiz-model"); if (sel) WIZ.model = sel.value;
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
    } catch (e) { toast("Setup failed", "err"); }
    return;
  }
  WIZ.step = Math.min(WIZ.step + 1, WIZ.steps.length - 1);
  wizRender();
}

export function initWizard() {
  $("#wiz-next").onclick = wizNext;
  $("#wiz-back").onclick = () => { WIZ.step = Math.max(0, WIZ.step-1); wizRender(); };
  $("#wiz-skip").onclick = async () => {
    await api("/api/config", {onboarded: true, ui_mode: "advanced"});
    applyMode("advanced"); wizHide();
  };
  // show it the first time state says onboarding hasn't happened
  on("state", s => {
    const el = $("#wizard");
    if (!(s.onboarding||{}).onboarded && el && el.hidden) wizShow();
  });
}
