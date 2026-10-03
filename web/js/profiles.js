// Launch profiles: a strip above the model list. One click switches to the
// profile's pinned engine build (if any), applies its preset, loads its model.
// Saved from a model's editor ("Save as profile"), which emits "profile-save".
import { $, esc, setHTML, api, toast } from "./core.js";
import { config as cfgOf } from "./state.js";
import { on, emit } from "./bus.js";
import { showModal } from "./models.js";

let shownSig = null;
let busy = false;

function describe(p) {
  return [p.model, p.preset ? `preset ${p.preset}` : "", p.engine ? `engine ${p.engine}` : ""]
    .filter(Boolean).join(" · ");
}

function render() {
  const el = $("#profiles"); if (!el) return;
  const P = cfgOf().profiles || {};
  const sig = JSON.stringify(P);
  if (sig === shownSig) return;          // the 4 s poll shouldn't churn the DOM
  shownSig = sig;
  const names = Object.keys(P).sort();
  if (!names.length) { setHTML(el, ""); return; }
  setHTML(el, `<div class="presetbar profbar">
    <span style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)">Profiles</span>
    ${names.map(n => `<span class="pchip" data-prof-launch="${esc(n)}" title="launch: ${esc(describe(P[n]))}">`
      + `&#9654; ${esc(n)}<span class="px" data-prof-del="${esc(n)}" title="delete profile">&times;</span></span>`).join("")}
  </div>`);
}

async function launch(name) {
  if (busy) return;
  busy = true;
  const p = (cfgOf().profiles || {})[name] || {};
  toast(p.engine ? `Launching ${name} (switching engine if needed)...` : `Launching ${name}...`, "ok");
  try {
    const r = await api("/api/profiles/launch", {name});
    if (r.ok) toast(`${name}: ${r.model} is loading${r.switched_engine ? " on " + p.engine : ""}`, "ok");
    else toast(`${name}: ${r.step === "engine" ? "engine switch failed - " : ""}${r.error || "launch failed"}`, "err");
  } finally {
    busy = false;
    emit("refresh", true);
  }
}

async function openSave({ id, backend }) {
  const c = cfgOf(), llama = backend !== "vllm";
  const engineKey = backend || c.active_engine || "llamacpp";
  const bound = ((c.preset_bindings || {})[engineKey] || {})[id] || "";
  let installs = [];
  if (llama) {
    const r = await api("/api/engine/prebuilt");
    installs = (r && r.installs) || [];
  }
  const base = id.split("/").pop().replace(/\.gguf$/i, "").slice(0, 40);
  const opt = (v, label, sel) => `<option value="${esc(v)}"${sel ? " selected" : ""}>${esc(label)}</option>`;
  const presetSel = llama ? `<div class="fld" style="margin-top:12px"><label>Preset</label><select id="prof-preset">
      ${opt("", "none (the model's saved settings)", !bound)}
      ${Object.keys(c.presets || {}).map(n => opt(n, n, n === bound)).join("")}
    </select></div>` : "";
  const engineSel = llama && installs.length ? `<div class="fld" style="margin-top:12px"><label>Engine</label><select id="prof-engine">
      ${opt("", "whichever build is active (follows updates)", true)}
      ${installs.map(i => {
        const dir = i.dir.split(/[\\/]/).pop();
        return opt(dir, `pin ${i.tag || dir} ${i.variant || ""}${i.active ? " (active now)" : ""}`, false);
      }).join("")}
    </select></div>` : "";
  const m = showModal("Save as profile", `
    <div class="note" style="margin-top:0">One click from the Models tab will load <b>${esc(id)}</b> with these choices.</div>
    <div class="fld" style="margin-top:12px"><label>Name</label><input id="prof-name" value="${esc(base)}" maxlength="40"></div>
    ${presetSel}${engineSel}
    <div style="display:flex;gap:8px;margin-top:14px"><button class="primary" id="prof-save">Save profile</button></div>`);
  const nameEl = $("#prof-name");
  nameEl.focus(); nameEl.select();
  $("#prof-save").onclick = async () => {
    const name = nameEl.value.trim();
    if (!name) { toast("Name the profile", "err"); return; }
    const profile = { model: id, backend: backend || "llamacpp",
      preset: ($("#prof-preset") || {}).value || "", engine: ($("#prof-engine") || {}).value || "" };
    const r = await api("/api/profiles/save", {name, profile});
    if (r.ok) { toast(`Saved profile "${name}"`, "ok"); m.close(); emit("refresh", true); }
    else toast(r.error || "save failed", "err");
  };
}

export function initProfiles() {
  on("state", render);
  on("profile-save", openSave);
  document.addEventListener("click", async e => {
    const del = e.target.closest("#profiles [data-prof-del]");
    if (del) {
      e.stopPropagation();
      const n = del.dataset.profDel;
      if (!confirm(`Delete profile "${n}"? The model and its settings stay.`)) return;
      await api("/api/profiles/delete", {name: n});
      toast("Profile deleted", "ok"); emit("refresh", true);
      return;
    }
    const chip = e.target.closest("#profiles [data-prof-launch]");
    if (chip) launch(chip.dataset.profLaunch);
  });
}
