// Launch profiles: a strip above the model list. One click switches to the
// profile's pinned engine build (if any), applies its preset, loads its model.
// Saved from a model's editor ("Save as profile"), which emits "profile-save".
// Profiles share as recipes (readable JSON, backend/recipes.py) and import back;
// "browse recipes" lists the community gallery (repo recipes/ folder, backend/gallery.py).
import { $, esc, setHTML, api, toast } from "./core.js";
import { config as cfgOf } from "./state.js";
import { on, emit } from "./bus.js";
import { showModal } from "./models.js";

let shownSig = null;
let busy = false;
let importPoll = null;

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
  setHTML(el, `<div class="presetbar profbar">
    <span style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)">Profiles</span>
    ${names.map(n => `<span class="pchip" data-prof-launch="${esc(n)}" title="launch: ${esc(describe(P[n]))}">`
      + `&#9654; ${esc(n)}`
      + (P[n].backend === "vllm" ? "" : `<span class="px" data-prof-share="${esc(n)}" title="share as a recipe">&#8599;</span>`)
      + `<span class="px" data-prof-del="${esc(n)}" title="delete profile">&times;</span></span>`).join("")}
    <span class="pchip" data-prof-gallery title="tested setups shared by the community">&#9776; browse recipes</span>
    <span class="pchip" data-prof-import title="paste a recipe someone shared">+ import recipe</span>
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

async function openShare(name) {
  const r = await api("/api/profiles/export", {name});
  if (!r.ok) { toast(r.error || "export failed", "err"); return; }
  const text = JSON.stringify(r.recipe, null, 2);
  const m = r.recipe.model, e = r.recipe.engine;
  showModal(`Share "${name}"`, `
    <div class="note" style="margin-top:0">Paste this anywhere (a Reddit comment, a gist, a chat). Anyone with LlamaForge can
      <b>+ import recipe</b> it${m.hf_repo ? " and download the same file from <b>" + esc(m.hf_repo) + "</b>" : ""}.
      Only tuning knobs are included: no paths, keys or hosts.${e ? " Made on llama.cpp <b>" + esc(e.tag) + "</b>." : ""}</div>
    <textarea id="prof-recipe" readonly spellcheck="false" style="width:100%;height:260px;margin-top:12px;font-family:var(--mono,monospace);font-size:11px">${esc(text)}</textarea>
    <div style="display:flex;gap:8px;margin-top:14px"><button class="primary" id="prof-copy">Copy recipe</button></div>`);
  $("#prof-copy").onclick = () => navigator.clipboard.writeText(text).then(() => toast("Recipe copied", "ok"));
}

function stopPoll() { if (importPoll) { clearInterval(importPoll); importPoll = null; } }

const SHARE_URL = "https://github.com/dadwritestech/LlamaForge/tree/master/recipes#share-yours";

async function openGallery(force = false) {
  const r = await api("/api/recipes/gallery" + (force ? "?force=1" : ""));
  const list = (r && r.recipes) || [];
  // every knob, never "+N more": you see exactly what an import will apply
  const knobs = s => Object.entries(s).map(([k, v]) => `<code>${esc(k)}=${esc(v)}</code>`).join(" ");
  const card = (e, i) => `<div class="rcard">
      <div style="display:flex;gap:8px;align-items:baseline">
        <b style="flex:1">${esc(e.title)}</b>
        ${e.have ? `<span class="rhave" title="this model file is already registered here">on this machine</span>` : ""}
        <button data-gal-import="${i}">Import</button>
      </div>
      <div class="rmeta">${esc(e.model.file || "")}${e.model.hf_repo ? " · " + esc(e.model.hf_repo) : ""}</div>
      <div class="rmeta">Tested on ${esc(e.hardware || "unspecified hardware")}${e.author ? " · by " + esc(e.author) : ""}</div>
      ${e.notes ? `<div class="rnotes">${esc(e.notes)}</div>` : ""}
      <div class="rknobs">${knobs(e.settings)}</div>
    </div>`;
  const m = showModal("Community recipes", `
    <div class="note" style="margin-top:0">Setups people have run on their own hardware. Import one to get the same model
      (downloaded from Hugging Face if missing) and the same settings as a profile. Only tuning knobs are imported.
      ${r && r.source === "bundled" ? " <b>Offline:</b> showing the copy that shipped with this version." : ""}</div>
    <div class="rlist">${list.length ? list.map(card).join("") : `<div class="note">No recipes found.</div>`}</div>
    <div class="note rfoot" style="display:flex;gap:12px">
      <a href="${SHARE_URL}" target="_blank" rel="noopener">Share yours &#8599;</a>
      <a href="#" id="gal-refresh">refresh</a></div>`);
  $("#gal-refresh").onclick = ev => { ev.preventDefault(); openGallery(true); };
  document.querySelectorAll("[data-gal-import]").forEach(b => {
    b.onclick = () => { const e = list[+b.dataset.galImport]; m.close(); openImport(JSON.stringify(e.recipe, null, 2)); };
  });
}

function openImport(prefill = "") {
  stopPoll();
  const m = showModal("Import a recipe", `
    <div class="note" style="margin-top:0">Paste a LlamaForge recipe. It becomes a preset and a profile. Unsafe knobs
      (paths, hosts, keys) are dropped.</div>
    <textarea id="prof-paste" spellcheck="false" placeholder='{"llamaforge_recipe": 1, ...}' style="width:100%;height:220px;margin-top:12px;font-family:var(--mono,monospace);font-size:11px"></textarea>
    <div id="prof-imp-msg" class="note"></div>
    <div style="display:flex;gap:8px;margin-top:14px"><button class="primary" id="prof-import">Import</button></div>`);
  const msg = $("#prof-imp-msg"), btn = $("#prof-import");
  const say = html => setHTML(msg, html);
  $("#prof-paste").focus();
  if (prefill) { $("#prof-paste").value = prefill; }

  const done = r => {
    const notes = [];
    if (r.engine_missing) notes.push(`it was made on llama.cpp ${r.engine_missing}, which isn't installed here, so it will use your active build`);
    if (r.dropped && r.dropped.length) notes.push(`skipped ${r.dropped.length} knob(s) your engine doesn't support or that aren't safe to import: ${r.dropped.join(", ")}`);
    toast(`Imported profile "${r.name}"`, "ok");
    emit("refresh", true);
    if (notes.length) say(`<b>Imported "${esc(r.name)}".</b> Note: ${esc(notes.join("; "))}.`);
    else m.close();
  };

  const watch = () => {
    stopPoll();
    importPoll = setInterval(async () => {
      if (!m.isOpen()) { stopPoll(); return; }
      const p = await api("/api/hub/progress");
      if (!p) return;
      if (p.phase === "done" && p.finished_path) {
        stopPoll();
        say("Downloaded. Registering the model...");
        await api("/api/hub/add", {path: p.finished_path});
        run(false);
      } else if (["failed", "cancelled", "paused"].includes(p.phase)) {
        stopPoll();
        say(`Download ${esc(p.phase)}${p.error ? ": " + esc(p.error) : ""}. You can resume it from Discover, then import again.`);
        btn.disabled = false;
      } else {
        const pct = p.total ? Math.floor(100 * p.downloaded / p.total) : 0;
        say(`Downloading <b>${esc(p.file || "")}</b>${p.total_files > 1 ? ` (file ${p.done_files + 1}/${p.total_files})` : ""}: ${pct}%`);
      }
    }, 1500);
  };

  const run = async download => {
    let recipe;
    try { recipe = JSON.parse($("#prof-paste").value); }
    catch { say("That isn't valid JSON - copy the whole recipe, braces included."); return; }
    btn.disabled = true;
    const r = await api("/api/profiles/import", {recipe, download});
    if (r.ok) { btn.disabled = false; done(r); return; }
    if (r.missing) {
      if (download && r.downloading) { say("Starting download..."); watch(); return; }
      btn.disabled = false;
      if (download) { say(esc(r.error || "download failed")); return; }
      const mm = r.missing;
      say(`<b>${esc(mm.file)}</b> isn't on this machine.` + (mm.hf_repo
        ? ` <button id="prof-dl" style="margin-left:8px">Download from ${esc(mm.hf_repo)} &amp; import</button>`
        : " The recipe doesn't say where it came from: download it yourself, then import again."));
      const dl = $("#prof-dl");
      if (dl) dl.onclick = () => run(true);
      return;
    }
    btn.disabled = false;
    say(esc(r.error || "import failed"));
  };
  btn.onclick = () => run(false);
  if (prefill) run(false);
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
    const share = e.target.closest("#profiles [data-prof-share]");
    if (share) { e.stopPropagation(); openShare(share.dataset.profShare); return; }
    if (e.target.closest("#profiles [data-prof-import]")) { openImport(); return; }
    if (e.target.closest("#profiles [data-prof-gallery]")) { openGallery(); return; }
    const chip = e.target.closest("#profiles [data-prof-launch]");
    if (chip) launch(chip.dataset.profLaunch);
  });
}
