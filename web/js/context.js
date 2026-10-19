// Context tab: markdown context docs, named profiles composed from them, the
// per-model active profile, and export into an agent's CLAUDE.md / AGENTS.md.
import { $, esc, setHTML, api, toast, askText } from "./core.js";
import { S, models } from "./state.js";

export async function loadContext() {
  const v = $("#view-context");
  // Deep-linking straight to #context renders before the first /api/state poll
  // lands, which left "Active profile per model" with an empty dropdown and no
  // way to set anything until you visited another tab. Fetch state ourselves
  // when it isn't there yet rather than rendering a knowingly empty control.
  const [d, pr] = await Promise.all([
    api("/api/wiki/docs"),
    api("/api/wiki/profiles"),
    S.STATE ? null : api("/api/state").then(s => { if (s && !s.error) S.STATE = s; }),
  ]);
  const docs = d.docs || [], profiles = pr.profiles || {};
  const ids = models().map(m => m.id);
  setHTML(v, `
    <div class="card"><h3>Context docs</h3>
      <div class="formrow">
        <label class="f grow"><span class="lbl">Document</span>
          <select id="wk-doc">${docs.map(n=>`<option>${esc(n)}</option>`).join("")}</select></label>
        <button id="wk-new">New</button>
        <button id="wk-del">Delete</button>
      </div>
      <textarea id="wk-text" rows="12" aria-label="Document content" placeholder="${docs.length?"":"No documents yet - hit New to create one."}"></textarea>
      <div class="actions"><button id="wk-save" class="primary">Save doc</button></div>
    </div>
    <div class="card"><h3>Profiles</h3>
      <div class="formrow">
        <label class="f grow"><span class="lbl">Profile name</span>
          <input id="wk-pname" placeholder="e.g. coding"></label>
        <label class="f grow"><span class="lbl">Documents in profile</span>
          ${docs.length
            ? `<select id="wk-pdocs" multiple size="4">${docs.map(n=>`<option>${esc(n)}</option>`).join("")}</select>`
            : `<select id="wk-pdocs" multiple size="4" disabled></select>`}</label>
        <button id="wk-psave" class="primary">Save profile</button>
      </div>
      <div class="note">${docs.length?"Ctrl-click to select more than one.":"Create a context doc above first."}</div>
      <div id="wk-plist" class="note">Saved: ${Object.keys(profiles).map(esc).join(", ")||"(none)"}</div>
    </div>
    <div class="card"><h3>Active profile per model</h3>
      <div class="formrow">
        <label class="f grow"><span class="lbl">Model</span>
          <select id="wk-model">${ids.map(m=>`<option>${esc(m)}</option>`).join("")}</select></label>
        <label class="f grow"><span class="lbl">Profile</span>
          <select id="wk-active"><option value="">(none)</option>${Object.keys(profiles).map(n=>`<option>${esc(n)}</option>`).join("")}</select></label>
        <button id="wk-setactive">Set active</button>
      </div>
    </div>
    <div class="card"><h3>Export to agent file</h3>
      <div class="formrow">
        <label class="f grow"><span class="lbl">Agent</span>
          <select id="wk-eagent"><option value="claude-code">Claude Code (CLAUDE.md)</option><option value="codex">Codex (AGENTS.md)</option><option value="pi">pi.dev (AGENTS.md)</option></select></label>
        <label class="f grow"><span class="lbl">Profile</span>
          <select id="wk-eprofile">${Object.keys(profiles).map(n=>`<option>${esc(n)}</option>`).join("")}</select></label>
        <label class="f grow"><span class="lbl">Project path</span>
          <input id="wk-epath" placeholder="(optional) blank = global"></label>
        <button id="wk-export">Export</button>
      </div>
    </div>`);
  const load = async () => {
    const n = $("#wk-doc").value;
    if (n) { const r = await api("/api/wiki/doc?name=" + encodeURIComponent(n)); $("#wk-text").value = r.text || ""; }
  };
  load();
  $("#wk-doc").onchange = load;
  $("#wk-new").onclick = async () => {
    const n = await askText("Doc name", {title: "New doc", ok: "Create", placeholder: "style"});
    if (n) {
      $("#wk-text").value = "";
      $("#wk-doc").insertAdjacentHTML("beforeend", `<option selected>${esc(n.endsWith(".md")?n:n+".md")}</option>`);
    }
  };
  $("#wk-save").onclick = async () => {
    const n = $("#wk-doc").value;
    if (!n) { toast("Pick or create a doc", "err"); return; }
    await api("/api/wiki/doc", {name: n, text: $("#wk-text").value});
    toast("Saved", "ok"); loadContext();
  };
  $("#wk-del").onclick = async () => {
    const n = $("#wk-doc").value;
    if (n) { await api("/api/wiki/doc/delete", {name: n}); toast("Deleted", "ok"); loadContext(); }
  };
  $("#wk-psave").onclick = async () => {
    const name = $("#wk-pname").value.trim();
    const chosen = [...$("#wk-pdocs").selectedOptions].map(o => o.value);
    if (!name) { toast("Profile name required", "err"); return; }
    await api("/api/wiki/profile", {name, docs: chosen});
    toast("Profile saved", "ok"); loadContext();
  };
  $("#wk-setactive").onclick = async () => {
    await api("/api/wiki/active", {model: $("#wk-model").value, profile: $("#wk-active").value});
    toast("Active profile set", "ok");
  };
  $("#wk-export").onclick = async () => {
    const r = await api("/api/wiki/export", {
      agent: $("#wk-eagent").value, profile: $("#wk-eprofile").value,
      path: $("#wk-epath").value.trim()});
    if (r.error) { toast(r.error, "err"); return; }
    toast(`${r.action}: ${r.path}`, "ok");
  };
}
