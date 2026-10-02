// Prebuilt engine card: install / update / roll back official ggml-org
// llama.cpp builds, no compiler needed. Mounted at the top of the Build tab
// and inside the first-run wizard, so it owns its own polling and imports
// neither view.
import { $, esc, setHTML, api, toast, meter, fmtAgo } from "./core.js";

const PHASES = {
  resolving: "finding the right build", downloading: "downloading",
  verifying: "verifying sha256", extracting: "extracting",
  testing: "test-running llama-server", activating: "starting the router",
};
const fmtMB = b => (b / 1e6).toFixed(0) + " MB";

export async function mountEngineCard(el, opts = {}) {
  if (!el) return;
  const force = !!opts.force;
  const prev = el.dataset.channel;
  const q = `?${force ? "force=1&" : ""}${prev ? "channel=" + encodeURIComponent(prev) : ""}`;
  const [info, st] = await Promise.all([api("/api/engine/prebuilt" + q),
                                        api("/api/engine/prebuilt/status")]);
  const L = info.latest || {};
  const installs = info.installs || [];
  const cur = installs.find(i => i.active);
  el.dataset.channel = info.channel;

  let headline, button;
  if (!L.ok) {
    headline = `<span class="v bad">can't reach GitHub releases: ${esc(L.error || "?")}</span>`;
    button = "";
  } else if (cur && !info.update_available) {
    headline = `<span class="v ok">up to date &middot; ${esc(cur.tag)} &middot; ${esc(cur.variant)}</span>`;
    button = `<button class="ghost" id="eng-install">Reinstall</button>`;
  } else if (cur) {
    headline = `<span class="v bad">${esc(L.tag)} available (you have ${esc(cur.tag)})</span>`;
    button = `<button class="primary" id="eng-install">Update to ${esc(L.tag)}</button>`;
  } else {
    headline = `<span class="v">${info.active_bin ? "using your own build" : "no engine installed yet"}</span>`;
    button = `<button class="primary" id="eng-install">Install llama.cpp ${esc(L.tag)} &middot; ${esc(fmtMB(L.download_bytes || 0))}</button>`;
  }
  const variants = (L.alternatives || []).map(v =>
    `<option value="${esc(v)}">${esc(v)}${v === L.variant ? " (recommended)" : ""}</option>`).join("");

  setHTML(el, `<div class="card engine-card"><h3>Engine &middot; official llama.cpp builds</h3>
    <div class="kv"><span class="k">status</span>${headline}</div>
    ${L.ok ? `<div class="kv"><span class="k">for this PC</span><span class="v">${esc(L.variant)} &middot; ${esc(L.reason)}</span></div>` : ""}
    <div class="kv"><span class="k">channel</span><span class="v">
      <select id="eng-channel">
        <option value="nightly" ${info.channel === "nightly" ? "selected" : ""}>nightly: every upstream merge, new models day 0</option>
        <option value="stable" ${info.channel === "stable" ? "selected" : ""}>stable: curated llama.cpp releases</option>
      </select>
      ${L.label && L.label !== L.tag ? `<span class="note" style="margin:0 0 0 6px">${esc(L.label)} = ${esc(L.tag)}</span>` : ""}
    </span></div>
    ${opts.compact ? "" : `<div class="kv"><span class="k">variant</span><span class="v">
      <select id="eng-variant"><option value="">auto (${esc(L.variant || "?")})</option>${variants}</select></span></div>`}
    <div class="actions" style="margin-top:6px">${button}
      ${opts.compact ? "" : `<button class="ghost" id="eng-refresh">Check now</button>`}
      <span class="msg" id="eng-msg"></span></div>
    <div id="eng-progress" style="display:none;margin-top:8px">
      <div class="meter" id="eng-meter"></div>
      <div class="note" id="eng-phase" style="margin:4px 0 0"></div>
    </div>
    <div class="note">Downloads the official binary from github.com/ggml-org/llama.cpp, checks it against
      GitHub's SHA-256, and test-runs it before switching. Old builds stay installed for rollback.</div>
    ${!opts.compact && installs.length ? `<div style="margin-top:8px">${installs.map(i => `
      <div class="kv"><span class="k">${esc(i.tag)} &middot; ${esc(i.variant)}</span>
        <span class="v">${esc(i.channel || "")} &middot; installed ${esc(fmtAgo(i.installed_at))}
        ${i.active ? `<strong class="ok">active</strong>`
                   : `<button class="ghost" data-use="${esc(i.dir)}" style="padding:2px 8px">Use</button>`}</span></div>`).join("")}</div>` : ""}
    ${opts.compact ? "" : `<div class="log" id="eng-log" style="display:none"></div>`}
  </div>`);

  $("#eng-channel", el).onchange = e => { el.dataset.channel = e.target.value; mountEngineCard(el, opts); };
  const ref = $("#eng-refresh", el);
  if (ref) ref.onclick = () => { ref.disabled = true; mountEngineCard(el, {...opts, force: true}); };
  const inst = $("#eng-install", el);
  if (inst) inst.onclick = async () => {
    inst.disabled = true;
    const variant = ($("#eng-variant", el) || {}).value || "";
    const r = await api("/api/engine/prebuilt/install", {channel: info.channel, variant});
    if (!r.started) toast("an install is already running", "err");
    poll(el, opts);
  };
  el.querySelectorAll("[data-use]").forEach(b => b.onclick = async () => {
    b.disabled = true;
    const r = await api("/api/engine/prebuilt/use", {dir: b.dataset.use});
    toast(r.ok ? "Switched engine build" : (r.error || "switch failed"), r.ok ? "ok" : "err");
    mountEngineCard(el, opts);
  });
  if (st.running) poll(el, opts);
}

function poll(el, opts) {
  clearInterval(el._poll);
  const tick = async () => {
    const s = await api("/api/engine/prebuilt/status");
    const box = $("#eng-progress", el), msg = $("#eng-msg", el), log = $("#eng-log", el);
    if (!box) { clearInterval(el._poll); return; }
    box.style.display = "";
    setHTML($("#eng-meter", el), meter(s.downloaded, Math.max(s.total, 1)));
    const what = PHASES[s.phase] || s.phase;
    $("#eng-phase", el).textContent = s.phase === "downloading"
      ? `${what} ${s.file} · ${fmtMB(s.downloaded)} / ${fmtMB(s.total)}` : what;
    if (log) { log.style.display = ""; log.textContent = s.log || ""; log.scrollTop = log.scrollHeight; }
    if (s.running) { msg.className = "msg work"; msg.textContent = "installing..."; return; }
    clearInterval(el._poll);
    if (s.phase === "done") {
      toast(`llama.cpp ${s.tag} installed`, "ok");
      if (opts.onDone) opts.onDone(s);
      mountEngineCard(el, opts);
    } else if (s.phase === "failed") {
      msg.className = "msg err"; msg.textContent = s.error || "install failed";
      const btn = $("#eng-install", el); if (btn) btn.disabled = false;
    } else {
      msg.className = "msg"; msg.textContent = s.phase;
    }
  };
  tick();
  el._poll = setInterval(tick, 1000);
}
