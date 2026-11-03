// Build / Update: "Try a llama.cpp PR". Look a PR up, build it beside your own
// build, pin a model to it. The router never switches to a PR build; only
// the models you pin run on it, each in its own process. GET /api/prtry.
import { $, $$, esc, setHTML, api, toast, askYes, fmtAgo } from "./core.js";
import { models } from "./state.js";
import { lookCard, phaseLine, pinChoices } from "./prview.js";

let poll = null;
let looked = null;     // the PR the user looked up, pinned to its SHA

export async function mountPrTry(el) {
  if (!el) return;
  clearInterval(poll);
  const d = await api("/api/prtry");
  draw(el, d);
  if (d.status && d.status.running) {
    poll = setInterval(async () => {
      if (!el.isConnected) return clearInterval(poll);
      const n = await api("/api/prtry");
      if (n.status && n.status.running) return tick(el, n);
      clearInterval(poll);
      if (n.status && n.status.phase === "done") toast(`PR #${n.status.pr} built`, "ok");
      draw(el, n);
    }, 2000);
  }
}

function tick(el, d) {
  const m = $("#pr-phase", el), log = $("#pr-log", el);
  if (m) m.textContent = phaseLine(d.status);
  if (log) { log.textContent = d.log || ""; log.scrollTop = log.scrollHeight; }
}

function lookHTML(running) {
  if (!looked) return "";
  const c = lookCard(looked);
  return `<div class="pr-look">
    <div class="pr-look-h"><a href="${esc(looked.url)}" target="_blank" rel="noopener">${esc(c.heading)}</a></div>
    <div class="note" style="margin:2px 0 0">${esc(c.byline)}</div>
    ${c.warn ? `<div class="note warn" style="margin:4px 0 0">${esc(c.warn)}</div>` : ""}
    <div class="actions" style="margin-top:6px">
      <button class="primary" id="pr-build" ${running ? "disabled" : ""}>Build PR #${esc(looked.number)}</button>
    </div></div>`;
}

function installHTML(i) {
  const opts = pinChoices(models(), i).map(o =>
    `<option value="${esc(o.id)}" ${o.pinned ? "disabled" : ""}>${esc(o.id)}${o.pinned ? " (on it)" : ""}</option>`).join("");
  return `<div class="pr-inst">
    <div class="kv"><span class="k"><a href="${esc(i.url)}" target="_blank" rel="noopener">${esc(i.label || i.name)}</a></span>
      <span class="v">${esc(String(i.sha || "").slice(0, 7))} &middot; built ${esc(fmtAgo(i.installed_at))}
        ${i.ok ? "" : `<strong class="bad">no llama-server</strong>`}</span></div>
    <div class="kv"><span class="k">pinned</span><span class="v">${i.models && i.models.length
      ? i.models.map(m => `${esc(m)} <button class="ghost pr-unpin" data-model="${esc(m)}" style="padding:1px 6px">unpin</button>`).join(" ")
      : `<span class="dim">no model yet</span>`}</span></div>
    <div class="actions" style="margin-top:4px">
      ${opts ? `<select class="pr-pick" data-build="${esc(i.name)}"><option value="">pick a model...</option>${opts}</select>
      <button class="ghost pr-pin" data-build="${esc(i.name)}">Run it on this build</button>` : `<span class="note" style="margin:0">add a model first</span>`}
      <button class="ghost danger pr-remove" data-build="${esc(i.name)}">Remove</button>
    </div></div>`;
}

function draw(el, d) {
  const st = d.status || {};
  const installs = d.installs || [];
  const showLog = st.running || ["failed", "cancelled"].includes(st.phase);
  setHTML(el, `<div class="card pr-try"><h3>Try a llama.cpp pull request</h3>
    <div class="note" style="margin:0 0 6px">Someone's PR makes your GPU faster or adds a model? Build it beside your
      own build and try it on one model. Your router keeps running your build.</div>
    <div class="actions">
      <input id="pr-input" type="text" spellcheck="false" autocomplete="off" style="min-width:300px"
        placeholder="PR number or github.com/ggml-org/llama.cpp/pull/... link" aria-label="llama.cpp PR number or link">
      <button class="ghost" id="pr-lookup">Look it up</button>
      <span class="msg" id="pr-msg"></span>
    </div>
    <div id="pr-lookbox">${lookHTML(st.running)}</div>
    ${st.phase && st.phase !== "idle" ? `<div class="actions" style="margin-top:6px">
      <span class="msg ${st.running ? "work" : st.phase === "done" ? "ok" : "err"}" id="pr-phase">${esc(phaseLine(st))}</span>
      ${st.running ? `<button class="ghost" id="pr-cancel">Cancel</button>` : ""}</div>` : ""}
    ${showLog ? `<div class="log" id="pr-log">${esc(d.log || "")}</div>` : ""}
    ${installs.length ? `<div style="margin-top:8px">${installs.map(installHTML).join("")}</div>` : ""}
    <div class="note">Fetches only that PR's commit from github.com/ggml-org/llama.cpp and builds it with your
      Build Flags (needs the same git, CMake and compiler as building from source). The commit is pinned when
      you look it up, so a push in between stops the build.</div>
  </div>`);
  const log = $("#pr-log", el);
  if (log) log.scrollTop = log.scrollHeight;
  wire(el, d);
}

function wire(el, d) {
  const input = $("#pr-input", el), msg = $("#pr-msg", el);
  const lookup = async () => {
    const pr = input.value.trim();
    if (!pr) return input.focus();
    msg.className = "msg work"; msg.textContent = "asking GitHub...";
    const r = await api("/api/prtry/look", {pr});
    if (r.error) { msg.className = "msg err"; msg.textContent = r.error; return; }
    looked = r; msg.textContent = "";
    setHTML($("#pr-lookbox", el), lookHTML(d.status && d.status.running));
    wireBuild(el);
  };
  $("#pr-lookup", el).onclick = lookup;
  input.onkeydown = e => { if (e.key === "Enter") lookup(); };
  wireBuild(el);
  const cancel = $("#pr-cancel", el);
  if (cancel) cancel.onclick = async () => { cancel.disabled = true; await api("/api/prtry/cancel", {}); };
  for (const b of $$(".pr-pin", el)) b.onclick = async () => {
    const sel = $(`.pr-pick[data-build="${CSS.escape(b.dataset.build)}"]`, el);
    if (!sel || !sel.value) return toast("pick a model first", "err");
    b.disabled = true;
    const r = await api("/api/model/build", {model: sel.value, build: b.dataset.build});
    if (r.error) { b.disabled = false; return toast(r.error, "err"); }
    toast(`${sel.value} now runs on ${b.dataset.build}. Load it to try it.`, "ok");
    mountPrTry(el);
  };
  for (const b of $$(".pr-unpin", el)) b.onclick = async () => {
    b.disabled = true;
    const r = await api("/api/model/build", {model: b.dataset.model, build: ""});
    if (r.error) { b.disabled = false; return toast(r.error, "err"); }
    toast(`${b.dataset.model} is back on your build`, "ok");
    mountPrTry(el);
  };
  for (const b of $$(".pr-remove", el)) b.onclick = async () => {
    const ok = await askYes(`Delete ${b.dataset.build}? Models pinned to it are unloaded and go back to your build.`,
                            {title: "Remove PR build", ok: "Remove", danger: true});
    if (!ok) return;
    b.disabled = true;
    const r = await api("/api/prtry/remove", {name: b.dataset.build});
    if (r.error) { b.disabled = false; return toast(r.error, "err"); }
    toast("PR build removed", "ok");
    mountPrTry(el);
  };
}

function wireBuild(el) {
  const b = $("#pr-build", el);
  if (!b || !looked) return;
  b.onclick = async () => {
    b.disabled = true;
    const r = await api("/api/prtry/build", {pr: looked.number, sha: looked.sha});
    if (r.error) { b.disabled = false; return toast(r.error, "err"); }
    looked = null;
    mountPrTry(el);
  };
}
