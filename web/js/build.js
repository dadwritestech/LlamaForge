// Build tab: current commit vs upstream, CMake flags, rebuild, log.
// Supports both llama.cpp and ik_llama build targets via a toggle.
// Also surfaces the vLLM pip package version, since updating it is a build-ish
// concern rather than a setup one.
import { $, esc, setHTML, api, toast, agoText, fmtDur } from "./core.js";

let buildPoll = null;
let _target = localStorage.getItem("build_target") || "llamacpp";

function setTarget(t) {
  _target = t;
  localStorage.setItem("build_target", t);
}

const ENGINE_LABELS = {
  llamacpp: "llama.cpp",
  ikllama: "ik_llama",
};

const ENGINE_REPOS = {
  llamacpp: "github.com/ggml-org/llama.cpp",
  ikllama: "github.com/ikawrakow/ik_llama.cpp",
};

export async function loadBuild(force) {
  const v = $("#view-build");
  const label = ENGINE_LABELS[_target] || _target;
  if (force) {
    const s = $("#upstream-status");
    if (s) { s.textContent = "checking github..."; s.className = "v work"; }
  } else setHTML(v, `<div class="skel"><div class="card"><span class="spinner"></span><div class="log card"><span class="log-item">Querying ${esc(label)} git...\n</span></div></div></div>`);
  const q = (force ? "?force=1&" : "?") + `target=${_target}`;
  
  const b = await api("/api/build/info" + q);
  
  if (!force) {
    const logContainer = $(".log", v);
    if (logContainer) {
      logContainer.innerHTML += `<span class="log-item">Validating local ${esc(label)} build state...\n</span>`;
    }
  }
  
  const st = await api("/api/state");
  const activeEngine = st.active_engine || "llamacpp";
  
  if (!force) {
    const logContainer = $(".log", v);
    if (logContainer) {
      logContainer.innerHTML += '<span class="log-item">Validating local vLLM state...\n</span>';
    }
  }
  
  const vver = await api("/api/vllm/version" + (force ? "?force=1" : ""));
  const cur = b.current||{}, up = b.updates||{};
  const flags = b.saved_flags && Object.keys(b.saved_flags).length ? b.saved_flags : b.recommended_flags||{};
  const behind = up.ok ? up.behind : 0;
  const checked = up.cached ? `checked ${agoText(up.checked_secs_ago)}` : "checked just now";
  const remoteUrl = b.remote || ENGINE_REPOS[_target] || "";
  const isActive = _target === activeEngine;

  setHTML(v, `
    <div class="card buildtarget">
      <span class="k">Build Target</span>
      <span class="mode-toggle">
        <button class="${_target==='llamacpp'?'active':''}" id="btn-tgt-llamacpp">llama.cpp</button>
        <button class="${_target==='ikllama'?'active':''}" id="btn-tgt-ikllama">ik_llama</button>
      </span>
      <span class="buildtarget-active">
        Active engine: <strong class="${isActive?'ok':'dim'}">${esc(ENGINE_LABELS[activeEngine]||activeEngine)}</strong>
        ${!isActive?`<button class="ghost" id="btn-switch-engine">Switch to ${esc(label)}</button>`:''}
      </span>
    </div>
    <div class="card"><h3>Current Build · ${esc(label)}</h3>
      <div class="kv"><span class="k">commit</span><span class="v">${esc(cur.hash||"?")} &middot; ${esc((cur.subject||"").slice(0,60))}</span></div>
      <div class="kv"><span class="k">branch</span><span class="v">${esc(cur.branch||"?")}</span></div>
      <div class="kv"><span class="k">date</span><span class="v">${esc(cur.date||"?")}</span></div>
    </div>
    <div class="card"><h3>Upstream (${esc(remoteUrl)})</h3>
      <div class="kv"><span class="k">status</span><span class="v ${behind>0?'bad':'ok'}" id="upstream-status">${up.ok?(behind>0?behind+" commits behind":"up to date"):"check failed"}</span></div>
      ${up.latest?`<div class="kv"><span class="k">latest</span><span class="v">${esc(up.latest.hash)} &middot; ${esc((up.latest.subject||"").slice(0,60))}</span></div>`:""}
      <div class="actions" style="margin-top:6px">
        <button class="ghost" id="btn-refresh-upstream">Check GitHub now</button>
        <span class="note" style="margin:0">${esc(checked)} &middot; auto-checks at most every 15 min</span>
      </div>
    </div>
    <div class="card"><h3>Build Flags · ${esc(label)}</h3>
      <div class="flags">${Object.entries(flags).map(([k,val])=>`<span class="flagpill">${esc(k)}=${esc(val)}</span>`).join("")}</div>
      <div class="actions">
        <button class="primary" id="btn-build">${behind>0?"Pull latest &amp; Rebuild":"Rebuild current"}</button>
        <label style="font-size:11px;color:var(--dim)"><input type="checkbox" id="opt-pull" ${behind>0?"checked":""}> git pull first</label>
        <span class="msg" id="build-msg"></span>
      </div>
      <div class="note">Rebuilds ${esc(label)} with CMake. Prior binaries are backed up first. Takes several minutes; watch the log below.</div>
    </div>
    <div class="card"><h3>Build Log · ${esc(label)}</h3><div class="log" id="build-log">idle</div></div>`
    + (vver.error ? "" : `<div class="card"><h3>vLLM (pip package in WSL)</h3>
      <div class="kv"><span class="k">installed</span><span class="v ${vver.installed&&vver.installed.present?'ok':'bad'}">${vver.installed&&vver.installed.present?"v"+esc(vver.installed.version):"not installed (see Setup)"}</span></div>
      <div class="kv"><span class="k">latest on PyPI</span><span class="v">${esc(vver.latest||"?")}</span></div>
      ${vver.installed&&vver.installed.present&&vver.latest&&vver.latest!==vver.installed.version?`<div class="actions"><button class="primary" id="btn-vllm-update">Update vLLM to ${esc(vver.latest)}</button><span class="msg" id="vllm-upd-msg"></span></div>`:`<div class="note">${vver.installed&&vver.installed.present?"vLLM is up to date.":"Install vLLM from the Setup tab first."}</div>`}
      <div class="log" id="vllm-update-log" style="display:none">idle</div>
    </div>`));

  // Target toggle
  $("#btn-tgt-llamacpp").onclick = () => { setTarget("llamacpp"); loadBuild(); };
  $("#btn-tgt-ikllama").onclick = () => { setTarget("ikllama"); loadBuild(); };

  // Engine switch
  const switchBtn = $("#btn-switch-engine");
  if (switchBtn) switchBtn.onclick = async () => {
    switchBtn.disabled = true;
    switchBtn.textContent = "switching...";
    const r = await api("/api/engine/switch", {engine: _target});
    if (r.ok) {
      toast(`Switched to ${label}`, "ok");
      setTimeout(loadBuild, 1500);
    } else {
      toast(r.error || "switch failed", "err");
      switchBtn.disabled = false;
      switchBtn.textContent = `Switch to ${label}`;
    }
  };

  $("#btn-build").onclick = startBuild;
  const refBtn = $("#btn-refresh-upstream");
  if (refBtn) refBtn.onclick = () => { refBtn.disabled = true; loadBuild(true); };
  pollBuild();
  const updBtn = $("#btn-vllm-update");
  if (updBtn) updBtn.onclick = async () => {
    const msg = $("#vllm-upd-msg"); msg.className = "msg work"; msg.textContent = "starting update...";
    const r = await api("/api/vllm/update");
    if (r.started) {
      toast("vLLM update started", "ok");
      $("#vllm-update-log").style.display = "";
      const iv = setInterval(async () => {
        const s = await api("/api/vllm/setup");
        const l = $("#vllm-update-log");
        if (l) { l.textContent = s.setup_log||""; l.scrollTop = l.scrollHeight; }
        if (s.setup_job && !s.setup_job.running) {
          clearInterval(iv); msg.className = "msg ok"; msg.textContent = "done";
          setTimeout(loadBuild, 1200);
        }
      }, 2000);
    } else msg.textContent = "a job is already running";
  };
}

async function startBuild() {
  const pull = $("#opt-pull").checked, msg = $("#build-msg");
  msg.className = "msg work"; msg.textContent = "starting build...";
  const r = await api("/api/build/start", {pull, target: _target});
  if (r.started) toast(`Build started (${ENGINE_LABELS[_target]})`, "ok");
  else msg.textContent = "a build is already running";
  pollBuild();
}

async function pollBuild() {
  clearInterval(buildPoll);
  const tick = async () => {
    const s = await api("/api/build/log?target=" + _target);
    const log = $("#build-log");
    if (log) { log.textContent = s.log||"idle"; log.scrollTop = log.scrollHeight; }
    const msg = $("#build-msg");
    if (msg && s.running) { msg.className = "msg work"; msg.textContent = "building: " + s.phase; }
    else if (msg && s.phase === "done") {
      msg.className = "msg ok";
      msg.textContent = "build OK" + (s.started&&s.finished?` in ${fmtDur(s.finished-s.started)}`:"");
      clearInterval(buildPoll);
    } else if (msg && s.phase === "done_warnings") {
      // llama-server built, but a non-essential later target (UI assets) failed.
      msg.className = "msg warn";
      msg.textContent = "built with warnings - " + (s.warning || "see log");
      clearInterval(buildPoll);
    } else if (msg && s.phase === "failed") {
      msg.className = "msg err"; msg.textContent = "build failed - see log";
      clearInterval(buildPoll);
    }
  };
  await tick();
  buildPoll = setInterval(tick, 2000);
}
