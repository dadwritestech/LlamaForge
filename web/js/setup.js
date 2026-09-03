// Setup tab: prerequisites, detected hardware, drive scanning, startup options,
// LAN access, the agent-connect panel, and vLLM/WSL installation.
import { $, $$, esc, setHTML, api, toast } from "./core.js";
import { models, config as cfgOf } from "./state.js";
import { emit } from "./bus.js";

let vllmSetupPoll = null;

function pollVllmSetup() {
  clearInterval(vllmSetupPoll);
  const log = $("#vllm-setup-log"); if (log) log.style.display = "";
  const tick = async () => {
    const s = await api("/api/vllm/setup");
    const l = $("#vllm-setup-log");
    if (l) { l.textContent = s.setup_log||"idle"; l.scrollTop = l.scrollHeight; }
    const msg = $("#vllm-inst-msg");
    const job = s.setup_job || {};
    if (job.running) { if (msg) { msg.className = "msg work"; msg.textContent = "installing..."; } }
    else if (job.phase === "done") {
      if (msg) { msg.className = "msg ok"; msg.textContent = "installed"; }
      clearInterval(vllmSetupPoll); toast("vLLM installed", "ok"); setTimeout(loadSetup, 1200);
    } else if (job.phase === "failed") {
      if (msg) { msg.className = "msg err"; msg.textContent = "install failed - see log"; }
      clearInterval(vllmSetupPoll);
    }
  };
  tick();
  vllmSetupPoll = setInterval(tick, 2000);
}

function networkMarkup(net) {
  const local = net.access_scope === "local";
  const hasKey = Boolean(net.has_api_key);
  const initialAction = hasKey ? "keep" : (local ? "clear" : "generate");
  const checked = value => initialAction === value ? " checked" : "";
  const status = net.configured_security_status.replaceAll("_", " ");
  const remediation = net.remediation_required
    ? `${net.message || "The stored network configuration must be repaired."}${
        net.router_running
          ? " A listener occupies the configured port; its process identity and protection cannot be verified."
          : ""}`
    : "";
  return `<section class="card" id="network-access" aria-labelledby="network-title">
    <h3 id="network-title">Network Access</h3>
    ${net.remediation_required ? `<div id="net-alert" class="net-alert" role="alert">
      ${esc(remediation)}
      <div class="actions">
        <button type="button" id="net-remediate-generate">Secure with a generated key</button>
        <button type="button" id="net-remediate-local">Return to local-only</button>
      </div>
    </div>` : ""}
    <div class="net-observation">
      <p><b>Configured:</b> <span id="net-configured">${esc(status)}</span></p>
      <p><b>Observed listener:</b> <span id="net-runtime">${
        net.router_running ? "listening on the configured port" : "not listening"}</span></p>
      <p class="note">A listening port does not prove process identity or authentication.
        LAN address: <b>http://${esc(net.lan_ip || "<lan-ip>")}:${esc(net.port)}/</b></p>
    </div>
    <fieldset id="net-scope-group" aria-describedby="net-scope-help net-apply-error">
      <legend>Access scope</legend>
      <label><input type="radio" name="net-scope" value="local"${
        local ? " checked" : ""}> This computer only — recommended</label>
      <label><input type="radio" name="net-scope" value="lan"${
        local ? "" : " checked"}> Devices on my local network</label>
      <p class="note" id="net-scope-help">LAN access always requires a usable API key.</p>
    </fieldset>
    <fieldset id="net-auth-group" aria-describedby="net-auth-help net-apply-error">
      <legend>Authentication</legend>
      <label><input type="radio" name="net-key-action" value="keep"${
        checked("keep")} ${hasKey ? "" : "disabled"}> Keep the configured key</label>
      <label><input type="radio" name="net-key-action" value="generate"${
        checked("generate")}> Generate a new strong key</label>
      <label><input type="radio" name="net-key-action" value="replace"${
        checked("replace")}> Replace with a key I provide</label>
      <label><input type="radio" name="net-key-action" value="clear"${
        checked("clear")}> Remove the key (local-only)</label>
      <p class="note" id="net-auth-help">Blank input never means keep or remove.
        Rotating a key requires updating every router client after restart.</p>
      <div class="fld" id="net-replace-wrap" hidden>
        <label for="net-replace-key">Replacement API key</label>
        <input id="net-replace-key" type="password" autocomplete="new-password"
               minlength="32" maxlength="256"
               aria-describedby="net-replace-help net-apply-error">
        <div class="hint" id="net-replace-help">32–256 URL-safe characters:
          letters, digits, dot, underscore, tilde, or hyphen.</div>
      </div>
    </fieldset>
    <div class="actions">
      <button type="button" class="primary" id="net-apply"
              aria-describedby="net-apply-error">Apply &amp; Restart Router</button>
      <span id="net-status" class="msg" role="status" aria-live="polite"></span>
    </div>
    <div id="net-apply-error" class="note" role="alert" hidden></div>
    <div id="net-generated" class="net-generated" hidden>
      <div>Generated API key — copy it now</div>
      <code id="net-generated-display" aria-label="Generated API key masked">••••••••••••••••</code>
      <button type="button" id="net-reveal-generated" aria-pressed="false">Reveal for 30 seconds</button>
      <button type="button" id="net-copy-generated">Copy</button>
      <button type="button" id="net-dismiss-generated">Done</button>
      <div class="note">Copy does not render plaintext. Reveal expires after 30 seconds.
        Retry, Done, navigation, or Setup rerender clears this one-time value.</div>
    </div>
  </section>`;
}

function confirmNetworkChange(title, message, confirmLabel) {
  return new Promise(resolve => {
    const root = $("#modal-root");
    const returnTo = document.activeElement;
    setHTML(root, `<dialog id="net-confirm" aria-labelledby="net-confirm-title"
      aria-describedby="net-confirm-message">
      <div class="modal">
        <h3 id="net-confirm-title">${esc(title)}</h3>
        <p id="net-confirm-message">${esc(message)}</p>
        <div class="actions">
          <button type="button" id="net-confirm-cancel">Cancel</button>
          <button type="button" class="primary" id="net-confirm-accept">${
            esc(confirmLabel)}</button>
        </div>
      </div>
    </dialog>`);
    const dialog = $("#net-confirm");
    const cancel = $("#net-confirm-cancel");
    const accept = $("#net-confirm-accept");
    let finished = false;
    const finish = answer => {
      if (finished) return;
      finished = true;
      if (dialog.open) dialog.close();
      setHTML(root, "");
      if (returnTo && returnTo.isConnected) returnTo.focus();
      resolve(answer);
    };
    cancel.onclick = () => finish(false);
    accept.onclick = () => finish(true);
    dialog.addEventListener("cancel", event => {
      event.preventDefault();
      finish(false);
    });
    dialog.addEventListener("keydown", event => {
      if (event.key !== "Tab") return;
      if (event.shiftKey && document.activeElement === cancel) {
        event.preventDefault();
        accept.focus();
      } else if (!event.shiftKey && document.activeElement === accept) {
        event.preventDefault();
        cancel.focus();
      }
    });
    dialog.showModal();
    cancel.focus();
  });
}

const STRONG_NETWORK_KEY = /^[A-Za-z0-9._~-]{32,256}$/;

function networkFailureMessage(scope, listenerObserved = false) {
  if (listenerObserved) {
    return scope === "lan"
      ? "A protected configuration was saved, but restart failed; a listener is present and its process identity and protection are not verified."
      : "The local-only configuration was saved, but restart failed; a listener is present and its process identity is not verified.";
  }
  return scope === "lan"
    ? "A protected configuration was saved, but the router is stopped; LAN protection is not currently active or verified."
    : "The local-only configuration was saved, but the router is stopped; its listener is not currently active or verified.";
}

async function pollNetworkRuntime(root, savedScope) {
  const status = $("#net-status", root);
  const runtime = $("#net-runtime", root);
  const error = $("#net-apply-error", root);
  let latest = null;
  for (let attempt = 0; attempt < 10; attempt++) {
    if (!root.isConnected) return;
    let net;
    try {
      net = await api("/api/network");
    } catch (requestError) {
      break;
    }
    if (!root.isConnected) return;
    latest = net;
    runtime.textContent = net.router_running
      ? "listening on the configured port"
      : "not listening";
    if (net.router_running) {
      const protection = String(
        net.configured_security_status || "configured").replaceAll("_", " ");
      const endpointHost = net.access_scope === "lan"
        ? (net.lan_ip || "<LAN-IP>") : "127.0.0.1";
      status.className = "msg ok";
      status.textContent = `Saved ${protection} settings; listener observed at http://${
        endpointHost}:${net.port}/.`;
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  if (root.isConnected) {
    status.className = "msg";
    status.textContent = "";
    error.hidden = false;
    error.textContent = `${networkFailureMessage(latest?.access_scope || savedScope)} ` +
      "Check Router Log for the launch error, then retry Apply.";
  }
}

function wireNetwork(net) {
  const root = $("#network-access");
  if (!root) return;
  const status = $("#net-status", root);
  const error = $("#net-apply-error", root);
  const apply = $("#net-apply", root);
  const replaceWrap = $("#net-replace-wrap", root);
  const replaceInput = $("#net-replace-key", root);
  const generatedPanel = $("#net-generated", root);
  const generatedDisplay = $("#net-generated-display", root);
  const generatedReveal = $("#net-reveal-generated", root);
  const generatedCopy = $("#net-copy-generated", root);
  const generatedDismiss = $("#net-dismiss-generated", root);

  const hideGeneratedPanel = () => {
    generatedDisplay.textContent = "";
    generatedDisplay.setAttribute("aria-label", "Generated API key cleared");
    generatedPanel.hidden = true;
    generatedDismiss.onclick = null;
  };
  let clearGenerated = hideGeneratedPanel;

  const installGeneratedSecret = rawSecret => {
    clearGenerated();
    const privateKey = {value: String(rawSecret)};
    let revealTimer = null;
    let observer = null;
    let leaveSetup = null;

    const destroy = () => {
      if (revealTimer !== null) clearTimeout(revealTimer);
      revealTimer = null;
      privateKey.value = "";
      generatedDisplay.textContent = "";
      generatedDisplay.setAttribute("aria-label", "Generated API key cleared");
      generatedReveal.onclick = null;
      generatedCopy.onclick = null;
      generatedDismiss.onclick = null;
      generatedReveal.disabled = true;
      generatedCopy.disabled = true;
      if (observer) observer.disconnect();
      if (leaveSetup) document.removeEventListener("click", leaveSetup, true);
      leaveSetup = null;
      generatedPanel.hidden = true;
      clearGenerated = hideGeneratedPanel;
    };

    const expire = () => {
      revealTimer = null;
      privateKey.value = "";
      generatedDisplay.textContent = "••••••••••••••••";
      generatedDisplay.setAttribute("aria-label", "Generated API key expired");
      generatedReveal.textContent = "Expired";
      generatedReveal.setAttribute("aria-pressed", "false");
      generatedReveal.disabled = true;
      generatedCopy.disabled = true;
      generatedReveal.onclick = null;
      generatedCopy.onclick = null;
      if (observer) observer.disconnect();
      if (leaveSetup) document.removeEventListener("click", leaveSetup, true);
      leaveSetup = null;
      generatedDismiss.onclick = hideGeneratedPanel;
      clearGenerated = hideGeneratedPanel;
    };

    generatedDisplay.textContent = "••••••••••••••••";
    generatedDisplay.setAttribute("aria-label", "Generated API key masked");
    generatedReveal.textContent = "Reveal for 30 seconds";
    generatedReveal.setAttribute("aria-pressed", "false");
    generatedReveal.disabled = false;
    generatedCopy.disabled = false;
    generatedPanel.hidden = false;

    generatedCopy.onclick = () => {
      if (!privateKey.value) return;
      navigator.clipboard.writeText(privateKey.value).then(() => {
        status.className = "msg ok";
        status.textContent = "Generated key copied.";
      });
    };
    generatedReveal.onclick = () => {
      if (!privateKey.value || revealTimer !== null) return;
      generatedDisplay.textContent = privateKey.value;
      generatedDisplay.setAttribute(
        "aria-label", "Generated API key revealed temporarily");
      generatedReveal.textContent = "Revealed — expires in 30 seconds";
      generatedReveal.setAttribute("aria-pressed", "true");
      generatedReveal.disabled = true;
      revealTimer = setTimeout(expire, 30_000);
    };
    generatedDismiss.onclick = destroy;
    leaveSetup = event => {
      const tab = event.target.closest?.(".tab");
      if (tab && tab.dataset.tab !== "setup") destroy();
    };
    document.addEventListener("click", leaveSetup, true);
    observer = new MutationObserver(() => {
      const view = root.closest(".view");
      if (!root.isConnected || !view || !view.classList.contains("active")) destroy();
    });
    observer.observe(document.body, {
      childList: true, subtree: true, attributes: true, attributeFilter: ["class"],
    });
    clearGenerated = destroy;
  };

  const scope = () => $('[name="net-scope"]:checked', root).value;
  const action = () => $('[name="net-key-action"]:checked', root).value;
  const chooseAction = value => {
    const radio = $(`[name="net-key-action"][value="${value}"]`, root);
    if (radio && !radio.disabled) radio.checked = true;
  };
  let invalidControl = null;
  const clearInvalid = () => {
    if (!invalidControl) return;
    invalidControl.removeAttribute("aria-invalid");
    invalidControl.removeAttribute("aria-errormessage");
    invalidControl = null;
  };
  const showError = (message, focus) => {
    clearInvalid();
    error.hidden = false;
    error.textContent = message;
    status.className = "msg";
    status.textContent = "";
    if (focus) {
      invalidControl = focus;
      focus.setAttribute("aria-invalid", "true");
      focus.setAttribute("aria-errormessage", "net-apply-error");
      focus.focus();
    }
  };
  const clearError = () => {
    clearInvalid();
    error.hidden = true;
    error.textContent = "";
  };
  const sync = () => {
    const lan = scope() === "lan";
    const clear = $('[name="net-key-action"][value="clear"]', root);
    clear.disabled = lan;
    if (lan && clear.checked) chooseAction(net.has_api_key ? "keep" : "generate");
    replaceWrap.hidden = action() !== "replace";
    clearError();
  };

  $$('[name="net-scope"], [name="net-key-action"]', root).forEach(
    control => control.onchange = sync);
  const localRemediation = $("#net-remediate-local", root);
  if (localRemediation) localRemediation.onclick = () => {
    $('[name="net-scope"][value="local"]', root).checked = true;
    sync();
    chooseAction(net.has_api_key ? "keep" : "clear");
    sync();
    apply.focus();
  };
  const generateRemediation = $("#net-remediate-generate", root);
  if (generateRemediation) generateRemediation.onclick = () => {
    $('[name="net-scope"][value="lan"]', root).checked = true;
    chooseAction("generate");
    sync();
    apply.focus();
  };

  apply.onclick = async () => {
    clearError();
    clearGenerated();
    const desiredScope = scope();
    const keyAction = action();
    if (desiredScope === "lan" && keyAction === "keep" && !net.has_api_key) {
      const generate = $('[name="net-key-action"][value="generate"]', root);
      showError("LAN access requires Generate or a valid replacement key.", generate);
      return;
    }
    if (desiredScope === "lan" && keyAction === "clear") {
      const generate = $('[name="net-key-action"][value="generate"]', root);
      showError("A key cannot be removed while LAN access is selected.", generate);
      return;
    }
    const body = {access_scope: desiredScope, key_action: keyAction};
    if (keyAction === "replace") {
      const replacement = replaceInput.value.trim();
      if (!STRONG_NETWORK_KEY.test(replacement)) {
        showError("Enter a 32–256 character URL-safe replacement key.",
                  replaceInput);
        return;
      }
      body.api_key = replacement;
      if (net.has_api_key && !await confirmNetworkChange(
          "Rotate router API key",
          "Existing clients will stop authenticating after restart until you update them.",
          "Rotate key")) return;
    }
    if (keyAction === "generate" && net.has_api_key &&
        !await confirmNetworkChange(
          "Rotate router API key",
          "Generating a new key immediately invalidates the current key after restart.",
          "Generate and rotate")) return;
    if (keyAction === "clear" && net.has_api_key && !await confirmNetworkChange(
        "Remove router API key",
        "Local clients will no longer need a key. LAN mode cannot run without one.",
        "Remove key")) return;

    apply.disabled = true;
    status.className = "msg work";
    status.textContent = "Saving safe policy and restarting router...";
    try {
      const r = await api("/api/network", body);
      if (!root.isConnected) return;
      $("#net-configured", root).textContent =
        String(r.configured_security_status || "saved").replaceAll("_", " ");
      net.has_api_key = Boolean(r.has_api_key);
      $('[name="net-key-action"][value="keep"]', root).disabled =
        !net.has_api_key;
      if (r.generated_api_key) {
        const issued = r.generated_api_key;
        delete r.generated_api_key;
        installGeneratedSecret(issued);
      }
      if (!r.ok) {
        status.className = "msg";
        status.textContent = "";
        error.hidden = false;
        error.textContent = `${networkFailureMessage(
          r.access_scope, Boolean(r.router_running))} ${
          r.error || "Open Router Log for details, then retry Apply."}`;
        $("#net-runtime", root).textContent = r.router_running
          ? "listener present / not verified" : "not running / not verified";
        return;
      }
      await pollNetworkRuntime(root, r.access_scope);
    } catch (requestError) {
      if (root.isConnected) {
        showError("The network change could not be completed.", apply);
      }
    } finally {
      if (root.isConnected) apply.disabled = false;
    }
  };
  sync();
}

export async function loadSetup() {
  const v = $("#view-setup");
  setHTML(v, `<div class="skel">PROBING SYSTEM...</div>`);
  const [s, net, vs] = await Promise.all([api("/api/setup"), api("/api/network"), api("/api/vllm/setup")]);
  const p = s.prereqs, hw = s.hardware;
  const toolRow = (name, t) => `<div class="kv"><span class="k">${esc(name)}</span>
    <span class="v ${t.present?'ok':'bad'}">${t.present?esc(t.version||"present"):"MISSING"}
    ${!t.present&&t.installable?` <button data-install="${esc(name)}" style="padding:3px 8px;margin-left:8px">Install</button>`:""}
    ${!t.present&&!t.installable&&t.hint?`<div class="note" style="margin-top:4px">${esc(t.hint)}</div>`:""}</span></div>`;
  const gpuLines = (hw.gpus||[]).map(g => `<div class="kv"><span class="k">GPU ${esc(g.index)}</span><span class="v">${esc(g.name)} &middot; cc ${esc(g.compute_cap||"?")}</span></div>`).join("");
  const bw = cfgOf().vram_bandwidths || {};
  setHTML(v, `
    <div class="card"><h3>Prerequisites</h3>
      ${Object.entries(p.tools).map(([n,t])=>toolRow(n,t)).join("")}
      <div class="kv"><span class="k">${esc(p.msvc.label||"C++ compiler")}</span><span class="v ${p.msvc.present?'ok':'bad'}">${p.msvc.present?"present":"MISSING"+(p.msvc.url?" &mdash; "+esc(p.msvc.url):"")}</span></div>
      ${p.cuda.applicable===false?"":`<div class="kv"><span class="k">CUDA toolkit</span><span class="v ${p.cuda.present?'ok':'bad'}">${p.cuda.present?esc(p.cuda.version||"present"):"not found (CPU build only)"}</span></div>`}
      <div class="kv"><span class="k">installers</span><span class="v">${esc(Object.keys(p.installers||{}).filter(k=>p.installers[k]).join(" ")||"none")}</span></div>
      <div class="note">Missing prerequisites can be installed with your permission where a package manager allows it (winget/choco/brew). On Linux the exact install command is shown instead &mdash; the dashboard never runs sudo.</div>
    </div>
    <div class="card"><h3>Detected Hardware</h3>
      <div class="kv"><span class="k">CPU</span><span class="v">${esc(hw.cpu.name||"?")} (${esc(hw.cpu.cores||"?")}c/${esc(hw.cpu.threads||"?")}t)</span></div>
      ${gpuLines}
      <div class="flags">${Object.entries(hw.cmake_flags).map(([k,val])=>`<span class="flagpill">${esc(k)}=${esc(val)}</span>`).join("")}</div>
      ${hw.notes.map(n=>`<div class="note">&bull; ${esc(n)}</div>`).join("")}
    </div>
    <div class="card"><h3>Speed Estimates <span style="color:var(--dim);font-weight:normal;font-size:11px">(advanced &mdash; optional)</span></h3>
      <div class="note">The "Will it run?" panel and Discover speed badges estimate tok/s from memory bandwidth. Detected GPU presets are used by default; override here only if you've measured your machine. Blank = use the preset/default.</div>
      <div class="row" style="gap:8px;margin-top:10px;flex-wrap:wrap;align-items:flex-end">
        <div class="fld"><label>VRAM GB/s</label><input id="bw-vram" type="number" min="0" step="any" placeholder="preset" value="${esc(String(bw.vram_bw ?? ""))}" style="width:110px"></div>
        <div class="fld"><label>RAM GB/s</label><input id="bw-ram" type="number" min="0" step="any" placeholder="50" value="${esc(String(bw.ram_bw ?? ""))}" style="width:110px"></div>
        <div class="fld"><label>Disk GB/s</label><input id="bw-disk" type="number" min="0" step="any" placeholder="5.7" value="${esc(String(bw.disk_bw ?? ""))}" style="width:110px"></div>
        <button id="bw-save">Save</button>
        <span class="msg" id="bw-msg"></span>
      </div>
    </div>
    <div class="card"><h3>Scan Drives for Models</h3>
      <div class="actions"><button id="btn-scan">Scan for GGUF models</button><button class="ghost" id="btn-missing">Check for deleted models</button><span class="msg" id="scan-msg"></span></div>
      <div id="scan-out"></div>
      <div id="missing-out"></div>
    </div>
    <div class="card"><h3>Startup</h3>
      <div class="kv"><label class="k" for="auto-load">auto-load a model on launch</label>
        <span class="v"><select id="auto-load" style="background:var(--inset);border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:6px">
          <option value="">none</option>
          ${models().map(m=>`<option value="${esc(m.id)}" ${cfgOf().auto_load_model===m.id?"selected":""}>${esc(m.id)}</option>`).join("")}
        </select></span></div>
      <div class="note">The selected model loads automatically once the router is ready after launch &mdash; handy for always-on setups. An optional tray icon (loaded-model count, quick open) is available if you <b>pip install pystray pillow</b>; without them LlamaForge stays pure-stdlib.</div>
    </div>
    ${networkMarkup(net)}
    <div id="agent-connect" class="card"></div>`
    + (vs.supported === false ? "" : `<div class="card"><h3>vLLM Backend (WSL2)</h3>
      <div class="kv"><span class="k">WSL2</span><span class="v ${vs.wsl.present?'ok':'bad'}">${vs.wsl.present?"installed":"NOT INSTALLED"}</span></div>
      ${vs.wsl.present?`<div class="kv"><span class="k">distro</span><span class="v">
        <select id="vllm-distro" style="background:var(--inset);border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:6px">
        ${(vs.distros||[]).map(d=>`<option value="${esc(d.name)}" ${d.name===vs.chosen?"selected":""}>${esc(d.name)} (${esc(d.state)})</option>`).join("")}
        </select></span></div>
      <div class="kv"><span class="k">GPU passthrough</span><span class="v ${vs.gpu.present?'ok':'bad'}">${vs.gpu.present?esc((vs.gpu.info||"").split("\n")[0]||"detected"):"NOT DETECTED (check NVIDIA driver)"}</span></div>
      <div class="kv"><span class="k">vLLM</span><span class="v ${vs.vllm.present?'ok':'bad'}">${vs.vllm.present?"v"+esc(vs.vllm.version):"not installed"}</span></div>`
      :`<div class="note">WSL2 is required to run vLLM. Install it (admin PowerShell): <b>wsl --install -d Ubuntu</b>, reboot, then reload this tab.</div>`}
      ${vs.wsl.present&&!vs.vllm.present?`<div class="actions"><button class="primary" id="btn-vllm-install">Install vLLM (uv, no sudo)</button><span class="msg" id="vllm-inst-msg"></span></div>
      <div class="note">Downloads uv + a standalone Python and installs vLLM into ~/.llamaforge/vllm-venv. Several GB; watch the log.</div>`:""}
      <div class="log" id="vllm-setup-log" style="display:${(vs.setup_job&&vs.setup_job.running)?"":"none"}">${esc(vs.setup_log||"idle")}</div>
    </div>`));
  wireNetwork(net);
  $$("[data-install]", v).forEach(b => b.onclick = async () => {
    b.disabled = true; b.textContent = "installing...";
    const r = await api("/api/setup/install", {tool: b.dataset.install});
    toast(r.ok?"Installed":"Install failed", r.ok?"ok":"err"); loadSetup();
  });
  $("#btn-scan").onclick = scanDrives;
  $("#btn-missing").onclick = checkMissing;
  const autoSel = $("#auto-load");
  if (autoSel) autoSel.onchange = async () => {
    await api("/api/config", {auto_load_model: autoSel.value});
    toast(autoSel.value?`Auto-load: ${autoSel.value}`:"Auto-load disabled", "ok");
  };
  const bwSave = $("#bw-save");
  if (bwSave) bwSave.onclick = async () => {
    const num = sel => { const val = $(sel).value.trim(); return val === "" ? undefined : Number(val); };
    const ov = {};
    const vram = num("#bw-vram"), ram = num("#bw-ram"), disk = num("#bw-disk");
    if (vram !== undefined && !Number.isNaN(vram)) ov.vram_bw = vram;
    if (ram !== undefined && !Number.isNaN(ram)) ov.ram_bw = ram;
    if (disk !== undefined && !Number.isNaN(disk)) ov.disk_bw = disk;
    await api("/api/config", {vram_bandwidths: ov});
    const m = $("#bw-msg"); m.className = "msg ok"; m.textContent = Object.keys(ov).length ? "saved" : "cleared (using defaults)";
  };
  const distroSel = $("#vllm-distro");
  if (distroSel) distroSel.onchange = () => api("/api/config", {wsl_distro: distroSel.value}).then(() => loadSetup());
  const instBtn = $("#btn-vllm-install");
  if (instBtn) instBtn.onclick = async () => {
    const msg = $("#vllm-inst-msg"); msg.className = "msg work"; msg.textContent = "starting install...";
    const r = await api("/api/vllm/setup/install", {distro: distroSel?distroSel.value:undefined});
    if (r.started) { toast("vLLM install started", "ok"); pollVllmSetup(); }
    else msg.textContent = "already running";
  };
  if (vs.setup_job && vs.setup_job.running) pollVllmSetup();
  renderAgentConnect();
}

/* ---------- connect an agent ---------- */
function agentModels() {
  const active = cfgOf().active_engine || "llamacpp";
  return models().filter(m =>
    (m.backend === "llamacpp" || m.backend === "ikllama") &&
    m.backend === active);
}

function agentModelOptions(selected = "") {
  return agentModels().map(m =>
    `<option value="${esc(m.id)}"${m.id === selected ? " selected" : ""}>${
      esc(m.id)}</option>`).join("");
}

function clearAgentPreview(message = "Choose settings, then show the configuration.") {
  const out = $("#ac-out");
  if (out) setHTML(out, `<div class="note">${esc(message)}</div>`);
}

function agentRequest() {
  const agent = $("#ac-agent").value;
  const model = $("#ac-model").value;
  const row = agentModels().find(m => m.id === model);
  if (!row) return null;
  return {
    agent,
    model,
    backend: row.backend,
    small: agent === "claude-code" ? $("#ac-small").value : "",
    inject: agent !== "claude-code" && $("#ac-inject").checked,
  };
}

function renderAgentConnect() {
  const host = $("#agent-connect");
  if (!host) return;
  setHTML(host, `<h3>Connect an agent</h3>
    <div class="note">Generate or apply configuration only when requested.
      Context injection uses this machine's loopback-only panel.</div>
    <div class="agent-controls">
      <label>Agent
        <select id="ac-agent">
          <option value="claude-code">Claude Code</option>
          <option value="codex">Codex</option>
          <option value="pi">pi.dev</option>
        </select>
      </label>
      <label>Model <select id="ac-model">${agentModelOptions()}</select></label>
      <label id="ac-small-wrap">Small model
        <select id="ac-small">${agentModelOptions()}</select>
      </label>
      <label id="ac-inject-wrap" hidden>
        <input id="ac-inject" type="checkbox">
        Inject local context through the panel (this machine only)
      </label>
      <button id="ac-show" type="button">Show configuration</button>
      <button id="ac-apply" type="button" class="primary">Apply</button>
    </div>
    <div id="ac-out" class="agent-out"></div>`);

  const sync = () => {
    const claude = $("#ac-agent").value === "claude-code";
    $("#ac-small-wrap").hidden = !claude;
    $("#ac-inject-wrap").hidden = claude;
    if (claude) $("#ac-inject").checked = false;
    clearAgentPreview();
  };
  $("#ac-agent").onchange = sync;
  $("#ac-model").onchange = () => clearAgentPreview();
  $("#ac-small").onchange = () => clearAgentPreview();
  $("#ac-inject").onchange = () => clearAgentPreview();
  $("#ac-show").onclick = showAgentConfig;
  $("#ac-apply").onclick = applyAgentConfig;
  sync();
}

async function showAgentConfig() {
  const body = agentRequest();
  if (!body) {
    clearAgentPreview("No llama-family model is available.");
    $("#ac-model").focus();
    return;
  }
  const out = $("#ac-out");
  const stillCurrent = () => {
    if (!out.isConnected || out !== $("#ac-out")) return false;
    const current = agentRequest();
    return current !== null && JSON.stringify(current) === JSON.stringify(body);
  };
  setHTML(out,
    `<div class="note" role="status">Generating configuration...</div>`);
  let r;
  try {
    r = await api("/api/agent/config", body);
  } catch (requestError) {
    if (stillCurrent()) {
      setHTML(out,
        `<div class="note" role="alert">Agent configuration is unavailable.</div>`);
    }
    return;
  }
  if (!stillCurrent()) return;
  if (r.error) {
    setHTML(out,
      `<div class="note" role="alert">${esc(r.error)}</div>`);
    return;
  }
  const privateValues = [r.content];
  setHTML(out,
    `<div class="note">Target: <b>${esc(r.target_path)}</b> · endpoint
      <b>${esc(r.endpoint)}</b><br>${esc(r.instructions)}</div>
      <div class="slabel">${esc(r.target_path)}</div>
      <div class="snip"><button type="button" class="qbtn scopy"
        data-agent-copy-index="0" aria-label="Copy agent configuration">Copy</button>${
        esc(r.content)}</div>`);
  $$("[data-agent-copy-index]", out).forEach(button => {
    const value = privateValues[Number(button.dataset.agentCopyIndex)];
    button.removeAttribute("data-agent-copy-index");
    button.onclick = () => navigator.clipboard.writeText(value).then(
      () => toast("Copied to clipboard", "ok"));
  });
}

async function applyAgentConfig() {
  const body = agentRequest();
  if (!body) {
    clearAgentPreview("No llama-family model is available.");
    $("#ac-model").focus();
    return;
  }
  let r;
  try {
    r = await api("/api/agent/apply", body);
  } catch (requestError) {
    setHTML($("#ac-out"),
      `<div class="note" role="alert">Agent configuration could not be applied.</div>`);
    return;
  }
  if (r.error) {
    setHTML($("#ac-out"),
      `<div class="note" role="alert">${esc(r.error)}</div>`);
    return;
  }
  clearAgentPreview(`${r.action}: ${r.path}${r.backup ? " (backup created)" : ""}`);
}

/* ---------- drive scanning ---------- */
async function scanDrives() {
  const msg = $("#scan-msg");
  msg.className = "msg work"; msg.textContent = "scanning all drives (may take a moment)...";
  const r = await api("/api/scan", {});
  const known = new Set(models().map(m => m.id));
  const fresh = r.entries.filter(e => !known.has(e.id));
  msg.className = "msg ok"; msg.textContent = `${r.entries.length} found, ${fresh.length} new`;
  setHTML($("#scan-out"), `<div class="note">${esc(fresh.length)} new models not yet in your config:</div>
    <div class="list" style="margin-top:10px">${fresh.map(e=>`<div class="row"><div class="rhead" style="cursor:default;grid-template-columns:1fr auto">
      <span class="mid">${esc(e.id)}${e.mmproj?'<span class="tag vis">vision</span>':''}${e.embeddings?'<span class="tag">embed</span>':''}</span>
      <span class="ctxpill">${esc(e.gib)} GiB</span></div></div>`).join("")||'<div class="note">nothing new</div>'}</div>
    ${fresh.length?`<div class="actions"><button class="primary" id="btn-apply">Add ${fresh.length} models to config</button><span class="msg" id="apply-msg"></span></div>`:""}`);
  if (fresh.length) $("#btn-apply").onclick = async () => {
    const am = $("#apply-msg"); am.className = "msg work"; am.textContent = "writing config...";
    const rr = await api("/api/scan/apply", {entries: fresh});
    am.className = "msg ok"; am.textContent = `added ${rr.added}`;
    toast("Models added", "ok"); emit("refresh", true);
  };
}

async function checkMissing() {
  const out = $("#missing-out");
  setHTML(out, `<div class="note">checking configured models against disk...</div>`);
  let r;
  try { r = await api("/api/scan/missing"); }
  catch (e) { setHTML(out, `<div class="note" style="color:var(--red)">backend unreachable</div>`); return; }
  const miss = (r && r.missing) || [];
  if (!miss.length) { setHTML(out, `<div class="note">All configured models still exist on disk.</div>`); return; }
  setHTML(out, `<div class="note">${esc(miss.length)} configured model(s) whose file is gone:</div>
    <div class="list" style="margin-top:10px">${miss.map(m=>`<div class="row"><div class="rhead" style="cursor:default;grid-template-columns:1fr auto">
      <span class="mid">${esc(m.id)}${m.loaded?'<span class="tag">loaded</span>':''}</span>
      <span class="ctxpill" title="${esc(m.model)}" style="color:var(--red);border-color:var(--red)">missing file</span></div></div>`).join("")}</div>
    <div class="actions"><button class="primary" id="btn-prune">Remove ${miss.length} missing</button><span class="msg" id="prune-msg"></span></div>`);
  $("#btn-prune").onclick = async () => {
    const pm = $("#prune-msg"); pm.className = "msg work"; pm.textContent = "removing...";
    const rr = await api("/api/scan/prune", {ids: miss.map(m => m.id)});
    toast(`Removed ${rr.removed.length} missing model(s)`, "ok");
    emit("refresh", true); checkMissing();
  };
}
