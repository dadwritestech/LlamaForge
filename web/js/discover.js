// Discover tab: search huggingface.co for GGUF (llama.cpp) or safetensors
// (vLLM) repos, rate each against real VRAM, and drive the download.
import { $, $$, esc, setHTML, api, toast, meter, fmtDur } from "./core.js";
import { S } from "./state.js";
import { emit } from "./bus.js";

let dlPoll = null, discoverLoaded = false;
let dlPrev = null;   // {t, bytes} from the previous progress poll -> speed/ETA

const PLAT_LABEL = {windows:"WIN", linux:"LINUX", macos:"MAC"};
const FIT_LABEL = {fits:["FITS VRAM","ok"], tight:["TIGHT","work"],
                   offload:["CPU OFFLOAD","err"], unknown:["?",""]};
const QUANT_BADGE = {nvfp4:["NVFP4","var(--green)"], fp8:["FP8","var(--cyan)"],
                     awq:["AWQ","var(--amber)"], gptq:["GPTQ","var(--amber)"],
                     bf16:["BF16","var(--dim)"], fp16:["FP16","var(--dim)"]};
const VFIT_LABEL = {fits:["FITS VRAM","var(--green)"], tight:["TIGHT","var(--amber)"],
                    wont:["WON'T FIT","var(--red)"], unknown:["?","var(--dim)"]};

function platTags(platforms) {
  if (!platforms || !platforms.length) return "";
  const cur = S.STATE && S.STATE.platform;
  let s = platforms.map(p => {
    const here = p === cur;
    return `<span class="tag" title="runs on ${esc(p)}${here?" (this machine)":""}" style="${here?"color:var(--amber);border-color:var(--amber)":""}">${PLAT_LABEL[p]||esc(p.toUpperCase())}</span>`;
  }).join("");
  if (cur && !platforms.includes(cur))
    s += `<span class="tag" style="color:var(--red);border-color:var(--red)" title="this backend does not run on ${esc(cur)}">NOT ON ${PLAT_LABEL[cur]||esc(cur.toUpperCase())}</span>`;
  return s;
}
function hubRow(m, installed, clickClass) {
  const inst = installed.has(m.repo);
  return `<div class="row" data-repo="${esc(m.repo)}">
    <div class="rhead ${clickClass}" style="grid-template-columns:1fr auto auto auto auto">
      <span class="mid">${esc(m.repo)}
        ${platTags(m.platforms)}
        ${m.gated?'<span class="tag" style="color:var(--red);border-color:var(--red)" title="gated repo - requires accepting terms + an HF token; downloads from here will fail">GATED</span>':''}
        ${inst?'<span class="tag" style="color:var(--green);border-color:var(--green)" title="already in your registry">INSTALLED</span>':''}
      </span>
      ${m.updated?`<span class="ctxpill"><span class="k">upd</span> ${esc(m.updated)}</span>`:""}
      <span class="ctxpill">${esc((m.downloads||0).toLocaleString())} dl</span>
      <span class="ctxpill" style="color:var(--cyan)">${esc(m.likes)} &hearts;</span>
      <span class="chev">&#9654;</span>
    </div>
    <div class="edit"></div>
  </div>`;
}
function dlSpeed(s) {
  const now = Date.now();
  let txt = "";
  if (dlPrev && s.downloaded >= dlPrev.bytes) {   // negative delta = next shard started
    const dt = (now - dlPrev.t) / 1000;
    if (dt > 0.2) {
      const bps = (s.downloaded - dlPrev.bytes) / dt;
      if (bps > 1e4) {
        txt = ` · ${(bps/1e6).toFixed(1)} MB/s`;
        if (s.total > s.downloaded) txt += ` · ETA ${fmtDur((s.total-s.downloaded)/bps)}`;
      }
    }
  }
  dlPrev = {t: now, bytes: s.downloaded};
  return txt;
}
function fitBadge(fit) {
  const [txt, cls] = FIT_LABEL[fit] || FIT_LABEL.unknown;
  const col = cls==="ok"?"var(--green)":cls==="work"?"var(--amber)":cls==="err"?"var(--red)":"var(--dim)";
  return `<span class="tag" style="color:${col};border-color:${col}">${txt}</span>`;
}
// vramwise placement + speed estimate (from /api/hub/files predict). Empty when
// unavailable so Discover degrades to the plain fit badge above.
const REGIME_LABEL = {
  "gpu-resident": ["FITS", "var(--green)"],
  "hybrid":       ["HYBRID", "var(--amber)"],
  "streaming":    ["STREAM", "var(--red)"],
};
function predictBadge(p) {
  if (!p || p.confidence === "unknown" || !p.regime) return "";
  const [txt, col] = REGIME_LABEL[p.regime] || ["?", ""];
  const tok = (p.tok_s != null) ? `~${esc(String(p.tok_s))} tok/s` : "";
  const faint = (p.confidence === "low") ? "opacity:.6" : "";
  return `<span class="tag" style="color:${col};border-color:${col};${faint}" title="${esc(p.note || "")}">${esc(txt)}${tok ? " &middot; " + tok : ""}</span>`;
}

export function loadDiscover() {
  if (discoverLoaded) return;
  discoverLoaded = true;
  setHTML($("#view-discover"), `
    <div class="card"><h3>Discover models on huggingface.co</h3>
      <div class="toolbar">
        <select id="hub-mode" style="background:var(--inset);border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:8px">
          <option value="gguf">GGUF (llama.cpp)</option>
          <option value="safetensors">safetensors (vLLM)</option>
        </select>
        <input class="search" id="hub-q" placeholder="search models (e.g. qwen coder, gemma vision)... blank = most downloaded">
        <select id="hub-sort" style="background:var(--inset);border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:8px">
          <option value="downloads">most downloaded</option>
          <option value="lastModified">newest</option>
          <option value="likes">most liked</option>
        </select>
        <button class="primary" id="hub-go">Search</button>
        <span class="msg" id="hub-msg"></span>
      </div>
      <div class="note">Fit ratings compare file size against your total VRAM (<span id="hub-vram">?</span> GB across all GPUs).
        FITS = full GPU offload with headroom &middot; TIGHT = loads but little room for context &middot; CPU OFFLOAD = larger than VRAM, will use system RAM (slower).</div>
    </div>
    <div id="hub-results"></div>
    <div class="card" id="hub-dlcard" style="display:none"><h3>Download</h3>
      <div class="kv"><span class="k">file</span><span class="v" id="dl-file">-</span></div>
      <div class="meter" style="margin-top:8px" id="dl-meter"></div>
      <div class="kv"><span class="k">progress</span><span class="v" id="dl-prog">-</span></div>
      <div class="actions" id="dl-run" style="display:none">
        <button class="ghost" id="dl-pause">Pause</button>
        <button class="ghost" id="dl-resume" style="display:none">Resume</button>
        <button class="ghost" id="dl-cancel">Cancel download</button>
      </div>
      <div class="actions" id="dl-done" style="display:none">
        <button class="primary" id="dl-add">Add to my models</button><span class="msg" id="dl-msg"></span>
      </div>
    </div>`);
  $("#dl-cancel").onclick = async () => { const r = await api("/api/hub/cancel", {}); toast(r.ok?"Cancelling...":"No download running", r.ok?"ok":"err"); };
  $("#dl-pause").onclick = async () => { const r = await api("/api/hub/pause", {}); toast(r.ok?"Pausing...":"No download running", r.ok?"ok":"err"); };
  $("#dl-resume").onclick = async () => {
    const r = await api("/api/hub/resume", {});
    if (r.ok) { toast("Resuming download", "ok"); ggufDlPoll(); } else toast("Nothing to resume", "err");
  };
  // vLLM (safetensors) is Windows/WSL-only; drop the mode on other platforms
  if (S.STATE && S.STATE.vllm_supported === false) {
    const opt = $('#hub-mode option[value="safetensors"]');
    if (opt) opt.remove();
  }
  // restore last search (mode/sort/query survive tab switches + reloads)
  try {
    const saved = JSON.parse(localStorage.getItem("lf_hub") || "{}");
    if (saved.mode && $(`#hub-mode option[value="${saved.mode}"]`)) $("#hub-mode").value = saved.mode;
    if (saved.sort) $("#hub-sort").value = saved.sort;
    if (saved.q) $("#hub-q").value = saved.q;
  } catch (e) {}
  $("#hub-go").onclick = hubSearch;
  $("#hub-mode").onchange = () => hubSearch();
  $("#hub-q").addEventListener("keydown", e => { if (e.key === "Enter") hubSearch(); });
  hubSearch();
}

async function hubSearch() {
  localStorage.setItem("lf_hub", JSON.stringify({
    mode: $("#hub-mode").value, sort: $("#hub-sort").value, q: $("#hub-q").value.trim()}));
  if ($("#hub-mode") && $("#hub-mode").value === "safetensors") return vllmHubSearch();
  const msg = $("#hub-msg"); msg.className = "msg work"; msg.textContent = "searching huggingface.co...";
  const r = await api("/api/hub/search", {query: $("#hub-q").value.trim(), sort: $("#hub-sort").value});
  if (r.error) { msg.className = "msg err"; msg.textContent = r.error.slice(0,80); return; }
  $("#hub-vram").textContent = (r.vram_mib/1024).toFixed(1);
  msg.className = "msg ok"; msg.textContent = `${r.results.length} repos`;
  const inst = new Set(r.installed || []);
  setHTML($("#hub-results"), `<div class="list">${r.results.map(m => hubRow(m, inst, "hub-repo")).join("")}</div>`);
  $$("#hub-results .hub-repo").forEach(h => h.onclick = () => hubFiles(h.parentElement));
}

async function hubFiles(row) {
  const open = row.classList.toggle("open");
  if (!open) return;
  const box = $(".edit", row);
  setHTML(box, `<div class="note">listing files...</div>`);
  const r = await api("/api/hub/files", {repo: row.dataset.repo});
  if (r.error) { setHTML(box, `<div class="note" style="color:var(--red)">${esc(r.error.slice(0,120))}</div>`); return; }
  const mm = r.mmproj && r.mmproj.length ? r.mmproj[0].path : "";
  setHTML(box, `
    ${mm?`<div class="note">vision model - the smallest mmproj (${esc(mm)}) will be downloaded too</div>`:""}
    ${r.mtp&&r.mtp.length?`<div class="note">matching MTP sidecars are downloaded automatically</div>`:""}
    <div class="list" style="margin-top:8px">${r.files.map(f=>`
      <div class="row"><div class="rhead" style="grid-template-columns:1fr auto auto auto auto;cursor:default">
        <span class="mid">${esc(f.path)}${f.shards>1?`<span class="tag">${f.shards} shards</span>`:""}${f.mtp?`<span class="tag" title="includes ${esc(f.mtp)}">MTP</span>`:""}</span>
        <span class="ctxpill">${esc((f.size/1e9).toFixed(2))} GB</span>
        ${fitBadge(f.fit)}
        ${predictBadge(f.predict)}
        <button data-dl="${esc(f.path)}" data-shards="${f.shards}" data-mtp="${esc(f.mtp||"")}" ${f.fit==="offload"?'title="larger than VRAM - will be slow"':""}>Download</button>
      </div></div>`).join("")}</div>`);
  $$("[data-dl]", box).forEach(b => b.onclick = () =>
    hubDownload(row.dataset.repo, b.dataset.dl, parseInt(b.dataset.shards), mm, b.dataset.mtp));
}

async function hubDownload(repo, path, shards, mmproj, mtp) {
  const r = await api("/api/hub/download", {repo, path, shards, mmproj, mtp});
  if (!r.started) { toast("A download is already running", "err"); return; }
  toast("Download started", "ok");
  $("#hub-dlcard").style.display = ""; $("#dl-done").style.display = "none";
  $("#dl-run").style.display = ""; dlPrev = null;
  ggufDlPoll();
}

// GGUF download progress loop, shared by a fresh download and by Resume.
function ggufDlPoll() {
  $("#hub-dlcard").style.display = ""; $("#dl-run").style.display = ""; dlPrev = null;
  clearInterval(dlPoll);
  dlPoll = setInterval(async () => {
    const s = await api("/api/hub/progress");
    $("#dl-file").textContent = `${s.repo} :: ${s.file||"-"} (${s.done_files+1 > s.total_files ? s.total_files : s.done_files+1}/${s.total_files})`;
    const pct = s.total ? Math.round(100*s.downloaded/s.total) : 0;
    setHTML($("#dl-meter"), meter(s.downloaded, Math.max(s.total,1)));
    $("#dl-prog").textContent = s.phase==="done" ? "complete"
      : s.phase==="failed" ? ("FAILED: " + s.error.slice(0,80))
      : s.phase==="cancelled" ? "cancelled"
      : s.phase==="paused" ? `paused at ${(s.downloaded/1e9).toFixed(2)} / ${(s.total/1e9).toFixed(2)} GB (${pct}%)`
      : `${(s.downloaded/1e9).toFixed(2)} / ${(s.total/1e9).toFixed(2)} GB (${pct}%)${dlSpeed(s)}`;
    const active = s.phase === "downloading" || s.phase === "starting";
    $("#dl-run").style.display = (active || s.phase === "paused") ? "" : "none";
    $("#dl-pause").style.display = active ? "" : "none";
    $("#dl-resume").style.display = s.phase === "paused" ? "" : "none";
    if (s.phase === "paused") clearInterval(dlPoll);
    if (s.phase === "cancelled") { clearInterval(dlPoll); toast("Download cancelled", "ok"); }
    if (s.phase === "done") {
      clearInterval(dlPoll); $("#dl-done").style.display = "";
      $("#dl-add").onclick = async () => {
        const m = $("#dl-msg"); m.className = "msg work"; m.textContent = "registering...";
        const rr = await api("/api/hub/add", {path: s.finished_path});
        if (rr.ok) { m.className = "msg ok"; m.textContent = "added: " + rr.added.join(", "); toast("Model added to registry","ok"); emit("refresh", true); }
        else { m.className = "msg err"; m.textContent = rr.error || "failed"; }
      };
    }
    if (s.phase === "failed") clearInterval(dlPoll);
  }, 1000);
}

/* ---------- vLLM (safetensors) ---------- */
async function vllmHubSearch() {
  const msg = $("#hub-msg"); msg.className = "msg work"; msg.textContent = "searching safetensors repos...";
  const r = await api("/api/vllm/hub/search", {query: $("#hub-q").value.trim(), sort: $("#hub-sort").value});
  if (r.error) { msg.className = "msg err"; msg.textContent = r.error.slice(0,80); return; }
  $("#hub-vram").textContent = (r.vram_mib/1024).toFixed(1);
  msg.className = "msg ok"; msg.textContent = `${r.results.length} repos`;
  const inst = new Set(r.installed || []);
  setHTML($("#hub-results"), `<div class="list">${r.results.map(m => hubRow(m, inst, "vhub-repo")).join("")}</div>`);
  $$("#hub-results .vhub-repo").forEach(h => h.onclick = () => vllmHubInfo(h.parentElement));
}

async function vllmHubInfo(row) {
  const open = row.classList.toggle("open");
  if (!open) return;
  const box = $(".edit", row);
  setHTML(box, `<div class="note">reading repo (summing shards, detecting quant)...</div>`);
  const r = await api("/api/vllm/hub/info", {repo: row.dataset.repo});
  if (r.error) { setHTML(box, `<div class="note" style="color:var(--red)">${esc(r.error.slice(0,120))}</div>`); return; }
  const [qtxt, qcol] = QUANT_BADGE[r.quant] || [r.quant.toUpperCase(), "var(--dim)"];
  const [ftxt, fcol] = VFIT_LABEL[r.fit] || VFIT_LABEL.unknown;
  const nvfp4Note = r.quant === "nvfp4" ? `<div class="note" style="color:var(--green)">NVFP4 &mdash; native on your Blackwell GPUs</div>` : "";
  setHTML(box, `
    <div class="kv"><span class="k">weights size</span><span class="v">${esc((r.size_bytes/1e9).toFixed(1))} GB</span></div>
    <div class="kv"><span class="k">quantization</span><span class="v"><span class="tag" style="color:${qcol};border-color:${qcol}">${qtxt}</span></span></div>
    <div class="kv"><span class="k">VRAM fit</span><span class="v"><span class="tag" style="color:${fcol};border-color:${fcol}">${ftxt}</span></span></div>
    ${nvfp4Note}
    <div class="actions">
      <button class="primary" data-vdl="${esc(row.dataset.repo)}" data-size="${r.size_bytes}" data-quant="${esc(r.quant)}" ${r.fit==="wont"?'title="larger than usable VRAM"':""}>Download to WSL</button>
      <span class="msg" data-vmsg></span>
    </div>`);
  $(`[data-vdl]`, box).onclick = e =>
    vllmHubDownload(e.target.dataset.vdl, parseInt(e.target.dataset.size), e.target.dataset.quant);
}

async function vllmHubDownload(repo, sizeBytes, quant) {
  const r = await api("/api/vllm/hub/download", {repo, size_bytes: sizeBytes});
  if (!r.started) { toast("A download is already running", "err"); return; }
  toast("Download started", "ok");
  $("#hub-dlcard").style.display = ""; $("#dl-done").style.display = "none";
  $("#dl-run").style.display = "none"; dlPrev = null;   // WSL transfer: no cancel
  clearInterval(dlPoll);
  dlPoll = setInterval(async () => {
    const s = await api("/api/vllm/hub/progress");
    $("#dl-file").textContent = `${s.repo} (WSL cache)`;
    const pct = s.total ? Math.round(100*s.downloaded/s.total) : 0;
    setHTML($("#dl-meter"), meter(s.downloaded, Math.max(s.total,1)));
    $("#dl-prog").textContent = s.phase==="done" ? "complete"
      : s.phase==="failed" ? ("FAILED: " + (s.error||"").slice(0,80))
      : `${(s.downloaded/1e9).toFixed(2)} / ${(s.total/1e9).toFixed(2)} GB (${pct}%)${dlSpeed(s)}`;
    if (s.phase === "done") {
      clearInterval(dlPoll); $("#dl-done").style.display = "";
      $("#dl-add").onclick = async () => {
        const m = $("#dl-msg"); m.className = "msg work"; m.textContent = "registering...";
        const rr = await api("/api/vllm/hub/register", {repo, size_bytes: sizeBytes, quant});
        if (rr.ok) { m.className = "msg ok"; m.textContent = "added: " + rr.added; toast("vLLM model registered","ok"); emit("refresh", true); }
        else { m.className = "msg err"; m.textContent = rr.error || "failed"; }
      };
    }
    if (s.phase === "failed") clearInterval(dlPoll);
  }, 1000);
}
