// "New this week" (top of Discover): model support llama.cpp just merged,
// marked against the engine you run, plus LlamaForge's own update.
import { $, esc, setHTML, api, toast } from "./core.js";
import { switchTab } from "./ui.js";

let updPoll = null;
// links come from GitHub's API; still, only ever https into an href
const href = u => esc(/^https:\/\//.test(u || "") ? u : "#");

function newsRow(n) {
  const badge = n.have === true
    ? `<span class="tag" style="color:var(--green);border-color:var(--green)" title="your engine is build ${esc(String(n.build))} or newer">IN YOUR ENGINE</span>`
    : n.have === false
      ? `<button class="ghost feed-engine" title="your engine predates ${esc(n.tag)}">Update engine</button>`
      : "";
  return `<div class="feed-item">
    <span class="ctxpill">${esc(n.date)}</span>
    <a class="feed-title" href="${href(n.pr || n.url)}" target="_blank" rel="noopener">${esc(n.title)}</a>
    <span class="tag">${esc(n.tag)}</span>${badge}
  </div>`;
}

function appBanner(a) {
  const j = (a && a.job) || {};
  if (!a || !a.managed) return "";
  if (j.state === "done")   // files are already on disk, so "available" is false now
    return `<div class="feed-app"><span><b>LlamaForge ${esc(j.tag)}</b> is installed.
      Restart to finish (loaded models keep running).</span>
      <button class="primary" id="app-restart">Restart now</button></div>`;
  if (!a.available) return "";
  const action = (j.state === "downloading" || j.state === "installing")
    ? `<span class="msg work">${esc(j.state)}...</span>`
    : `<button class="primary" id="app-update">Update now</button>`;
  return `<div class="feed-app">
    <span><b>LlamaForge ${esc(a.latest)}</b> is out (you have ${esc(a.installed)}).
      <a href="${href(a.url)}" target="_blank" rel="noopener">What's new</a></span>
    ${action}${j.state === "error" ? `<span class="msg err">${esc((j.error || "").slice(0, 100))}</span>` : ""}
  </div>`;
}

function render(r) {
  const news = r.engine_news || [];
  const engine = r.engine_build ? `your engine: build ${esc(String(r.engine_build))}` : "";
  setHTML($("#feed"), `<div class="card"><h3>New this week</h3>
    ${appBanner(r.app)}
    <div class="note" style="margin-top:0">llama.cpp just learned to run ${engine ? `&middot; ${engine}` : ""}</div>
    ${news.length ? `<div class="feed-list">${news.map(newsRow).join("")}</div>`
      : `<div class="note">${r.engine_error ? "couldn't reach GitHub: " + esc(r.engine_error.slice(0, 80)) : "no new architectures in the last few weeks"}</div>`}
    <div class="note">Below: GGUFs trending on Hugging Face that were published in the last 14 days, rated for your VRAM.</div>
  </div>`);
  $("#feed").querySelectorAll(".feed-engine").forEach(b => b.onclick = () => switchTab("build"));
  const up = $("#app-update");
  if (up) up.onclick = async () => {
    const s = await api("/api/app/update", {tag: r.app.latest});
    if (!s.ok) return toast(s.error || "update failed", "err");
    watchUpdate();
  };
  const rs = $("#app-restart");
  if (rs) rs.onclick = restart;
}

function watchUpdate() {
  clearInterval(updPoll);
  updPoll = setInterval(async () => {
    const j = await api("/api/app/update");
    if (j.state === "downloading" || j.state === "installing") return;
    clearInterval(updPoll);
    loadFeed();
    if (j.state === "done") toast("Update installed - restart to finish", "ok");
  }, 1000);
}

async function restart() {
  const r = await api("/api/app/restart", {});
  if (!r.ok) return toast(r.error || "restart failed", "err");
  toast("Restarting LlamaForge... models keep running", "ok");
  const t0 = Date.now();
  await new Promise(res => setTimeout(res, 3000));
  const tick = async () => {
    try {
      const resp = await fetch("/api/state", {cache: "no-store"});
      if (resp.ok) return location.reload();
    } catch (e) {}
    if (Date.now() - t0 < 60000) setTimeout(tick, 1000);
    else toast("LlamaForge didn't come back - start it from the Start menu / llamaforge", "err");
  };
  tick();
}

export async function loadFeed(force) {
  if (!$("#feed")) return;
  const r = await api("/api/feed" + (force ? "?force=1" : ""));
  if (r && !r.error) render(r);
}
