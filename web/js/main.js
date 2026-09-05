// Entry point. Owns two things and no view logic:
//   1. which loader runs when a tab is shown
//   2. the polling timers
//
// There are no `window`-assigned globals: every handler is wired with
// addEventListener (toolbar controls by id, dynamic rows by delegation), so the
// HTML and view templates carry no inline on* attributes to keep in sync.
import { $, api, esc, setHTML } from "./core.js";
import { S } from "./state.js";
import * as ui from "./ui.js";
import * as models from "./models.js";
import * as stats from "./stats.js";
import { loadDiscover } from "./discover.js";
import { loadWillRun } from "./willrun.js";
import { loadBuild } from "./build.js";
import { leaveSetup, loadSetup } from "./setup.js";
import { loadContext } from "./context.js";
import { loadDocs } from "./help.js";
import { initWizard } from "./wizard.js";
import { initOnboarding } from "./onboarding.js";

/* ---------- tab loaders ---------- */
ui.onTabShown("build", loadBuild);
ui.onTabShown("setup", loadSetup);
ui.onTabHidden("setup", leaveSetup);
ui.onTabShown("discover", loadDiscover);
ui.onTabShown("willrun", loadWillRun);
ui.onTabShown("stats", stats.loadStats);
ui.onTabShown("context", loadContext);
ui.onTabShown("help", loadDocs);

/* ---------- boot ---------- */
ui.initTabs();
ui.initModeToggle();
ui.initThemeControls();
ui.initSidebar();
ui.initDrawer();
ui.updatePageTitle();
initWizard();
initOnboarding();
models.initModels();
stats.initStats();

// deep-linkable tabs: #<tab> in the URL activates that tab (docs deep-links +
// tools/shoot.py). Runs after initTabs() has wired the click handlers.
window.addEventListener("hashchange", () => {
  const h = location.hash.slice(1);
  if (h) ui.switchTab(h);
});
if (location.hash) ui.switchTab(location.hash.slice(1));

// The engine badge sits beside the clock but changes about once a session,
// so it gets its own element: the clock stays a textContent write, and the
// badge is only re-rendered (through the audited setHTML/esc sink) when the
// engine actually changes. Rebuilding markup once a second would both churn
// the DOM and put a dynamic value into innerHTML on every tick.
const ENGINE_LABEL = { llamacpp: "llama.cpp", ikllama: "ik_llama" };
let shownEngine = null;

function renderEngineBadge() {
  const engine = (S.STATE && S.STATE.active_engine) || "";
  if (engine === shownEngine) return;
  shownEngine = engine;
  const el = $("#engine-badge");
  if (!el) return;
  setHTML(el, engine
    ? `<span class="tag be-${esc(engine)}">${esc(ENGINE_LABEL[engine] || engine)}</span>`
    : "");
}

function clock() {
  const el = $("#clock");
  if (el) el.textContent = new Date().toLocaleTimeString("en-GB") + " LOCAL";
  renderEngineBadge();
}
clock();
setInterval(clock, 1000);

(async () => {
  S.SCHEMA = await api("/api/schema");
  await models.refresh();
  // theme/cvd defaults from config.json, used only when this device hasn't chosen
  try {
    const cfg = (S.STATE && S.STATE.config) || {};
    if (!localStorage.getItem("theme") && cfg.theme) ui.applyTheme(cfg.theme);
    if (localStorage.getItem("cvd") === null && cfg.cvd) ui.applyCvd(true);
    ui.applyMode(((S.STATE||{}).onboarding||{}).ui_mode || "lite");
  } catch (e) {}
})();

/* ---------- polls (idle unless their tab is showing) ---------- */
setInterval(() => { if (ui.activeTab() === "models") models.refresh(true); }, 4000);
setInterval(() => { if (ui.activeTab() === "stats") stats.loadStats(true); }, 4000);
setInterval(() => { if (ui.activeTab() === "models") models.refreshRouterLog(); }, 3000);
setInterval(() => { if (ui.activeTab() === "models") models.refreshVllmLog(); }, 3000);
models.refreshRouterLog();
models.refreshVllmLog();
