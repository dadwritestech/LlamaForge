// Asking for a GitHub star, once, at a moment LlamaForge just earned it.
//
// The one-line installer clones the repo, so most people who use LlamaForge
// never see its GitHub page. This asks them in the app instead, but only right
// after it did something for them (an Auto-Forge run that beat their setup, or
// the 5th model they loaded), and never again once it has asked. It makes no
// network request; the button is a plain link.

import { esc, setHTML } from "./core.js";

export const REPO = "https://github.com/dadwritestech/LlamaForge";
const KEY = "lf_star_ask", LOADS = "lf_loads", LOADS_TO_ASK = 5;

const store = {
  get: k => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode: ask at most per session */ } },
};
let askedThisSession = false;

/** What to say for an event, or null when it shouldn't ask. Pure; tested.
 *  asked: anything stored under KEY (it asks once, ever). */
export function askFor(event, asked, loads = 0) {
  if (asked) return null;
  if (event.kind === "forge" && event.x >= 1.1) {
    return `Auto-Forge just made ${event.model} ${event.x.toFixed(1)}x faster.`;
  }
  if (event.kind === "load" && loads === LOADS_TO_ASK) return `That's ${LOADS_TO_ASK} models you've run with LlamaForge.`;
  return null;
}

function ask(event, loads) {
  const line = askFor(event, askedThisSession || store.get(KEY), loads);
  if (!line) return;
  askedThisSession = true;
  store.set(KEY, "asked");
  let el = document.getElementById("star-ask");
  if (!el) {
    el = Object.assign(document.createElement("div"), {id: "star-ask", className: "star-ask"});
    el.setAttribute("role", "status");
    document.body.appendChild(el);
  }
  setHTML(el, `<div class="sa-line">${esc(line)}</div>
    <div class="sa-sub">If it's useful, a star on GitHub is how other people find it. It's a one-person project.</div>
    <div class="sa-act"><a class="qbtn load" href="${REPO}" target="_blank" rel="noopener" data-sa="star">&#9733; Star on GitHub</a>
      <button type="button" class="qbtn" data-sa="no">Not now</button></div>`);
  el.hidden = false;
  el.onclick = e => {
    const t = e.target.closest("[data-sa]");
    if (!t) return;
    store.set(KEY, t.dataset.sa === "star" ? "starred" : "no");
    el.hidden = true;
  };
}

/** A model finished loading. */
export function noteLoad() {
  const n = (parseInt(store.get(LOADS), 10) || 0) + 1;
  store.set(LOADS, String(n));
  ask({kind: "load"}, n);
}

/** An Auto-Forge run the user watched finished; x = best speedup over their setup. */
export function noteForgeWin(model, x) {
  if (Number.isFinite(x)) ask({kind: "forge", model, x}, 0);
}
