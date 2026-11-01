// LlamaForge live demo: the real panel (web/) with fetch() answered from
// fixtures.json, which tools/demo_record.py recorded from a real two-GPU box and
// scrubbed. Nothing runs: reads come from the recording, every change is
// answered with DEMO_MSG. Loaded as a classic script before the panel's module,
// so the first /api call already goes through here.
(function () {
  "use strict";
  var DEMO_MSG = "This is the live demo, so nothing runs here. Install LlamaForge to do this on your own GPUs.";
  var INSTALL = "https://github.com/dadwritestech/LlamaForge#install";
  var realFetch = window.fetch.bind(window);
  var fixtures = realFetch("fixtures.json", {cache: "no-cache"})
    .then(function (r) { return r.json(); })
    .catch(function () { return {}; });

  function json(v, status) {
    return new Response(JSON.stringify(v), {status: status || 200, headers: {"Content-Type": "application/json"}});
  }

  function urlOf(input) {
    return typeof input === "string" ? input : (input && input.url) || String(input);
  }

  var NOTE = "Recorded from a real RTX 5080 + 5060 Ti box. Click around: nothing runs here.";
  var nudged = 0, back = 0;
  // a refused change: shake the bar and say why there for a few seconds
  function nudge() {
    var bar = document.getElementById("lf-demo-bar");
    if (!bar || Date.now() - nudged < 1500) return;
    nudged = Date.now();
    bar.classList.remove("nudge"); void bar.offsetWidth; bar.classList.add("nudge");
    var span = bar.querySelector("span");
    if (!span) return;
    span.textContent = DEMO_MSG;
    clearTimeout(back);
    back = setTimeout(function () { span.textContent = NOTE; }, 5000);
  }

  window.fetch = function (input, init) {
    var url = urlOf(input);
    if (url.indexOf("/api/") !== 0) {
      // the router's own web UI (Chat tab) and anything else off-panel: not here
      if (/^https?:/.test(url)) return Promise.reject(new TypeError("demo: no router here"));
      return realFetch(input, init);
    }
    var method = ((init && init.method) || "GET").toUpperCase();
    var q = url.indexOf("?");
    var path = q < 0 ? url : url.slice(0, q);
    return fixtures.then(function (fx) {
      var hit = fx[method + " " + url];
      if (hit === undefined) hit = fx[method + " " + path];
      if (hit !== undefined) return json(hit);
      if (method === "GET") return json({});
      nudge();
      // the panel's handlers read either {ok, error} or {success, reason}
      return json({ok: false, success: false, error: DEMO_MSG, reason: DEMO_MSG, demo: true});
    });
  };

  function bar() {
    if (document.getElementById("lf-demo-bar")) return;
    var css = document.createElement("style");
    css.textContent =
      "#lf-demo-bar{position:fixed;left:50%;bottom:14px;transform:translateX(-50%);z-index:9999;" +
      "display:flex;gap:12px;align-items:center;padding:8px 10px 8px 14px;background:#13243f;color:#e8eef6;" +
      "border:1px solid #ffc21a;font:500 13px/1.3 'Barlow',system-ui,sans-serif;box-shadow:0 6px 24px rgba(0,0,0,.35);" +
      "max-width:calc(100vw - 24px)}" +
      "#lf-demo-bar b{color:#ffc21a;letter-spacing:.08em;font-family:'Barlow Condensed',system-ui,sans-serif}" +
      "#lf-demo-bar a{color:#13243f;background:#ffc21a;padding:5px 10px;text-decoration:none;font-weight:600;white-space:nowrap}" +
      "#lf-demo-bar.nudge{animation:lfnudge .5s}" +
      "@keyframes lfnudge{0%,100%{transform:translateX(-50%)}25%{transform:translateX(calc(-50% - 6px))}75%{transform:translateX(calc(-50% + 6px))}}" +
      "@media (max-width:640px){#lf-demo-bar span{display:none}}";
    document.head.appendChild(css);
    var el = document.createElement("div");
    el.id = "lf-demo-bar";
    el.setAttribute("role", "note");
    var tag = document.createElement("b"); tag.textContent = "LIVE DEMO";
    var txt = document.createElement("span");
    txt.textContent = NOTE;
    var a = document.createElement("a"); a.href = INSTALL; a.target = "_blank"; a.rel = "noopener";
    a.textContent = "Install it";
    el.appendChild(tag); el.appendChild(txt); el.appendChild(a);
    document.body.appendChild(el);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bar);
  else bar();
})();
