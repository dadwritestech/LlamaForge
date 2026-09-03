const SENTINEL = "fixture-secret-must-not-leak-" + "f".repeat(16);

function check(condition, code) {
  if (!condition) throw new Error(code);
}

async function waitFor(selector, predicate = () => true, code = "missing-selector") {
  const deadline = Date.now() + 8000;
  while (Date.now() < deadline) {
    const element = document.querySelector(selector);
    if (element && await predicate(element)) return element;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(code);
}

async function waitUntil(predicate, code, timeout = 8000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(code);
}

function click(selector, root = document, code = "missing-click-target") {
  const element = root.querySelector(selector);
  check(element, code);
  element.click();
  return element;
}

function hasMinimumPointerTarget(element) {
  const rect = element.getBoundingClientRect();
  return rect.width >= 48 && rect.height >= 48;
}

async function requests() {
  const response = await fetch("/_fixture/requests", {cache: "no-store"});
  const payload = await response.json();
  return payload.requests;
}

async function setScenario(name) {
  const response = await fetch("/_fixture/scenario", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name}),
  });
  const payload = await response.json();
  check(response.ok && payload.ok, "scenario-selection-failed");
}

function apiRequests(list, path, method = null) {
  return list.filter(entry => entry.path === path && (!method || entry.method === method));
}

function setValue(element, value) {
  element.value = value;
  element.dispatchEvent(new Event("change", {bubbles: true}));
}

function bodyHasSecret() {
  return document.body.textContent.includes(SENTINEL);
}

function secretTextIsInside(container) {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    if (node.nodeValue.includes(SENTINEL) && !container.contains(node.parentElement)) {
      return false;
    }
  }
  return true;
}

function accessibleName(element) {
  const labelled = element.getAttribute("aria-labelledby");
  if (labelled) {
    return labelled.split(/\s+/).map(id => document.getElementById(id)?.textContent || "")
      .join(" ").trim();
  }
  return (element.getAttribute("aria-label") || element.textContent || "").trim();
}

function dispatchTab(element, shiftKey) {
  element.dispatchEvent(new KeyboardEvent("keydown", {
    key: "Tab", shiftKey, bubbles: true, cancelable: true,
  }));
}

function parseRgb(value) {
  const match = value.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  return match ? match.slice(1, 4).map(Number) : null;
}

function luminance(rgb) {
  const values = rgb.map(value => {
    const normalized = value / 255;
    return normalized <= 0.03928
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  });
  return values[0] * 0.2126 + values[1] * 0.7152 + values[2] * 0.0722;
}

function contrastRatio(element) {
  const foreground = parseRgb(getComputedStyle(element).color);
  let current = element;
  let background = null;
  while (current && !background) {
    const color = getComputedStyle(current).backgroundColor;
    if (color && color !== "transparent" && !color.endsWith(", 0)")) {
      background = parseRgb(color);
    }
    current = current.parentElement;
  }
  background ||= parseRgb(getComputedStyle(document.body).backgroundColor);
  if (!foreground || !background) return 0;
  const a = luminance(foreground);
  const b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

function hasLabel(control) {
  return Boolean(control.closest("label") || control.getAttribute("aria-label") ||
    control.getAttribute("aria-labelledby") ||
    (control.id && document.querySelector(`label[for="${CSS.escape(control.id)}"]`)));
}

async function renderScenario(name) {
  await setScenario(name);
  const previous = document.querySelector("#network-access");
  click('.tab[data-tab="setup"]', document, "missing-setup-tab");
  return waitFor("#network-access", element => element !== previous,
    "missing-network-access");
}

function scrubSecretDom() {
  document.querySelector("#modal-root")?.replaceChildren();
  document.querySelector("#ac-out")?.replaceChildren();
  for (const element of document.querySelectorAll("#net-generated-display")) {
    element.textContent = "";
  }
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    if (node.nodeValue.includes(SENTINEL)) {
      node.nodeValue = node.nodeValue.replaceAll(SENTINEL, "[fixture secret redacted]");
    }
  }
  for (const element of document.querySelectorAll("*")) {
    for (const attribute of element.attributes) {
      if (attribute.value.includes(SENTINEL)) element.removeAttribute(attribute.name);
    }
  }
  for (const key of Object.keys(localStorage)) {
    if (String(localStorage.getItem(key)).includes(SENTINEL)) localStorage.removeItem(key);
  }
}

async function run() {
  let clipboardValue = "";
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: {
      writeText(value) {
        clipboardValue = String(value);
        return Promise.resolve();
      },
    },
  });

  const captureSecretCopy = async (button, code, exact = false) => {
    clipboardValue = "";
    button.click();
    await waitUntil(() => clipboardValue !== "", code);
    check(exact ? clipboardValue === SENTINEL : clipboardValue.includes(SENTINEL), code);
    clipboardValue = "";
  };

  await waitFor("#agent-connect #ac-apply",
    element => typeof element.onclick === "function", "agent-controls-not-wired");
  await waitFor("#ac-out .note", () => true, "agent-initial-state-missing");
  await waitUntil(async () => {
    const list = await requests();
    return apiRequests(list, "/api/network", "GET").length > 0;
  }, "setup-network-request-missing");
  const routine = await requests();
  check(apiRequests(routine, "/api/agent/config").length === 0,
    "automatic-agent-request");
  check(apiRequests(routine, "/api/client/config").length === 0,
    "automatic-client-request");
  check(document.querySelector("#network-access"), "missing-network-access");

  let root = await renderScenario("local-no-key");
  check(root.querySelector('[name="net-scope"][value="local"]').checked,
    "local-scope-state");
  check(root.querySelector("#net-configured").textContent.trim() === "local",
    "local-configured-state");
  check(root.querySelector("#net-runtime").textContent.includes("listening"),
    "local-listener-state");

  root = await renderScenario("local-keyed");
  check(root.querySelector("#net-configured").textContent.trim() === "local keyed",
    "local-keyed-configured-state");
  check(!root.querySelector('[name="net-key-action"][value="keep"]').disabled,
    "local-keyed-keep-state");

  root = await renderScenario("lan-protected");
  check(root.querySelector('[name="net-scope"][value="lan"]').checked,
    "lan-scope-state");
  check(root.querySelector("#net-configured").textContent.trim() === "protected",
    "lan-configured-state");
  check(!root.querySelector("#net-runtime").textContent.includes("protected"),
    "listener-policy-conflation");

  root = await renderScenario("unsafe-legacy");
  const alert = root.querySelector('#net-alert[role="alert"]');
  check(alert && alert.textContent.includes("cannot be verified"),
    "unsafe-legacy-alert");
  check(root.querySelector("#net-configured").textContent.trim() === "unsafe legacy",
    "unsafe-legacy-configured-state");

  root = await renderScenario("local-no-key");
  click('[name="net-scope"][value="lan"]', root, "missing-lan-radio");
  click('[name="net-key-action"][value="replace"]', root,
    "missing-replace-radio");
  const replacement = root.querySelector("#net-replace-key");
  replacement.value = "";
  const beforeBlank = apiRequests(await requests(), "/api/network", "POST").length;
  click("#net-apply", root, "missing-network-apply");
  await waitFor("#net-apply-error", element => !element.hidden,
    "blank-replacement-error-missing");
  check(document.activeElement === replacement, "blank-replacement-focus");
  check(replacement.getAttribute("aria-invalid") === "true",
    "blank-replacement-invalid-state");
  check(apiRequests(await requests(), "/api/network", "POST").length === beforeBlank,
    "blank-replacement-posted");

  const beforeSelectors = apiRequests(await requests(), "/api/agent/config").length;
  const agent = document.querySelector("#ac-agent");
  setValue(agent, "codex");
  const injection = document.querySelector("#ac-inject");
  injection.checked = true;
  injection.dispatchEvent(new Event("change", {bubbles: true}));
  setValue(document.querySelector("#ac-model"), "llama-fixture");
  check(apiRequests(await requests(), "/api/agent/config").length === beforeSelectors,
    "selector-triggered-agent-request");
  click("#ac-show", document, "missing-agent-show");
  const agentOut = await waitFor("#ac-out .snip",
    element => element.textContent.includes(SENTINEL), "agent-preview-missing");
  const agentPosts = apiRequests(await requests(), "/api/agent/config", "POST");
  check(agentPosts.length === beforeSelectors + 1, "agent-preview-post-count");
  check(JSON.stringify(agentPosts.at(-1).body) === JSON.stringify({
    agent: "codex", model: "llama-fixture", backend: "llamacpp",
    small: "", inject: true,
  }), "agent-preview-request-shape");
  const agentContainer = document.querySelector("#ac-out");
  check(secretTextIsInside(agentContainer), "agent-secret-outside-preview");
  const agentCopy = agentContainer.querySelector(".scopy");
  check(agentCopy && !agentCopy.hasAttribute("data-agent-copy-index"),
    "agent-copy-binding-exposed");
  check(hasMinimumPointerTarget(agentCopy), "agent-copy-pointer-target");
  await captureSecretCopy(agentCopy, "agent-private-copy");
  injection.checked = false;
  injection.dispatchEvent(new Event("change", {bubbles: true}));
  check(!document.querySelector("#ac-out .snip") && !bodyHasSecret(),
    "agent-preview-not-destroyed");

  click('.tab[data-tab="models"]', document, "missing-models-tab");
  const llamaRow = await waitFor('.row[data-id="llama-fixture"]', () => true,
    "missing-llama-row");
  click(".rhead", llamaRow, "missing-llama-row-head");
  const llamaClient = await waitFor(
    '.row[data-id="llama-fixture"].open [data-act="client"]', () => true,
    "missing-llama-client-action");
  const clientBefore = apiRequests(await requests(), "/api/client/config", "POST").length;
  llamaClient.focus();
  llamaClient.click();
  const llamaDialog = await waitFor("#modal-root dialog .snip",
    element => element.textContent.includes(SENTINEL), "llama-client-preview-missing");
  const clientPosts = apiRequests(await requests(), "/api/client/config", "POST");
  check(clientPosts.length === clientBefore + 1, "llama-client-post-count");
  check(JSON.stringify(clientPosts.at(-1).body) === JSON.stringify({
    model: "llama-fixture", backend: "llamacpp",
  }), "llama-client-request-shape");
  check(secretTextIsInside(llamaDialog.closest("dialog")),
    "client-secret-outside-dialog");
  const clientCopy = llamaDialog.closest("dialog").querySelector(".scopy");
  check(clientCopy && !clientCopy.hasAttribute("data-private-copy-index"),
    "client-copy-binding-exposed");
  check(hasMinimumPointerTarget(clientCopy), "client-copy-pointer-target");
  await captureSecretCopy(clientCopy, "client-private-copy");
  click("[data-mclose]", llamaDialog.closest("dialog"), "missing-client-close");
  await waitUntil(() => !document.querySelector("#modal-root dialog"),
    "client-dialog-not-closed");
  check(!bodyHasSecret(), "client-secret-after-close");
  check(document.activeElement === llamaClient, "client-dialog-focus-return");

  const vllmRow = document.querySelector('.row[data-id="vllm-fixture"]');
  click(".rhead", vllmRow, "missing-vllm-row-head");
  const vllmClient = await waitFor(
    '.row[data-id="vllm-fixture"].open [data-act="client"]', () => true,
    "missing-vllm-client-action");
  vllmClient.click();
  const vllmDialog = await waitFor("#modal-root dialog .snip",
    element => element.textContent.includes("127.0.0.1:8081"),
    "vllm-client-preview-missing");
  check(!vllmDialog.closest("dialog").textContent.includes(SENTINEL) && !bodyHasSecret(),
    "vllm-client-secret-leak");
  click("[data-mclose]", vllmDialog.closest("dialog"), "missing-vllm-close");
  await waitUntil(() => !document.querySelector("#modal-root dialog"),
    "vllm-dialog-not-closed");

  const generateSecretPanel = async () => {
    const networkRoot = await renderScenario("local-no-key");
    const postCount = apiRequests(await requests(), "/api/network", "POST").length;
    click('[name="net-key-action"][value="generate"]', networkRoot,
      "missing-generate-action");
    click("#net-apply", networkRoot, "missing-generate-apply");
    const panel = await waitFor("#net-generated", element => !element.hidden,
      "generated-panel-missing");
    const display = panel.querySelector("#net-generated-display");
    check(display.textContent === "••••••••••••••••", "generated-not-masked");
    check(!bodyHasSecret(), "generated-secret-rendered-without-reveal");
    await waitUntil(() => !networkRoot.querySelector("#net-apply").disabled,
      "network-apply-not-restored");
    const posts = apiRequests(await requests(), "/api/network", "POST");
    check(posts.length === postCount + 1, "generate-post-count");
    check(JSON.stringify(posts.at(-1).body) === JSON.stringify({
      access_scope: "local", key_action: "generate",
    }), "generate-request-shape");
    return {root: networkRoot, panel, display};
  };

  let generated = await generateSecretPanel();
  check(hasMinimumPointerTarget(
    generated.panel.querySelector("#net-copy-generated")),
  "generated-copy-pointer-target");
  await captureSecretCopy(generated.panel.querySelector("#net-copy-generated"),
    "generated-private-copy", true);
  check(!bodyHasSecret(), "generated-copy-rendered-secret");
  click("#net-reveal-generated", generated.panel, "missing-generated-reveal");
  await waitUntil(() => generated.display.textContent === SENTINEL,
    "generated-reveal-missing");
  check(secretTextIsInside(generated.display), "generated-secret-outside-display");
  await waitUntil(() => {
    const reveal = generated.panel.querySelector("#net-reveal-generated");
    return reveal.disabled && generated.display.getAttribute("aria-label")?.includes("expired");
  }, "generated-secret-did-not-expire", 35000);
  check(!bodyHasSecret(), "generated-secret-after-expiry");
  check(generated.panel.querySelector("#net-copy-generated").disabled,
    "generated-copy-active-after-expiry");
  clipboardValue = "";
  generated.panel.querySelector("#net-copy-generated").click();
  await Promise.resolve();
  check(clipboardValue === "", "generated-closure-live-after-expiry");

  generated = await generateSecretPanel();
  const retryCopy = generated.panel.querySelector("#net-copy-generated");
  click("#net-apply", generated.root, "missing-retry-apply");
  const retryDialog = await waitFor("#net-confirm", element => element.open,
    "retry-confirm-missing");
  await waitUntil(() => retryCopy.onclick === null, "retry-secret-not-cleared");
  check(!bodyHasSecret(), "retry-secret-in-dom");
  retryDialog.dispatchEvent(new Event("cancel", {cancelable: true}));
  await waitUntil(() => !document.querySelector("#net-confirm"),
    "retry-confirm-not-cancelled");

  generated = await generateSecretPanel();
  const doneCopy = generated.panel.querySelector("#net-copy-generated");
  click("#net-dismiss-generated", generated.panel, "missing-generated-done");
  check(doneCopy.onclick === null && !bodyHasSecret(), "done-secret-not-cleared");

  generated = await generateSecretPanel();
  const navCopy = generated.panel.querySelector("#net-copy-generated");
  click('.tab[data-tab="models"]', document, "missing-models-tab");
  await waitUntil(() => navCopy.onclick === null, "navigation-secret-not-cleared");
  check(!bodyHasSecret(), "navigation-secret-in-dom");

  generated = await generateSecretPanel();
  const rerenderCopy = generated.panel.querySelector("#net-copy-generated");
  const oldNetworkRoot = generated.root;
  click('.tab[data-tab="setup"]', document, "missing-setup-tab");
  await waitFor("#network-access", element => element !== oldNetworkRoot,
    "setup-rerender-missing");
  await waitUntil(() => rerenderCopy.onclick === null, "rerender-secret-not-cleared");
  check(!bodyHasSecret(), "rerender-secret-in-dom");

  root = await renderScenario("local-keyed");
  click('[name="net-key-action"][value="generate"]', root,
    "missing-rotate-action");
  const rotateApply = root.querySelector("#net-apply");
  const rotatePostCount = apiRequests(await requests(), "/api/network", "POST").length;
  rotateApply.focus();
  rotateApply.click();
  let confirm = await waitFor("#net-confirm", element => element.open,
    "rotate-confirm-missing");
  check(accessibleName(confirm) === "Rotate router API key",
    "rotate-confirm-name");
  let cancel = confirm.querySelector("#net-confirm-cancel");
  let accept = confirm.querySelector("#net-confirm-accept");
  check(document.activeElement === cancel, "rotate-cancel-initial-focus");
  check(hasMinimumPointerTarget(cancel), "dialog-cancel-pointer-target");
  check(apiRequests(await requests(), "/api/network", "POST").length === rotatePostCount,
    "rotate-post-before-confirm");
  dispatchTab(cancel, true);
  check(document.activeElement === accept, "dialog-shift-tab-trap");
  dispatchTab(accept, false);
  check(document.activeElement === cancel, "dialog-tab-trap");
  confirm.dispatchEvent(new Event("cancel", {cancelable: true}));
  await waitUntil(() => !document.querySelector("#net-confirm"),
    "rotate-escape-cancel");
  check(document.activeElement === rotateApply, "rotate-focus-return");

  root = await renderScenario("local-keyed");
  click('[name="net-key-action"][value="clear"]', root,
    "missing-clear-action");
  const clearApply = root.querySelector("#net-apply");
  const clearPostCount = apiRequests(await requests(), "/api/network", "POST").length;
  clearApply.focus();
  clearApply.click();
  confirm = await waitFor("#net-confirm", element => element.open,
    "clear-confirm-missing");
  check(accessibleName(confirm) === "Remove router API key", "clear-confirm-name");
  cancel = confirm.querySelector("#net-confirm-cancel");
  check(document.activeElement === cancel, "clear-cancel-initial-focus");
  check(apiRequests(await requests(), "/api/network", "POST").length === clearPostCount,
    "clear-post-before-confirm");
  cancel.click();
  await waitUntil(() => !document.querySelector("#net-confirm"),
    "clear-confirm-not-cancelled");
  check(document.activeElement === clearApply, "clear-focus-return");

  root = await renderScenario("forced-restart-failure");
  click('[name="net-scope"][value="lan"]', root, "missing-failure-lan-action");
  click("#net-apply", root, "missing-failure-apply");
  const failure = await waitFor("#net-apply-error", element => !element.hidden,
    "restart-failure-message-missing");
  const failureText = failure.textContent.toLowerCase();
  const observedText = root.querySelector("#net-runtime").textContent.toLowerCase();
  check(failureText.includes("saved") && failureText.includes("restart failed") &&
    failureText.includes("not verified"), "restart-failure-wording");
  check(!observedText.includes("protected") && !observedText.includes("running") &&
    observedText.includes("not verified"), "restart-observation-overclaim");

  root = document.querySelector("#network-access");
  check(root.querySelectorAll("fieldset").length === 2 &&
    [...root.querySelectorAll("fieldset")].every(fieldset => fieldset.querySelector("legend")),
  "network-fieldset-legend");
  check([...root.querySelectorAll("input, select, textarea")].every(hasLabel),
    "network-orphaned-control");
  const described = root.querySelector("#net-replace-key").getAttribute("aria-describedby") || "";
  check(described.includes("net-replace-help") && described.includes("net-apply-error"),
    "network-help-association");
  for (const label of root.querySelectorAll('fieldset > label')) {
    check(hasMinimumPointerTarget(label), "radio-pointer-target");
  }
  for (const button of [root.querySelector("#net-apply")]) {
    check(hasMinimumPointerTarget(button), "button-pointer-target");
  }

  for (const theme of ["dark", "light"]) {
    for (const cvd of [false, true]) {
      document.documentElement.dataset.theme = theme;
      if (cvd) document.documentElement.dataset.cvd = "safe";
      else delete document.documentElement.dataset.cvd;
      for (const element of [
        root.querySelector("#network-title"),
        root.querySelector("#net-configured"),
        root.querySelector('#net-scope-group > label'),
      ]) {
        check(contrastRatio(element) >= 4.5, "network-color-contrast");
      }
    }
  }
  document.documentElement.dataset.theme = "dark";
  delete document.documentElement.dataset.cvd;

  check([...document.querySelectorAll('[role="status"], [role="alert"], #toast')]
    .every(element => !element.textContent.includes(SENTINEL)),
  "secret-in-live-region");
  check(!location.href.includes(SENTINEL), "secret-in-url");
  check(Object.keys(localStorage).every(
    key => !String(localStorage.getItem(key)).includes(SENTINEL)),
  "secret-in-local-storage");
  check((await requests()).every(entry =>
    !JSON.stringify(entry.body).includes(SENTINEL)), "secret-in-operation-log");
  check(!bodyHasSecret(), "secret-in-final-dom");
}

function finish(status, code) {
  scrubSecretDom();
  const result = document.createElement("pre");
  result.id = "lf-security-result";
  result.dataset.status = status;
  result.textContent = status === "pass" ? "PASS" : code;
  document.body.append(result);
}

run().then(
  () => finish("pass", "PASS"),
  error => {
    const code = /^[a-z0-9-]+$/.test(String(error?.message || ""))
      ? error.message : "unexpected-fixture-failure";
    finish("fail", code);
  },
);
