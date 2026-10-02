# Contributing to LlamaForge

Thanks for helping! LlamaForge is a small, solo-maintained project, so a few
conventions keep things moving fast.

## Ground rules
- **Backend is Python standard library only.** No new required dependencies.
  Optional extras (like the tray icon) must be imported lazily and degrade to a no-op.
- **Frontend is vanilla JS** in `web/` — no framework, no build step. Every dynamic
  value goes through `esc()`, and `setHTML()` is the single HTML sink.
- **LlamaForge never does inference itself.** It drives llama.cpp / ik_llama / vLLM.
  Features that belong upstream should go upstream.

## Running tests
```bash
cd tests
python -m unittest discover -s . -p "test_*.py"
```
Tests must be run from `tests/`. Tests that touch config must point
`config.CONFIG` at a temp file — never the real `config.json`.

## Running the UI
`python backend/server.py` starts the panel on port 8090. It renders even with no
engine running, which is handy for UI work.

## Pull requests
- Keep PRs focused — one fix or feature each.
- Include a screenshot for UI changes, ideally in both light and dark themes, and
  check narrow widths (the sidebar changes layout below 900px and 600px).
- Small PRs get reviewed fastest. For bigger ideas, open an issue or Discussion first.

## Good first issues
Look for the [`good first issue`](https://github.com/dadwritestech/LlamaForge/labels/good%20first%20issue)
label.
