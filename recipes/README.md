# Community recipes

Tested llama.cpp setups for specific models, one JSON file each. LlamaForge lists
this folder live (**Profiles → recipes**), so a merged recipe shows up for everyone
without a release. Import one and you get the same model file (downloaded from
Hugging Face if you don't have it) and the same settings.

## Share yours

1. In LlamaForge, save a launch profile for a setup that works well, then click **↗** on it
   and copy the recipe.
2. Save it here as `recipes/<short-name>.json` and add an `about` block:
   ```json
   "about": {
     "title": "Gemma 4 26B-A4B on one 16 GB GPU, 90k context",
     "hardware": "RTX 4070 Ti Super 16 GB, 64 GB DDR5",
     "author": "your-github-name",
     "notes": "What the key knobs do and when to change them."
   }
   ```
3. Open a pull request.

## Rules

- **Only setups you've run on your own hardware.** Say what that hardware is.
- `model.hf_repo` must point at a public Hugging Face repo that has the file.
- Notes explain the knobs. No speed claims we can't reproduce; if you include a
  number, say how you measured it.
- One recipe per file. Short lowercase file names, `-` between words.
- Recipes carry tuning knobs only, from an explicit allowlist in
  `backend/recipes.py` (context, KV cache, offload, sampling, speculative,
  chat/reasoning). Anything else is dropped on import: paths, hosts, URLs, keys,
  tools, server behaviour such as timeouts or logging, and your GPU layout. CI
  rejects a recipe here that contains one (`tests/test_gallery.py`).
- Use the canonical knob names (`ctx-size`, not `c`). Leave out `gpu-layers`
  and `tensor-split`, so `--fit` can size the model to the importer's card.
