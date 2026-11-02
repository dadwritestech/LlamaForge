---
title: Blind Arena
section: guides
order: 16
---

# Blind Arena

Which of your models is actually better at the things you ask? The **Arena** tab answers that the way LMArena does, but on your own prompts and your own GPUs. Two models answer the same prompt; you see them only as **Left** and **Right**, in random order. You vote, and only then do the names and speeds show. Every vote updates an Elo board kept on your PC.

## Playing a round

1. Pick two models, or click **Surprise me** for a random pair. Left and Right are shuffled after you click **Ask both**, so the order of the two boxes gives nothing away.
2. Type a prompt you care about: a bug you hit this week, an email to rewrite, a question you know the answer to.
3. Read both answers, then pick **Left is better**, **Tie** or **Right is better**. The names and tokens per second appear after the vote.

Both sides get the same request: your prompt as a single user message, the model's own sampler settings from `models.ini`, and at most 1024 tokens. An answer that hits the limit says so. If a model thinks before answering, its reasoning is folded away under **thinking**, so the vote is about the answer.

## Loading

The arena loads a model when it isn't running. Models it loaded itself are unloaded when the next round needs a different one; models you had loaded before are left alone. With **Multi-model** on, the planner may refuse a load that doesn't fit beside what's running; the round then shows why on that side, and you can unload something or set a `ctx-size` for the model.

A model without a `ctx-size` takes its full trained context, which for some small models is far more memory than the weights. If a round says a model needs more than your GPUs have, give it a `ctx-size` in its settings.

Only llama.cpp and ik_llama.cpp chat models take part. Embedding and reranker models are left out, and so is vLLM.

## The board

Each model starts at 1000. A win against an equal model moves 16 points each way; beating a higher-rated model moves more (K = 32). A few dozen votes are needed before the order means much.

The board lives in `arena.json` next to `config.json`. **Share image** draws the board as a 1200x630 card (top six models, Elo, W-L-T, the vote count and your GPU names) to download or copy; it holds no prompts, paths or hosts. **Copy as markdown** copies a table you can paste into a post or an issue. **Reset board** removes every vote.

## API

| Route | What it does |
|---|---|
| `GET /api/arena` | The models that can take part, the board, the number of games, the markdown table and the GPU names. |
| `POST /api/arena/pair` | Two different models at random: `{a, b}`. |
| `POST /api/arena/answer` | `{model, prompt}` → `{text, thinking, tokens, tok_s, cut, seconds}`. Loads the model if needed. |
| `POST /api/arena/vote` | `{a, b, winner}` with `winner` `"a"`, `"b"` or `"tie"`. Returns the new board. |
| `POST /api/arena/reset` | Removes every vote. |
