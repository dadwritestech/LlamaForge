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
4. Play again: **Same prompt, new pair** asks two other models at random; **Rematch these two** asks the same two again, with Left and Right shuffled anew. The **Recent prompts** list above the prompt box holds your last ten prompts (in this browser only).

Both sides get the same request: your prompt as a single user message, the model's own sampler settings from `models.ini`, and at most 1024 tokens. An answer that hits the limit says so. If a model thinks before answering, its reasoning is folded away under **thinking**, so the vote is about the answer.

## Loading

The arena loads a model when it isn't running. Models it loaded itself are unloaded when the next round needs a different one; models you had loaded before are left alone. With **Multi-model** on, the planner may refuse a load that doesn't fit beside what's running; the round then shows why on that side, and you can unload something or set a `ctx-size` for the model.

A model without a `ctx-size` takes its full trained context, which for some small models is far more memory than the weights. If a round says a model needs more than your GPUs have, give it a `ctx-size` in its settings.

Only llama.cpp and ik_llama.cpp chat models take part. Embedding and reranker models are left out, and so is vLLM.

## The board

Each model starts at 1000. A win against an equal model moves 16 points each way; beating a higher-rated model moves more (K = 32). A few dozen votes are needed before the order means much.

The board lives in `arena.json` next to `config.json`. **Share image** draws the board as a 1200x630 card (top six models, Elo, W-L-T, the vote count and your GPU names) to download or copy; it holds no prompts, paths or hosts. **Copy as markdown** copies a table you can paste into a post or an issue. **Reset board** removes every vote.

## Checked tasks

Votes tell you which answers you like. **Checked tasks**, below the board, tells you which models get things right. It asks each model 15 short tasks that have one right answer, three in each of five groups:

| Group | Example |
|---|---|
| Extract | The new invoice number in a message that mentions two. |
| Math | The largest context that fits in a GPU, given weights and KV cache per token. |
| JSON | A JSON object with given keys, from a sentence. |
| Follow | Exactly seven words, or three lines with no numbering. |
| Reason | A day of the week 100 days from now. |

A script marks each reply against the known answer; model output is only compared, never run. Thinking is ignored, so a model is marked on its final answer. Each answer may use up to 4096 tokens so thinking models can finish; one that still runs out is marked wrong and flagged as cut off.

Tick the models and press **Run checks**. Models load one at a time, as in a round, and the ones you had loaded stay loaded. A run takes a few minutes per model and keeps going if you leave the tab; **Cancel** stops it and keeps the models that finished. Arena rounds wait while a run is going.

The table shows passed tasks per group and the median decode speed. The model in your main slot is marked as your main model, and every other model gets one line against it, for example "gemma-4-12b beats qwen3-8b on 3 of 5 categories (1 tie), 40% slower." A model that wins no category "trails" or "matches" instead. **What ... got wrong** shows each missed task with the start of the model's reply.

Fifteen tasks with one try each is a smoke test, not a benchmark: a small model can pass a task in one run and miss it in the next. Results live in `proving.json` next to `config.json`; running a model again replaces its row.

**Share image** makes a 1200x630 picture of the table: one square per task (green right, dark wrong), passed count, speed and your GPU names. When your fastest model isn't the one that gets most right, the card says so in two lines. It holds no answers, prompts, paths or hosts.

**Every night at** runs the checks overnight on its own. Pick a time and press **Schedule**: from then until 06:00, once nobody has used a model for 20 minutes, LlamaForge runs the models that were ticked when you scheduled it (or every chat model if all were ticked). It runs once a night, only while LlamaForge is running, and the results are waiting in the table in the morning. **Turn off** stops it.

## API

| Route | What it does |
|---|---|
| `GET /api/arena` | The models that can take part, the board, the number of games, the markdown table and the GPU names. |
| `POST /api/arena/pair` | Two different models at random: `{a, b}`. |
| `POST /api/arena/answer` | `{model, prompt}` → `{text, thinking, tokens, tok_s, cut, seconds}`. Loads the model if needed. |
| `POST /api/arena/vote` | `{a, b, winner}` with `winner` `"a"`, `"b"` or `"tie"`. Returns the new board. |
| `POST /api/arena/reset` | Removes every vote. |
| `GET /api/proving` | Checked tasks: `status` (`running`, `model`, `done`, `total`, `errors`, `cancelled`), the models, the tasks, `scores` and `results` per model, `driver` (the main slot's model) and `verdicts`. |
| `POST /api/proving/run` | `{models?}` starts a run on those chat models (all of them if left out). 409 while one is going. |
| `POST /api/proving/cancel` | Stops the run after the current answer. |
| `POST /api/proving/schedule` | `{at, models?}` runs the checks every night from `at` (`HH:MM`) to 06:00 once the models have been idle 20 minutes; `models` left out or `null` means every chat model. `{at: null}` turns it off. |
