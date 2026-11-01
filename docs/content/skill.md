---
title: Agent Skill
section: guides
order: 14
---

# Agent Skill

LlamaForge ships an [agent skill](https://github.com/dadwritestech/LlamaForge/blob/master/skills/llamaforge/SKILL.md): a short playbook that teaches Claude Code, Codex and other coding agents how to drive LlamaForge. With it, an agent can tell you which models fit your GPUs, download and load one, Auto-Forge it to its fastest setup, and hand grunt work to it, without you explaining any of it.

## Install

Any agent that reads skills (Claude Code, Codex, Cursor and others, through [skills](https://github.com/vercel-labs/skills)):

```bash
npx skills add dadwritestech/LlamaForge
```

Claude Code as a plugin:

```
/plugin marketplace add dadwritestech/LlamaForge
/plugin install llamaforge@llamaforge
```

The skill needs LlamaForge running on the same machine. If it isn't installed, the skill tells the agent to ask you before installing it.

## What it does

- **Connects the MCP tools.** It reads `GET /api/mcp/setup` and runs the one-line command for your agent, so the [MCP server](mcp.md) tools show up in its next session. Until then it uses the plain HTTP API.
- **"What can I run?"** It reads your GPUs, searches Hugging Face, and picks the largest quant that fits with room for context. It asks before downloading.
- **Make it fast.** It starts an [Auto-Forge](autoforge.md) run (after asking: the run unloads every model), follows it, shows you the before and after, and applies the winner only when you say yes.
- **Hand off grunt work.** Summaries, boilerplate and bulk edits go to the local model through `ask` or `pi_run`, and the agent checks what comes back.
- **Claude Code on a local model.** It checks whether a model can run Claude Code and opens it on a folder.

## Ground rules it gives the agent

- Ask before anything slow, big or disruptive: downloads, Auto-Forge, applying a result, replacing your main model.
- Go through LlamaForge rather than editing `models.ini` or killing `llama-server`.
- Never print the router API key.
- Quote speeds measured on your machine, never promised ones.
