# Knowl

A voice-first personal context manager that makes AI interactions effortless.

Knowl captures, organizes, and surfaces the right context so that every conversation with an AI — regardless of provider — feels like talking to someone who already knows you, your projects, and your preferences.

## What It Does

- **Voice-driven context capture**: Speak to add knowledge, facts, and preferences to your personal context store.
- **Transparent context management**: Always see exactly what context is being sent to your AI. Nothing hidden.
- **Global + project context**: Maintain always-on personal context alongside project-specific knowledge. Promote project context to global when it proves universal.
- **Bring your own LLM**: Works with Claude, GPT, Gemini, Ollama, or any OpenAI-compatible endpoint. You choose the model.
- **Smart curation**: The system helps identify context that should be promoted from project to global scope.
- **Agent dispatch**: Hand off tasks to Claude Code, Codex, or other coding agents with full context assembled automatically.

## Status

Early development. See [docs/prd.md](docs/prd.md) for the full product vision and implementation plan.

## Context Store

Knowl stores context as plain markdown files on disk:

```
~/.knowl/
├── global/           # Always-active context (identity, preferences, style)
├── projects/         # Per-project context files
│   ├── my-app/
│   └── client-work/
├── index.json        # Master index with summaries
└── config.json       # LLM provider config, voice settings
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prerequisites

- Python 3.10+
- `ffmpeg` on PATH (required by Whisper for voice input)
- An API key for your LLM provider of choice
