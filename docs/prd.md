# Knowl — Personal Context Manager for AI

## 1. Product Overview

Knowl is a voice-first personal context manager that makes AI interactions effortless. It captures, organizes, and surfaces the right context so that every conversation with an AI assistant feels like talking to someone who already knows you, your projects, and your preferences.

Knowl borrows the hierarchical context file model pioneered by tools like Claude Code (CLAUDE.md), but makes the context **visible, editable, and user-controlled**. Users can see exactly what context is active, organize it across projects, and manage it with their voice.

## 2. Core Concepts

### Context Files
Markdown files containing structured knowledge — facts about the user, project details, coding conventions, preferences, domain knowledge, etc. These are the atomic units of context.

### Global Context
A set of context files that are **always active** across every project and conversation. Contains personal identity, universal preferences, communication style, and other information that applies everywhere.

### Project Context
Context files scoped to a specific project or domain of work. Only active when that project is selected. Examples: a codebase's architecture decisions, a client's brand guidelines, a research topic's key papers and findings.

### Context Promotion
Project-level context can be **promoted** to global scope when it proves to be universally relevant. The system actively monitors project context and suggests promotions when it detects patterns that look global in nature (e.g., a coding style preference appearing identically across multiple projects).

### Context Transparency
The user always sees which context files are active in any given interaction. Context is never hidden or implicit — it is a first-class, inspectable part of the experience.

## 3. Goals

- **Voice-first interaction**: Speaking is the primary way to capture context, ask questions, and manage projects. Typing is supported but secondary.
- **Effortless AI conversations**: The right context is always present, so the user never has to repeat themselves or re-explain background.
- **Full transparency**: Users see exactly what context is being sent to the AI. No black-box behavior.
- **Hierarchical and indexed**: Context is structured so that only relevant pieces are loaded, avoiding context window overload.
- **Smart curation**: The system helps users manage their context by suggesting promotions, detecting redundancy, and identifying gaps.
- **Project organization**: Users can maintain multiple projects, each with its own context, and switch between them fluidly.

## 4. Non-Goals

- Replacing Claude Code or other AI coding assistants directly.
- Building a general-purpose note-taking app (context files are specifically for AI interaction).
- Real-time collaborative editing of context files.
- Hosting or running LLMs locally (Knowl manages context and sends it to the Claude API).
- Supporting multiple LLM providers (Claude-only for now; other providers can be added later).
- Mobile app (desktop/web first).

## 5. User Personas & Stories

### Power User / Developer
- As a developer, I want to speak my project context into Knowl so it's captured without breaking my flow.
- As a developer, I want to select which context files are active before starting an AI conversation about a specific project.
- As a developer, I want to see exactly what context the AI is receiving so I can debug unexpected responses.

### Knowledge Worker / Consultant
- As a consultant, I want separate projects for each client with distinct context, so AI interactions are appropriately scoped.
- As a knowledge worker, I want my global preferences (writing style, terminology) to apply everywhere automatically.
- As a knowledge worker, I want the system to notice when a client-specific insight is actually universal and suggest I promote it.

### Researcher
- As a researcher, I want to capture key findings by voice as I read papers, building up project context over time.
- As a researcher, I want to switch between research projects and have the AI immediately understand the relevant domain.

## 6. System Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│                   Knowl UI                       │
│  (Desktop app — PySide6 or Web)                 │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
│  │  Voice   │ │ Context  │ │    Project       │  │
│  │  Input   │ │ Viewer   │ │    Selector      │  │
│  │ (Whisper)│ │ & Editor │ │                  │  │
│  └──────────┘ └──────────┘ └─────────────────┘  │
│  ┌──────────────────────────────────────────┐    │
│  │         Active Context Panel             │    │
│  │  (shows what's being sent to AI)         │    │
│  └──────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────┐    │
│  │         AI Conversation View             │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────────────┐
│  Context Store  │  │   Promotion Engine       │
│  (filesystem)   │  │   (analyzes context      │
│                 │  │    across projects,       │
│  ~/.knowl/      │  │    suggests promotions)   │
│  ├── global/    │  └─────────────────────────┘
│  ├── projects/  │
│  │   ├── proj1/ │
│  │   └── proj2/ │
│  └── index.json │
└─────────────────┘
```

### Context Store Layout

```
~/.knowl/
├── global/                        # Always-active context
│   ├── identity.md                # Who the user is
│   ├── preferences.md             # Universal preferences and style
│   ├── coding-style.md            # Coding conventions (if applicable)
│   └── ...
├── projects/
│   ├── my-saas-app/
│   │   ├── project.md             # Project overview and goals
│   │   ├── architecture.md        # Technical architecture
│   │   ├── conventions.md         # Project-specific conventions
│   │   └── context.json           # Metadata: which files are active, last used, etc.
│   └── client-acme/
│       ├── project.md
│       ├── brand-guidelines.md
│       └── context.json
├── index.json                     # Master index of all context files with summaries
└── config.json                    # App settings, API keys reference, voice config
```

### Context Loading Strategy

1. **Always load**: All files in `~/.knowl/global/` (kept concise — aim for < 2K tokens total).
2. **Load on project select**: Active files from the selected project's directory.
3. **Index-based retrieval**: For large projects, maintain a summary index so the AI can request specific context files on demand rather than loading everything upfront.
4. **Token budgeting**: Track estimated token count of active context and warn the user when approaching limits.

## 7. Key Features

### 7.1 Voice Capture & Intent Routing
- Press-and-hold or toggle to record voice input.
- Transcription via local Whisper (reuses existing transcription infrastructure).
- Voice input is routed based on **explicit mode switching** (not ML classification). Three modes:
  - **Chat** (default): Transcribed text goes to the Claude conversation.
  - **Capture**: Transcribed text is appended to the active context file (user selects target, or defaults to a scratchpad).
  - **Command**: Text is parsed against a fixed command vocabulary (`switch to <project>`, `promote <file>`, `list projects`, `show context`).
- User switches mode via voice prefix ("note:", "hey knowl:") or UI toggle button.
- Fallback: if command parsing fails, treat as chat. The active mode is always shown in the UI.

### 7.2 Context Editor
- View and edit any context file in a markdown editor.
- Files are plain markdown on disk — editable outside Knowl too.
- Create, rename, delete context files within a project or global scope.

### 7.3 Active Context Panel
- Always-visible panel showing which context files are currently active.
- Toggle individual files on/off for the current conversation.
- Shows estimated token count per file and total.
- Expandable to preview file contents inline.

### 7.4 Project Manager
- Create/switch/archive projects.
- Each project has its own set of context files and conversation history.
- Quick-switch between projects with voice or UI.

### 7.5 Context Promotion
- Manual: user selects a project context file and promotes it to global.
- Assisted: system analyzes context across projects and surfaces suggestions.
  - "You have similar coding style notes in 3 projects — promote to global?"
  - "This preference appears project-specific but matches your global identity — merge?"
- Promotion creates a copy in global (or merges into existing global file) and optionally removes the project-level version.

### 7.6 Claude Integration
- Chat interface that sends active context + user message to **Anthropic Claude** via the `anthropic` Python SDK.
- Configuration in `~/.knowl/config.json`: model name (default: `claude-sonnet-4-6`) and `ANTHROPIC_API_KEY` environment variable.
- Context files are concatenated into the system prompt with section headers (e.g., "## Global: identity.md", "## Project: architecture.md").
- Conversation history is scoped per project.
- User can inspect the full prompt (context + message) before sending.

#### Token Budgeting
- Default budget: 8,000 tokens for context (configurable per project in `context.json`).
- Token estimation uses a character-based heuristic (~4 chars per token) as a conservative estimate.
- Loading priority when budget is exceeded: (1) global context (mandatory, always loaded), (2) project files in `active_files` order, (3) skip remaining files with a warning in the Active Context Panel.
- If a single file exceeds the budget: truncate with a "[truncated — full file: <path>]" marker.

#### Context Conflict Resolution
- **Project overrides global** on conflict (same precedence model as Claude Code's CLAUDE.md hierarchy).
- No automatic merging — conflicts are surfaced to the user via the Active Context Panel.
- The promotion engine flags conflicts during promotion and asks the user to resolve before completing.

### 7.7 Agent Dispatch (Build Mode)
- Knowl can dispatch tasks to **CLI-based coding agents** like Claude Code, Aider, or Continue.dev, passing along the assembled context.
- The user describes what they want built (by voice or text), Knowl assembles the relevant context, and hands the task off to the chosen agent.
- **Supported agents**:
  - **Claude Code**: Launches with assembled context injected via CLAUDE.md files or CLI arguments.
  - **Other agents**: Extensible to future coding agents that accept context + task descriptions.
- Context files from Knowl are translated into the agent's native format (e.g., CLAUDE.md for Claude Code, system prompts for other agents).
- The user stays in control: they can review what context will be sent before dispatch, and see the agent's output within Knowl.
- This makes Knowl a **context orchestration layer** — it doesn't replace coding agents, it makes them more effective by giving them the right context every time.

## 8. Technical Considerations

### Tech Stack
- **Language**: Python 3.10+
- **Voice**: OpenAI Whisper (local, already in repo)
- **UI**: PySide6 (desktop) — can evaluate web (Flask/FastAPI + React) later
- **AI API**: Anthropic Claude via the `anthropic` Python SDK. Users provide an `ANTHROPIC_API_KEY` environment variable.
- **Storage**: Filesystem-based (markdown files + JSON metadata)
- **Dependencies**: Minimal — avoid heavy frameworks

### Security
- **API key**: `ANTHROPIC_API_KEY` is read from an environment variable only — never stored in config files or on disk.
- **Path validation**: All store operations validate that paths resolve within `~/.knowl/`. Path traversal (`..`) is rejected.
- **File size limit**: 100KB per context file (configurable). Prevents accidental inclusion of large files.
- **Prompt injection**: Context files are user-controlled and should only contain trusted content. No automatic inclusion of files from untrusted sources.

### Context Format
- Plain markdown for maximum portability and human readability.
- JSON metadata files for indexing, token estimates, and project state.
- Files are version-friendly (work well with git).

### Voice Pipeline
- Reuse existing Whisper transcription code from `app.py`.
- Add intent routing layer on top of raw transcription.
- Consider wake-word or hotkey activation for hands-free operation.

## 9. Success Metrics

- User can go from opening Knowl to having a context-rich AI conversation in under 30 seconds.
- Context files are always visible and the user never asks "what context is the AI seeing?"
- Voice capture feels natural — transcription latency < 3 seconds for short utterances.
- Promotion suggestions are relevant at least 70% of the time.
- Switching between projects takes < 2 seconds.

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Context files grow too large for AI context windows | Token budgeting, index-based lazy loading, summarization |
| Voice transcription errors corrupt context | Always show transcription for confirmation before saving to context |
| Promotion engine suggests irrelevant promotions | Start with conservative heuristics; let user train with accept/dismiss |
| Users don't maintain context files | Voice-first capture lowers friction; AI can help draft/refine context |
| Scope creep into note-taking app | Stay focused: context files exist to make AI conversations better, not as general knowledge management |

## 11. Implementation Plan

### Phase 1: Foundation
- Restructure repo: extract Whisper transcription into a reusable module.
- Implement context store (filesystem layout, read/write, index management).
- Build basic project manager (create, switch, list projects).
- Build global context management (create, edit, list global files).

### Phase 2: Context-Aware Conversations
- Integrate Claude API via `anthropic` SDK for AI conversations.
- Build active context panel — show what's being sent.
- Implement context assembly: global + project context → structured system prompt.
- Add token estimation and budgeting.

### Phase 3: Voice Integration
- Wire Whisper transcription into the UI as primary input.
- Add intent routing (conversation vs. context capture vs. commands).
- Implement voice-driven project switching and context management.

### Phase 4: Smart Promotion
- Build promotion engine: analyze context across projects for patterns.
- Surface promotion suggestions in UI.
- Implement merge/promote workflows.

### Phase 5: Agent Dispatch
- Implement Claude Code integration: generate CLAUDE.md from active context files and launch tasks.
- Build dispatch UI: review context → choose agent → send task → view results.
- Add extensibility hooks for future coding agents (Aider, Continue.dev, etc.).

### Phase 6: Polish & Extensibility
- Refine UI/UX based on usage.
- Add conversation history management.
- Consider web UI option.
- Explore MCP server integration for bidirectional Claude Code ↔ Knowl communication.
