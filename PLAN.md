# Knowl Web UI — Implementation Plan

## Architecture Overview

**Backend:** FastAPI (Python) — thin REST API over existing `store`, `claude`, and `promotion` modules
**Frontend:** React + Vite (TypeScript) — single-page app with hot reload
**Chat streaming:** Server-Sent Events (SSE) via Anthropic streaming API
**Launch:** `knowl serve` CLI command starts FastAPI + serves built React assets

---

## File Structure

```
knowl/
├── ui/
│   ├── __init__.py          # (existing, empty)
│   ├── server.py            # FastAPI app + all API routes
│   └── frontend/            # React/Vite project
│       ├── package.json
│       ├── vite.config.ts
│       ├── tsconfig.json
│       ├── index.html
│       └── src/
│           ├── main.tsx      # React entry point
│           ├── App.tsx       # Main layout: sidebar + main panel
│           ├── api.ts        # Fetch helpers for all API endpoints
│           ├── components/
│           │   ├── ProjectSelector.tsx    # Dropdown to switch projects
│           │   ├── ContextPanel.tsx       # File list with toggle + token counts
│           │   ├── ChatView.tsx           # Conversation UI with streaming
│           │   ├── ContextEditor.tsx      # Markdown editor for context files
│           │   ├── ContextInspect.tsx     # Preview assembled system prompt
│           │   └── TokenBudget.tsx        # Token budget bar/display
│           └── styles/
│               └── app.css               # Minimal CSS (or use Tailwind)
```

---

## Backend: FastAPI Routes (`knowl/ui/server.py`)

### Project endpoints
| Method | Path | Maps to | Purpose |
|--------|------|---------|---------|
| `GET` | `/api/projects` | `store.list_projects()` | List all projects |
| `POST` | `/api/projects` | `store.create_project(name)` | Create project |
| `DELETE` | `/api/projects/{name}` | `store.delete_project(name)` | Delete project |
| `GET` | `/api/config` | `store.load_config()` | Get current config (active project, model) |
| `PUT` | `/api/config` | `store.save_config(config)` | Update config (switch project, change model) |

### Context file endpoints
| Method | Path | Maps to | Purpose |
|--------|------|---------|---------|
| `GET` | `/api/context/global` | `store.list_global_files()` | List global files with token counts |
| `GET` | `/api/context/project/{name}` | `store.list_project_files(name)` | List project files with token/active status |
| `GET` | `/api/context/file` | `store.read_context_file(path)` | Read file content (query param: `path`) |
| `PUT` | `/api/context/file` | `store.write_context_file(path, content)` | Write file content |
| `DELETE` | `/api/context/file` | `store.delete_context_file(path)` | Delete file |
| `PUT` | `/api/context/active/{project}` | `store.set_active_files(project, files)` | Toggle active files |
| `POST` | `/api/context/promote` | `store.promote_to_global(project, file)` | Promote file to global |

### Context assembly endpoints
| Method | Path | Maps to | Purpose |
|--------|------|---------|---------|
| `GET` | `/api/inspect` | `store.assemble_context() + claude.format_system_prompt()` | Preview system prompt |
| `GET` | `/api/tokens/{project}` | `store.estimate_active_context_tokens(project)` | Token breakdown |

### Chat endpoints
| Method | Path | Maps to | Purpose |
|--------|------|---------|---------|
| `GET` | `/api/history/{project}` | `store.load_history(project)` | Load chat history |
| `DELETE` | `/api/history/{project}` | `store.clear_history(project)` | Clear history |
| `POST` | `/api/chat` | SSE streaming via `claude.send_message()` (streaming variant) | Send message, stream response |

### Promotion endpoints
| Method | Path | Maps to | Purpose |
|--------|------|---------|---------|
| `GET` | `/api/promotions` | `promotion.scan_for_promotions()` | Get promotion suggestions |
| `POST` | `/api/promotions/apply` | `promotion.apply_promotion()` | Apply a promotion |

### Static files
- In production: serve built React assets from `knowl/ui/frontend/dist/`
- In development: Vite dev server on port 5173, FastAPI on port 8000, Vite proxies `/api` to FastAPI

---

## Chat Streaming Design

The existing `claude.send_message()` uses `client.messages.create()` (non-streaming). We need a streaming variant:

1. Add `stream_message()` to `knowl/llm/claude.py`:
   - Uses `client.messages.stream()` instead of `.create()`
   - Yields text delta chunks as a generator

2. The `/api/chat` endpoint:
   - Accepts `POST` with `{ message, project?, model? }`
   - Assembles context, loads history
   - Calls `stream_message()` and returns SSE response
   - After streaming completes, saves to history

3. Frontend `ChatView.tsx`:
   - Uses `EventSource` or `fetch()` with streaming reader
   - Appends chunks to the current message bubble in real-time

---

## Frontend Components

### `App.tsx` — Layout
```
┌──────────────────────────────────────────────┐
│  Knowl         [Project ▼]     [model badge] │
├──────────────┬───────────────────────────────┤
│              │                               │
│  Context     │         Chat View             │
│  Panel       │                               │
│              │   user> ...                    │
│  ☑ global/   │   claude> ...                 │
│    file.md   │                               │
│  ☑ project/  │                               │
│    api.md    │                               │
│  ☐ project/  │                               │
│    old.md    │                               │
│              │  ┌─────────────────┐ [Send]   │
│  [Token bar] │  │ Type message... │          │
│  [+ Add]     │  └─────────────────┘          │
│  [Inspect]   │                               │
├──────────────┴───────────────────────────────┤
│  [Editor tab]  [Inspect tab]                 │
└──────────────────────────────────────────────┘
```

- **Sidebar:** Project selector, context file list (checkboxes for active), token budget bar
- **Main panel:** Chat view (default), context editor (on file click), context inspect (on button click)
- **Bottom section:** Tabbed editor / inspector (optional — could be modal instead)

### `ProjectSelector.tsx`
- Dropdown of projects from `GET /api/projects`
- Switching calls `PUT /api/config` with new `active_project`
- Refreshes context panel and chat history

### `ContextPanel.tsx`
- Lists global files (always shown, always active) + project files
- Checkboxes to toggle active status → `PUT /api/context/active/{project}`
- Shows token count per file
- Click file name → opens in ContextEditor
- "Add file" button → creates new file

### `ChatView.tsx`
- Loads history from `GET /api/history/{project}`
- Input box at bottom
- Submit → `POST /api/chat` → reads SSE stream → renders chunks
- "Clear history" button
- Markdown rendering for assistant responses

### `ContextEditor.tsx`
- Textarea/code editor for markdown
- Loads from `GET /api/context/file?path=...`
- Save button → `PUT /api/context/file`
- Delete button → `DELETE /api/context/file`

### `ContextInspect.tsx`
- Reads from `GET /api/inspect?project=...&budget=...`
- Shows formatted system prompt in read-only view
- Shows file list with token counts

### `TokenBudget.tsx`
- Visual bar showing used vs. budget
- Per-file breakdown from `GET /api/tokens/{project}`

---

## CLI Command: `knowl serve`

Add to `knowl/cli.py`:
```python
def cmd_serve(args):
    from knowl.ui.server import create_app
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
```

Flags: `--host 0.0.0.0`, `--port 8000`, `--dev` (enables CORS for Vite dev server)

---

## Implementation Order

### Step 1: Backend API (`knowl/ui/server.py`)
- FastAPI app with all routes
- SSE streaming for chat
- Add `stream_message()` to `knowl/llm/claude.py`
- CORS middleware for dev mode
- Static file serving for production

### Step 2: React Frontend scaffold
- `npm create vite@latest` in `knowl/ui/frontend/`
- Configure proxy to backend in `vite.config.ts`
- Set up `api.ts` with all fetch helpers

### Step 3: Core UI components
- `App.tsx` layout with sidebar + main panel
- `ProjectSelector.tsx`
- `ContextPanel.tsx` with file listing + toggle
- `TokenBudget.tsx`

### Step 4: Chat
- `ChatView.tsx` with SSE streaming
- Message history display
- Markdown rendering

### Step 5: Editor + Inspector
- `ContextEditor.tsx` — edit markdown files
- `ContextInspect.tsx` — preview system prompt

### Step 6: Polish + CLI integration
- `knowl serve` command
- Build script (`npm run build` → `dist/`)
- Serve built assets from FastAPI in production mode
- Tests for API endpoints

---

## New Dependencies

**Python (add to requirements.txt):**
- `fastapi`
- `uvicorn[standard]`

**Node (in `knowl/ui/frontend/package.json`):**
- `react`, `react-dom`
- `@vitejs/plugin-react`
- `typescript`

No heavy UI libraries — keep it lean. Use CSS modules or a small CSS file for styling.

---

## What NOT to change
- All existing modules (`store`, `claude`, `promotion`, `dispatch`, `voice`) remain unchanged
- All existing CLI commands remain unchanged
- All 153 existing tests remain passing
- The only new code is `knowl/ui/server.py`, the React frontend, and the `knowl serve` CLI command
