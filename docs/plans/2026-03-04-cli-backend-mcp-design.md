# Design: CLI Backend with MCP Tool Integration

**Date:** 2026-03-04
**Status:** Approved

## Problem

Knowl currently requires an Anthropic API key for all LLM interactions. When API credits run out, the app is unusable. Users with Claude Max subscriptions should be able to use Knowl through their existing subscription.

## Solution

Add a CLI backend that routes LLM requests through the `claude -p` CLI (which uses the user's Claude Max subscription) and exposes Knowl's tools as an MCP server so Claude Code can call them.

## Architecture

```
User -> Knowl Web UI -> claude -p (with Knowl MCP server) -> Anthropic (via subscription)
                              |
                     Knowl MCP Server (stdio)
                     +-- web_search
                     +-- fetch_page
                     +-- list_context_files
                     +-- read_context_file
                     +-- write_context_file
                     +-- delete_context_file
                     +-- create_tool
                     +-- [approved custom tools]
```

## Components

### 1. Knowl MCP Server (`knowl/mcp/server.py`)

A stdio-transport MCP server exposing Knowl's existing tools. Wraps the same functions that `execute_tool()` in `server.py` already calls.

- **Entry point:** `python -m knowl.mcp.server --project <name>`
- **Transport:** stdio (JSON-RPC over stdin/stdout)
- **Tools:** All built-in tools + approved custom tools for the active project
- **Registration:** Added to `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "knowl": {
      "command": "python3",
      "args": ["-m", "knowl.mcp.server", "--project", "knowl"],
      "env": { "KNOWL_DIR": "/path/to/.knowl" }
    }
  }
}
```

### 2. CLI Backend (`knowl/llm/cli_backend.py`)

Replaces `anthropic.Anthropic()` calls with `claude -p` subprocess calls. Same async generator interface as `claude.py`.

**Invocation:**
```bash
claude -p "<prompt>" \
  --output-format stream-json \
  --verbose \
  --allowedTools "mcp__knowl__*" \
  --append-system-prompt "<knowl context>"
```

**Key behaviors:**
- System prompt (assembled context) passed via `--append-system-prompt`
- `--allowedTools "mcp__knowl__*"` auto-approves Knowl MCP tools
- Conversation continuity via `--resume <session_id>`
- Parses NDJSON stream, yields same `StreamEvent` types (TextChunkEvent, ToolCallEvent, DoneEvent)

**Key difference from API mode:** Claude Code runs the tool loop. Knowl observes tool calls for UI rendering but doesn't execute them — the MCP server handles execution when Claude Code calls it.

### 3. Configuration

`.knowl/config.json` gains a `backend` field:

```json
{
  "llm": {
    "model": "claude-sonnet-4-6",
    "backend": "cli"
  }
}
```

- `"api"` — current behavior, requires `ANTHROPIC_API_KEY`
- `"cli"` — uses Claude Code CLI with subscription auth (default)

### 4. Server Integration

Minimal change in `server.py`. Chat route picks backend based on config:

```python
backend = config.get("llm", {}).get("backend", "cli")
if backend == "cli":
    from knowl.llm.cli_backend import stream_message_with_tools
else:
    from knowl.llm.claude import stream_message_with_tools
```

Both return the same `StreamEvent` types — SSE event_stream logic stays identical.

The `format-file` endpoint also needs a CLI-mode path using `claude -p` with a one-shot prompt.

### 5. Setup Command

`python -m knowl.mcp.setup` registers the Knowl MCP server in `~/.claude.json`. Run once.

### 6. Frontend

Small toggle in settings to switch between "API Key" and "Claude Code" backend modes. `UpdateConfig` pydantic model extended with `backend` field.

## What Stays the Same

- All tool implementations (web_search, context tools, custom tools) — unchanged, wrapped in MCP handlers
- Tool management UI (approve/reject/disable) — unchanged
- History persistence — unchanged

## Trade-offs

| Aspect | API mode | CLI mode |
|--------|----------|----------|
| Auth | API key required | Claude Max subscription |
| Tool loop | Knowl controls it | Claude Code controls it |
| Streaming | Real-time token streaming | NDJSON parsing (slight latency) |
| Attachments | Images, PDFs, text files | Text files only (initially) |
| Session mgmt | Knowl manages history | Claude Code session + Knowl history |
| Cost | Pay per token | Included in subscription |

## Prior Art

This follows the same pattern as OpenClaw's architecture: an independent tool runtime that feeds tools to the LLM via MCP, with the LLM provider handling the agent loop. OpenClaw uses `pi-agent-core` for its own loop; we use Claude Code's built-in loop via `claude -p`, which is simpler and Anthropic-sanctioned for subscription use.

## Dependencies

- `mcp` Python package for MCP server
- `claude` CLI installed and authenticated with Max subscription
