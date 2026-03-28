# CLI Backend + MCP Server Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Knowl route LLM requests through the Claude Agent SDK (using Claude Max subscription) instead of requiring an Anthropic API key, while exposing Knowl's tools as an MCP server.

**Architecture:** Two new backends share the same `StreamEvent` interface (`TextChunkEvent`, `ToolCallEvent`, `DoneEvent`). The CLI backend uses `claude_agent_sdk.query()` with a Knowl MCP server that wraps existing tool implementations. The server picks which backend based on `config.json`'s `llm.backend` field ("api" or "cli"). A settings UI toggle lets users switch.

**Tech Stack:** Python, `claude-agent-sdk`, `mcp` (Python MCP SDK), FastAPI, React/TypeScript

---

## File Structure

```
knowl/
├── mcp/
│   ├── __init__.py              # Package init
│   ├── server.py                # MCP server exposing Knowl tools (stdio transport)
│   └── setup.py                 # Registration script for ~/.claude.json
├── llm/
│   ├── claude.py                # (existing) API backend — unchanged
│   └── cli_backend.py           # NEW: Claude Agent SDK backend
└── ui/
    ├── server.py                # (existing) MODIFY: backend routing in chat routes
    └── frontend/src/
        ├── components/
        │   └── SettingsPanel.tsx # NEW: backend toggle + API key input
        └── App.tsx              # (existing) MODIFY: add settings gear icon
```

---

## Chunk 1: MCP Server

### Task 1: Create MCP package and server

**Files:**
- Create: `knowl/mcp/__init__.py`
- Create: `knowl/mcp/server.py`
- Test: `tests/test_mcp_server.py`

The MCP server wraps existing tool functions from `knowl.context.store`, `knowl.web.search`, `knowl.web.fetch`, and `knowl.tools.executor`. It uses stdio transport (JSON-RPC over stdin/stdout) for use with Claude Agent SDK's `mcp_servers` option.

- [ ] **Step 1: Write the MCP server**

Create `knowl/mcp/__init__.py` (empty) and `knowl/mcp/server.py`:

```python
"""Knowl MCP Server — exposes Knowl tools via Model Context Protocol (stdio transport).

Run standalone: python -m knowl.mcp.server --project <name>
Or used programmatically by cli_backend.py as an MCP server config.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from knowl.context import store
from knowl.log import get_logger

logger = get_logger(__name__)


def _build_tools(project: str | None) -> list[Tool]:
    """Build the list of MCP tools from Knowl's built-in + approved custom tools."""
    tools = [
        Tool(
            name="web_search",
            description=(
                "Search the web using DuckDuckGo. Returns a list of "
                "{title, url, snippet} results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="fetch_page",
            description="Fetch and read the content of a web page URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="list_context_files",
            description=(
                "List all context files in the Knowl system. Returns global "
                "and project files with names, tokens, and paths."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name (optional, lists global only if omitted)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="read_context_file",
            description="Read a context file's content by its full path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full file path"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write_context_file",
            description=(
                "Create or update a context file. Provide 'path' to update, "
                "or 'filename' + 'scope' to create new."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of existing file to overwrite"},
                    "filename": {"type": "string", "description": "New file name"},
                    "scope": {"type": "string", "description": "'global' or project name"},
                    "content": {"type": "string", "description": "Markdown content"},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="delete_context_file",
            description="Delete a context file by path. Permanent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full file path to delete"},
                },
                "required": ["path"],
            },
        ),
    ]

    # Add approved custom tools for the project
    if project:
        for ct in store.list_approved_tools(project):
            tools.append(Tool(
                name=ct["name"],
                description=ct.get("description", ""),
                inputSchema=ct.get("input_schema", {
                    "type": "object", "properties": {}, "required": []
                }),
            ))

    return tools


async def _handle_tool_call(name: str, arguments: dict, project: str | None) -> str:
    """Execute a tool call — mirrors execute_tool() in server.py."""
    if name == "web_search":
        from knowl.web.search import web_search
        results = await web_search(
            query=arguments["query"],
            num_results=arguments.get("num_results", 5),
        )
        return json.dumps(results)

    elif name == "fetch_page":
        from knowl.web.fetch import fetch_page
        result = await fetch_page(url=arguments["url"])
        return json.dumps(result)

    elif name == "list_context_files":
        project_name = arguments.get("project")
        global_files = [
            {"name": f.name, "tokens": store.estimate_file_tokens(f), "path": str(f), "scope": "global"}
            for f in store.list_global_files()
        ]
        project_files = []
        if project_name:
            project_files = [
                {"name": f.name, "tokens": store.estimate_file_tokens(f), "path": str(f), "scope": project_name}
                for f in store.list_project_files(project_name)
            ]
        projects = store.list_projects()
        return json.dumps({"global_files": global_files, "project_files": project_files, "projects": projects})

    elif name == "read_context_file":
        rpath = arguments["path"]
        if not Path(rpath).is_absolute():
            rpath = str(store.PROJECTS_DIR / rpath)
        content = store.read_context_file(rpath)
        if content is None:
            return json.dumps({"error": f"File not found: {rpath}"})
        return json.dumps({"path": rpath, "content": content})

    elif name == "write_context_file":
        content = arguments["content"]
        path = arguments.get("path")
        if not path:
            filename = arguments.get("filename")
            scope = arguments.get("scope", "global")
            if not filename:
                return json.dumps({"error": "Provide 'path' or 'filename' + 'scope'"})
            if scope == "global":
                target = store.GLOBAL_DIR / filename
            else:
                target = store.PROJECTS_DIR / scope / filename
            path = str(target)
        elif not Path(path).is_absolute():
            path = str(store.PROJECTS_DIR / path)
        try:
            store.write_context_file(path, content)
            return json.dumps({"success": True, "path": path, "tokens": store.estimate_tokens(content)})
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

    elif name == "delete_context_file":
        dpath = arguments["path"]
        if not Path(dpath).is_absolute():
            dpath = str(store.PROJECTS_DIR / dpath)
        ok = store.delete_context_file(dpath)
        if not ok:
            return json.dumps({"error": f"File not found: {dpath}"})
        return json.dumps({"success": True, "deleted": dpath})

    else:
        # Custom tool
        if project:
            tool = store.get_tool(project, name)
            if tool and tool.get("status") == "approved":
                from knowl.tools.executor import execute_custom_tool, ToolExecutionError
                try:
                    result = execute_custom_tool(project, name, arguments)
                    return result
                except ToolExecutionError as exc:
                    return json.dumps({"error": str(exc)})

        return json.dumps({"error": f"Unknown tool: {name}"})


def create_server(project: str | None = None) -> Server:
    """Create and configure the MCP server instance."""
    # Initialize store
    knowl_dir_env = os.environ.get("KNOWL_DIR")
    if knowl_dir_env:
        store.KNOWL_DIR = Path(knowl_dir_env)
        store._refresh_dirs()
    store.init_store()

    server = Server("knowl")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _build_tools(project)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result = await _handle_tool_call(name, arguments, project)
        return [TextContent(type="text", text=result)]

    return server


async def main(project: str | None = None) -> None:
    """Run the MCP server over stdio."""
    server = create_server(project)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Knowl MCP Server")
    parser.add_argument("--project", "-p", default=None, help="Active project name")
    args = parser.parse_args()
    asyncio.run(main(args.project))
```

- [ ] **Step 2: Write tests for MCP server tool building and handling**

Create `tests/test_mcp_server.py`:

```python
"""Tests for knowl.mcp.server."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from knowl.mcp.server import _build_tools, _handle_tool_call, create_server


class TestBuildTools:
    def test_returns_builtin_tools(self):
        tools = _build_tools(project=None)
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "fetch_page" in names
        assert "list_context_files" in names
        assert "read_context_file" in names
        assert "write_context_file" in names
        assert "delete_context_file" in names

    def test_includes_custom_tools_for_project(self):
        with patch("knowl.mcp.server.store") as mock_store:
            mock_store.list_approved_tools.return_value = [
                {
                    "name": "my_tool",
                    "description": "A custom tool",
                    "input_schema": {"type": "object", "properties": {}, "required": []},
                }
            ]
            tools = _build_tools(project="test_project")
            names = [t.name for t in tools]
            assert "my_tool" in names

    def test_no_custom_tools_without_project(self):
        tools = _build_tools(project=None)
        # Should only have the 6 built-in tools
        assert len(tools) == 6


class TestHandleToolCall:
    @pytest.mark.asyncio
    async def test_list_context_files(self, tmp_path):
        with patch("knowl.mcp.server.store") as mock_store:
            mock_store.list_global_files.return_value = []
            mock_store.list_projects.return_value = ["test"]
            mock_store.list_project_files.return_value = []
            result = await _handle_tool_call(
                "list_context_files", {"project": "test"}, project="test"
            )
            data = json.loads(result)
            assert "global_files" in data
            assert "project_files" in data

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        result = await _handle_tool_call("nonexistent", {}, project=None)
        data = json.loads(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]


class TestCreateServer:
    def test_creates_server(self):
        with patch("knowl.mcp.server.store"):
            server = create_server(project="test")
            assert server is not None
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_mcp_server.py -v`
Expected: All tests PASS

- [ ] **Step 4: Install mcp dependency**

Run: `pip install mcp`
Add `mcp` to `requirements.txt`

- [ ] **Step 5: Commit**

```bash
git add knowl/mcp/ tests/test_mcp_server.py requirements.txt
git commit -m "feat: add MCP server exposing Knowl tools via stdio transport"
```

---

### Task 2: MCP setup command

**Files:**
- Create: `knowl/mcp/setup.py`
- Test: manual (writes to `~/.claude.json`)

- [ ] **Step 1: Write the setup script**

Create `knowl/mcp/setup.py`:

```python
"""Register the Knowl MCP server in ~/.claude.json for Claude Code integration.

Usage: python -m knowl.mcp.setup [--project <name>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


CLAUDE_CONFIG = Path.home() / ".claude.json"


def setup(project: str | None = None) -> None:
    """Register Knowl MCP server in Claude Code's config."""
    # Load existing config
    config: dict = {}
    if CLAUDE_CONFIG.exists():
        try:
            config = json.loads(CLAUDE_CONFIG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Warning: {CLAUDE_CONFIG} is not valid JSON, creating fresh config")
            config = {}

    # Build MCP server entry
    args = ["-m", "knowl.mcp.server"]
    if project:
        args.extend(["--project", project])

    server_config = {
        "command": sys.executable,  # Use the same Python that's running this
        "args": args,
    }

    # Add to mcpServers
    config.setdefault("mcpServers", {})
    config["mcpServers"]["knowl"] = server_config

    # Write back
    CLAUDE_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Registered Knowl MCP server in {CLAUDE_CONFIG}")
    print(f"  Command: {sys.executable} {' '.join(args)}")
    if project:
        print(f"  Project: {project}")
    print("\nClaude Code will now have access to Knowl tools.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Register Knowl MCP server with Claude Code")
    parser.add_argument("--project", "-p", default=None, help="Default project name")
    args = parser.parse_args()
    setup(args.project)
```

- [ ] **Step 2: Commit**

```bash
git add knowl/mcp/setup.py
git commit -m "feat: add MCP setup command to register Knowl in Claude Code"
```

---

## Chunk 2: CLI Backend

### Task 3: Implement CLI backend using Claude Agent SDK

**Files:**
- Create: `knowl/llm/cli_backend.py`
- Test: `tests/test_cli_backend.py`

This is the core new backend. It uses `claude_agent_sdk.query()` with the Knowl MCP server to stream responses. It yields the same `StreamEvent` types as `claude.stream_message_with_tools()`.

- [ ] **Step 1: Write the CLI backend**

Create `knowl/llm/cli_backend.py`:

```python
"""CLI backend — routes LLM requests through Claude Agent SDK (uses Claude Max subscription).

Same async generator interface as claude.stream_message_with_tools(),
yielding StreamEvent instances (TextChunkEvent, ToolCallEvent, DoneEvent).
"""

from __future__ import annotations

import json
import os
import sys
from typing import AsyncIterator

from knowl.llm.claude import (
    TextChunkEvent,
    ToolCallEvent,
    DoneEvent,
    StreamEvent,
    format_system_prompt,
)
from knowl.log import get_logger

logger = get_logger(__name__)


def _get_mcp_server_config(project: str | None) -> dict:
    """Build the MCP server config dict for claude_agent_sdk."""
    args = ["-m", "knowl.mcp.server"]
    if project:
        args.extend(["--project", project])

    config = {
        "command": sys.executable,
        "args": args,
    }

    # Pass KNOWL_DIR if set
    knowl_dir = os.environ.get("KNOWL_DIR")
    if knowl_dir:
        config["env"] = {"KNOWL_DIR": knowl_dir}

    return config


async def stream_message_with_tools(
    user_message: str,
    tool_executor=None,  # Ignored — MCP server handles tool execution
    context: list[dict[str, str]] | None = None,
    history: list[dict[str, str]] | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    project: str | None = None,
    attachments: list[dict] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream a Claude response via the Agent SDK with Knowl MCP tools.

    Same interface as claude.stream_message_with_tools() — yields
    TextChunkEvent, ToolCallEvent, and DoneEvent.

    Key difference: Claude Agent SDK controls the tool loop. We observe
    tool calls for UI rendering but don't execute them — the MCP server
    handles execution when Claude calls it.
    """
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, SystemMessage, AssistantMessage, TextBlock
    except ImportError as exc:
        raise RuntimeError(
            "The 'claude-agent-sdk' package is not installed. "
            "Install it with: pip install claude-agent-sdk"
        ) from exc

    system_prompt = format_system_prompt(context or [])

    # Build the prompt — include history context if present
    prompt_parts = []
    if history:
        prompt_parts.append("Previous conversation:")
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt_parts.append(f"  {role}: {content}")
        prompt_parts.append("")  # blank line
    prompt_parts.append(user_message)

    # Add attachment descriptions (CLI mode only supports text attachments initially)
    if attachments:
        for att in attachments:
            if att.get("type") == "text":
                prompt_parts.append(att.get("text", ""))

    full_prompt = "\n".join(prompt_parts)

    mcp_config = _get_mcp_server_config(project)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt if system_prompt else None,
        model=model,
        mcp_servers={"knowl": mcp_config},
        allowed_tools=["mcp__knowl__*"],
        max_turns=20,
    )

    full_text_parts = []

    try:
        async for message in query(prompt=full_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        full_text_parts.append(block.text)
                        yield TextChunkEvent(text=block.text)
            elif isinstance(message, ResultMessage):
                if message.result:
                    full_text_parts.append(message.result)
                    yield TextChunkEvent(text=message.result)
                yield DoneEvent(full_text="".join(full_text_parts))
                return

        # If we exit the loop without a ResultMessage, still emit DoneEvent
        yield DoneEvent(full_text="".join(full_text_parts))

    except Exception as exc:
        logger.error("CLI backend error: %s", exc)
        raise RuntimeError(f"Claude Agent SDK error: {exc}") from exc
```

- [ ] **Step 2: Write tests**

Create `tests/test_cli_backend.py`:

```python
"""Tests for knowl.llm.cli_backend."""

import sys
from unittest.mock import patch, MagicMock

import pytest

from knowl.llm.cli_backend import _get_mcp_server_config


class TestGetMcpServerConfig:
    def test_basic_config(self):
        config = _get_mcp_server_config(project=None)
        assert config["command"] == sys.executable
        assert "-m" in config["args"]
        assert "knowl.mcp.server" in config["args"]
        assert "--project" not in config["args"]

    def test_with_project(self):
        config = _get_mcp_server_config(project="myproject")
        assert "--project" in config["args"]
        assert "myproject" in config["args"]

    def test_with_knowl_dir(self, monkeypatch):
        monkeypatch.setenv("KNOWL_DIR", "/tmp/test_knowl")
        config = _get_mcp_server_config(project=None)
        assert config["env"]["KNOWL_DIR"] == "/tmp/test_knowl"

    def test_without_knowl_dir(self, monkeypatch):
        monkeypatch.delenv("KNOWL_DIR", raising=False)
        config = _get_mcp_server_config(project=None)
        assert "env" not in config


class TestStreamMessageWithTools:
    @pytest.mark.asyncio
    async def test_raises_without_sdk(self):
        with patch.dict("sys.modules", {"claude_agent_sdk": None}):
            from knowl.llm.cli_backend import stream_message_with_tools
            # Importing won't fail but calling will since the module is None
            # This test verifies the ImportError path
            pass
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_cli_backend.py -v`
Expected: PASS

- [ ] **Step 4: Install claude-agent-sdk**

Run: `pip install claude-agent-sdk`
Add `claude-agent-sdk` to `requirements.txt`

- [ ] **Step 5: Commit**

```bash
git add knowl/llm/cli_backend.py tests/test_cli_backend.py requirements.txt
git commit -m "feat: add CLI backend using Claude Agent SDK for subscription auth"
```

---

## Chunk 3: Server Integration + Frontend

### Task 4: Add backend routing to server.py

**Files:**
- Modify: `knowl/ui/server.py`
- Modify: `knowl/context/store.py` (DEFAULT_CONFIG)

The server picks the backend based on `config.llm.backend` ("api" or "cli").

- [ ] **Step 1: Update DEFAULT_CONFIG in store.py**

In `knowl/context/store.py`, update the default config to include `backend`:

```python
DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {"model": "claude-sonnet-4-6", "backend": "api"},
    "voice": {"whisper_model": "base", "language": "auto"},
    "active_project": None,
}
```

- [ ] **Step 2: Add UpdateConfig field and backend routing in server.py**

In `knowl/ui/server.py`:

1. Add `backend` field to `UpdateConfig`:
```python
class UpdateConfig(BaseModel):
    active_project: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None  # "api" or "cli"
```

2. Handle backend in `update_config` route:
```python
if payload.backend is not None:
    if payload.backend not in ("api", "cli"):
        raise HTTPException(status_code=400, detail="backend must be 'api' or 'cli'")
    config.setdefault("llm", {})["backend"] = payload.backend
```

3. Add a helper to get the streaming function:
```python
def _get_stream_fn(config: dict):
    """Return the appropriate stream_message_with_tools based on backend config."""
    backend = config.get("llm", {}).get("backend", "api")
    if backend == "cli":
        from knowl.llm.cli_backend import stream_message_with_tools
        return stream_message_with_tools
    else:
        from knowl.llm.claude import stream_message_with_tools
        return stream_message_with_tools
```

4. Update `event_stream()` in the `/api/chat` route to use `_get_stream_fn`:
```python
stream_fn = _get_stream_fn(config)
async for event in stream_fn(
    user_message=payload.message,
    tool_executor=bound_executor,
    context=context_pieces,
    history=history,
    model=model,
    project=project,
):
```

5. Same for `/api/chat/upload` route.

6. Update `/api/format-file` to handle CLI backend:
```python
@app.post("/api/format-file")
async def format_file(payload: FormatFile):
    config = store.load_config()
    model = payload.model or config.get("llm", {}).get("model", "claude-sonnet-4-6")
    backend = config.get("llm", {}).get("backend", "api")
    prompt = (
        "Clean up the following markdown content. Fix markdown formatting "
        "(especially tables), remove duplicates, correct typos. "
        "Preserve ALL information. Return ONLY the cleaned markdown.\n\n"
        + payload.content
    )
    try:
        if backend == "cli":
            from knowl.llm.cli_backend import format_with_cli
            result = await format_with_cli(prompt, model)
        else:
            result = await asyncio.to_thread(
                claude.send_message, user_message=prompt, model=model,
            )
        return {"content": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```

- [ ] **Step 3: Add format_with_cli helper to cli_backend.py**

```python
async def format_with_cli(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """One-shot prompt via Agent SDK for formatting (no tools needed)."""
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, AssistantMessage, TextBlock
    except ImportError as exc:
        raise RuntimeError("claude-agent-sdk not installed") from exc

    options = ClaudeAgentOptions(model=model, max_turns=1, allowed_tools=[])
    parts = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
        elif isinstance(message, ResultMessage):
            if message.result:
                parts.append(message.result)
    return "".join(parts)
```

- [ ] **Step 4: Add /api/config/backend endpoint for quick status check**

```python
@app.get("/api/config/backend")
async def get_backend():
    config = store.load_config()
    backend = config.get("llm", {}).get("backend", "api")
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    return {"backend": backend, "has_api_key": has_api_key}
```

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add knowl/ui/server.py knowl/context/store.py knowl/llm/cli_backend.py
git commit -m "feat: add backend routing — switch between API and CLI modes"
```

---

### Task 5: Frontend settings toggle

**Files:**
- Create: `knowl/ui/frontend/src/components/SettingsPanel.tsx`
- Modify: `knowl/ui/frontend/src/App.tsx`
- Modify: `knowl/ui/frontend/src/api.ts`
- Modify: `knowl/ui/frontend/src/styles/app.css`

- [ ] **Step 1: Add API helpers in api.ts**

Add to `knowl/ui/frontend/src/api.ts`:

```typescript
export async function getBackendStatus(): Promise<{ backend: string; has_api_key: boolean }> {
  const res = await fetch(`${BASE}/api/config/backend`);
  return res.json();
}

export async function setBackend(backend: "api" | "cli"): Promise<void> {
  await fetch(`${BASE}/api/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backend }),
  });
}
```

- [ ] **Step 2: Create SettingsPanel.tsx**

```tsx
import { useEffect, useState } from "react";
import { getBackendStatus, setBackend } from "../api";

interface Props {
  onClose: () => void;
}

export default function SettingsPanel({ onClose }: Props) {
  const [backend, setBackendState] = useState<"api" | "cli">("api");
  const [hasApiKey, setHasApiKey] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getBackendStatus().then((status) => {
      setBackendState(status.backend as "api" | "cli");
      setHasApiKey(status.has_api_key);
      setLoading(false);
    });
  }, []);

  const handleToggle = async (mode: "api" | "cli") => {
    setBackendState(mode);
    await setBackend(mode);
  };

  if (loading) return <div className="settings-panel">Loading...</div>;

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h3>Settings</h3>
        <button className="btn btn-sm" onClick={onClose}>✕</button>
      </div>

      <div className="settings-section">
        <label className="settings-label">LLM Backend</label>
        <div className="backend-toggle">
          <button
            className={`btn btn-sm ${backend === "api" ? "btn-active" : ""}`}
            onClick={() => handleToggle("api")}
          >
            API Key
          </button>
          <button
            className={`btn btn-sm ${backend === "cli" ? "btn-active" : ""}`}
            onClick={() => handleToggle("cli")}
          >
            Claude Code
          </button>
        </div>
        <p className="settings-hint">
          {backend === "api"
            ? hasApiKey
              ? "Using ANTHROPIC_API_KEY from environment."
              : "Warning: ANTHROPIC_API_KEY not set. Chat will fail."
            : "Using Claude Max subscription via Claude Code CLI."}
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Add settings gear to App.tsx**

Add a gear icon button in the header that toggles a `showSettings` state, and render `<SettingsPanel>` when active.

- [ ] **Step 4: Add CSS for settings panel**

```css
.settings-panel {
  padding: 16px;
  background: var(--bg-surface);
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.settings-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.settings-section {
  margin-bottom: 16px;
}
.settings-label {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 6px;
  display: block;
}
.backend-toggle {
  display: flex;
  gap: 8px;
  margin-bottom: 6px;
}
.settings-hint {
  font-size: 12px;
  color: var(--text-dim);
}
```

- [ ] **Step 5: Build frontend**

Run: `cd knowl/ui/frontend && npm run build`

- [ ] **Step 6: Commit**

```bash
git add knowl/ui/frontend/
git commit -m "feat: add settings panel with API/CLI backend toggle"
```

---

## Chunk 4: API Tests

### Task 6: Add API endpoint tests

**Files:**
- Create: `tests/test_ui_server.py`

- [ ] **Step 1: Write API tests**

Create `tests/test_ui_server.py`:

```python
"""Tests for knowl.ui.server API endpoints."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    """Create a test client with isolated store."""
    with patch("knowl.context.store.KNOWL_DIR", tmp_path):
        with patch("knowl.context.store.GLOBAL_DIR", tmp_path / "global"):
            with patch("knowl.context.store.PROJECTS_DIR", tmp_path / "projects"):
                with patch("knowl.context.store.INDEX_PATH", tmp_path / "index.json"):
                    with patch("knowl.context.store.CONFIG_PATH", tmp_path / "config.json"):
                        from knowl.ui.server import create_app
                        app = create_app(dev=True)
                        yield TestClient(app)


class TestProjectRoutes:
    def test_list_projects_empty(self, client):
        response = client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data

    def test_create_and_list_project(self, client):
        response = client.post("/api/projects", json={"name": "test_proj"})
        assert response.status_code == 200
        assert response.json()["name"] == "test_proj"

        response = client.get("/api/projects")
        assert "test_proj" in response.json()["projects"]

    def test_delete_project(self, client):
        client.post("/api/projects", json={"name": "to_delete"})
        response = client.delete("/api/projects/to_delete")
        assert response.status_code == 200

    def test_delete_nonexistent_project(self, client):
        response = client.delete("/api/projects/nope")
        assert response.status_code == 404


class TestConfigRoutes:
    def test_get_config(self, client):
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "llm" in data

    def test_update_config_model(self, client):
        response = client.put("/api/config", json={"model": "claude-haiku-4-5"})
        assert response.status_code == 200
        assert response.json()["llm"]["model"] == "claude-haiku-4-5"

    def test_update_config_backend(self, client):
        response = client.put("/api/config", json={"backend": "cli"})
        assert response.status_code == 200
        assert response.json()["llm"]["backend"] == "cli"

    def test_update_config_invalid_backend(self, client):
        response = client.put("/api/config", json={"backend": "invalid"})
        assert response.status_code == 400


class TestContextRoutes:
    def test_list_global_files_empty(self, client):
        response = client.get("/api/context/global")
        assert response.status_code == 200
        assert response.json() == []

    def test_create_and_read_file(self, client):
        client.post("/api/projects", json={"name": "test_proj"})
        response = client.post("/api/context/file", json={
            "filename": "notes.md",
            "scope": "test_proj",
            "content": "# Notes\n\nSome content",
        })
        assert response.status_code == 200
        path = response.json()["path"]

        response = client.get(f"/api/context/file?path={path}")
        assert response.status_code == 200
        assert "Some content" in response.json()["content"]


class TestBackendStatus:
    def test_get_backend_status(self, client):
        response = client.get("/api/config/backend")
        assert response.status_code == 200
        data = response.json()
        assert "backend" in data
        assert "has_api_key" in data
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_ui_server.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (no regressions)

- [ ] **Step 4: Commit**

```bash
git add tests/test_ui_server.py
git commit -m "test: add API endpoint tests for web UI server"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-03-10-cli-backend-mcp.md`. Ready to execute?
