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
