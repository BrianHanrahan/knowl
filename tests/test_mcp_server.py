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
    async def test_list_context_files(self):
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
            assert "projects" in data

    @pytest.mark.asyncio
    async def test_read_context_file_not_found(self):
        with patch("knowl.mcp.server.store") as mock_store:
            mock_store.PROJECTS_DIR = Path("/tmp/test_knowl/projects")
            mock_store.read_context_file.return_value = None
            result = await _handle_tool_call(
                "read_context_file", {"path": "/nonexistent"}, project=None
            )
            data = json.loads(result)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_write_context_file_success(self):
        with patch("knowl.mcp.server.store") as mock_store:
            mock_store.GLOBAL_DIR = Path("/tmp/test_knowl/global")
            mock_store.PROJECTS_DIR = Path("/tmp/test_knowl/projects")
            mock_store.write_context_file.return_value = None
            mock_store.estimate_tokens.return_value = 10
            result = await _handle_tool_call(
                "write_context_file",
                {"filename": "test.md", "scope": "global", "content": "# Test"},
                project=None,
            )
            data = json.loads(result)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_delete_context_file_success(self):
        with patch("knowl.mcp.server.store") as mock_store:
            mock_store.PROJECTS_DIR = Path("/tmp/test_knowl/projects")
            mock_store.delete_context_file.return_value = True
            result = await _handle_tool_call(
                "delete_context_file", {"path": "/tmp/test_knowl/global/test.md"}, project=None
            )
            data = json.loads(result)
            assert data["success"] is True

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
            assert server.name == "knowl"
