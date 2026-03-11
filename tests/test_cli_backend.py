"""Tests for knowl.llm.cli_backend."""

import sys
from unittest.mock import patch

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


class TestStreamMessageInterface:
    """Verify the function signature matches the API backend."""

    def test_function_exists(self):
        from knowl.llm.cli_backend import stream_message_with_tools
        import inspect
        sig = inspect.signature(stream_message_with_tools)
        params = list(sig.parameters.keys())
        assert "user_message" in params
        assert "tool_executor" in params
        assert "context" in params
        assert "history" in params
        assert "model" in params
        assert "project" in params
        assert "attachments" in params

    def test_format_with_cli_exists(self):
        from knowl.llm.cli_backend import format_with_cli
        import inspect
        sig = inspect.signature(format_with_cli)
        params = list(sig.parameters.keys())
        assert "prompt" in params
        assert "model" in params
