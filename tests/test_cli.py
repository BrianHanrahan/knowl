"""Tests for knowl.cli."""

import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from knowl.context import store
from knowl.cli import main


@pytest.fixture(autouse=True)
def use_tmp_knowl_dir(tmp_path):
    """Point the store at a temp directory for every test."""
    store.set_knowl_dir(tmp_path / ".knowl")
    store.init_store()
    yield


def run_cli(*args: str, expect_exit: int | None = None) -> str:
    """Run the CLI with the given args, capturing stdout."""
    captured = StringIO()
    with patch.object(sys, "argv", ["knowl"] + list(args)), \
         patch("sys.stdout", captured):
        try:
            main()
        except SystemExit as e:
            if expect_exit is not None:
                assert e.code == expect_exit
            elif e.code not in (0, None):
                raise
    return captured.getvalue()


class TestInit:
    def test_init(self):
        output = run_cli("init")
        assert "initialized" in output.lower()


class TestProjectCommands:
    def test_create_and_list(self):
        run_cli("project", "create", "my-app")
        output = run_cli("project", "list")
        assert "my-app" in output

    def test_switch(self):
        run_cli("project", "create", "proj1")
        run_cli("project", "switch", "proj1")
        config = store.load_config()
        assert config["active_project"] == "proj1"

    def test_switch_nonexistent(self):
        output = run_cli("project", "switch", "nope", expect_exit=1)


class TestContextCommands:
    def test_list_global(self):
        (store.GLOBAL_DIR / "identity.md").write_text("# Me")
        output = run_cli("context", "list", "--global")
        assert "identity.md" in output

    def test_add_from_stdin(self):
        with patch("sys.stdin", StringIO("# Test content")), \
             patch("sys.stdin.isatty", return_value=False):
            run_cli("context", "add", "test.md", "--global")
        content = store.read_context_file(store.GLOBAL_DIR / "test.md")
        assert content == "# Test content"

    def test_show(self):
        (store.GLOBAL_DIR / "show-me.md").write_text("hello world")
        output = run_cli("context", "show", "show-me.md")
        assert "hello world" in output

    def test_promote(self):
        run_cli("project", "create", "proj")
        (store.PROJECTS_DIR / "proj" / "conventions.md").write_text("# Conventions")
        run_cli("context", "promote", "conventions.md", "--from", "proj")
        assert (store.GLOBAL_DIR / "conventions.md").exists()


class TestContextInspect:
    def test_inspect_empty(self):
        output = run_cli("context", "inspect")
        assert "No context files" in output

    def test_inspect_with_files(self):
        (store.GLOBAL_DIR / "identity.md").write_text("# Identity\nI am a developer.")
        output = run_cli("context", "inspect")
        assert "global/identity.md" in output
        assert "System Prompt Preview" in output
        assert "I am a developer." in output

    def test_inspect_shows_token_count(self):
        (store.GLOBAL_DIR / "identity.md").write_text("# Identity\nI am a developer.")
        output = run_cli("context", "inspect")
        assert "tokens" in output


class TestChat:
    def test_chat_quit(self):
        """Test that /quit exits the chat loop."""
        (store.GLOBAL_DIR / "identity.md").write_text("# Identity")
        with patch("builtins.input", side_effect=["/quit"]), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            output = run_cli("chat")
        assert "Knowl Chat" in output
        assert "Goodbye" in output

    def test_chat_clear(self):
        """Test that /clear resets history."""
        store.save_history(None, [{"role": "user", "content": "old"}])
        with patch("builtins.input", side_effect=["/clear", "/quit"]), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            output = run_cli("chat")
        assert "History cleared" in output
        assert store.load_history(None) == []

    def test_chat_context_command(self):
        """Test that /context shows active files."""
        (store.GLOBAL_DIR / "identity.md").write_text("# Identity")
        with patch("builtins.input", side_effect=["/context", "/quit"]), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            output = run_cli("chat")
        assert "global/identity.md" in output

    def test_chat_sends_message(self):
        """Test that a regular message gets sent to Claude."""
        (store.GLOBAL_DIR / "identity.md").write_text("# Identity")
        with patch("builtins.input", side_effect=["Hello Claude", "/quit"]), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}), \
             patch("knowl.llm.claude.send_message", return_value="Hello human!") as mock_send:
            output = run_cli("chat")
        mock_send.assert_called_once()
        assert "Hello human!" in output

    def test_chat_persists_history(self):
        """Test that conversation history is saved."""
        with patch("builtins.input", side_effect=["Hi", "/quit"]), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}), \
             patch("knowl.llm.claude.send_message", return_value="Hello!"):
            run_cli("chat")
        history = store.load_history(None)
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hi"}
        assert history[1] == {"role": "assistant", "content": "Hello!"}

    def test_chat_no_api_key(self, monkeypatch):
        """Test that missing API key gives a clear error."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        run_cli("chat", expect_exit=1)


class TestHistoryClear:
    def test_clear_history(self):
        store.save_history(None, [{"role": "user", "content": "old"}])
        output = run_cli("history", "clear")
        assert "cleared" in output.lower()
        assert store.load_history(None) == []

    def test_clear_project_history(self):
        store.create_project("proj")
        store.save_history("proj", [{"role": "user", "content": "old"}])
        output = run_cli("history", "clear", "--project", "proj")
        assert "cleared" in output.lower()
        assert store.load_history("proj") == []


class TestStatus:
    def test_status(self):
        output = run_cli("status")
        assert "Knowl store" in output
        assert "claude-sonnet-4-6" in output
