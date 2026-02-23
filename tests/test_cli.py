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


class TestStatus:
    def test_status(self):
        output = run_cli("status")
        assert "Knowl store" in output
        assert "claude-sonnet-4-6" in output
