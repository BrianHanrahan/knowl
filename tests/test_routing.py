"""Tests for knowl.voice.routing — intent classification."""

import pytest

from knowl.voice.routing import Intent, classify


class TestClassifyChat:
    """Default — anything that doesn't match a command or capture prefix."""

    def test_plain_text(self):
        intent = classify("What is the weather today?")
        assert intent.kind == "chat"
        assert intent.text == "What is the weather today?"

    def test_empty_string(self):
        intent = classify("")
        assert intent.kind == "chat"
        assert intent.text == ""

    def test_whitespace_only(self):
        intent = classify("   ")
        assert intent.kind == "chat"
        assert intent.text == ""

    def test_partial_prefix(self):
        # "noted" shouldn't trigger capture
        intent = classify("noted that the tests pass")
        assert intent.kind == "chat"


class TestClassifyCapture:
    """Capture mode — triggered by "note", "remember", "save", etc."""

    def test_note_prefix(self):
        intent = classify("note the API uses OAuth 2.0")
        assert intent.kind == "capture"
        assert intent.text == "the API uses OAuth 2.0"

    def test_note_with_colon(self):
        intent = classify("note: always use UTC for timestamps")
        assert intent.kind == "capture"
        assert intent.text == "always use UTC for timestamps"

    def test_remember_prefix(self):
        intent = classify("remember that Brian prefers Tailwind CSS")
        assert intent.kind == "capture"
        assert intent.text == "that Brian prefers Tailwind CSS"

    def test_save_prefix(self):
        intent = classify("save the database runs on port 5432")
        assert intent.kind == "capture"
        assert intent.text == "the database runs on port 5432"

    def test_capture_prefix(self):
        intent = classify("capture this is a test")
        assert intent.kind == "capture"
        assert intent.text == "this is a test"

    def test_add_to_context(self):
        intent = classify("add to context the backend uses FastAPI")
        assert intent.kind == "capture"
        assert intent.text == "the backend uses FastAPI"

    def test_note_only_no_text(self):
        # Just "note" with nothing after it — fall through to chat
        intent = classify("note")
        assert intent.kind == "chat"

    def test_case_insensitive(self):
        intent = classify("Note use camelCase for variables")
        assert intent.kind == "capture"
        assert intent.text == "use camelCase for variables"


class TestClassifyCommand:
    """Command mode — fixed vocabulary for project/context management."""

    def test_switch_project(self):
        intent = classify("switch to my-app")
        assert intent.kind == "command"
        assert intent.command == "switch_project"
        assert intent.args["name"] == "my-app"

    def test_switch_project_shorthand(self):
        intent = classify("switch my-app")
        assert intent.kind == "command"
        assert intent.command == "switch_project"
        assert intent.args["name"] == "my-app"

    def test_use_project(self):
        intent = classify("use project my-app")
        assert intent.kind == "command"
        assert intent.command == "switch_project"
        assert intent.args["name"] == "my-app"

    def test_list_projects(self):
        intent = classify("list projects")
        assert intent.kind == "command"
        assert intent.command == "list_projects"

    def test_show_projects(self):
        intent = classify("show all projects")
        assert intent.kind == "command"
        assert intent.command == "list_projects"

    def test_list_context(self):
        intent = classify("list context files")
        assert intent.kind == "command"
        assert intent.command == "list_context"

    def test_show_context(self):
        intent = classify("show context")
        assert intent.kind == "command"
        assert intent.command == "list_context"

    def test_status(self):
        intent = classify("status")
        assert intent.kind == "command"
        assert intent.command == "show_status"

    def test_show_status(self):
        intent = classify("show status")
        assert intent.kind == "command"
        assert intent.command == "show_status"

    def test_create_project(self):
        intent = classify("create project new-thing")
        assert intent.kind == "command"
        assert intent.command == "create_project"
        assert intent.args["name"] == "new-thing"

    def test_new_project(self):
        intent = classify("new project test-app")
        assert intent.kind == "command"
        assert intent.command == "create_project"
        assert intent.args["name"] == "test-app"

    def test_promote(self):
        intent = classify("promote conventions.md from my-app")
        assert intent.kind == "command"
        assert intent.command == "promote"
        assert intent.args["file"] == "conventions.md"
        assert intent.args["project"] == "my-app"

    def test_promote_no_project(self):
        intent = classify("promote conventions.md")
        assert intent.kind == "command"
        assert intent.command == "promote"
        assert intent.args["file"] == "conventions.md"
        assert "project" not in intent.args

    def test_clear_history(self):
        intent = classify("clear history")
        assert intent.kind == "command"
        assert intent.command == "clear_history"

    def test_clear_chat_history(self):
        intent = classify("clear chat history")
        assert intent.kind == "command"
        assert intent.command == "clear_history"

    def test_inspect_context(self):
        intent = classify("inspect context")
        assert intent.kind == "command"
        assert intent.command == "inspect_context"

    def test_preview_context(self):
        intent = classify("preview context")
        assert intent.kind == "command"
        assert intent.command == "inspect_context"


class TestCommandPriority:
    """Commands take priority over capture prefixes."""

    def test_command_over_capture(self):
        # "switch to X" should be a command, not captured text
        intent = classify("switch to my-app")
        assert intent.kind == "command"
