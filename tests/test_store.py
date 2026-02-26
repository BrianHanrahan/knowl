"""Tests for knowl.context.store."""

import json

import pytest

from knowl.context import store


@pytest.fixture(autouse=True)
def use_tmp_knowl_dir(tmp_path):
    """Point the store at a temp directory for every test."""
    store.set_knowl_dir(tmp_path / ".knowl")
    yield
    # Restore default (not strictly necessary since tests are isolated)
    store.set_knowl_dir(tmp_path / ".knowl")


class TestInitStore:
    def test_creates_directory_structure(self):
        store.init_store()
        assert store.GLOBAL_DIR.exists()
        assert store.PROJECTS_DIR.exists()
        assert store.INDEX_PATH.exists()
        assert store.CONFIG_PATH.exists()

    def test_default_config(self):
        store.init_store()
        config = store.load_config()
        assert config["llm"]["model"] == "claude-sonnet-4-6"
        assert config["active_project"] is None

    def test_idempotent(self):
        store.init_store()
        store.init_store()  # Should not error
        assert store.GLOBAL_DIR.exists()


class TestCreateProject:
    def test_creates_project_directory(self):
        store.init_store()
        path = store.create_project("my-app")
        assert path.exists()
        assert (path / "project.md").exists()
        assert (path / "context.json").exists()

    def test_default_files_content(self):
        store.init_store()
        store.create_project("test-proj")
        md = (store.PROJECTS_DIR / "test-proj" / "project.md").read_text()
        assert "# test-proj" in md
        ctx = json.loads((store.PROJECTS_DIR / "test-proj" / "context.json").read_text())
        assert "project.md" in ctx["active_files"]

    def test_idempotent(self):
        store.init_store()
        store.create_project("proj")
        # Write custom content
        (store.PROJECTS_DIR / "proj" / "project.md").write_text("custom")
        store.create_project("proj")
        # Should not overwrite existing files
        assert (store.PROJECTS_DIR / "proj" / "project.md").read_text() == "custom"


class TestListOperations:
    def test_list_projects(self):
        store.init_store()
        assert store.list_projects() == []
        store.create_project("alpha")
        store.create_project("beta")
        assert store.list_projects() == ["alpha", "beta"]

    def test_list_global_files(self):
        store.init_store()
        assert store.list_global_files() == []
        (store.GLOBAL_DIR / "identity.md").write_text("# Me")
        files = store.list_global_files()
        assert len(files) == 1
        assert files[0].name == "identity.md"

    def test_list_project_files(self):
        store.init_store()
        store.create_project("proj")
        files = store.list_project_files("proj")
        assert any(f.name == "project.md" for f in files)


class TestReadWriteContextFile:
    def test_round_trip(self):
        store.init_store()
        path = store.GLOBAL_DIR / "test.md"
        store.write_context_file(path, "hello world")
        content = store.read_context_file(path)
        assert content == "hello world"

    def test_read_nonexistent(self):
        store.init_store()
        result = store.read_context_file(store.GLOBAL_DIR / "nonexistent.md")
        assert result is None

    def test_size_limit(self):
        store.init_store()
        path = store.GLOBAL_DIR / "big.md"
        big_content = "x" * (store.MAX_CONTEXT_FILE_SIZE + 1)
        with pytest.raises(ValueError, match="exceeds limit"):
            store.write_context_file(path, big_content)


class TestDeleteOperations:
    def test_delete_context_file(self):
        store.init_store()
        path = store.GLOBAL_DIR / "temp.md"
        path.write_text("temporary")
        assert store.delete_context_file(path) is True
        assert not path.exists()

    def test_delete_nonexistent_file(self):
        store.init_store()
        assert store.delete_context_file(store.GLOBAL_DIR / "nope.md") is False

    def test_delete_project(self):
        store.init_store()
        store.create_project("to-delete")
        assert store.delete_project("to-delete") is True
        assert "to-delete" not in store.list_projects()

    def test_delete_nonexistent_project(self):
        store.init_store()
        assert store.delete_project("nope") is False


class TestPromoteToGlobal:
    def test_promote(self):
        store.init_store()
        store.create_project("proj")
        source = store.PROJECTS_DIR / "proj" / "conventions.md"
        source.write_text("# Conventions\n\nUse snake_case.")
        result = store.promote_to_global("proj", "conventions.md")
        assert result is not None
        assert result.exists()
        assert result.parent == store.GLOBAL_DIR
        assert result.read_text() == "# Conventions\n\nUse snake_case."

    def test_promote_nonexistent(self):
        store.init_store()
        store.create_project("proj")
        result = store.promote_to_global("proj", "nope.md")
        assert result is None


class TestPathTraversal:
    def test_read_rejects_traversal(self):
        store.init_store()
        with pytest.raises(ValueError, match="escapes allowed directory"):
            store.read_context_file(store.KNOWL_DIR / ".." / ".." / "etc" / "passwd")

    def test_write_rejects_traversal(self):
        store.init_store()
        with pytest.raises(ValueError, match="escapes allowed directory"):
            store.write_context_file(store.KNOWL_DIR / ".." / "evil.md", "pwned")

    def test_create_project_rejects_traversal(self):
        store.init_store()
        with pytest.raises(ValueError, match="escapes allowed directory"):
            store.create_project("../../evil")


class TestTokenEstimation:
    def test_estimate_tokens(self):
        # ~4 chars per token
        assert store.estimate_tokens("a" * 400) == 100
        assert store.estimate_tokens("") == 1  # minimum 1

    def test_estimate_file_tokens(self):
        store.init_store()
        path = store.GLOBAL_DIR / "test.md"
        path.write_text("a" * 400)
        assert store.estimate_file_tokens(path) == 100

    def test_estimate_active_context_tokens(self):
        store.init_store()
        (store.GLOBAL_DIR / "identity.md").write_text("I am a developer." * 10)
        store.create_project("proj")
        tokens = store.estimate_active_context_tokens("proj")
        assert "total" in tokens
        assert tokens["total"] > 0


class TestContextAssembly:
    def test_basic_assembly(self):
        store.init_store()
        (store.GLOBAL_DIR / "identity.md").write_text("I am a developer.")
        store.create_project("proj")
        pieces = store.assemble_context("proj")
        assert len(pieces) >= 1
        assert any("identity.md" in p["source"] for p in pieces)

    def test_budget_respected(self):
        store.init_store()
        # Create a file with ~100 tokens (400 chars)
        (store.GLOBAL_DIR / "big.md").write_text("x" * 400)
        store.create_project("proj")
        # Tight budget
        pieces = store.assemble_context("proj", token_budget=50)
        # Should still include (truncated) content
        assert len(pieces) >= 1

    def test_assembly_without_project(self):
        store.init_store()
        (store.GLOBAL_DIR / "identity.md").write_text("I am a developer.")
        pieces = store.assemble_context(None)
        assert len(pieces) == 1


class TestActiveFiles:
    def test_list_active_files(self):
        store.init_store()
        store.create_project("proj")
        active = store.list_active_files("proj")
        assert any(f.name == "project.md" for f in active)

    def test_set_active_files(self):
        store.init_store()
        store.create_project("proj")
        # Add another file
        (store.PROJECTS_DIR / "proj" / "arch.md").write_text("# Arch")
        store.set_active_files("proj", ["project.md", "arch.md"])
        active = store.list_active_files("proj")
        names = [f.name for f in active]
        assert "project.md" in names
        assert "arch.md" in names


class TestConfig:
    def test_load_default_config(self):
        # No store initialized yet — should return defaults
        config = store.load_config()
        assert config["llm"]["model"] == "claude-sonnet-4-6"

    def test_save_and_load(self):
        store.init_store()
        config = store.load_config()
        config["active_project"] = "test"
        store.save_config(config)
        loaded = store.load_config()
        assert loaded["active_project"] == "test"


class TestIndex:
    def test_rebuild_index(self):
        store.init_store()
        (store.GLOBAL_DIR / "identity.md").write_text("# Identity\n\nI am a developer.")
        store.create_project("proj")
        index = store.rebuild_index()
        assert len(index["global"]) >= 1
        assert "proj" in index["projects"]

    def test_update_index_entry(self):
        store.init_store()
        (store.GLOBAL_DIR / "test.md").write_text("hello")
        store.update_index_entry("global", None, "test.md")
        index = json.loads(store.INDEX_PATH.read_text())
        assert any(e["file"] == "test.md" for e in index["global"])


class TestConversationHistory:
    def test_save_and_load_project_history(self):
        store.init_store()
        store.create_project("proj")
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        store.save_history("proj", history)
        loaded = store.load_history("proj")
        assert len(loaded) == 2
        assert loaded[0]["content"] == "Hello"
        assert loaded[1]["content"] == "Hi there!"

    def test_save_and_load_global_history(self):
        store.init_store()
        history = [{"role": "user", "content": "Test"}]
        store.save_history(None, history)
        loaded = store.load_history(None)
        assert len(loaded) == 1

    def test_load_empty_history(self):
        store.init_store()
        assert store.load_history("nonexistent") == []

    def test_clear_history(self):
        store.init_store()
        store.create_project("proj")
        store.save_history("proj", [{"role": "user", "content": "Hi"}])
        store.clear_history("proj")
        assert store.load_history("proj") == []

    def test_clear_nonexistent_history(self):
        store.init_store()
        # Should not raise
        store.clear_history("nonexistent")
