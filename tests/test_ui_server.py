"""Tests for knowl.ui.server API endpoints."""

import json
import pytest
from pathlib import Path

from fastapi.testclient import TestClient

from knowl.context import store


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with isolated store."""
    monkeypatch.setenv("KNOWL_DIR", str(tmp_path))
    store.set_knowl_dir(tmp_path)
    store.init_store()
    from knowl.ui.server import create_app
    app = create_app(dev=True)
    yield TestClient(app)


class TestProjectRoutes:
    def test_list_projects_empty(self, client):
        response = client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert isinstance(data["projects"], list)

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
        assert response.json()["deleted"] == "to_delete"

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

    def test_update_config_active_project(self, client):
        client.post("/api/projects", json={"name": "myproj"})
        response = client.put("/api/config", json={"active_project": "myproj"})
        assert response.status_code == 200
        assert response.json()["active_project"] == "myproj"

    def test_update_config_nonexistent_project(self, client):
        response = client.put("/api/config", json={"active_project": "nope"})
        assert response.status_code == 404


class TestContextRoutes:
    def test_list_global_files(self, client):
        response = client.get("/api/context/global")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_and_read_file(self, client):
        client.post("/api/projects", json={"name": "test_proj"})
        response = client.post("/api/context/file", json={
            "filename": "notes.md",
            "scope": "test_proj",
            "content": "# Notes\n\nSome content here",
        })
        assert response.status_code == 200
        path = response.json()["path"]

        response = client.get(f"/api/context/file?path={path}")
        assert response.status_code == 200
        assert "Some content" in response.json()["content"]

    def test_create_duplicate_file(self, client):
        client.post("/api/projects", json={"name": "test_proj"})
        client.post("/api/context/file", json={
            "filename": "dup.md", "scope": "test_proj", "content": "first"
        })
        response = client.post("/api/context/file", json={
            "filename": "dup.md", "scope": "test_proj", "content": "second"
        })
        assert response.status_code == 409

    def test_write_and_read_file(self, client):
        client.post("/api/projects", json={"name": "test_proj"})
        client.post("/api/context/file", json={
            "filename": "edit.md", "scope": "test_proj", "content": "original"
        })
        path = str(store.PROJECTS_DIR / "test_proj" / "edit.md")
        response = client.put("/api/context/file", json={
            "path": path, "content": "updated content"
        })
        assert response.status_code == 200

        response = client.get(f"/api/context/file?path={path}")
        assert response.json()["content"] == "updated content"

    def test_delete_file(self, client):
        client.post("/api/projects", json={"name": "test_proj"})
        resp = client.post("/api/context/file", json={
            "filename": "del.md", "scope": "test_proj", "content": "bye"
        })
        path = resp.json()["path"]
        response = client.delete(f"/api/context/file?path={path}")
        assert response.status_code == 200

    def test_read_nonexistent_file(self, client):
        # Use a path under KNOWL_DIR that doesn't exist
        path = str(store.KNOWL_DIR / "global" / "nonexistent.md")
        response = client.get(f"/api/context/file?path={path}")
        assert response.status_code == 404

    def test_list_project_files(self, client):
        client.post("/api/projects", json={"name": "test_proj"})
        client.post("/api/context/file", json={
            "filename": "a.md", "scope": "test_proj", "content": "aaa"
        })
        response = client.get("/api/context/project/test_proj")
        assert response.status_code == 200
        names = [f["name"] for f in response.json()]
        assert "a.md" in names


class TestBackendStatus:
    def test_get_backend_status(self, client):
        response = client.get("/api/config/backend")
        assert response.status_code == 200
        data = response.json()
        assert "backend" in data
        assert "has_api_key" in data
        assert data["backend"] in ("api", "cli")

    def test_backend_reflects_config(self, client):
        client.put("/api/config", json={"backend": "cli"})
        response = client.get("/api/config/backend")
        assert response.json()["backend"] == "cli"


class TestHistoryRoutes:
    def test_get_empty_history(self, client):
        response = client.get("/api/history")
        assert response.status_code == 200
        assert response.json()["history"] == []

    def test_clear_history(self, client):
        response = client.delete("/api/history")
        assert response.status_code == 200
        assert response.json()["cleared"] is True


class TestInspectRoutes:
    def test_inspect_context(self, client):
        response = client.get("/api/inspect")
        assert response.status_code == 200
        data = response.json()
        assert "system_prompt" in data
        assert "total_tokens" in data

    def test_token_breakdown(self, client):
        client.post("/api/projects", json={"name": "test_proj"})
        response = client.get("/api/tokens/test_proj")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
