"""
Context store — manages the ~/.knowl/ filesystem layout.

Handles reading, writing, indexing, and organizing context files
at both global and project scope.
"""

import json
from pathlib import Path

KNOWL_DIR = Path.home() / ".knowl"
GLOBAL_DIR = KNOWL_DIR / "global"
PROJECTS_DIR = KNOWL_DIR / "projects"
INDEX_PATH = KNOWL_DIR / "index.json"
CONFIG_PATH = KNOWL_DIR / "config.json"


def init_store():
    """Create the ~/.knowl directory structure if it doesn't exist."""
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text(json.dumps({"global": [], "projects": {}}, indent=2))
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps({
            "llm": {
                "provider": "",
                "model": "",
                "api_key_env": "",
                "base_url": ""
            },
            "voice": {
                "whisper_model": "base",
                "language": "auto"
            }
        }, indent=2))


def list_global_files():
    """Return all markdown files in the global context directory."""
    if not GLOBAL_DIR.exists():
        return []
    return sorted(GLOBAL_DIR.glob("*.md"))


def list_projects():
    """Return all project directory names."""
    if not PROJECTS_DIR.exists():
        return []
    return sorted([p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()])


def list_project_files(project_name):
    """Return all markdown files in a project's context directory."""
    project_dir = PROJECTS_DIR / project_name
    if not project_dir.exists():
        return []
    return sorted(project_dir.glob("*.md"))


def create_project(name):
    """Create a new project directory with a default project.md."""
    project_dir = PROJECTS_DIR / name
    project_dir.mkdir(parents=True, exist_ok=True)
    project_md = project_dir / "project.md"
    if not project_md.exists():
        project_md.write_text(f"# {name}\n\nProject context goes here.\n")
    context_json = project_dir / "context.json"
    if not context_json.exists():
        context_json.write_text(json.dumps({
            "active_files": ["project.md"],
            "last_used": None
        }, indent=2))
    return project_dir


def read_context_file(path):
    """Read a context file and return its contents."""
    path = Path(path)
    if path.exists():
        return path.read_text()
    return None


def write_context_file(path, content):
    """Write content to a context file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def promote_to_global(project_name, filename):
    """Copy a project context file to the global directory."""
    source = PROJECTS_DIR / project_name / filename
    dest = GLOBAL_DIR / filename
    if source.exists():
        content = source.read_text()
        dest.write_text(content)
        return dest
    return None
