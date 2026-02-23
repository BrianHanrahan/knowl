"""
Context store — manages the ~/.knowl/ filesystem layout.

Handles reading, writing, indexing, and organizing context files
at both global and project scope.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from knowl.log import get_logger

logger = get_logger(__name__)

# Allow overriding via env var for testing.
KNOWL_DIR = Path(os.environ.get("KNOWL_DIR", str(Path.home() / ".knowl")))
GLOBAL_DIR = KNOWL_DIR / "global"
PROJECTS_DIR = KNOWL_DIR / "projects"
INDEX_PATH = KNOWL_DIR / "index.json"
CONFIG_PATH = KNOWL_DIR / "config.json"

MAX_CONTEXT_FILE_SIZE = 100 * 1024  # 100 KB

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {"model": "claude-sonnet-4-6"},
    "voice": {"whisper_model": "base", "language": "auto"},
    "active_project": None,
}


def _refresh_dirs() -> None:
    """Recompute directory paths from KNOWL_DIR (call after changing KNOWL_DIR)."""
    global GLOBAL_DIR, PROJECTS_DIR, INDEX_PATH, CONFIG_PATH
    GLOBAL_DIR = KNOWL_DIR / "global"
    PROJECTS_DIR = KNOWL_DIR / "projects"
    INDEX_PATH = KNOWL_DIR / "index.json"
    CONFIG_PATH = KNOWL_DIR / "config.json"


def set_knowl_dir(path: Path) -> None:
    """Override the KNOWL_DIR for testing."""
    global KNOWL_DIR
    KNOWL_DIR = path
    _refresh_dirs()


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def _validate_path(path: Path) -> Path:
    """Resolve path and verify it's under KNOWL_DIR."""
    resolved = path.resolve()
    knowl_resolved = KNOWL_DIR.resolve()
    if not str(resolved).startswith(str(knowl_resolved) + os.sep) and resolved != knowl_resolved:
        raise ValueError(f"Path {path} escapes allowed directory {KNOWL_DIR}")
    return resolved


# ---------------------------------------------------------------------------
# Store initialization
# ---------------------------------------------------------------------------

def init_store() -> None:
    """Create the ~/.knowl directory structure if it doesn't exist."""
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text(json.dumps({"global": [], "projects": {}}, indent=2))
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG.copy())
    rebuild_index()
    logger.info("Store initialized at %s", KNOWL_DIR)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    """Read ~/.knowl/config.json. Returns defaults if missing or corrupt."""
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read config: %s — using defaults", exc)
    return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any]) -> None:
    """Write config atomically (write to .tmp, then rename)."""
    KNOWL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CONFIG_PATH.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        tmp_path.replace(CONFIG_PATH)
    except OSError as exc:
        logger.error("Failed to save config: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_global_files() -> list[Path]:
    """Return all markdown files in the global context directory."""
    if not GLOBAL_DIR.exists():
        return []
    return sorted(GLOBAL_DIR.glob("*.md"))


def list_projects() -> list[str]:
    """Return all project directory names."""
    if not PROJECTS_DIR.exists():
        return []
    return sorted([p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()])


def list_project_files(project_name: str) -> list[Path]:
    """Return all markdown files in a project's context directory."""
    project_dir = PROJECTS_DIR / project_name
    _validate_path(project_dir)
    if not project_dir.exists():
        return []
    return sorted(project_dir.glob("*.md"))


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def create_project(name: str) -> Path:
    """Create a new project directory with a default project.md."""
    project_dir = PROJECTS_DIR / name
    _validate_path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    project_md = project_dir / "project.md"
    if not project_md.exists():
        project_md.write_text(f"# {name}\n\nProject context goes here.\n")
    context_json = project_dir / "context.json"
    if not context_json.exists():
        context_json.write_text(json.dumps({
            "active_files": ["project.md"],
            "token_budget": 8000,
            "last_used": None,
        }, indent=2))
    update_index_entry("project", name, "project.md")
    logger.info("Created project: %s", name)
    return project_dir


def delete_project(name: str) -> bool:
    """Delete a project directory and remove it from the index."""
    project_dir = PROJECTS_DIR / name
    _validate_path(project_dir)
    if not project_dir.exists():
        return False
    shutil.rmtree(project_dir)
    # Remove from index
    try:
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else {}
        index.get("projects", {}).pop(name, None)
        INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to update index after project deletion: %s", exc)
    logger.info("Deleted project: %s", name)
    return True


def read_context_file(path: str | Path) -> str | None:
    """Read a context file and return its contents."""
    path = Path(path)
    _validate_path(path)
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to read %s: %s", path, exc)
    return None


def write_context_file(path: str | Path, content: str) -> None:
    """Write content to a context file."""
    path = Path(path)
    _validate_path(path)
    size = len(content.encode("utf-8"))
    if size > MAX_CONTEXT_FILE_SIZE:
        raise ValueError(
            f"Content size ({size} bytes) exceeds limit ({MAX_CONTEXT_FILE_SIZE} bytes). "
            "Split the file or increase the limit."
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to write %s: %s", path, exc)
        raise


def delete_context_file(path: str | Path) -> bool:
    """Delete a context file."""
    path = Path(path)
    _validate_path(path)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.error("Failed to delete %s: %s", path, exc)
        return False


def rename_context_file(old_path: str | Path, new_path: str | Path) -> Path:
    """Rename a context file."""
    old_path = Path(old_path)
    new_path = Path(new_path)
    _validate_path(old_path)
    _validate_path(new_path)
    if not old_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {old_path}")
    old_path.rename(new_path)
    return new_path


def promote_to_global(project_name: str, filename: str) -> Path | None:
    """Copy a project context file to the global directory."""
    source = PROJECTS_DIR / project_name / filename
    dest = GLOBAL_DIR / filename
    _validate_path(source)
    _validate_path(dest)
    if not source.exists():
        return None
    content = source.read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    update_index_entry("global", None, filename)
    logger.info("Promoted %s/%s to global", project_name, filename)
    return dest


# ---------------------------------------------------------------------------
# Active files
# ---------------------------------------------------------------------------

def list_active_files(project_name: str) -> list[Path]:
    """Return the active file paths for a project."""
    project_dir = PROJECTS_DIR / project_name
    _validate_path(project_dir)
    context_json = project_dir / "context.json"
    if not context_json.exists():
        return []
    try:
        data = json.loads(context_json.read_text(encoding="utf-8"))
        filenames = data.get("active_files", [])
        return [project_dir / f for f in filenames if (project_dir / f).exists()]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read context.json for %s: %s", project_name, exc)
        return []


def set_active_files(project_name: str, files: list[str]) -> None:
    """Update the active files list for a project."""
    project_dir = PROJECTS_DIR / project_name
    _validate_path(project_dir)
    context_json = project_dir / "context.json"
    try:
        data = json.loads(context_json.read_text(encoding="utf-8")) if context_json.exists() else {}
    except (json.JSONDecodeError, OSError):
        data = {}
    data["active_files"] = files
    context_json.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count. Uses ~4 chars per token as a conservative heuristic."""
    return max(1, len(text) // 4)


def estimate_file_tokens(path: str | Path) -> int:
    """Estimate token count for a file."""
    path = Path(path)
    _validate_path(path)
    if not path.exists():
        return 0
    try:
        content = path.read_text(encoding="utf-8")
        return estimate_tokens(content)
    except OSError:
        return 0


def estimate_active_context_tokens(project_name: str) -> dict[str, int]:
    """Return token counts for each active file and the total."""
    result: dict[str, int] = {}
    total = 0
    # Global files
    for f in list_global_files():
        tokens = estimate_file_tokens(f)
        result[f"global/{f.name}"] = tokens
        total += tokens
    # Project files
    for f in list_active_files(project_name):
        tokens = estimate_file_tokens(f)
        result[f"project/{f.name}"] = tokens
        total += tokens
    result["total"] = total
    return result


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def assemble_context(
    project_name: str | None,
    token_budget: int = 8000,
) -> list[dict[str, str]]:
    """Load global + active project files within token budget.

    Returns a list of dicts with keys: role, content, source.
    """
    pieces: list[dict[str, str]] = []
    tokens_used = 0

    # 1. Global files (mandatory)
    for f in list_global_files():
        try:
            content = f.read_text(encoding="utf-8")
        except OSError:
            continue
        tokens = estimate_tokens(content)
        if tokens_used + tokens > token_budget:
            # Truncate to fit
            chars_remaining = (token_budget - tokens_used) * 4
            if chars_remaining > 100:
                content = content[:chars_remaining] + f"\n\n[truncated — full file: {f}]"
                tokens = estimate_tokens(content)
            else:
                logger.warning("Skipping global file %s — budget exhausted", f.name)
                continue
        pieces.append({
            "role": "system",
            "content": content,
            "source": f"global/{f.name}",
        })
        tokens_used += tokens

    # 2. Project files (in active_files order)
    if project_name:
        for f in list_active_files(project_name):
            try:
                content = f.read_text(encoding="utf-8")
            except OSError:
                continue
            tokens = estimate_tokens(content)
            if tokens_used + tokens > token_budget:
                chars_remaining = (token_budget - tokens_used) * 4
                if chars_remaining > 100:
                    content = content[:chars_remaining] + f"\n\n[truncated — full file: {f}]"
                    tokens = estimate_tokens(content)
                else:
                    logger.warning("Skipping project file %s — budget exhausted", f.name)
                    continue
            pieces.append({
                "role": "system",
                "content": content,
                "source": f"project/{f.name}",
            })
            tokens_used += tokens

    return pieces


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def rebuild_index() -> dict[str, Any]:
    """Scan all files and rebuild index.json."""
    index: dict[str, Any] = {"global": [], "projects": {}}

    # Global files
    for f in list_global_files():
        try:
            content = f.read_text(encoding="utf-8")
            tokens = estimate_tokens(content)
            summary = content[:100].replace("\n", " ").strip()
            index["global"].append({
                "file": f.name,
                "tokens": tokens,
                "summary": summary,
            })
        except OSError:
            continue

    # Project files
    for project_name in list_projects():
        project_files = []
        for f in list_project_files(project_name):
            try:
                content = f.read_text(encoding="utf-8")
                tokens = estimate_tokens(content)
                summary = content[:100].replace("\n", " ").strip()
                project_files.append({
                    "file": f.name,
                    "tokens": tokens,
                    "summary": summary,
                })
            except OSError:
                continue
        index["projects"][project_name] = project_files

    try:
        INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to write index: %s", exc)

    return index


def update_index_entry(scope: str, project: str | None, filename: str) -> None:
    """Update a single entry in the index after file create/edit/delete."""
    try:
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else {"global": [], "projects": {}}
    except (json.JSONDecodeError, OSError):
        index = {"global": [], "projects": {}}

    if scope == "global":
        path = GLOBAL_DIR / filename
        # Remove existing entry
        index["global"] = [e for e in index.get("global", []) if e.get("file") != filename]
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                index["global"].append({
                    "file": filename,
                    "tokens": estimate_tokens(content),
                    "summary": content[:100].replace("\n", " ").strip(),
                })
            except OSError:
                pass
    elif scope == "project" and project:
        path = PROJECTS_DIR / project / filename
        project_entries = index.get("projects", {}).get(project, [])
        project_entries = [e for e in project_entries if e.get("file") != filename]
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                project_entries.append({
                    "file": filename,
                    "tokens": estimate_tokens(content),
                    "summary": content[:100].replace("\n", " ").strip(),
                })
            except OSError:
                pass
        if "projects" not in index:
            index["projects"] = {}
        index["projects"][project] = project_entries

    try:
        INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to update index: %s", exc)
