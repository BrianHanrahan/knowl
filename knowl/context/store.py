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
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {"model": "claude-sonnet-4-6", "backend": "api"},
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
    """Resolve path and verify it's under KNOWL_DIR (follows symlinks)."""
    resolved = path.resolve()
    # Resolve KNOWL_DIR and GLOBAL_DIR separately — GLOBAL_DIR may be a
    # symlink to an external directory (e.g. ~/Documents/claude-context/context).
    knowl_resolved = KNOWL_DIR.resolve()
    global_resolved = GLOBAL_DIR.resolve()

    under_knowl = (
        str(resolved).startswith(str(knowl_resolved) + os.sep)
        or resolved == knowl_resolved
    )
    under_global = (
        str(resolved).startswith(str(global_resolved) + os.sep)
        or resolved == global_resolved
    )
    if not under_knowl and not under_global:
        raise ValueError(f"Path {path} escapes allowed directory {KNOWL_DIR}")
    return resolved


# ---------------------------------------------------------------------------
# Store initialization
# ---------------------------------------------------------------------------

def _verify_writable(path: Path, label: str) -> None:
    """Verify a directory exists and is writable, raising a clear error if not."""
    resolved = path.resolve()
    if not resolved.exists():
        raise PermissionError(
            f"{label} directory does not exist and could not be created: {resolved}"
        )
    if not os.access(resolved, os.W_OK):
        raise PermissionError(
            f"{label} directory is not writable: {resolved}. "
            f"Fix with: chmod u+w '{resolved}'"
        )


def init_store() -> None:
    """Create the ~/.knowl directory structure if it doesn't exist."""
    # Create directories first.
    try:
        KNOWL_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PermissionError(
            f"Cannot create knowl directory {KNOWL_DIR}: {exc}"
        ) from exc

    try:
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PermissionError(
            f"Cannot create projects directory {PROJECTS_DIR}: {exc}"
        ) from exc

    # Global dir may be a symlink to an external directory — create the
    # target only if it's a real (non-symlink) path.  If it's a symlink,
    # verify the target exists and is writable.
    if GLOBAL_DIR.is_symlink():
        target = GLOBAL_DIR.resolve()
        if not target.exists():
            raise PermissionError(
                f"Global context symlink {GLOBAL_DIR} points to "
                f"{target} which does not exist."
            )
        _verify_writable(target, "Global context (symlink target)")
    else:
        try:
            GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PermissionError(
                f"Cannot create global directory {GLOBAL_DIR}: {exc}"
            ) from exc

    # Verify all key directories are writable.
    _verify_writable(KNOWL_DIR, "Knowl root")
    _verify_writable(PROJECTS_DIR, "Projects")

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
    if not os.access(PROJECTS_DIR, os.W_OK):
        raise PermissionError(
            f"Projects directory is not writable: {PROJECTS_DIR}. "
            f"Fix with: chmod u+w '{PROJECTS_DIR}'"
        )
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
    # Check parent directory is writable (or can be created).
    parent = path.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        raise PermissionError(
            f"Directory is not writable: {parent}. "
            f"Fix with: chmod u+w '{parent}'"
        )
    try:
        parent.mkdir(parents=True, exist_ok=True)
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
    """Rename a context file and update active_files / index accordingly."""
    old_path = Path(old_path)
    new_path = Path(new_path)
    _validate_path(old_path)
    _validate_path(new_path)
    if not old_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {old_path}")
    old_path.rename(new_path)

    # Update active_files in context.json if this is a project file
    old_name = old_path.name
    new_name = new_path.name
    parent = new_path.parent
    context_json = parent / "context.json"
    if context_json.exists():
        try:
            data = json.loads(context_json.read_text(encoding="utf-8"))
            active = data.get("active_files", [])
            if old_name in active:
                data["active_files"] = [new_name if f == old_name else f for f in active]
                context_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to update context.json after rename: %s", exc)

    # Update index
    if parent == GLOBAL_DIR:
        update_index_entry("global", None, new_name)
    else:
        project_name = parent.name
        update_index_entry("project", project_name, new_name)

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


def move_context_file(
    filename: str,
    from_scope: str,
    to_scope: str,
    from_project: str | None = None,
    to_project: str | None = None,
) -> Path:
    """Move a context file between scopes (global ↔ project, or project ↔ project).

    Args:
        filename: The file name (e.g. 'notes.md').
        from_scope: 'global' or 'project'.
        to_scope: 'global' or 'project'.
        from_project: Source project name (required when from_scope='project').
        to_project: Destination project name (required when to_scope='project').

    Returns:
        The destination Path.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FileExistsError: If a file with the same name already exists at dest.
        ValueError: If scope/project args are invalid.
    """
    # Resolve source path
    if from_scope == "global":
        source = GLOBAL_DIR / filename
    elif from_scope == "project":
        if not from_project:
            raise ValueError("from_project is required when from_scope='project'")
        source = PROJECTS_DIR / from_project / filename
    else:
        raise ValueError(f"Invalid from_scope: {from_scope}")

    # Resolve destination path
    if to_scope == "global":
        dest = GLOBAL_DIR / filename
    elif to_scope == "project":
        if not to_project:
            raise ValueError("to_project is required when to_scope='project'")
        dest_dir = PROJECTS_DIR / to_project
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
    else:
        raise ValueError(f"Invalid to_scope: {to_scope}")

    _validate_path(source)
    _validate_path(dest)

    if not source.exists():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    if dest.exists():
        raise FileExistsError(f"Destination file already exists: {dest}")

    # Read, write to dest, delete source (not rename — may cross filesystems)
    content = source.read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    source.unlink()

    # Update index for both source and destination
    if from_scope == "global":
        update_index_entry("global", None, filename)
    else:
        update_index_entry("project", from_project, filename)

    if to_scope == "global":
        update_index_entry("global", None, filename)
    else:
        update_index_entry("project", to_project, filename)

    logger.info("Moved %s from %s/%s to %s/%s",
                filename,
                from_scope, from_project or "",
                to_scope, to_project or "")
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

def _build_context_catalog(project_name: str | None) -> str:
    """Build a compact catalog of available context files from index.json.

    Returns a human-readable summary the LLM can use to decide which files
    to load via `read_context_file`, without needing an extra tool call.
    """
    try:
        if not INDEX_PATH.exists():
            return ""
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    # Check if there's anything to list
    global_entries = index.get("global", [])
    project_entries = (
        index.get("projects", {}).get(project_name, []) if project_name else []
    )
    all_projects = list(index.get("projects", {}).keys())
    other_projects = [p for p in all_projects if p != project_name]
    if not global_entries and not project_entries and not other_projects:
        return ""

    lines = ["# Available Context Files", ""]
    lines.append("Use `read_context_file` to load any of these. "
                 "Do NOT reference these paths directly — use the context tools.")
    lines.append("")

    # Global files
    if global_entries:
        lines.append("## Global")
        for entry in global_entries:
            name = entry.get("file", "")
            tokens = entry.get("tokens", 0)
            summary = entry.get("summary", "")
            # Trim summary to first ~80 chars for compactness
            if len(summary) > 80:
                summary = summary[:77] + "..."
            lines.append(f"- **{name}** ({tokens} tok): {summary}")
        lines.append("")

    # Current project files
    if project_entries:
        lines.append(f"## Project: {project_name}")
        for entry in project_entries:
            name = entry.get("file", "")
            tokens = entry.get("tokens", 0)
            summary = entry.get("summary", "")
            if len(summary) > 80:
                summary = summary[:77] + "..."
            lines.append(f"- **{name}** ({tokens} tok): {summary}")
        lines.append("")

    # Other projects (just names, for awareness)
    if other_projects:
        lines.append(f"## Other projects: {', '.join(other_projects)}")
        lines.append("")

    return "\n".join(lines)


def assemble_context(
    project_name: str | None,
    token_budget: int = 8000,
) -> list[dict[str, str]]:
    """Load context catalog + active project files within token budget.

    Returns a list of dicts with keys: role, content, source.
    The catalog is a compact summary built from index.json — the LLM
    uses it to decide which files to load on demand via tools.
    """
    pieces: list[dict[str, str]] = []
    tokens_used = 0

    # 1. Compact catalog from index.json (replaces raw INDEX.md loading)
    catalog = _build_context_catalog(project_name)
    if catalog:
        tokens = estimate_tokens(catalog)
        if tokens <= token_budget:
            pieces.append({
                "role": "system",
                "content": catalog,
                "source": "context-catalog",
                "tokens": tokens,
            })
            tokens_used += tokens
        else:
            logger.warning("Context catalog exceeds budget — truncating")
            chars = (token_budget - tokens_used) * 4
            if chars > 100:
                catalog = catalog[:chars] + "\n\n[truncated]"
                tokens = estimate_tokens(catalog)
                pieces.append({
                    "role": "system",
                    "content": catalog,
                    "source": "context-catalog",
                    "tokens": tokens,
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
                "tokens": tokens,
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


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

def _history_path(project_name: str | None) -> Path:
    """Return the path for a conversation history file."""
    if project_name:
        return PROJECTS_DIR / project_name / "history.json"
    return KNOWL_DIR / "global_history.json"


def load_history(project_name: str | None) -> list[dict[str, str]]:
    """Load conversation history for a project (or global if None)."""
    path = _history_path(project_name)
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read history: %s", exc)
    return []


def save_history(project_name: str | None, history: list[dict[str, str]]) -> None:
    """Save conversation history for a project (or global if None)."""
    path = _history_path(project_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to save history: %s", exc)


def clear_history(project_name: str | None) -> None:
    """Clear conversation history for a project (or global if None)."""
    path = _history_path(project_name)
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            logger.error("Failed to clear history: %s", exc)


# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------

def _tools_path(project_name: str) -> Path:
    """Return the path for a project's tools.json."""
    return PROJECTS_DIR / project_name / "tools.json"


def load_tools(project_name: str) -> dict[str, Any]:
    """Load all custom tools for a project."""
    path = _tools_path(project_name)
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "tools" in data:
                return data["tools"]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read tools for %s: %s", project_name, exc)
    return {}


def save_tools(project_name: str, tools: dict[str, Any]) -> None:
    """Write all custom tools for a project."""
    path = _tools_path(project_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tools": tools}, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to save tools for %s: %s", project_name, exc)


def get_tool(project_name: str, name: str) -> dict[str, Any] | None:
    """Get a single custom tool by name."""
    tools = load_tools(project_name)
    return tools.get(name)


def save_tool(project_name: str, tool: dict[str, Any]) -> None:
    """Upsert a single custom tool."""
    tools = load_tools(project_name)
    tools[tool["name"]] = tool
    save_tools(project_name, tools)


def delete_tool(project_name: str, name: str) -> bool:
    """Delete a custom tool by name. Returns True if it existed."""
    tools = load_tools(project_name)
    if name not in tools:
        return False
    del tools[name]
    save_tools(project_name, tools)
    return True


# ---------------------------------------------------------------------------
# File uploads
# ---------------------------------------------------------------------------

def uploads_dir(project_name: str) -> Path:
    """Return the uploads directory for a project, creating it if needed."""
    d = PROJECTS_DIR / project_name / "uploads"
    _validate_path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(project_name: str, filename: str, data: bytes) -> Path:
    """Save an uploaded file to the project's uploads directory."""
    if len(data) > MAX_UPLOAD_SIZE:
        raise ValueError(
            f"File size ({len(data)} bytes) exceeds limit ({MAX_UPLOAD_SIZE} bytes)."
        )
    dest = uploads_dir(project_name) / filename
    _validate_path(dest)
    dest.write_bytes(data)
    logger.info("Saved upload %s/%s (%d bytes)", project_name, filename, len(data))
    return dest


def list_uploads(project_name: str) -> list[dict[str, Any]]:
    """List uploaded files for a project."""
    d = PROJECTS_DIR / project_name / "uploads"
    _validate_path(d)
    if not d.exists():
        return []
    result = []
    for f in sorted(d.iterdir()):
        if f.is_file():
            result.append({
                "name": f.name,
                "size": f.stat().st_size,
                "path": str(f),
            })
    return result


def delete_upload(project_name: str, filename: str) -> bool:
    """Delete an uploaded file. Returns True if it existed."""
    path = PROJECTS_DIR / project_name / "uploads" / filename
    _validate_path(path)
    if not path.exists():
        return False
    path.unlink()
    logger.info("Deleted upload %s/%s", project_name, filename)
    return True


def list_approved_tools(project_name: str) -> list[dict[str, Any]]:
    """Return API-format tool dicts for approved (enabled) tools only."""
    tools = load_tools(project_name)
    result = []
    for tool in tools.values():
        if tool.get("status") == "approved":
            result.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }),
            })
    return result
