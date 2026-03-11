"""Knowl Web UI — FastAPI backend serving REST API + React frontend."""


import asyncio
import functools
import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
import base64
import mimetypes

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from knowl.context import store
from knowl.llm import claude
from knowl.llm.claude import TextChunkEvent, ToolCallEvent, DoneEvent


# ── Pydantic request models (must be at module level for FastAPI) ────

class CreateProject(BaseModel):
    name: str

class UpdateConfig(BaseModel):
    active_project: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None  # "api" or "cli"

class WriteFile(BaseModel):
    path: str
    content: str

class CreateFile(BaseModel):
    filename: str
    scope: str  # "global" or project name
    content: str = ""

class SetActiveFiles(BaseModel):
    files: list[str]

class PromoteFile(BaseModel):
    project: str
    filename: str

class RenameFile(BaseModel):
    path: str
    new_name: str

class FormatFile(BaseModel):
    content: str
    model: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    project: Optional[str] = None
    model: Optional[str] = None

class ApplyPromotion(BaseModel):
    filename: str
    source_project: Optional[str] = None
    force: bool = False
    cleanup: bool = False

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


def create_app(dev: bool = False) -> FastAPI:
    """Create and configure the FastAPI application."""
    load_dotenv()
    app = FastAPI(title="Knowl", version="0.1.0")

    if dev:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Re-read KNOWL_DIR from env (dotenv may have set it after store import)
    knowl_dir_env = os.environ.get("KNOWL_DIR")
    if knowl_dir_env:
        store.KNOWL_DIR = Path(knowl_dir_env)
        store._refresh_dirs()

    # Ensure store is initialized
    store.init_store()

    # ── Browser lifecycle ────────────────────────────────────────────
    @app.on_event("shutdown")
    async def shutdown_browser():
        from knowl.web.browser import BrowserPool
        await BrowserPool.get().shutdown()

    # ── Register routes ──────────────────────────────────────────────
    _register_project_routes(app)
    _register_config_routes(app)
    _register_context_routes(app)
    _register_inspect_routes(app)
    _register_chat_routes(app)
    _register_tool_routes(app)
    _register_upload_routes(app)
    _register_promotion_routes(app)

    # Serve React build in production
    if not dev and FRONTEND_DIST.is_dir():
        # Catch-all: serve index.html for client-side routing
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file_path = FRONTEND_DIST / full_path
            if full_path and file_path.is_file():
                from fastapi.responses import FileResponse
                return FileResponse(file_path)
            return _serve_index()

    return app


def _serve_index():
    from fastapi.responses import FileResponse
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend not built. Run: cd knowl/ui/frontend && npm run build")


# ── Project routes ───────────────────────────────────────────────────

def _register_project_routes(app: FastAPI) -> None:

    @app.get("/api/projects")
    async def list_projects():
        projects = store.list_projects()
        config = store.load_config()
        active = config.get("active_project")
        return {"projects": projects, "active_project": active}

    @app.post("/api/projects")
    async def create_project(payload: CreateProject):
        path = store.create_project(payload.name)
        return {"name": payload.name, "path": str(path)}

    @app.delete("/api/projects/{name}")
    async def delete_project(name: str):
        ok = store.delete_project(name)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Project '{name}' not found")
        return {"deleted": name}


# ── Config routes ────────────────────────────────────────────────────

def _register_config_routes(app: FastAPI) -> None:

    @app.get("/api/config")
    async def get_config():
        return store.load_config()

    @app.put("/api/config")
    async def update_config(payload: UpdateConfig):
        config = store.load_config()
        if payload.active_project is not None:
            # Validate project exists
            if payload.active_project and payload.active_project not in store.list_projects():
                raise HTTPException(status_code=404, detail=f"Project '{payload.active_project}' not found")
            config["active_project"] = payload.active_project or None
        if payload.model is not None:
            config.setdefault("llm", {})["model"] = payload.model
        if payload.backend is not None:
            if payload.backend not in ("api", "cli"):
                raise HTTPException(status_code=400, detail="backend must be 'api' or 'cli'")
            config.setdefault("llm", {})["backend"] = payload.backend
        store.save_config(config)
        return config

    @app.get("/api/config/backend")
    async def get_backend():
        config = store.load_config()
        backend = config.get("llm", {}).get("backend", "api")
        has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
        return {"backend": backend, "has_api_key": has_api_key}


# ── Backend routing ───────────────────────────────────────────────────

def _get_stream_fn(config: dict):
    """Return the appropriate stream_message_with_tools based on backend config."""
    backend = config.get("llm", {}).get("backend", "api")
    if backend == "cli":
        from knowl.llm.cli_backend import stream_message_with_tools
        return stream_message_with_tools
    else:
        from knowl.llm.claude import stream_message_with_tools
        return stream_message_with_tools


# ── Context file routes ──────────────────────────────────────────────

def _register_context_routes(app: FastAPI) -> None:

    @app.get("/api/context/global")
    async def list_global_files():
        files = store.list_global_files()
        return [
            {"name": f.name, "tokens": store.estimate_file_tokens(f), "path": str(f)}
            for f in files
        ]

    @app.get("/api/context/project/{name}")
    async def list_project_files(name: str):
        files = store.list_project_files(name)
        active_paths = {str(f) for f in store.list_active_files(name)}
        return [
            {
                "name": f.name,
                "tokens": store.estimate_file_tokens(f),
                "path": str(f),
                "active": str(f) in active_paths,
            }
            for f in files
        ]

    @app.get("/api/context/file")
    async def read_file(path: str = Query(...)):
        content = store.read_context_file(path)
        if content is None:
            raise HTTPException(status_code=404, detail="File not found")
        return {"path": path, "content": content}

    @app.put("/api/context/file")
    async def write_file(payload: WriteFile):
        try:
            store.write_context_file(payload.path, payload.content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"path": payload.path, "tokens": store.estimate_tokens(payload.content)}

    @app.post("/api/context/file")
    async def create_file(payload: CreateFile):
        if payload.scope == "global":
            target = store.GLOBAL_DIR / payload.filename
        else:
            target = store.PROJECTS_DIR / payload.scope / payload.filename
        if target.exists():
            raise HTTPException(status_code=409, detail="File already exists")
        content = payload.content or f"# {payload.filename}\n\n"
        try:
            store.write_context_file(target, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"path": str(target), "tokens": store.estimate_tokens(content)}

    @app.delete("/api/context/file")
    async def delete_file(path: str = Query(...)):
        ok = store.delete_context_file(path)
        if not ok:
            raise HTTPException(status_code=404, detail="File not found")
        return {"deleted": path}

    @app.put("/api/context/active/{project}")
    async def set_active_files(project: str, payload: SetActiveFiles):
        store.set_active_files(project, payload.files)
        return {"project": project, "active_files": payload.files}

    @app.post("/api/context/rename")
    async def rename_file(payload: RenameFile):
        old = Path(payload.path)
        new_path = old.parent / payload.new_name
        try:
            result = store.rename_context_file(old, new_path)
            return {"old_path": payload.path, "new_path": str(result)}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/format-file")
    async def format_file(payload: FormatFile):
        config = store.load_config()
        model = payload.model or config.get("llm", {}).get("model", "claude-sonnet-4-6")
        backend = config.get("llm", {}).get("backend", "api")
        prompt = (
            "Clean up the following markdown content. Fix markdown formatting "
            "(especially tables — ensure proper | pipe | syntax), remove duplicate "
            "content, correct obvious typos and errors. Preserve ALL information — "
            "do not remove content, just clean it up. Return ONLY the cleaned "
            "markdown, no commentary or explanation.\n\n"
            + payload.content
        )
        try:
            if backend == "cli":
                from knowl.llm.cli_backend import format_with_cli
                result = await format_with_cli(prompt, model)
            else:
                result = await asyncio.to_thread(
                    claude.send_message,
                    user_message=prompt,
                    model=model,
                )
            return {"content": result}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/context/promote")
    async def promote_file(payload: PromoteFile):
        result = store.promote_to_global(payload.project, payload.filename)
        if not result:
            raise HTTPException(status_code=404, detail="File not found in project")
        return {"promoted": payload.filename, "path": str(result)}


# ── Inspect routes ───────────────────────────────────────────────────

def _register_inspect_routes(app: FastAPI) -> None:

    @app.get("/api/inspect")
    async def inspect_context(
        project: Optional[str] = Query(None),
        budget: int = Query(8000),
    ):
        if project is None:
            config = store.load_config()
            project = config.get("active_project")
        pieces = store.assemble_context(project, token_budget=budget)
        system_prompt = claude.format_system_prompt(pieces)
        files = [
            {"source": p["source"], "tokens": store.estimate_tokens(p.get("content", ""))}
            for p in pieces
        ]
        total_tokens = sum(f["tokens"] for f in files)
        return {
            "files": files,
            "total_tokens": total_tokens,
            "budget": budget,
            "system_prompt": system_prompt,
        }

    @app.get("/api/tokens/{project}")
    async def token_breakdown(project: str):
        tokens = store.estimate_active_context_tokens(project)
        return tokens


# ── Chat routes ──────────────────────────────────────────────────────

def _register_chat_routes(app: FastAPI) -> None:

    @app.get("/api/history")
    async def get_history(project: Optional[str] = Query(None)):
        if project is None:
            config = store.load_config()
            project = config.get("active_project")
        history = store.load_history(project)
        return {"project": project, "history": history}

    @app.delete("/api/history")
    async def clear_history(project: Optional[str] = Query(None)):
        if project is None:
            config = store.load_config()
            project = config.get("active_project")
        store.clear_history(project)
        return {"cleared": True, "project": project}

    async def execute_tool(name: str, input_data: dict, project: Optional[str] = None) -> str:
        """Dispatch a tool call to the appropriate implementation."""
        # Web tools
        if name == "web_search":
            from knowl.web.search import web_search
            results = await web_search(
                query=input_data["query"],
                num_results=input_data.get("num_results", 5),
            )
            return json.dumps(results)
        elif name == "fetch_page":
            from knowl.web.fetch import fetch_page
            result = await fetch_page(url=input_data["url"])
            return json.dumps(result)

        # Context tools
        elif name == "list_context_files":
            project_name = input_data.get("project")
            global_files = [
                {"name": f.name, "tokens": store.estimate_file_tokens(f), "path": str(f), "scope": "global"}
                for f in store.list_global_files()
            ]
            project_files = []
            if project_name:
                project_files = [
                    {"name": f.name, "tokens": store.estimate_file_tokens(f), "path": str(f), "scope": project_name}
                    for f in store.list_project_files(project_name)
                ]
            projects = store.list_projects()
            return json.dumps({"global_files": global_files, "project_files": project_files, "projects": projects})

        elif name == "read_context_file":
            rpath = input_data["path"]
            if not Path(rpath).is_absolute():
                rpath = str(store.PROJECTS_DIR / rpath)
            content = store.read_context_file(rpath)
            if content is None:
                return json.dumps({"error": f"File not found: {rpath}"})
            return json.dumps({"path": rpath, "content": content})

        elif name == "write_context_file":
            content = input_data["content"]
            path = input_data.get("path")
            if not path:
                # Creating a new file
                filename = input_data.get("filename")
                scope = input_data.get("scope", "global")
                if not filename:
                    return json.dumps({"error": "Must provide either 'path' (to update) or 'filename' + 'scope' (to create)."})
                if scope == "global":
                    target = store.GLOBAL_DIR / filename
                else:
                    target = store.PROJECTS_DIR / scope / filename
                path = str(target)
            elif not Path(path).is_absolute():
                # Resolve relative paths against the knowl projects directory
                target = store.PROJECTS_DIR / path
                path = str(target)
            try:
                store.write_context_file(path, content)
                return json.dumps({"success": True, "path": path, "tokens": store.estimate_tokens(content)})
            except ValueError as exc:
                return json.dumps({"error": str(exc)})

        elif name == "delete_context_file":
            dpath = input_data["path"]
            if not Path(dpath).is_absolute():
                dpath = str(store.PROJECTS_DIR / dpath)
            ok = store.delete_context_file(dpath)
            if not ok:
                return json.dumps({"error": f"File not found: {dpath}"})
            return json.dumps({"success": True, "deleted": dpath})

        # Meta-tool: create a new custom tool
        elif name == "create_tool":
            if not project:
                return json.dumps({"error": "Cannot create tools without an active project."})
            from datetime import datetime, timezone
            tool_data = {
                "name": input_data["name"],
                "description": input_data["description"],
                "input_schema": input_data["input_schema"],
                "implementation": input_data["implementation"],
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": 1,
            }
            store.save_tool(project, tool_data)
            return json.dumps({
                "success": True,
                "message": (
                    f"Tool '{input_data['name']}' has been proposed and is awaiting user approval. "
                    f"IMPORTANT: You CANNOT call this tool yet. The user must approve it first via the UI. "
                    f"Do NOT attempt to call '{input_data['name']}' — it will fail until approved. "
                    f"Tell the user the tool has been proposed and they can approve it in the sidebar or inline."
                ),
                "status": "pending",
            })

        else:
            # Check if this is a custom tool
            if project:
                tool = store.get_tool(project, name)
                if tool:
                    status = tool.get("status")
                    if status == "approved":
                        from knowl.tools.executor import execute_custom_tool, ToolExecutionError
                        try:
                            result = await asyncio.to_thread(
                                execute_custom_tool, project, name, input_data
                            )
                            return result
                        except ToolExecutionError as exc:
                            return json.dumps({"error": str(exc)})
                    elif status == "pending":
                        return json.dumps({
                            "error": f"Tool '{name}' exists but is pending user approval. "
                            f"Do NOT retry — wait for the user to approve it in the UI first."
                        })
                    elif status == "disabled":
                        return json.dumps({
                            "error": f"Tool '{name}' is currently disabled by the user."
                        })
                    else:
                        return json.dumps({
                            "error": f"Tool '{name}' has status '{status}' and cannot be used."
                        })

            available = "web_search, fetch_page, list_context_files, read_context_file, write_context_file, delete_context_file, create_tool"
            return json.dumps({
                "error": f"Unknown tool: {name}. Available tools: {available}"
            })

    @app.post("/api/chat")
    async def chat(payload: ChatMessage):
        config = store.load_config()
        project = payload.project if payload.project is not None else config.get("active_project")
        model = payload.model or config.get("llm", {}).get("model", "claude-sonnet-4-6")

        context_pieces = store.assemble_context(project)
        history = store.load_history(project)

        async def event_stream():
            full_response = []
            try:
                stream_fn = _get_stream_fn(config)
                bound_executor = functools.partial(execute_tool, project=project)
                async for event in stream_fn(
                    user_message=payload.message,
                    tool_executor=bound_executor,
                    context=context_pieces,
                    history=history,
                    model=model,
                    project=project,
                ):
                    if isinstance(event, TextChunkEvent):
                        full_response.append(event.text)
                        yield f"data: {json.dumps({'type': 'chunk', 'text': event.text})}\n\n"
                    elif isinstance(event, ToolCallEvent):
                        if event.name == "create_tool":
                            yield f"data: {json.dumps({'type': 'tool_proposal', 'tool': event.input})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'tool_call', 'name': event.name, 'input': event.input})}\n\n"
                    elif isinstance(event, DoneEvent):
                        response_text = event.full_text
                        try:
                            history.append({"role": "user", "content": payload.message})
                            history.append({"role": "assistant", "content": response_text})
                            store.save_history(project, history)
                        except Exception:
                            pass
                        yield f"data: {json.dumps({'type': 'done', 'full_text': response_text})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    @app.post("/api/chat/upload")
    async def chat_with_files(
        message: str = Form(...),
        project: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        files: list[UploadFile] = File(default=[]),
    ):
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

        config = store.load_config()
        proj = project if project is not None else config.get("active_project")
        mdl = model or config.get("llm", {}).get("model", "claude-sonnet-4-6")

        # Build attachment content blocks
        IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
        TEXT_EXTENSIONS = {
            ".txt", ".md", ".csv", ".json", ".xml", ".py", ".js", ".ts",
            ".html", ".css", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".sh", ".bash", ".rb", ".go", ".rs", ".java", ".c", ".cpp",
            ".h", ".hpp", ".sql", ".r", ".swift", ".kt", ".lua",
        }

        attachments: list[dict] = []
        file_descriptions: list[str] = []

        for f in files:
            data = await f.read()
            if len(data) > 20 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{f.filename}' exceeds 20MB limit",
                )

            mime = f.content_type or mimetypes.guess_type(f.filename or "")[0] or ""
            ext = Path(f.filename or "").suffix.lower()

            if mime in IMAGE_MIMES:
                b64 = base64.standard_b64encode(data).decode("ascii")
                attachments.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                })
                file_descriptions.append(f"[Image: {f.filename}]")
            elif mime == "application/pdf":
                b64 = base64.standard_b64encode(data).decode("ascii")
                attachments.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64,
                    },
                })
                file_descriptions.append(f"[PDF: {f.filename}]")
            elif ext in TEXT_EXTENSIONS or mime.startswith("text/"):
                try:
                    text_content = data.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = data.decode("latin-1")
                attachments.append({
                    "type": "text",
                    "text": f"--- {f.filename} ---\n{text_content}",
                })
                file_descriptions.append(f"[File: {f.filename}]")
            else:
                # Treat unknown as text if small, else skip
                if len(data) < 1024 * 1024:
                    try:
                        text_content = data.decode("utf-8")
                        attachments.append({
                            "type": "text",
                            "text": f"--- {f.filename} ---\n{text_content}",
                        })
                        file_descriptions.append(f"[File: {f.filename}]")
                    except UnicodeDecodeError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported file type: {f.filename} ({mime})",
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {f.filename} ({mime})",
                    )

        context_pieces = store.assemble_context(proj)
        history = store.load_history(proj)

        # For history, store text description instead of base64
        history_content = message
        if file_descriptions:
            history_content = message + "\n\nAttachments: " + ", ".join(file_descriptions)

        async def event_stream():
            try:
                stream_fn = _get_stream_fn(config)
                bound_executor = functools.partial(execute_tool, project=proj)
                async for event in stream_fn(
                    user_message=message,
                    tool_executor=bound_executor,
                    context=context_pieces,
                    history=history,
                    model=mdl,
                    project=proj,
                    attachments=attachments if attachments else None,
                ):
                    if isinstance(event, TextChunkEvent):
                        yield f"data: {json.dumps({'type': 'chunk', 'text': event.text})}\n\n"
                    elif isinstance(event, ToolCallEvent):
                        if event.name == "create_tool":
                            yield f"data: {json.dumps({'type': 'tool_proposal', 'tool': event.input})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'tool_call', 'name': event.name, 'input': event.input})}\n\n"
                    elif isinstance(event, DoneEvent):
                        response_text = event.full_text
                        try:
                            history.append({"role": "user", "content": history_content})
                            history.append({"role": "assistant", "content": response_text})
                            store.save_history(proj, history)
                        except Exception:
                            pass
                        yield f"data: {json.dumps({'type': 'done', 'full_text': response_text})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )


# ── Upload management routes ─────────────────────────────────────

def _register_upload_routes(app: FastAPI) -> None:

    @app.post("/api/uploads/{project}")
    async def upload_file(project: str, file: UploadFile = File(...)):
        if project not in store.list_projects():
            raise HTTPException(status_code=404, detail=f"Project '{project}' not found")
        data = await file.read()
        try:
            path = store.save_upload(project, file.filename or "unnamed", data)
            return {"name": file.filename, "path": str(path), "size": len(data)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/uploads/{project}")
    async def list_uploads(project: str):
        if project not in store.list_projects():
            raise HTTPException(status_code=404, detail=f"Project '{project}' not found")
        return store.list_uploads(project)

    @app.delete("/api/uploads/{project}/{filename}")
    async def delete_upload(project: str, filename: str):
        ok = store.delete_upload(project, filename)
        if not ok:
            raise HTTPException(status_code=404, detail="Upload not found")
        return {"deleted": filename}


# ── Tool management routes ───────────────────────────────────────────

class ToolAction(BaseModel):
    action: str  # "approve", "reject", "disable", "enable"

def _register_tool_routes(app: FastAPI) -> None:

    @app.get("/api/tools/{project}")
    async def list_tools(project: str):
        tools = store.load_tools(project)
        return {"tools": list(tools.values())}

    @app.put("/api/tools/{project}/{name}/approve")
    async def approve_tool(project: str, name: str):
        tool = store.get_tool(project, name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
        if tool["status"] not in ("pending", "disabled"):
            raise HTTPException(status_code=400, detail=f"Tool is {tool['status']}, cannot approve")
        tool["status"] = "approved"
        store.save_tool(project, tool)
        # Write the executable script with HMAC auth
        from knowl.tools.executor import write_tool_script
        write_tool_script(project, tool)
        return {"name": name, "status": "approved"}

    @app.put("/api/tools/{project}/{name}/reject")
    async def reject_tool(project: str, name: str):
        tool = store.get_tool(project, name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
        tool["status"] = "rejected"
        store.save_tool(project, tool)
        return {"name": name, "status": "rejected"}

    @app.put("/api/tools/{project}/{name}/disable")
    async def disable_tool(project: str, name: str):
        tool = store.get_tool(project, name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
        if tool["status"] != "approved":
            raise HTTPException(status_code=400, detail=f"Tool is {tool['status']}, cannot disable")
        tool["status"] = "disabled"
        store.save_tool(project, tool)
        return {"name": name, "status": "disabled"}

    @app.put("/api/tools/{project}/{name}/enable")
    async def enable_tool(project: str, name: str):
        tool = store.get_tool(project, name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
        if tool["status"] != "disabled":
            raise HTTPException(status_code=400, detail=f"Tool is {tool['status']}, cannot enable")
        tool["status"] = "approved"
        store.save_tool(project, tool)
        # Re-write script in case implementation changed
        from knowl.tools.executor import write_tool_script
        write_tool_script(project, tool)
        return {"name": name, "status": "approved"}

    @app.delete("/api/tools/{project}/{name}")
    async def delete_tool_route(project: str, name: str):
        ok = store.delete_tool(project, name)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
        # Clean up script file if it exists
        script_path = store.PROJECTS_DIR / project / "tools" / f"{name}.py"
        if script_path.exists():
            script_path.unlink()
        return {"deleted": name}


# ── Promotion routes ─────────────────────────────────────────────────

def _register_promotion_routes(app: FastAPI) -> None:

    @app.get("/api/promotions")
    async def get_promotions(threshold: float = Query(0.3)):
        from knowl.context.promotion import scan_for_promotions
        suggestions = scan_for_promotions(similarity_threshold=threshold)
        return [
            {
                "filename": s.filename,
                "projects": s.projects,
                "similarity": s.similarity,
                "reason": s.reason,
                "conflict": s.conflict,
            }
            for s in suggestions
        ]

    @app.post("/api/promotions/apply")
    async def apply_promotion(payload: ApplyPromotion):
        from knowl.context.promotion import scan_for_promotions, apply_promotion as do_apply

        suggestions = scan_for_promotions()
        matching = [s for s in suggestions if s.filename == payload.filename]
        if not matching:
            raise HTTPException(status_code=404, detail=f"No promotion suggestion for '{payload.filename}'")

        suggestion = matching[0]
        if suggestion.conflict and not payload.force:
            raise HTTPException(status_code=409, detail="File exists in global. Use force=true to overwrite.")

        source = payload.source_project or suggestion.projects[0]
        result = do_apply(suggestion, source_project=source, remove_from_projects=payload.cleanup)
        if not result:
            raise HTTPException(status_code=500, detail="Promotion failed")
        return {"promoted": payload.filename, "path": str(result)}
