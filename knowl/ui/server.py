"""Knowl Web UI — FastAPI backend serving REST API + React frontend."""

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from knowl.context import store
from knowl.llm import claude


# ── Pydantic request models (must be at module level for FastAPI) ────

class CreateProject(BaseModel):
    name: str

class UpdateConfig(BaseModel):
    active_project: Optional[str] = None
    model: Optional[str] = None

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
    app = FastAPI(title="Knowl", version="0.1.0")

    if dev:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Ensure store is initialized
    store.init_store()

    # ── Register routes ──────────────────────────────────────────────
    _register_project_routes(app)
    _register_config_routes(app)
    _register_context_routes(app)
    _register_inspect_routes(app)
    _register_chat_routes(app)
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
    async def create_project(body: CreateProject):
        path = store.create_project(body.name)
        return {"name": body.name, "path": str(path)}

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
    async def update_config(body: UpdateConfig):
        config = store.load_config()
        if body.active_project is not None:
            # Validate project exists
            if body.active_project and body.active_project not in store.list_projects():
                raise HTTPException(status_code=404, detail=f"Project '{body.active_project}' not found")
            config["active_project"] = body.active_project or None
        if body.model is not None:
            config.setdefault("llm", {})["model"] = body.model
        store.save_config(config)
        return config


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
    async def write_file(body: WriteFile):
        try:
            store.write_context_file(body.path, body.content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"path": body.path, "tokens": store.estimate_tokens(body.content)}

    @app.post("/api/context/file")
    async def create_file(body: CreateFile):
        if body.scope == "global":
            target = store.GLOBAL_DIR / body.filename
        else:
            target = store.PROJECTS_DIR / body.scope / body.filename
        if target.exists():
            raise HTTPException(status_code=409, detail="File already exists")
        content = body.content or f"# {body.filename}\n\n"
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
    async def set_active_files(project: str, body: SetActiveFiles):
        store.set_active_files(project, body.files)
        return {"project": project, "active_files": body.files}

    @app.post("/api/context/promote")
    async def promote_file(body: PromoteFile):
        result = store.promote_to_global(body.project, body.filename)
        if not result:
            raise HTTPException(status_code=404, detail="File not found in project")
        return {"promoted": body.filename, "path": str(result)}


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

    @app.post("/api/chat")
    async def chat(body: ChatMessage):
        config = store.load_config()
        project = body.project if body.project is not None else config.get("active_project")
        model = body.model or config.get("llm", {}).get("model", "claude-sonnet-4-6")

        context_pieces = store.assemble_context(project)
        history = store.load_history(project)

        def event_stream():
            full_response = []
            try:
                for chunk in claude.stream_message(
                    user_message=body.message,
                    context=context_pieces,
                    history=history,
                    model=model,
                ):
                    full_response.append(chunk)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
            except RuntimeError as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
                return

            # Save to history after streaming completes
            response_text = "".join(full_response)
            history.append({"role": "user", "content": body.message})
            history.append({"role": "assistant", "content": response_text})
            store.save_history(project, history)
            yield f"data: {json.dumps({'type': 'done', 'full_text': response_text})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")


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
    async def apply_promotion(body: ApplyPromotion):
        from knowl.context.promotion import scan_for_promotions, apply_promotion as do_apply

        suggestions = scan_for_promotions()
        matching = [s for s in suggestions if s.filename == body.filename]
        if not matching:
            raise HTTPException(status_code=404, detail=f"No promotion suggestion for '{body.filename}'")

        suggestion = matching[0]
        if suggestion.conflict and not body.force:
            raise HTTPException(status_code=409, detail="File exists in global. Use force=true to overwrite.")

        source = body.source_project or suggestion.projects[0]
        result = do_apply(suggestion, source_project=source, remove_from_projects=body.cleanup)
        if not result:
            raise HTTPException(status_code=500, detail="Promotion failed")
        return {"promoted": body.filename, "path": str(result)}
