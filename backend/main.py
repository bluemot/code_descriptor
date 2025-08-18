"""
FastAPI application exposing /upload and /ask endpoints.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import load_env  # noqa: E402
load_env()
import tempfile
from pathlib import Path
from .ast_gen import run_ast_generation
from .rag_service import build_index, answer_question

# Include build_rag and ask_rag routers
from .build_rag import router as build_rag_router
from .ask_rag import router as ask_rag_router
from .build_ast import router as build_ast_router


app = FastAPI()

# Disable caching for app.js to force front-end reloads
@app.middleware("http")
async def no_cache_js(request, call_next):
    response = await call_next(request)
    # Avoid caching for JS static files (e.g. app.js)
    if request.url.path.startswith("/static/") and request.url.path.endswith(".js"):
        response.headers["Cache-Control"] = "no-store"
    return response

# Serve frontend static files and index
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import threading
import queue
import json

# Mount the frontend directory for static assets
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def serve_index():
    """Serve the SPA entry point."""
    return FileResponse('frontend/index.html')

# Allow CORS for all origins (for front-end usage)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload(request: Request):
    """
    Upload multiple source files or use a server-side project directory to generate ASTs and build the index.
    """
    data = None
    content_type = request.headers.get('content-type', '')
    # Handle file upload (prepare project directory)
    if content_type.startswith('multipart/form-data'):
        form = await request.form()
        files = form.getlist('files')
        project_dir = tempfile.mkdtemp()
        temp_path = Path(project_dir)
        for file in files:
            dest = temp_path / file.filename
            content = await file.read()
            dest.write_bytes(content)
        # Return the server-side project_dir for subsequent Run
        return {"status": "success", "project_dir": project_dir}

    # Handle Run action via JSON project_dir with streaming progress
    if content_type.startswith('application/json'):
        data = await request.json()
        project_dir = data.get('project_dir')
        if not project_dir:
            return {"status": "error", "detail": "project_dir is required"}
        p = Path(project_dir)
        if not p.exists() or not p.is_dir():
            return {"status": "error", "detail": "Invalid project directory"}

        def event_stream():
            q = queue.Queue()

            def worker():
                # Generate ASTs, streaming subprocess output
                ast_dir = run_ast_generation(p, on_line=q.put)
                try:
                    build_index(ast_dir)
                except Exception as e:
                    q.put(f"__INDEX_ERROR__:{e}")
                # Signal completion
                q.put(None)

            threading.Thread(target=worker, daemon=True).start()

            while True:
                item = q.get()
                if item is None:
                    break
                yield item

        return StreamingResponse(event_stream(), media_type="text/plain")

    return {"status": "error", "detail": "Unsupported content type"}


@app.post("/ask")
async def ask(q: Question):
    """
    Receive a question, retrieve context via embeddings, and generate an answer.
    """
    try:
        ans = answer_question(q.question)
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    return {"answer": ans}

# RAG build and ask endpoints
app.include_router(build_rag_router)
app.include_router(ask_rag_router)
app.include_router(build_ast_router)
