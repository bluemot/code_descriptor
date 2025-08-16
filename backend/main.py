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


app = FastAPI()

# Serve frontend static files and index
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

    # Handle Run action via JSON project_dir
    if content_type.startswith('application/json'):
        data = await request.json()
        project_dir = data.get('project_dir')
        if not project_dir:
            return {"status": "error", "detail": "project_dir is required"}
        p = Path(project_dir)
        if not p.exists() or not p.is_dir():
            return {"status": "error", "detail": "Invalid project directory"}
        try:
            ast_dir = run_ast_generation(p)
            build_index(ast_dir)
        except Exception as e:
            return {"status": "error", "detail": str(e)}
        return {"status": "success"}

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
