"""
FastAPI application exposing /upload and /ask endpoints.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import load_env  # noqa: E402
load_env()
import tempfile
import os
from pathlib import Path
from .ast_gen import run_ast_generation
from .rag_service import build_index, answer_question

# Include build_rag and ask_rag routers
from .build_rag import router as build_rag_router
from .ask_rag import router as ask_rag_router
from .build_ast import router as build_ast_router

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware


app = FastAPI()

# ───────────────── Frontend path detection ─────────────────
# Default to <repo_root>/frontend/dist (Vite/most setups). Allow override via FRONTEND_DIR.
# We resolve relative to this file so it doesn't depend on current working dir.
REPO_ROOT = Path(__file__).resolve().parents[1]
_frontend_env = os.getenv("FRONTEND_DIR", "")
FRONTEND_DIR = Path(_frontend_env).resolve() if _frontend_env else (REPO_ROOT / "frontend" / "dist")
INDEX_HTML = FRONTEND_DIR / "index.html"

if INDEX_HTML.exists():
    print(f"[info] Serving SPA from: {FRONTEND_DIR}")
else:
    print(f"[warn] index.html not found at: {INDEX_HTML}")
    print("       Hints: (1) cd frontend && npm run build  → produces frontend/dist")
    print("              (2) or export FRONTEND_DIR=/absolute/path/to/your/build")

class CacheControlForSpa(BaseHTTPMiddleware):
    """
    Best-practice SPA caching:
    - index.html:      no-cache (so clients pick up new hashed assets)
    - /assets/* files: long cache with immutable (hashed filenames)
    Legacy:
    - /static/* also long cache (temporary compatibility; prefer removing /static usage).
    """
    ASSET_SUFFIXES = (".js", ".css", ".map", ".woff2", ".woff", ".ttf", ".eot",
                      ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        p = request.url.path
        pl = p.lower()
        # index.html (root or explicit)
        if p == "/" or pl.endswith("/index.html"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
        # Vite assets (hashed)
        if pl.startswith("/assets/") and pl.endswith(self.ASSET_SUFFIXES):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response
        # Temporary legacy: /static/* assets
        if pl.startswith("/static/") and pl.endswith(self.ASSET_SUFFIXES):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response
        return response

app.add_middleware(CacheControlForSpa)

# Mount SPA static dir when available (html=True enables index fallback for static handler)
if FRONTEND_DIR.exists():
    # Mount SPA static dir when available (html=True enables index fallback for static handler)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="spa")
    # Temporary legacy compatibility for older /static/* references.
    # This points to the same dist directory; prefer migrating HTML to Vite-managed paths.
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="legacy-static")
    print("[warn] Mounted legacy /static → dist/. Update your HTML to rely on Vite output (e.g. /assets/*).")

@app.get("/")
async def serve_index():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return PlainTextResponse(
        "frontend build not found. Run `npm run build` in ./frontend or set FRONTEND_DIR.",
        status_code=500,
    )

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


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return PlainTextResponse(
        "frontend build not found. Run `npm run build` in ./frontend or set FRONTEND_DIR.",
        status_code=500,
    )

# Silence Chrome DevTools /.well-known probe to avoid noisy 404 logs.
@app.get("/.well-known/{path:path}")
async def well_known_probe(path: str):
    # Respond No Content; adjust if you need to serve a real JSON later.
    return Response(status_code=204)
