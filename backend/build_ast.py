from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import threading
import queue
from pathlib import Path

from .ast_gen import run_ast_generation

router = APIRouter()


class AstRequest(BaseModel):
    project: str
    project_dir: str


@router.post("/build_ast")
async def build_ast(req: AstRequest):
    """
    Build AST by running gen_ast_all_no_multi.py under project_dir, streaming progress.
    Returns status JSON upon completion.
    """
    project_dir = Path(req.project_dir)
    q = queue.Queue()

    def worker():
        ast_dir = run_ast_generation(project_dir, on_line=q.put)
        # Signal completion with JSON status
        q.put(f"{{""status"":""success"", ""project"":""{req.project}"", ""ast_dir"":""{ast_dir}""}}")
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="text/plain")
