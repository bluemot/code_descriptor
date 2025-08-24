from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import threading
import queue
import os
import io
import contextlib
import qdrant_client as qc
from qdrant_client.http.models import CollectionStatus
from pathlib import Path

from .rag_demo6_1 import get_coll_name, build_index as rd_build_index

router = APIRouter()


class BuildRequest(BaseModel):
    project: str
    project_dir: str
    force: bool = False


@router.post("/build_rag")
async def build_rag(req: BuildRequest):
    """
    Build RAG index, stream progress logs to client, and send final JSON status.
    """
    q: queue.Queue[str] = queue.Queue()

    class _QueueWriter:
        """File-like writer that streams lines into the queue in real time."""
        def __init__(self, q: queue.Queue[str]):
            self.q = q
            self._buf = ''

        def write(self, s: str) -> None:
            self._buf += s
            if '\n' in self._buf:
                parts = self._buf.split('\n')
                for line in parts[:-1]:
                    self.q.put(line)
                self._buf = parts[-1]

        def flush(self) -> None:
            if self._buf:
                self.q.put(self._buf)
                self._buf = ''

    def worker():
        try:
            #coll = get_coll_name(req.project)
            coll = f"proj_ast_{req.project}"
            url = os.getenv("QDRANT_URL", None)
            key = os.getenv("QDRANT_KEY", None) or None
            if url:
                client = qc.QdrantClient(url=url, api_key=key, timeout=1200)
                q.put(f"Using remote Qdrant at {url}")
            else:
                client = qc.QdrantClient(path="rag_demo_qdrant")
                q.put("Using local Qdrant at path rag_demo_qdrant")

            # Check AST output
            ast_out = Path(req.project_dir) / "ast_out"
            if not ast_out.exists() or not any(ast_out.iterdir()):
                q.put(f"{{\"status\":\"error\",\"message\":\"AST output not found, please build AST first.\"}}")
                q.put(None)
                return

            # Skip existing if not forced
            if client.collection_exists(coll) and not req.force:
                info = client.get_collection(coll)
                q.put(f"Collection {coll} exists with {info.points_count} points (skipped build)")
                q.put(f"{{\"status\":\"exists\",\"project\":\"{req.project}\",\"points\":{info.points_count}}}")
                q.put(None)
                return

            # Creation and upsert are handled inside build_index (_ensure_collection)

            # Switch to project directory and build, streaming output
            os.chdir(req.project_dir)
            q.put("Starting build_index")
            writer = _QueueWriter(q)
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                rd_build_index(client, req.project_dir, coll)
            writer.flush()
            q.put("Finished build_index")

            # Final status
            info = client.get_collection(coll)
            points_count = info.points_count if info is not None else 0
            if points_count == 0:
                q.put(f"{{\"status\":\"error\",\"project\":\"{req.project}\",\"message\":\"RAG build failed, no points indexed\"}}")
            else:
                q.put(
                    f"{{\"status\":\"success\",\"project\":\"{req.project}\",\"collection\":\"{coll}\",\"points\":{points_count},\"message\":\"RAG build finished successfully\"}}"
                )
        except Exception as e:
            msg = str(e).replace('"', '\\"')
            q.put(f"{{\"status\":\"error\",\"message\":\"Build exception: {msg}\"}}")
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield item + "\n"

    return StreamingResponse(event_stream(), media_type="text/plain")
