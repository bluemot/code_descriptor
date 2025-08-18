from fastapi import APIRouter
from pydantic import BaseModel
import os
import qdrant_client as qc
from pathlib import Path

from .rag_demo6_1 import get_coll_name, build_index as rd_build_index

router = APIRouter()


class BuildRequest(BaseModel):
    project: str
    project_dir: str
    force: bool = False


@router.post("/build_rag")
async def build_rag(req: BuildRequest):
    coll = get_coll_name(req.project)
    url = os.getenv("QDRANT_URL", None)
    key = os.getenv("QDRANT_KEY", None) or None
    if url:
        client = qc.QdrantClient(url=url, api_key=key)
    else:
        client = qc.QdrantClient(path="rag_demo_qdrant")

    # 確認 AST 輸出已在 project_dir/ast_out
    ast_out = Path(req.project_dir) / 'ast_out'
    if not ast_out.exists() or not any(ast_out.iterdir()):
        return {"status": "error", "message": "AST output not found, please build AST first."}

    # If exists and not forced, return exists
    if client.collection_exists(coll) and not req.force:
        info = client.get_collection(coll)
        return {"status": "exists", "project": req.project, "points": info.points_count}

    # If forced and exists, delete first
    if client.collection_exists(coll) and req.force:
        client.delete_collection(coll)

    # Change working directory to project_dir before indexing
    os.chdir(req.project_dir)
    # Build index
    rd_build_index(client, req.project_dir, coll)
    info = client.get_collection(coll)
    return {"status": "success", "project": req.project, "points": info.points_count}
