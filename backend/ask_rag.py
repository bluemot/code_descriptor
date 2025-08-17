from fastapi import APIRouter
from pydantic import BaseModel
import os
import qdrant_client as qc

from .rag_demo6_1 import get_coll_name, answer as rd_answer

router = APIRouter()


class AskRequest(BaseModel):
    project: str
    question: str


@router.post("/ask_rag")
async def ask_rag(req: AskRequest):
    coll = get_coll_name(req.project)
    url = os.getenv("QDRANT_URL", None)
    key = os.getenv("QDRANT_KEY", None) or None
    if url:
        client = qc.QdrantClient(url=url, api_key=key)
    else:
        client = qc.QdrantClient(path="rag_demo_qdrant")

    if not client.collection_exists(coll):
        return {"status": "not_found", "message": f"Project '{req.project}' not found, please build RAG first."}

    ans = rd_answer(client, coll, req.question)
    return {"status": "success", "project": req.project, "answer": ans}
