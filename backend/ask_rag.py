from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
import os
import qdrant_client as qc

from .rag_demo6_1 import get_coll_name, answer as rd_answer

router = APIRouter()


class AskRequest(BaseModel):
    project: str
    question: str


@router.post("/ask_rag")
async def ask_rag(req: AskRequest, background_tasks: BackgroundTasks):
    coll = get_coll_name(req.project)
    url = os.getenv("QDRANT_URL", None)
    key = os.getenv("QDRANT_KEY", None) or None
    if url:
        client = qc.QdrantClient(url=url, api_key=key)
    else:
        client = qc.QdrantClient(path="rag_demo_qdrant")

    if not client.collection_exists(coll):
        return {"status": "not_found", "message": f"Project '{req.project}' not found, please build RAG first."}

    # New retrieval pipeline: hybrid recall -> rerank -> pack -> LLM, with on-demand deepen in background
    try:
        # retrieval modules
        from .retrieval.hybrid import hybrid_recall
        from .retrieval.rerank import rerank_and_fuse
        from .retrieval.pack import expand_neighbors, pack_by_lines
        from .retrieval.deepen import on_demand_deepen
        from .retrieval.types import PackedContext

        # retrieval params
        te = int(os.getenv('EMBED_TOP_K', '90'))
        tb = int(os.getenv('BM25_TOP_K', '80'))
        ti = int(os.getenv('ID_TOP_K', '40'))
        mc = int(os.getenv('HYBRID_MAX_CAND', '120'))

        # hybrid recall
        cands = hybrid_recall(req.question, topk_embed=te, topk_bm25=tb, topk_id=ti, max_cand=mc)
        # rerank & fuse
        ranked = rerank_and_fuse(req.question, cands)
        # expand & pack context
        expanded = expand_neighbors(ranked)
        packed: PackedContext = pack_by_lines(expanded)

        # background deepen
        background_tasks.add_task(on_demand_deepen, req.project, packed.files_covered)

        # prompt assembly
        blocks = []
        for it in packed.items:
            blocks.append(f"```{it.file}:{it.start_line}-{it.end_line}\n{it.text}\n```")
        prompt = (
            "你是一位資深 C/C++ 網路驅動工程師，以下程式片段可作為上下文：\n" +
            "\n---\n".join(blocks) +
            f"\n\n{req.question}\n不要加任何客套、署名，回答完輸出 <END>。\nA:")

        ans = rd_answer(client, coll, prompt)
        return {"status": "success", "project": req.project, "answer": ans}
    except Exception:
        # fallback original
        ans = rd_answer(client, coll, req.question)
        return {"status": "success", "project": req.project, "answer": ans}
