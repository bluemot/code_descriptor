from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import os
import re
import time
import qdrant_client as qc

# 我們仍使用 rag_demo6_1 的 collection 命名邏輯與 LLM 實例
from .rag_demo6_1 import get_coll_name, LLM_SUM

router = APIRouter()


class AskRequest(BaseModel):
    project: Optional[str] = None
    collection: Optional[str] = None
    question: str


@router.post("/ask_rag")
async def ask_rag(req: AskRequest, background_tasks: BackgroundTasks):
    # determine collection from request or project
    if not req.collection and not req.project:
        return {"status": "error", "message": "collection or project must be provided"}
    coll = req.collection or get_coll_name(req.project)
    url = os.getenv("QDRANT_URL", None)
    key = os.getenv("QDRANT_KEY", None) or None
    if url:
        client = qc.QdrantClient(url=url, api_key=key)
    else:
        client = qc.QdrantClient(path="rag_demo_qdrant")

    if not client.collection_exists(coll):
        return {"status": "not_found", "message": f"Project '{req.project}' not found, please build RAG first."}

    print(f"Qdrant URL:{url} coll {coll}")
    from .retrieval.hybrid import hybrid_recall
    # 先抽 query 裡的 code-like tokens，方便後續診斷
    tokens = [t for t in re.findall(r"[A-Za-z_]\w+", req.question or "") if len(t) >= 3]
    print(f"[ask_rag] query_tokens={tokens[:6]}")

    # hybrid recall（把使用者選的 collection 傳下去）
    t0 = time.time()
    cands = hybrid_recall(req.question, 40, collection=coll)
    t1 = time.time()
    print(f"[ask_rag] hybrid recall got {len(cands)} candidates in {(t1 - t0)*1000:.1f} ms")

    # 顯示 top5 候選摘要，包含管道/分數/檔案/行範圍與文字預覽
    for i, c in enumerate(cands[:5]):
        preview = (c.text or "")[:120].replace("\n", " ")
        print(f"[ask_rag][top{i}] src={getattr(c,'source','?')} score={getattr(c,'score',0):.4f} "
              f"file={getattr(c,'file','?')} span={getattr(c,'start_line','?')}-{getattr(c,'end_line','?')} "
              f"preview={preview}")

    # 統計 tokens 在候選中的命中分佈
    if tokens:
        hit_stats = {t: sum(1 for c in cands if (c.text and t in c.text)) for t in tokens[:6]}
        print(f"[ask_rag] token_hit_stats={hit_stats}")

    # assemble context text
    used_cnt = sum(1 for c in cands if getattr(c, "text", None))
    context = "-----\n".join(c.text for c in cands if getattr(c, "text", None))
    print(f"[ask_rag] context length = {len(context)} chars, used_snippets={used_cnt}")
    if not context:
        print("[ask_rag] no context -> return (no context)")
        return {"status": "success", "project": req.project, "collection": coll, "answer": "(no context)"}

    # 嚴格防胡謅的提示：沒有根據 context 的資訊時不得猜測
    prompt = (
        "你是一位資深 C/C++ 網路驅動工程師，請『只根據』以下片段回答問題；"
        "若片段中找不到答案，請回覆「(no context)」。\n"
        "禁止臆測或編造檔名/函式/檔案路徑。\n\n"
        f"{context}\n\n"
        f"問題：{req.question}\n"
        "回答時簡潔直接；若無根據請回「(no context)」。\nA:"
    )
    print(f"[ask_rag] prompt_chars={len(prompt)}")
    try:
        t2 = time.time()
        ans = LLM_SUM.invoke(prompt)
        t3 = time.time()
        print(f"[ask_rag] LLM answered in {(t3 - t2)*1000:.1f} ms, ans_len={len(str(ans))}, "
              f"ans_preview={(str(ans)[:160]).replace(chr(10),' ')}")
    except Exception as e:
        print(f"[ask_rag] LLM invoke failed: {type(e).__name__}: {e}")
        ans = "(no context)"
    return {"status": "success", "project": req.project, "collection": coll, "answer": ans}


@router.get("/collections")
async def list_collections():
    """List projects and collections from Qdrant."""
    url = os.getenv("QDRANT_URL", None)
    key = os.getenv("QDRANT_KEY", None) or None
    client = qc.QdrantClient(url=url, api_key=key) if url else qc.QdrantClient(path="rag_demo_qdrant")
    try:
        result = client.get_collections()
    except Exception as e:
        return {"status": "error", "message": f"Failed to retrieve collections: {e}"}
    raw = []
    if hasattr(result, "collections"):
        raw = [info.name for info in result.collections]
    elif isinstance(result, dict) and "collections" in result:
        raw = [c.get("name") for c in result["collections"]]
    projects = []
    mapping: dict[str, str] = {}
    for coll in raw:
        if coll.startswith("proj_ast_"):
            proj = coll[len("proj_ast_"):]
            projects.append(proj)
            mapping[proj] = coll
    return {"status": "success", "collections": list(mapping.values()), "projects": projects, "mapping": mapping}
