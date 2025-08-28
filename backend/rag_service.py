"""
Wrapper for Qdrant indexing and question-answering using the rag_demo6_1 script.
"""
from .config import load_env  # noqa: E402
load_env()

import os
import qdrant_client as qc
from pathlib import Path
from .retrieval.types import Candidate
from .rag_demo6_1 import EMBEDDER, QDRANT_URL
import time
import re


def qdrant_embedding_recall(query: str, topk: int, collection: str | None = None) -> list[Candidate]:
    """
    Embed query and recall topk via Qdrant, with dimension check and fallback to [].
    """
    url = os.getenv("QDRANT_URL", QDRANT_URL)
    key = os.getenv("QDRANT_KEY", None)
    try:
        client = qc.QdrantClient(url=url, api_key=key) if url else qc.QdrantClient(path="rag_demo_qdrant")
    except Exception:
        return []

    coll = collection or os.getenv("RAG_COLLECTION")
    if not coll or not client.collection_exists(coll):
        return []

    try:
        info = client.get_collection(coll)
        pts = getattr(info, 'points_count', None) or getattr(getattr(info,'result',None),'points_count',0)
        cfg = getattr(getattr(info,'config',None),'params',None)
        coll_dim = getattr(getattr(cfg,'vectors',None),'size',None) or cfg.vectors.get('size', None)
    except Exception:
        return []
    if not pts:
        return []

    try:
        # 取得 collection 維度與 points 數，協助診斷
        coll_dim = None
        points_count = None
        try:
            info = client.get_collection(coll)
            coll_dim = getattr(getattr(getattr(info, 'config', None), 'params', None), 'vectors', None)
            if isinstance(coll_dim, dict) and 'size' in coll_dim:
                coll_dim = coll_dim['size']
            elif hasattr(info, 'vectors_config') and hasattr(info.vectors_config, 'size'):
                coll_dim = info.vectors_config.size
            else:
                coll_dim = getattr(getattr(info, 'vectors', None), 'size', None)
            points_count = getattr(info, 'points_count', None)
        except Exception as e:
            print(f"[embed] get_collection info fail: {type(e).__name__}: {e}")
        t0 = time.time()
        vec = EMBEDDER.encode([query])[0]
        qv = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        model_dim = len(qv)
    except Exception:
        return []
    if model_dim != coll_dim:
        return []

    # Perform search and log diagnostics
    try:
        qdim = len(qv) if isinstance(qv, list) else None
        hits = client.search(collection_name=coll, query_vector=qv, limit=topk, with_payload=True)
        t1 = time.time()
        print(f"[embed] coll={coll} coll_dim={coll_dim} points={points_count} qdim={qdim} "
              f"hits={len(hits)} time_ms={(t1 - t0)*1000:.1f}")
        for i, h in enumerate(hits[:5]):
            pay = getattr(h, 'payload', {}) or {}
            print(f"[embed:hit{i}] score={getattr(h,'score',0):.4f} file={pay.get('file')} "
                  f"oid={(pay.get('original_id') or '')[:80]} txt_len={len(pay.get('text') or '')}")
    except Exception:
        return []

    results: list[Candidate] = []
    for h in hits:
        pay = h.payload or {}
        txt = pay.get('text', '')
        if not txt:
            continue
        f = pay.get('file', '(unknown)')
        score = float(getattr(h, 'score', 0.0) or 0.0)
        hval = (hash(txt) & 0x7FFFFFFF) % 100000
        sl = hval
        el = hval + max(0, txt.count("\n"))
        cand = Candidate(
            id=str(h.id), file=f, function=None, signature=None,
            start_line=sl, end_line=el, scope='func', score=score,
            source='embed', text=txt, meta={}
        )
        results.append(cand)
    return results


def qdrant_symbol_lookup(query: str, limit: int = 10, collection: str | None = None) -> list[Candidate]:
    """
    Identifier/symbol shortcut via Qdrant payload search.
    - 優先以 payload.original_id 的全文 match 搜尋；其次嘗試 payload.text；
    - 若伺服器無啟用 text index，退化為小批 scroll + 本地 substring 比對；
    - 任何例外不拋出，回傳 []。
    回傳的 Candidate 會給很高的 score，確保在 hybrid 合併時能浮上來。
    """
    try:
        if not query or len(query) < 3:
            return []
        url = os.getenv("QDRANT_URL", QDRANT_URL)
        key = os.getenv("QDRANT_KEY", None) or None
        client = qc.QdrantClient(url=url, api_key=key) if url else qc.QdrantClient(path="rag_demo_qdrant")
        coll = collection
        if not coll:
            coll = os.getenv("RAG_COLLECTION", None)
            if not coll:
                return []

        # 小工具：嘗試建立 text 索引（若未關閉）
        def _ensure_text_index():
            if os.getenv("CREATE_TEXT_INDEX_ON_DEMAND", "1").lower() in ("1", "true", "yes"):
                try:
                    client.create_payload_index(coll, field_name="original_id", field_schema={"type": "text"})
                    print("[symbol] created/ensured text index on 'original_id'")
                except Exception:
                    pass
                try:
                    client.create_payload_index(coll, field_name="text", field_schema={"type": "text"})
                    print("[symbol] created/ensured text index on 'text'")
                except Exception:
                    pass

        points = []
        # 1) 以 original_id 做全文 match（若伺服器支援 text index）
        for attempt in (0, 1):
            try:
                pts, _ = client.scroll(
                    collection_name=coll,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter={"must": [{"key": "original_id", "match": {"text": query}}]},
                )
                points = pts or []
            except Exception:
                points = []
            if points:
                print(f"[symbol] original_id match success on attempt={attempt}, hits={len(points)}")
                break
            if attempt == 0:
                print("[symbol] original_id match empty -> ensure text index and retry")
                _ensure_text_index()  # 第一次失敗時嘗試建立索引，然後重試一次

        # 2) 若沒命中，改用 payload.text
        if not points:
            for attempt in (0, 1):
                try:
                    pts, _ = client.scroll(
                        collection_name=coll,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False,
                        scroll_filter={"must": [{"key": "text", "match": {"text": query}}]},
                    )
                    points = pts or []
                except Exception:
                    points = []
                if points:
                    print(f"[symbol] text match success on attempt={attempt}, hits={len(points)}")
                    break
                if attempt == 0:
                    print("[symbol] text match empty -> ensure text index and retry")
                    _ensure_text_index()

        # 3) 退化方案：分頁掃描 + 本地 substring（不再只限 scope=func）
        if not points:
            hits = []
            next_off = None
            scanned = 0
            cap = 50000  # 最多掃 5 萬點避免卡死
            while True:
                try:
                    batch, next_off = client.scroll(
                        collection_name=coll,
                        limit=1000,
                        with_payload=True,
                        with_vectors=False,
                        offset=next_off
                    )
                except Exception:
                    break
                scanned += len(batch or [])
                for p in (batch or []):
                    pay = p.payload or {}
                    oid = str(pay.get("original_id", ""))
                    txt = str(pay.get("text", ""))
                    if query in oid or query in txt:
                        hits.append(p)
                        if len(hits) >= limit:
                            break
                if len(hits) >= limit or not next_off or scanned >= cap:
                    break
            t0 = time.time()
            points = hits
            t1 = time.time()
            print(f"[symbol] fallback scan scanned={scanned} collected={len(points)} "
                  f"time_ms={(t1 - t0)*1000:.1f}")

        out: list[Candidate] = []
        base = 1_000_000.0  # 讓符號命中優先浮上

        def _extract_sig(txt: str, func: str) -> str:
            if not txt or not func:
                return ""
            m = re.search(rf"[^\n;{{]{{0,200}}\b{re.escape(func)}\s*\([^)]*\)", txt)
            return m.group(0).strip() if m else ""

        for i, p in enumerate(points or []):
            pay = p.payload or {}
            txt = pay.get("text") or ""
            if not txt:
                continue
            fpath = pay.get("file") or "(unknown)"
            original_id = pay.get("original_id") or ""
            scope = pay.get("scope") or "func"
            pid = getattr(p, "id", None)
            cid = str(pid) if pid is not None else (original_id or str((hash(txt) & 0x7FFFFFFF)))
            # 儘量從 original_id 推出 function 名稱："<path>::<function>"
            func_name = ""
            if isinstance(original_id, str) and "::" in original_id:
                func_name = original_id.split("::", 1)[1]
            sig = pay.get("signature") or _extract_sig(txt, func_name)
            # pseudo 行號（與既有 merge key 相容）
            h = (hash(txt) & 0x7FFFFFFF) % 100000
            sl = h
            el = h + max(0, txt.count("\n"))
            score = base - float(i)
            out.append(Candidate(
                id=cid,
                file=fpath,
                function=func_name,
                signature=sig,
                start_line=sl,
                end_line=el,
                scope=scope,
                score=score,
                source="sym",
                text=txt,
                meta={"original_id": original_id},
            ))
        # 顯示前幾筆 symbol 命中，便於確認是否抓到預期函式
        print(f"[retrieve:symbol] query='{query}' hits={len(out)} coll={coll}")
        for i, c in enumerate(out[:5]):
            pv = (c.text or "")[:100].replace("\n", " ")
            print(f"[symbol:hit{i}] id={c.id} func={c.function} scope={c.scope} score={c.score:.1f} sig_preview={(c.signature or '')[:80]} preview={pv}")
        return out
    except Exception as e:
        print(f"[retrieve:symbol] error={type(e).__name__} msg={e}")
        return []
def build_index(ast_dir: Path) -> None:
    """
    Build or update the Qdrant index based on AST files in ast_dir.
    """
    try:
        import qdrant_client as qc
        from .rag_demo6_1 import build_index as rd_build_index, QDRANT_URL

        url = os.getenv("QDRANT_URL", QDRANT_URL)
        key = os.getenv("QDRANT_KEY", None)
        if url:
            client = qc.QdrantClient(url=url, api_key=key)
        else:
            client = qc.QdrantClient(path="rag_demo_qdrant")
        rd_build_index(client, str(ast_dir))
    except Exception:
        # Swallow any errors during indexing
        pass


def answer_question(question: str) -> str:
    """
    Retrieve and generate an answer for the given question using the RAG service.
    """
    try:
        import qdrant_client as qc
        from .rag_demo6_1 import answer as rd_answer, QDRANT_URL

        url = os.getenv("QDRANT_URL", QDRANT_URL)
        key = os.getenv("QDRANT_KEY", None)
        if url:
            client = qc.QdrantClient(url=url, api_key=key)
        else:
            client = qc.QdrantClient(path="rag_demo_qdrant")
        return rd_answer(client, question)
    except Exception:
        # On error, return empty answer
        return ""
