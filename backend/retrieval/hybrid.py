"""
Hybrid recall combining Embedding (Qdrant), BM25, and Identifier channels.
"""
from typing import List, Dict, Optional
import re
import time

from .types import Candidate



def hybrid_recall(query: str, topk: int, collection: Optional[str] = None) -> List[Candidate]:
    """
    Perform hybrid recall combining Qdrant embedding, BM25, and identifier matching.
    Merge duplicate spans, sort by score, and return up to topk candidates.
    `collection`：若提供，會傳遞給 Qdrant 向量通道以查指定 collection。
    """
    from ..rag_service import qdrant_embedding_recall, qdrant_symbol_lookup
    from .bm25_index import bm25_recall
    from .identifier_index import id_recall


    # 0) 符號名捷徑（強化版）：
    #    從整句 query 中抽取所有 code-like tokens（含中文/標點時也能抓到）
    sym_c: List[Candidate] = []
    if query:
        # 抽取所有符合程式識別子的片段；長度>=3 避免太短的雜訊
        tokens = [t for t in re.findall(r"[A-Za-z_]\w+", query) if len(t) >= 3]
        # 保留原始順序，照在 query 裡出現的先後進行 lookup
        seen = set()
        ordered_tokens = [t for t in tokens if not (t in seen or seen.add(t))]
        t0 = time.time()
        for tok in ordered_tokens[:4]:
            hits = qdrant_symbol_lookup(query=tok, limit=min(10, topk), collection=collection)
            for c in hits:
                c.source = 'sym'
            sym_c.extend(hits)
        t1 = time.time()
        # 去重（依 file/span）以防多個 token 命中同一段
        dedup: Dict[str, Candidate] = {}
        for c in sym_c:
            k = f"{c.file}:{c.start_line}-{c.end_line}"
            if k not in dedup or c.score > dedup[k].score:
                dedup[k] = c
        sym_c = list(dedup.values())
        print(f"[retrieve:symbol_tokens] tokens={ordered_tokens[:4]} sym_hits={len(sym_c)} coll={collection} "
              f"in {(t1 - t0)*1000:.1f} ms")

    # 1) Embedding (Qdrant) — 傳入使用者選的 collection
    t0 = time.time()
    embed_c = qdrant_embedding_recall(query, topk, collection=collection)
    t1 = time.time()
    for c in embed_c:
        c.source = 'embed'
    print(f"[retrieve:embed] got={len(embed_c)} in {(t1 - t0)*1000:.1f} ms")

    # 2) BM25 on local source tree
    t0 = time.time()
    bm25_c = bm25_recall(query, topk)
    t1 = time.time()
    for c in bm25_c:
        c.source = 'bm25'
    print(f"[retrieve:bm25] got={len(bm25_c)} in {(t1 - t0)*1000:.1f} ms")

    # 3) Identifier-based recall (filename / token)
    t0 = time.time()
    id_c = id_recall(query, topk)
    t1 = time.time()
    for c in id_c:
        c.source = 'id'
    print(f"[retrieve:id] got={len(id_c)} in {(t1 - t0)*1000:.1f} ms")

    # Merge & dedupe: keep highest score per file-span
    merged: Dict[str, Candidate] = {}
    for c in (sym_c + embed_c + bm25_c + id_c):
        key = f"{c.file}:{c.start_line}-{c.end_line}"
        if key not in merged or c.score > merged[key].score:
            merged[key] = c

    result = sorted(merged.values(), key=lambda x: x.score, reverse=True)
    print(f"[retrieve] sym={len(sym_c)} embed={len(embed_c)} bm25={len(bm25_c)} id={len(id_c)} -> merged={len(result)}")
    # 顯示合併後 top5 候選摘要，方便比對
    for i, c in enumerate(result[:5]):
        pv = (c.text or "")[:100].replace("\n", " ")
        print(f"[retrieve:top{i}] src={getattr(c,'source','?')} score={getattr(c,'score',0):.4f} "
              f"file={getattr(c,'file','?')} span={getattr(c,'start_line','?')}-{getattr(c,'end_line','?')} "
              f"preview={pv}")
    return result[:topk]
