"""
Hybrid recall combining Embedding (Qdrant), BM25, and Identifier channels.
"""
import os
import time
from typing import List, Dict

from .types import Candidate

from ..config import load_env


def hybrid_recall(
    query: str,
    *,
    topk_embed: int,
    topk_bm25: int,
    topk_id: int,
    max_cand: int,
) -> List[Candidate]:
    """
    Perform hybrid recall: Qdrant embedding, BM25 inverted index, and identifier match.
    Merge, dedupe overlapping line ranges, sort by score, return up to max_cand.
    """
    env = load_env()
    start = time.time()
    cands: List[Candidate] = []

    # 1. Embedding recall (via Qdrant)
    try:
        from ..rag_service import qdrant_embedding_recall

        embed_c = qdrant_embedding_recall(query, topk_embed)
    except ImportError:
        embed_c = []
        print('[deg] embedding_unavailable')
    for c in embed_c:
        c.source = 'embed'
    cands.extend(embed_c)

    # 2. BM25 recall
    try:
        from .bm25_index import bm25_recall

        bm25_c = bm25_recall(query, topk_bm25)
    except ImportError:
        bm25_c = []
        print('[deg] bm25_unavailable')
    for c in bm25_c:
        c.source = 'bm25'
    cands.extend(bm25_c)

    # 3. Identifier recall
    try:
        from .identifier_index import id_recall

        id_c = id_recall(query, topk_id)
    except ImportError:
        id_c = []
        print('[deg] id_unavailable')
    for c in id_c:
        c.source = 'id'
    cands.extend(id_c)

    # Merge & dedupe: if same file & overlapping lines, keep highest score
    merged: Dict[str, Candidate] = {}
    for c in cands:
        key = f"{c.file}:{c.start_line}-{c.end_line}"
        if key not in merged or c.score > merged[key].score:
            merged[key] = c

    result = sorted(merged.values(), key=lambda x: x.score, reverse=True)
    took = time.time() - start
    print(f'[retrieve] embed={len(embed_c)} bm25={len(bm25_c)} id={len(id_c)} -> merged={len(result)} took={took:.3f}s')
    return result[:max_cand]
