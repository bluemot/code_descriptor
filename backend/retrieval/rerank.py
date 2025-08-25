"""
Rerank and fuse candidate scores using a cross-encoder and channel weights.
"""
import os
import time
from typing import List

from .types import Candidate
from ..config import load_env


def rerank_and_fuse(query: str, candidates: List[Candidate]) -> List[Candidate]:
    """
    Rerank candidates with a cross-encoder and fuse scores by configured weights.
    Degrade to embed-only scores if reranker unavailable.
    """
    env = load_env()
    start = time.time()
    model = os.getenv('RERANK_MODEL', 'bge-reranker-base')
    top_k = int(os.getenv('RERANK_TOP_K', '36'))
    weights = {
        'w_rerank': float(os.getenv('FUSE_W_RERANK', '0.55')),
        'w_embed': float(os.getenv('FUSE_W_EMBED', '0.30')),
        'w_bm25': float(os.getenv('FUSE_W_BM25', '0.10')),
        'w_signal': float(os.getenv('FUSE_W_SIGNAL', '0.05')),
    }

    # Limit to top_k for reranking
    to_rerank = candidates[:top_k]
    # Compute rerank scores
    try:
        from ..rag_service import reranker_score

        rerank_scores = reranker_score(query, to_rerank, model)
    except ImportError:
        rerank_scores = {c.id: c.score for c in to_rerank}
        print('[deg] reranker_unavailable')

    fused: List[Candidate] = []
    for c in to_rerank:
        r = rerank_scores.get(c.id, 0.0)
        e = c.score if c.source == 'embed' else 0.0
        b = c.score if c.source == 'bm25' else 0.0
        s = c.meta.get('signal', 0.0)
        final = (
            weights['w_rerank'] * r +
            weights['w_embed'] * e +
            weights['w_bm25'] * b +
            weights['w_signal'] * s
        )
        c.meta['score_fused'] = final
        fused.append(c)

    result = sorted(fused, key=lambda x: x.meta['score_fused'], reverse=True)[:top_k]
    took = time.time() - start
    print(f'[rerank] took={took:.3f}s -> top_k={len(result)}')
    return result
