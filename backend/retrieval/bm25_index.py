"""
Lightweight BM25 inverted index stored under .rag_state/inverted/.
"""
import os
import json
from typing import List

from .types import Candidate
from ..config import load_env


def bm25_recall(query: str, topk: int) -> List[Candidate]:
    """
    Retrieve top-k candidates via BM25 inverted index under .rag_state/inverted.
    Builds or loads index lazily.
    """
    from ..config import load_env
    env = load_env()
    idx_dir = os.getenv('INDEX_STATE_DIR', '.rag_state') + '/inverted'
    os.makedirs(idx_dir, exist_ok=True)
    # TODO: load or build inverted index, run BM25 recall
    # Placeholder: no candidates
    return []
