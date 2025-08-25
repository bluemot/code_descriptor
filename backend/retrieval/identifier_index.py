"""
Lightweight identifier/symbol recall index under .rag_state/symbols/.
"""
import os
from typing import List

from .types import Candidate


def id_recall(query: str, topk: int) -> List[Candidate]:
    """
    Retrieve top-k candidates via identifier/symbol index under .rag_state/symbols.
    Builds or loads index lazily.
    """
    from ..config import load_env
    env = load_env()
    idx_dir = os.getenv('INDEX_STATE_DIR', '.rag_state') + '/symbols'
    os.makedirs(idx_dir, exist_ok=True)
    # TODO: extract symbols from AST/state and perform identifier recall
    # Placeholder: no candidates
    return []
