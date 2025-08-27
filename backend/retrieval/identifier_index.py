"""
Lightweight identifier/symbol recall index based on file-name tokens.
"""
import os
import re
import pathlib
from hashlib import blake2b
from typing import List

from .types import Candidate
from ..config import load_env


_symbol_index: List[pathlib.Path] | None = None

def _ensure_index():
    global _symbol_index
    if _symbol_index is not None:
        return
    load_env()
    root = os.getenv('PROJECT_DIR', None) or os.getcwd()
    paths: List[pathlib.Path] = []
    for p in pathlib.Path(root).rglob('*'):
        if p.suffix in ('.c', '.cpp', '.h', '.hpp', '.py'):
            paths.append(p)
    _symbol_index = paths


def id_recall(query: str, topk: int) -> List[Candidate]:
    """
    Retrieve top-k candidates by matching query tokens to file/path tokens.
    Builds an index of source files and returns files whose name tokens intersect query tokens.
    """
    _ensure_index()
    files = _symbol_index or []
    # tokenize query
    q_tokens = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
    matches: List[tuple[Candidate, float]] = []
    for p in files:
        name_tokens = set(re.findall(r"[A-Za-z0-9_]+", p.stem.lower()))
        common = q_tokens & name_tokens
        if not common:
            continue
        # score by token overlap ratio
        score = len(common) / max(len(name_tokens), 1)
        # read snippet (first 200 lines)
        try:
            lines = p.read_text(errors='ignore').splitlines()
        except Exception:
            continue
        snippet = '\n'.join(lines[:200])
        # build candidate
        key = f"{p}:{1}-{min(len(lines),200)}"
        cid = int.from_bytes(blake2b(key.encode(), digest_size=8).digest(), 'big')
        cand = Candidate(
            id=str(cid), file=str(p), function=None, signature=None,
            start_line=1, end_line=min(len(lines),200), scope='file',
            score=score, source='id', text=snippet, meta={}
        )
        matches.append((cand, score))
    # sort and take topk
    matches.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in matches[:topk]]
