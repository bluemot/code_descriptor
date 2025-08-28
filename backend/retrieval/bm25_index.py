"""
Lightweight BM25-like recall using TF-IDF on source files under project directory.
Index stored under .rag_state/inverted/ as pickle.
"""
import os
import pathlib
import pickle
from hashlib import blake2b
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .types import Candidate
from ..config import load_env

_bm25_index: dict | None = None

def _ensure_index():
    """Load or build the TF-IDF index for BM25 recall."""
    global _bm25_index
    if _bm25_index is not None:
        return
    load_env()
    root = os.getenv('PROJECT_DIR', None) or os.getcwd()
    idx_dir = pathlib.Path(os.getenv('INDEX_STATE_DIR', '.rag_state')) / 'inverted'
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_file = idx_dir / 'bm25_index.pkl'
    if idx_file.exists():
        with open(idx_file, 'rb') as f:
            _bm25_index = pickle.load(f)
        return

    # build index: each file as document
    files = []
    docs = []
    for p in pathlib.Path(root).rglob('*'):
        if p.suffix in ('.c', '.cpp', '.h', '.hpp', '.py'):
            try:
                txt = p.read_text(errors='ignore')
            except Exception:
                continue
            files.append(p)
            docs.append(txt)

    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9_]+", max_features=int(os.getenv('BM25_MAX_FEATURES', '10000')))
    mat = vec.fit_transform(docs)
    _bm25_index = {'vectorizer': vec, 'matrix': mat, 'files': files}
    with open(idx_file, 'wb') as f:
        pickle.dump(_bm25_index, f)

def bm25_recall(query: str, topk: int) -> List[Candidate]:
    """
    Recall top-k candidates by TF-IDF cosine similarity on file contents.
    """
    _ensure_index()
    vec = _bm25_index['vectorizer']
    mat = _bm25_index['matrix']
    files = _bm25_index['files']

    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat)[0]
    idx = np.argsort(sims)[::-1][:topk]
    cands: List[Candidate] = []
    for i in idx:
        p = files[i]
        score = float(sims[i])
        # snippet first 200 lines
        try:
            lines = p.read_text(errors='ignore').splitlines()
        except Exception:
            continue
        snippet = '\n'.join(lines[:200])
        cid = int.from_bytes(blake2b(f"{p}:{1}-{min(len(lines),200)}".encode(), digest_size=8).digest(), 'big')
        cands.append(Candidate(
            id=str(cid), file=str(p), function=None, signature=None,
            start_line=1, end_line=min(len(lines),200), scope='file',
            score=score, source='bm25', text=snippet, meta={}
        ))
    print(f"[retrieve:bm25] returned={len(cands)} topk={topk}")
    # 顯示 BM25 前幾筆命中，便於判斷是否都落在正確模組
    try:
        for i, c in enumerate(cands[:5]):
            pv = (c.text or "")[:100].replace("\n", " ")
            print(f"[bm25:top{i}] score={getattr(c,'score',0):.4f} file={getattr(c,'file','?')} preview={pv}")
    except Exception:
        pass
    return cands
