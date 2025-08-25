"""
Core data types for retrieval workflow: Candidate, PackedContext, DeepenReport.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Candidate:
    """
    A retrieval candidate item from embedding, BM25, or identifier channels.
    """
    id: str
    file: str
    function: str | None
    signature: str | None
    start_line: int
    end_line: int
    scope: str
    score: float
    source: str  # 'embed' | 'bm25' | 'id'
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PackedContext:
    """
    A packed context consisting of multiple candidate snippets,
    trimmed to a token limit for prompt construction.
    """
    items: List[Candidate]
    total_tokens: int
    files_covered: List[str]
    notes: str | None = None


@dataclass
class DeepenReport:
    """
    Report of on-demand deepening run, including cache hits and new upserts.
    """
    files: List[str]
    new_functions: int
    new_points: int
    took_ms: float
    cached_hit: bool
