"""
Context expansion and packing by lines and token limits.
"""
import os
from typing import List
from ..config import load_env

from .types import Candidate, PackedContext


def expand_neighbors(candidates: List[Candidate], expand: int | None = None) -> List[Candidate]:
    """
    For function-level candidates, include neighboring chunks ±expand (default from env).
    """
    load_env()
    exp = expand if expand is not None else int(os.getenv('NEIGHBOR_EXPAND', '1'))
    # TODO: implement AST-based lookup for same function neighbors ±exp
    return candidates


def pack_by_lines(candidates: List[Candidate], token_limit: int | None = None) -> PackedContext:
    """
    Merge overlapping line ranges per file, trim to token_limit, return PackedContext.
    """
    load_env()
    limit = token_limit if token_limit is not None else int(os.getenv('PACK_TOKEN_LIMIT', '1400'))
    # naive sort and merge with token limit
    items = sorted(candidates, key=lambda c: (c.file, c.start_line))
    merged = []
    total = 0
    files = set()
    for c in items:
        files.add(c.file)
        # approximate token count as lines range
        total += c.end_line - c.start_line + 1
        if total > limit:
            break
        merged.append(c)
    files_list = list(files)
    print(f"[pack] neighbors=±{os.getenv('NEIGHBOR_EXPAND', '1')} -> packed_tokens={total} files={files_list}")
    return PackedContext(
        items=merged,
        total_tokens=total,
        files_covered=files_list,
        notes=None,
    )
