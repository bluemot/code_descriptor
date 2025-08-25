"""
On-demand AST-aware function-level deepening and embedding upsert.
"""
import os
import time
from typing import List

from .types import DeepenReport
from ..config import load_env


def on_demand_deepen(project: str, files: List[str], env=None) -> DeepenReport:
    """
    Deepen given files at function-level, generate summaries and embeddings, upsert to vector store.
    """
    env = env or load_env()
    start = time.time()
    enabled = int(os.getenv('DEEPEN_ENABLE', '1'))
    top_n = int(os.getenv('DEEPEN_TOP_FILES', '3'))
    max_lines = int(os.getenv('MAX_FUNC_LINES', '500'))
    target = int(os.getenv('FUNC_CHUNK_TARGET', '220'))
    overlap = int(os.getenv('FUNC_CHUNK_OVERLAP', '12'))

    if not enabled:
        print(f"[deepen] enabled=0, skip deepening")
        return DeepenReport(
            files=[], new_functions=0, new_points=0,
            took_ms=0.0, cached_hit=True
        )

    # Select top files for deepening
    to_process = files[:top_n]
    print(f"[deepen] enabled=1 files={to_process} max_lines={max_lines} target={target} overlap={overlap}")
    # TODO: implement AST-aware function splitting, embedding upsert, summary generation
    took = (time.time() - start) * 1000
    report = DeepenReport(
        files=to_process,
        new_functions=0,
        new_points=0,
        took_ms=took,
        cached_hit=False,
    )
    print(f"[deepen] modules upserted funcs={report.new_functions} points={report.new_points} took={report.took_ms:.0f}ms")
    return report
