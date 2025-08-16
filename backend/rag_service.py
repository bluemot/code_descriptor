"""
Wrapper for Qdrant indexing and question-answering using the rag_demo6_1 script.
"""
from .config import load_env  # noqa: E402
load_env()

import os
from pathlib import Path


def build_index(ast_dir: Path) -> None:
    """
    Build or update the Qdrant index based on AST files in ast_dir.
    """
    try:
        import qdrant_client as qc
        from .rag_demo6_1 import build_index as rd_build_index, QDRANT_URL

        url = os.getenv("QDRANT_URL", QDRANT_URL)
        key = os.getenv("QDRANT_KEY", None)
        if url:
            client = qc.QdrantClient(url=url, api_key=key)
        else:
            client = qc.QdrantClient(path="rag_demo_qdrant")
        rd_build_index(client, str(ast_dir))
    except Exception:
        # Swallow any errors during indexing
        pass


def answer_question(question: str) -> str:
    """
    Retrieve and generate an answer for the given question using the RAG service.
    """
    try:
        import qdrant_client as qc
        from .rag_demo6_1 import answer as rd_answer, QDRANT_URL

        url = os.getenv("QDRANT_URL", QDRANT_URL)
        key = os.getenv("QDRANT_KEY", None)
        if url:
            client = qc.QdrantClient(url=url, api_key=key)
        else:
            client = qc.QdrantClient(path="rag_demo_qdrant")
        return rd_answer(client, question)
    except Exception:
        # On error, return empty answer
        return ""
