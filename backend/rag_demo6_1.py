#!/usr/bin/env python3
"""
RAG DEMO ✦ v6.1  (2025‑06‑02)
============================================================
一次到位：
  • 解析 Clang JSON AST → 抽出 Function / Method 片段
  • 產生 File / Project 摘要 (LLM) 並一併向量化
  • 本機或遠端 Qdrant：若 collection 不存在先建索引，否則直接回答
  • 檢索：向量 k=40 → TF‑IDF re‑rank 25 → LLM

依賴：
  pip install sentence_transformers scikit-learn qdrant_client langchain_openai tqdm gzip
"""
from __future__ import annotations
import argparse, pathlib, json, os, sys, itertools, warnings
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import qdrant_client as qc
from qdrant_client import models as qm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gzip, torch # Removed zstandard import
from langchain_openai import OpenAI

# Load configuration from env.txt
from .config import load_env  # noqa: E402
load_env()

QDRANT_URL = os.getenv("QDRANT_URL", None)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", None)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", None)
LLM_KEY = os.getenv("LLM_KEY", None)
LLM_BASE = os.getenv("LLM_BASE", None)
LLM_MODEL = os.getenv("LLM_MODEL", None)

# ──────────────────── 全域設定 ────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDER    = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
COLL        = "proj_ast"

# LLM (OpenAI 介面)
def get_llm():
    """
    Initialize LLM client: Ollama if configured, else OpenAI-compatible.
    """
    if OLLAMA_HOST and OLLAMA_MODEL:
        try:
            from langchain_ollama import Ollama
        except ImportError:
            sys.exit("✗ langchain_ollama not installed for Ollama usage")
        return Ollama(
            model=OLLAMA_MODEL,
            host=OLLAMA_HOST,
        )
    if not (LLM_KEY and LLM_BASE and LLM_MODEL):
        sys.exit("✗ 缺少 LLM_KEY/LLM_BASE/LLM_MODEL (請於 env.txt 中設定)")
    return OpenAI(
        openai_api_key=LLM_KEY,
        openai_api_base=LLM_BASE,
        model_name=LLM_MODEL,
        max_tokens=512,
        temperature=0.2,
        timeout=60,
        stop=["<END>", "Best regards"],
    )

LLM_SUM = get_llm()

warnings.filterwarnings("ignore", category=DeprecationWarning, message="`search` method")

# ──────────────────── AST 解析 ─────────────────────────────

def _open_stream(p: pathlib.Path):
    # if p.suffix == ".zst": # Removed zstandard handling
    #     return zstd.ZstdDecompressor().stream_reader(open(p, "rb"))
    if p.suffix == ".gz":
        return gzip.open(p, "rb") # gzip.open returns a binary file object
    # For uncompressed files (e.g., .json) or any other suffix.
    # If a .zst file is encountered, json.load() will likely fail.
    return open(p, "rb") # open in binary mode is also fine for json.load()


def stream_functions(ast_path: pathlib.Path, src_lines: List[str]):
    """Yield (name, begin, end, snippet) from a single AST."""
    with _open_stream(ast_path) as fp: # fp will be a binary stream
        data = json.load(fp) # json.load can handle a binary fp if it contains UTF-8 text
    stack = [data]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        kind = node.get("kind")
        if kind in ("FunctionDecl", "CXXMethodDecl") and "range" in node:
            name = node.get("name", "anon")
            b = node["range"]["begin"].get("line", 1)
            e = node["range"]["end"].get("line", b)
            if 1 <= b <= len(src_lines):
                snippet = "\n".join(src_lines[b-1 : min(e, b+400)])
                yield name, b, e, snippet
        stack.extend(node.get("inner", []))

# ──────────────────── 摘要 ────────────────────────────────

def summarize(texts: List[str]) -> str:
    joined = "\n---\n".join(texts[:30])  # 避免 prompt 過長
    prompt = (
        "以下是同一檔案 / 專案的程式片段，請用 1~2 句歸納它們的主要功能：\n" + joined)
    return LLM_SUM.invoke(prompt)

# ──────────────────── 建索引 ─────────────────────────────

def build_index(qc_cli: qc.QdrantClient, ast_dir: str):
    ast_root = pathlib.Path(ast_dir)
    # The glob will find *.ast.json, *.ast.json.gz, etc.
    ast_files = [ast_root] if ast_root.is_file() else sorted(ast_root.rglob("*.ast.json*"))
    if not ast_files:
        sys.exit("✗ 找不到 AST 檔")

    docs: List[Dict] = []
    per_file: Dict[str, List[str]] = defaultdict(list)

    for ast_path in tqdm(ast_files, desc="AST"):
        # Determine the base name for the source file
        # Example: file.c.ast.json.gz -> file.c
        # Example: path__to__file.c.ast.json.gz -> path__to__file.c
        original_stem = ast_path.name
        if original_stem.endswith(".ast.json.gz"):
            prefix = original_stem[:-len(".ast.json.gz")]
        elif original_stem.endswith(".ast.json.zst"): # Should ideally not be processed
            prefix = original_stem[:-len(".ast.json.zst")]
        elif original_stem.endswith(".ast.json"):
            prefix = original_stem[:-len(".ast.json")]
        else: # Fallback, might not be correct for all naming schemes
            prefix = ast_path.stem

        # If ast_path.stem was used:
        # file.c.ast.json.gz -> stem is file.c.ast.json -> prefix from this would be file.c.ast
        # Need to be careful with how prefix is derived if complex names are involved.
        # The original code's logic for prefix:
        # prefix = ast_path.stem.split(".ast.json")[0] # foo.ast.json.gz -> stem=foo.ast.json -> prefix = foo
        # prefix = prefix.rsplit(".", 1)[0] # if foo was foo.c -> prefix = foo (incorrect)
                                           # if foo was just foo -> prefix = foo
        # Let's refine prefix extraction based on the full name for robustness
        
        name_parts = ast_path.name.split('.')
        # Assuming format like <name_parts...>.c.ast.json.gz or <name_parts...>.ast.json
        if "ast" in name_parts and "json" in name_parts:
            idx_ast = name_parts.index("ast")
            # prefix is parts before ".ast.json..."
            # e.g., ["path__to__file", "c", "ast", "json", "gz"] -> "path__to__file.c"
            # e.g., ["file", "c", "ast", "json", "gz"] -> "file.c"
            # e.g., ["some_other_file", "ast", "json"] -> "some_other_file" (if no language ext)
            prefix = ".".join(name_parts[:idx_ast])
        else:
            # Fallback if ".ast.json" is not in the name as expected (e.g. single file input)
            prefix = ast_path.stem # This might be foo.ast.json if input is foo.ast.json.gz

        # The original logic for splitting prefix by "__" to reconstruct path
        # This prefix is what's used for src_code path finding.
        # Example: `safe = "__".join(rel.parts)` from gen_ast_all.py
        # So, `prefix` here should correspond to that `safe` name.
        # If ast_path.name is "path__to__file.c.ast.json.gz"
        # Correct prefix should be "path__to__file.c"
        
        # Revised prefix logic based on common output from gen_ast_all.py
        current_name = ast_path.name
        if current_name.endswith(".ast.json.gz"):
            prefix = current_name[:-len(".ast.json.gz")]
        elif current_name.endswith(".ast.json.zst"): # Still check for safety, though not processed
            prefix = current_name[:-len(".ast.json.zst")]
        elif current_name.endswith(".ast.json"):
            prefix = current_name[:-len(".ast.json")]
        else:
            # If it's a direct ast file path like "my_code.ast.json" from args
            if ast_root.is_file() and current_name.endswith(".ast.json"):
                 prefix = current_name[:-len(".ast.json")]
            else:
                # This case might need adjustment based on exact file naming from gen_ast
                # Assuming the part before .ast.json.* is the "safe" name
                print(f"[warn] Unexpected AST filename format: {ast_path.name}, attempting to derive prefix.")
                prefix = ast_path.name.split(".ast.json")[0]


        parts  = prefix.split("__") # This is for reconstructing the original relative path
        src_code: pathlib.Path | None = None

        # (1) 絕對路徑 (if prefix started with "__" from root, parts[0] would be empty)
        if parts[0] == "" and len(parts) > 1: # Check len(parts) > 1 for actual path segments
            # Original path was /a/b/c.c -> __a__b__c.c
            # parts = ["", "a", "b", "c.c"]
            # pathlib.Path("/").joinpath(*parts[1:]) should give /a/b/c.c
            # .with_suffix(".cpp") etc. is problematic if original suffix is in parts[-1]
            potential_path_str = "/".join(parts[1:])
            potential_path = pathlib.Path("/" + potential_path_str)

            # Try with original extension first if it's part of the prefix
            if potential_path.exists() and potential_path.is_file():
                src_code = potential_path
            else:
                # Try common C/C++ extensions if original extension was stripped or different
                for ext in (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"):
                    cand = potential_path.with_suffix(ext)
                    if cand.exists():
                        src_code = cand; break
        
        # (2) 工作樹尋找
        if not src_code:
            # prefix is like "dir__file.c" or "file.c"
            # parts = ["dir", "file.c"] or ["file.c"]
            path_from_parts = pathlib.Path(*parts) # forms dir/file.c or file.c
            for root in (pathlib.Path.cwd(), pathlib.Path("..")): # Check current and parent dir
                # Try with the extension embedded in 'prefix' first
                cand_exact = root / path_from_parts
                if cand_exact.exists() and cand_exact.is_file():
                    src_code = cand_exact
                    break
                
                if src_code: break
                # If not found, try changing extension (path_from_parts might be "file" if ext was ".c")
                # This part assumes `prefix` might not have the original extension correctly.
                # The `prefix` from `gen_ast_all.py` should be like `path__to__file.c`
                # So `path_from_parts` would be `path/to/file.c`
                # The loop below is more of a fallback if `path_from_parts` is extensionless
                if '.' not in path_from_parts.name: # If it looks extensionless
                    for ext in (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"):
                        cand = root / path_from_parts.with_suffix(ext)
                        if cand.exists():
                            src_code = cand; break
                    if src_code: break
        
        if not src_code:
            print(f"[warn] Source code not found for AST: {ast_path.name} (derived prefix: {prefix})")
            continue

        src_lines = src_code.read_text(errors="ignore").splitlines()
        func_snips: List[str] = []
        for name, b, e, snippet in stream_functions(ast_path, src_lines):
            # ID includes original source suffix, e.g. path__to__file.c::my_function
            docs.append({"id": f"{prefix}::{name}", "text": snippet})
            func_snips.append(snippet)
            per_file[prefix].append(snippet)

        # 檔案級摘要
        if func_snips:
            file_sum = summarize(func_snips)
            # ID for file summary, e.g. path__to__file.c::__file_summary__
            docs.append({"id": f"{prefix}::__file_summary__", "text": file_sum})
            per_file[prefix].append(file_sum)

    if not docs:
        sys.exit("✗ 沒抓到任何可索引的程式片段，請確認 AST 與原始碼路徑")

    # 專案級摘要
    project_sum = summarize(list(itertools.chain.from_iterable(per_file.values())))
    docs.append({"id": "__project_summary__", "text": project_sum})

    vecs = EMBEDDER.encode([d["text"] for d in docs], batch_size=256,
                            convert_to_numpy=True, show_progress_bar=True)
    
    vector_size = EMBEDDER.get_sentence_embedding_dimension()
    if vecs.shape[1] != vector_size: # Should not happen with bge-large-zh-v1.5 (1024)
        print(f"[warn] Vector dimension mismatch. Expected {vector_size}, got {vecs.shape[1]}. Will use {vecs.shape[1]}.")

    if qc_cli.collection_exists(COLL):
        print(f"[info] Collection '{COLL}' exists, deleting and recreating.")
        qc_cli.delete_collection(COLL)
    
    qc_cli.create_collection(COLL, vectors_config=qm.VectorParams(size=int(vecs.shape[1]),
                                                                  distance=qm.Distance.COSINE))
    
    # Generate integer IDs for Qdrant if not using UUIDs or specific string IDs
    # Qdrant SDK can take integer IDs directly.
    # The 'id' field in 'docs' is for human readability/debugging, Qdrant gets integer IDs here.
    qdrant_ids = list(range(len(docs)))
    
    # Prepare payloads. Qdrant payload should not contain the 'id' we made for docs.
    # It will store the text and any other metadata we want to retrieve.
    payloads = [{"text": d["text"], "original_id": d["id"]} for d in docs]


    qc_cli.upload_collection(COLL, 
                             vectors=vecs,
                             payload=payloads,
                             ids=qdrant_ids,
                             batch_size=256) # Added batch_size for potentially large uploads
    print(f"[build] 完成，共 {len(docs)} 向量")

# ──────────────────── 檢索 ───────────────────────────────

def search(client: qc.QdrantClient, q: str, k: int = 40) -> List[str]:
    q_vec = EMBEDDER.encode(q, convert_to_numpy=True)
    # Ensure q_vec is 1D for Qdrant client if it's not already
    query_vector = q_vec.tolist()
    if isinstance(query_vector[0], list): # if encode returns a list of lists for single query
        query_vector = query_vector[0]

    hits = client.search(collection_name=COLL, query_vector=query_vector, limit=k, with_payload=True)
    if not hits:
        return []
    
    texts = []
    seen_texts = set() # Avoid duplicates from Qdrant if any (should not happen with distinct payloads)
    for h in hits:
        if h.payload and "text" in h.payload:
            if h.payload["text"] not in seen_texts:
                texts.append(h.payload["text"])
                seen_texts.add(h.payload["text"])
        else:
            print(f"[warn] Hit with no payload or no text: {h.id}")


    if not texts:
        return []

    # TF-IDF Re-ranking (original logic)
    # To re-rank, we need to re-encode the retrieved texts if we want semantic similarity for re-ranking
    # Or use TF-IDF as originally intended.
    # The original code re-encodes texts and uses cosine_similarity.
    
    # Option 1: Re-rank based on cosine similarity with the query_vector (already done by Qdrant)
    # The hits are already sorted by similarity by Qdrant.
    # The original re-ranking step might be redundant or aiming for a different re-ranking logic.
    # "sub_vecs = EMBEDDER.encode(texts, convert_to_numpy=True)"
    # "scores = cosine_similarity(q_vec.reshape(1,-1), sub_vecs)[0]"
    # This re-calculates scores for the texts Qdrant already found.
    # If Qdrant's COSINE distance is used, this re-ranking should yield a similar order unless
    # there are precision differences or if `texts` were manipulated.
    # For simplicity, let's trust Qdrant's order for the top-k, then slice to 25.
    # Or, implement the TF-IDF re-ranking as the comment suggested was the goal.
    # The current code is doing semantic re-ranking, not TF-IDF.
    
    # Keeping the original re-ranking logic for now:
    sub_vecs = EMBEDDER.encode(texts, convert_to_numpy=True) # Re-encode the K texts
    scores = cosine_similarity(q_vec.reshape(1,-1), sub_vecs)[0]
    order = scores.argsort()[::-1][:25] # Get top 25 from the K=40
    
    return [texts[i] for i in order]


def answer(client: qc.QdrantClient, question: str) -> str:
    """
    Perform RAG-based answer generation for the question using the Qdrant client.
    Returns the generated answer text.
    """
    ctx = search(client, question)
    if not ctx:
        return "(no context)"
    prompt = (
        "你是一位資深 C/C++ 網路驅動工程師，以下程式片段與摘要可作為上下文：\n" +
        "\n---\n".join(ctx) +
        f"\n\n{question}\n不要加任何客套、署名，回答完輸出 <END>。\nA:")
    return LLM_SUM.invoke(prompt)

# ──────────────────── CLI ───────────────────────────────

# ... other imports and code ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", help="AST 路徑 (檔或資料夾 containing .ast.json.gz or .ast.json files)")
    ap.add_argument("--ask", help="提問")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", QDRANT_URL), help=f"Qdrant URL (default: {QDRANT_URL} or QDRANT_URL env var)")
    ap.add_argument("--qdrant-key", default=os.getenv("QDRANT_KEY"), help="Qdrant API Key (optional, from QDRANT_KEY env var)")
    args = ap.parse_args()

    qdrant_url_to_use = args.qdrant_url
    qdrant_api_key_to_use = args.qdrant_key if args.qdrant_key else None # Ensure None if empty

    if not qdrant_url_to_use: # If empty string from arg and env
        print("[info] QDRANT_URL not set, using local Qdrant (path='rag_demo_qdrant').")
        qc_cli = qc.QdrantClient(path="rag_demo_qdrant")
    else:
        print(f"[info] Connecting to Qdrant at {qdrant_url_to_use}")
        qc_cli = qc.QdrantClient(url=qdrant_url_to_use, api_key=qdrant_api_key_to_use)
    
    # Check Qdrant connection by trying a simple operation
    try:
        qc_cli.get_collections() # This will attempt to connect and list collections
        print("[info] Qdrant connection successful (able to list collections).")
    except Exception as e:
        # More specific error checking could be added here if needed
        # e.g. for qdrant_client.http.exceptions.ResponseHandlingException or ConnectionError
        sys.exit(f"✗ Failed to connect/communicate with Qdrant at {qdrant_url_to_use if qdrant_url_to_use else 'local path'}: {e}")


    if args.build:
        build_index(qc_cli, args.build)

    if args.ask:
        try:
            collection_info = qc_cli.get_collection(COLL)
            print(f"[info] Found collection '{COLL}' with {collection_info.points_count} points.")
        except Exception: 
            print(f"[info] Collection '{COLL}' does not exist.")
            if not args.build:
                 sys.exit(f"✗ Collection '{COLL}' not found. Please run with --build <ast_dir> first.")
            # If --build was also specified, build_index would have run already or will run.
            # If collection still not found after build (if build was specified and ran), then it's an issue.
            # For simplicity, if --ask is used and collection is missing, error out if --build wasn't *also* given.

        answer(qc_cli, args.ask)

if __name__ == "__main__":
    main()
