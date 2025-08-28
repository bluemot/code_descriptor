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
import argparse, pathlib, json, os, sys, itertools, warnings, time
from collections import defaultdict
from hashlib import blake2b
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
# Embedding setup: choose between SentenceTransformer or OllamaEmbeddings based on environment
# Default to a smaller model for faster embed speed, override via EMBED_MODEL env
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-small-en-v1.5")

class EmbedderWrapper:
    """Wrapper to unify interface between SentenceTransformer and OllamaEmbeddings"""
    def __init__(self, embedder):
        self._embedder = embedder

    def encode(self, texts, batch_size=None, convert_to_numpy=False, show_progress_bar=False):
        # texts: list[str] for documents or single str for query
        if isinstance(texts, list):
            vectors = self._embedder.embed_documents(texts)
            arr = np.array(vectors)
            return arr if convert_to_numpy else vectors
        # single text query
        vector = self._embedder.embed_query(texts)
        arr = np.array(vector)
        return arr if convert_to_numpy else vector

    def get_sentence_embedding_dimension(self):
        try:
            return self._embedder.get_sentence_embedding_dimension()
        except AttributeError:
            vec = self._embedder.embed_query("")
            return len(vec)

def get_embedder():
    # Prefer OllamaEmbeddings if configured, else use SentenceTransformer
    cuda = torch.cuda.is_available()
    use_fp16 = cuda
    device = "cuda" if cuda else "cpu"
    # If Ollama is configured, prefer it; else use SentenceTransformer
    if OLLAMA_HOST and EMBED_MODEL:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            sys.exit(
                "✗ langchain_ollama installed but OllamaEmbeddings class not found, please upgrade/downgrade package"
            )

        embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
        print(f"[info] EMBED_MODEL={EMBED_MODEL} via OllamaEmbeddings host={OLLAMA_HOST} device={device} batch={EMBED_BATCH} fp16={use_fp16}")
        return EmbedderWrapper(embed)

    # Fallback to SentenceTransformer
    print(f"[info] EMBED_MODEL={EMBED_MODEL} via SentenceTransformer device={device} batch={EMBED_BATCH} fp16={use_fp16}")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    if use_fp16:
        model = model.half()
    return model

## performance knobs (env-overridable) must come before embedder init
MAX_FUNC_LINES_BUILD = int(os.getenv('MAX_FUNC_LINES_BUILD', '80'))
MAX_BYTES_PER_SNIPPET = int(os.getenv('MAX_BYTES_PER_SNIPPET', '4096'))
MAX_POINTS_PER_FILE = int(os.getenv('MAX_POINTS_PER_FILE', '800'))
print(f"[config] MAX_FUNC_LINES_BUILD={MAX_FUNC_LINES_BUILD}, MAX_BYTES_PER_SNIPPET={MAX_BYTES_PER_SNIPPET}, MAX_POINTS_PER_FILE={MAX_POINTS_PER_FILE}")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "192"))              # texts per embedding call
QDRANT_UPSERT_CHUNK = int(os.getenv("QDRANT_UPSERT_CHUNK", "1024"))  # points per upsert request
FILE_SUM_SNIPPETS = int(os.getenv("FILE_SUM_SNIPPETS", "12"))  # snippets per file summary
INDEX_STATE_DIR = os.getenv("INDEX_STATE_DIR", ".rag_state")   # local checkpoint directory
RESUME = int(os.getenv("RESUME", "1"))                         # 1=resume from checkpoints

# Embedding entry point
EMBEDDER = get_embedder()

# batching knobs (env-overridable)
# Increase batch size for throughput; fall back on OOM


# ──────────────────── Checkpoint utilities ────────────────────
def _safe_mkdir(p: pathlib.Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _state_key_for_file(ast_path: pathlib.Path) -> str:
    from hashlib import blake2b
    return blake2b(str(ast_path).encode("utf-8"), digest_size=8).hexdigest()

def _state_path(ast_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(INDEX_STATE_DIR) / f"{_state_key_for_file(ast_path)}.json"

def _done_path(ast_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(INDEX_STATE_DIR) / f"{_state_key_for_file(ast_path)}.done"

def _load_checkpoint(ast_path: pathlib.Path) -> Dict:
    sp = _state_path(ast_path)
    if sp.exists():
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_checkpoint(ast_path: pathlib.Path, data: Dict):
    _safe_mkdir(pathlib.Path(INDEX_STATE_DIR))
    sp = _state_path(ast_path)
    data = dict(data)
    data["ast_path"] = str(ast_path)
    data["ts"] = time.time()
    sp.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")

def _mark_done(ast_path: pathlib.Path):
    _safe_mkdir(pathlib.Path(INDEX_STATE_DIR))
    _state_path(ast_path).unlink(missing_ok=True)
    _done_path(ast_path).write_text("done", encoding="utf-8")

def _is_done(ast_path: pathlib.Path) -> bool:
    return _done_path(ast_path).exists()

# LLM (OpenAI 介面)
def get_llm():
    if OLLAMA_HOST and OLLAMA_MODEL:
        try:
            from langchain_ollama import OllamaLLM
        except ImportError:
            sys.exit("✗ langchain_ollama installed but OllamaLLM class not found, please upgrade/downgrade package")

        return OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
        )

LLM_SUM = get_llm()

warnings.filterwarnings("ignore", category=DeprecationWarning, message="`search` method")

# ──────────────────── Qdrant 輔助 ─────────────────────────
def _stable_id(s: str) -> int:
    """Deterministic 64-bit integer ID from a string."""
    return int.from_bytes(blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")

def _ensure_collection(cli: qc.QdrantClient, coll_name: str, dim: int):
    if not cli.collection_exists(coll_name):
        try:
            cli.create_collection(
                coll_name,
                vectors_config=qm.VectorParams(size=int(dim), distance=qm.Distance.COSINE),
            )
        except Exception as e:
            print(f"[warning] Could not create collection {coll_name}: {e}")
            # double-check
            if not cli.collection_exists(coll_name):
                raise RuntimeError(f"Collection {coll_name} not usable, aborting.")
    else:
        print(f"[info] Collection {coll_name} already exists, will upsert data instead of creating.")

def _upsert_docs(cli: qc.QdrantClient, coll_name: str, docs: list, *,
                 start_chunk: int = 0,
                 on_chunk_done=None):
    """
    Batched embed + upsert.
    - start_chunk: resume index (0-based). Each chunk size is QDRANT_UPSERT_CHUNK.
    - on_chunk_done: optional callback(idx, total_chunks) after each upsert returns.
    """
    total = len(docs)
    num_chunks = (total + QDRANT_UPSERT_CHUNK - 1) // QDRANT_UPSERT_CHUNK
    print(f"[upsert] preparing {total} docs (chunk={QDRANT_UPSERT_CHUNK}, embed_batch={EMBED_BATCH}, start_chunk={start_chunk}/{num_chunks})")
    if start_chunk >= num_chunks:
        return
    # loop from start_chunk
    for chunk_idx in range(start_chunk, num_chunks):
        j = chunk_idx * QDRANT_UPSERT_CHUNK
        sub_docs = docs[j:j+QDRANT_UPSERT_CHUNK]
        t0 = time.time()
        sub_vecs = EMBEDDER.encode(
            [d["text"] for d in sub_docs],
            convert_to_numpy=True,
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
        )
        t1 = time.time()
        points = []
        for i, d in enumerate(sub_docs):
            points.append(
                qm.PointStruct(
                    id=_stable_id(d["id"]),
                    vector=sub_vecs[i].tolist(),
                    payload={
                        "text": d["text"],
                        "original_id": d["id"],
                        "scope": d.get("scope", "func"),
                        "file": d.get("file"),
                    },
                )
            )
        print(f"[upsert] encoded {len(sub_docs)} in {t1 - t0:.1f}s; sending {len(points)} points to Qdrant… (chunk {chunk_idx+1}/{num_chunks})")
        t2 = time.time()
        cli.upsert(collection_name=coll_name, points=points, wait=False)
        t3 = time.time()
        print(f"[upsert] upsert returned in {t3 - t2:.2f}s")
        if on_chunk_done:
            try:
                on_chunk_done(chunk_idx, num_chunks)
            except Exception as e:
                print(f"[warn] on_chunk_done callback failed: {e}")

def get_coll_name(ast_dir: str) -> str:
    """
    Determine Qdrant collection name based on the AST directory or file name.
    Returns 'proj_ast_<name>' where <name> is the basename of ast_dir without AST-related suffixes.
    """
    base = pathlib.Path(ast_dir).name
    # strip common AST filename suffixes
    for suf in (".ast.json.gz", ".ast.json.zst", ".ast.json"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return f"proj_ast_{base}"

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
    # performance tunables
    _MAX_FUNC_LINES_BUILD = int(os.getenv('MAX_FUNC_LINES_BUILD', '80'))
    _SKIP_TRIVIAL_FUNCS = int(os.getenv('SKIP_TRIVIAL_FUNCS', '1'))
    _MAX_BYTES_PER_SNIPPET = int(os.getenv('MAX_BYTES_PER_SNIPPET', '4096'))
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
                length = e - b + 1
                if _SKIP_TRIVIAL_FUNCS and length < 3:
                    continue
                end_line = min(e, b + _MAX_FUNC_LINES_BUILD)
                snippet = "\n".join(src_lines[b-1:end_line])
                if len(snippet.encode('utf-8')) > _MAX_BYTES_PER_SNIPPET:
                    snippet = snippet.encode('utf-8')[:_MAX_BYTES_PER_SNIPPET].decode('utf-8', errors='ignore')
                yield name, b, e, snippet
        stack.extend(node.get("inner", []))

# ──────────────────── 摘要 ────────────────────────────────

def summarize(texts: List[str]) -> str:
    joined = "\n---\n".join(texts[:30])  # 避免 prompt 過長
    prompt = (
        "以下是同一檔案 / 專案的程式片段，請用 1~2 句歸納它們的主要功能：\n" + joined)
    return LLM_SUM.invoke(prompt)

# ──────────────────── 建索引 ─────────────────────────────

def build_index(qc_cli: qc.QdrantClient, ast_dir: str, coll_name: str):
    ast_root = pathlib.Path(ast_dir)
    # The glob will find *.ast.json, *.ast.json.gz, etc.
    ast_files = [ast_root] if ast_root.is_file() else sorted(ast_root.rglob("*.ast.json*"))
    if not ast_files:
        sys.exit("✗ 找不到 AST 檔")

    print(f"[build] Found {len(ast_files)} AST file(s) in {ast_dir}")
    per_file: Dict[str, List[str]] = defaultdict(list)

    # Ensure collection exists with the correct dimension
    vector_size = EMBEDDER.get_sentence_embedding_dimension()
    _ensure_collection(qc_cli, coll_name, vector_size)

    for idx, ast_path in enumerate(ast_files, 1):
        print(f"[build] Processing AST file {idx}/{len(ast_files)}: {ast_path}")
        # Resume: skip finished files entirely
        if RESUME and _is_done(ast_path):
            print(f"[build] Skipping (done): {ast_path}")
            continue
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
        docs_batch: List[Dict] = []
        for name, b, e, snippet in stream_functions(ast_path, src_lines):
            # per-function point
            docs_batch.append({"id": f"{prefix}::{name}", "text": snippet, "scope": "func", "file": prefix})
            func_snips.append(snippet)
            per_file[prefix].append(snippet)

        # 檔案級摘要（有內容才做）
        if func_snips:
            print(f"[build] Summarizing file-level snippets for {prefix} ({len(func_snips)} items)")
            file_sum = summarize(func_snips)
            print(f"[build] Generated file summary for {prefix}")
            docs_batch.append({"id": f"{prefix}::__file_summary__", "text": file_sum, "scope": "file", "file": prefix})
            per_file[prefix].append(file_sum)

        # ⬅️ 立刻上傳本檔案的 points（支援中斷續跑）
        if docs_batch:
            # enforce per-file point limit
            if len(docs_batch) > MAX_POINTS_PER_FILE:
                # sort by snippet length desc, then keyword priority
                keywords = ["init", "encode", "decode", "probe", "aes", "ccm", "tx", "rx"]
                def score_item(d):
                    txt = d.get('text','')
                    s = len(txt)
                    for kw in keywords:
                        if kw in txt.lower(): s += 10000
                    return s

                docs_batch.sort(key=score_item, reverse=True)
                docs_batch = docs_batch[:MAX_POINTS_PER_FILE]
            print(f"[build] Upserting {len(docs_batch)} point(s) for {prefix} (batch embed {EMBED_BATCH}, upsert chunk {QDRANT_UPSERT_CHUNK})")
            # load checkpoint for this file
            start_chunk = 0
            if RESUME:
                ckpt = _load_checkpoint(ast_path)
                start_chunk = int(ckpt.get("completed_chunks", 0))
                if start_chunk:
                    print(f"[build] Resuming {ast_path} from chunk {start_chunk}")
            def _on_done(chunk_idx, total_chunks):
                # mark completed_chunks as chunk_idx+1
                _save_checkpoint(ast_path, {
                    "completed_chunks": int(chunk_idx + 1),
                    "total_chunks": int(total_chunks),
                    "qdrant_chunk": int(QDRANT_UPSERT_CHUNK),
                })
            _upsert_docs(qc_cli, coll_name, docs_batch, start_chunk=start_chunk, on_chunk_done=_on_done)
            # if finished all chunks, mark done
            fin = _load_checkpoint(ast_path)
            if int(fin.get("completed_chunks", 0)) >= ((len(docs_batch) + QDRANT_UPSERT_CHUNK - 1)//QDRANT_UPSERT_CHUNK):
                _mark_done(ast_path)
                print(f"[build] Marked done: {ast_path}")

    # 若整個專案沒有可索引片段，直接退出
    if not per_file:
        sys.exit("✗ 沒抓到任何可索引的程式片段，請確認 AST 與原始碼路徑")

    # —— 最後：專案級摘要（全域視角） ——
    print(f"[build] Summarizing project-level snippets from {len(per_file)} files")
    project_sum = summarize(list(itertools.chain.from_iterable(per_file.values())))
    print(f"[build] Generated project summary, upserting...")
    proj_doc = [{"id": "__project_summary__", "text": project_sum, "scope": "project", "file": None}]
    _upsert_docs(qc_cli, coll_name, proj_doc)
    print(f"[build] Incremental upload done + project summary uploaded.")

# ──────────────────── 檢索 ───────────────────────────────

def search(client: qc.QdrantClient, coll_name: str, q: str, k: int = 40) -> List[str]:
    # 1) Embed query for local re-ranking
    q_vec = EMBEDDER.encode(q, convert_to_numpy=True)
    if isinstance(q_vec, list):
        q_vec = np.array(q_vec)
    if q_vec.ndim == 1:
        qv = q_vec.reshape(1, -1)
        q_for_qdrant = q_vec.tolist()
    else:
        qv = q_vec
        q_for_qdrant = q_vec.tolist()

    # 2) Initial semantic search from Qdrant (with payloads and scores)
    hits = client.search(
        collection_name=coll_name,
        query_vector=q_for_qdrant,
        limit=k,
        with_payload=True,
    )
    if not hits:
        return []

    # Collect candidates with text, scope, base score
    items = []
    seen_texts = set()
    for h in hits:
        if not h.payload or "text" not in h.payload:
            print(f"[warn] Hit with no payload or no text: {getattr(h, 'id', None)}")
            continue
        tx = h.payload["text"]
        if tx in seen_texts:
            continue
        seen_texts.add(tx)
        scope = h.payload.get("scope", "func")
        base = float(getattr(h, "score", 0.0) or 0.0)
        items.append({"text": tx, "scope": scope, "base": base})

    if not items:
        return []

    texts = [it["text"] for it in items]

    # 3) Local re-encode for cosine similarity
    sub_vecs = EMBEDDER.encode(texts, convert_to_numpy=True)
    if isinstance(sub_vecs, list):
        sub_vecs = np.array(sub_vecs)
    if sub_vecs.ndim == 1:
        sub_vecs = sub_vecs.reshape(1, -1)
    sims = cosine_similarity(qv, sub_vecs)[0]

    # 4) Normalize Qdrant scores to [0,1]
    base_scores = np.array([it["base"] for it in items], dtype=float)
    if np.ptp(base_scores) > 0:
        # Normalize base scores; use np.ptp for peak-to-peak (NumPy 2.0 compatibility)
        base_norm = (base_scores - base_scores.min()) / (np.ptp(base_scores) if hasattr(np, 'ptp') else base_scores.ptp())
    else:
        base_norm = np.zeros_like(base_scores)

    # 5) Scope bonus
    def bonus(scope: str) -> float:
        if scope == "project":
            return 0.03
        if scope == "file":
            return 0.01
        return 0.0
    bonuses = np.array([bonus(it["scope"]) for it in items], dtype=float)

    # 6) Final score and top-N
    final = 0.8 * sims + 0.2 * base_norm + bonuses
    top_n = min(25, len(items))
    order = np.argsort(final)[::-1][:top_n]
    return [texts[i] for i in order]


def answer(client: qc.QdrantClient, coll_name: str, question: str) -> str:
    """
    Perform RAG-based answer generation for the question using the Qdrant client.
    Returns the generated answer text.
    """
    ctx = search(client, coll_name, question)
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
        coll_name = get_coll_name(args.build)
        build_index(qc_cli, args.build, coll_name)

    if args.ask:
        if not args.build:
            sys.exit("✗ 要使用 --ask 請同時提供 --build <ast_dir> 以決定 collection 名稱")
        coll_name = get_coll_name(args.build)
        try:
            info = qc_cli.get_collection(coll_name)
            print(f"[info] Found collection '{coll_name}' with {info.points_count} points.")
        except Exception:
            sys.exit(f"✗ Collection '{coll_name}' not found. Please run with --build <ast_dir> first.")
        ans = answer(qc_cli, coll_name, args.ask)
        print(ans)

if __name__ == "__main__":
    main()
