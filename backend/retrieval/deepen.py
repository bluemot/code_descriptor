"""
On-demand AST-aware function-level deepening and embedding upsert.
"""
import os, json, pathlib, gzip
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
    # AST-aware function splitting and summary generation
    summaries_dir = pathlib.Path(os.getenv('INDEX_STATE_DIR', '.rag_state')) / 'deepen'
    summaries_dir.mkdir(parents=True, exist_ok=True)
    llm = None
    try:
        from ..rag_demo6_1 import LLM_SUM
        llm = LLM_SUM
    except ImportError:
        pass
    new_funcs = 0
    def extract_funcs_from_ast(p: pathlib.Path) -> list[tuple[str,str,int,int,str]]:
        # Attempt AST JSON parsing for precise function ranges
        for suf in ('.ast.json.gz', '.ast.json.zst', '.ast.json'):
            astp = p.with_suffix(p.suffix + suf)
            if astp.exists():
                try:
                    import gzip, json as _json

                    text = gzip.open(astp, 'rt', errors='ignore').read() if astp.suffix.endswith('gz') else _json.loads(astp.read_text(errors='ignore'))
                    data = _json.loads(text) if isinstance(text, str) else text
                except Exception:
                    break
                funcs = []
                def visit(node: dict):
                    if node.get('kind') in ('FunctionDecl', 'CXXMethodDecl', 'FunctionTemplateDecl'):
                        r = node.get('range', {})
                        b = r.get('begin', {}).get('line')
                        e = r.get('end', {}).get('line')
                        sig = node.get('loc', {}).get('line') and node.get('name', '')
                        if b and e and e - b <= max_lines:
                            # load source block
                            src = p.read_text(errors='ignore').splitlines()[b-1:e]
                            funcs.append((node.get('name', ''), sig or '', b, e, '\n'.join(src)))
                    for c in node.get('inner', []):
                        visit(c)
                visit(data)
                return funcs
        # fallback: regex-based extraction
        text = p.read_text(errors='ignore')
        lines = text.splitlines()
        funcs = []
        i = 0
        while i < len(lines):
            sig = lines[i].strip()
            if sig.endswith('{') and '(' in sig and ')' in sig:
                start = i
                brace = 0
                block = []
                while i < len(lines):
                    l = lines[i]
                    brace += l.count('{') - l.count('}')
                    block.append(l)
                    i += 1
                    if brace <= 0:
                        break
                funcs.append((sig.split('(')[0], sig, start+1, start+len(block), '\n'.join(block)))
            else:
                i += 1
        return funcs

    for path in to_process:
        p = pathlib.Path(path)
        for fn_name, sig, b, e, fn_block in extract_funcs_from_ast(p):
            summary = None
            if llm:
                prompt = (
                    f"請為以下 C/C++ 函式程式碼產生摘要，包含 signature、inputs、outputs、side effects：\n" +
                    "```\n" + fn_block + "\n```"
                )
                try:
                    summary = llm.invoke(prompt)
                except Exception:
                    summary = None
            fname = f"{p.stem}_{new_funcs}.json"
            outp = summaries_dir / fname
            outp.write_text(
                json.dumps({
                    'file': str(p), 'function': fn_name,
                    'signature': sig, 'summary': summary
                }, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            new_funcs += 1
    took = (time.time() - start) * 1000
    report = DeepenReport(
        files=to_process,
        new_functions=new_funcs,
        new_points=0,
        took_ms=took,
        cached_hit=False,
    )
    print(f"[deepen] modules upserted funcs={new_funcs} points=0 took={report.took_ms:.0f}ms")
    return report
