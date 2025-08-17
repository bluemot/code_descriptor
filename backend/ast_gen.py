"""
Wrapper for AST generation using the existing gen_ast_all_no_multi script.
"""
import os
from pathlib import Path




from typing import Optional, Callable

def run_ast_generation(source_dir: Path, on_line: Optional[Callable[[str], None]] = None) -> Path:
    """
    Generate AST files for all source files in source_dir using gen_ast_all_no_multi.
    Returns the directory where AST outputs are written.
    """
    # Invoke the gen_ast script in a subprocess to avoid import-time side effects
    import sys
    import subprocess
    cwd = os.getcwd()
    try:
        print(f"[debug ast_gen] cwd before chdir: {cwd}", file=sys.stderr)
        os.chdir(source_dir)
        print(f"[debug ast_gen] changed cwd to source_dir: {source_dir}", file=sys.stderr)
        script = Path(__file__).parent / 'gen_ast_all_no_multi.py'
        print(f"[debug ast_gen] script path: {script}", file=sys.stderr)
        try:
            # Use unbuffered Python (-u) and line-buffered IO (bufsize=1) for real-time stdout streaming
            cmd = [sys.executable, '-u', str(script), '--sequential']
            print(f"[debug ast_gen] launching command: {cmd}", file=sys.stderr)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if proc.stdout is not None:
                for line in proc.stdout:
                    print(f"[debug ast_gen stdout] {line.strip()}", file=sys.stderr)
                    if on_line:
                        on_line(line)
            ret = proc.wait()
            print(f"[debug ast_gen] process exit code: {ret}", file=sys.stderr)
            if ret != 0:
                print(f"[debug ast_gen] non-zero exit, fallback to source_dir", file=sys.stderr)
                return source_dir
        except Exception as e:
            print(f"[debug ast_gen] exception during subprocess: {e}", file=sys.stderr)
            return source_dir
        out_dir = source_dir / 'ast_out'
        if not out_dir.exists():
            out_dir = source_dir / 'ast_out_raw'
        print(f"[debug ast_gen] returning out_dir: {out_dir}", file=sys.stderr)
        return out_dir
    finally:
        os.chdir(cwd)
