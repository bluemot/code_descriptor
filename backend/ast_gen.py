"""
Wrapper for AST generation using the existing gen_ast_all_no_multi script.
"""
import os
from pathlib import Path




def run_ast_generation(source_dir: Path) -> Path:
    """
    Generate AST files for all source files in source_dir using gen_ast_all_no_multi.
    Returns the directory where AST outputs are written.
    """
    # Invoke the gen_ast script in a subprocess to avoid import-time side effects
    import sys, subprocess
    cwd = os.getcwd()
    try:
        os.chdir(source_dir)
        script = Path(__file__).parent / 'gen_ast_all_no_multi.py'
        try:
            subprocess.run([sys.executable, str(script)], check=True)
        except subprocess.CalledProcessError:
            # ignore errors (e.g., missing compile_commands.json) and fallback
            return source_dir
        out_dir = source_dir / 'ast_out'
        if not out_dir.exists():
            out_dir = source_dir / 'ast_out_raw'
        return out_dir
    finally:
        os.chdir(cwd)
