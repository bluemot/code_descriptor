#!/usr/bin/env python3
"""
gen_ast_universal.py
===================

• 讀 compile_commands.json，逐檔產生 *.ast.json
• 支援 kernel 和 user space 代碼的通用 AST 生成
• 自動檢測 kernel 環境並添加相應編譯選項
• 立即以 gzip (specified level, default 9) 背景壓縮成
  <relative_path_with__>.ast.json.gz，並刪原 JSON
• 相對路徑基準 = 現在工作目錄 (`project_root`)
• 提供 load_ast() 方便其他腳本讀取（自動解壓 .gz and .zst）
• Fail-fast: Exits on any significant error during processing.
"""

import json
import subprocess
import pathlib
import shlex
import re
import sys
import os
import concurrent.futures
import shutil
import threading
import argparse
import gzip
from typing import List, Dict, Optional

# ─────────────────────────── 命令行參數 ───────────────────────────
parser = argparse.ArgumentParser(description='Generate AST files for kernel and user space code. Exits on first error.')
parser.add_argument('--no-compress', action='store_true', help='Disable compression, keep raw JSON files')
parser.add_argument('--demo', action='store_true', help='Show demo and statistics (runs after generation if successful)')
parser.add_argument('--compress-level', type=int, choices=range(1, 10), default=9,
                    help='Gzip compression level (1-9, default: 9)')
parser.add_argument('--sequential', action='store_true', help='Process files sequentially (disable multi-threading)')
args = parser.parse_args()

ENABLE_COMPRESSION = not args.no_compress
GZIP_COMPRESS_LEVEL = min(args.compress_level, 9) if args.compress_level is not None else 9
IS_SEQUENTIAL = args.sequential

MAX_PENDING = 12
sem_clang = threading.Semaphore(MAX_PENDING)

# ─────────────────────────── 基本路徑 ───────────────────────────
PROJECT_ROOT = pathlib.Path.cwd()
RAW_DIR  = pathlib.Path("ast_out_raw")
COMP_DIR = pathlib.Path("ast_out")
OUTPUT_DIR = RAW_DIR if not ENABLE_COMPRESSION else COMP_DIR

RAW_DIR.mkdir(exist_ok=True)
if ENABLE_COMPRESSION:
    COMP_DIR.mkdir(exist_ok=True)

CDB_PATH = pathlib.Path("compile_commands.json")
if not CDB_PATH.exists():
    print(f"[CRITICAL ERROR] {CDB_PATH} not found. Exiting.", file=sys.stderr)
    sys.exit(1)

try:
    CDB = json.load(open(CDB_PATH))
except json.JSONDecodeError as e:
    print(f"[CRITICAL ERROR] Failed to parse {CDB_PATH}: {e}. Exiting.", file=sys.stderr)
    sys.exit(1)

# ────────────────────── 壓縮函數 (Gzip only) ───────────────────
COMP_EXT = ".gz"

def compress_content_gzip(src_path: pathlib.Path, dst_path: pathlib.Path):
    """Compresses src_path to dst_path using gzip. Raises Exception on error."""
    print(f"[debug gzip] Compressing {src_path.name} to {dst_path.name}")
    if not src_path.exists():
        raise FileNotFoundError(f"Source file for compression not found: {src_path}")
    src_size = src_path.stat().st_size
    print(f"[debug gzip] Source size: {src_size} bytes for {src_path.name}")
    processed_uncompressed_bytes = 0
    try:
        with open(src_path, "rb") as fin, gzip.open(dst_path, "wb", compresslevel=GZIP_COMPRESS_LEVEL) as fout:
            while True:
                chunk = fin.read(1 << 20) # Read 1MB
                if not chunk:
                    break
                bytes_written_in_chunk = fout.write(chunk)
                if bytes_written_in_chunk is None:
                    processed_uncompressed_bytes += len(chunk)
                else:
                    processed_uncompressed_bytes += bytes_written_in_chunk
            
        print(f"[debug gzip] gzip.open context exited for {dst_path.name}. File should be closed and finalized.")
        
        if not dst_path.exists():
            raise IOError(f"Compressed file {dst_path.name} not created after GzipFile close.")

        dst_size_compressed = dst_path.stat().st_size
        print(f"[debug gzip] Compressed file size: {dst_size_compressed} bytes for {dst_path.name}")

        if processed_uncompressed_bytes != src_size:
            if dst_path.exists(): dst_path.unlink(missing_ok=True)
            raise ValueError(f"Mismatch in processed uncompressed bytes for {src_path.name}! "
                             f"Expected {src_size}, but processed {processed_uncompressed_bytes}")

    except Exception as e_compress:
        print(f"[error debug gzip] EXCEPTION during compression of {src_path.name} to {dst_path.name}: {e_compress}", file=sys.stderr)
        if dst_path.exists():
            print(f"[error debug gzip] Deleting partially written/corrupt file {dst_path.name}", file=sys.stderr)
            dst_path.unlink(missing_ok=True)
        raise # Re-raise to be handled by caller


if ENABLE_COMPRESSION:
    print(f"[info] Compression ENABLED: output will be *{COMP_EXT} (gzip), level={GZIP_COMPRESS_LEVEL}")
else:
    print("[info] Compression DISABLED: output will be raw *.ast.json files")

# ───────────────────────── Kernel 檢測 ────────────────────────────
def detect_kernel_source(src_path: pathlib.Path, flags: List[str]) -> bool:
    path_indicators = ["kernel", "arch", "drivers", "fs", "mm", "net", "security", "sound", "crypto", "block", "ipc", "init", "lib", "scripts", "include/linux", "include/asm"]
    src_str = str(src_path).lower()
    if any(indicator in src_str for indicator in path_indicators): return True
    flag_indicators = ["-D__KERNEL__", "-DKBUILD_MODNAME", "-DMODULE", "-nostdinc", "-isystem", "-fno-builtin"]
    flags_str = " ".join(flags)
    if any(indicator in flags_str for indicator in flag_indicators): return True
    if src_path.exists() and src_path.suffix in ['.c', '.h']:
        try:
            with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2048) # Read a small chunk
                kernel_includes = ['#include <linux/', '#include <asm/', '#include "linux/', 'EXPORT_SYMBOL', 'MODULE_LICENSE', '__init', '__exit']
                if any(inc in content for inc in kernel_includes): return True
        except Exception: pass # Ignore errors reading file for this heuristic
    return False

def get_kernel_arch() -> str:
    arch = os.environ.get('ARCH', '')
    if arch: return arch
    for entry in CDB[:min(5, len(CDB))]: # Check a few entries
        flags_list = entry.get("arguments", []) or shlex.split(entry.get("command", ""))
        for flag in flags_list:
            if flag.startswith("-march="):
                march_val = flag[7:]
                if "x86" in march_val or "i386" in march_val: return "x86"
                elif "arm64" in march_val or "aarch64" in march_val: return "arm64"
                elif "arm" in march_val: return "arm"
    return "x86_64" # Default

# ───────────────────────── 旗標過濾與增強 ────────────────────────────
KEEP = re.compile(r"^-D | ^-I | ^-isystem |^--sysroot= |^-std= | ^-nostdinc |^-include |^-x(c)?$ |^-pthread$", re.X)
EXPECT_ARG = {"-I", "-isystem", "-include", "--sysroot"}
GCC_ONLY_FLAGS = {
    "-fconserve-stack", "-fno-allow-store-data-races", "-fno-var-tracking-assignments",
    "-femit-struct-debug-baseonly", "-fno-var-tracking", "-g1", "-feliminate-unused-debug-symbols",
    "-fmerge-debug-strings", "-fdebug-prefix-map=", "-fmacro-prefix-map=", "-ffile-prefix-map=",
    "-fno-canonical-system-headers", "-Wno-maybe-uninitialized", "-Wframe-larger-than=",
    "-fno-reorder-blocks", "-fno-ipa-cp-clone", "-fno-partial-inlining", "-mindirect-branch=",
    "-mindirect-branch-register", "-fno-jump-tables", "-fno-guess-branch-probability",
    "-Wno-packed-not-aligned", "-Wno-format-truncation", "-Wno-format-overflow",
    "-Wno-stringop-truncation", "-falign-jumps=", "-falign-loops=", "-mfunction-return=",
    "-fstack-protector-strong"}

def is_gcc_only_flag(flag: str) -> bool:
    if flag in GCC_ONLY_FLAGS: return True
    gcc_prefixes = ["-falign-jumps=", "-falign-loops=", "-Wframe-larger-than=", "-fdebug-prefix-map=", "-fmacro-prefix-map=", "-ffile-prefix-map=", "-mindirect-branch=", "-mfunction-return="]
    return any(flag.startswith(prefix) for prefix in gcc_prefixes)

def scrub(argv: List[str]) -> List[str]:
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-c", "-o"): i += 2; continue
        if tok.startswith(("-c", "-o")) and "=" not in tok: i += 1; continue
        if is_gcc_only_flag(tok): i += 1; continue
        if KEEP.match(tok):
            out.append(tok)
            if tok in EXPECT_ARG and i + 1 < len(argv) and not any(tok.startswith(p) for p in ["-I/", "--sysroot=", "--param="]):
                i += 1
                if i < len(argv): out.append(argv[i])
        i += 1
    return out

def enhance_kernel_flags(flags: List[str], src_path: pathlib.Path) -> List[str]:
    enhanced = flags.copy()
    arch = get_kernel_arch()
    kernel_defines = [f"-D__KERNEL__=1", f"-DKBUILD_MODNAME=KBUILD_STR(dummy)", f"-DCONFIG_{arch.upper()}=1", f"-DMODULE=1", f"-D__LINUX_KERNEL__=1"]
    for define in kernel_defines:
        if not any(f.startswith(define.split('=')[0]) for f in enhanced): enhanced.append(define)
    
    arch_specific_flags = []
    if arch in ["x86", "x86_64"]: arch_specific_flags = ["-D__x86_64__=1", "-DCONFIG_X86_64=1", "-D__LP64__=1"]
    elif arch == "arm64": arch_specific_flags = ["-D__aarch64__=1", "-DCONFIG_ARM64=1", "-D__LP64__=1"]
    elif arch == "arm": arch_specific_flags = ["-D__arm__=1", "-DCONFIG_ARM=1"]
    for flag in arch_specific_flags:
        if not any(f.startswith(flag.split('=')[0]) for f in enhanced): enhanced.append(flag)

    common_kernel_defines = ["-DCONFIG_64BIT=1", "-DCONFIG_SMP=1", "-DCONFIG_PREEMPT_NONE=1", "-D__STDC_HOSTED__=0", "-fno-builtin", "-ffreestanding"]
    for define in common_kernel_defines:
        if not any(f.startswith(define.split('=')[0]) for f in enhanced): enhanced.append(define)

    kernel_includes_paths_str = ["include", f"arch/{arch}/include", "include/uapi", f"arch/{arch}/include/uapi"]
    if arch in ["x86", "x86_64"]:
        kernel_includes_paths_str = ["include", "arch/x86/include", "include/uapi", "arch/x86/include/uapi"]
    
    for inc_path_s in kernel_includes_paths_str:
        full_inc_path = PROJECT_ROOT / inc_path_s
        if full_inc_path.exists():
            inc_flag = f"-I{full_inc_path.resolve()}"
            if inc_flag not in enhanced: enhanced.append(inc_flag)
    return enhanced

# ────────────────── Core Compression & Validation Logic ───────────────────
def _compress_validate_and_remove_json(json_path: pathlib.Path):
    """
    Performs GZIP compression, validation, and removes the original JSON file.
    Raises RuntimeError on any failure.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Original JSON file {json_path.name} not found for compression.")

    original_json_size = json_path.stat().st_size
    compressed_path = COMP_DIR / f"{json_path.name}{COMP_EXT}"

    print(f"[debug _compress_validate_and_remove_json] Starting compression for {json_path.name} (size: {original_json_size}) to {compressed_path.name}")

    try:
        compress_content_gzip(json_path, compressed_path) # Can raise Exception
        print(f"[debug _compress_validate_and_remove_json] GZIP compression function returned for {compressed_path.name}")

        if not compressed_path.exists():
            raise RuntimeError(f"Compressed file {compressed_path.name} does not exist after compression call!")

        compressed_file_size = compressed_path.stat().st_size
        print(f"[debug _compress_validate_and_remove_json] Compressed file {compressed_path.name} current size: {compressed_file_size}")

        # Validation Step 1: gzip -t
        validation_passed_external = False
        try:
            subprocess.run(["gzip", "-t", str(compressed_path)], 
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[debug _compress_validate_and_remove_json] `gzip -t` validation PASSED for {compressed_path.name}")
            validation_passed_external = True
        except subprocess.CalledProcessError as val_err:
            stderr_output = val_err.stderr.decode('utf-8', errors='ignore').strip() if val_err.stderr else "No stderr output"
            msg = f"`gzip -t` validation FAILED for {compressed_path.name}: {val_err}. Stderr: {stderr_output}"
            print(f"[error _compress_validate_and_remove_json] {msg}", file=sys.stderr)
            # Don't raise yet, try internal validation
        except FileNotFoundError:
            print(f"[warn _compress_validate_and_remove_json] `gzip` command not found for external validation of {compressed_path.name}. Skipping.")

        # Validation Step 2: Internal Python gzip read
        validation_passed_internal = False
        try:
            with gzip.open(compressed_path, 'rb') as f_val:
                data_chunk = f_val.read(1024 * 1024) # Try to read up to 1MB
                # Check if file is empty or fully read
                if compressed_file_size == 0: # Original was empty, GZ is small
                    validation_passed_internal = True
                elif len(data_chunk) == compressed_file_size and f_val.peek(1) == b'': # For small files fully read
                     validation_passed_internal = True
                elif len(data_chunk) > 0 and f_val.peek(1) == b'': # Read some, and it's EOF
                     validation_passed_internal = True
                elif len(data_chunk) == 0 and compressed_file_size > 0:
                    # Read nothing from a non-empty compressed file
                    raise gzip.BadGzipFile(f"Internal Python gzip read: read 0 bytes from non-empty file {compressed_path.name}")
                else: # General case, assume OK if read some data. EOF check above is better.
                    validation_passed_internal = True # Tentative
            if validation_passed_internal:
                print(f"[debug _compress_validate_and_remove_json] Internal Python gzip read validation PASSED for {compressed_path.name}")
        except Exception as e_val_internal:
            msg = f"Internal Python gzip read validation FAILED for {compressed_path.name}: {e_val_internal}"
            print(f"[error _compress_validate_and_remove_json] {msg}", file=sys.stderr)
            # If external validation also failed or was skipped, this is critical
            if not validation_passed_external:
                if compressed_path.exists(): compressed_path.unlink(missing_ok=True)
                raise RuntimeError(msg + " (and external validation failed/skipped)") from e_val_internal

        if not (validation_passed_external or validation_passed_internal):
            # This case should be caught by the raise in internal validation if external also failed
            msg = f"ALL gzip validations (external and internal) FAILED for {compressed_path.name}."
            if compressed_path.exists(): compressed_path.unlink(missing_ok=True)
            raise RuntimeError(msg)

        # Validation Step 3: Decompress and compare size
        temp_decompressed_path = None
        try:
            temp_decomp_name = f"{compressed_path.stem}_temp_decomp_{os.getpid()}_{threading.get_ident()}"
            temp_decompressed_path = compressed_path.parent / temp_decomp_name

            with gzip.open(compressed_path, 'rb') as f_gz_in, open(temp_decompressed_path, 'wb') as f_plain_out:
                shutil.copyfileobj(f_gz_in, f_plain_out)
            
            decompressed_size = temp_decompressed_path.stat().st_size
            print(f"[debug _compress_validate_and_remove_json] Decompressed {compressed_path.name} to {temp_decompressed_path.name} (size: {decompressed_size})")
            
            if original_json_size != decompressed_size:
                msg = (f"Decompressed size ({decompressed_size}) MISMATCHES original JSON size ({original_json_size}) "
                       f"for {json_path.name}! Original: {json_path}, Compressed: {compressed_path}, Test: {temp_decompressed_path}")
                print(f"[CRITICAL ERROR _compress_validate_and_remove_json] {msg}", file=sys.stderr)
                # Keep temp_decompressed_path for inspection
                raise RuntimeError(msg)
            else:
                print(f"[debug _compress_validate_and_remove_json] Decompressed size MATCHES original for {json_path.name}.")
                if temp_decompressed_path.exists(): temp_decompressed_path.unlink()
        
        except Exception as e_final_check:
            msg = f"Final decompression check FAILED for {compressed_path.name}: {e_final_check}"
            print(f"[CRITICAL ERROR _compress_validate_and_remove_json] {msg}", file=sys.stderr)
            if temp_decompressed_path and temp_decompressed_path.exists():
                 print(f"[debug _compress_validate_and_remove_json] Keeping temp decompressed file for inspection: {temp_decompressed_path.name}")
            if compressed_path.exists(): # Keep potentially corrupt compressed file if original still exists
                 print(f"[debug _compress_validate_and_remove_json] Keeping compressed file: {compressed_path.name}")
            raise RuntimeError(msg) from e_final_check

        # If all checks passed:
        print(f"[info] {compressed_path.name} compress completed and validated. Deleting original: {json_path.name}")
        json_path.unlink(missing_ok=True)

    except Exception as e: # Catch-all for this function's scope, including re-raised ones
        # Ensure potentially corrupt compressed file is removed if original still exists
        if compressed_path.exists() and json_path.exists():
            print(f"[debug _compress_validate_and_remove_json] Cleaning up potentially bad compressed file {compressed_path.name} due to error: {e}", file=sys.stderr)
            compressed_path.unlink(missing_ok=True)
        # Re-raise the original error or a new one wrapping it
        if not isinstance(e, RuntimeError): # If it's not one of our specific RuntimeErrors
            raise RuntimeError(f"Unexpected error during compression/validation for {json_path.name}: {e}") from e
        else:
            raise # Re-raise the specific RuntimeError

# ────────────────── Task for Parallel Processing (Clang + Compression) ─────────
def _process_entry_task(entry_data: Dict):
    """
    Processes a single compile command: AST generation and optional compression.
    This function is submitted to ThreadPoolExecutor. Raises Exception on failure.
    Acquires and releases sem_clang.
    """
    raw_json_path_obj = None
    src_file_name_for_log = pathlib.Path(entry_data.get('file', 'unknown_file')).name
    
    sem_clang.acquire()
    try:
        file_entry_path_str = entry_data["file"]
        directory_entry_path_str = entry_data["directory"]

        workdir = pathlib.Path(directory_entry_path_str)
        if not workdir.is_absolute():
            cdb_parent_dir = CDB_PATH.parent.resolve()
            workdir = (cdb_parent_dir / workdir).resolve()
        
        temp_src_path = pathlib.Path(file_entry_path_str)
        src_path = (workdir / temp_src_path).resolve() if not temp_src_path.is_absolute() else temp_src_path.resolve()
        
        argv_original = entry_data.get("arguments") or shlex.split(entry_data["command"])

        if src_path.name.endswith((".mod.c", ".s", ".S")) or src_path.suffix not in ['.c', '.h', '.cpp', '.cc', '.cxx']:
            print(f"[skip task] Skipping non-C/C++/H file: {src_path.name}")
            return
        
        flags = scrub(argv_original[1:])
        is_kernel = detect_kernel_source(src_path, flags)
        log_prefix = "[kernel task]" if is_kernel else "[user task]"
        if is_kernel:
            flags = enhance_kernel_flags(flags, src_path)
        print(f"{log_prefix} {src_path.name}")

        try:
            rel_path = src_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = pathlib.Path(*src_path.parts[1:])
    
        safe_filename_base = "__".join(rel_path.parts).replace(":", "_").replace(" ", "_")
        raw_json_path_obj = RAW_DIR / f"{safe_filename_base}.ast.json"

        clang_cmd = ["clang", *flags, str(src_path), "-Xclang", "-ast-dump=json", "-fsyntax-only"]
        print(f"[clang task] Generating AST for: {rel_path} -> {raw_json_path_obj.name}")
        
        if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)

        try:
            with open(raw_json_path_obj, "wb") as fh_out:
                # Increased timeout slightly
                result = subprocess.run(clang_cmd, cwd=workdir, stdout=fh_out, stderr=subprocess.PIPE, timeout=180)
            
            if result.returncode != 0:
                stderr_msg = result.stderr.decode('utf-8', errors='ignore')[:500] if result.stderr else "No stderr."
                msg = (f"Clang failed for {src_file_name_for_log} with code {result.returncode}. "
                       f"Command: {' '.join(clang_cmd)}. Stderr: {stderr_msg}")
                if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)
                raise RuntimeError(msg)
            
            if not raw_json_path_obj.exists() or raw_json_path_obj.stat().st_size == 0:
                msg = f"Clang produced no/empty output file {raw_json_path_obj.name} for {src_file_name_for_log}."
                if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)
                raise RuntimeError(msg)

        except subprocess.TimeoutExpired:
            if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)
            raise RuntimeError(f"Clang timeout for {src_file_name_for_log}. Command: {' '.join(clang_cmd)}")
        except Exception as e_clang: # Other Clang execution errors
            if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)
            raise RuntimeError(f"Clang execution error for {src_file_name_for_log}: {e_clang}") from e_clang

        # AST Generation successful
        if ENABLE_COMPRESSION:
            print(f"[info task] AST generated for {raw_json_path_obj.name}. Proceeding to compression.")
            _compress_validate_and_remove_json(raw_json_path_obj) # Raises RuntimeError on failure
            print(f"[info task] Compression successful for {raw_json_path_obj.name} (original JSON removed).")
        else:
            print(f"[info task] {raw_json_path_obj.name} saved (uncompressed in {RAW_DIR})")

    except Exception as e_task: # Catch all exceptions from this task to ensure semaphore release
        # Error message already printed by specific failure points or will be printed by main
        # Re-raise to be caught by main's future.result()
        raise RuntimeError(f"Error processing entry for {src_file_name_for_log} in task: {e_task}") from e_task
    finally:
        sem_clang.release()

# ────────────────── AST Generation and Compression (Sequential Mode) ───────────────
def handle_single_entry_sequentially(entry: Dict):
    raw_json_path_obj = None
    src_file_name_for_log = pathlib.Path(entry.get('file', 'unknown_file')).name
    try:
        file_entry_path_str = entry["file"]
        directory_entry_path_str = entry["directory"]

        workdir = pathlib.Path(directory_entry_path_str)
        if not workdir.is_absolute():
            cdb_parent_dir = CDB_PATH.parent.resolve()
            workdir = (cdb_parent_dir / workdir).resolve()
        
        temp_src_path = pathlib.Path(file_entry_path_str)
        src_path = (workdir / temp_src_path).resolve() if not temp_src_path.is_absolute() else temp_src_path.resolve()
        
        argv_original = entry.get("arguments") or shlex.split(entry["command"])

        if src_path.name.endswith((".mod.c", ".s", ".S")) or src_path.suffix not in ['.c', '.h', '.cpp', '.cc', '.cxx']:
            print(f"[skip sequential] Skipping non-C/C++/H file: {src_path.name}")
            return 
        
        flags = scrub(argv_original[1:])
        is_kernel = detect_kernel_source(src_path, flags)
        log_prefix = "[kernel sequential]" if is_kernel else "[user sequential]"
        if is_kernel:
            flags = enhance_kernel_flags(flags, src_path)
        print(f"{log_prefix} {src_path.name}")

        try:
            rel_path = src_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = pathlib.Path(*src_path.parts[1:])
    
        safe_filename_base = "__".join(rel_path.parts).replace(":", "_").replace(" ", "_")
        raw_json_path_obj = RAW_DIR / f"{safe_filename_base}.ast.json"

        clang_cmd = ["clang", *flags, str(src_path), "-Xclang", "-ast-dump=json", "-fsyntax-only"]
        print(f"[clang sequential] Generating AST for: {rel_path} -> {raw_json_path_obj.name}")
        
        if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)

        try:
            with open(raw_json_path_obj, "wb") as fh_out:
                result = subprocess.run(clang_cmd, cwd=workdir, stdout=fh_out, stderr=subprocess.PIPE, timeout=180)
            
            if result.returncode != 0:
                stderr_msg = result.stderr.decode('utf-8', errors='ignore')[:500] if result.stderr else "No stderr."
                msg = (f"Clang failed for {src_file_name_for_log} with code {result.returncode}. "
                       f"Command: {' '.join(clang_cmd)}. Stderr: {stderr_msg}")
                print(f"[CRITICAL ERROR sequential] {msg}", file=sys.stderr)
                if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)
                sys.exit(1)
            
            if not raw_json_path_obj.exists() or raw_json_path_obj.stat().st_size == 0:
                msg = f"Clang produced no/empty output file {raw_json_path_obj.name} for {src_file_name_for_log}."
                print(f"[CRITICAL ERROR sequential] {msg}", file=sys.stderr)
                if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)
                sys.exit(1)

        except subprocess.TimeoutExpired:
            msg = f"Clang timeout for {src_file_name_for_log}. Command: {' '.join(clang_cmd)}"
            print(f"[CRITICAL ERROR sequential] {msg}", file=sys.stderr)
            if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)
            sys.exit(1)
        except Exception as e_clang:
            msg = f"Clang execution error for {src_file_name_for_log}: {e_clang}"
            print(f"[CRITICAL ERROR sequential] {msg}", file=sys.stderr)
            if raw_json_path_obj.exists(): raw_json_path_obj.unlink(missing_ok=True)
            sys.exit(1)

        # AST Generation successful
        if ENABLE_COMPRESSION:
            print(f"[info sequential] AST generated for {raw_json_path_obj.name}. Proceeding to compression.")
            try:
                _compress_validate_and_remove_json(raw_json_path_obj) # Raises RuntimeError on failure
                print(f"[info sequential] Compression successful for {raw_json_path_obj.name} (original JSON removed).")
            except Exception as e_compress_validate:
                print(f"[CRITICAL ERROR sequential] Compression/Validation failed for {raw_json_path_obj.name}: {e_compress_validate}", file=sys.stderr)
                # _compress_validate_and_remove_json might have cleaned up compressed_path
                # Original json_path might still exist if validation failed.
                sys.exit(1)
        else:
            print(f"[info sequential] {raw_json_path_obj.name} saved (uncompressed in {RAW_DIR})")

    except SystemExit: # Propagate sys.exit()
        raise
    except Exception as e_process_entry: # Catch any other unexpected error
        print(f"[CRITICAL ERROR sequential] Unexpected error processing entry for {src_file_name_for_log}: {e_process_entry}", file=sys.stderr)
        if raw_json_path_obj and raw_json_path_obj.exists():
             raw_json_path_obj.unlink(missing_ok=True) # Cleanup
        sys.exit(1)

# ─────────────────────────── 主流程 ────────────────────────────
def main():
    if IS_SEQUENTIAL:
        print(f"[info] Sequential processing ENABLED. Processing {len(CDB)} files...")
        for i, entry_data in enumerate(CDB):
            print(f"---> Processing file {i+1}/{len(CDB)} (sequential): {pathlib.Path(entry_data['file']).name} <---")
            handle_single_entry_sequentially(entry_data) # Will sys.exit(1) on error
        print(f"[progress sequential] Successfully processed all {len(CDB)} files.")
    else: # Parallel processing
        print(f"[info] Parallel processing ENABLED. Processing {len(CDB)} files...")
        # Adjust num_workers: more than MAX_PENDING if compression is lighter than clang
        # For simplicity, let's use MAX_PENDING, as each task does clang then maybe compression.
        num_workers = MAX_PENDING
        
        all_task_futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            for i, entry_data in enumerate(CDB):
                if i > 0 and i % 50 == 0:
                    print(f"[progress parallel] Submitted {i}/{len(CDB)} tasks for processing...")
                future = pool.submit(_process_entry_task, entry_data)
                all_task_futures.append(future)
            
            print(f"[info parallel] All {len(all_task_futures)} tasks submitted. Waiting for completion...")
            completed_count = 0
            try:
                for f_task in concurrent.futures.as_completed(all_task_futures):
                    f_task.result() # Will re-raise exception from _process_entry_task if any
                    completed_count += 1
                    if completed_count % 20 == 0 or completed_count == len(all_task_futures):
                        print(f"[progress parallel] {completed_count}/{len(all_task_futures)} tasks completed successfully.")
            except Exception as e:
                print(f"[CRITICAL ERROR parallel] A task failed: {e}", file=sys.stderr)
                print("[CRITICAL ERROR parallel] Attempting to shutdown worker pool and exit.", file=sys.stderr)
                # Shutdown the pool: try to cancel pending futures (Py3.9+) or just don't wait for them.
                if sys.version_info >= (3, 9):
                    pool.shutdown(wait=False, cancel_futures=True)
                else:
                    pool.shutdown(wait=False) # For older Python, more abrupt.
                sys.exit(1) # Exit the main process
            
            print(f"[info parallel] All {len(all_task_futures)} tasks completed successfully.")

    print(f"\n[done] All AST files processed successfully. Output in {OUTPUT_DIR.resolve()}")
    
    if ENABLE_COMPRESSION:
        total_files = len(list(COMP_DIR.glob(f"*{COMP_EXT}")))
        print(f"[stats] Total {total_files} compressed AST files (*{COMP_EXT}) in {COMP_DIR}")
    else:
        total_files = len(list(RAW_DIR.glob("*.ast.json")))
        print(f"[stats] Total {total_files} uncompressed AST files in {RAW_DIR}")

# ──────────────────────── 輔助：load_ast ───────────────────────
def load_ast(pathlike) -> Optional[dict]:
    p = pathlib.Path(pathlike)
    ext_tuple = p.suffixes
    
    try:
        if ext_tuple[-1] == ".gz":
            with gzip.open(p, "rt", encoding="utf-8") as fp_gz:
                return json.load(fp_gz)
        elif ext_tuple[-1] == ".zst":
            try:
                import zstandard as zstd_load
                with open(p, "rb") as fh_zst_bin:
                    decompressed_bytes = zstd_load.ZstdDecompressor().decompress(fh_zst_bin.read())
                return json.loads(decompressed_bytes.decode('utf-8'))
            except ImportError:
                print(f"[warn load_ast] zstandard library not found, cannot load {p.name}", file=sys.stderr)
                return None
            except Exception as e_zstd_load:
                print(f"[error load_ast] Failed to load zstd file {p.name}: {e_zstd_load}", file=sys.stderr)
                return None
        elif ext_tuple[-1] == ".json":
            return json.loads(p.read_text(encoding='utf-8'))
        else:
            print(f"[warn load_ast] Unknown file extension for AST: {p.name}", file=sys.stderr)
            return None
    except FileNotFoundError:
        print(f"[error load_ast] File not found: {p}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e_json:
        print(f"[error load_ast] Invalid JSON in {p.name}: {e_json}", file=sys.stderr)
        return None
    except Exception as e_load: # Catch other potential errors like gzip.BadGzipFile
        print(f"[error load_ast] Failed to load AST from {p.name}: {e_load}", file=sys.stderr)
        return None

# ─────────────────── demo 和統計 ──────────────────────────────────
def show_demo():
    print("\n--- Demo and Statistics ---")
    current_output_dir = COMP_DIR if ENABLE_COMPRESSION else RAW_DIR
    file_glob_pattern = f"*{COMP_EXT}" if ENABLE_COMPRESSION else "*.ast.json"
    
    ast_files_list = list(current_output_dir.glob(file_glob_pattern))
    file_type_desc = f"compressed (*{COMP_EXT})" if ENABLE_COMPRESSION else "uncompressed (*.ast.json)"
    
    if not ast_files_list:
        print(f"[demo] No {file_type_desc} AST files found in {current_output_dir}.")
        return
    
    print(f"[stats demo] Found {len(ast_files_list)} {file_type_desc} AST files in {current_output_dir}.")
    
    kernel_keywords = ["kernel", "drivers", "arch", "mm", "fs", "net"]
    kernel_files_count = sum(1 for f_path in ast_files_list if any(k_word in f_path.name.lower() for k_word in kernel_keywords))
    user_files_count = len(ast_files_list) - kernel_files_count
    print(f"[stats demo] Estimated kernel files: {kernel_files_count}, user space files: {user_files_count}")
    
    sample_file_for_demo = ast_files_list[0]
    print(f"[demo] Analyzing sample file: {sample_file_for_demo.name}")
    ast_content = load_ast(sample_file_for_demo)
    
    if ast_content:
        functions_found = []
        def extract_func_names_recursive(node_element):
            if isinstance(node_element, dict):
                if node_element.get("kind") == "FunctionDecl" and node_element.get("name"):
                    functions_found.append(node_element["name"])
                for key, value in node_element.items():
                    if key == "inner" and isinstance(value, list):
                        for child_node in value: extract_func_names_recursive(child_node)
                    elif isinstance(value, dict): extract_func_names_recursive(value)
                    elif isinstance(value, list):
                         for item_in_list in value: extract_func_names_recursive(item_in_list)
            elif isinstance(node_element, list):
                for item in node_element: extract_func_names_recursive(item)

        extract_func_names_recursive(ast_content.get("inner", [])) # Clang AST usually has a top "inner" list
        
        print(f"[demo] Sample {sample_file_for_demo.name}: found {len(functions_found)} unique functions (approx).")
        if functions_found:
            unique_funcs = sorted(list(set(functions_found)))
            print(f"[demo] First 5 unique functions: {', '.join(unique_funcs[:5])}")
    else:
        print(f"[demo] Could not load or parse AST for sample file {sample_file_for_demo.name}.")

if __name__ == "__main__":
    if args.sequential:
        print("[info] Running in --sequential mode.")
    
    main_executed_successfully = False
    try:
        main()
        main_executed_successfully = True # Set flag if main completes without sys.exit
    except SystemExit as e:
        # main() or one of its called functions called sys.exit()
        # Error messages should have been printed already.
        # The exit code from sys.exit() will be used.
        if e.code == 0: # Should not happen if error occurred, but as safeguard
            main_executed_successfully = True # If sys.exit(0) was somehow called
        else:
            print(f"[info] Script terminated early due to error (exit code: {e.code}).", file=sys.stderr)
            raise # Re-raise to ensure script exits with the correct code
    except Exception as e_unhandled:
        # Catch any other unexpected error not handled by sys.exit in main paths
        print(f"[CRITICAL UNHANDLED ERROR] An unexpected error occurred: {e_unhandled}", file=sys.stderr)
        sys.exit(1) # Ensure non-zero exit code

    if main_executed_successfully and args.demo:
        show_demo()
    elif args.demo and not main_executed_successfully:
        print("[info] Skipping demo due to errors during main processing.", file=sys.stderr)