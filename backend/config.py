"""
Load environment variables from env.txt at project root.
"""
import os
from pathlib import Path


def load_env():
    root = Path(__file__).resolve().parent.parent
    env_file = root / "env.txt"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        # remove inline comments
        if "#" in val:
            val = val.split("#", 1)[0].strip()
        # strip quotes
        val = val.strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val

# load on import
load_env()
