"""
Pytest configuration and path adjustment.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path for module imports
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
