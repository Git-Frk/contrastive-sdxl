# utils/setup_paths.py
import sys
from pathlib import Path

roma_path = Path(__file__).resolve().parent.parent/'modles'/'external'/'roma'
if str(roma_path) not in sys.path:
    sys.path.insert(0, str(roma_path))