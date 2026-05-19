#!/usr/bin/env python
"""CLI shim. Run from the repo root:

    python scripts/generate_dataset.py --n 100 --out data/raw/smoke.parquet
"""
import sys
from pathlib import Path

# Repo root onto sys.path so `from src.data_generation import ...` works
# without requiring the project to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generation.generate_dataset import main  # noqa: E402

if __name__ == "__main__":
    main()
