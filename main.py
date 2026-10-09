"""Thin wrapper so `uv run main.py <cmd>` works alongside the `qqbot` script.

All logic lives in `scripts/cli.py`; this deliberately does not re-export or
duplicate it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
