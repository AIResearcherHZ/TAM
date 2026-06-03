#!/usr/bin/env python3
"""Run external-mode trajectory tracking error evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simadaptor.deploy.trajectory_tracking_eval import main


if __name__ == "__main__":
    main()
