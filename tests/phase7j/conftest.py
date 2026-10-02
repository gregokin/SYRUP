import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Phase 4 synthetic terrain/graph builders; `phase7h` (a package under tests/) supplies the committed case builders,
# comparison helpers and failure mutations that Phase 7j reuses.
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
sys.path.insert(0, str(ROOT / "tests"))
