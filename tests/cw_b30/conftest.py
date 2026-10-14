"""CW-B30 outer-frame lifetime tests: shared paths."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/phase5", "tests/conservative_water", "tests/cw_b30"):
    sys.path.insert(0, str(ROOT / sub))
