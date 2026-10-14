"""CW-B20 private accounting-step tests: shared paths and the case builders."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/phase5", "tests/conservative_water", "tests/cw_b20"):
    sys.path.insert(0, str(ROOT / sub))
