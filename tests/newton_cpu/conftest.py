"""Shared paths of the Newton CPU tests. Reuses the synthetic graph builders of Phase 4, the actual-MAPLE bed
constructors of Phases 5/6, the prepared-hydrology case builder of Phase 7h and the RFID pit chain."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/phase4", "tests/phase5", "tests/phase6", "tests/phase7h", "tests/rfid", "tests"):
    sys.path.insert(0, str(ROOT / sub))
