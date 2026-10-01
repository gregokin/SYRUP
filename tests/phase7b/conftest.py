import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Reuse the actual-MAPLE bed constructors of the Phase 5 event tests and the
# Phase 6 completion fixture for the integration tests of the corrected scheme.
sys.path.insert(0, str(ROOT / "tests" / "phase5"))
sys.path.insert(0, str(ROOT / "tests" / "phase6"))
