import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Reuse the verified actual-MAPLE bed constructors and the Plot 1 case fixture
# from Phase 5, and import the comparator script as a module.
sys.path.insert(0, str(ROOT / "tests" / "phase5"))
sys.path.insert(0, str(ROOT / "benchmarks" / "phase7"))

from test_sediment_experiment import plot1_case  # noqa: F401
