import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Reuse the synthetic terrain/graph builders of Phase 4, and the verified Plot 1 case fixture of Phase 5.
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
sys.path.insert(0, str(ROOT / "tests" / "phase5"))
sys.path.insert(0, str(ROOT / "benchmarks" / "phase7"))

from test_sediment_experiment import plot1_case  # noqa: F401
