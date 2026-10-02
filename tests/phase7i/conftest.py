import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Synthetic terrain/graph builders of Phase 4 (test_routing), the Phase 7 comparator module and the Phase 7i scripts.
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
sys.path.insert(0, str(ROOT / "benchmarks" / "phase7"))
sys.path.insert(0, str(ROOT / "benchmarks" / "phase7i"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
