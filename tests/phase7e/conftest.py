import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "phase7e"))
sys.path.insert(0, str(ROOT / "benchmarks" / "phase7d"))
