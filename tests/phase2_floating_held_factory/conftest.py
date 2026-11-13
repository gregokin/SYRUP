import sys
from pathlib import Path

T = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(T / "phase2_floating_checkpoint"), str(T / "phase2_floating_pond")]
