import sys
from pathlib import Path

_T = Path(__file__).resolve().parents[1]
for sub in ("phase2_native_floating_soil", "phase2_compact_soil_profile", "phase2_floating_held_context", "phase2_floating_pond", "pond_event", "pond_windowed"):
    sys.path.insert(0, str(_T / sub))
