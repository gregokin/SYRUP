import sys
from pathlib import Path

T = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(T / "phase2_floating_checkpoint_cuda"), str(T / "phase2_floating_context_checkpoint"), str(T / "phase2_floating_checkpoint"), str(T / "phase2_floating_pond"),
                str(T / "phase2_native_floating_soil"), str(T / "phase2_compact_soil_profile"), str(T / "phase2_floating_held_context"), str(T / "pond_event"), str(T / "pond_windowed")]
