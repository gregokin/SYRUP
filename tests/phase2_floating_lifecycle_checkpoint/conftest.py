import sys
from pathlib import Path

T = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(T / "phase2_floating_event_daily_bridge"), str(T / "phase2_floating_continued_checkpoint"), str(T / "phase2_floating_event_lifecycle"),
                str(T / "phase2_floating_daily"), str(T / "phase2_floating_checkpoint"), str(T / "phase2_floating_pond"), str(T / "phase2_native_floating_soil"), str(T / "phase2_floating_context_checkpoint"), str(T / "pond_event"), str(T / "pond_windowed")]
