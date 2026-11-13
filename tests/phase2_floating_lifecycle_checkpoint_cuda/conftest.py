import sys
from pathlib import Path

T = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent)] + [str(T / x) for x in ("phase2_floating_lifecycle_checkpoint", "phase2_floating_event_daily_bridge_cuda", "phase2_floating_event_daily_bridge", "phase2_floating_event_lifecycle_cuda",
                                                                              "phase2_floating_event_lifecycle", "phase2_floating_event_refresh_cuda", "phase2_floating_event_refresh", "phase2_floating_daily",
                                                                              "phase2_floating_continued_checkpoint_cuda", "phase2_floating_continued_checkpoint", "phase2_floating_checkpoint_cuda",
                                                                              "phase2_floating_checkpoint", "phase2_floating_context_checkpoint", "phase2_floating_pond", "phase2_native_floating_soil",
                                                                              "phase2_compact_soil_profile", "pond_event", "pond_windowed")]
