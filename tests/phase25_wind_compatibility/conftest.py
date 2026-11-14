"""Fixtures of the Phase 2.5 wind-compatibility suite: the ACTUAL LEAF candidate package (read-only, by path; LEAF_CANDIDATE_ROOT, default the final frozen H3 snapshot) plus the WH1 native-protocol builders and the vegetation-lifecycle
builders (unique helper module names). LEAF is never faked: if it cannot be imported the consumer tests skip (no claim)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

T = Path(__file__).resolve().parents[1]
TASK = Path(__file__).resolve().parents[3]
LEAF_ROOT = Path(os.environ.get("LEAF_CANDIDATE_ROOT", str(TASK / "root_compact_helpers_h3_snapshot")))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(T / "phase2_wind_handoff"), str(T / "phase25_vegetation_lifecycle"), str(T / "phase2_floating_lifecycle_checkpoint"),
                str(T / "phase2_floating_event_daily_bridge"), str(T / "phase2_floating_event_lifecycle"), str(T / "phase2_floating_daily"), str(T / "phase2_floating_checkpoint"), str(T / "phase2_floating_pond"),
                str(T / "phase2_native_floating_soil"), str(T / "phase2_floating_continued_checkpoint"), str(T / "phase2_floating_context_checkpoint"), str(T / "pond_event"), str(T / "pond_windowed")]
if str(LEAF_ROOT) not in sys.path and (LEAF_ROOT / "leaf" / "water_vegetation.py").is_file():
    sys.path.insert(0, str(LEAF_ROOT))


@pytest.fixture(scope="session")
def leaf():
    try:
        from leaf import state, traits
        from leaf import water_vegetation as wv
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the actual LEAF candidate is not importable ({type(exc).__name__}: {exc}); the consumer is UNQUALIFIED (no claim)")
    return {"state": state, "traits": traits, "wv": wv, "root": LEAF_ROOT}
