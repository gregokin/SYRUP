"""Real Native/LEAF helper closure; no substitute projector or runtime."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
_HELPERS = (
    "phase2_wind_handoff", "phase25_vegetation_lifecycle", "phase2_floating_lifecycle_checkpoint",
    "phase2_floating_event_daily_bridge", "phase2_floating_event_lifecycle", "phase2_floating_daily",
    "phase2_floating_checkpoint", "phase2_floating_pond", "phase2_native_floating_soil",
    "phase2_floating_continued_checkpoint", "phase2_floating_context_checkpoint", "phase2_compact_soil_profile",
    "pond_event", "pond_windowed",
)
sys.path[:0] = [str(_TESTS / name) for name in _HELPERS]
_ROOT = os.environ.get("LEAF_CANDIDATE_ROOT")
if _ROOT:
    sys.path.insert(0, str(Path(_ROOT).resolve()))


@pytest.fixture(scope="session")
def leaf():
    if not _ROOT:
        pytest.skip("LEAF_CANDIDATE_ROOT is not configured; genuine vegetation qualification is unissued")
    from leaf import state, traits, water_vegetation

    return {"state": state, "traits": traits, "wv": water_vegetation, "root": Path(_ROOT).resolve()}
