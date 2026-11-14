"""Fixtures of the Phase 2.5 vegetation-lifecycle suite: the ACTUAL LEAF candidate package (read-only, imported by path; LEAF_CANDIDATE_ROOT) and real native floating lifecycles (thin-lake storms). LEAF is NEVER faked:
when it cannot be imported the consumer tests skip (no claim). Helper module names are unique (`veg_lc_*`) so the older suites' collection does not collide."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

T = Path(__file__).resolve().parents[1]
TASK = Path(__file__).resolve().parents[3]
LEAF_ROOT = Path(os.environ.get("LEAF_CANDIDATE_ROOT", str(TASK / "leaf_candidate")))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(T / "phase2_floating_event_lifecycle"), str(T / "phase2_floating_checkpoint"), str(T / "phase2_floating_pond"),
                str(T / "phase2_floating_event_daily_bridge")]
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
