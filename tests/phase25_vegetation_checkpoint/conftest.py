"""Fixtures of the Phase 2.5 vegetation physical-checkpoint suite: the ACTUAL LEAF candidate package (read-only, by path; LEAF_CANDIDATE_ROOT) and REAL native lifecycle states. LEAF is never faked: if it cannot be imported the
consumer tests skip (no claim). Helper module names are unique (`vck_*`)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

T = Path(__file__).resolve().parents[1]
TASK = Path(__file__).resolve().parents[3]
LEAF_ROOT = Path(os.environ.get("LEAF_CANDIDATE_ROOT", str(TASK / "leaf_candidate")))
sys.path[:0] = [str(Path(__file__).resolve().parent)] + [str(T / x) for x in (
    "phase25_vegetation_lifecycle", "phase25_vegetation_frames", "phase2_floating_lifecycle_checkpoint", "phase2_floating_event_daily_bridge", "phase2_floating_continued_checkpoint", "phase2_floating_event_lifecycle", "phase2_floating_daily",
    "phase2_floating_checkpoint", "phase2_floating_pond", "phase2_native_floating_soil", "phase2_floating_context_checkpoint", "pond_event", "pond_windowed")]
if str(LEAF_ROOT) not in sys.path and (LEAF_ROOT / "leaf" / "water_vegetation.py").is_file():
    sys.path.insert(0, str(LEAF_ROOT))


MAPLE_ROOT = Path(os.environ.get("VEG_CHECKPOINT_MAPLE_ROOT", "/home/okin/SYRUP/agent_handoffs/tasks/phase1_publication_c1_20261112/runs/runtime-store-cpu/objects/maple-materialized/5d850789f9e82dc456227cb1eb4dfcc255b3ee4dfb0ed74883813c93e98d0ab3"))
MAPLE_PACKAGE = os.environ.get("VEG_CHECKPOINT_MAPLE_PACKAGE", "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e")
LEAF_PACKAGE = "38009872f8969f86b56c720adc1b5d9ffefc554fb9cacf90855d999db604530f"


def _digest(mod) -> dict:
    from maple_syrup.provenance import source_tree_digest

    d = source_tree_digest(Path(mod.__file__).resolve().parent)
    return {"module_file": str(mod.__file__), "digest": d.digest_sha256, "files": d.file_count}


@pytest.fixture(scope="session", autouse=True)
def pinned_sources():
    """The imported MAPLE must come from the configured immutable materialization (the CPU pin by default; the actual-GPU pin through VEG_CHECKPOINT_MAPLE_ROOT), with the FULL package digest (fbc504bb..., 412 files), and the actual LEAF
    package must be the FINAL frozen H3 (38009872..., 28 files): asserted before AND after the session. A missing pinned source is a setup failure, never a fallback."""
    assert (MAPLE_ROOT / "source" / "src" / "maple" / "__init__.py").is_file(), f"the configured MAPLE source does not exist: {MAPLE_ROOT}"
    import maple

    import maple_syrup

    assert Path(maple.__file__).resolve().is_relative_to(MAPLE_ROOT / "source" / "src"), f"MAPLE comes from {maple.__file__}"
    before = {"MAPLE": _digest(maple), "SYRUP": _digest(maple_syrup)}
    assert before["MAPLE"]["digest"] == MAPLE_PACKAGE and before["MAPLE"]["files"] == 412, before["MAPLE"]
    import leaf

    before["LEAF"] = _digest(leaf)
    assert before["LEAF"]["digest"] == LEAF_PACKAGE and before["LEAF"]["files"] == 28, before["LEAF"]
    yield before
    after = {"MAPLE": _digest(maple), "SYRUP": _digest(maple_syrup), "LEAF": _digest(leaf)}
    assert after == before, (before, after)


@pytest.fixture(scope="session")
def leaf():
    try:
        from leaf import state, traits
        from leaf import water_vegetation as wv
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the actual LEAF candidate is not importable ({type(exc).__name__}: {exc}); the consumer is UNQUALIFIED (no claim)")
    return {"state": state, "traits": traits, "wv": wv, "root": LEAF_ROOT}
