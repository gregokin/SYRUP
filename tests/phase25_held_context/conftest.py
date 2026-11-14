"""Fixtures of the dynamic-vegetation held-context suite. The imported `maple` must come from the pinned CPU materialization (file prefix) and its FULL package digest (fbc504bb…, 412 files), the actual LEAF package digest
(38009872…, 28 files, frozen H3 snapshot) are asserted before AND after the session; a missing pinned source is a setup failure, never a fallback. The expected MAPLE prefix / digest are configurable through
`HELD_CONTEXT_MAPLE_ROOT` / `HELD_CONTEXT_MAPLE_PACKAGE` (default: the CPU pin) so that the actual-GPU pin (same package digest) is selectable without editing a test; no CPU-only hard-code of the fixture path."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
T = _HERE.parent
for sub in ("phase25_vegetation_binding", "phase25_cached_inputs", "phase25_vegetation_lifecycle", "phase2_floating_event_daily_bridge", "phase2_floating_event_lifecycle", "phase2_floating_checkpoint",
            "phase2_floating_pond", "phase2_native_floating_soil", "phase2_floating_held_context", "phase2_compact_soil_profile"):
    sys.path.append(str(T / sub))
sys.path.insert(0, str(_HERE))
_LEAF_ROOT = Path(os.environ.get("LEAF_CANDIDATE_ROOT", str(T.parents[1] / "root_compact_helpers_h3_snapshot")))   # the FINAL frozen LEAF (never a mutable tree)
if str(_LEAF_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEAF_ROOT))

from hc_sources import (
    LEAF_PACKAGE,
    MAPLE_PACKAGE,
    MAPLE_ROOT,
)


def _digest(mod) -> dict:
    from maple_syrup.provenance import source_tree_digest
    d = source_tree_digest(Path(mod.__file__).resolve().parent)
    return {"module_file": str(mod.__file__), "digest": d.digest_sha256, "files": d.file_count}


@pytest.fixture(scope="session", autouse=True)
def pinned_sources():
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
    from leaf import state, traits
    from leaf import water_vegetation as wv
    return {"state": state, "traits": traits, "wv": wv}
