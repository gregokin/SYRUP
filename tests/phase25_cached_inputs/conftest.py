"""Fixtures of the cached-inputs suite. The S0 support module (distinct names) is reused by path; LEAF is the ACTUAL package of LEAF_CANDIDATE_ROOT (default: Root's frozen H3 compact-helpers
snapshot), never faked. The native MAPLE source is asserted EVERY session: the imported `maple` must come from the pinned 9f materialization (file prefix) and its FULL package digest must equal
the recorded one before AND after the tests (no live-MAPLE fallback)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase25_vegetation_binding"))
from cached_support import LEAF_PACKAGE, MAPLE_PACKAGE, configured_maple_root


def _digest(mod) -> dict:
    from maple_syrup.provenance import source_tree_digest
    d = source_tree_digest(Path(mod.__file__).resolve().parent)
    return {"module_file": str(mod.__file__), "digest": d.digest_sha256, "files": d.file_count}


@pytest.fixture(scope="session", autouse=True)
def pinned_sources():
    try:
        root = configured_maple_root()  # EXPLICIT configuration only: the CPU constant, or (GPU lane) the root the lane configured; anything else is a hard failure
    except ValueError as exc:
        pytest.fail(f"the expected MAPLE materialization is not (validly) configured: {exc}", pytrace=False)
    import maple

    import maple_syrup
    assert Path(maple.__file__).resolve().is_relative_to(root / "source" / "src"), f"MAPLE comes from {maple.__file__}, not the configured qualified materialization {root}"
    before = {"MAPLE": _digest(maple), "SYRUP": _digest(maple_syrup)}
    assert before["MAPLE"]["digest"] == MAPLE_PACKAGE, before["MAPLE"]
    try:
        import leaf
        before["LEAF"] = _digest(leaf)
        assert before["LEAF"]["digest"] == LEAF_PACKAGE, before["LEAF"]
    except ImportError:
        before["LEAF"] = None
    yield before
    if before["LEAF"] is not None:
        import leaf
        assert _digest(leaf)["digest"] == LEAF_PACKAGE
    after = {"MAPLE": _digest(maple), "SYRUP": _digest(maple_syrup)}
    assert after["MAPLE"]["digest"] == MAPLE_PACKAGE and after["SYRUP"]["digest"] == before["SYRUP"]["digest"], (before, after)
    out = os.environ.get("CACHED_INPUTS_SOURCES_OUT")
    if out:
        Path(out).write_text(json.dumps({"before": before, "after": after}, indent=2) + "\n")


@pytest.fixture(scope="session")
def leaf():
    try:
        import leaf as pkg  # noqa: F401
        from leaf import state, traits
        from leaf import water_vegetation as wv
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the actual LEAF package is not importable ({type(exc).__name__}: {exc}); the consumer is UNQUALIFIED (no claim)")
    for name in ("compact_content_sha256", "rederive_fields_from_compact", "preferential_inputs_from_compact"):
        if not hasattr(wv, name):
            pytest.skip(f"the imported LEAF lacks the public {name} (H1/H2/H3): the cached-inputs consumer is UNQUALIFIED (no claim)")
    return {"state": state, "traits": traits, "wv": wv}
