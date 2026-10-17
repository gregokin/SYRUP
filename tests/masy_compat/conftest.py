"""Bind THIS tree's `maple_syrup` for tests/masy_compat and build a small, real MAPLE Dense bed through the existing
native initializers (`probe.build_minimal_state`, MAPLE's committed-topography and availability factories). No terrain,
saved arrays or payloads are read; every fixture is a few cells and is rebuilt per test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
sys.path.insert(0, str(CANDIDATE_SRC))
sys.path.insert(1, str(Path(__file__).resolve().parent))  # masy_compat_support

import maple_syrup  # noqa: E402

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/masy_compat must exercise {CANDIDATE_SRC}, but maple_syrup was imported from {_PACKAGE_DIR}")


def pytest_report_header(config):
    """Record which MAPLE the run actually bound and whether it exposes the optional windowed class."""
    try:
        import maple
        from maple.core.types import voxel
    except Exception as exc:  # noqa: BLE001 - reported verbatim, the tests then fail on their own
        return [f"masy_compat: MAPLE import failed ({type(exc).__name__}: {exc})"]
    windowed = 'WindowedVoxelColumnState' in vars(voxel)
    return [f"masy_compat: maple_syrup={_PACKAGE_DIR}",
            f"masy_compat: maple={Path(maple.__file__).resolve().parent} WindowedVoxelColumnState present={windowed}"]


@pytest.fixture
def dense_bed():
    """(state, context, graph, reference) of a genuine MAPLE Dense (`VoxelColumnState`) bed: the phase5 3x4 fixture."""
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_availability import (
        sediment_availability_state_from_active_layer,
    )
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.surface.topographic_commit import (
        initial_committed_topography_state_from_physical_state,
    )

    from maple_syrup.probe import build_minimal_state
    from maple_syrup.routing import build_routing_graph
    from maple_syrup.sediment_bed import BedContext, BedState, terrain_reference

    p = build_minimal_state()
    committed = initial_committed_topography_state_from_physical_state(
        p.voxel_column, p.active_layer, p.geometry, p.mass_resolution_kg)
    available = sediment_availability_state_from_active_layer(p.active_layer.mass_kg,
                                                              np.ones_like(p.active_layer.mass_kg))
    state = BedState(p.voxel_column, p.active_layer, p.water, p.ledger, committed, available)
    ctx = BedContext(p.geometry, p.grain_classes, p.mass_resolution_kg, TopographicCommitSpec(),
                     AvalanchingSpec(enabled=False), np.ones_like(p.active_layer.mass_kg))
    z = np.ones((5, 6))
    z[1:-1, 1:-1] = committed_elevation_m(committed)
    z[0, 1:-1] = .1 + .005 * np.arange(4)
    exports = np.zeros_like(z, bool)
    exports[0, 1:-1] = True
    graph = build_routing_graph(z, exports, np.full((3, 4), 21.45), .5)
    ref = terrain_reference(z, exports, state, graph)
    return state, ctx, graph, ref


from masy_compat_support import NATIVE_PHYSICAL_ENTRIES  # noqa: E402,F401  (the spied native entries, test-only list)


@pytest.fixture
def native_spy(monkeypatch, dense_bed):
    """Replace every available native physical entry with a recorder that fails loudly, AFTER the real Dense fixture
    is built (hence the `dense_bed` dependency). Returns the call list.

    All entry modules are imported BEFORE the first spy is installed (a later-imported module doing
    `from module import function` would otherwise capture an earlier spy permanently). The spies are undone explicitly at
    teardown and the fixture then fails if any `maple*` / `maple_syrup*` module attribute is still a spy."""
    from masy_compat_support import NativeEntrySpy, leaked_native_spies, resolve_native_entries

    calls: list[str] = []
    patch = pytest.MonkeyPatch()
    try:
        for module, attribute, _original in resolve_native_entries(NATIVE_PHYSICAL_ENTRIES):
            patch.setattr(module, attribute, NativeEntrySpy(f'{module.__name__}.{attribute}', calls))
        yield calls
    finally:
        patch.undo()
    assert leaked_native_spies() == []
