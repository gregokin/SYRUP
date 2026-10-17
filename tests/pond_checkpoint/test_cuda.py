"""Real CuPy restart of the pond event (same-backend), with an honest skip when no CUDA device is usable (not run by its author)."""

from __future__ import annotations

import numpy as np
import pytest

from maple_syrup.pond_checkpoint import PondCheckpointError, load_pond_checkpoint, save_pond_checkpoint

from .pond_fixture import IDENTITY, advance, assert_tree_equal, host, make_case


@pytest.fixture(scope="module")
def cp():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.zeros(1)
    except Exception as exc:  # noqa: BLE001 - CuPy raises several runtime types when no device is usable
        pytest.skip(f"no usable CUDA device: {exc}")
    return cupy


def test_cuda_wet_restart_equals_uninterrupted_and_stays_on_device(cp, tmp_path):
    from maple.core.backend import array_namespace

    case = make_case(xp=cp, backend="cuda", pickup_kg_per_class=5e-4)
    event = case.event()
    paused = advance(event, event.initial_state(case.bed, case.soil), 6)
    assert float(np.abs(host(paused.hydraulic.qx_m2_s)).max()) > 0.0
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)

    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    # device arrays are CuPy, the topology / datum / class-total anchors are host
    for array in (loaded.hydraulic.depth_m, loaded.hydraulic.qx_m2_s, loaded.phase_fraction, loaded.held_source_m,
                  loaded.bed.water.mobile_mass_by_cell_class_kg, loaded.water.rain, loaded.sediment.pickup_cell):
        assert array_namespace(array) is cp
    assert isinstance(loaded.topology.source_z_full, np.ndarray) and isinstance(loaded.datum.initial_full_m, np.ndarray)
    assert isinstance(loaded.sediment.inventory0, np.ndarray)
    assert_tree_equal(paused, loaded)
    assert_tree_equal(advance(event, paused, 6), advance(fresh_event, loaded, 6))


def test_cuda_checkpoint_refused_by_a_numpy_event_and_vice_versa(cp, tmp_path):
    cuda_case = make_case(xp=cp, backend="cuda", pickup_kg_per_class=5e-4)
    cuda_event = cuda_case.event()
    cuda_state = advance(cuda_event, cuda_event.initial_state(cuda_case.bed, cuda_case.soil), 3)
    cuda_path = save_pond_checkpoint(tmp_path / "cuda", cuda_state, cuda_event, IDENTITY)

    cpu_case = make_case(pickup_kg_per_class=5e-4)
    cpu_event = cpu_case.event()
    cpu_state = advance(cpu_event, cpu_event.initial_state(cpu_case.bed, cpu_case.soil), 3)
    cpu_path = save_pond_checkpoint(tmp_path / "cpu", cpu_state, cpu_event, IDENTITY)

    with pytest.raises(PondCheckpointError, match="cross-backend"):
        load_pond_checkpoint(cuda_path, cpu_case.event(), IDENTITY)
    with pytest.raises(PondCheckpointError, match="cross-backend"):
        load_pond_checkpoint(cpu_path, cuda_case.event(), IDENTITY)
