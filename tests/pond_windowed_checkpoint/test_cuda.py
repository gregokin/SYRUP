"""Real CuPy restart of the windowed pond event (same backend), with an honest skip when no CUDA device is usable. The resident
window, base, prefix metadata and the derived height packets must come back on the DEVICE through MAPLE's counted moves; there
is no implicit NumPy conversion of device data (not run by its author)."""

from __future__ import annotations

import numpy as np
import pytest

from maple_syrup.pond_checkpoint import PondCheckpointError, load_pond_checkpoint, save_pond_checkpoint

from .wfixture import IDENTITY, advance, assert_tree_equal, host, make_case, physical, published, store_pins


@pytest.fixture(scope="module")
def cp():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.zeros(1)
    except Exception as exc:  # noqa: BLE001 - CuPy raises several runtime types when no device is usable
        pytest.skip(f"no usable CUDA device: {exc}")
    return cupy


def test_cuda_wet_restart_moves_the_height_packets_and_continues_exactly(cp, tmp_path):
    from maple.core.backend import array_namespace
    from maple.surface.voxels.windowed_prefix import require_prefix_packets

    case = make_case(tmp_path, xp=cp, backend="cuda", pickup_kg_per_class=5e-4)
    event = case.event()
    paused = published(advance(event, event.initial_state(case.bed, case.soil), 6), event)
    physical(paused)
    assert np.any(host(paused.bed.voxel_column.base) > 0)
    pins = store_pins(case.store_dir)
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)

    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    column = loaded.bed.voxel_column
    for array in (loaded.hydraulic.depth_m, loaded.hydraulic.qx_m2_s, loaded.phase_fraction, loaded.held_source_m,
                  loaded.bed.water.mobile_mass_by_cell_class_kg, loaded.water.rain, loaded.sediment.pickup_cell,
                  column.window_kg, column.base, column.deep_by_class_kg, column.deep_total_kg, column.deep_room_kg,
                  loaded.bed.active_layer.mass_kg):
        assert array_namespace(array) is cp
    # the hidden derived packets are valid, bound to THIS column and on the device (never silently NumPy)
    packets = require_prefix_packets(column, operation="test")
    assert array_namespace(packets.packet) is cp and packets.describes(column)
    assert isinstance(loaded.topology.source_z_full, np.ndarray) and isinstance(loaded.datum.initial_full_m, np.ndarray)
    assert column.pending_deep is None
    assert column.deep_store.manifest_digest == paused.bed.voxel_column.deep_store.manifest_digest
    assert store_pins(case.store_dir) == pins
    assert_tree_equal(paused, loaded)
    assert_tree_equal(advance(event, paused, 6), advance(fresh_event, loaded, 6))


def test_cuda_overlay_v2_roundtrip_and_device_continuation(cp, tmp_path):
    """A real v2 (overlay) deep version under a CuPy event: physical / logical identities are recorded, the packets come back
    on the device, and the continuation equals the uninterrupted device run."""
    from maple.core.backend import array_namespace
    from maple.surface.voxels.windowed_prefix import require_prefix_packets

    case = make_case(tmp_path, xp=cp, backend="cuda", pickup_kg_per_class=5e-4, encoding="overlay")
    event = case.event()
    paused = published(advance(event, event.initial_state(case.bed, case.soil), 5), event)
    store = paused.bed.voxel_column.deep_store
    assert store.format_version == 2 and store.manifest_digest != store.logical_manifest_digest
    physical(paused)
    pins = store_pins(case.store_dir)
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)

    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    column = loaded.bed.voxel_column
    assert column.deep_store.format_version == 2
    assert column.deep_store.manifest_digest == store.manifest_digest
    assert column.deep_store.logical_manifest_digest == store.logical_manifest_digest
    assert array_namespace(column.window_kg) is cp and array_namespace(column.base) is cp
    packets = require_prefix_packets(column, operation="test")
    assert array_namespace(packets.packet) is cp and packets.describes(column) and column.pending_deep is None
    assert store_pins(case.store_dir) == pins
    assert_tree_equal(paused, loaded)
    assert_tree_equal(published(advance(event, paused, 5), event), published(advance(fresh_event, loaded, 5), fresh_event))


def test_cuda_and_numpy_checkpoints_do_not_cross(cp, tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "c").mkdir()
    cuda_case = make_case(tmp_path / "g", xp=cp, backend="cuda", pickup_kg_per_class=5e-4)
    cuda_event = cuda_case.event()
    cuda_state = published(advance(cuda_event, cuda_event.initial_state(cuda_case.bed, cuda_case.soil), 3), cuda_event)
    cuda_path = save_pond_checkpoint(tmp_path / "cuda", cuda_state, cuda_event, IDENTITY)

    cpu_case = make_case(tmp_path / "c", pickup_kg_per_class=5e-4)
    cpu_event = cpu_case.event()
    cpu_state = published(advance(cpu_event, cpu_event.initial_state(cpu_case.bed, cpu_case.soil), 3), cpu_event)
    cpu_path = save_pond_checkpoint(tmp_path / "cpu", cpu_state, cpu_event, IDENTITY)

    with pytest.raises(PondCheckpointError, match="cross-backend"):
        load_pond_checkpoint(cuda_path, cpu_case.event(), IDENTITY)
    with pytest.raises(PondCheckpointError, match="cross-backend"):
        load_pond_checkpoint(cpu_path, cuda_case.event(), IDENTITY)


def test_cuda_completed_handoff_roundtrip(cp, tmp_path):
    from .wfixture import dry_case

    case = dry_case(tmp_path, xp=cp, backend="cuda")
    event = case.event()
    handoff = event.complete(event.initial_state(case.bed, case.soil), 120.0)
    path = save_pond_checkpoint(tmp_path / "done", handoff, event, IDENTITY)
    loaded = load_pond_checkpoint(path, case.event(), IDENTITY, allow_complete=True)
    assert_tree_equal(handoff, loaded)
