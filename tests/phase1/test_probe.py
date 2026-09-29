"""Real MAPLE state, the water seam MAPLE-SYRUP builds on, and the probe CLI.

Expected values are computed from the authored inputs (literal geometry and
fractions), never by calling the MAPLE routine under test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from maple_syrup.probe import (
    build_minimal_state,
    check_inventories,
    exercise_zero_demand_exchange,
    main,
    run_probe,
)

# Literal inputs matching build_minimal_state's defaults.
NY, NX, N_CLASSES = 3, 4, 6
CELL_AREA_M2 = 0.5 * 0.5
BULK_DENSITY = 1250.0
ACTIVE_TARGET_KG = BULK_DENSITY * CELL_AREA_M2 * 0.002
FRACTIONS = np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.10])
ATOL_KG = 1e-11


@pytest.fixture(scope="module")
def state():
    return build_minimal_state()


def _bed_by_class(voxel_column, active_layer):
    return voxel_column.mass_kg.sum(axis=(0, 1, 2)) + active_layer.mass_kg.sum(axis=(0, 1))


def test_bed_inventory_and_partition_match_authored_inputs(state):
    rows, cols = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    elevation = 0.12 + 0.01 * rows + 0.005 * cols
    expected_cell_class = (elevation * BULK_DENSITY * CELL_AREA_M2)[:, :, None] * FRACTIONS

    voxel = state.voxel_column.mass_kg
    active = state.active_layer.mass_kg
    assert voxel.shape == (NY, NX, 3, N_CLASSES) and voxel.dtype == np.float64
    assert active.shape == (NY, NX, N_CLASSES)

    np.testing.assert_allclose(voxel.sum(axis=2) + active, expected_cell_class, rtol=0, atol=ATOL_KG)
    np.testing.assert_allclose(active.sum(axis=2), ACTIVE_TARGET_KG, rtol=0, atol=ATOL_KG)
    np.testing.assert_allclose(active / active.sum(axis=2, keepdims=True), np.broadcast_to(FRACTIONS, active.shape), rtol=0, atol=1e-12)
    # Subactive column holds exactly the rest, bottom-up: one full voxel
    # (0.1 m) plus a partial one in every cell, nothing in the third.
    full_voxel_kg = BULK_DENSITY * CELL_AREA_M2 * 0.10
    np.testing.assert_allclose(voxel[:, :, 0, :].sum(axis=2), full_voxel_kg, rtol=0, atol=ATOL_KG)
    assert not np.any(voxel[:, :, 2, :])

    assert np.array_equal(state.water.depth_m, np.full((NY, NX), 0.001))
    assert not np.any(state.water.mobile_mass_by_cell_class_kg)
    report = check_inventories(state)
    assert report["maple_partition_validator"] == "passed"


def test_zero_demand_exchange_moves_nothing(state):
    from maple.core.backend import resolve_backend

    report = exercise_zero_demand_exchange(state, resolve_backend("numpy"))
    assert report["state_unchanged"] is True
    assert report["transfer_counter_delta"]["host_to_device"] == 0


def test_local_pickup_conserves_mass_but_water_step_has_no_lateral_transfer(state):
    """MAPLE's water step is local: mass picked up at cell A stays in A's
    mobile pool, and deposition requested at cell B is refused because B
    carries nothing. SYRUP must supply lateral mobile transfer itself
    (interface contract section 4.1)."""
    from maple.water import WaterProcessDemand, apply_water_process_demand

    removal = np.zeros((NY, NX, N_CLASSES))
    removal[0, 0, 0] = 0.01
    removal[0, 0, 2] = 0.02
    deposition = np.zeros_like(removal)
    deposition[2, 3, 0] = 0.01

    inputs = [a.copy() for a in (state.voxel_column.mass_kg, state.active_layer.mass_kg, state.water.depth_m)]
    result = apply_water_process_demand(
        state.voxel_column, state.active_layer, state.water, state.ledger,
        WaterProcessDemand(removal, deposition), state.geometry, state.grain_classes,
        state.mass_resolution_kg, adapter_name="phase1_test",
    )

    bed_before = _bed_by_class(state.voxel_column, state.active_layer)
    bed_after = _bed_by_class(result.new_voxel_column, result.new_active_layer)
    mobile_after = result.new_water.mobile_mass_by_cell_class_kg
    np.testing.assert_allclose(bed_before, bed_after + mobile_after.sum(axis=(0, 1)), rtol=0, atol=ATOL_KG)
    np.testing.assert_allclose(bed_before - bed_after, removal.sum(axis=(0, 1)), rtol=0, atol=ATOL_KG)
    np.testing.assert_allclose(mobile_after[0, 0], removal[0, 0], rtol=0, atol=ATOL_KG)
    assert not np.any(result.deposition_by_cell_class_kg)
    assert mobile_after[2, 3, 0] == 0.0
    # Active layer was refilled to target from the subactive column (MAPLE
    # may leave a sub-resolution replenishment residual).
    np.testing.assert_allclose(
        result.new_active_layer.mass_kg[0, 0].sum(), ACTIVE_TARGET_KG,
        rtol=0, atol=ATOL_KG + state.mass_resolution_kg,
    )
    # Depth passes through untouched and inputs are not mutated.
    assert np.array_equal(result.new_water.depth_m, state.water.depth_m)
    for before, after in zip(
        inputs, (state.voxel_column.mass_kg, state.active_layer.mass_kg, state.water.depth_m), strict=True
    ):
        assert np.array_equal(before, after)


def test_malformed_water_state_and_demand_are_refused_without_mutation(state):
    from maple.core.types.water import WaterState
    from maple.water import (
        WaterProcessDemand,
        WaterStateValidationError,
        apply_water_process_demand,
        validate_water_state,
    )

    wrong = WaterState(depth_m=np.zeros((NY, NX + 1)), mobile_mass_by_cell_class_kg=np.zeros((NY, NX, N_CLASSES)))
    with pytest.raises(WaterStateValidationError, match="shape"):
        validate_water_state(wrong, NY, NX, N_CLASSES)

    voxel_before = state.voxel_column.mass_kg.copy()
    bad = np.zeros((NY, NX, N_CLASSES - 1))
    with pytest.raises(ValueError, match="shape"):
        apply_water_process_demand(
            state.voxel_column, state.active_layer, state.water, state.ledger,
            WaterProcessDemand(bad, bad.copy()), state.geometry, state.grain_classes,
            state.mass_resolution_kg,
        )
    assert np.array_equal(state.voxel_column.mass_kg, voxel_before)


def test_selected_constant_depth_rule_leaves_solver_depth_alone(state):
    """The contract selects `constant_depth` so MAPLE's commit stage does
    not rewrite SYRUP-owned depth; MAPLE's default rule would."""
    from maple.water import apply_depth_after_bed_change

    depth = np.full((NY, NX), 0.004)
    bed_rise = np.full((NY, NX), 0.001)
    kept = apply_depth_after_bed_change(depth, bed_rise, state.geometry, "constant_depth")
    assert np.array_equal(kept.new_depth_m, depth)
    assert kept.displaced_volume_m3 == 0.0

    default = apply_depth_after_bed_change(depth, bed_rise, state.geometry, "constant_free_surface")
    np.testing.assert_allclose(default.new_depth_m, 0.003, rtol=0, atol=1e-15)


def test_unavailable_gpu_backend_fails_loudly_instead_of_falling_back():
    from maple.core.backend import BackendUnavailableError, cupy_available

    if cupy_available():
        pytest.skip("CuPy present; the device round trip is not exercised by default")
    with pytest.raises(BackendUnavailableError):
        run_probe(backend="cupy")


def test_cli_reports_real_maple_and_numpy_backend(capsys):
    assert main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["source_stable_during_probe"] is True
    assert report["backend"]["resolved"] == "numpy"
    assert report["gpu_readiness_claimed"] is False
    assert report["maple"]["dependency"]["project_name"] == "maple"
    assert report["zero_demand_exchange"]["state_unchanged"] is True


def test_cli_output_file_is_new_only(tmp_path, capsys):
    existing = tmp_path / "existing.json"
    existing.write_text("keep")
    assert main(["--output", str(existing)]) == 2
    assert existing.read_text() == "keep"

    target = tmp_path / "report.json"
    assert main(["--output", str(target)]) == 0
    assert json.loads(target.read_text()) == json.loads(capsys.readouterr().out)
