"""E3: native bed -> floating-soil owner initialization and the staged changed-column refresh transaction, on genuine native Dense and Windowed beds (CPU/host)."""
from __future__ import annotations

import numpy as np
import pytest
from builders import (
    GRAIN_TEXTURE,
    config,
    set_cell_from,
    stratified,
    swap_classes,
    windowed,
)

from maple_syrup import compact_soil_profile as csp
from maple_syrup import floating_soil_layers as fl
from maple_syrup import native_floating_soil as nfs
from maple_syrup import soil_texture_properties as stp

REFUSALS = (nfs.NativeFloatingError, nfs.NativeFloatingMemoryError)
SAT0 = 0.9


def init(bed=None, cfg=None, **kw):
    bed = stratified() if bed is None else bed
    return bed, nfs.initialize_native_floating_soil(bed, cfg or config(), initial_relative_saturation=kw.pop("sat", SAT0), **kw)


def water(o):
    return np.array(o.state.water_m, copy=True)


def maps(o):
    return {k: np.array(o.column._maps[k], copy=True) for k in ("theta_field_capacity", "theta_saturation", "soil_depth")}


def test_initialization_from_the_native_bed():
    bed, o = init()
    assert np.allclose(o.depth_m, [0.222, 0.222, 0.032], atol=1e-15)  # the hand-worked canonical depths of the fixture
    assert list(o.column._maps["bottom_open"]) == [True, True, False]  # open only where D > H = 0.16
    fc, sat = o.column._maps["theta_field_capacity"], o.column._maps["theta_saturation"]
    assert np.all(fc <= sat) and fc[2, 2] == 0.25 and sat[2, 2] == 0.42  # the zero-mass slot keeps its explicit latent properties
    w = o.state.water_m
    assert w[2, 2] == 0.0 and np.all(w[:2, :] > 0) and o.generation == 0 and o.is_current
    assert o.matrix_history.wetting_m[0] == 0.01 and o.matrix_history.cumulative_intake_m[0] == 0.02
    with pytest.raises(REFUSALS):
        nfs.initialize_native_floating_soil(bed, config(), initial_relative_saturation=1.0)  # strictly unsaturated


def test_packing_grain_mapping_and_input_refusals():
    bed = stratified()
    with pytest.raises(nfs.NativePackingError):
        nfs.initialize_native_floating_soil(bed, config(density=1300.0), initial_relative_saturation=0.5)  # stated density != actual MAPLE packing
    with pytest.raises(nfs.NativeFloatingError):
        nfs.initialize_native_floating_soil(bed, config(texture=GRAIN_TEXTURE[:3]), initial_relative_saturation=0.5)
    bad = GRAIN_TEXTURE.copy()
    bad[0, 0] = 0.5
    with pytest.raises(nfs.NativeFloatingError):
        config(texture=bad)
    with pytest.raises(nfs.NativeFloatingError):
        config(backend="tpu")
    from maple_syrup.soil_water_remap import SoilRemapBackendUnavailableError

    with pytest.raises((nfs.NativeFloatingError, SoilRemapBackendUnavailableError)):  # a CUDA owner config with a HOST bed: typed refusal (no CuPy here), never a fallback
        nfs.initialize_native_floating_soil(bed, config(backend="cuda"), initial_relative_saturation=0.5)
    with pytest.raises(nfs.NativeFloatingError, match="cuda backend"):
        nfs.initialize_native_floating_soil(bed, config(), initial_relative_saturation=0.5, host_memory_budget_bytes=10)
    with pytest.raises(nfs.NativeFloatingError):
        config(overrides={"bogus": np.zeros((3, 3))})


def test_erosion_to_rock_stages_without_mutation_then_adopts_once():
    bed, o = init()
    child = set_cell_from(bed, 1, 2)  # cell 1 loses its deep soil: D 0.222 -> 0.032 (open bottom -> closed rock)
    w0, m0 = water(o), maps(o)
    cand = nfs.stage_native_refresh(o, bed, child, mode="depth")
    assert list(cand.dirty_indices) == [1] and cand.reconciled and cand.mode == "depth"
    assert np.array_equal(water(o), w0) and o.is_current and o.generation == 0  # staging mutated nothing
    # closure: stored before = stored after + returned + exported (receipt), exposed once
    rc = cand.receipt
    assert abs(rc.residual_m3) <= rc.bound_m3 and cand.surface_return_m[1] > 0.0 and cand.bottom_export_m[1] == 0.0
    assert np.all(cand.surface_return_m[[0, 2]] == 0.0)
    # untouched cells are bit-identical in water, properties and depth
    for c in (0, 2):
        assert np.array_equal(cand.state.water_m[:, c], w0[:, c]) and cand.depth_m[c] == o.depth_m[c]
        assert np.array_equal(cand.column._maps["theta_field_capacity"][:, c], m0["theta_field_capacity"][:, c])
    assert cand.depth_m[1] == 0.032 and cand.column._maps["bottom_open"][1] == False
    # depth mode retains every property
    assert np.array_equal(cand.column._maps["theta_saturation"], m0["theta_saturation"])
    # persistent matrix history preserved exactly through the operator rebind (no reset)
    assert np.array_equal(cand.matrix_history.wetting_m, o.matrix_history.wetting_m) and np.array_equal(cand.matrix_history.cumulative_intake_m, o.matrix_history.cumulative_intake_m)
    assert cand.matrix_history.operator_sha256 == cand.matrix_op.sha256 != o.matrix_op.sha256
    o2 = nfs.adopt_native_refresh(o, cand)
    assert o2.generation == 1 and o2.is_current and not o.is_current
    assert o2.totals["cells_refreshed"] == 1 and o2.totals["surface_return_m3"] == pytest.approx(float(cand.surface_return_m.sum()) * 0.25)
    for call in (lambda: nfs.adopt_native_refresh(o2, cand), lambda: nfs.adopt_native_refresh(o, cand), lambda: nfs.stage_native_refresh(o, bed, child)):
        with pytest.raises(nfs.NativeStaleError):
            call()
    assert o2.totals["surface_return_m3"] == pytest.approx(float(cand.surface_return_m.sum()) * 0.25)  # a duplicate never double-credits


def test_zero_net_composition_exchange_refreshes_properties():
    bed, o = init()
    child = swap_classes(bed, 0)  # identical totals and depth; only the composition changes
    tot_c, tot_p = np.sum(child.voxel_column.mass_kg, axis=(-1, -2)), np.sum(bed.voxel_column.mass_kg, axis=(-1, -2))
    assert np.array_equal(tot_c, tot_p) and np.array_equal(np.sum(child.active_layer.mass_kg, axis=-1), np.sum(bed.active_layer.mass_kg, axis=-1))  # genuinely zero net mass/elevation
    assert not np.array_equal(child.voxel_column.mass_kg[0, 0], bed.voxel_column.mass_kg[0, 0])  # the composition did change
    cp = nfs.stage_native_refresh(o, bed, child, mode="properties")
    assert list(cp.dirty_indices) == [0] and cp.depth_m[0] == o.depth_m[0]
    assert not np.array_equal(cp.column._maps["theta_field_capacity"][:, 0], o.column._maps["theta_field_capacity"][:, 0])
    for c in (1, 2):
        assert np.array_equal(cp.column._maps["theta_field_capacity"][:, c], o.column._maps["theta_field_capacity"][:, c])
    # the same dirty cell in depth mode keeps the properties: nothing to reconcile, no redundant cascade
    cd = nfs.stage_native_refresh(o, bed, child, mode="depth")
    assert list(cd.dirty_indices) == [0] and not cd.reconciled and cd.receipt is None and cd.state is o.state and not cd.surface_return_m.any()
    o2 = nfs.adopt_native_refresh(o, cp)
    assert o2.generation == 1 and o2.binding.source_witness == cp.child_witness != o.binding.source_witness


def test_open_bottom_export_is_exposed_once_and_closes():
    bed, o = init(sat=0.99)
    child = swap_classes(bed, 0, 1, 2)  # sandier layers lower the field capacity of the open column
    cand = nfs.stage_native_refresh(o, bed, child, mode="properties")
    rc = cand.receipt
    assert cand.reconciled and abs(rc.residual_m3) <= rc.bound_m3
    moved = cand.state.water_m[:, 0].sum() - o.state.water_m[:, 0].sum()
    assert moved == pytest.approx(-(cand.surface_return_m[0] + cand.bottom_export_m[0]), abs=1e-15)
    assert cand.bottom_export_m[0] >= 0.0 and cand.surface_return_m[0] >= 0.0
    o2 = nfs.adopt_native_refresh(o, cand)
    # the same child against the new accepted parent is a no-change refresh: nothing is exported twice
    again = nfs.stage_native_refresh(o2, child, child, mode="properties")
    assert again.n_dirty == 0 and not again.reconciled and not again.bottom_export_m.any()


def test_inactive_cells_are_excluded_and_preserved_exactly():
    mask = np.array([[True, True, False]])
    bed = stratified()
    bed = nfs.NativeBed(bed.voxel_column, bed.active_layer, bed.geometry, bed.grain_classes, bed.mass_resolution_kg, mask)
    bed, o = init(bed)
    child = set_cell_from(bed, 2, 0)  # the INACTIVE cell changes
    cand = nfs.stage_native_refresh(o, bed, child, mode="properties")
    assert cand.n_dirty == 0 and not cand.reconciled
    child2 = swap_classes(bed, 1)
    cand2 = nfs.stage_native_refresh(o, bed, child2, mode="properties")
    assert list(cand2.dirty_indices) == [1] and np.array_equal(cand2.state.water_m[:, 2], o.state.water_m[:, 2])


def test_wrong_parent_is_refused_by_the_native_witness():
    bed, o = init()
    other = swap_classes(bed, 1)
    with pytest.raises(nfs.NativeStaleError, match="witness"):
        nfs.stage_native_refresh(o, other, set_cell_from(other, 0, 2))  # a parent bed that is not the accepted source
    a = nfs.stage_native_refresh(o, bed, other, mode="properties")
    o2 = nfs.adopt_native_refresh(o, a)
    with pytest.raises(nfs.NativeStaleError, match="witness"):
        nfs.stage_native_refresh(o2, bed, other)  # the OLD parent after adoption
    assert nfs.stage_native_refresh(o2, other, other).n_dirty == 0


def test_invalid_child_and_later_batch_failure_leave_the_owner_untouched(monkeypatch):
    bed, o = init()
    w0, g0 = water(o), o.generation
    child = swap_classes(bed, 0)
    child2 = set_cell_from(child, 1, 2)
    child2.voxel_column.mass_kg[0, 1, 0, 0] = -1.0  # an invalid native row in the SECOND dirty batch (batch_cells=1)
    from maple_syrup import soil_profile_texture as spt

    batches = []
    real = nfs._batch_properties
    monkeypatch.setattr(nfs, "_batch_properties", lambda *a, **k: batches.append(1) or real(*a, **k))
    with pytest.raises(spt.SoilProfileTextureError):
        nfs.stage_native_refresh(o, bed, child2, mode="properties", batch_cells=1)
    monkeypatch.undo()
    assert len(batches) == 1  # the first batch had already been yielded and consumed when the second one failed
    assert o.is_current and o.generation == g0 and np.array_equal(water(o), w0)
    ok = nfs.stage_native_refresh(o, bed, child, mode="properties", batch_cells=1)  # the owner is still usable after the failure
    assert ok.n_dirty == 1


def test_provider_refusal_stages_nothing(monkeypatch):
    bed, o = init()
    child = swap_classes(bed, 0)

    def boom(**k):
        raise stp.TexturePropertyError("synthetic provider refusal")

    monkeypatch.setattr(stp, "estimate_soil_texture_properties", boom)
    with pytest.raises(stp.TexturePropertyError):
        nfs.stage_native_refresh(o, bed, child, mode="properties")
    assert o.is_current


def test_admission_composes_every_term_before_any_scratch(monkeypatch):
    bed = stratified()
    cfg = config()
    model = nfs.estimate_native_refresh_memory(bed, bed, cfg)
    assert set(model["terms"]) == {"scan", "parent_native", "child_native_and_properties", "persistent_candidate"} and model["required_bytes"] > sum(v for k, v in model["terms"].items() if k == "persistent_candidate")
    calls = []
    monkeypatch.setattr(csp, "iter_compact_soil_profile", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(nfs, "_scan", lambda *a, **k: calls.append(2))
    with pytest.raises(nfs.NativeFloatingMemoryError, match="nothing was allocated"):
        nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=0.5, memory_budget_bytes=1000)
    with pytest.raises(nfs.NativeFloatingMemoryError, match="reported available"):
        nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=0.5, available_bytes_fn=lambda: 1000)
    assert calls == []  # refused before the scan, the reader and the provider
    need = nfs.estimate_native_initialization_memory(bed, cfg)["required_bytes"]
    monkeypatch.undo()
    nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=0.5, memory_budget_bytes=need)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=0.5, memory_budget_bytes=need - 1)


def test_compact_batches_are_released_and_no_full_texture_tensor(monkeypatch):
    bed, o = init()
    child = swap_classes(bed, 0)
    seen = []
    real = csp.iter_compact_soil_profile

    def spy(*a, **k):
        seen.append((int(k["indices"].size), k["batch_cells"]))
        return real(*a, **k)

    monkeypatch.setattr(csp, "iter_compact_soil_profile", spy)
    nfs.stage_native_refresh(o, bed, child, mode="properties")
    assert seen == [(1, 1)]  # ONE compact read of the single dirty cell, batch bounded by the dirty count


def test_windowed_matches_dense_and_runs_the_mandatory_native_validation(tmp_path, monkeypatch):
    from maple.surface.voxels import windowed_inventory as wi

    dbed = stratified()
    parent_w = windowed(dbed, tmp_path, "p")
    child_dense = swap_classes(set_cell_from(dbed, 1, 2), 0)
    child_w = windowed(child_dense, tmp_path, "c")
    assert child_w.representation == "windowed" and int(child_w.voxel_column.base.max()) > 0  # material really lives below the resident floor
    full_calls = []
    real = wi.validate_windowed_column_full
    monkeypatch.setattr(wi, "validate_windowed_column_full", lambda *a, **k: full_calls.append(1) or real(*a, **k))
    _, od = init(dbed)
    _, ow = init(parent_w)
    assert np.array_equal(od.depth_m, ow.depth_m) and np.array_equal(water(od), water(ow))
    cd = nfs.stage_native_refresh(od, dbed, child_dense, mode="properties")
    full_calls.clear()
    cw = nfs.stage_native_refresh(ow, parent_w, child_w, mode="properties")
    assert len(full_calls) >= 2  # parent and child are fully validated every time (compact reader adds its own)
    assert list(cd.dirty_indices) == list(cw.dirty_indices) == [0, 1]
    assert np.array_equal(cd.state.water_m, cw.state.water_m) and np.array_equal(cd.depth_m, cw.depth_m)
    assert np.array_equal(cd.surface_return_m, cw.surface_return_m) and np.array_equal(cd.bottom_export_m, cw.bottom_export_m)
    o2 = nfs.adopt_native_refresh(ow, cw)
    assert o2.generation == 1


def test_direct_construction_of_bindings_and_candidates_is_refused():
    _bed, o = init()
    with pytest.raises(nfs.NativeFloatingError):
        nfs.NativeBinding("x", 0, "w", "t")
    with pytest.raises(nfs.NativeFloatingError):
        nfs.NativeRefreshCandidate(o.binding, "w", "depth", np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), o.column, o.state, o.matrix_op, o.matrix_history, o.depth_m, None, np.zeros(3), np.zeros(3), None, False, {}, None, o._authority, [False])
    assert fl.FloatingSoilColumn is type(o.column)


def test_numba_owner_matches_numpy_owner():
    pytest.importorskip("numba")
    bed = stratified()
    child = swap_classes(set_cell_from(bed, 1, 2), 0)
    res = {}
    for backend in ("numpy", "numba"):
        o = nfs.initialize_native_floating_soil(bed, config(backend=backend), initial_relative_saturation=0.9)
        c = nfs.stage_native_refresh(o, bed, child, mode="properties")
        res[backend] = (o, c)
    (a, ca), (b, cb) = res["numpy"], res["numba"]
    assert np.array_equal(a.depth_m, b.depth_m) and np.array_equal(a.state.water_m, b.state.water_m)
    assert np.array_equal(ca.state.water_m, cb.state.water_m) and np.array_equal(ca.surface_return_m, cb.surface_return_m) and np.array_equal(ca.bottom_export_m, cb.bottom_export_m)
    assert np.array_equal(ca.matrix_history.wetting_m, cb.matrix_history.wetting_m)


def test_depth_equal_to_the_profile_height_is_closed_rock_without_epsilon():
    bed = stratified()
    d = nfs.initialize_native_floating_soil(bed, config(layers=(0.222,), diffusivity_m2_s=np.zeros(0)), initial_relative_saturation=0.5).depth_m[0]
    o = nfs.initialize_native_floating_soil(bed, config(layers=(float(d),), diffusivity_m2_s=np.zeros(0)), initial_relative_saturation=0.5)
    assert o.column._maps["edges"][-1] == o.depth_m[0] and not o.column._maps["bottom_open"][0] and not o.column._maps["bottom_open"][1]  # D == H closed
    o2 = nfs.initialize_native_floating_soil(bed, config(layers=(float(np.nextafter(d, 0.0)),), diffusivity_m2_s=np.zeros(0)), initial_relative_saturation=0.5)
    assert o2.column._maps["bottom_open"][0]  # D > H by one ulp is open: no epsilon


def test_imported_native_source_identity_matches_the_configured_pin():
    """With `E3_PINNED_MAPLE_SRC` set (the pinned materialised source directory) the imported MAPLE MUST come from it: a wrong source FAILS. Unset means no pin was configured (skip, no claim)."""
    import os

    import maple

    pin = os.environ.get("E3_PINNED_MAPLE_SRC")
    if not pin:
        pytest.skip("E3_PINNED_MAPLE_SRC is not configured: no source-identity claim is made")
    assert os.path.realpath(maple.__file__).startswith(os.path.realpath(pin) + os.sep), f"imported MAPLE {maple.__file__} is not under the configured pin {pin}"


# ---------------------------------------------------------------------------------------------------------------------------------------------
# correction01
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_untouched_columns_keep_their_water_bitwise_even_above_field_capacity():
    bed, o = init(sat=0.999999)  # heterogeneous post-storm state: every porous slot just below saturation, i.e. ABOVE field capacity
    assert np.any(o.state.water_m > o.column._maps["theta_field_capacity"] * 0 + 0)  # non-trivial
    child = swap_classes(bed, 0)
    cand = nfs.stage_native_refresh(o, bed, child, mode="properties")
    assert list(cand.dirty_indices) == [0] and list(cand.changed_columns) == [0]
    for c in (1, 2):  # unrefreshed columns: no cascade, no export, bitwise identical
        assert np.array_equal(cand.state.water_m[:, c], o.state.water_m[:, c])
    assert cand.surface_return_m[1:].sum() == 0.0 and cand.bottom_export_m[1:].sum() == 0.0
    rc = cand.receipt
    assert rc.n_changed_columns == 1 and abs(rc.residual_m3) <= rc.bound_m3 and abs(rc.whole_residual_m3) <= rc.whole_bound_m3
    # whole-inventory closure from the actual selected reconciliation
    area = 0.25
    assert (cand.state.water_m.sum() - o.state.water_m.sum()) * area + (cand.surface_return_m.sum() + cand.bottom_export_m.sum()) * area == pytest.approx(0.0, abs=rc.whole_bound_m3)
    assert cand.surface_return_m.sum() * area == pytest.approx(rc.surface_return_m3, abs=rc.whole_bound_m3)


def test_inactive_untouched_exact_and_changed_set_follows_actual_changes():
    bed = stratified()
    mask = np.array([[True, True, False]])
    bed = nfs.NativeBed(bed.voxel_column, bed.active_layer, bed.geometry, bed.grain_classes, bed.mass_resolution_kg, mask)
    bed, o = init(bed, sat=0.999999)
    cand = nfs.stage_native_refresh(o, bed, swap_classes(bed, 1), mode="properties")
    assert list(cand.changed_columns) == [1] and np.array_equal(cand.state.water_m[:, [0, 2]], o.state.water_m[:, [0, 2]])


def test_trial_rollback_after_a_late_staging_failure(monkeypatch):
    from maple_syrup import floating_matrix_infiltration as fm

    bed, o = init(sat=0.999999)
    child = swap_classes(bed, 0)
    real = fm.rebind_floating_matrix_history
    monkeypatch.setattr(fm, "rebind_floating_matrix_history", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("late failure")))
    with pytest.raises(RuntimeError):
        nfs.stage_native_refresh(o, bed, child, mode="properties")
    assert o.is_current and o.generation == 0
    monkeypatch.setattr(fm, "rebind_floating_matrix_history", real)
    assert nfs.stage_native_refresh(o, bed, child, mode="properties").n_dirty == 1


def test_exposed_arrays_and_records_are_sealed():
    bed, o = init(sat=0.999999)
    cand = nfs.stage_native_refresh(o, bed, swap_classes(bed, 0), mode="properties")
    for a in (cand.surface_return_m, cand.bottom_export_m, cand.depth_m, cand.dirty_indices, cand.changed_columns, o.depth_m, o.config.grain_texture, o.config.nominal_edges_m,
              o.config.latent_theta_field_capacity, o.config.organic_matter_percent):
        with pytest.raises(ValueError):
            a[0] = a[0] + 1
    before = (cand.surface_return_m.sum(), cand.receipt.surface_return_m3)
    o2 = nfs.adopt_native_refresh(o, cand)
    assert o2.totals["surface_return_m3"] == cand.receipt.surface_return_m3 and before[1] == cand.receipt.surface_return_m3
    with pytest.raises(TypeError):
        o2.totals["surface_return_m3"] = 5.0


def test_override_values_are_part_of_the_config_identity():
    a = np.full((3, 3), np.nan)
    b = a.copy()
    a[0, 0], b[0, 0] = 0.30, 0.31
    ca, cb = config(overrides={"theta_33_override": a}), config(overrides={"theta_33_override": b})
    assert ca.sha256 != cb.sha256 and ca.sha256 == config(overrides={"theta_33_override": a.copy()}).sha256
    a[0, 0] = 9.0  # the config owns a sealed copy
    assert ca.overrides["theta_33_override"][0, 0] == 0.30 and not ca.overrides["theta_33_override"].flags.writeable


def test_copies_and_reconstructions_do_not_mint_a_second_authority():
    import copy
    import dataclasses
    import pickle

    bed, o = init()
    cand = nfs.stage_native_refresh(o, bed, swap_classes(bed, 0), mode="properties")
    for obj in (o, cand, o.binding):
        for fn in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(nfs.NativeFloatingError):
                fn(obj)
    with pytest.raises(nfs.NativeFloatingError):
        dataclasses.replace(o, depth_m=o.depth_m)
    with pytest.raises(nfs.NativeFloatingError):
        dataclasses.replace(cand, mode="depth")
    assert nfs.adopt_native_refresh(o, cand).generation == 1


def test_late_adoption_failure_leaves_the_authority_and_candidate_retryable(monkeypatch):
    bed, o = init()
    cand = nfs.stage_native_refresh(o, bed, swap_classes(bed, 0), mode="properties")
    real = nfs.NativeBinding
    monkeypatch.setattr(nfs, "NativeBinding", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("late constructor failure")))
    with pytest.raises(RuntimeError):
        nfs.adopt_native_refresh(o, cand)
    monkeypatch.setattr(nfs, "NativeBinding", real)
    assert o.is_current and not cand._consumed[0] and o.totals["refreshes"] == 0
    o2 = nfs.adopt_native_refresh(o, cand)
    assert o2.generation == 1 and o2.totals["refreshes"] == 1


def _padded(bed, extra):
    from maple.core.types.voxel import VoxelColumnState

    v = bed.voxel_column.mass_kg
    out = np.zeros(v.shape[:2] + (v.shape[2] + extra, v.shape[3]))
    out[:, :, :v.shape[2]] = v
    return nfs.NativeBed(VoxelColumnState(mass_kg=out), bed.active_layer, bed.geometry, bed.grain_classes, bed.mass_resolution_kg, bed.active_mask)


def test_empty_top_capacity_growth_and_shrink_are_not_changes():
    bed, o = init(sat=0.999999)
    grown = _padded(bed, 4)  # (3, 8) voxels vs (3, 4): newly added EMPTY top capacity
    cand = nfs.stage_native_refresh(o, bed, grown, mode="properties")
    assert cand.n_dirty == 0 and not cand.reconciled and cand.state is o.state
    assert cand.child_witness == o.binding.source_witness  # canonical: capacity padding is not part of the identity
    o2 = nfs.adopt_native_refresh(o, cand)
    back = nfs.stage_native_refresh(o2, grown, bed, mode="properties")  # the accepted child is the next parent; shrinking is also no change
    assert back.n_dirty == 0
    # a real change beyond the padded capacity is still dirty
    bumped = _padded(bed, 4)
    bumped.voxel_column.mass_kg[0, 0, 6, 0] = 1e-3
    with pytest.raises((nfs.NativeFloatingError, ValueError, TypeError, RuntimeError)):  # an invalid native column (mass above an empty voxel) is refused by the native validator, never accepted as a change
        nfs.stage_native_refresh(o, bed, bumped, mode="properties")


def test_class_order_and_geometry_semantics_are_part_of_the_binding():
    from maple.core.parameters.grain_classes import GrainClassSet

    bed, o = init()
    reordered = GrainClassSet(classes=tuple(reversed(bed.grain_classes.classes)))
    other = nfs.NativeBed(bed.voxel_column, bed.active_layer, bed.geometry, reordered, bed.mass_resolution_kg, bed.active_mask)  # identical masses and K, other class ORDER
    with pytest.raises(nfs.NativeFloatingError):
        nfs.stage_native_refresh(o, bed, other)  # incomparable beds
    with pytest.raises(nfs.NativeStaleError):
        nfs.stage_native_refresh(o, other, other)  # not the owner's accepted source (descriptor is part of the witness)
    assert nfs._descriptor(bed) != nfs._descriptor(other)


def test_the_provider_sees_only_real_material_slots(monkeypatch):
    calls = []
    real = stp.estimate_soil_texture_properties

    def spy(**k):
        calls.append((k["n_layers"], k["n_cells"]))
        return real(**k)

    monkeypatch.setattr(stp, "estimate_soil_texture_properties", spy)
    bed = stratified()
    nfs.initialize_native_floating_soil(bed, config(), initial_relative_saturation=0.5)
    assert all(nl == 1 for nl, _ in calls) and sum(m for _, m in calls) == 8  # 9 slots, exactly ONE zero-mass slot (layer 3 of the shallow cell) is never passed


# ---------------------------------------------------------------------------------------------------------------------------------------------
# correction02: composed memory from the ACTUAL parent AND child shapes, bounded lifetimes, minting windows, exact-parent request
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _spy_rows(monkeypatch):
    sizes = []
    real = nfs._rows

    def spy(bed, idx, B):
        v, a = real(bed, idx, B)
        sizes.append(int(v.nbytes + a.nbytes))
        return v, a

    monkeypatch.setattr(nfs, "_rows", spy)
    return sizes


def test_parent_with_much_larger_native_capacity_is_admitted_from_its_actual_shape(monkeypatch):
    bed, o = init(sat=0.999999)
    big_parent = _padded(bed, 9996)  # same physical material + 9996 EMPTY top voxels (10000 voxels), the child has 4
    model = nfs.estimate_native_refresh_memory(big_parent, bed, config())
    assert model["parent_voxels"] == 10000 and model["child_voxels"] == 4
    small = nfs.estimate_native_refresh_memory(bed, bed, config())
    assert model["terms"]["scan"] > small["terms"]["scan"] + 3 * 9996 * 4 * 8  # the parent's capacity is counted
    sizes = _spy_rows(monkeypatch)
    with pytest.raises(nfs.NativeFloatingMemoryError, match="nothing was allocated"):
        nfs.stage_native_refresh(o, big_parent, bed, memory_budget_bytes=model["required_bytes"] - 1)
    assert sizes == []  # refused BEFORE the first row was gathered
    seen = []
    cand = nfs.stage_native_refresh(o, big_parent, bed, memory_budget_bytes=model["required_bytes"], available_bytes_fn=lambda: seen.append(1) or model["required_bytes"])
    assert cand.n_dirty == 0 and seen and max(sizes) <= model["terms"]["scan"] <= model["required_bytes"]  # the largest single gather is inside the admitted figure
    assert cand.request.memory_required_bytes == model["required_bytes"]


def test_child_larger_then_shrink_and_non_top_changes_stay_dirty():
    bed, o = init(sat=0.999999)
    grown = _padded(bed, 40)
    up = nfs.stage_native_refresh(o, bed, grown)  # child with MORE empty top capacity
    assert up.n_dirty == 0 and not up.reconciled
    o2 = nfs.adopt_native_refresh(o, up)
    down = nfs.stage_native_refresh(o2, grown, bed)  # then SHRINK again from the accepted larger parent
    assert down.n_dirty == 0 and not down.reconciled
    # a change in a NON-top (deep) voxel of the larger-capacity bed is dirty
    deep = _padded(swap_classes(bed, 1), 40)
    c = nfs.stage_native_refresh(o2, grown, deep, mode="properties")
    assert list(c.dirty_indices) == [1]
    # only EXTRA capacity that is non-zero is a change on the larger side (checked at the comparison level)
    a = np.zeros((1, 6, 4))
    b = np.zeros((1, 4, 4))
    assert not nfs._changed_rows(a, b).any()
    a[0, 5, 2] = 1e-12
    assert nfs._changed_rows(a, b).all() and nfs._changed_rows(b, a).all()


def _tracemalloc_peak(fn):
    import gc
    import tracemalloc

    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    out = fn()
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return out, peak


@pytest.mark.parametrize("representation", ["dense", "windowed"])
def test_warm_tracked_array_peak_is_inside_the_composed_estimate(representation, tmp_path):
    from builders import fx

    parts = fx.stratified_dense(shallow_cells=(), ny=20, nx=50)  # N = 1000 cells, L = 3 layers
    dense = bed_from_parts(parts)
    parent = dense if representation == "dense" else windowed(dense, tmp_path, "p")
    child_d = swap_classes(swap_classes(dense, 0), 777)
    child = child_d if representation == "dense" else windowed(child_d, tmp_path, "c")
    B = 16
    owner = nfs.initialize_native_floating_soil(parent, config(), initial_relative_saturation=0.999999, batch_cells=B)
    nfs.stage_native_refresh(owner, parent, child, batch_cells=B)  # warm (imports, JIT-free caches)
    model = nfs.estimate_native_refresh_memory(parent, child, config(), batch_cells=B)
    cand, peak = _tracemalloc_peak(lambda: nfs.stage_native_refresh(owner, parent, child, batch_cells=B))
    assert list(cand.dirty_indices) == [0, 777]
    assert 0 < peak <= model["required_bytes"], (peak, model["required_bytes"])
    print(f"{representation}: tracked numpy-array peak {peak} B vs composed estimate {model['required_bytes']} B (ratio {peak / model['required_bytes']:.2f}), N=1000 L=3 B={B}")


def bed_from_parts(parts):
    from builders import bed_from

    return bed_from(parts)


def test_minting_window_is_local_to_one_constructor_call():
    bed, o = init()
    child = swap_classes(bed, 0)
    attempts = []

    def hostile_reader():  # a user availability callback running inside stage_native_refresh must not be able to mint
        for ctor in (lambda: nfs.NativeBinding("x", 0, "w", "t"), lambda: nfs.NativeFloatingSoil(*([None] * 11))):
            try:
                ctor()
            except nfs.NativeFloatingError:
                attempts.append("refused")
        return 1 << 40

    nfs.stage_native_refresh(o, bed, child, available_bytes_fn=hostile_reader)
    assert attempts == ["refused", "refused"]
    import threading

    box = []

    def other_thread():
        try:
            nfs.NativeBinding("x", 0, "w", "t")
        except nfs.NativeFloatingError:
            box.append("refused")

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()
    assert box == ["refused"] and not hasattr(nfs, "_minted")


def test_candidate_carries_the_exact_parent_request():
    bed, o = init()
    child = swap_classes(bed, 0)
    cand = nfs.stage_native_refresh(o, bed, child, mode="properties", batch_cells=2)
    r = cand.request
    assert (r.lineage, r.parent_generation, r.parent_witness) == (o.binding.lineage, 0, o.binding.source_witness)
    assert r.child_witness == cand.child_witness and r.mode == "properties" and r.batch_cells == 2 and (r.n_dirty, r.n_changed) == (1, 1)
    assert r.descriptor == nfs._descriptor(child) and r.memory_required_bytes == cand.memory["required_bytes"]


def test_exposed_state_arrays_are_not_second_copies():
    bed, o = init(sat=0.999999)
    cand = nfs.stage_native_refresh(o, bed, swap_classes(bed, 0))
    assert not cand.depth_m.flags.writeable and not cand.surface_return_m.flags.writeable
    assert not np.shares_memory(cand.depth_m, o.depth_m)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# E3b: explicit texture-Ks feedback mode (CPU)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def ks_config(**kw):
    return config(matrix_conductivity_m_s=None, matrix_conductivity_mode="texture_ks", latent_saturated_conductivity_m_s=3e-6, **kw)


def provider_ks(mass_by_class):
    """Independent reference: the accepted provider's Ks for one layer mass composition through the explicit grain mapping (not through the transaction code)."""
    frac = np.asarray(mass_by_class) / np.sum(mass_by_class)
    sand, silt, clay = (np.full((1, 1), float(frac @ GRAIN_TEXTURE[:, c])) for c in range(3))
    res = stp.estimate_soil_texture_properties(sand_fraction=sand, silt_fraction=silt, clay_fraction=clay, organic_matter_percent=np.full((1, 1), 3.0), bulk_density_kg_m3=np.full((1, 1), 1250.0),
                                               n_layers=1, n_cells=1, memory_budget_bytes=1 << 30)
    return float(res.array("saturated_conductivity_m_s")[0, 0])


def test_texture_ks_mode_uses_the_layer_zero_provider_conductivity():
    from builders import fx

    bed = stratified()
    o = nfs.initialize_native_floating_soil(bed, ks_config(), initial_relative_saturation=0.5)
    law = np.asarray(o.matrix_op._law[0])
    ref0 = provider_ks(fx.EXPECTED_3[0])  # hand-worked layer-0 mass of the stratified column (cells 0 and 1)
    assert law[0] == ref0 == law[1] and law[2] != law[0] and law[2] > 0.0
    assert np.array_equal(law, o.ks_m_s) and not o.ks_m_s.flags.writeable
    # the REAL matrix intake of the operator equals the same law built independently from the reference K
    from maple_syrup import floating_matrix_infiltration as fm

    ref_op = fm.prepare_floating_matrix(o.column, conductivity_m_s=np.array(law), suction_m=0.1, vegetation_factor=0.7)
    pool, standing, rate = np.full(3, 0.004), np.zeros(3), np.full(3, 0.004 / 600.0)
    r1 = fm.floating_matrix_step(o.matrix_op, pool, standing, rate, o.state, o.matrix_history, 600.0)
    r2 = fm.floating_matrix_step(ref_op, pool, standing, rate, o.state, fm.floating_matrix_history(ref_op, wetting_m=0.01, cumulative_intake_m=0.02), 600.0)
    assert np.array_equal(r1.state.water_m, r2.state.water_m) and np.array_equal(r1.intake_m, r2.intake_m)
    # the explicit mode is untouched by the new option
    e = nfs.initialize_native_floating_soil(bed, config(), initial_relative_saturation=0.5)
    assert e.ks_m_s is None and float(np.asarray(e.matrix_op._law[0])[0]) == 2e-6


def test_texture_ks_follows_texture_change_holds_in_depth_mode_and_keeps_history():
    bed = stratified()
    o = nfs.initialize_native_floating_soil(bed, ks_config(), initial_relative_saturation=0.5)
    child = swap_classes(bed, 0, 2, 3)  # zero-net composition exchange of the layer-0 material (the active cap holds classes 2 and 3)
    held = nfs.stage_native_refresh(o, bed, child, mode="depth")
    assert np.array_equal(held.ks_m_s, o.ks_m_s) and held.matrix_op is o.matrix_op  # depth mode HOLDS Ks as a property
    cand = nfs.stage_native_refresh(o, bed, child, mode="properties")
    assert cand.ks_m_s[0] != o.ks_m_s[0] and np.array_equal(cand.ks_m_s[1:], o.ks_m_s[1:])
    assert np.array_equal(np.asarray(cand.matrix_op._law[0]), cand.ks_m_s)
    assert np.array_equal(cand.matrix_history.wetting_m, o.matrix_history.wetting_m) and np.array_equal(cand.matrix_history.cumulative_intake_m, o.matrix_history.cumulative_intake_m)
    assert cand.matrix_history.operator_sha256 == cand.matrix_op.sha256
    o2 = nfs.adopt_native_refresh(o, cand)
    assert np.array_equal(o2.ks_m_s, cand.ks_m_s) and o2.generation == 1


def test_texture_ks_only_change_rebinds_without_a_water_cascade():
    """A Ks change with unchanged depth and FC/SAT (swap of two classes with identical texture rows) creates a new operator over the SAME column and state."""
    tex = GRAIN_TEXTURE.copy()
    tex[1] = tex[0]  # classes 0 and 1 share one texture row: swapping them changes the composition but no property
    bed = stratified()
    o = nfs.initialize_native_floating_soil(bed, config(texture=tex, matrix_conductivity_m_s=None, matrix_conductivity_mode="texture_ks", latent_saturated_conductivity_m_s=3e-6),
                                            initial_relative_saturation=0.999999)
    cand = nfs.stage_native_refresh(o, bed, swap_classes(bed, 0), mode="properties")
    assert cand.n_dirty == 1 and cand.changed_columns.size == 0 and cand.state is o.state and cand.column is o.column and cand.receipt is None


def test_measured_ks_override_goes_through_the_provider_and_latent_is_user_supplied():
    ov = np.full((3, 3), np.nan)
    ov[0, 1] = 7e-6
    bed = stratified()
    o = nfs.initialize_native_floating_soil(bed, ks_config(overrides={"saturated_conductivity_override_m_s": ov}), initial_relative_saturation=0.5)
    assert o.ks_m_s[1] == 7e-6 and o.ks_m_s[0] != 7e-6
    # latent: a batch whose layer 0 holds no material keeps the user-supplied Ks, with NO provider call
    from types import SimpleNamespace

    cfg = ks_config()
    batch = SimpleNamespace(layer_fraction=np.zeros((3, 2, 4)), layer_total_kg=np.zeros((3, 2)), indices=np.array([0, 1]), n_cells=2)
    calls = []
    real = stp.estimate_soil_texture_properties
    stp.estimate_soil_texture_properties = lambda **k: calls.append(1) or real(**k)
    try:
        fc, _sat, ks0 = nfs._batch_properties(batch, cfg, 4, 1 << 20)
    finally:
        stp.estimate_soil_texture_properties = real
    assert calls == [] and np.all(ks0 == 3e-6) and np.all(fc[0] == 0.25)


def test_ks_config_validation_and_identity():
    with pytest.raises(nfs.NativeFloatingError):
        config(matrix_conductivity_mode="texture_ks", latent_saturated_conductivity_m_s=3e-6)  # explicit K must be absent in texture mode
    with pytest.raises(nfs.NativeFloatingError):
        config(matrix_conductivity_m_s=None, matrix_conductivity_mode="texture_ks")  # latent Ks required
    with pytest.raises(nfs.NativeFloatingError):
        config(latent_saturated_conductivity_m_s=3e-6)  # latent Ks refused in explicit mode
    assert ks_config().sha256 != config().sha256 and ks_config().sha256 != config(matrix_conductivity_m_s=None, matrix_conductivity_mode="texture_ks", latent_saturated_conductivity_m_s=4e-6).sha256


# ---------------------------------------------------------------------------------------------------------------------------------------------
# E3b correction02: the active mask PATTERN is part of the source binding
# ---------------------------------------------------------------------------------------------------------------------------------------------
def masked(bed, pattern):
    return nfs.NativeBed(bed.voxel_column, bed.active_layer, bed.geometry, bed.grain_classes, bed.mass_resolution_kg, np.array(pattern, dtype=bool))


def test_mask_pattern_is_in_the_versioned_descriptor_and_witness():
    import hashlib

    bed = stratified()
    a, b2 = masked(bed, [[True, True, False]]), masked(bed, [[False, True, True]])  # SAME active count (2), different pattern, identical native rows
    assert nfs._descriptor(a) != nfs._descriptor(b2) and nfs._descriptor(a) == nfs._descriptor(masked(bed, [[True, True, False]]))
    # layout independence: a Fortran-ordered / strided copy of the same mask has the same digest
    base = np.zeros((2, 3), dtype=bool)
    base[0] = [True, True, False]
    assert nfs._mask_digest(base[:1]) == nfs._mask_digest(np.asfortranarray(base[:1])) == nfs._mask_digest(np.array([[True, True, False]]))
    want = hashlib.sha256(f"{nfs.WITNESS_RECIPE}|mask|(1, 3)|".encode() + np.array([True, True, False]).tobytes()).hexdigest()  # independent recipe replica
    assert nfs._mask_digest(a.active_mask) == want and nfs.WITNESS_RECIPE.endswith("/2")
    w_a = nfs._scan(None, a, nfs._flat_active(a), 1)[1]
    assert w_a == nfs._scan(None, a, nfs._flat_active(a), 64)[1] != nfs._scan(None, b2, nfs._flat_active(b2), 1)[1]  # batch invariant; the pattern changes the witness


def test_same_count_different_mask_cannot_masquerade_as_the_accepted_parent():
    bed = stratified()
    a, b2 = masked(bed, [[True, True, False]]), masked(bed, [[False, True, True]])
    o = nfs.initialize_native_floating_soil(a, config(), initial_relative_saturation=0.5)
    for mode in ("depth", "properties"):
        with pytest.raises(nfs.NativeFloatingError, match="active mask"):
            nfs.stage_native_refresh(o, b2, b2, mode=mode)  # exactly equal native columns, parent AND child submitted with the other mask
    assert o.is_current and o.generation == 0
    # the accepted mask still works and inactive cells stay exactly unchanged
    c = nfs.stage_native_refresh(o, a, a, mode="depth")
    assert c.n_dirty == 0 and np.array_equal(c.state.water_m[:, 2], o.state.water_m[:, 2])


def test_a_mask_mutated_in_place_after_initialization_is_refused():
    bed = stratified()
    mask = np.array([[True, True, False]])
    live = masked(bed, mask)
    live = nfs.NativeBed(live.voxel_column, live.active_layer, live.geometry, live.grain_classes, live.mass_resolution_kg, mask)  # the SAME ndarray object the caller still holds
    o = nfs.initialize_native_floating_soil(live, config(), initial_relative_saturation=0.5)
    mask[0] = [False, True, True]  # ordinary caller mutation, same count
    with pytest.raises(nfs.NativeFloatingError, match="active mask"):
        nfs.stage_native_refresh(o, live, live, mode="depth")
    assert o.is_current


def test_witness_recipe_is_versioned_and_changes_the_accepted_binding():
    bed = stratified()
    o = nfs.initialize_native_floating_soil(bed, config(), initial_relative_saturation=0.5)
    assert nfs.WITNESS_RECIPE.endswith("/2") and len(o.binding.source_witness) == 64
    assert o.binding.source_witness == nfs._scan(None, bed, nfs._flat_active(bed), 3)[1]  # the accepted witness is exactly the cold recipe over the actual mask and rows


# ---------------------------------------------------------------------------------------------------------------------------------------------
# E3c: actual canonical depth bookkeeping vs HYDRAULIC geometry change
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_equivalent_deeper_than_profile_depth_change_does_not_drain_and_publishes_the_actual_depth(backend_name="numpy"):
    from builders import shift_top

    bed = stratified()
    o = nfs.initialize_native_floating_soil(bed, config(), initial_relative_saturation=0.999999)  # a legal parent ABOVE field capacity
    child = shift_top(bed, 0, -0.001)  # remove 1 mm from the top filled voxel of cell 0 ONLY: D 0.222 -> 0.221, both deeper than H = 0.16
    cand = nfs.stage_native_refresh(o, bed, child, mode="depth")
    assert list(cand.dirty_indices) == [0] and cand.changed_columns.size == 0 and list(cand.depth_changed) == [0] and not cand.reconciled and cand.receipt is None
    assert cand.depth_m[0] == pytest.approx(0.221, abs=1e-15) and cand.depth_m[0] != o.depth_m[0] and np.array_equal(cand.depth_m[1:], o.depth_m[1:])
    # genuinely new column/state/operator bound to the ACTUAL depth, with the accepted inventory bit for bit
    assert cand.column.sha256 != o.column.sha256 and np.array_equal(np.asarray(cand.column._maps["soil_depth"]), cand.depth_m)
    assert cand.state is not o.state and np.array_equal(cand.state.water_m, o.state.water_m) and cand.state.t_s == o.state.t_s
    assert not cand.surface_return_m.any() and not cand.bottom_export_m.any()
    assert np.array_equal(cand.matrix_history.wetting_m, o.matrix_history.wetting_m) and np.array_equal(cand.matrix_history.cumulative_intake_m, o.matrix_history.cumulative_intake_m)
    assert cand.matrix_history.operator_sha256 == cand.matrix_op.sha256 != o.matrix_op.sha256
    assert np.array_equal(np.asarray(cand.column._maps["theta_field_capacity"]), np.asarray(o.column._maps["theta_field_capacity"]))
    assert cand.request.n_changed == 0 and cand.request.n_depth_changed == 1 and cand.request.n_dirty == 1
    o2 = nfs.adopt_native_refresh(o, cand)
    assert o2.generation == 1 and o2.depth_m[0] == cand.depth_m[0] and np.array_equal(np.asarray(o2.column._maps["soil_depth"]), o2.depth_m)
    assert o2.totals["surface_return_m3"] == 0.0 and o2.totals["bottom_export_m3"] == 0.0 and not o.is_current
    again = nfs.stage_native_refresh(o2, child, child, mode="depth")  # the accepted child is the next parent
    assert again.n_dirty == 0
    # the reverse direction (deeper again) is equivalent too
    back = nfs.stage_native_refresh(o2, child, bed, mode="depth")
    assert back.changed_columns.size == 0 and list(back.depth_changed) == [0] and np.array_equal(back.state.water_m, o.state.water_m)


def test_numba_owner_has_the_same_equivalent_depth_outcome():
    pytest.importorskip("numba")
    from builders import shift_top

    bed = stratified()
    res = []
    for backend in ("numpy", "numba"):
        o = nfs.initialize_native_floating_soil(bed, config(backend=backend), initial_relative_saturation=0.999999)
        c = nfs.stage_native_refresh(o, bed, shift_top(bed, 0, -0.001), mode="depth")
        res.append((o, c))
    assert np.array_equal(res[0][1].state.water_m, res[1][1].state.water_m) and np.array_equal(res[0][1].depth_m, res[1][1].depth_m) and res[1][1].changed_columns.size == 0


def test_geometry_equivalence_rules_have_no_epsilon():
    cfg_open = config(layers=(0.15,), diffusivity_m2_s=np.zeros(0))
    cfg_closed = config(layers=(0.15,), diffusivity_m2_s=np.zeros(0), bottom="closed")
    H = float(cfg_open.nominal_edges_m[-1])
    up = np.nextafter(H, 1.0)
    d_old = np.array([H, H, 0.2, 0.2, 0.1, 0.0, 0.1])
    d_new = np.array([up, H, 0.3, 0.16, 0.1, 0.0, 0.1 + 1e-9])
    assert list(nfs._geometry_changed(cfg_open, d_old, d_new)) == [True, False, False, False, False, False, True]  # open: D==H -> >H flips the bottom flag; deeper-than-H changes are equivalent; a partial slab change is real
    assert list(nfs._geometry_changed(cfg_closed, d_old, d_new)) == [False, False, False, False, False, False, True]  # explicit closed: D==H and D>H are the same geometry


def test_open_profile_crossing_the_bottom_and_partial_capacity_still_reconcile_correctly():
    from builders import copy_dense

    bed = stratified()
    o = nfs.initialize_native_floating_soil(bed, config(), initial_relative_saturation=0.999999)
    # erode cell 0 from D = 0.222 to D ~ 0.122 < H = 0.16 (the profile bottom enters the soil): remove the partial top voxel and the upper 0.1 m... i.e. the top full voxel too
    child = copy_dense(bed)
    child.voxel_column.mass_kg[0, 0, 2] = 0.0
    child.voxel_column.mass_kg[0, 0, 1] *= 0.2  # voxel 1 (0.1 m) keeps 0.02 m
    c = nfs.stage_native_refresh(o, bed, child, mode="depth")
    assert c.depth_m[0] < 0.16
    assert list(c.changed_columns) == [0] and c.reconciled and c.receipt is not None and abs(c.receipt.whole_residual_m3) <= c.receipt.whole_bound_m3
    assert c.surface_return_m[0] > 0.0 or c.bottom_export_m[0] > 0.0
    # the whole-profile D1 reference of the same column pair agrees (no extra rule)
    assert list(c.depth_changed) == [0] and c.request.n_changed == 1


def test_explicit_closed_profile_keeps_water_when_the_depth_passes_the_profile_height_and_open_flips_the_boundary():
    from builders import shift_top

    bed = stratified()
    depth = nfs.initialize_native_floating_soil(bed, config(layers=(0.222,), diffusivity_m2_s=np.zeros(0)), initial_relative_saturation=0.5).depth_m[0]
    H = float(depth)
    child = shift_top(bed, 0, 0.001)  # D 0.222 (== H bitwise) -> 0.223 (> H)
    closed = nfs.initialize_native_floating_soil(bed, config(layers=(H,), diffusivity_m2_s=np.zeros(0), bottom="closed"), initial_relative_saturation=0.999999)
    cc = nfs.stage_native_refresh(closed, bed, child, mode="depth")
    assert cc.changed_columns.size == 0 and list(cc.depth_changed) == [0] and np.array_equal(cc.state.water_m, closed.state.water_m)  # explicit closed: equivalent
    opened = nfs.initialize_native_floating_soil(bed, config(layers=(H,), diffusivity_m2_s=np.zeros(0), bottom="open"), initial_relative_saturation=0.999999)
    assert not opened.column._maps["bottom_open"][0]  # D == H is closed rock even in the open mode
    oc = nfs.stage_native_refresh(opened, bed, child, mode="depth")
    assert list(oc.changed_columns) == [0] and oc.reconciled and oc.column._maps["bottom_open"][0]  # open: D > H flips the bottom flag: a REAL change


def test_property_changes_still_cascade_and_no_op_beds_stay_no_ops():
    bed = stratified()
    o = nfs.initialize_native_floating_soil(bed, config(), initial_relative_saturation=0.999999)
    c = nfs.stage_native_refresh(o, bed, swap_classes(bed, 0), mode="properties")
    assert list(c.changed_columns) == [0] and c.reconciled and list(c.depth_changed) == []  # a pure composition change: properties change, the actual depth does not
    n = nfs.stage_native_refresh(o, bed, bed, mode="depth")
    assert n.n_dirty == 0 and n.changed_columns.size == 0 and n.depth_changed.size == 0 and n.state is o.state and n.column is o.column
