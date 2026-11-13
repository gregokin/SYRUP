"""Phase 2 E1: explicit `coverage="partial_over_base"` texture policy on genuine native Dense/Windowed beds.
CPU only here; the CUDA definitions at the bottom run only when E1_ALLOW_GPU=1 is set by Root (never by the author)."""
import os
import tracemalloc

import numpy as np
import pytest
from texture_fixtures import (
    BUDGET,
    CAP_KG,
    CAP_M,
    MIX_A,
    MIX_B,
    MIX_C,
    MIX_D,
    PER_M,
    full_mask,
    host,
    native_deep_bed,
    stratified_dense,
    texture_bytes,
    windowed_twin,
)

from maple_syrup.soil_profile_texture import (
    SoilProfileGeometryError,
    SoilProfileTextureError,
    build_soil_profile_texture,
    soil_profile_texture_from_bed,
)

TIGHT = {"rtol": 1e-12, "atol": 1e-13}
CAPACITY = PER_M * 0.1
POLICY = "partial_over_base"
ALL = np.array([[True, True, True]])


def build(layers, vc=None, al=None, g=None, gc=None, mres=None, *, mask=ALL, **kw):
    if vc is None:
        vc, al, g, gc, mres = stratified_dense()
    return build_soil_profile_texture(vc, al, g, gc, mres, layers, active_mask=mask, memory_budget_bytes=BUDGET, **kw)


def test_default_is_strict_and_unchanged():
    with pytest.raises(SoilProfileGeometryError, match="do not represent enough"):
        build((0.1, 0.1, 0.1))  # depth 0.222 < 0.3 on cells 0, 1 (and 0.032 on the shallow cell)
    t = build((0.05, 0.05), mask=np.array([[True, True, False]]))
    assert t.coverage_policy == "full" and "coverage_policy" not in t.source


def test_hand_worked_partial_bottom_layer():
    t = build((0.1, 0.1, 0.1), mask=np.array([[True, True, False]]), coverage=POLICY)
    assert t.coverage_policy == POLICY
    pol = t.source["coverage_policy"]
    assert pol["policy"] == POLICY and "NOT the canonical hydraulic" in pol["represented_depth_m_meaning"]
    # segments: cap [0,0.002) D; voxel2 [0.002,0.022) 6.25 kg C; voxel1 [0.022,0.122) 31.25 kg B; voxel0 [0.122,0.222) 31.25 kg A
    exp = np.array([
        0.625 * MIX_D + 6.25 * MIX_C + (0.078 / 0.1) * 31.25 * MIX_B,
        (0.022 / 0.1) * 31.25 * MIX_B + (0.078 / 0.1) * 31.25 * MIX_A,
        (0.022 / 0.1) * 31.25 * MIX_A,
    ])
    for cell in (0, 1):
        np.testing.assert_allclose(t.layer_mass_kg[:, cell, :], exp, **TIGHT)
        np.testing.assert_allclose(t.coverage_m[:, cell], (0.1, 0.1, 0.022), **TIGHT)
        np.testing.assert_allclose(t.layer_fraction[2, cell], MIX_A, **TIGHT)  # actual fractions, nothing renormalised away
    assert bool((t.layer_total_kg[:, :2] > 0.0).all())
    assert t.represented_depth_m[0] == pytest.approx(0.222, abs=1e-15)


def test_multiple_rock_slots_are_exactly_empty_and_not_invented():
    t = build((0.1, 0.1, 0.1, 0.1, 0.1), mask=np.array([[True, True, False]]), coverage=POLICY)
    for cell in (0, 1):
        assert t.layer_total_kg[3, cell] == 0.0 and t.layer_total_kg[4, cell] == 0.0
        assert not t.layer_mass_kg[3:, cell].any() and not t.layer_fraction[3:, cell].any()
        assert t.coverage_m[3, cell] == 0.0 and t.coverage_m[4, cell] == 0.0
    assert (t.layer_total_kg[3:, :2] == 0.0).all() and (t.layer_total_kg[:3, :2] > 0.0).all()
    assert (t.layer_total_kg[:, 2] == 0.0).all()  # inactive cell: no profile
    assert not hasattr(t, "empty_layer_mask")  # no unaccounted L x N convenience tensor


def test_shallow_cell_zero_and_partial_layers_and_mass_conservation():
    vc, al, g, gc, mres = stratified_dense()
    t = build((0.01, 0.05, 0.1), vc, al, g, gc, mres, coverage=POLICY)
    cell = 2  # 0.03 m partial voxel 0 of mixture B below the 0.002 m cap
    seg = PER_M * 0.03
    np.testing.assert_allclose(t.layer_mass_kg[0, cell], CAP_KG * MIX_D + (0.008 / 0.03) * seg * MIX_B, **TIGHT)
    np.testing.assert_allclose(t.layer_mass_kg[1, cell], (0.022 / 0.03) * seg * MIX_B, **TIGHT)
    assert t.layer_total_kg[2, cell] == 0.0
    np.testing.assert_allclose(t.coverage_m[:, cell], (0.01, 0.022, 0.0), **TIGHT)
    # all actual material (cap + voxel) is accounted for in the layers, none is created
    total = t.layer_mass_kg[:, cell].sum()
    assert total == pytest.approx(CAP_KG + seg, rel=1e-12)


def test_wholly_eroded_active_cell_is_valid_and_all_empty():
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    vc, al, g, gc, mres = stratified_dense()
    v, a = np.array(vc.mass_kg), np.array(al.mass_kg)
    v[0, 1], a[0, 1] = 0.0, 0.0
    vc, al = VoxelColumnState(mass_kg=v), ActiveLayerState(mass_kg=a)
    with pytest.raises(SoilProfileGeometryError):
        build((0.05, 0.05), vc, al, g, gc, mres)
    t = build((0.05, 0.05), vc, al, g, gc, mres, coverage=POLICY)
    assert t.represented_depth_m[1] == 0.0 and (t.layer_total_kg[:, 1] == 0.0).all()
    assert not t.layer_mass_kg[:, 1].any() and not t.coverage_m[:, 1].any()
    assert t.layer_total_kg[0, 0] > 0.0  # neighbours untouched


def test_fully_covered_arrays_are_byte_identical_across_policies():
    m = np.array([[True, True, False]])
    a = build((0.01, 0.05, 0.1), mask=m)
    b = build((0.01, 0.05, 0.1), mask=m, coverage=POLICY)
    assert texture_bytes(a) == texture_bytes(b)
    assert b.coverage_policy == POLICY and a.coverage_policy == "full"


def test_top_layer_cap_constraint_is_retained():
    with pytest.raises(SoilProfileGeometryError, match="active-layer cap thicker"):
        build((CAP_M / 2, 0.05, 0.3), coverage=POLICY)


def test_inactive_cells_stay_zero_and_never_refuse():
    m = np.array([[True, False, False]])
    t = build((0.1, 0.1, 0.1), mask=m, coverage=POLICY)
    for c in (1, 2):
        assert not t.layer_mass_kg[:, c].any() and not t.coverage_m[:, c].any() and t.represented_depth_m[c] == 0.0


@pytest.mark.parametrize("bad", [None, "Partial", "partial", 1, True, b"full", ("full",)])
def test_invalid_policy_refused_before_work(bad):
    with pytest.raises(SoilProfileTextureError, match="coverage must be one of"):
        build((0.05, 0.05), coverage=bad)


def test_inputs_not_mutated_and_refusal_publishes_nothing():
    vc, al, g, gc, mres = stratified_dense()
    v0, a0 = vc.mass_kg.copy(), al.mass_kg.copy()
    build((0.1, 0.1, 0.3), vc, al, g, gc, mres, coverage=POLICY)
    with pytest.raises(SoilProfileGeometryError):
        build((CAP_M / 4, 0.1), vc, al, g, gc, mres, coverage=POLICY)
    assert np.array_equal(vc.mass_kg, v0) and np.array_equal(al.mass_kg, a0)


def test_coverage_inconsistency_with_actual_material_is_refused(monkeypatch):
    import maple_syrup.soil_profile_texture as spt

    original = spt.reference_texture_batch

    def broken(*args):
        out = list(original(*args))
        out[3] = out[3] + 0.01  # coverage claims more material than the depth supports
        return tuple(out)

    monkeypatch.setattr(spt, "reference_texture_batch", broken)
    with pytest.raises(SoilProfileGeometryError, match="disagrees with the actual represented"):
        build((0.1, 0.1, 0.1), coverage=POLICY)


@pytest.mark.parametrize("field,value", [("coverage_m", np.nan), ("coverage_m", np.inf), ("coverage_m", -0.01),
                                         ("represented_depth_m", np.nan), ("represented_depth_m", np.inf),
                                         ("represented_depth_m", -0.5)])
@pytest.mark.parametrize("cell,active", [(0, True), (2, False)])
def test_invalid_coverage_or_depth_fails_closed_on_active_cells_only(monkeypatch, field, value, cell, active):
    import maple_syrup.soil_profile_texture as spt

    original = spt.reference_texture_batch

    def broken(*args):
        out = [np.array(o) for o in original(*args)]
        if field == "coverage_m":
            out[3][1, cell] = value
        else:
            out[4][cell] = value
        return tuple(out)

    monkeypatch.setattr(spt, "reference_texture_batch", broken)
    mask = np.array([[True, True, False]])
    if active:
        with pytest.raises(SoilProfileGeometryError, match="non-finite/negative"):
            build((0.1, 0.1, 0.1), mask=mask, coverage=POLICY)
    else:  # inactive-cell output follows the existing mask policy: not inspected by the opt-in check, no refusal
        build((0.1, 0.1, 0.1), mask=mask, coverage=POLICY)


def test_stage_is_itemised_admitted_and_transfers_only_chunk_flags(monkeypatch):
    from maple.core import backend as mb

    import maple_syrup.soil_profile_texture as spt

    vc, al, g, gc, mres = native_deep_bed(ny=2, nx=3)
    mask = full_mask(g.ny, g.nx)
    calls = []
    real = mb.to_host
    monkeypatch.setattr(mb, "to_host", lambda a: (calls.append(np.asarray(a).shape), real(a))[1])
    t = build((0.1, 0.2, 0.3, 0.3), vc, al, g, gc, mres, mask=mask, coverage=POLICY, batch_cells=4)
    ns, hostb = spt.partial_stage_bytes(4)
    assert (ns, hostb) == (6 * 8 * 4 + 8 * 4, 4) and ns <= t.memory.validation_bytes and hostb <= t.memory.host_bytes
    # one (<=4,) bool flag copy per chunk of 4 cells: 6 cells -> chunks of 4 and 2 (the kernels' own flag reads are (4,)/(2,) uint8)
    flag_copies = [c for c in calls if len(c) == 1]
    assert flag_copies.count((4,)) >= 2 and all(c[0] <= 4 for c in flag_copies)
    assert t.source["coverage_policy"]["validation_stage"]["model_bytes_namespace_host"] == [ns, hostb]
    full = build((0.1, 0.1), vc, al, g, gc, mres, mask=mask, batch_cells=4)
    assert "coverage_policy" not in full.source


def test_host_flag_buffer_is_single_and_mask_and_outputs_unchanged():
    import maple_syrup.soil_profile_texture as spt

    N = 7
    layers = spt.layer_geometry((0.1, 0.1))
    out = {"coverage_m": np.array([[0.1] * N, [0.05] * N]), "represented_depth_m": np.full(N, 0.15)}
    out["coverage_m"][1, 3] = 0.09  # inconsistent on cells 3 (active) and 5 (inactive)
    out["coverage_m"][1, 5] = 0.09
    mask = np.array([True, True, True, True, True, False, True])
    before = (out["coverage_m"].copy(), out["represented_depth_m"].copy(), mask.copy())
    assert spt._partial_coverage_inconsistent(out, mask, layers, 1e-13, 3) == 1
    assert all(np.array_equal(a, b) for a, b in zip(before, (out["coverage_m"], out["represented_depth_m"], mask)))


def test_stage_admission_refuses_when_not_covered():
    import dataclasses

    import maple_syrup.soil_profile_texture as spt

    vc, al, g, gc, mres = native_deep_bed(ny=1, nx=2)
    t = build((0.1, 0.2, 0.3, 0.3), vc, al, g, gc, mres, mask=full_mask(1, 2), coverage=POLICY)
    tiny = dataclasses.replace(t.memory, validation_bytes=1)
    with pytest.raises(spt.SoilProfileMemoryError, match="partial-coverage validation stage"):
        spt._admit_partial_stage(tiny)


def test_numba_equals_numpy_bitwise(numba_module):
    m = np.array([[True, True, False]])
    ref = build((0.1, 0.1, 0.1, 0.1), mask=m, coverage=POLICY)
    t = build((0.1, 0.1, 0.1, 0.1), mask=m, coverage=POLICY, backend="numba")
    assert texture_bytes(t) == texture_bytes(ref) and t.backend == "numba"


def test_from_bed_wrapper_passes_the_policy():
    from maple_syrup.sediment_bed import BedContext, BedState

    vc, al, g, gc, mres = stratified_dense()
    state = BedState(voxel_column=vc, active_layer=al, water=None, ledger=None, committed_topography=None,
                     sediment_availability=None)
    ctx = BedContext(geometry=g, grain_classes=gc, mass_resolution_kg=mres, topographic_commit_spec=None,
                     avalanching_spec=None, initial_available_fraction=None)
    t = soil_profile_texture_from_bed(state, ctx, (0.1, 0.1, 0.3), active_mask=ALL, memory_budget_bytes=BUDGET,
                                      coverage=POLICY)
    assert t.coverage_policy == POLICY and t.layer_total_kg[2, 2] == 0.0


def test_partial_check_stays_within_memory_model():
    vc, al, g, gc, mres = native_deep_bed(ny=2, nx=3)
    mask = full_mask(g.ny, g.nx)
    layers = (0.1, 0.2, 0.3, 0.3)
    build(layers, vc, al, g, gc, mres, mask=mask, coverage=POLICY)  # warm
    tracemalloc.start()
    t = build(layers, vc, al, g, gc, mres, mask=mask, coverage=POLICY)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= t.memory.required_bytes


@pytest.fixture
def pair(tmp_path):
    vc, al, g, gc, mres = native_deep_bed()
    w = windowed_twin(vc, g, mres, tmp_path / "store")
    assert int(host(w.base).min()) > 0  # resident window floor is ABOVE the physical base
    return vc, w, al, g, gc, mres


def test_genuine_windowed_equals_dense_and_window_floor_is_not_bedrock(pair):
    vc, w, al, g, gc, mres = pair
    layers = (0.1, 0.2, 0.2, 0.4)  # bottom edge 0.9 m >> surface 0.25-0.30 m: the base is hit, material exists below the window
    mask = full_mask(g.ny, g.nx)
    ref = build(layers, vc, al, g, gc, mres, mask=mask, coverage=POLICY)
    t = build(layers, w, al, g, gc, mres, mask=mask, coverage=POLICY)
    assert t.representation == "windowed" and texture_bytes(t) == texture_bytes(ref)
    assert "validate_windowed_column_full (always)" in t.source["windowed"]["validation"]
    # deep archived material (below the resident window) is read: the layers it covers are NOT empty
    assert bool((t.layer_total_kg[:2] > 0.0).all())
    deep_m = 0.01 * int(host(w.base).min())  # nominal height of the archived part of the column
    assert deep_m > 0.0 and float(t.represented_depth_m.min()) > deep_m
    assert (t.layer_total_kg[3] == 0.0).all()
    with pytest.raises(SoilProfileGeometryError):
        build(layers, w, al, g, gc, mres, mask=mask)  # strict default still refuses on the windowed bed


GPU = pytest.mark.skipif(os.environ.get("E1_ALLOW_GPU") != "1", reason="GPU not touched by the author; Root sets E1_ALLOW_GPU=1")


@GPU
def test_cuda_dense_equals_numpy(gpu):
    import cupy as cp
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    vc, al, g, gc, mres = stratified_dense()
    ref = build((0.1, 0.1, 0.1, 0.1), vc, al, g, gc, mres, mask=ALL, coverage=POLICY)
    dvc, dal = VoxelColumnState(mass_kg=cp.asarray(vc.mass_kg)), ActiveLayerState(mass_kg=cp.asarray(al.mass_kg))
    t = build((0.1, 0.1, 0.1, 0.1), dvc, dal, g, gc, mres, mask=ALL, coverage=POLICY, backend="cuda")
    assert t.namespace == "cupy" and texture_bytes(t) == texture_bytes(ref)


@GPU
def test_device_resident_windowed_host_compute_equals_dense(gpu, pair):
    """Device-resident genuine Windowed arrays with HOST (numpy) compute and device output placement, like the original
    qualified test_native_windowed.py:139-159; CUDA compute for windowed storage stays refused by the baseline."""


    vc, w, al, g, gc, mres = pair
    ref = build((0.1, 0.2, 0.2, 0.4), vc, al, g, gc, mres, mask=full_mask(g.ny, g.nx), coverage=POLICY)
    # same route as the original qualified suite: maple.core.backend.to_device_tree(obj, xp) (xp REQUIRED; arrays.py:204)
    from maple.core.backend import to_device_tree

    dw, dal = to_device_tree(w, gpu), to_device_tree(al, gpu)
    t = build((0.1, 0.2, 0.2, 0.4), dw, dal, g, gc, mres, mask=full_mask(g.ny, g.nx), coverage=POLICY, backend="numpy")
    assert t.namespace == "cupy" and type(t.layer_mass_kg) is gpu.ndarray and texture_bytes(t) == texture_bytes(ref)
