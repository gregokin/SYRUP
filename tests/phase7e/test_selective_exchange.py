"""Phase 7e isolated MAPLE candidate: selective surface-window bed exchange with cached voxel totals.

Opt-in: runs only when the selected MAPLE is the recorded Phase 7e candidate (benchmarks/phase7e/
candidate_digest.txt); the accepted dependency skips. Original kernels are loaded from the accepted
72310c49 snapshot by file path, never copied. The two-environment transaction differential
(transaction_sequence.py) is exercised in-process against a saved accepted-environment NPZ when present.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from maple.core.backend import (
    resolve_backend,
    to_device,
    to_device_tree,
    to_host,
    to_host_tree,
)
from maple.core.boundaries import AxisBoundary, BoundaryKind
from maple.core.parameters.geometry import GeometrySpec
from maple.core.types.active_layer import ActiveLayerState
from maple.core.types.sediment_ledger import zeros_sediment_ledger_state
from maple.core.types.voxel import VoxelColumnState
from maple.coupling.sediment_ledger import accumulate
from maple.surface.active_layer.capacity import compute_active_layer_target_mass_kg
from maple.surface.active_layer.exchange import (
    update_active_layer_after_deposition,
    update_active_layer_after_erosion,
)
from maple.surface.voxels import transfer
from maple.surface.voxels.capacity import max_voxel_mass_kg
from maple.surface.voxels.serialization import (
    load_voxel_column_state,
    save_voxel_column_state,
)

from maple_syrup.dependency import resolve_maple_dependency
from maple_syrup.provenance import source_tree_digest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "outputs/dependencies/maple_72310c49/source/src/maple"
EXPECTED = (ROOT / "benchmarks/phase7e/candidate_digest.txt").read_text().strip()


@pytest.fixture(scope="module", autouse=True)
def candidate_identity():
    digest = source_tree_digest(resolve_maple_dependency().package_dir).digest_sha256
    if digest != EXPECTED:
        pytest.skip(f"selected MAPLE {digest[:8]} is not the Phase 7e candidate {EXPECTED[:8]}")


def reference(relative, name):
    spec = importlib.util.spec_from_file_location(name, BASELINE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def old_transfer():
    return reference("surface/voxels/transfer.py", "phase7e_original_transfer")


@pytest.fixture(scope="module")
def old_ledger():
    return reference("coupling/sediment_ledger/accumulate.py", "phase7e_original_accumulate")


@pytest.fixture(params=["numpy", "cupy"])
def xp(request):
    try:
        return resolve_backend(request.param).xp
    except Exception as exc:
        if request.param == "cupy":
            pytest.skip(f"GPU unavailable: {exc}")
        raise


def geometry(nx=3, ny=4, dz=0.1):
    return GeometrySpec(nx=nx, ny=ny, dx_m=0.5, dy_m=0.5, voxel_dz_m=dz, bulk_density_kg_m3=1250.0,
                        active_layer_thickness_m=0.002, boundary_x=AxisBoundary(kind=BoundaryKind.PERIODIC),
                        boundary_y=AxisBoundary(kind=BoundaryKind.PERIODIC))


def random_columns(rng, g, nv, nc, *, deep_room=False, empty_fraction=0.0):
    cap = max_voxel_mass_kg(g)
    mass = np.zeros((g.ny, g.nx, nv, nc))
    for y in range(g.ny):
        for x in range(g.nx):
            if rng.random() < empty_fraction:
                continue
            full = rng.integers(0, nv + 1)
            for k in range(full):
                mass[y, x, k] = rng.dirichlet(np.ones(nc)) * cap
                if deep_room and rng.random() < 0.5:
                    mass[y, x, k] *= 1.0 - 4.0 * np.finfo(float).eps
            if 0 < full < nv and rng.random() < 0.7:
                mass[y, x, full] = rng.dirichlet(np.ones(nc)) * cap * rng.uniform(0.01, 0.99)
    return mass


def result_arrays(result):
    return {f.name: np.asarray(to_host(getattr(result, f.name))) for f in dataclasses.fields(result)}


def assert_same_result(a, b):
    for key, value in result_arrays(a).items():
        assert np.array_equal(value, result_arrays(b)[key]), key


MRES = 1.0e-10
REQUEST_KINDS = ("zero", "tiny", "sub_bound", "small", "top_plus", "exhaust", "over")


def extraction_requests(kind, mass, cap, nv, rng):
    total = mass.sum((-1, -2))
    bound = transfer.summation_error_bound_kg(nv, cap)
    top_mass = np.take_along_axis(mass.sum(-1), np.maximum((mass.sum(-1) > 0).sum(-1) - 1, 0)[..., None], -1)[..., 0]
    return {
        "zero": np.zeros_like(total), "tiny": np.full_like(total, 1e-14), "sub_bound": np.full_like(total, bound * 0.5),
        "small": rng.random(total.shape) * 1e-3, "top_plus": top_mass + 1e-3, "exhaust": total * 1.1,
        "over": np.nextafter(total, np.inf),
    }[kind]


@pytest.mark.parametrize("nc", [1, 6, 9])
@pytest.mark.parametrize("nv", [1, 2, 3, 5, 20])
@pytest.mark.parametrize("kind", REQUEST_KINDS)
def test_selective_extraction_matches_original(xp, nc, nv, kind, old_transfer):
    g = geometry()
    rng = np.random.default_rng(7000 + nc * 31 + nv * 7 + REQUEST_KINDS.index(kind))
    mass = random_columns(rng, g, nv, nc, empty_fraction=0.2)
    request = extraction_requests(kind, mass, max_voxel_mass_kg(g), nv, rng)
    a = to_device(mass.copy(), xp)
    b = to_device(mass.copy(), xp)
    totals = b.sum(axis=-1)
    old = old_transfer._extract_surface_mixture_batched_inplace(a, to_device(request, xp), g, MRES)
    new = transfer._extract_surface_mixture_selective_inplace(b, to_device(request, xp), g, MRES, totals)
    assert np.array_equal(to_host(a), to_host(b))
    assert np.array_equal(to_host(totals), to_host(b.sum(axis=-1)))
    assert_same_result(old, new)


DEPOSIT_KINDS = ("zero", "sub_bound", "small", "fill_top", "cross_boundary", "deep_room", "mixed")


@pytest.mark.parametrize("nc", [1, 6, 9])
@pytest.mark.parametrize("nv", [1, 2, 3, 5, 20])
@pytest.mark.parametrize("kind", DEPOSIT_KINDS)
def test_selective_deposition_matches_original(xp, nc, nv, kind, old_transfer):
    g = geometry()
    cap = max_voxel_mass_kg(g)
    rng = np.random.default_rng(9000 + nc * 31 + nv * 7 + DEPOSIT_KINDS.index(kind))
    mass = random_columns(rng, g, nv, nc, deep_room=(kind in ("deep_room", "mixed")), empty_fraction=0.2)
    totals_host = mass.sum(-1)
    room = np.maximum(cap - totals_host, 0.0).sum(-1)  # total room per cell
    bound = transfer.summation_error_bound_kg(nv, cap)
    mix = rng.dirichlet(np.ones(nc), size=(g.ny, g.nx))
    scale = {
        "zero": np.zeros((g.ny, g.nx)), "sub_bound": np.full((g.ny, g.nx), bound * 0.5),
        "small": rng.random((g.ny, g.nx)) * 1e-3, "fill_top": np.minimum(room, cap * 0.5),
        "cross_boundary": np.minimum(room, cap * 1.5), "deep_room": rng.random((g.ny, g.nx)) * 1e-3,
        "mixed": np.where(rng.random((g.ny, g.nx)) < 0.5, bound * 0.5, np.minimum(room, cap * 1.2)),
    }[kind]
    request = mix * scale[..., None]
    a = to_device(mass.copy(), xp)
    b = to_device(mass.copy(), xp)
    totals = b.sum(axis=-1)
    totals_before = to_host(totals).copy()
    try:
        old = old_transfer._deposit_surface_mixture_batched_inplace(a, to_device(request, xp), g, MRES)
    except ValueError as exc:
        # The original refuses this request (over capacity): the selective entry must refuse
        # identically and leave both the mass and the totals untouched.
        with pytest.raises(ValueError, match="insufficient allocated voxel capacity"):
            transfer._deposit_surface_mixture_selective_inplace(b, to_device(request, xp), g, MRES, totals)
        assert "insufficient allocated voxel capacity" in str(exc)
        assert np.array_equal(to_host(a), mass) and np.array_equal(to_host(b), mass)
        assert np.array_equal(to_host(totals), totals_before)
        return
    new = transfer._deposit_surface_mixture_selective_inplace(b, to_device(request, xp), g, MRES, totals)
    assert np.array_equal(to_host(a), to_host(b))
    assert np.array_equal(to_host(totals), to_host(b.sum(axis=-1)))
    assert_same_result(old, new)


def test_deposition_capacity_failure_is_atomic(xp):
    g = geometry()
    cap = max_voxel_mass_kg(g)
    nv, nc = 4, 3
    mass = np.full((g.ny, g.nx, nv, nc), cap / nc)  # every column full
    mass[0, 0, -1] = 0.0  # one column with room in its top voxel
    request = np.zeros((g.ny, g.nx, nc))
    request[0, 0, 0] = cap * 0.5
    request[1, 1, 1] = 1.0  # genuinely over capacity -> whole call must refuse
    b = to_device(mass.copy(), xp)
    totals = b.sum(axis=-1)
    before, totals_before = to_host(b).copy(), to_host(totals).copy()
    with pytest.raises(ValueError, match="insufficient allocated voxel capacity"):
        transfer._deposit_surface_mixture_selective_inplace(b, to_device(request, xp), g, MRES, totals)
    assert np.array_equal(to_host(b), before)
    assert np.array_equal(to_host(totals), totals_before)


def test_exchange_routines_keep_cache_current_and_rebuild_on_new_binding(xp):
    g = geometry(nx=4, ny=5)
    nv, nc = 12, 6
    rng = np.random.default_rng(11)
    mass = random_columns(rng, g, nv, nc, deep_room=True)
    mass[:, :, -3:] = 0.0  # guarantee headroom: repeated burial below must never exceed allocated capacity
    target = compute_active_layer_target_mass_kg(g)
    active = rng.dirichlet(np.ones(nc), size=(g.ny, g.nx)) * target
    column = VoxelColumnState(mass_kg=to_device(mass, xp))
    layer = ActiveLayerState(mass_kg=to_device(active, xp))
    assert column.valid_surface_cache() is None
    transfer.reset_selective_statistics()
    removal = to_device(np.minimum(rng.random((g.ny, g.nx, nc)) * 1e-4, active), xp)
    column, layer, _ = update_active_layer_after_erosion(column, layer, removal, g, MRES)
    assert transfer.selective_statistics["cache_rebuilds"] == 1
    assert transfer.validate_voxel_surface_cache(column)
    deposit = to_device(rng.random((g.ny, g.nx, nc)) * 2e-5, xp)
    column, layer, _ = update_active_layer_after_deposition(column, layer, deposit, g, MRES)
    assert transfer.selective_statistics["cache_rebuilds"] == 1 + int(xp is not np)
    assert transfer.selective_statistics["cache_hits"] == int(xp is np)
    assert transfer.validate_voxel_surface_cache(column)
    transfer.reset_selective_statistics()
    for _ in range(20):
        column, layer, _ = update_active_layer_after_erosion(column, layer, removal, g, MRES, trust_surface_cache=True)
        column, layer, _ = update_active_layer_after_deposition(column, layer, deposit, g, MRES, trust_surface_cache=True)
        assert transfer.validate_voxel_surface_cache(column)
    assert transfer.selective_statistics["cache_rebuilds"] == 0
    assert transfer.selective_statistics["cache_hits"] == 40
    stats = transfer.selective_statistics
    assert stats["deposit_cells_fast"] + stats["deposit_cells_fallback"] + stats["deposit_cells_skipped"] == 20 * g.ny * g.nx
    # The cache never propagates: replace, tree conversion and ordinary construction drop it.
    assert dataclasses.replace(column, mass_kg=column.mass_kg.copy()).surface_cache is None
    assert to_host_tree(column).surface_cache is None
    assert VoxelColumnState(mass_kg=column.mass_kg).surface_cache is None
    with pytest.raises(TypeError):
        VoxelColumnState(mass_kg=column.mass_kg, surface_cache=column.surface_cache)  # init=False
    # A stale cache behind a new array is rebuilt, and the result equals the cached-path result bitwise.
    fresh = VoxelColumnState(mass_kg=column.mass_kg.copy())
    c1, l1, r1 = update_active_layer_after_erosion(column, layer, removal, g, MRES, trust_surface_cache=True)
    c2, l2, r2 = update_active_layer_after_erosion(fresh, layer, removal, g, MRES)
    assert transfer.selective_statistics["cache_rebuilds"] == 1
    assert np.array_equal(to_host(c1.mass_kg), to_host(c2.mass_kg))
    assert np.array_equal(to_host(l1.mass_kg), to_host(l2.mass_kg))
    assert_same_result(r1, r2)


def test_restart_from_snapshot_without_cache_is_equivalent(xp, tmp_path):
    g = geometry(nx=3, ny=3)
    nv, nc = 8, 4
    rng = np.random.default_rng(5)
    mass = random_columns(rng, g, nv, nc)
    target = compute_active_layer_target_mass_kg(g)
    active = rng.dirichlet(np.ones(nc), size=(g.ny, g.nx)) * target
    column = VoxelColumnState(mass_kg=to_device(mass, xp))
    layer = ActiveLayerState(mass_kg=to_device(active, xp))
    removal = to_device(np.minimum(rng.random((g.ny, g.nx, nc)) * 1e-4, active), xp)
    deposit = to_device(rng.random((g.ny, g.nx, nc)) * 3e-4, xp)
    for _ in range(5):
        column, layer, _ = update_active_layer_after_erosion(column, layer, removal, g, MRES, trust_surface_cache=True)
        column, layer, _ = update_active_layer_after_deposition(column, layer, deposit, g, MRES, trust_surface_cache=True)
    path = save_voxel_column_state(column, tmp_path / "column")
    restored = load_voxel_column_state(path)
    assert restored.surface_cache is None and np.array_equal(to_host(column.mass_kg), restored.mass_kg)
    restored = VoxelColumnState(mass_kg=to_device(restored.mass_kg, xp))
    a = update_active_layer_after_deposition(column, layer, deposit, g, MRES)
    b = update_active_layer_after_deposition(restored, layer, deposit, g, MRES)
    assert np.array_equal(to_host(a[0].mass_kg), to_host(b[0].mass_kg))
    assert np.array_equal(to_host(a[1].mass_kg), to_host(b[1].mass_kg))
    assert_same_result(a[2], b[2])


def ledger_fields(ledger):
    return {f.name: np.asarray(to_host(getattr(ledger, f.name))) for f in dataclasses.fields(ledger)}


@pytest.mark.parametrize("foreign_compensation", [False, True])
def test_single_process_ledger_matches_original(xp, old_ledger, foreign_compensation):
    ny, nx, nc = 5, 4, 6
    rng = np.random.default_rng(3 + foreign_compensation)
    ledger = zeros_sediment_ledger_state(ny, nx, nc)
    process = accumulate.process_index("water_erosion_deposition")
    # Drive nontrivial compensation on the water slice with the ORIGINAL routine first.
    for _ in range(3):
        actual = -rng.random((ny, nx, nc)) * np.logspace(-12, 2, nc)
        ledger = old_ledger.accumulate_process_transfer(ledger, process, actual, np.zeros_like(actual), MRES)
    if foreign_compensation:
        other = accumulate.process_index("permanent_deposition")
        ledger = old_ledger.accumulate_process_transfer(ledger, other, rng.random((ny, nx, nc)) * 1e-3,
                                                       np.zeros((ny, nx, nc)), MRES)
    ledger = type(ledger)(**{k: (to_device(v, xp) if isinstance(v, np.ndarray) else v) for k, v in
                            {f.name: getattr(ledger, f.name) for f in dataclasses.fields(ledger)}.items()})
    actual = to_device(-rng.random((ny, nx, nc)) * 1e-3, xp)
    residual = to_device(-rng.random((ny, nx, nc)) * 5e-11, xp)
    new = accumulate.accumulate_process_transfer(ledger, process, actual, residual, MRES)
    old = old_ledger.accumulate_process_transfer(ledger, process, actual, residual, MRES)
    for key, value in ledger_fields(old).items():
        assert np.array_equal(value, ledger_fields(new)[key]), key
    deposition = to_device(rng.random((ny, nx, nc)) * 1e-3, xp)
    new2 = accumulate.accumulate_process_transfer(new, process, deposition, to_device(np.zeros((ny, nx, nc)), xp), MRES)
    old2 = old_ledger.accumulate_process_transfer(old, process, deposition, to_device(np.zeros((ny, nx, nc)), xp), MRES)
    for key, value in ledger_fields(old2).items():
        assert np.array_equal(value, ledger_fields(new2)[key]), key


def test_ledger_sign_rule_still_enforced(xp):
    ny, nx, nc = 2, 2, 3
    ledger = to_device_tree(zeros_sediment_ledger_state(ny, nx, nc), xp)
    process = accumulate.process_index("aerodynamic_bed_removal")
    bad = to_device(np.full((ny, nx, nc), 1e-3), xp)  # positive on a nonpositive-only channel
    with pytest.raises(ValueError, match="nonpositive-only"):
        accumulate.accumulate_process_transfer(ledger, process, bad, to_device(np.zeros((ny, nx, nc)), xp), MRES)


ACCEPTED_SEQUENCE = ROOT / "outputs/phase7e/evidence/transaction_sequence_accepted.npz"


@pytest.mark.skipif(not ACCEPTED_SEQUENCE.exists(), reason="accepted-environment sequence evidence not present")
def test_transaction_sequence_matches_accepted_environment(tmp_path):
    import transaction_sequence  # benchmarks/phase7e, via conftest path

    out = tmp_path / "candidate.npz"
    transaction_sequence.main(out, steps=48)
    with np.load(ACCEPTED_SEQUENCE) as old, np.load(out) as new:
        assert set(old.files) == set(new.files)
        bad = [k for k in old.files if not k.endswith(".error") and not np.array_equal(old[k], new[k])]
        for k in old.files:
            if k.endswith(".error"):
                prefix = "deposit_surface_mixture: insufficient allocated voxel capacity"
                assert old[k].tobytes().decode().startswith("ValueError("), k
                assert old[k].tobytes().decode()[12:].startswith(prefix), k
                assert new[k].tobytes().decode().startswith("ValueError("), k
                assert new[k].tobytes().decode()[12:].startswith(prefix), k
    assert not bad, bad[:10]


# --- trust boundary regressions (Codex reproducer: in-place mutation behind a bound cache) ------------------
def _cached_column(xp, g, nv, nc, rng):
    target = compute_active_layer_target_mass_kg(g)
    mass = random_columns(rng, g, nv, nc)
    cap = max_voxel_mass_kg(g)
    for y in range(g.ny):  # at least two full voxels everywhere so refills can always complete
        for x in range(g.nx):
            if mass[y, x, 1].sum() < cap:
                mass[y, x, :2] = rng.dirichlet(np.ones(nc), size=2) * cap
    mass[:, :, -2:] = 0.0
    active = rng.dirichlet(np.ones(nc), size=(g.ny, g.nx)) * target
    column = VoxelColumnState(mass_kg=to_device(mass, xp))
    layer = ActiveLayerState(mass_kg=to_device(active, xp))
    column, layer, _ = update_active_layer_after_erosion(column, layer, to_device(np.zeros((g.ny, g.nx, nc)), xp), g, MRES)
    assert column.surface_cache is not None
    return column, layer


def test_cached_column_mass_is_read_only_where_enforceable(xp):
    g = geometry()
    column, _ = _cached_column(xp, g, 6, 3, np.random.default_rng(1))
    if xp is np:
        assert not column.mass_kg.flags.writeable
        with pytest.raises(ValueError):
            column.mass_kg[0, 0, 0, 0] = 0.0
        assert column.valid_surface_cache() is not None
    else:
        # CuPy cannot enforce read-only arrays: the cache is NOT relied upon without declared trust.
        assert column.valid_surface_cache() is None
        assert column.valid_surface_cache(trusted=True) is not None


def test_mutated_writable_array_behind_cache_is_not_trusted(xp):
    """Codex reproducer: make the array writable again, mutate it to another valid (shallower) column
    without changing identity, then exchange. The cache must be ignored and the result must equal a
    fresh column's result (active layer refilled to target: 0.625 kg per cell for Plot1 geometry)."""
    g = geometry()
    nv, nc = 6, 3
    column, layer = _cached_column(xp, g, nv, nc, np.random.default_rng(2))
    host = to_host(column.mass_kg).copy()
    host[:, :, 2:] = 0.0  # shallower, still bottom-full and valid
    host[:, :, 1] *= 0.5
    if xp is np:
        column.mass_kg.flags.writeable = True  # deliberate contract break
        column.mass_kg[...] = host
        assert column.valid_surface_cache() is None  # writable: identity binding alone is not trusted
    else:
        column.mass_kg[...] = to_device(host, xp)
    removal = to_device(np.full((g.ny, g.nx, nc), 1e-4), xp)
    fresh = VoxelColumnState(mass_kg=to_device(host.copy(), xp))
    c1, l1, r1 = update_active_layer_after_erosion(column, layer, removal, g, MRES)
    c2, l2, r2 = update_active_layer_after_erosion(fresh, layer, removal, g, MRES)
    assert np.array_equal(to_host(c1.mass_kg), to_host(c2.mass_kg))
    assert np.array_equal(to_host(l1.mass_kg), to_host(l2.mass_kg))
    assert_same_result(r1, r2)
    target = compute_active_layer_target_mass_kg(g)
    assert np.allclose(to_host(l1.mass_kg).sum(-1), target)
    assert transfer.validate_voxel_surface_cache(c1)
    if xp is not np:
        # Declared trust on a mutated CuPy array IS the caller's contract violation; the validator detects it.
        with pytest.raises(ValueError):
            transfer.validate_voxel_surface_cache(column, trusted=True)


def test_corrupted_cached_totals_are_detected_by_validator(xp):
    g = geometry()
    column, _ = _cached_column(xp, g, 6, 3, np.random.default_rng(3))
    bad_totals = to_host(column.surface_cache.voxel_totals_kg).copy()
    bad_totals[0, 0, 0] += 1e-12
    corrupted = transfer.column_with_surface_cache(to_device(to_host(column.mass_kg).copy(), xp), to_device(bad_totals, xp))
    with pytest.raises(ValueError, match="disagree"):
        transfer.validate_voxel_surface_cache(corrupted)
    assert transfer.validate_voxel_surface_cache(column)


def test_backend_conversion_and_restore_drop_cache_and_rebuild(xp, tmp_path):
    g = geometry()
    column, layer = _cached_column(xp, g, 6, 3, np.random.default_rng(4))
    converted = to_host_tree(column)
    assert converted.surface_cache is None
    if xp is not np:
        from maple.core.backend import to_device_tree
        back = to_device_tree(converted, xp)
        assert back.surface_cache is None and type(back.mass_kg) is type(column.mass_kg)
    path = save_voxel_column_state(column, tmp_path / "c")
    restored = load_voxel_column_state(path)
    assert restored.surface_cache is None
    transfer.reset_selective_statistics()
    dep = to_device(np.full((g.ny, g.nx, 3), 1e-5), xp)
    a = update_active_layer_after_deposition(column, layer, dep, g, MRES, trust_surface_cache=True)
    hits_after_cached = transfer.selective_statistics["cache_hits"]
    b = update_active_layer_after_deposition(VoxelColumnState(mass_kg=to_device(restored.mass_kg, xp)), layer, dep, g, MRES)
    assert hits_after_cached == 1 and transfer.selective_statistics["cache_rebuilds"] == 1
    assert np.array_equal(to_host(a[0].mass_kg), to_host(b[0].mass_kg))
    assert_same_result(a[2], b[2])


def test_touched_storage_bed_change_matches_full_inventory_within_validator_bound(xp):
    from maple.core.types.sediment_availability import SedimentAvailabilityState
    from maple.core.types.water import WaterState
    from maple.water import WaterProcessDemand, apply_water_process_demand
    from maple.water.validation import validate_water_process_result

    from maple_syrup.case_import import verify_plot1_case
    from maple_syrup.sediment_bed import bed_from_case

    g = geometry(nx=4, ny=5)
    classes = bed_from_case(verify_plot1_case(ROOT / "outputs/plot1", allow_maple_source_change=True).case)[1].grain_classes
    nv, nc = 10, len(classes.classes)
    rng = np.random.default_rng(6)
    column, layer = _cached_column(xp, g, nv, nc, rng)
    avail = SedimentAvailabilityState(available_mass_kg=layer.mass_kg, bound_mass_kg=xp.zeros_like(layer.mass_kg))
    frac = to_device(np.ones((g.ny, g.nx, nc)), xp)
    water = WaterState(depth_m=to_device(np.full((g.ny, g.nx), 0.01), xp),
                       mobile_mass_by_cell_class_kg=to_device(rng.random((g.ny, g.nx, nc)) * 1e-2, xp))
    ledger = to_device_tree(zeros_sediment_ledger_state(g.ny, g.nx, nc), xp)
    removal = to_device(np.minimum(rng.random((g.ny, g.nx, nc)) * 1e-4, to_host(layer.mass_kg)), xp)
    deposition = water.mobile_mass_by_cell_class_kg * 0.7
    demand = WaterProcessDemand(removal, deposition)
    full = apply_water_process_demand(column, layer, water, ledger, demand, g, classes, MRES,
                                      sediment_availability=avail, initial_available_fraction=frac)
    touched = apply_water_process_demand(column, layer, water, ledger, demand, g, classes, MRES,
                                         sediment_availability=avail, initial_available_fraction=frac,
                                         trust_surface_cache=True, bed_change_from_touched_storage=True)
    # Physical state and transfers identical; only the bed-change DIAGNOSTIC's summation differs.
    assert np.array_equal(to_host(full.new_voxel_column.mass_kg), to_host(touched.new_voxel_column.mass_kg))
    assert np.array_equal(to_host(full.new_active_layer.mass_kg), to_host(touched.new_active_layer.mass_kg))
    assert np.array_equal(to_host(full.actual_removal_by_class_kg), to_host(touched.actual_removal_by_class_kg))
    assert np.array_equal(to_host(full.deposited_mass_by_class_kg), to_host(touched.deposited_mass_by_class_kg))
    gap = np.abs(to_host(full.bed_change_by_class_kg) - to_host(touched.bed_change_by_class_kg))
    scale = float(to_host(full.new_voxel_column.mass_kg).sum() + to_host(full.new_active_layer.mass_kg).sum())
    bound = transfer.summation_error_bound_kg(2 * g.ny * g.nx * (nv + 1) + 2 * g.ny * g.nx, max(scale, 1.0))
    assert np.all(gap <= bound)
    expected = to_host(touched.deposited_mass_by_class_kg) - to_host(touched.actual_removal_by_class_kg)
    assert np.all(np.abs(to_host(touched.bed_change_by_class_kg) - expected) <= bound)
    validate_water_process_result(touched, ny=g.ny, nx=g.nx, n_classes=nc, mass_resolution_kg=MRES)
    stats = transfer.selective_statistics
    assert stats["touched_delta_bed_changes"] >= 1 and stats["inventory_scans"] >= 2


def test_zero_effect_cells_are_skipped_but_reconciliation_case_is_not(xp, old_transfer):
    g = geometry()
    nv, nc = 5, 2
    cap = max_voxel_mass_kg(g)
    mass = np.zeros((g.ny, g.nx, nv, nc))
    mass[:, :, :2] = cap / nc  # two full voxels, empty top everywhere ...
    mass[0, 0, :] = cap / nc  # ... except one column whose TOP ALLOCATED voxel holds mass
    mass[0, 0, -1] = 1e-14  # sub-bound: the zero request's reconciliation drains it
    request = np.zeros((g.ny, g.nx))
    a, b = to_device(mass.copy(), xp), to_device(mass.copy(), xp)
    totals = b.sum(axis=-1)
    transfer.reset_selective_statistics()
    old = old_transfer._extract_surface_mixture_batched_inplace(a, to_device(request, xp), g, MRES)
    new = transfer._extract_surface_mixture_selective_inplace(b, to_device(request, xp), g, MRES, totals)
    assert np.array_equal(to_host(a), to_host(b)) and to_host(b)[0, 0, -1].sum() == 0.0
    assert_same_result(old, new)
    s = transfer.selective_statistics
    assert s["extract_cells_skipped"] == g.ny * g.nx - 1 and s["extract_cells_fast"] + s["extract_cells_fallback"] == 1
    # zero deposition request: every cell skipped, no kernel entries touched
    transfer.reset_selective_statistics()
    z = to_device(np.zeros((g.ny, g.nx, nc)), xp)
    new = transfer._deposit_surface_mixture_selective_inplace(b, z, g, MRES, totals)
    assert transfer.selective_statistics["deposit_cells_skipped"] == g.ny * g.nx
    assert transfer.selective_statistics["deposit_kernel_entries_fast"] == 0
    assert float(to_host(new.actual_total_mass_kg).sum()) == 0.0


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -1.0])
def test_inventory_scale_rejects_invalid_values(scale):
    from maple.water.validation import validate_water_process_result
    with pytest.raises(ValueError, match="bed_inventory_scale_kg"):
        validate_water_process_result(None, ny=1, nx=1, n_classes=1,
                                      mass_resolution_kg=MRES, bed_inventory_scale_kg=scale)
