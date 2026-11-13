"""Phase 2 E2: compact selected-cell texture batches + cold canonical depth on genuine native Dense/Windowed beds.
CPU/host only. Gated CUDA definitions at the bottom run only with E1_ALLOW_GPU=1 (set by Root, never by the author)."""
import dataclasses
import gc as pygc
import json
import os
import weakref
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from texture_fixtures import (
    BUDGET,
    full_mask,
    host,
    native_deep_bed,
    random_native_bed,
    stratified_dense,
    windowed_twin,
)

from maple_syrup.compact_soil_profile import (
    CompactBatch,
    canonical_depth_m,
    estimate_compact_memory,
    iter_compact_soil_profile,
)
from maple_syrup.soil_profile_texture import (
    SoilProfileGeometryError,
    SoilProfileMemoryError,
    SoilProfileTextureError,
    build_soil_profile_texture,
)

CONTROLS = Path(__file__).resolve().parents[3] / "root_e1_partial_texture_fraction_controls.json"
LAYERS = (0.1, 0.2, 0.2, 0.3)


def compact(vc, al, g, gc, mres, layers, idx, *, mask=None, **kw):
    kw.setdefault("memory_budget_bytes", BUDGET)
    mask = full_mask(g.ny, g.nx) if mask is None else mask
    return iter_compact_soil_profile(vc, al, g, gc, mres, layers, indices=np.asarray(idx, dtype=np.int64), active_mask=mask, **kw)


def whole(vc, al, g, gc, mres, layers, mask, **kw):
    return build_soil_profile_texture(vc, al, g, gc, mres, layers, active_mask=mask, memory_budget_bytes=BUDGET, **kw)


def collect(it):
    return list(it)


def stack(batches, name, axis):
    return np.concatenate([host(getattr(b, name)) for b in batches], axis=axis)


def assert_matches_whole(batches, t, idx):
    idx = np.asarray(idx)
    assert np.array_equal(np.concatenate([b.indices for b in batches]), idx)
    for name, axis in (("layer_mass_kg", 1), ("layer_fraction", 1), ("layer_total_kg", 1), ("coverage_m", 1),
                       ("represented_depth_m", 0), ("active_cap_thickness_m", 0)):
        got = stack(batches, name, axis)
        ref = host(getattr(t, name))
        ref = ref[:, idx] if axis == 1 else ref[idx]
        assert got.tobytes() == np.ascontiguousarray(ref).tobytes(), name


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("batch", [1, 2, 3, 100])
def test_dense_compact_equals_whole_e1_bytes(backend, batch):
    if backend == "numba":
        pytest.importorskip("numba")
    vc, al, g, gc, mres = random_native_bed(3, ny=3, nx=5)
    idx = [0, 2, 3, 7, 14]
    mask = full_mask(g.ny, g.nx)
    sel = np.zeros(15, bool)
    sel[idx] = True
    t = whole(vc, al, g, gc, mres, LAYERS, sel.reshape(3, 5), coverage="partial_over_base", backend=backend)
    it = compact(vc, al, g, gc, mres, LAYERS, idx, mask=mask, coverage="partial_over_base", backend=backend, batch_cells=batch)
    bs = collect(it)
    assert all(isinstance(b, CompactBatch) and b.n_cells <= batch for b in bs)
    assert_matches_whole(bs, t, idx)
    assert bs[0].source["result_kind"] == "compact_selected_columns" and bs[0].source["whole_grid"] is False
    assert it.source["selection"]["n_selected"] == 5 and bs[0].layer_mass_kg.shape == (4, bs[0].n_cells, 4)


def test_windowed_compact_equals_whole_and_reads_archive(tmp_path):
    vc, al, g, gc, mres = native_deep_bed()
    w = windowed_twin(vc, g, mres, tmp_path / "store")
    assert int(host(w.base).min()) > 0
    idx = [1, 2, 4]
    sel = np.zeros(g.ny * g.nx, bool)
    sel[idx] = True
    t = whole(w, al, g, gc, mres, LAYERS, sel.reshape(g.ny, g.nx), coverage="partial_over_base")
    it = compact(w, al, g, gc, mres, LAYERS, idx, coverage="partial_over_base", batch_cells=2)
    bs = collect(it)
    assert_matches_whole(bs, t, idx)
    assert "validate_windowed_column_full" in it.source["validation"] and it.source["windowed"]["resident_floor_is_not_bedrock"]
    # the compact result equals the Dense compact result bitwise (archive below the window was read)
    dense = collect(compact(vc, al, g, gc, mres, LAYERS, idx, coverage="partial_over_base", batch_cells=2))
    for name, axis in (("layer_mass_kg", 1), ("canonical_depth_m", 0)):
        assert stack(bs, name, axis).tobytes() == stack(dense, name, axis).tobytes()
    assert float(np.min(stack(bs, "canonical_depth_m", 0))) > 0.01 * int(host(w.base).min())


def test_fraction_control_stratified_shallow_and_all_rock():
    with open(CONTROLS) as handle:
        d = json.load(handle)
    edges = [Fraction(e) for e in d["edges_m"]]
    layers = tuple(float(edges[i + 1] - edges[i]) for i in range(len(edges) - 1))
    vc, al, g, gc, mres = stratified_dense()
    v, a = np.array(vc.mass_kg), np.array(al.mass_kg)
    v[0, 1], a[0, 1] = 0.0, 0.0  # cell 1 becomes the all-rock control (active, nothing)
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    vc, al = VoxelColumnState(mass_kg=v), ActiveLayerState(mass_kg=a)
    bs = collect(compact(vc, al, g, gc, mres, layers, [0, 1, 2], mask=np.array([[True, True, True]]), coverage="partial_over_base"))
    mass = stack(bs, "layer_mass_kg", 1)
    for cell, label in ((0, "stratified"), (1, "all_rock"), (2, "shallow")):
        ctrl = next(c for c in d["controls"] if c["label"] == label)
        exp = np.array([[float(Fraction(x)) for x in row] for row in ctrl["mass_kg"]])
        np.testing.assert_allclose(mass[:, cell, :], exp, rtol=1e-12, atol=1e-13, err_msg=label)
        np.testing.assert_allclose(stack(bs, "coverage_m", 1)[:, cell], [float(Fraction(x)) for x in ctrl["coverage_m"]],
                                   rtol=1e-12, atol=1e-13)


def native_depth(vc, al, g, mres):
    from maple.surface.active_layer.diagnostics import diagnose_combined_surface_state

    return np.asarray(diagnose_combined_surface_state(vc, al, g, mres).combined_elevation_m).reshape(-1)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("batch", [1, 3, 7, 64])
def test_canonical_depth_is_bitwise_the_native_diagnostic(seed, batch):
    vc, al, g, gc, mres = random_native_bed(seed, ny=3, nx=5, n_voxels=30)
    ref = native_depth(vc, al, g, mres)
    idx = np.arange(15, dtype=np.int64)[seed % 2::2] if seed % 3 else np.arange(15, dtype=np.int64)
    bs = collect(compact(vc, al, g, gc, mres, (0.05, 0.05), idx, coverage="partial_over_base", batch_cells=batch))
    got = np.concatenate([b.canonical_depth_m for b in bs])
    assert got.tobytes() == ref[idx].tobytes()
    # the sequential S1 diagnostic is a different quantity (kept distinct, never substituted)
    seq = stack(bs, "represented_depth_m", 0)
    assert np.allclose(seq, got, rtol=1e-9)


def test_canonical_depth_zero_edges_and_total_depth():
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    vc, al, g, gc, mres = stratified_dense()
    v, a = np.array(vc.mass_kg), np.array(al.mass_kg)
    v[0, 1], a[0, 1] = 0.0, 0.0  # zero depth
    v[0, 2] = 0.0  # active cap only
    vc2, al2 = VoxelColumnState(mass_kg=v), ActiveLayerState(mass_kg=a)
    ref = native_depth(vc2, al2, g, mres)
    bs = collect(compact(vc2, al2, g, gc, mres, (0.01, 0.01, 0.01, 0.5), [0, 1, 2], coverage="partial_over_base"))
    got = np.concatenate([b.canonical_depth_m for b in bs])
    assert got.tobytes() == ref.tobytes() and got[1] == 0.0
    # exactly the nominal total depth of the stratified column (0.222 m): equal to the native value, never to a clipped one
    assert got[0] == ref[0] and abs(got[0] - 0.222) < 1e-15


def test_canonical_depth_function_validates_inputs():
    vc, al, g, _gc, _mres = stratified_dense()
    with pytest.raises(SoilProfileTextureError):
        canonical_depth_m(np.asfortranarray(np.array(vc.mass_kg.reshape(3, 4, 4))), np.array(al.mass_kg.reshape(3, 4)), g)


def test_empty_selection_is_a_noop(monkeypatch):
    import maple_syrup.soil_profile_texture as spt

    def boom(*a, **k):
        raise AssertionError("work performed for an empty selection")

    for name in ("_runner", "_native_identity", "_host_mask", "_validate_dense_batch"):
        monkeypatch.setattr(spt, name, boom)
    vc, al, g, gc, mres = stratified_dense()
    it = compact(vc, al, g, gc, mres, LAYERS, [])
    assert it.n_selected == 0 and it.memory is None and it.source is None and list(it) == []


@pytest.mark.parametrize("bad", [[2, 1], [1, 1], [-1, 0], [0, 99], np.zeros((1, 2), np.int64), np.array([0], np.int32), [0, 1], "x"])
def test_invalid_selection_refused(bad):
    vc, al, g, gc, mres = stratified_dense()
    arr = bad if isinstance(bad, np.ndarray) else (bad if isinstance(bad, str) else np.asarray(bad, dtype=np.int64))
    if isinstance(bad, list) and bad == [0, 1]:
        pytest.skip("valid selection control")
    with pytest.raises(SoilProfileTextureError):
        iter_compact_soil_profile(vc, al, g, gc, mres, LAYERS, indices=arr, active_mask=full_mask(1, 3), memory_budget_bytes=BUDGET)


def test_inactive_selected_cell_refused_and_inactive_unselected_ignored():
    vc, al, g, gc, mres = stratified_dense()
    mask = np.array([[True, False, True]])
    with pytest.raises(SoilProfileTextureError, match="must be active"):
        compact(vc, al, g, gc, mres, (0.05, 0.05), [0, 1], mask=mask)
    bs = collect(compact(vc, al, g, gc, mres, (0.05, 0.05), [0, 2], mask=mask, coverage="partial_over_base"))
    assert np.array_equal(np.concatenate([b.indices for b in bs]), [0, 2])


def test_value_refusals_and_no_mutation():
    vc, al, g, gc, mres = stratified_dense()
    v0, a0 = vc.mass_kg.copy(), al.mass_kg.copy()
    with pytest.raises(SoilProfileGeometryError, match="do not represent enough"):
        collect(compact(vc, al, g, gc, mres, (0.1, 0.1, 0.3), [0]))
    with pytest.raises(SoilProfileGeometryError, match="cap thicker"):
        collect(compact(vc, al, g, gc, mres, (0.001, 0.1), [0], coverage="partial_over_base"))
    with pytest.raises(SoilProfileTextureError, match="coverage must be"):
        compact(vc, al, g, gc, mres, (0.1,), [0], coverage="x")
    assert np.array_equal(vc.mass_kg, v0) and np.array_equal(al.mass_kg, a0)


def test_unpublished_journal_rejected_before_any_batch(tmp_path):
    from maple.surface.voxels.deep_store import DeepJournal
    from maple.surface.voxels.windowed_pending import attach_pending_journal

    vc, al, g, gc, mres = native_deep_bed()
    w = windowed_twin(vc, g, mres, tmp_path / "store")
    deep = w.deep_store.gather_columns(np.zeros(1, dtype=np.int64))[0]
    journal = DeepJournal.from_records(np.zeros(2, dtype=np.int64), np.arange(2, dtype=np.int64), np.array(deep[:2]), header=w.deep_store.header)
    twin = attach_pending_journal(dataclasses.replace(w), journal)
    with pytest.raises(SoilProfileTextureError, match="unpublished deep journal"):
        compact(twin, al, g, gc, mres, LAYERS, [0], coverage="partial_over_base")
    assert twin.pending_deep is not None


def test_budget_refusal_before_any_gather(tmp_path):
    from maple.surface.voxels.deep_store import (
        read_deep_store_counters,
        reset_deep_store_counters,
    )

    vc, al, g, gc, mres = native_deep_bed()
    w = windowed_twin(vc, g, mres, tmp_path / "store")
    reset_deep_store_counters()
    with pytest.raises(SoilProfileMemoryError, match="nothing was gathered"):
        compact(w, al, g, gc, mres, LAYERS, [0, 1], memory_budget_bytes=1000, batch_cells=2)
    assert read_deep_store_counters()["columns_read"] == 0
    it = compact(vc, al, g, gc, mres, LAYERS, [0, 1, 2, 3, 4], coverage="partial_over_base", memory_budget_bytes=10**9)
    full_b = it.memory.batch_cells
    small = estimate_compact_memory(representation="dense", backend="numpy", output_namespace="numpy", n_cells=6, n_selected=5,
                                    n_layers=4, n_classes=4, n_voxels=40, batch_cells=1).required_bytes
    it2 = compact(vc, al, g, gc, mres, LAYERS, [0, 1, 2, 3, 4], coverage="partial_over_base", memory_budget_bytes=small + 1)
    assert it2.memory.batch_cells <= full_b and it2.source["selection"]["explicit_batch"] is False


def test_compact_memory_has_no_full_domain_terms():
    kw = {"representation": "dense", "backend": "numpy", "output_namespace": "numpy", "n_layers": 6, "n_classes": 5,
          "n_voxels": 300, "n_selected": 100, "batch_cells": 100}
    small = estimate_compact_memory(n_cells=10**4, **kw)
    big = estimate_compact_memory(n_cells=10**7, **kw)
    assert big.required_bytes - small.required_bytes == 10**7 - 10**4  # only the host mask copy scales with N
    whole_outputs = (2 * 6 * 10**7 * 5 + 2 * 6 * 10**7 + 2 * 10**7) * 8
    assert big.required_bytes < whole_outputs / 100
    assert all(v >= 0 for v in dataclasses.astuple(big) if isinstance(v, int))
    assert big.required_bytes == big.host_bytes + big.device_bytes + big.reserve_bytes


def test_batch_peak_vs_model_and_release_behaviour():
    import tracemalloc

    vc, al, g, gc, mres = random_native_bed(1, ny=3, nx=5, n_voxels=30)
    it = compact(vc, al, g, gc, mres, (0.05, 0.05), list(range(15)), coverage="partial_over_base", batch_cells=2)
    collect(compact(vc, al, g, gc, mres, (0.05, 0.05), [0], coverage="partial_over_base"))  # warm
    tracemalloc.start()
    refs = []
    for b in it:
        refs.append(weakref.ref(b))
        del b
        pygc.collect()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert all(r() is None for r in refs), "the iterator retained a batch"
    assert peak <= it.memory.required_bytes  # ndarray payload + the declared interpreter allowance, bounded by B not N
    assert it.memory.batch_cells == 2 and len(refs) == 8


def test_from_gc_iterator_holds_no_history():
    vc, al, g, gc, mres = random_native_bed(2, ny=3, nx=5, n_voxels=30)
    it = compact(vc, al, g, gc, mres, (0.05, 0.05), list(range(15)), coverage="partial_over_base", batch_cells=4)
    first = next(it)
    r = weakref.ref(first)
    del first
    pygc.collect()
    next(it)
    assert r() is None
    it.close()
    with pytest.raises(StopIteration):
        next(it)


def test_selection_is_owned_and_caller_mutation_cannot_change_batches_or_digest():
    vc, al, g, gc, mres = random_native_bed(5, ny=3, nx=5, n_voxels=30)
    mask = full_mask(3, 5)
    v0, a0, m0 = vc.mass_kg.copy(), al.mass_kg.copy(), mask.copy()
    caller = np.array([1, 3, 4, 8, 11], dtype=np.int64)
    it = iter_compact_soil_profile(vc, al, g, gc, mres, (0.05, 0.05), indices=caller, active_mask=mask, memory_budget_bytes=BUDGET,
                                   coverage="partial_over_base", batch_cells=2)
    digest = it.source["selection"]["indices_sha256"]
    import hashlib

    assert digest == hashlib.sha256(np.array([1, 3, 4, 8, 11], dtype=np.int64).tobytes()).hexdigest()
    first = next(it)
    caller[:] = [0, 2, 5, 6, 7]  # caller mutates its array after construction
    mask[:] = False  # and its mask
    rest = list(it)
    got = np.concatenate([first.indices] + [b.indices for b in rest])
    assert np.array_equal(got, [1, 3, 4, 8, 11]) and it.source["selection"]["indices_sha256"] == digest
    assert np.array_equal(vc.mass_kg, v0) and np.array_equal(al.mass_kg, a0) and not mask.any() and m0.all()
    assert first.source["selection"]["indices_sha256"] == digest


def test_one_byte_budget_refuses_before_selection_copy_mask_or_hash(monkeypatch):
    import maple_syrup.compact_soil_profile as cs
    import maple_syrup.soil_profile_texture as spt

    def boom(*a, **k):
        raise AssertionError("work before admission")

    monkeypatch.setattr(cs, "_seal_selection", boom)
    monkeypatch.setattr(spt, "_host_mask", boom)
    monkeypatch.setattr(spt, "_native_identity", boom)
    vc, al, g, gc, mres = stratified_dense()
    with pytest.raises(SoilProfileMemoryError, match="nothing was gathered"):
        compact(vc, al, g, gc, mres, (0.05, 0.05), [0, 1], memory_budget_bytes=1)


def test_selection_scratch_is_counted_and_chunked():
    est = estimate_compact_memory(representation="dense", backend="numpy", output_namespace="numpy", n_cells=1000, n_selected=900,
                                  n_layers=3, n_classes=4, n_voxels=40, batch_cells=100)
    assert est.selection_bytes == 8 * 900 + 4 * 100
    # a strictly-increasing violation exactly at a chunk boundary is found without a full-C boolean
    vc, al, g, gc, mres = random_native_bed(6, ny=3, nx=5, n_voxels=30)
    bad = np.array([0, 1, 2, 2, 5], dtype=np.int64)
    with pytest.raises(SoilProfileTextureError, match="strictly increasing"):
        compact(vc, al, g, gc, mres, (0.05, 0.05), bad, batch_cells=2)  # duplicate sits across chunk edge 2|2
    with pytest.raises(SoilProfileTextureError, match="strictly increasing"):
        compact(vc, al, g, gc, mres, (0.05, 0.05), np.array([0, 3, 2, 9], dtype=np.int64), batch_cells=2)


def test_empty_selection_returns_before_layers_budget_and_geometry_values(monkeypatch):
    import maple_syrup.soil_profile_texture as spt

    monkeypatch.setattr(spt, "canonical_layers", lambda *a: (_ for _ in ()).throw(AssertionError("layers canonicalised")))
    vc, al, g, gc, mres = stratified_dense()
    it = iter_compact_soil_profile(vc, al, g, gc, mres, "not-even-layers", indices=np.zeros(0, np.int64), active_mask=full_mask(1, 3),
                                   memory_budget_bytes=0)  # cheap metadata checks only; budget/layers never examined
    assert list(it) == []
    with pytest.raises(SoilProfileTextureError):  # cheap checks are still honest
        iter_compact_soil_profile(vc, al, g, gc, mres, (0.1,), indices=np.zeros(0, np.int32), active_mask=full_mask(1, 3), memory_budget_bytes=1)


def test_device_guard_spy_refuses_before_any_work(monkeypatch):
    import maple_syrup.compact_soil_profile as cs
    import maple_syrup.soil_profile_texture as spt

    vc, al, g, gc, mres = stratified_dense()

    class Dev:
        def __init__(self, i):
            self.id = i

    class Arr:  # narrow stand-in for a device array's `.device` only (no native class faked)
        def __init__(self, i):
            self.device = Dev(i)

    class Xp:
        class cuda:
            class runtime:
                getDevice = staticmethod(lambda: 0)

    assert cs._guard_device("numpy", None, []) is None
    assert cs._guard_device("cupy", Xp, [Arr(0), Arr(0)]) == 0
    with pytest.raises(SoilProfileTextureError, match="different devices"):
        cs._guard_device("cupy", Xp, [Arr(0), Arr(1)])
    with pytest.raises(SoilProfileTextureError, match="never switched"):
        cs._guard_device("cupy", Xp, [Arr(1), Arr(1)])
    # the guard sits before the mask copy and validators in the real entry point
    calls = []
    monkeypatch.setattr(cs, "_guard_device", lambda *a: calls.append("guard") or None)
    monkeypatch.setattr(spt, "_host_mask", lambda *a: calls.append("mask") or (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        compact(vc, al, g, gc, mres, (0.05, 0.05), [0])
    assert calls == ["guard", "mask"]


def test_device_mask_with_cpu_resident_source_is_refused_before_any_transfer(monkeypatch):
    import maple_syrup.soil_profile_texture as spt

    vc, al, g, gc, mres = stratified_dense()

    class DeviceMask(np.ndarray):  # narrow metadata stand-in: only the namespace probe is intercepted
        pass

    mask = full_mask(1, 3).view(DeviceMask)
    import maple.core.backend as mb

    real = mb.array_namespace
    monkeypatch.setattr(mb, "array_namespace", lambda *a: DeviceMask if isinstance(a[0], DeviceMask) else real(*a))
    monkeypatch.setattr(mb, "namespace_name", lambda xp: "cupy" if xp is DeviceMask else "numpy")

    def boom(*a, **k):
        raise AssertionError("transfer before the device-mask policy check")

    monkeypatch.setattr(spt, "_host_mask", boom)
    with pytest.raises(SoilProfileTextureError, match="device-resident active_mask"):
        iter_compact_soil_profile(vc, al, g, gc, mres, (0.05, 0.05), indices=np.array([0], np.int64), active_mask=mask,
                                  memory_budget_bytes=BUDGET)


def test_source_device_is_in_metadata_and_host_is_none():
    vc, al, g, gc, mres = stratified_dense()
    assert compact(vc, al, g, gc, mres, (0.05, 0.05), [0]).source["device_id"] is None


def test_canonical_depth_exact_types_and_required_order_metadata(monkeypatch):
    vc, al, g, gc, mres = stratified_dense()
    v = np.ascontiguousarray(np.array(vc.mass_kg.reshape(3, 4, 4)))
    a = np.ascontiguousarray(np.array(al.mass_kg.reshape(3, 4)))

    class Sub(np.ndarray):
        pass

    with pytest.raises(SoilProfileTextureError):
        canonical_depth_m(v.view(Sub), a, g)
    with pytest.raises(SoilProfileTextureError):
        canonical_depth_m(v, np.ma.masked_array(a), g)
    assert canonical_depth_m(v, a, g).shape == (3,)
    import sys
    import types

    stub = types.ModuleType("maple.core.backend.numpy_pairwise_order")  # lacks ORDER_ID
    monkeypatch.setitem(sys.modules, "maple.core.backend.numpy_pairwise_order", stub)
    import maple.core.backend as mb

    monkeypatch.setattr(mb, "numpy_pairwise_order", stub, raising=False)
    with pytest.raises(SoilProfileTextureError, match="ORDER_ID"):
        compact(vc, al, g, gc, mres, (0.05, 0.05), [0])


def test_no_untracked_device_transfer_in_source():
    import inspect

    import maple_syrup.compact_soil_profile as cs

    src = inspect.getsource(cs)
    assert "xp.asarray(" not in src and "cp.asarray(" not in src  # every transfer goes through MAPLE to_device/to_host


GPU = pytest.mark.skipif(os.environ.get("E1_ALLOW_GPU") != "1", reason="GPU not touched by the author; Root sets E1_ALLOW_GPU=1")


@GPU
def test_cuda_dense_equals_host_and_counts_transfers(gpu):
    import cupy as cp
    from maple.core.backend import read_transfer_counters
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    vc, al, g, gc, mres = random_native_bed(4, ny=3, nx=5, n_voxels=30)
    idx = [1, 4, 5, 9]
    ref = collect(compact(vc, al, g, gc, mres, LAYERS, idx, coverage="partial_over_base", batch_cells=2))
    dvc, dal = VoxelColumnState(mass_kg=cp.asarray(vc.mass_kg)), ActiveLayerState(mass_kg=cp.asarray(al.mass_kg))
    before = read_transfer_counters()  # TransferCounters dataclass snapshot (pinned arrays.py); use .delta()
    got = collect(compact(dvc, dal, g, gc, mres, LAYERS, idx, coverage="partial_over_base", backend="cuda", batch_cells=2))
    counters = read_transfer_counters().delta(before)
    for name, axis in (("layer_mass_kg", 1), ("layer_fraction", 1), ("coverage_m", 1), ("canonical_depth_m", 0)):
        assert stack(got, name, axis).tobytes() == stack(ref, name, axis).tobytes()
    assert got[0].namespace == "cupy" and counters.device_to_host > 0 and counters.host_to_device > 0
    assert counters.device_to_host_bytes > 0 and counters.host_to_device_bytes > 0


@GPU
def test_cuda_device_resident_windowed_host_compute_equals_dense(gpu, tmp_path):
    from maple.core.backend import to_device_tree

    vc, al, g, gc, mres = native_deep_bed()
    w = windowed_twin(vc, g, mres, tmp_path / "store")
    idx = [0, 2, 5]
    ref = collect(compact(vc, al, g, gc, mres, LAYERS, idx, coverage="partial_over_base", batch_cells=2))
    got = collect(compact(to_device_tree(w, gpu), to_device_tree(al, gpu), g, gc, mres, LAYERS, idx, coverage="partial_over_base",
                          backend="numpy", batch_cells=2))
    assert got[0].namespace == "cupy" and type(got[0].layer_mass_kg) is gpu.ndarray
    for name, axis in (("layer_mass_kg", 1), ("canonical_depth_m", 0)):
        assert stack(got, name, axis).tobytes() == stack(ref, name, axis).tobytes()


@GPU
def test_cuda_current_device_mismatch_refused_two_devices(gpu):
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two visible devices (Root's allocation is GPU0 only)")
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    vc, al, g, gc, mres = random_native_bed(4, ny=3, nx=5, n_voxels=30)
    with cp.cuda.Device(1):
        dvc, dal = VoxelColumnState(mass_kg=cp.asarray(vc.mass_kg)), ActiveLayerState(mass_kg=cp.asarray(al.mass_kg))
    with pytest.raises(SoilProfileTextureError, match="never switched"):
        compact(dvc, dal, g, gc, mres, LAYERS, [0], backend="cuda", coverage="partial_over_base")
