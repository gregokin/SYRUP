"""E5 correction01: (a) the dimension-aware HOST model of the CUDA vertical constructor (cold mask download, staging, depth record, edges) and the host-pool source scan; (b) the canonical NumPy-order depth contract of the Dense
and Windowed device helpers against the cold native oracle (`compact_soil_profile.canonical_depth_m`). Host execution only: the plumbing arrays are NumPy, so the ORDER discrimination of the real device reduction is Root's
(test_gpu_definitions.py); here the contract is enforced structurally (the ordered native helper is the ONLY active/voxel reduction used) and numerically against the oracle."""
from __future__ import annotations

import importlib.util
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hostd2 import fake

from maple_syrup import compact_soil_profile as csp
from maple_syrup import floating_pond as fp
from maple_syrup import floating_pond_cuda as fpc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[0] / "phase2_floating_pond"))
_spec = importlib.util.spec_from_file_location("e5c_host_helpers", _here / "test_cuda_exchange_host.py")
hx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hx)
AVAIL = lambda: 1 << 40

pytestmark = hx.pytestmark


def _vertical(cp, monkeypatch, **kw):
    bed = hx._b.stratified(shallow=((0, 2),))
    dbed = hx.device_bed(bed)
    owner = hx.gpu_owner(dbed, bed, monkeypatch)
    return owner, bed, dbed


# ---- (a) host model ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_constructor_host_model_is_dimension_aware_and_admitted_before_the_readback(cp, monkeypatch):
    owner, _bed, _d = _vertical(cp, monkeypatch)
    N, L = owner.column.n_cells, owner.column.n_layers
    good = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    assert good.cold_host_bytes == 2 * N + 8 * N + 2 * (L + 1) * 8 + fpc.HOST_OP_BYTES and good.retained_host_bytes == N + (L + 1) * 8
    need = good.cold_host_bytes + fsl.fcu_py_allowance() if hasattr(fsl, "fcu_py_allowance") else good.cold_host_bytes + 65536
    n_alloc, d2h = len(cp.allocs), cp.count.d2h
    with pytest.raises(fsl.FloatingMemoryError, match="host side"):
        fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=lambda: need - 1)
    assert (len(cp.allocs), cp.count.d2h) == (n_alloc, d2h)  # refused before any owned allocation and before the mask readback
    assert fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=lambda: need).cold_host_bytes == good.cold_host_bytes
    # the host reserve is counted ONCE
    with pytest.raises(fsl.FloatingMemoryError, match="host side"):
        fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=lambda: need + 6, host_reserve_bytes=7)
    assert fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=lambda: need + 7, host_reserve_bytes=7)


def test_constructor_traced_host_peak_stays_inside_the_model(cp, monkeypatch):
    owner, _bed, _d = _vertical(cp, monkeypatch)
    import gc

    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    v = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak <= v.cold_host_bytes + 65536, (peak, v.cold_host_bytes)  # tracemalloc sees the NumPy host allocations of the constructor (not the interpreter pool or the device)


def test_the_cold_source_scan_uses_the_host_pool_not_the_device_reader(cp, monkeypatch):
    owner, bed, dbed = _vertical(cp, monkeypatch)
    dev_calls, host_calls = [], []
    v = fp.FloatingPondVertical(owner, available_bytes_fn=lambda: (dev_calls.append(1), 1 << 40)[1], host_available_bytes_fn=lambda: (host_calls.append(1), 1 << 40)[1])
    v.bind_shape(tuple(bed.active_layer.mass_kg.shape[:2]))
    dev_calls.clear()
    host_calls.clear()
    v.check_initial(hx.bed_state(dbed), hx.context(dbed), 0.0)
    assert host_calls and not dev_calls  # the witness scan is admitted from the HOST reader only (the preflight allocates nothing on the device)
    v2 = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    v2.bind_shape(tuple(bed.active_layer.mass_kg.shape[:2]))
    v2.quota = dict(v2.quota, host_available_bytes_fn=lambda: 0)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        v2.check_initial(hx.bed_state(dbed), hx.context(dbed), 0.0)


# ---- (b) canonical NumPy-order contract ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def _geometry(nx=7, ny=1):
    fx_path = _here.parents[0] / "phase2_compact_soil_profile" / "texture_fixtures.py"
    spec = importlib.util.spec_from_file_location("e5c_texture_fixtures", fx_path)
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    return fx.geometry(ny, nx)


def _oracle(vox, act, g, idx):
    """The cold native canonical depth of the flat cells `idx` (host rows, native NumPy order)."""
    K = act.shape[-1]
    V = vox.shape[2]
    v = np.ascontiguousarray(vox.reshape(-1, V, K)[idx])
    a = np.ascontiguousarray(act.reshape(-1, K)[idx])
    return csp.canonical_depth_m(v, a, g)


@pytest.mark.parametrize("K", [1, 2, 3, 6, 7, 8, 9, 17])
def test_dense_device_depth_equals_the_cold_oracle_bitwise_for_heterograin_columns(K, monkeypatch):
    rng = np.random.default_rng(K)
    g = _geometry(7, 1)
    V = 12
    vox = rng.uniform(1e-6, 31.0, (1, 7, V, K)) * rng.choice([1.0, 1e-9, 1e6], size=(1, 7, V, K))  # wide dynamic range: summation-order sensitive
    act = rng.uniform(1e-9, 0.6, (1, 7, K)) * rng.choice([1.0, 1e-7, 1e4], size=(1, 7, K))
    bed = SimpleNamespace(voxel_column=SimpleNamespace(mass_kg=vox), active_layer=SimpleNamespace(mass_kg=act))
    v = SimpleNamespace(cp=np, batch_cells=3)
    idx = np.array([0, 2, 3, 6])
    calls = []
    import maple.core.backend.cell_reductions as cr

    real = cr.sum_per_cell_numpy_order
    monkeypatch.setattr(cr, "sum_per_cell_numpy_order", lambda a, *, trailing_axes: (calls.append((a.shape, trailing_axes)), real(a, trailing_axes=trailing_axes))[1])
    import maple.surface.active_layer.capacity as cap

    want = _oracle(vox, act, g, idx)  # the cold oracle (it legitimately uses the host native helper) BEFORE the device helper is guarded
    monkeypatch.setattr(cap, "compute_active_layer_actual_thickness_m", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the native raw active sum must not be used")))
    got = fpc.dense_canonical_depth(v, bed, SimpleNamespace(geometry=g), idx)
    assert np.asarray(got).tobytes() == want.tobytes()  # bit for bit, no epsilon
    assert {t for _s, t in calls} == {1, 2} and sum(1 for _s, t in calls if t == 1) == sum(1 for _s, t in calls if t == 2) == 2  # voxel (trailing 2) AND active (trailing 1), two batches of <= 3 rows
    assert all(s[1] == 1 for s, _t in calls)  # bounded contiguous (B, 1, ...) rows


def test_windowed_device_depth_equals_the_dense_oracle_bitwise_and_never_uses_the_raw_active_sum(tmp_path, monkeypatch):
    import fpond_builders as fb

    wh = fb._windowed_helpers()
    z = fb.ph.plane(base=0.12, drop=0.004)
    dense, ctx = wh.dense_deep_bed(z, depth=0.003)
    wbed, wctx = wh.windowed_counterpart(dense, ctx, tmp_path / "store")
    g = ctx.geometry
    ny, nx = dense.active_layer.mass_kg.shape[:2]
    idx = np.arange(ny * nx)
    import maple.surface.active_layer.capacity as cap

    want = _oracle(np.asarray(dense.voxel_column.mass_kg), np.asarray(dense.active_layer.mass_kg), g, idx)
    monkeypatch.setattr(cap, "compute_active_layer_actual_thickness_m", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the native raw active sum must not be used")))
    v = SimpleNamespace(cp=np, batch_cells=4)
    got = fpc.windowed_canonical_depth(v, wbed, SimpleNamespace(geometry=wctx.geometry), idx)
    assert np.asarray(got).tobytes() == want.tobytes()


def test_strict_d_equals_h_and_nextafter_geometry_selection_matches_the_owner_rule(cp):
    cfg = hx._b.config()
    e = np.asarray(cfg.nominal_edges_m)
    H = float(e[-1])
    d_old = np.array([H, H, np.nextafter(H, 1.0), H - 1e-12, 0.0, e[1], np.nextafter(e[1], 0.0)])
    d_new = np.array([np.nextafter(H, 1.0), H, H, H, 1e-15, np.nextafter(e[1], 1.0), e[1]])
    v = SimpleNamespace(cp=np, _edges_host=e, config=cfg)
    got = fpc.geometry_changed_device(v, d_old, d_new)
    assert np.array_equal(got, nfs._geometry_changed(cfg, d_old, d_new))  # the same floating-point formula, no epsilon, strict D > H bottom flag
    assert not got[1] and got[0] and got[2]  # D == H -> D == H unchanged; H -> nextafter(H) flips the open flag
    assert fake is not None
