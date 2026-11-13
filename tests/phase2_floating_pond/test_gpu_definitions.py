"""E5 REAL GPU definitions (not placeholders, not xfail): the actual CUDA continuous floating Pond on native DENSE and native WINDOWED device beds, compared with the independent CPU Pond on the same terrain, forcing and
parameters. Run only on Root's hardware with SYRUP_RELEASE_GPU=1 and a free device. Author did not run them (no CuPy/GPU here).

Comparison contracts: floating water / history / canonical depth within the existing 2e-12 / 1e-14; hydraulic and sediment fields within rtol 1e-9 / atol 1e-12 (device reduction order differs from NumPy); budgets through the event's
own audits (water and sediment `closes`)."""
from __future__ import annotations

import gc
import os
from types import SimpleNamespace

import numpy as np
import pytest
from fpond_builders import AVAIL, NC, config, make_pair, native_bed, ph

pytestmark = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1", reason="GPU definitions: set SYRUP_RELEASE_GPU=1 on the hardware qualification host (free device)")

RAIN = 100.0 / 3.6e6


def _cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("no CUDA device")
    return backend.cupy_module()


def device_event(tmp_path, *, windowed, surface=None, depth=0.003, mobile=None, closed=False, policy=None, bottom="open", saturation=0.5, rain_mm_h=100.0, end_s=100.0, splash=True, max_dt_s=0.2):
    """`(event, state, vertical, owner)` of the CUDA floating Pond on a device Dense/Windowed bed built from the qualified native pond fixtures (initial availability on the device)."""
    from dataclasses import replace

    import fpond_builders as fb
    from maple.core.backend import to_device, to_device_tree

    from maple_syrup import floating_pond as fp
    from maple_syrup import floating_soil_layers_cuda as fcu
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup.pond_event import PondEvent
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    cp = _cp()
    wh = fb._windowed_helpers()
    z = ph.plane(base=0.12, drop=0.004) if surface is None else surface
    dense, ctx = wh.dense_deep_bed(z, depth=depth, mobile=mobile)
    topo = ph.make_topology(dense, [], allow_closed=True) if closed else ph.make_topology(dense, ph.east_outlets(*z.shape), drop=0.02)
    if windowed:
        bed, dctx = wh.windowed_counterpart(dense, ctx, tmp_path / "store", xp=cp)
    else:
        bed = to_device_tree(dense, cp)
        dctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, cp))
    cfg = config(backend="cuda", bottom=bottom)
    owner = nfs.initialize_native_floating_soil(native_bed(bed, dctx), cfg, initial_relative_saturation=saturation)  # real default readers
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=fcu.device_reader(cp))
    be = SimpleNamespace(hydraulics="cuda", xp=cp)
    ctl = wh.control_for(be, windowed=windowed, max_dt_s=max_dt_s)
    sp = splash_parameters(np.linspace(0.4, 3.0, NC), xp=cp) if splash else None
    ev = PondEvent(dctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, end_s, rain_mm_h), topo, control=ctl, policy=policy, splash=sp, floating_vertical=vert)
    return ev, ev.initial_state(bed, vert.initial_soil_water()), vert, owner


def run(ev, st, n, rate=RAIN, dt=0.2):
    for _ in range(n):
        st = ev.step(st, rate, dt).state
    return st


def host(a):
    from maple.core.backend import to_host

    return np.asarray(to_host(a))


@pytest.mark.parametrize("windowed", [False, True])
def test_device_pond_matches_the_cpu_pond_through_rain_matrix_gravity_wet_and_splash_exchange(tmp_path, windowed):
    cp = _cp()
    (cev, cst, cvert, *_a), _ = make_pair(tmp_path / "cpu", splash=True)
    gev, gst, gvert, _o = device_event(tmp_path / "gpu", windowed=windowed, splash=True)
    cst, gst = run(cev, cst, 12), run(gev, gst, 12)
    assert gvert.stats.depth_changed_columns == cvert.stats.depth_changed_columns and gvert.stats.touched_columns == cvert.stats.touched_columns
    np.testing.assert_allclose(host(gst.hydraulic.depth_m), np.asarray(cst.hydraulic.depth_m), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(host(gst.hydraulic.soil_water_m), np.asarray(cst.hydraulic.soil_water_m), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(gvert.current_water(gst.hydraulic.soil_water_m), cvert.current_water(cst.hydraulic.soil_water_m), rtol=2e-12, atol=1e-14)
    pc, pg = cvert.current_payload(cst.hydraulic.soil_water_m), gvert.current_payload(gst.hydraulic.soil_water_m)
    assert host(pg.depth_m).tobytes() == pc.depth_m.tobytes()  # the device canonical depth (Dense: native NumPy-order reduction; Windowed: native prefix) is bitwise the cold native depth
    np.testing.assert_allclose(host(pg.history.wetting_m), pc.history.wetting_m, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(host(gst.bed.water.mobile_mass_by_cell_class_kg), np.asarray(cst.bed.water.mobile_mass_by_cell_class_kg), rtol=1e-9, atol=1e-12)
    a = gev.audit(gst)
    assert a["water"]["closes"] and a["sediment"]["closes"]
    assert cp.get_default_memory_pool() is not None


@pytest.mark.parametrize("windowed", [False, True])
def test_device_terminal_thin_lake_reconciles_before_quiet_and_recession_recertifies(tmp_path, windowed):
    from maple_syrup.pond_event import PondCompletionPolicy

    z = ph.basin() - 0.08  # thin closed profile
    depth = np.maximum(0.135 - z, 0.0)
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    kw = {"surface": z, "depth": depth, "mobile": mob, "closed": True, "bottom": "closed", "saturation": 0.999999, "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0,
          "policy": PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)}
    ev, st, _vert, _o = device_event(tmp_path, windowed=windowed, **kw)
    q = ev.evolve(st, 60.0, stop_when_quiet=True)
    assert q.quiet_ready
    h = ev.settle_retaining_water(q.state)
    assert h.status == "requires_recession" and h.reason == "floating_terminal_capacity_reconciled/v1"
    assert h.accounting["final_audit"]["water"]["closes"] and h.accounting["final_audit"]["sediment"]["closes"]
    assert np.array_equal(host(h.state.bed.water.depth_m), host(h.state.hydraulic.depth_m))  # ONE surface depth
    res = ev.evolve(h.state, 120.0, stop_when_quiet=True)
    assert res.quiet_ready and ev.settle_retaining_water(res.state).complete


def test_device_shallow_real_erosion_matches_the_cpu_ordinary_reconciliation(tmp_path):
    z = ph.plane(base=0.12, drop=0.004)
    (cev, cst, cvert, *_a), _ = make_pair(tmp_path / "cpu", surface=z, bottom="closed", saturation=0.97, splash=True, depth=0.02)
    gev, gst, gvert, _o = device_event(tmp_path / "gpu", windowed=False, surface=z, bottom="closed", saturation=0.97, splash=True, depth=0.02, max_dt_s=0.001)
    for _ in range(200):
        cst = cev.step(cst, 300 / 3.6e6, 0.001).state
        gst = gev.step(gst, 300 / 3.6e6, 0.001).state
    assert gvert.stats.reconciled_columns == cvert.stats.reconciled_columns
    np.testing.assert_allclose(gvert.current_water(gst.hydraulic.soil_water_m), cvert.current_water(cst.hydraulic.soil_water_m), rtol=2e-12, atol=1e-14)
    assert gev.audit(gst)["water"]["closes"]


def test_device_initial_guards_and_budget_refusals(tmp_path):
    from maple_syrup import floating_pond as fp

    cp = _cp()
    ev, st, vert, owner = device_event(tmp_path, windowed=False)
    bed = st.bed
    with pytest.raises(fp.FloatingPondError, match="clock"):
        ev.initial_state(bed, vert.initial_soil_water(), t_s=7.0)
    from dataclasses import replace

    from maple.core.types.voxel import VoxelColumnState

    v = cp.array(bed.voxel_column.mass_kg, copy=True)
    v[..., [0, 1]] = v[..., [1, 0]]  # a bed with swapped grain classes is not the owner's native source
    with pytest.raises(Exception, match="witness|source"):
        ev.initial_state(replace(bed, voxel_column=VoxelColumnState(mass_kg=v)), vert.initial_soil_water())
    with pytest.raises(Exception):  # noqa: B017 - typed memory refusal before any owned allocation
        fp.FloatingPondVertical(owner, available_bytes_fn=lambda: 0)
    with pytest.raises(fp.FloatingPondError, match="cuda"):
        from maple_syrup import native_floating_soil as nfs

        cpu_owner = nfs.initialize_native_floating_soil(native_bed(ph.make_bed(ph.plane())[0], ph.make_bed(ph.plane())[1]), config(), initial_relative_saturation=0.5, available_bytes_fn=AVAIL)
        fp.FloatingPondVertical(cpu_owner, available_bytes_fn=AVAIL, host_reserve_bytes=1)  # a meaningless host argument on a CPU owner


def test_device_warm_pool_plateau_over_many_steps(tmp_path):
    cp = _cp()
    ev, st, vert, _o = device_event(tmp_path, windowed=False, end_s=1000.0)
    st = run(ev, st, 10)
    gc.collect()
    pool = cp.get_default_memory_pool()
    used0 = pool.used_bytes()
    st = run(ev, st, 60)
    gc.collect()
    assert pool.used_bytes() - used0 <= 64 * vert.n_cells * 8  # a plateau (only the live state), no per-step growth
    assert len(vert._registry) <= 6


# ---- E5 correction01 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("K", [1, 2, 3, 6, 7, 8, 9, 17])
def test_device_dense_canonical_depth_equals_the_cold_host_oracle_bitwise_for_heterograin_columns(K):
    """REAL device arrays, wide dynamic-range heterograin masses at varied K: the ordered native reduction (voxel trailing 2, active trailing 1) must equal the cold host oracle bit for bit (a raw `cupy.sum` would not
    in general)."""
    import importlib.util
    from pathlib import Path

    from maple.core.backend import to_device

    from maple_syrup import compact_soil_profile as csp
    from maple_syrup import floating_pond_cuda as fpc

    cp = _cp()
    spec = importlib.util.spec_from_file_location("e5c_tf", Path(__file__).resolve().parents[1] / "phase2_compact_soil_profile" / "texture_fixtures.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    g = fx.geometry(1, 257)
    rng = np.random.default_rng(K)
    V = 12
    vox = rng.uniform(1e-6, 31.0, (1, 257, V, K)) * rng.choice([1.0, 1e-9, 1e6], size=(1, 257, V, K))
    act = rng.uniform(1e-9, 0.6, (1, 257, K)) * rng.choice([1.0, 1e-7, 1e4], size=(1, 257, K))
    idx = np.arange(257)
    want = csp.canonical_depth_m(np.ascontiguousarray(vox.reshape(-1, V, K)), np.ascontiguousarray(act.reshape(-1, K)), g)
    bed = SimpleNamespace(voxel_column=SimpleNamespace(mass_kg=to_device(vox, cp)), active_layer=SimpleNamespace(mass_kg=to_device(act, cp)))
    v = SimpleNamespace(cp=cp, batch_cells=64)
    got = host(fpc.dense_canonical_depth(v, bed, SimpleNamespace(geometry=g), to_device(idx, cp)))
    assert got.tobytes() == want.tobytes()


def test_device_windowed_canonical_depth_equals_the_dense_oracle_bitwise(tmp_path):
    """The Windowed device bed is built the native way (canonical host dense baseline -> `windowed_counterpart(xp=cupy)` -> device packets), then the hot depth equals the host Dense canonical depth bit for bit."""
    import fpond_builders as fb

    from maple_syrup import compact_soil_profile as csp
    from maple_syrup import floating_pond_cuda as fpc

    cp = _cp()
    wh = fb._windowed_helpers()
    z = ph.plane(base=0.12, drop=0.004)
    dense, ctx = wh.dense_deep_bed(z, depth=0.003)
    wbed, wctx = wh.windowed_counterpart(dense, ctx, tmp_path / "store", xp=cp)
    ny, nx = dense.active_layer.mass_kg.shape[:2]
    V, K = dense.voxel_column.mass_kg.shape[2:]
    idx = np.arange(ny * nx)
    want = csp.canonical_depth_m(np.ascontiguousarray(np.asarray(dense.voxel_column.mass_kg).reshape(-1, V, K)), np.ascontiguousarray(np.asarray(dense.active_layer.mass_kg).reshape(-1, K)), ctx.geometry)
    v = SimpleNamespace(cp=cp, batch_cells=4)
    got = host(fpc.windowed_canonical_depth(v, wbed, SimpleNamespace(geometry=wctx.geometry), cp.asarray(idx)))
    assert got.tobytes() == want.tobytes()


def test_device_constructor_host_peak_at_n100003_stays_inside_the_declared_cold_host_model():
    import tracemalloc
    from maple_syrup import floating_pond as fp
    from maple_syrup import floating_soil_layers_cuda as fcu
    from maple_syrup import native_floating_soil as nfs

    cp = _cp()
    # Use the native scalable flat fixture, not make_bed's internally tilted
    # default surface (which exceeds its voxel allocation on this domain).
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "e5_large_native_builders",
        Path(__file__).resolve().parents[1] / "phase2_native_floating_soil_cuda" / "test_gpu_definitions.py",
    )
    gb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gb)
    native = gb.b.bed_from(gb.b.fx.stratified_dense(shallow_cells=(), ny=1, nx=100003))
    dbed = gb.device_bed(cp, native)
    owner = nfs.initialize_native_floating_soil(dbed, gb.b.config(backend="cuda"), initial_relative_saturation=0.5)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    v = fp.FloatingPondVertical(owner, available_bytes_fn=fcu.device_reader(cp))
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak <= v.cold_host_bytes + fcu.PY_ALLOWANCE_BYTES, (peak, v.cold_host_bytes)
    assert v.cold_host_bytes > 100003 * 10  # the model is dimension-aware (the Root defect admitted a constant 69632 B)
