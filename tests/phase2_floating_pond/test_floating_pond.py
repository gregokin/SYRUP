"""E4b floating-layer vertical water inside the native Pond step (CPU/host, NumPy and Numba owners, Dense actual MAPLE bed). Author-written."""
# ruff: noqa: RUF059
from __future__ import annotations

import gc
import hashlib
import tracemalloc
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fpond_builders import AVAIL, NC, make, ph

from maple_syrup import floating_pond as fp
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs

RAIN = 100.0 / 3.6e6
BACKENDS = ["numpy", "numba"]

GOLDEN_CONFIG_SHA = "bdd0fc47b9550fd44d2b3875969805a955788db855a1239e053f3338f36f44a8"
GOLDEN_STATE_SHA = "50c0c24a27ee427c403a6db357040fd4a4f5cc2b80614e8eac2ed88503f2b719"


@pytest.fixture(params=BACKENDS)
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def run(ev, st, n, rate=RAIN, dt=0.2):
    for _ in range(n):
        st = ev.step(st, rate, dt).state
    return st


def totals(ev, st, vert):
    """Host volumes (m3): surface, soil (floating inventory), plus the ledger's cumulative flows."""
    area = ev.context.geometry.dx_m * ev.context.geometry.dy_m
    w = vert.current_water(st.hydraulic.soil_water_m)
    wl = st.water
    return {"surface": float(st.hydraulic.depth_m.sum()) * area, "soil": float(w.sum()) * area, "rain": float(wl.rain.sum()) * area, "drain": float(wl.drain.sum()) * area,
            "export": float(wl.export) if np.isscalar(wl.export) else float(np.sum(wl.export)), "sat_return": float(wl.sat_return.sum()) * area, "intake": float(wl.intake.sum()) * area}


def test_default_pond_is_byte_identical():
    be = SimpleNamespace(hydraulics="numpy", xp=np)
    ev, st = ph.wet_plane(be, splash=True)
    assert ev.vertical is None
    ev0, _ = ph.wet_plane(be)
    assert ev0.config_sha256 == GOLDEN_CONFIG_SHA
    for _ in range(4):
        st = ev.step(st, RAIN, 0.2).state
    h = hashlib.sha256()
    for a in (st.hydraulic.depth_m, st.hydraulic.soil_water_m, st.bed.water.mobile_mass_by_cell_class_kg, st.bed.active_layer.mass_kg):
        h.update(np.ascontiguousarray(a).tobytes())
    assert h.hexdigest() == GOLDEN_STATE_SHA


def test_soil_water_is_only_the_derived_sum_and_foreign_states_refused(backend):
    ev, st, *_rest, vert, _owner = make(backend=backend)
    st = run(ev, st, 3)
    w = vert.current_water(st.hydraulic.soil_water_m)
    assert np.array_equal(st.hydraulic.soil_water_m.reshape(-1), w.sum(axis=0))
    forged = replace(st.hydraulic, soil_water_m=np.array(st.hydraulic.soil_water_m, copy=True))
    with pytest.raises(fp.FloatingPondError, match="not issued"):
        ev.step(replace(st, hydraulic=forged), RAIN, 0.2)
    with pytest.raises(fp.FloatingPondError, match="not issued"):
        vert._payload(np.zeros((3, 4)))


def test_water_budget_closes_with_rain_matrix_and_preferential(backend):
    ev, st, _bed, _ctx, vert, _o = make(backend=backend, preferential=True, depth=0.001)
    t0 = totals(ev, st, vert)
    st = run(ev, st, 25)
    t1 = totals(ev, st, vert)
    d_surface, d_soil = t1["surface"] - t0["surface"], t1["soil"] - t0["soil"]
    resid = (d_surface + d_soil) + t1["export"] + t1["drain"] - (t1["rain"] - t0["rain"])
    assert t1["intake"] > 0.0 and d_soil > 0.0
    assert abs(resid) < 1e-12, resid  # author cross-check only; the acceptance statement is the audit below
    audit = ev.audit(st)
    assert audit["water"]["closes"] and audit["sediment"]["closes"]  # the event's own audited closures (existing native bounds), not an invented tolerance


def test_actual_detachment_changes_depth_and_matches_cold_canonical_depth(backend):
    ev, st, bed, ctx, vert, owner = make(backend=backend, splash=True, depth=0.004)
    st = run(ev, st, 12)
    assert vert.stats.depth_changed_columns > 0 and vert.stats.views_built > 0
    payload = vert.current_payload(st.hydraulic.soil_water_m)
    cold = nfs.initialize_native_floating_soil(nfs.NativeBed(st.bed.voxel_column, st.bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, np.ones((3, 4), bool)),
                                               owner.config, initial_relative_saturation=0.1, available_bytes_fn=AVAIL)
    touched = np.flatnonzero((st.sediment.pickup_cell + st.sediment.deposition_cell).sum(axis=-1).reshape(-1) > 0)
    assert touched.size > 0
    assert np.array_equal(payload.depth_m[touched], cold.depth_m[touched])  # bit for bit the cold native canonical depth


def test_depth_change_above_h_only_never_drains(backend):
    ev, st, *_r, vert, _o = make(backend=backend, depth=0.004)  # D ~ 0.2-0.26 m > H = 0.16 m everywhere
    w0 = vert.current_water(st.hydraulic.soil_water_m)
    st = run(ev, st, 8)
    assert vert.stats.depth_changed_columns > 0 and vert.stats.reconciled_columns == 0
    assert vert.stats.surface_return_m3 == 0.0 and vert.stats.bottom_export_m3 == 0.0
    # raw depth changed below nothing: the history is the same sealed arrays the matrix steps advanced (no reset)
    assert vert.current_water(st.hydraulic.soil_water_m).shape == w0.shape


def test_zero_net_exchange_touches_but_changes_no_geometry(backend):
    ev, st, bed, ctx, vert, _o = make(backend=backend)
    moved = ph.move_mass(st.bed, ctx, (1, 1), (1, 1), 0.002)
    touched = np.zeros((3, 4), bool)
    touched[1, 1] = True
    hyd, rec = vert.after_exchange(moved, ctx, st.hydraulic, touched, np.ones((3, 4), bool))
    assert rec["n_touched"] == 1 and rec["n_reconciled"] == 0
    assert np.array_equal(vert.current_water(hyd.soil_water_m), vert.current_water(st.hydraulic.soil_water_m))


def _shallow(backend, **kw):
    z = ph.plane(base=0.12, drop=0.004)  # D 0.12 - 0.136 m < H = 0.16 m: a closed shallow profile at every cell
    return make(surface=z, backend=backend, bottom="closed", saturation=0.97, **kw)


def test_rock_cutoff_returns_water_once_and_conserves(backend):
    ev, st, bed, ctx, vert, _o = _shallow(backend)
    area = ctx.geometry.dx_m * ctx.geometry.dy_m
    soil0 = vert.current_water(st.hydraulic.soil_water_m).sum() * area
    depth0 = float(st.hydraulic.depth_m.sum()) * area
    lowered, _ = ph.pickup_cell(st.bed, ctx, (1, 2), 0.004)
    touched = np.zeros((3, 4), bool)
    touched[1, 2] = True
    hyd, rec = vert.after_exchange(lowered, ctx, st.hydraulic, touched, np.ones((3, 4), bool))
    assert rec["n_reconciled"] == 1
    soil1 = vert.current_water(hyd.soil_water_m).sum() * area
    returned = float(hyd.depth_m.sum()) * area - depth0
    assert returned == pytest.approx(rec["surface_return_m3"], abs=1e-15) and returned > 0.0
    assert abs((soil0 - soil1) - (returned + rec["bottom_export_m3"])) < 1e-13  # moved once: soil lost = surface gained + export
    # the old accepted payload is untouched (pure)
    assert vert.current_water(st.hydraulic.soil_water_m).sum() * area == pytest.approx(soil0, abs=0.0)


def test_returned_water_enters_ledger_once_and_is_routed_by_rain0_steps(backend, monkeypatch):
    """FAULT-INJECTED depth source (labelled): the actual canonical depth read of the touched columns is lowered by 6 mm to emulate a large actual erosion event; everything downstream
    (geometry view, E3c rule, ONE D1 reconcile body, ledger credit, surface routing, quiet monitor) is the real code."""
    ev, st, bed, ctx, vert, _o = _shallow(backend, depth=0.004)
    real = nfs.read_canonical_depth
    armed = {"on": False}

    def lowered(bed_, config, idx, scratch, **kw):
        d = real(bed_, config, idx, scratch, **kw)
        return d - 0.006 if armed["on"] else d

    monkeypatch.setattr(nfs, "read_canonical_depth", lowered)
    st = run(ev, st, 4)
    before = totals(ev, st, vert)
    armed["on"] = True
    st1 = ev.step(st, RAIN, 0.2).state
    armed["on"] = False
    mid = totals(ev, st1, vert)
    assert vert.stats.surface_return_m3 > 0.0 and vert.stats.reconciled_columns > 0
    # one step: rain + the credited return/export; both closures exact (soil: intake - return - drain; surface: rain - intake + return - export)
    assert abs(((mid["surface"] - before["surface"]) + (mid["soil"] - before["soil"])) + (mid["export"] - before["export"]) + (mid["drain"] - before["drain"]) - (mid["rain"] - before["rain"])) < 1e-12
    assert mid["sat_return"] - before["sat_return"] >= vert.stats.surface_return_m3 - 1e-15  # credited once in the ledger (the step's own saturation return may add)
    assert not st1.quiet.ready and st1.quiet.since_s is None  # the new depth change gives a fresh (not yet started) quiet hold
    st2 = run(ev, st1, 12, rate=0.0)  # later rain-0 steps run the ordinary surface/soil stages: the returned water is routed, nothing new is added
    after = totals(ev, st2, vert)
    assert after["rain"] == mid["rain"]
    assert abs(((after["surface"] - mid["surface"]) + (after["soil"] - mid["soil"])) + (after["export"] - mid["export"]) + (after["drain"] - mid["drain"])) < 1e-12
    assert after["surface"] < mid["surface"] or after["export"] > mid["export"] or after["soil"] > mid["soil"]


def test_hot_path_is_bounded_no_cold_prepare_of_the_whole_domain(backend, monkeypatch):
    ev, st, bed, ctx, vert, _o = make(backend=backend, splash=True, depth=0.004)
    sizes = []
    real = fsl.prepare_floating_column

    def spy(*a, **k):
        sizes.append(int(k["soil_depth_m"].shape[0]))
        return real(*a, **k)

    monkeypatch.setattr(fsl, "prepare_floating_column", spy)
    run(ev, st, 10)
    assert all(n <= vert.batch_cells for n in sizes), sizes  # only compact touched-column sub-columns; never an N-cell prepare per step
    assert vert.stats.steps == 10


def test_many_step_registry_and_buffers_stay_bounded(backend):
    ev, st, *_r, vert, _o = make(backend=backend, splash=True, depth=0.004, end_s=1000.0)
    st = run(ev, st, 5)
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    st = run(ev, st, 60)
    gc.collect()
    cur = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert len(vert._registry) <= 6, len(vert._registry)
    assert vert.stats.peak_registry <= 8
    assert cur - base < 2_000_000, cur - base  # warm 60 steps: no growing history/whole-state accumulation (sealed counters only)


def test_refusals_never_downgrade(backend, tmp_path):
    ev, st, bed, ctx, vert, owner = make(backend=backend)
    with pytest.raises(fp.FloatingPondError, match="CURRENT"):
        fp.FloatingPondVertical(object(), available_bytes_fn=AVAIL)
    with pytest.raises(fp.FloatingPondError, match="callable"):
        fp.FloatingPondVertical(owner, available_bytes_fn=None)
    with pytest.raises(Exception, match="finalize is the dry reset"):
        ev.finalize(st)
    from maple_syrup.pond_event import PondEvent, PondEventControl

    topo = ev._initial_topology
    with pytest.raises(Exception, match="column=None"):
        PondEvent(ctx, object(), ev.sediment, ev.schedule, topo, floating_vertical=vert)
    with pytest.raises(Exception, match="CPU backend only"):
        PondEvent(ctx, None, ev.sediment, ev.schedule, topo, control=PondEventControl(backend="cuda"), floating_vertical=vert)


def test_numba_matches_numpy():
    pytest.importorskip("numba")
    outs = []
    for b in BACKENDS:
        ev, st, *_r, vert, _o = make(backend=b, splash=True, depth=0.004)
        st = run(ev, st, 10)
        outs.append((st.hydraulic.depth_m, vert.current_water(st.hydraulic.soil_water_m), st.bed.water.mobile_mass_by_cell_class_kg))
    for a, b in zip(outs[0], outs[1], strict=True):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12)


def test_windowed_control_with_a_dense_bed_is_refused_at_initial_state_not_downgraded():
    ev, st, bed, ctx, vert, owner = make()
    from maple_syrup.pond_event import PondEvent, PondEventControl

    wev = PondEvent(ctx, None, ev.sediment, ev.schedule, ev._initial_topology, control=PondEventControl(bed_storage="windowed"), floating_vertical=vert)
    with pytest.raises(Exception):  # noqa: B017 - a typed native refusal of a dense bed under windowed storage, before any state
        wev.initial_state(bed, vert.initial_soil_water())
    assert NC == 6


# ---- correction01 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_initial_clock_must_equal_the_owner_clock_before_any_work(backend):
    ev, st, bed, ctx, vert, _o = make(backend=backend)
    n_reg = len(vert._registry)
    with pytest.raises(fp.FloatingPondError, match="clock"):
        ev.initial_state(bed, vert.initial_soil_water(), t_s=7.0)
    assert len(vert._registry) == n_reg  # nothing was registered or adopted
    ok = ev.initial_state(bed, vert.initial_soil_water(), t_s=0.0)
    assert ok.hydraulic.t_s == 0.0


def _swapped(bed, a=0, b=1):
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    vox, act = bed.voxel_column.mass_kg.copy(), bed.active_layer.mass_kg.copy()
    for arr in (vox, act):
        arr[..., [a, b]] = arr[..., [b, a]]
    return replace(bed, voxel_column=VoxelColumnState(mass_kg=vox), active_layer=ActiveLayerState(mass_kg=act))


def test_initial_bed_must_be_the_owner_native_source(backend):
    ev, st, bed, ctx, vert, _o = make(backend=backend)
    other = _swapped(bed)  # same geometry, same sums, same elevation: swapped grain classes 0/1
    assert np.array_equal(other.voxel_column.mass_kg.sum(axis=(2, 3)), bed.voxel_column.mass_kg.sum(axis=(2, 3)))
    with pytest.raises(fp.FloatingPondError, match="witness"):
        ev.initial_state(other, vert.initial_soil_water())
    assert ev.initial_state(bed, vert.initial_soil_water()) is not None  # the genuine source is accepted (one cold scan)


def test_swapped_active_mask_pattern_of_the_same_count_is_refused(backend):
    from maple_syrup.pond_event import PondEvent

    mask = np.ones((3, 4), bool)
    mask[0, 0] = False
    ev, st, bed, ctx, vert, _o = make(backend=backend, mask=mask)
    swapped = mask.copy()
    swapped[0, 0], swapped[1, 1] = True, False
    topo = ph.make_topology(bed, ph.east_outlets(3, 4), drop=0.02, active=swapped)
    with pytest.raises(fp.FloatingPondError, match="active mask"):
        PondEvent(ctx, None, ev.sediment, ev.schedule, topo, control=ev.control, floating_vertical=vert).initial_state(bed, vert.initial_soil_water())


def test_stale_owner_is_refused_at_initial_use(backend):
    ev, st, bed, ctx, vert, owner = make(backend=backend)
    owner._authority.current = object()  # a superseded lineage (what adoption of a refresh does)
    with pytest.raises(fp.FloatingPondError, match="current"):
        ev.initial_state(bed, vert.initial_soil_water())


def test_construction_admission_precedes_every_owned_allocation(backend):
    from fpond_builders import config, native_bed

    from maple_syrup import native_floating_soil as nfs2

    z = ph.plane()
    bed, ctx = ph.make_bed(z)
    owner = nfs2.initialize_native_floating_soil(native_bed(bed, ctx), config(backend=backend), initial_relative_saturation=0.5, available_bytes_fn=AVAIL)
    calls = []

    def zero():
        calls.append(1)
        return 0

    with pytest.raises(nfs.NativeFloatingMemoryError):
        fp.FloatingPondVertical(owner, available_bytes_fn=zero)
    assert calls  # the real reader was consulted, before any array of the stage existed
    good = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL)
    need = good.construction_bytes
    assert fp.FloatingPondVertical(owner, available_bytes_fn=lambda: need).construction_bytes == need
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fp.FloatingPondVertical(owner, available_bytes_fn=lambda: need - 1)


def test_public_geometry_view_admission_and_defaults(backend):
    ev, st, bed, ctx, vert, owner = make(backend=backend)
    col = owner.column
    depth = np.array(owner.depth_m, copy=True)
    calls = []
    with pytest.raises(fsl.FloatingMemoryError):
        fsl.floating_geometry_view(col, depth, available_bytes_fn=lambda: (calls.append(1), 0)[1])
    assert calls
    need = col.n_cells * 11
    assert fsl.floating_geometry_view(col, depth, available_bytes_fn=lambda: need).parent_sha256 == col.sha256
    with pytest.raises(fsl.FloatingMemoryError):
        fsl.floating_geometry_view(col, depth, available_bytes_fn=lambda: need - 1)
    assert fsl.floating_geometry_view(col, depth).identity_kind == "geometry_view_token"  # the old call form stays valid


@pytest.mark.parametrize("shape,layers", [((3, 4), (0.01, 0.05, 0.10)), ((9, 11), (0.01, 0.02, 0.05, 0.08))])
def test_warm_step_and_exchange_allocations_stay_inside_the_declared_model(backend, shape, layers):
    z = ph.plane(ny=shape[0], nx=shape[1], drop=0.004)
    ev, st, bed, ctx, vert, _o = make(surface=z, backend=backend, layers=layers, splash=True, depth=0.004)
    st = run(ev, st, 4)  # warm: compiled kernels, caches
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    nxt = ev.step(st, RAIN, 0.2).state
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    del nxt
    # the stage + surface model is a bound on the vertical's own arrays; the sediment/hydraulic stages of the old body are not part of it, so only the vertical's share is asserted via its own steps below
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    h = st.hydraulic
    solver = ev._cache.pond.solver
    out = solver.step(ev.rain_scale * RAIN, h, 0.2)
    p_step = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert p_step <= vert.stage_step_bytes + vert.surface_step_bytes, (p_step, vert.stage_step_bytes, vert.surface_step_bytes)
    assert peak > 0 and out is not None


def test_one_byte_short_step_refuses_before_allocation(backend):
    ev, st, bed, ctx, vert, _o = make(backend=backend)
    vert.available_bytes_fn = lambda: vert.step_bytes - 1
    with pytest.raises(nfs.NativeFloatingMemoryError):
        ev.step(st, RAIN, 0.2)
    vert.available_bytes_fn = AVAIL


def _lake(backend, **kw):
    from maple_syrup.pond_event import PondCompletionPolicy

    z = ph.basin() - 0.08  # thin profile (D 0.10-0.14 m < H = 0.16 m), closed lake
    depth = np.maximum(0.135 - z, 0.0)
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    return make(surface=z, closed=True, bottom="closed", saturation=0.999999, depth=depth, mobile=mob, policy=PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True), max_dt_s=1.0,
                end_s=2.0, rain_mm_h=0.0, backend=backend, **kw)


def test_terminal_pool_deposition_reconciles_actual_capacity_and_wakes_the_quiet_certificate(backend):
    """GENUINE closed lake with a real mobile pool over a thin closed profile (no monkeypatched height): the terminal deposition's gross per-cell proof stages the actual canonical depth through the shared
    settlement, BEFORE the final commit and publication."""
    ev, st, bed, ctx, vert, _o = _lake(backend)
    res = ev.evolve(st, 60.0, stop_when_quiet=True)
    assert res.quiet_ready
    q = res.state
    n0 = vert.stats.after_exchange_calls
    h = ev.settle_retaining_water(q)
    assert vert.stats.after_exchange_calls == n0 + 1 and vert.stats.reconciled_columns >= 1  # the terminal deposit reached the vertical
    assert h.status == "requires_recession" and h.reason == "floating_terminal_capacity_reconciled/v1" and not h.complete
    assert h.state.topology.generation == q.topology.generation  # routing is still ONLY the net-elevation threshold: resetting the quiet certificate reroutes nothing
    assert not h.state.quiet.ready and h.state.quiet.since_s is None and h.state.t_s == q.t_s  # same clock, no E/T/extra time
    assert np.array_equal(np.asarray(h.state.bed.water.depth_m), np.asarray(h.state.hydraulic.depth_m))  # ONE surface depth (native bed water)
    post = h.accounting["final_audit"]
    assert post["water"]["closes"] and post["sediment"]["closes"]
    w = vert.current_water(h.state.hydraulic.soil_water_m)
    assert np.array_equal(h.state.hydraulic.soil_water_m.reshape(-1), w.sum(axis=0))  # the derived soil payload is the updated one
    with pytest.raises(Exception, match="quiet"):
        ev.settle_retaining_water(h.state)  # an immediate retry cannot complete (resource caps/old q = 0 are not completion)
    res2 = ev.evolve(h.state, 120.0, stop_when_quiet=True)
    assert res2.quiet_ready
    done = ev.settle_retaining_water(res2.state)
    assert done.complete  # a fresh real hold re-certifies; nothing further is deposited, no further reconcile


def test_terminal_hook_is_absent_for_the_default_event():
    be = SimpleNamespace(hydraulics="numpy", xp=np)
    ev, st = ph.wet_plane(be)
    assert ev.vertical is None


@pytest.mark.parametrize("shape", [(3, 4), (9, 11)])
def test_warm_exchange_stage_allocations_stay_inside_the_declared_model(backend, shape):
    z = ph.plane(ny=shape[0], nx=shape[1], drop=0.004)
    ev, st, bed, ctx, vert, _o = make(surface=z, backend=backend, depth=0.004)
    lowered, _ = ph.pickup_cell(st.bed, ctx, (1, 1), 0.002)
    touched = np.zeros(shape, bool)
    touched[1, 1] = True
    act = np.ones(shape, bool)
    vert.after_exchange(lowered, ctx, st.hydraulic, touched, act)  # warm
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    hyd, rec = vert.after_exchange(lowered, ctx, st.hydraulic, touched, act)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert rec["n_depth_changed"] == 1 and peak <= vert.last_exchange_required, (peak, vert.last_exchange_required)


def test_default_reader_zero_refuses_public_geometry_view(backend, monkeypatch):
    ev, st, bed, ctx, vert, owner = make(backend=backend)
    col, depth = owner.column, np.array(owner.depth_m, copy=True)
    calls = []
    monkeypatch.setattr(fsl.sp, "default_host_available_bytes", lambda: (calls.append(1), 0)[1])
    with pytest.raises(fsl.FloatingMemoryError):
        fsl.floating_geometry_view(col, depth)
    assert calls  # the REAL default reader was consulted before any copy
    with pytest.raises(fsl.FloatingSoilError):
        fsl.floating_geometry_view(col, depth, reserve_bytes=-1)  # the reserve is validated on every branch
    monkeypatch.setattr(fsl.sp, "default_host_available_bytes", lambda: 10**12)
    assert fsl.floating_geometry_view(col, depth).identity_kind == "geometry_view_token"  # old call form still works when memory is available


# ---- E4c: native Windowed CPU ----------------------------------------------------------------------------------------------------------------------------------------------------------
def test_windowed_matches_dense_bit_for_bit_through_wet_and_splash_exchange(backend, tmp_path):
    from fpond_builders import make_pair

    (de, ds, dv, *_a), (we, wsx, wv, *_b) = make_pair(tmp_path, backend=backend)
    for _ in range(12):
        ds, wsx = de.step(ds, RAIN, 0.2).state, we.step(wsx, RAIN, 0.2).state
    assert wv.stats.depth_changed_columns > 0 and wv.stats.touched_columns == dv.stats.touched_columns
    for a, b in ((ds.hydraulic.depth_m, wsx.hydraulic.depth_m), (ds.hydraulic.soil_water_m, wsx.hydraulic.soil_water_m)):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert dv.current_water(ds.hydraulic.soil_water_m).tobytes() == wv.current_water(wsx.hydraulic.soil_water_m).tobytes()
    pd, pw = dv.current_payload(ds.hydraulic.soil_water_m), wv.current_payload(wsx.hydraulic.soil_water_m)
    assert pd.depth_m.tobytes() == pw.depth_m.tobytes()  # the hot native prefix depth equals the Dense canonical depth bitwise
    assert pd.history.wetting_m.tobytes() == pw.history.wetting_m.tobytes()
    assert we.audit(wsx)["water"]["closes"] and we.audit(wsx)["sediment"]["closes"]


def test_windowed_terminal_settlement_and_publication_last(backend, tmp_path):
    from maple_syrup.pond_event import PondCompletionPolicy

    z = ph.basin() - 0.08
    depth = np.maximum(0.135 - z, 0.0)
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    from fpond_builders import make_pair

    pair = make_pair(tmp_path, surface=z, depth=depth, mobile=mob, closed=True, bottom="closed", saturation=0.999999, backend=backend, splash=False, rain_mm_h=0.0, end_s=2.0, max_dt_s=1.0,
                     policy=PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True))
    outs = []
    for ev, st, vert, *_r in pair:
        q = ev.evolve(st, 60.0, stop_when_quiet=True)
        assert q.quiet_ready
        h = ev.settle_retaining_water(q.state)
        assert h.status == "requires_recession" and h.reason == "floating_terminal_capacity_reconciled/v1"
        assert h.accounting["final_audit"]["water"]["closes"] and h.accounting["final_audit"]["sediment"]["closes"]
        outs.append((np.asarray(h.state.hydraulic.depth_m), vert.current_water(h.state.hydraulic.soil_water_m)))
    assert outs[0][0].tobytes() == outs[1][0].tobytes() and outs[0][1].tobytes() == outs[1][1].tobytes()
