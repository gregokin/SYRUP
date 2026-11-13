"""H2 correction02: VALID wide/deep composed-need controls, terminal/second-settlement, D > D0, recomputed-hash H0 tampering and the windowed pending-journal order through the container (genuine producers and public file paths only)."""
from __future__ import annotations

import hashlib
import json
import shutil
import tracemalloc

import fpond_builders as fb
import numpy as np
import pytest
import static_case as sc
from case import AVAIL, snapshot
from fpond_builders import ph

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import floating_pond as fp
from maple_syrup import native_floating_soil as nfs
from maple_syrup import pond_checkpoint as pc
from maple_syrup.pond_event import PondCompletionPolicy

IDENT = {"case": "h2c2", "v": 1}


@pytest.fixture
def case_state():
    keep = (dict(sc.CASE), dict(sc.TERRAIN), sc.DZ)
    yield
    sc.CASE.clear()
    sc.CASE.update(keep[0])
    sc.TERRAIN.clear()
    sc.TERRAIN.update(keep[1])
    sc.DZ = keep[2]


def same(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), k


def wide_original(n=32):
    """Root's valid wide fixture: actual native 32x32 cells, 4 voxels, 6 grains, phase bins 32 (original deep bed exists only here)."""
    from maple.core.types.sediment_availability import (
        sediment_availability_state_from_active_layer,
    )
    from maple.core.types.sediment_ledger import zeros_sediment_ledger_state
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.core.types.water import water_state_from_depth
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.topographic_commit import (
        initial_committed_topography_state_from_physical_state,
    )
    from maple.surface.voxels import deposit_surface_mixture_batch

    from maple_syrup import probe
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondEvent, PondEventControl
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_bed import BedState
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    sc.TERRAIN.update(ny=n, nx=n)
    sc.DZ = 0.1
    context = sc.bed_context()
    g = context.geometry
    z = sc.terrain()
    col = zeros_voxel_column_state(n, n, 4, 6)
    req = (z * g.bulk_density_kg_m3 * g.dx_m * g.dy_m)[:, :, None] * np.asarray(probe.DEFAULT_CLASS_FRACTIONS)[None, None, :]
    col, _ = deposit_surface_mixture_batch(col, req, g, context.mass_resolution_kg)
    col, act, _ = initialize_active_layer_from_voxels(col, g, context.mass_resolution_kg)
    committed = initial_committed_topography_state_from_physical_state(col, act, g, context.mass_resolution_kg)
    avail = sediment_availability_state_from_active_layer(act.mass_kg, np.ones_like(act.mass_kg))
    bed = BedState(col, act, water_state_from_depth(np.full((n, n), 0.004), 6), zeros_sediment_ledger_state(n, n, 6), committed, avail)
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, context), fb.config(bottom="closed"), initial_relative_saturation=0.97, available_bytes_fn=AVAIL)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL)
    ctl = PondEventControl(backend="numpy", time=ExperimentalControl(max_dt_s=0.2, min_dt_s=1 / 1024, max_retries=12, max_steps=50000))
    ev = PondEvent(context, None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, 1000.0, 300.0), sc.topology(), control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, 6), xp=np), floating_vertical=vert)
    return ev, ev.initial_state(bed, vert.initial_soil_water())


def test_valid_wide_container_exact_need_loads_need_minus_one_refuses_and_peak_is_inside(tmp_path, case_state):
    ev, st = wide_original()
    ck = tmp_path / "ck"
    cc.save_floating_context_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    need = cc.context_load_need_bytes(ck, IDENT, available_bytes_fn=AVAIL)
    loads = []
    real = np.load
    np.load = lambda *a, **k: (loads.append(1), real(*a, **k))[1]
    try:
        with pytest.raises(nfs.NativeFloatingMemoryError):
            cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), IDENT, available_bytes_fn=lambda: need - 1)
        assert loads == []
    finally:
        np.load = real
    reserve_need = cc.context_load_need_bytes(ck, IDENT, available_bytes_fn=AVAIL, reserve_bytes=9)
    assert reserve_need == need + 9
    with pytest.raises(nfs.NativeFloatingMemoryError):
        cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), IDENT, available_bytes_fn=lambda: reserve_need - 1, reserve_bytes=9)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    e2, s2 = cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), IDENT, available_bytes_fn=lambda: need)  # EXACT need
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert s2.counters.n_accepted_steps == 0 and e2.vertical._owner is None
    assert peak <= need, (peak, need)  # traced NumPy/Python peak (includes the caller factory's own small static setup; interpreter/JIT excluded)


def test_ordinary_v300_container_exact_need_loads(tmp_path):  # renamed: this is the ordinary V=300 case, NOT the Root 10000-voxel fixture (see test_outer_c3.py)
    (tmp_path / "a").mkdir()
    from case import build, run

    ev, st, _v = build(tmp_path / "a")
    st = run(ev, st, 4)
    ck = tmp_path / "ck"
    cc.save_floating_context_checkpoint(ck, st, ev, sc_ident(), pref_spec=None if ev.vertical.held_template().preferential is None else _pref(ev), available_bytes_fn=AVAIL)
    need = cc.context_load_need_bytes(ck, sc_ident(), available_bytes_fn=AVAIL)
    cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), sc_ident(), available_bytes_fn=lambda: need)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), sc_ident(), available_bytes_fn=lambda: need - 1)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), sc_ident(), available_bytes_fn=lambda: 0)


def sc_ident():
    from case import IDENT as I

    return I


def _pref(ev):
    from maple_syrup import floating_held_context as fh

    return fh.PreferentialSpec(np.full(ev.vertical.n_cells, 2.0e-3), 0.5, 1e-5, np.array([0.0, 1.0, 1.0]), (), None)


def lake_original():
    z = ph.basin() - 0.08
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    pol = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)
    ev, st, *_r = fb.make(surface=z, closed=True, bottom="closed", saturation=0.999999, depth=np.maximum(0.135 - z, 0.0), mobile=mob, policy=pol, max_dt_s=1.0, end_s=2.0, rain_mm_h=0.0, splash=False)
    sc.DZ = 0.1  # fb.make beds use the small-voxel MAPLE probe geometry; z below is the RECORDED committed source elevation of the original topology
    sc.CASE.update(z=np.array(ev._initial_topology.source_bed_m), closed=True, policy=pol, max_dt_s=1.0, splash=False, rain=(0.0, 2.0))
    sc.TERRAIN.update(ny=3, nx=5)
    return ev, st


def test_terminal_settlement_and_a_later_empty_second_settlement_through_the_container(tmp_path, case_state):
    ev, st = lake_original()
    q = ev.evolve(st, 60.0, stop_when_quiet=True)
    wake = ev.settle_retaining_water(q.state).state
    assert wake.sediment.n_terminal_native_calls == 1 and wake.sediment.retained_quiet_reset_step == wake.counters.n_accepted_steps
    ck = tmp_path / "ck"
    cc.save_floating_context_checkpoint(ck, wake, ev, IDENT, available_bytes_fn=AVAIL)
    ref = ev.evolve(wake, 120.0, stop_when_quiet=True)  # real recession re-certifies, uninterrupted
    ev2, got = cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), IDENT, available_bytes_fn=AVAIL)
    same(snapshot(ev2, got, ev2.vertical), snapshot(ev, wake, ev.vertical))
    cont = ev2.evolve(got, 120.0, stop_when_quiet=True)
    same(snapshot(ev2, cont.state, ev2.vertical), snapshot(ev, ref.state, ev.vertical))
    done_ref, done = ev.settle_retaining_water(ref.state), ev2.settle_retaining_water(cont.state)  # EMPTY second settlement: no native call
    assert done.complete and done.state.sediment.n_terminal_native_calls == 1 == done_ref.state.sediment.n_terminal_native_calls
    same(snapshot(ev2, done.state, ev2.vertical), snapshot(ev, done_ref.state, ev.vertical))


def test_thickened_columns_d_above_d0_round_trip_through_the_container(tmp_path, case_state):
    mob = np.zeros((3, 4, 6))
    mob[:, 0, :] = 0.05
    z = ph.plane(base=0.12, drop=0.004)
    ev, st, *_r = fb.make(surface=z, bottom="closed", saturation=0.97, depth=0.004, mobile=mob, splash=False, max_dt_s=0.2)
    sc.DZ = 0.1
    sc.CASE.update(splash=False, rain=(100.0, 100.0))
    for _ in range(25):
        st = ev.step(st, 300 / 3.6e6, 0.2).state
    p, t = ev.vertical.current_payload(st.hydraulic.soil_water_m), ev.vertical.held_template()
    assert (p.depth_m > t.depth0_m).any()
    ck = tmp_path / "ck"
    cc.save_floating_context_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    ref = ev.step(st, 300 / 3.6e6, 0.2).state
    ev2, got = cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), IDENT, available_bytes_fn=AVAIL)
    same(snapshot(ev2, ev2.step(got, 300 / 3.6e6, 0.2).state, ev2.vertical), snapshot(ev, ref, ev.vertical))


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    base = tmp_path_factory.mktemp("c2")
    sc_keep = dict(sc.CASE)
    ev, st, *_r = fb.make(surface=ph.plane(base=0.12, drop=0.004), bottom="closed", saturation=0.97, depth=0.004, splash=False, max_dt_s=0.2)
    keep_dz = sc.DZ
    sc.DZ = 0.1
    sc.CASE.update(splash=False, rain=(100.0, 100.0))
    st = ev.step(st, 300 / 3.6e6, 0.2).state
    ck = base / "ck"
    cc.save_floating_context_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    yield base, ck
    sc.DZ = keep_dz
    sc.CASE.clear()
    sc.CASE.update(sc_keep)


def recompute(d, edit_arrays=None, edit_meta=None):
    """Mutate the H0 arrays/meta, then RECOMPUTE the archive size/hash and the outer seal."""
    path = d / "context_arrays.npz"
    with np.load(path) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    if edit_arrays:
        edit_arrays(arrays)
        np.savez(path, **arrays)
    man = json.loads((d / cc.MANIFEST_NAME).read_text())
    if edit_meta:
        edit_meta(man["context_meta"])
    man["files"]["context_arrays.npz"] = {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (d / cc.MANIFEST_NAME).write_text(pc._json_dumps(cc._seal(man)))


@pytest.mark.parametrize("which", ["theta_field_capacity", "theta_saturation", "thickness_m", "depth0_m"])
def test_recomputed_hash_h0_array_tampering_is_refused_by_the_identity_reproduction(plain, tmp_path, which, case_state):
    d = tmp_path / which
    shutil.copytree(plain[1], d)
    recompute(d, lambda a: a[which].__setitem__(tuple(0 for _ in a[which].shape), a[which].flat[0] * 1.0001 + 1e-9))
    with pytest.raises(fc.PondCheckpointError):
        cc.load_floating_context_checkpoint(d, sc.factory("numpy", False), IDENT, available_bytes_fn=AVAIL)


def test_recomputed_hash_h0_scalar_tampering_is_refused(plain, tmp_path, case_state):
    d = tmp_path / "scalar"
    shutil.copytree(plain[1], d)
    recompute(d, edit_meta=lambda m: m["scalars"].update(matrix_suction_m=0.2.hex()))
    with pytest.raises(fc.PondCheckpointError):
        cc.load_floating_context_checkpoint(d, sc.factory("numpy", False), IDENT, available_bytes_fn=AVAIL)


def test_a_bad_late_factory_leaves_nothing_registered_and_a_good_factory_retries(plain, case_state):
    made = []

    def bad(ctx):
        ev = sc.factory("numpy", False)(ctx)
        made.append(ev)
        sc.TERRAIN["base"] = 0.13  # a static recipe that no longer equals the original topology
        try:
            return sc.factory("numpy", False)(ctx)
        finally:
            sc.TERRAIN["base"] = 0.12

    with pytest.raises(fc.PondCheckpointError):
        cc.load_floating_context_checkpoint(plain[1], bad, IDENT, available_bytes_fn=AVAIL)
    assert made and len(made[0].vertical._registry) == 0 and made[0].vertical._provisional is None
    assert cc.load_floating_context_checkpoint(plain[1], sc.factory("numpy", False), IDENT, available_bytes_fn=AVAIL)[1] is not None


def test_windowed_pending_journal_refusal_precedes_any_publication(plain, tmp_path, monkeypatch):
    ev, st, *_r = fb.make(surface=ph.plane(base=0.12, drop=0.004), bottom="closed", saturation=0.97, splash=False)
    order = []
    monkeypatch.setattr(fc, "is_windowed_bed", lambda b: True)
    monkeypatch.setattr(fc, "_require_published", lambda b: (order.append("published-check"), (_ for _ in ()).throw(fc.PondCheckpointError("pending journal")))[1])
    with pytest.raises(fc.PondCheckpointError, match="pending journal"):
        cc.save_floating_context_checkpoint(tmp_path / "w", st, ev, IDENT, available_bytes_fn=AVAIL)
    assert order == ["published-check"] and not (tmp_path / "w").exists() and not list(tmp_path.glob(".w.partial-*"))
