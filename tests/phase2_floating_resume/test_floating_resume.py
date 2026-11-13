"""R1a: genuine in-memory capture/restore of the floating Pond state (CPU/host). Real MAPLE beds, real native owners/verticals, real events; no fake native classes, no file codec."""
# ruff: noqa: RUF059
from __future__ import annotations

import copy
import dataclasses
import hashlib
import pickle
import tracemalloc

import numpy as np
import pytest
from fpond_builders import AVAIL, make, make_pair, ph

from maple_syrup import floating_pond as fp
from maple_syrup import floating_resume as fr
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs

RAIN = 100.0 / 3.6e6
BACKENDS = ["numpy", "numba"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def run(ev, st, n, rate=RAIN, dt=0.2):
    for _ in range(n):
        st = ev.step(st, rate, dt).state
    return st


def _arrays(obj, out, prefix=""):
    if isinstance(obj, np.ndarray):
        out[prefix] = obj
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _arrays(getattr(obj, f.name), out, prefix + "." + f.name)
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            _arrays(v, out, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float, str, bool, type(None))):
        out[prefix] = np.asarray(repr(obj))
    elif isinstance(obj, dict):
        for k in sorted(obj):
            _arrays(obj[k], out, f"{prefix}[{k}]")


def fingerprint(ev, st, vert):
    out = {}
    _arrays(st.hydraulic, out, "hyd")
    _arrays(st.bed.water, out, "bedwater")
    _arrays(st.bed.active_layer, out, "active")
    _arrays(st.bed.voxel_column, out, "voxel") if not hasattr(st.bed.voxel_column, "deep_store") else _arrays(st.bed.voxel_column.window_kg, out, "window")
    for name in ("sediment_velocity_m_s", "phase_fraction", "phase_position_m", "held_source_m"):
        out[name] = np.asarray(getattr(st, name))
    _arrays(st.water, out, "water")
    _arrays(st.sediment, out, "sediment")
    _arrays(st.counters, out, "counters")
    _arrays(st.quiet, out, "quiet")
    _arrays(st.topology.source_z_full, out, "topo")
    p = vert.current_payload(st.hydraulic.soil_water_m)
    out["W"], out["J"], out["I"], out["D"] = (np.asarray(p.state.water_m), np.asarray(p.history.wetting_m), np.asarray(p.history.cumulative_intake_m), np.asarray(p.depth_m))
    out["t"] = np.asarray(repr(p.state.t_s))
    return out


def same(a, b, what=""):
    assert a.keys() == b.keys(), what
    for k in a:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, (what, k)
        assert a[k].tobytes() == b[k].tobytes(), (what, k)


def state_hash(st):
    h = hashlib.sha256()
    out = {}
    _arrays(st, out)
    for k in sorted(out):
        h.update(k.encode())
        h.update(np.ascontiguousarray(out[k]).tobytes())
    return h.hexdigest()


def roundtrip(kw, steps_before, steps_after, *, backend, pre=None, rate=RAIN, dt=0.2, rate_after=None):
    ev, st, bed, ctx, vert, _o = make(backend=backend, **kw)
    if pre is not None:
        st = pre(ev, st, vert, ctx)
    st = run(ev, st, steps_before, rate=rate, dt=dt)
    cap = fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    ev2, _st2, _b2, _c2, vert2, _o2 = make(backend=backend, **kw)
    before_hash = state_hash(st)
    r = fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    same(fingerprint(ev, st, vert), fingerprint(ev2, r, vert2), "restored == captured")
    ra = RAIN if rate_after is None else rate_after
    ua = run(ev, st, steps_after, rate=ra, dt=dt)
    rb = run(ev2, r, steps_after, rate=ra, dt=dt)
    same(fingerprint(ev, ua, vert), fingerprint(ev2, rb, vert2), "continued")
    assert state_hash(st) == before_hash  # the captured parent value is bit for bit unchanged by both continuations
    for e, s in ((ev, ua), (ev2, rb)):
        a = e.audit(s)
        assert a["water"]["closes"] and a["sediment"]["closes"]
    return ev, st, vert, ev2, r, vert2, cap


def test_roundtrip_actual_continuation_with_erosion_preferential_and_history(backend):
    ev, st, vert, ev2, r, vert2, cap = roundtrip({"splash": True, "depth": 0.004, "preferential": True}, 14, 8, backend=backend)
    assert vert.stats.depth_changed_columns > 0  # the captured payload is a genuine geometry VIEW, not the prepared column
    p = vert.current_payload(st.hydraulic.soil_water_m)
    assert p.column.identity_kind == "geometry_view_token" and p.history.cumulative_intake_m.max() > 0.0 and p.history.wetting_m.max() > 0.0
    q = vert2.current_payload(r.hydraulic.soil_water_m)
    assert q.column.identity_kind == "geometry_view_token" and q.column.sha256 != p.column.sha256  # generation tokens are NOT durable and never compared
    assert q.column.parent_sha256 == p.column.parent_sha256 == vert.held_template().column.sha256


def test_deposition_grown_depth_with_water_above_the_original_D0_capacity(backend):
    """A real deposition raises D of a shallow closed profile; later rain fills the new capacity so that W EXCEEDS what the ORIGINAL prepared (D0) column could hold. Restoring W against the held D0
    column would be refused; against the CURRENT geometry it is valid (and the restored continuation is bit for bit the uninterrupted one)."""
    z = ph.plane(base=0.12, drop=0.004)
    kw = {"surface": z, "bottom": "closed", "saturation": 0.999, "splash": True, "end_s": 100000.0, "max_dt_s": 0.5, "diffusivity_m2_s": np.full(2, 1e-5), "rain_mm_h": 500.0}

    def raise_cell(ev, st, vert, ctx):
        raised = ph.move_mass(st.bed, ctx, (1, 2), (1, 1), 0.01)
        touched = np.zeros((3, 4), bool)
        touched[1, 1] = True
        hyd, rec = vert.after_exchange(raised, ctx, st.hydraulic, touched, np.ones((3, 4), bool))
        assert rec["n_depth_changed"] >= 1
        return dataclasses.replace(st, bed=raised, hydraulic=hyd)

    ev, st, vert, ev2, r, vert2, cap = roundtrip(kw, 400, 6, backend=backend, pre=raise_cell, rate=5 * RAIN, dt=0.5, rate_after=5 * RAIN)
    t = vert.held_template()
    capacity0 = np.stack([fsl.porous_thickness(t.column, k) * t.column._maps["theta_saturation"][k] for k in range(t.column.n_layers)])
    w = cap.current["water_m"]
    assert (w > capacity0 + 1e-12).any(), "fixture must put W above the original D0 capacity (not vacuous)"
    with pytest.raises(fsl.FloatingSoilError):
        fsl.floating_continued_state(t.column, w, cap.soil_t_s)  # the OLD owner-style validation against D0 refuses exactly this state


def test_windowed_published_boundary_roundtrip(tmp_path):
    (_d, _s, _v, _o, _b, _c), (ev, st, vert, _ow, bed, ctx) = make_pair(tmp_path, splash=True, deep_publication="step")
    (_ed, _sd, _vd, _od, _bd, _cd) = (None,) * 6
    st = run(ev, st, 12)
    cap = fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    assert cap.bed_storage == "windowed"
    _dense, (ev2, _st2, vert2, _o2, _b2, _c2) = make_pair(tmp_path / "second", splash=True, deep_publication="step")
    r = fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    same(fingerprint(ev, st, vert), fingerprint(ev2, r, vert2), "windowed restored")
    ua, rb = run(ev, st, 6), run(ev2, r, 6)
    same(fingerprint(ev, ua, vert), fingerprint(ev2, rb, vert2), "windowed continued")


def test_midrain_recession_and_retained_quiet_states_roundtrip(backend):
    kw = {"splash": True, "depth": 0.004, "end_s": 1000.0, "closed": True}
    roundtrip(kw, 10, 6, backend=backend)  # mid-rain
    roundtrip(kw, 10, 25, backend=backend, rate_after=0.0)  # recession from the restored state (rain 0)
    from maple_syrup.pond_event import PondCompletionPolicy

    rk = {"surface": ph.basin(), "closed": True, "end_s": 3.0, "policy": PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True, max_unit_discharge_m2_s=2.0e-6, max_depth_change_m=1.0e-4), "splash": True, "saturation": 0.3, "depth": 0.001}
    ev, st, bed, ctx, vert, _o = make(backend=backend, **rk)
    st = run(ev, st, 12)
    res = ev.evolve(st, 300.0, stop_when_quiet=True)
    assert res.quiet_ready
    h = ev.settle_retaining_water(res.state)
    assert h.status in ("complete", "requires_recession")
    cap = fr.capture_floating_pond(ev, h.state, available_bytes_fn=AVAIL)  # a retained wet state (complete or requires_recession) is captured as the wet value it is
    ev2, *_x, vert2, _o2 = make(backend=backend, **rk)
    r = fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    same(fingerprint(ev, h.state, vert), fingerprint(ev2, r, vert2), "retained")
    same(fingerprint(ev, run(ev, h.state, 5, rate=0.0), vert), fingerprint(ev2, run(ev2, r, 5, rate=0.0), vert2), "retained continued")


def test_capture_and_restore_do_not_consume_or_mutate_anything(backend):
    ev, st, bed, ctx, vert, owner = make(backend=backend, splash=True, depth=0.004)
    st = run(ev, st, 10)
    h0, w0 = state_hash(st), vert.current_water(st.hydraulic.soil_water_m).tobytes()
    reg0 = len(vert._registry)
    cap = fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    assert state_hash(st) == h0 and len(vert._registry) == reg0 and owner.is_current
    ev2, *_x, vert2, owner2 = make(backend=backend, splash=True, depth=0.004)
    r1 = fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    r2 = fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)  # two independent restored branches of one capture
    same(fingerprint(ev2, r1, vert2), fingerprint(ev2, r2, vert2), "two restores")
    a, b = run(ev2, r1, 5), run(ev2, r2, 5)
    same(fingerprint(ev2, a, vert2), fingerprint(ev2, b, vert2), "two branches continue identically")
    assert state_hash(st) == h0 and vert.current_water(st.hydraulic.soil_water_m).tobytes() == w0 and owner2.is_current
    assert cap.current["water_m"].flags.writeable is False and cap.held["theta_saturation"].flags.writeable is False


def test_restore_never_reads_a_property_from_the_current_bed(backend, monkeypatch):
    ev, st, bed, ctx, vert, _o = make(backend=backend, splash=True, depth=0.004)
    st = run(ev, st, 12)
    cap = fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    ev2, *_x, _v2, _o2 = make(backend=backend, splash=True, depth=0.004)
    from maple_syrup import soil_texture_properties as stp

    def boom(*a, **k):
        raise AssertionError("the property provider / texture reader must not run during restore")

    for obj, name in ((stp, "estimate_soil_texture_properties"), (nfs, "_read_texture"), (nfs, "read_canonical_depth"), (nfs, "initialize_native_floating_soil"), (nfs, "stage_native_refresh"),
                      (nfs, "native_source_witness"), (fsl, "prepare_floating_column")):
        monkeypatch.setattr(obj, name, boom)
    fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)


def test_measured_override_semantics_are_not_changed_by_capture_or_restore(backend):
    over = {"theta_33_override": np.full((3, 12), np.nan)}
    over["theta_33_override"][1, 5] = 0.3
    ev, st, *_r, vert, owner = make(backend=backend, depth=0.004, splash=True, overrides=over)
    st = run(ev, st, 8)
    cap = fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    ev2, *_x, vert2, owner2 = make(backend=backend, depth=0.004, splash=True, overrides=over)
    sha_cfg, gen = owner2.config.sha256, owner2.generation
    fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    assert owner2.config.sha256 == sha_cfg == owner.config.sha256 and owner2.generation == gen and owner2.is_current
    assert cap.owner_config_sha256 == sha_cfg  # the override is part of the config identity; restore neither adds nor freezes one


def test_refusals_precede_any_mutation(backend):
    ev, st, bed, ctx, vert, owner = make(backend=backend, splash=True, depth=0.004)
    st = run(ev, st, 6)
    cap = fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    ev2, *_x, vert2, _o2 = make(backend=backend, splash=True, depth=0.004)
    reg = len(vert2._registry)
    with pytest.raises(fr.FloatingResumeError):
        fr.restore_floating_pond(ev2, {"schema": fr.SCHEMA}, available_bytes_fn=AVAIL)  # plain caller record
    for bad in (lambda: copy.copy(cap), lambda: copy.deepcopy(cap), lambda: pickle.dumps(cap), lambda: dataclasses.replace(cap, generation=9)):
        with pytest.raises(fr.FloatingResumeError):
            bad()
    with pytest.raises(fr.FloatingResumeError, match="explicit available_bytes_fn"):
        fr.restore_floating_pond(ev2, cap, available_bytes_fn=None)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fr.restore_floating_pond(ev2, cap, available_bytes_fn=lambda: 0)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fr.capture_floating_pond(ev, st, available_bytes_fn=lambda: 0)
    other_rain, *_y, _vo, _oo = make(backend=backend, splash=True, depth=0.004, rain_mm_h=50.0)
    with pytest.raises(fr.FloatingResumeError, match="static configuration"):
        fr.restore_floating_pond(other_rain, cap, available_bytes_fn=AVAIL)
    other_depth, *_y, _vd, _od = make(backend=backend, splash=True, depth=0.004, organic_matter_percent=np.full(3, 6.0))  # another held property source -> another held template/identity
    with pytest.raises(fr.FloatingResumeError):
        fr.restore_floating_pond(other_depth, cap, available_bytes_fn=AVAIL)
    with pytest.raises(fr.FloatingResumeError):
        fr.capture_floating_pond(ev, dataclasses.replace(st, hydraulic=dataclasses.replace(st.hydraulic, soil_water_m=np.array(st.hydraulic.soil_water_m, copy=True))), available_bytes_fn=AVAIL)
    with pytest.raises(fr.FloatingResumeError):
        fr.capture_floating_pond(ev, "state", available_bytes_fn=AVAIL)
    for plain in ({"x": 1}, cap, None):
        with pytest.raises(fr.FloatingResumeError):
            vert2.adopt_resumed(plain)  # neither a plain record nor a genuine capture is a prepared payload
    with pytest.raises(TypeError):
        vert2.adopt_resumed(cap, np.zeros((3, 4)), column=None)  # free keyword parts no longer exist
    assert len(vert2._registry) == reg  # nothing registered by any refusal


def test_other_backend_cannot_restore(backend):
    if backend == "numba":
        pytest.skip("pair covered from numpy side")
    pytest.importorskip("numba")
    ev, st, *_r = make(backend="numpy", splash=True, depth=0.004)
    st = run(ev, st, 4)
    cap = fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    evn, *_x = make(backend="numba", splash=True, depth=0.004)
    with pytest.raises(fr.FloatingResumeError, match="static configuration"):
        fr.restore_floating_pond(evn, cap, available_bytes_fn=AVAIL)


def test_windowed_pending_journal_is_refused(tmp_path):
    from types import SimpleNamespace

    from fpond_builders import NC, _windowed_helpers, config, native_bed

    from maple_syrup.pond_event import PondEvent
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_bed import pending_journal_records
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    wh = _windowed_helpers()
    z = ph.plane(base=0.12, drop=0.004)
    dense, ctx = wh.dense_deep_bed(z, depth=0.003, mobile=None)
    wbed, wctx = wh.windowed_counterpart(dense, ctx, tmp_path / "store")
    topo = ph.make_topology(dense, ph.east_outlets(*z.shape), drop=0.02)
    owner = nfs.initialize_native_floating_soil(native_bed(wbed, wctx), config(), initial_relative_saturation=0.5, available_bytes_fn=AVAIL)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL)
    ctl = wh.control_for(SimpleNamespace(hydraulics="numpy", xp=np), windowed=True, max_dt_s=0.2, deep_publication="boundary", high_water=500_000)
    ev = PondEvent(wctx, None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, 100.0, 100.0), topo, control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, NC), xp=np), floating_vertical=vert)
    st = ev.initial_state(wbed, vert.initial_soil_water())
    for _ in range(60):
        st = ev.step(st, RAIN, 0.2).state
        if pending_journal_records(st.bed):
            break
    assert pending_journal_records(st.bed) > 0, "fixture must reach a genuine pending deep journal (not vacuous)"
    with pytest.raises(fr.FloatingResumeError, match="pending deep journal"):
        fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)


def test_bounded_memory_of_capture_and_restore(backend):
    ev, st, *_r, vert, _o = make(backend=backend, splash=True, depth=0.004)
    st = run(ev, st, 8)
    ev2, *_x, vert2, _o2 = make(backend=backend, splash=True, depth=0.004)
    L, N = vert.n_layers, vert.n_cells
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    cap = fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    cpeak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    rpeak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert cpeak <= (3 * L + 5) * N * 8 + 200_000, cpeak  # small fixture; the declared estimate plus bounded interpreter overhead
    assert rpeak <= 4 * ((2 * L + 8) * N * 8 + 16 * N) + 2_000_000, rpeak


def test_default_view_pond_is_untouched_by_the_additive_hooks():
    """A default (bucket) Pond never imports the resume module nor gains a vertical hook."""
    from types import SimpleNamespace

    be = SimpleNamespace(hydraulics="numpy", xp=np)
    ev, _st = ph.wet_plane(be)
    assert ev.vertical is None
    with pytest.raises(fr.FloatingResumeError):
        fr.capture_floating_pond(ev, _st, available_bytes_fn=AVAIL)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# correction01: integrity, foreign adoption, strict reserve, registration last
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _cap(backend, steps=8, **kw):
    ev, st, *_r, vert, _o = make(backend=backend, splash=True, depth=0.004, **kw)
    st = run(ev, st, steps)
    cap = fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    ev2, *_x, vert2, _o2 = make(backend=backend, splash=True, depth=0.004, **kw)
    return ev, st, vert, cap, ev2, vert2


def test_capture_integrity_is_bound_to_every_captured_value(backend):
    ev, st, vert, cap, ev2, vert2 = _cap(backend)
    reg = len(vert2._registry)
    # ordinary container replacement is impossible (read-only mappings)
    for mapping in (cap.current, cap.held, cap.source_pair):
        with pytest.raises(TypeError):
            mapping["water_m"] = np.zeros((3, 12))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.generation = 5
    # a legal redistribution between layers (the layer swap preserves every derived cell sum exactly) replaced through low-level API is refused
    w = np.array(cap.current["water_m"], copy=True)
    swapped = w.copy()
    swapped[[0, 1]] = w[[1, 0]]
    assert np.array_equal(w.sum(axis=0), swapped.sum(axis=0)) and not np.array_equal(w, swapped)
    original = cap.current
    object.__setattr__(cap, "current", {**dict(original), "water_m": swapped})
    with pytest.raises(fr.FloatingResumeError):
        fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    from types import MappingProxyType

    sealed = swapped.copy()
    sealed.flags.writeable = False
    object.__setattr__(cap, "current", MappingProxyType({**dict(original), "water_m": sealed}))
    with pytest.raises(fr.FloatingResumeError, match="altered"):
        fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    object.__setattr__(cap, "current", original)
    fr.require_capture(cap)  # restoring the original object restores integrity: the digest binds the VALUES
    # every other value is bound: J, intake, D, held maps, scalar facts, the source pair
    for key in ("wetting_m", "cumulative_intake_m", "depth_m"):
        a = original[key]
        keep = a.copy()
        a.flags.writeable = True  # low-level re-enabling of a sealed array
        a.reshape(-1)[0] += 1e-9
        with pytest.raises(fr.FloatingResumeError):
            fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
        a[...] = keep
        a.flags.writeable = False
        fr.require_capture(cap)
    for key in cap.held:
        a = cap.held[key]
        keep = a.copy()
        a.flags.writeable = True
        flat = a.reshape(-1)
        flat[0] = np.nextafter(flat[0], np.inf)
        with pytest.raises(fr.FloatingResumeError):
            fr.require_capture(cap)
        a[...] = keep
        a.flags.writeable = False
        fr.require_capture(cap)
    for field, value in (("generation", 7), ("source_witness", "0" * 64), ("root_column_sha256", "1" * 64), ("soil_t_s", cap.soil_t_s + 1.0), ("backend", "numba" if backend == "numpy" else "numpy")):
        old = getattr(cap, field)
        object.__setattr__(cap, field, value)
        with pytest.raises(fr.FloatingResumeError):
            fr.require_capture(cap)
        object.__setattr__(cap, field, old)
    fr.require_capture(cap)
    assert len(vert2._registry) == reg
    r = fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)  # the intact capture still restores exactly
    same(fingerprint(ev, st, vert), fingerprint(ev2, r, vert2), "intact after refusals")


def test_adoption_accepts_only_a_minted_prepared_record_and_reverifies_every_part(backend):
    ev, st, vert, cap, ev2, vert2 = _cap(backend)
    reg = len(vert2._registry)
    import weakref

    from maple_syrup import floating_matrix_infiltration as fmi
    # a FOREIGN genuine column/operator/history/state (another held property source) cannot be registered even if a prepared record is forced through the private window
    other, *_y, vo, _oo = make(backend=backend, splash=True, depth=0.004, organic_matter_percent=np.full(3, 6.0))
    tf = vo.held_template()
    foreign_hist = fmi.floating_matrix_history(tf.matrix_op, wetting_m=np.array(cap.current["wetting_m"]), cumulative_intake_m=np.array(cap.current["cumulative_intake_m"]))
    foreign_state = fsl.floating_continued_state(tf.column, np.zeros((3, 12)), cap.soil_t_s)
    soil = np.asarray(foreign_state.water_m).sum(axis=0).reshape(tuple(vert2.shape))
    parts = fr._PreparedParts(soil, tf.column, tf.matrix_op, foreign_hist, None, foreign_state, np.array(tf.depth0_m))
    with fr._minting():
        forged = fr._PreparedResume(cap, weakref.ref(vert2), parts, [False])
    with pytest.raises(fr.FloatingResumeError, match="foreign|differ"):
        vert2.adopt_resumed(forged)
    with pytest.raises(fr.FloatingResumeError), vert2.resume_scope(forged):
        pass
    assert len(vert2._registry) == reg and vert2._provisional is None
    # a prepared record cannot be constructed outside the window, copied or serialised; another vertical's record is refused
    with pytest.raises(fr.FloatingResumeError):
        fr._PreparedResume(cap, weakref.ref(vert2), parts, [False])
    with pytest.raises(fr.FloatingResumeError):
        pickle.dumps(forged)
    with pytest.raises(fr.FloatingResumeError):
        vert.adopt_resumed(forged)


@pytest.mark.parametrize("bad", [-1, -(10**9), True, False, 1.0, 1e9, float("nan"), float("inf"), "0", None, np.float64(3.0), np.bool_(True)])
def test_reserve_bytes_is_a_strict_nonnegative_integer_before_any_read_or_copy(backend, bad):
    ev, st, vert, cap, ev2, vert2 = _cap(backend)
    reads = []

    def reader():
        reads.append(1)
        return 10**12

    with pytest.raises(fr.FloatingResumeError):
        fr.capture_floating_pond(ev, st, available_bytes_fn=reader, reserve_bytes=bad)
    with pytest.raises(fr.FloatingResumeError):
        fr.restore_floating_pond(ev2, cap, available_bytes_fn=reader, reserve_bytes=bad)
    assert reads == []  # refused BEFORE the availability reader is consulted


def test_reserve_accepts_nonbool_integers_and_zero_memory_fails_exactly_before_allocation(backend):
    ev, st, vert, cap, ev2, vert2 = _cap(backend)
    fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL, reserve_bytes=np.int64(5))
    fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL, reserve_bytes=0)
    L, N = vert.n_layers, vert.n_cells
    need = (3 * L + 5) * N * 8 + fr.HASH_ALLOWANCE_BYTES
    fr.capture_floating_pond(ev, st, available_bytes_fn=lambda: need)  # exactly the declared estimate
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fr.capture_floating_pond(ev, st, available_bytes_fn=lambda: need - 1)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fr.capture_floating_pond(ev, st, available_bytes_fn=lambda: 0)
    reg = len(vert2._registry)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fr.restore_floating_pond(ev2, cap, available_bytes_fn=lambda: 0)
    assert len(vert2._registry) == reg


def test_registration_is_last_a_late_failure_leaves_registry_stats_and_capture_untouched(backend, monkeypatch):
    ev, st, vert, cap, ev2, vert2 = _cap(backend)
    reg0, peak0, steps0 = len(vert2._registry), vert2.stats.peak_registry, vert2.stats.steps
    audit = ev2.audit
    sizes = []

    def failing(state):
        sizes.append(len(vert2._registry))  # the candidate is validated BEFORE registration: the registry is still unchanged here
        out = audit(state)
        out["water"] = dict(out["water"], closes=False)
        return out

    monkeypatch.setattr(ev2, "audit", failing)
    held = None
    try:
        fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    except fr.FloatingResumeError as exc:
        held = exc  # the exception (and its traceback) is HELD: no reliance on garbage collection
    assert held is not None and sizes == [reg0]
    assert len(vert2._registry) == reg0 and vert2.stats.peak_registry == peak0 and vert2.stats.steps == steps0 and vert2._provisional is None
    fr.require_capture(cap)
    monkeypatch.undo()

    def raising(state):
        raise fp.FloatingPondError("injected")

    monkeypatch.setattr(ev2, "_check_state", raising)
    try:
        fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)
    except fr.FloatingResumeError as exc:
        held = exc
    assert len(vert2._registry) == reg0 and vert2._provisional is None
    monkeypatch.undo()
    r = fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)  # the same capture still restores after the refused attempts
    assert len(vert2._registry) == reg0 + 1
    same(fingerprint(ev, st, vert), fingerprint(ev2, r, vert2), "after late failure")
    assert held is not None


def test_source_hash_and_imports_are_computed_outside_the_minting_window(backend, monkeypatch):
    ev, st, *_r = make(backend=backend, splash=True, depth=0.004)
    st = run(ev, st, 4)
    seen = []
    real = fr._source_pair

    def spy():
        seen.append(getattr(fr._MINT, "ok", False))
        return real()

    monkeypatch.setattr(fr, "_source_pair", spy)
    fr.capture_floating_pond(ev, st, available_bytes_fn=AVAIL)
    assert seen == [False]


# ---------------------------------------------------------------------------------------------------------------------------------------------
# correction02: cold first-use source hash is modelled and admitted; malformed metadata is refused before any copy/hash
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _cold(monkeypatch):
    """Actual process-local source caches EMPTY (test setup only: the real source hashes then run on first use; monkeypatch restores the warm dicts)."""
    from maple_syrup import pond_checkpoint as pc
    from maple_syrup import sediment_bed as sb

    monkeypatch.setattr(pc, "_SOURCE_CACHE", {})
    monkeypatch.setattr(sb, "_MAPLE_DIGEST_CACHE", {})
    return pc, sb


def test_cold_first_use_source_hash_is_inside_the_declared_admission(backend, monkeypatch):
    ev, st, vert, cap, ev2, vert2 = _cap(backend)
    L, N = vert.n_layers, vert.n_cells
    base = (3 * L + 5) * N * 8 + fr.HASH_ALLOWANCE_BYTES
    from maple_syrup import provenance

    cold = int(provenance._READ_CHUNK_BYTES) + fr.COLD_METADATA_ALLOWANCE_BYTES
    pc, sb = _cold(monkeypatch)
    assert fr._cold_source_bytes() == cold
    need = base + cold
    reads = []
    with pytest.raises(nfs.NativeFloatingMemoryError):  # one byte short: refused BEFORE any source hash / owned allocation (the caches are still empty afterwards)
        fr.capture_floating_pond(ev, st, available_bytes_fn=lambda: reads.append(1) or need - 1)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fr.capture_floating_pond(ev, st, available_bytes_fn=lambda: 0)
    assert "pair" not in pc._SOURCE_CACHE and "digest" not in sb._MAPLE_DIGEST_CACHE and len(reads) == 1
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    cap2 = fr.capture_floating_pond(ev, st, available_bytes_fn=lambda: need)  # the exact declared need, REAL first-use hashing
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert "pair" in pc._SOURCE_CACHE and "digest" in sb._MAPLE_DIGEST_CACHE, "the control must have run the real first-use source hash"
    assert peak <= need, (peak, need)  # actual peak within the model (the arrays are tiny; the cold hash dominates)
    assert peak > int(provenance._READ_CHUNK_BYTES), "the cold hash must actually dominate (not a vacuous control)"
    assert fr._cold_source_bytes() == 0  # warm now: nothing is charged twice
    fr.capture_floating_pond(ev, st, available_bytes_fn=lambda: base)  # warm need is exactly the owned-array model
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fr.capture_floating_pond(ev, st, available_bytes_fn=lambda: base - 1)
    # restore: the cold hash is admitted BEFORE the source pair is computed
    pc, sb = _cold(monkeypatch)
    rneed = (2 * L + 8) * N * 8 + 16 * N + fr.HASH_ALLOWANCE_BYTES + cold
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fr.restore_floating_pond(ev2, cap2, available_bytes_fn=lambda: rneed - 1)
    assert "pair" not in pc._SOURCE_CACHE and "digest" not in sb._MAPLE_DIGEST_CACHE
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    fr.restore_floating_pond(ev2, cap2, available_bytes_fn=lambda: rneed)
    rpeak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert rpeak <= rneed + 2_000_000, (rpeak, rneed)  # + the existing validation/audit working set of the event (not part of the capture model), see report


def test_wrong_source_pair_still_fails(backend, monkeypatch):
    ev, st, vert, cap, ev2, vert2 = _cap(backend)
    real = fr._source_pair
    monkeypatch.setattr(fr, "_source_pair", lambda: {**real(), "maple_syrup_tree_sha256": "0" * 64})
    with pytest.raises(fr.FloatingResumeError, match="source pair"):
        fr.restore_floating_pond(ev2, cap, available_bytes_fn=AVAIL)


def _tamper(cap, name, value, group="current"):
    cur = dict(getattr(cap, group))
    cur[name] = value
    object.__setattr__(cap, group, __import__("types").MappingProxyType(cur))


class _Boom:
    def __array__(self, *a, **k):
        raise AssertionError("an __array__ callback must never run")


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_malformed_array_metadata_is_refused_before_any_copy_hash_or_reader_call(backend, monkeypatch):
    ev, st, vert, cap, ev2, vert2 = _cap(backend)
    good = cap.current["water_m"]
    L, N = good.shape
    wide = np.empty((L, N * 2))
    wide[:, ::2] = good
    bads = [
        wide[:, ::2],  # same shape, same dtype, NON-contiguous
        np.asfortranarray(good),  # same shape, Fortran order
        good.astype(np.float32),  # dtype
        good.reshape(-1),  # shape
        good[:, :-1].copy(),  # shape
        np.matrix(good),  # ndarray subclass
        [list(r) for r in good],  # not an ndarray
        _Boom(),  # array-like with a callback
        None,
    ]
    reads = []
    copies = []
    real = np.ascontiguousarray

    def watch(*a, **k):
        copies.append(1)
        return real(*a, **k)

    monkeypatch.setattr(np, "ascontiguousarray", watch)
    original = cap.current
    for bad in bads:
        _tamper(cap, "water_m", bad)
        for call in (lambda: fr.restore_floating_pond(ev2, cap, available_bytes_fn=lambda: reads.append(1) or 0), lambda: fr.require_capture(cap)):
            with pytest.raises(fr.FloatingResumeError):
                call()
    assert reads == [] and copies == []  # typed refusals, zero reader calls, no ascontiguousarray value copy
    for group, name in (("held", "theta_saturation"), ("current", "depth_m"), ("current", "wetting_m")):
        _tamper(cap, name, np.asarray(getattr(cap, group)[name])[::-1].astype(np.float32), group)
        with pytest.raises(fr.FloatingResumeError):
            fr.require_capture(cap)
    object.__setattr__(cap, "current", dict(original))  # a plain dict container instead of the read-only mapping
    with pytest.raises(fr.FloatingResumeError):
        fr.require_capture(cap)
    object.__setattr__(cap, "current", original)
    object.__setattr__(cap, "shape", (3,))
    with pytest.raises(fr.FloatingResumeError):
        fr.require_capture(cap)
    assert reads == [] and copies == []


def test_value_integrity_is_unchanged_by_the_metadata_gate(backend):
    ev, st, vert, cap, ev2, vert2 = _cap(backend)
    w = np.array(cap.current["water_m"], copy=True)
    w[[0, 1]] = w[[1, 0]]  # exact-metadata, altered values (cell sums preserved): still refused by the value digest
    w.flags.writeable = False
    _tamper(cap, "water_m", w)
    with pytest.raises(fr.FloatingResumeError, match="altered"):
        fr.require_capture(cap)
