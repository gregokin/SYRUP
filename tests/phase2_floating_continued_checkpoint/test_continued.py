# ruff: noqa: C408, PLC0206, RUF059  (declared table-style forge cases; unused unpack names mirror the lifecycle helper tuple)
"""CC1: explicit continued-origin CPU physical checkpoint (NumPy/Numba x Dense/published Windowed), on REAL native lifecycle states (no fake generation, capability, producer or owner)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from cont_builders import (
    AVAIL,
    IDENT,
    RAIN,
    cc,
    fc,
    lineage,
    restart_target,
    run,
    save,
    snapshot,
)

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_pond as fp
from maple_syrup import native_floating_soil as nfs
from maple_syrup import pond_checkpoint as pc

REFUSE = (fc.FloatingCheckpointError, pc.PondCheckpointError)
MATRIX = [("numpy", False), ("numpy", True), ("numba", False), ("numba", True)]


@pytest.fixture(params=MATRIX, ids=lambda p: f"{p[0]}-{'windowed' if p[1] else 'dense'}")
def ns(request, tmp_path):
    backend, windowed = request.param
    if backend == "numba":
        pytest.importorskip("numba")
    return lineage(tmp_path / "a", backend, windowed)


def same(a: dict, b: dict, skip=()):
    assert set(a) == set(b)
    for k in a:
        if not any(s in k for s in skip):
            assert a[k].tobytes() == b[k].tobytes(), k


def test_the_real_state_is_continued_origin_and_the_historical_profile_refuses_it_without_effect(ns, tmp_path):
    s = ns.state
    assert s.topology.generation >= 1 and s.counters.n_refreshes == 0 and ns.event._initial_topology.generation == s.topology.generation  # the genuine continued shape (Root's probe)
    assert not np.array_equal(s.datum.initial_full_m, ns.event._initial_topology.source_z_full)  # the GLOBAL original datum is NOT the carried routing source
    before = snapshot(ns.event, s, ns.occ.vertical)
    with pytest.raises(pc.PondCheckpointError, match="ORIGINAL initial topology / source datum"):
        fc.save_floating_pond_checkpoint(tmp_path / "old", s, ns.event, IDENT, available_bytes_fn=AVAIL)
    assert not (tmp_path / "old").exists() and not list(tmp_path.glob(".old.partial-*"))
    ck = save(ns, tmp_path / "ck")
    assert ns.lin.head is ns.occ  # the accepted head is neither advanced nor consumed by a save
    same(before, snapshot(ns.event, s, ns.occ.vertical))  # a save changes no physical state
    with pytest.raises(REFUSE, match="not a floating pond checkpoint schema"):
        e, _v, _o = restart_target(ns, ck)
        fc.load_floating_pond_checkpoint(ck, e, IDENT, available_bytes_fn=AVAIL)  # the historical loader refuses a continued file (typed, no migration)


def test_roundtrip_is_exact_and_the_continuation_is_bit_exact(ns, tmp_path):
    ck = save(ns, tmp_path / "ck")
    ev2, v2, origin = restart_target(ns, ck)
    assert origin.routing_generation == ns.state.topology.generation and origin.storage == ("windowed" if ns.windowed else "dense")
    s2 = cc.load_floating_continued_checkpoint(ck, ev2, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
    assert s2.topology.generation == ns.state.topology.generation and s2.counters.n_refreshes == 0 and v2._owner is None  # truthful local counter; no owner, no fake generation
    assert s2.topology.topology_sha256 == ns.state.topology.topology_sha256 and np.array_equal(s2.datum.initial_full_m, ns.original.source_z_full)  # carried routing + GLOBAL datum exact
    skip = ("window_cache",)
    same(snapshot(ns.event, ns.state, ns.occ.vertical), snapshot(ev2, s2, v2), skip)  # W/J/I/D, bed, phase, ledgers, counters, clocks
    a, b = ns.state, s2
    for _ in range(6):  # real native steps from both
        a = ns.event.step(a, RAIN, 0.2).state
        b = ev2.step(b, RAIN, 0.2).state
    same(snapshot(ns.event, a, ns.occ.vertical), snapshot(ev2, b, v2), skip)
    assert b.topology.topology_sha256 == ns.state.topology.topology_sha256 and b.counters.n_refreshes == a.counters.n_refreshes == 0  # below threshold: no event build, the carried routing stands
    au = ev2.audit(b)
    assert au["water"]["closes"] is True and au["sediment"]["closes"] is True


def test_current_D_and_the_one_water_are_saved_not_a_stale_initial(ns, tmp_path):
    ck = save(ns, tmp_path / "ck")
    with np.load(ck / "pond_continuation.npz") as z:
        p = ns.occ.vertical.current_payload(ns.state.hydraulic.soil_water_m)
        assert z["fl_depth"].tobytes() == np.asarray(p.depth_m).tobytes() and z["fl_water"].tobytes() == np.asarray(p.state.water_m).tobytes()
        assert z["fl_wetting"].tobytes() == np.asarray(p.history.wetting_m).tobytes() and z["fl_cum_intake"].tobytes() == np.asarray(p.history.cumulative_intake_m).tobytes()
        assert z["co_routing_source"].tobytes() == np.asarray(ns.event._initial_topology.source_z_full).tobytes()
        assert z["datum_full"].tobytes() == np.asarray(ns.original.source_z_full).tobytes()


def test_a_generation_0_historical_file_is_unchanged_and_the_two_profiles_refuse_each_other(tmp_path):
    from cont_builders import fcc, fh
    from lifecycle_builders import build

    ns = lineage(tmp_path / "a", "numpy", False)
    ck = save(ns, tmp_path / "ck")
    owner, bed, factory, pref, _ev0 = build(tmp_path / "b", "numpy", False, loose=True)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)
    rec0 = fh.export_held_context(occ.vertical, None, available_bytes_fn=AVAIL)  # the ORIGINAL root of the generation-0 lineage
    st = run(occ.event, st, 3)
    old = fc.save_floating_pond_checkpoint(tmp_path / "old", st, occ.event, IDENT, available_bytes_fn=AVAIL)
    assert json.loads((old / fc.MANIFEST_NAME).read_text())["schema"] == fc.SCHEMA_V3  # the historical writer's profile is unchanged
    ctx0 = fh.reconstruct_held_context(rec0, fcc._expected_from(rec0.meta()), available_bytes_fn=AVAIL)
    v0 = fp.FloatingPondVertical.from_held_context(ctx0, available_bytes_fn=AVAIL)
    ev_old = factory(v0)
    back = fc.load_floating_pond_checkpoint(old, ev_old, IDENT, available_bytes_fn=AVAIL)  # the historical loader still restores the historical file bit-exactly
    same(snapshot(occ.event, st, occ.vertical), snapshot(ev_old, back, v0), ("window_cache",))
    assert back.topology.generation == back.counters.n_refreshes == 0
    e, _v, _o = restart_target(ns, ck)
    with pytest.raises(REFUSE, match="not a continued-origin"):
        cc.load_floating_continued_checkpoint(old, e, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)  # an old file is refused by the continued loader
    with pytest.raises(REFUSE, match="not a continued-origin"):
        cc.read_continued_origin(old, IDENT, available_bytes_fn=AVAIL)
    with pytest.raises(REFUSE, match="not a floating pond checkpoint schema"):
        fc.load_floating_pond_checkpoint(ck, e, IDENT, available_bytes_fn=AVAIL)  # and a continued file by the historical loader
    with pytest.raises(REFUSE, match="generation-0"):
        cc.save_floating_continued_checkpoint(tmp_path / "x", st, occ.event, IDENT, original_topology=ns.state.topology, available_bytes_fn=AVAIL)  # a later routing generation cannot attest the datum
    assert not (tmp_path / "x").exists()


def _forge(src, tmp_path, name, *, manifest=None, arrays=None, binding=None):
    """A container forged by RESEALING: edits the companion arrays and/or manifest, recomputes every file hash and the manifest seal (a recomputed hash is not a proof)."""
    d = tmp_path / name
    shutil.copytree(src, d)
    if arrays:
        with np.load(d / "pond_continuation.npz") as z:
            data = {k: np.array(z[k], copy=True) for k in z.files}
        arrays(data)
        np.savez(d / "pond_continuation.npz", **data)
    man = json.loads((d / fc.MANIFEST_NAME).read_text())
    if arrays:
        man["files"] = {n: pc._hash_file(d / n, 1 << 40)[0] for n in fc.PAYLOADS}
    if manifest:
        manifest(man)
    (d / fc.MANIFEST_NAME).write_text(fc._json_dumps(fc._seal(man)))
    return d


def test_resealed_tampers_of_origin_generation_datum_source_and_counters_are_refused_before_adoption(tmp_path):
    ns = lineage(tmp_path / "a", "numpy", False)
    ck = save(ns, tmp_path / "ck")
    ev2, v2, _o = restart_target(ns, ck)
    reg0 = len(v2._registry)
    g = ns.state.topology.generation

    def bump(name, delta):
        def f(d):
            d[name] = np.array(d[name], copy=True)
            d[name].flat[len(d[name].flat) // 2] += delta
        return f

    def scal(path, value):
        def f(m):
            node = m["payload"]["scalars"]
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
        return f

    cases = [("datum", dict(arrays=bump("datum_full", 1e-3)), "datum"),
             ("route", dict(arrays=bump("co_routing_source", 1e-3)), "carried routing"),
             ("gen_meta", dict(manifest=lambda m: m["continued_origin"].update(routing_generation=g + 1)), "generation"),
             ("gen_bind", dict(manifest=lambda m: m["binding"]["initial_topology"].update(generation=g + 1)), "binding|generation"),
             ("orig_bind", dict(manifest=lambda m: m["binding"]["continued_origin"].update(original_source_sha256="0" * 64)), "binding"),
             ("n_ref", dict(manifest=scal(("counters", "n_refreshes"), g)), "refresh|scalars"),
             ("n_ref1", dict(manifest=scal(("counters", "n_refreshes"), 1)), "refresh|scalars"),
             ("topo_gen", dict(manifest=scal(("topology", "generation"), g + 1)), "generation|topology|scalars"),
             ("profile", dict(manifest=lambda m: m.update(profile=fc.PROFILE)), "profile"),
             ("schema", dict(manifest=lambda m: m.update(schema=fc.SCHEMA_V3)), "schema|keys")]
    for name, kw, rx in cases:
        with pytest.raises(REFUSE, match=rx):
            cc.load_floating_continued_checkpoint(_forge(ck, tmp_path, name, **kw), ev2, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
        assert len(v2._registry) == reg0 and v2._provisional is None, name
    # a wrong attested ORIGINAL (another datum) is refused before the arrays are used
    other = __import__("dataclasses").replace(ns.original, source_z_full=np.array(ns.original.source_z_full) + 1e-3, source_sha256="0" * 64)
    with pytest.raises(REFUSE, match="binding|original"):
        cc.load_floating_continued_checkpoint(ck, ev2, IDENT, original_topology=other, available_bytes_fn=AVAIL)
    assert len(v2._registry) == reg0
    ok = cc.load_floating_continued_checkpoint(ck, ev2, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)  # the untouched file still restores after every refusal
    assert ok.counters.n_refreshes == 0


def test_wrong_root_and_wrong_static_case_are_refused(tmp_path):
    ns = lineage(tmp_path / "a", "numpy", False)
    ck = save(ns, tmp_path / "ck")
    # an event on the ORIGINAL-generation routing (not the carried one) is refused by the binding
    ev_wrong, v_wrong, _o = restart_target(ns, ck)
    bad_event = ns.factory(v_wrong, ns.original)
    reg0 = len(v_wrong._registry)
    with pytest.raises(REFUSE, match="binding"):
        cc.load_floating_continued_checkpoint(ck, bad_event, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
    assert len(v_wrong._registry) == reg0
    # an origin of another static case cannot rebuild the carried routing
    o = cc.read_continued_origin(ck, IDENT, available_bytes_fn=AVAIL)
    shifted = __import__("dataclasses").replace(ns.original, friction=np.asarray(ns.original.friction) * 2.0)
    with pytest.raises(REFUSE, match="differs|rebuild"):
        cc.origin_topology(o, shifted)
    assert ev_wrong is not None


class _Reached(BaseException):  # (a BaseException so the loader's `except Exception` cannot convert the tripwire)
    """Raised by the np.load tripwire: the composed Stage-1 admission passed and the first payload array read is about to happen."""


def _first_read_reached(ck, ev, original, available):
    """True iff the composed admission admitted `available` bytes (the load reached its first array read, aborted by the tripwire); False iff it refused with the typed memory error before any read."""
    real, reads = np.load, []

    def trip(*a, **k):
        if str(a[0]).endswith("pond_continuation.npz") and not k.get("_origin"):
            reads.append(a[0])
            raise _Reached
        return real(*a, **k)

    np.load = trip
    try:
        try:
            cc.load_floating_continued_checkpoint(ck, ev, IDENT, original_topology=original, available_bytes_fn=lambda: available)
        except _Reached:
            return True
        except nfs.NativeFloatingMemoryError:
            assert reads == []
            return False
        raise AssertionError("the load neither reached its first read nor refused on memory")
    finally:
        np.load = real


def test_admission_precedes_every_array_read_and_the_published_need_is_exact(ns, tmp_path):
    ck = save(ns, tmp_path / "ck")
    ev2, v2, _o = restart_target(ns, ck)
    lo, hi = 0, 1 << 33
    assert _first_read_reached(ck, ev2, ns.original, hi) and not _first_read_reached(ck, ev2, ns.original, 0)
    while hi - lo > 1:  # the exact composed Stage-1 need (integer bisection over the real admission)
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if _first_read_reached(ck, ev2, ns.original, mid) else (mid, hi)
    need = hi
    assert _first_read_reached(ck, ev2, ns.original, need) and not _first_read_reached(ck, ev2, ns.original, need - 1)
    assert v2._provisional is None and len(v2._registry) == 0  # every probe left the target untouched
    table = json.loads((ck / fc.MANIFEST_NAME).read_text())["payload"]["arrays"]
    assert need > sum(int(np.prod(r["shape"], dtype=np.int64)) * np.dtype(r["dtype"]).itemsize for r in table.values())  # the composed model covers the declared companion table (incl. the origin array) and more
    if ns.windowed:
        deep = json.loads((ck / fc.MANIFEST_NAME).read_text())["deep_storage"]
        assert need >= fc.deep_cache_bound_bytes(deep)  # the declared Windowed mapping-cache bound is inside the composed need (never an archive-size proxy)
    cc.load_floating_continued_checkpoint(ck, ev2, IDENT, original_topology=ns.original, available_bytes_fn=lambda: max(need, 1 << 33))


def test_no_original_owner_probe_or_property_provider_runs_on_the_restore_path(ns, tmp_path, monkeypatch):
    ck = save(ns, tmp_path / "ck")
    from maple_syrup import probe

    def boom(*a, **k):
        raise AssertionError("original-case/owner machinery must not run on the restore path")

    for mod, name in ((nfs, "initialize_native_floating_soil"), (nfs, "stage_native_refresh"), (probe, "build_minimal_state")):
        monkeypatch.setattr(mod, name, boom)
    ev2, v2, _o = restart_target(ns, ck)
    s2 = cc.load_floating_continued_checkpoint(ck, ev2, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
    assert v2._owner is None and s2.counters.n_refreshes == 0


def test_failed_load_and_failed_save_leave_everything_retryable(ns, tmp_path):
    ck = save(ns, tmp_path / "ck")
    with pytest.raises(REFUSE, match="already exists"):
        save(ns, ck)  # no overwrite
    assert ns.lin.head is ns.occ
    ev2, v2, _o = restart_target(ns, ck)
    with pytest.raises(REFUSE):
        cc.load_floating_continued_checkpoint(ck, ev2, {"case": "other"}, original_topology=ns.original, available_bytes_fn=AVAIL)  # wrong identity
    assert len(v2._registry) == 0 and v2._provisional is None
    s2 = cc.load_floating_continued_checkpoint(ck, ev2, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)  # retry succeeds
    assert v2.current_payload(s2.hydraulic.soil_water_m) is not None and len(v2._registry) == 1
