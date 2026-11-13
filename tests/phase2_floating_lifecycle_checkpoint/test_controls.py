"""S1 controls: save authority/refusals/retry, expected seal, resealed tampers, composed admission, unpublished Windowed, parent release and re-save of a restored head. REAL native lifecycle states only."""
from __future__ import annotations

import dataclasses
import gc
import json
import shutil
import weakref
from pathlib import Path

import numpy as np
import pytest
from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_lifecycle_checkpoint as lcp
from maple_syrup import floating_pond as fp
from maple_syrup import native_floating_soil as nfs
from maple_syrup import pond_checkpoint as pc
from maple_syrup import soil_evaporation as se
from maple_syrup.pond_event import PondEvent

from lcp_builders import (
    AVAIL,
    IDENT,
    PART,
    boundary_ns,
    closed_day_ns,
    load,
    plan,
    run_day,
    save,
)

REFUSE = (lcp.LifecycleCheckpointError, fc.FloatingCheckpointError, pc.PondCheckpointError, nfs.NativeFloatingMemoryError)


def _forge(src: Path, tmp: Path, name: str, edit) -> tuple:
    """Copy the container, edit the OUTER manifest, RESEAL it (a recomputed seal is not a proof): returns `(dir, new_seal)`."""
    d = tmp / name
    shutil.copytree(src, d)
    m = json.loads((d / lcp.MANIFEST_NAME).read_text())
    edit(m)
    from maple_syrup import floating_context_checkpoint as fcc

    sealed = fcc._seal(m)
    (d / lcp.MANIFEST_NAME).write_text(pc._json_dumps(sealed))
    return d, sealed["manifest_sha256"]


@pytest.fixture(scope="module")
def cday(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cday")
    ns = closed_day_ns(tmp / "a", "numpy", False, days=2)
    saved = save(ns, tmp / "ck", ns.closed)
    return ns, saved, tmp


def test_save_authority_refusals_and_retry_leave_the_head_and_the_directory_clean(tmp_path):
    ns = closed_day_ns(tmp_path / "a", "numpy", False, days=1)
    head = ns.lin.head
    other = closed_day_ns(tmp_path / "b", "numpy", False, days=1)
    n = ns.closed.ctx.n_cells
    trial = br.day_action(ns.closed, plan=plan(1), pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL).state  # a mid-day interevent trial
    bad = [(None, "closed"), (other.closed, "genuine closed day"), (trial, "genuine closed day")]
    for closed, rx in bad:
        with pytest.raises(lcp.LifecycleCheckpointError, match=rx):
            save(ns, tmp_path / "x", closed)
    with pytest.raises(lcp.LifecycleCheckpointError, match="genuine closed daily state"):
        lcp.save_floating_lifecycle_checkpoint(tmp_path / "x", other.lin, other.factory, IDENT, closed=None, original_topology=other.original, available_bytes_fn=AVAIL)  # hold head, no closed
    assert not (tmp_path / "x").exists() and not list(tmp_path.glob(".x.partial-*")) and ns.lin.head is head
    saved = save(ns, tmp_path / "ok", ns.closed)
    assert ns.lin.head is head and saved.kind == "closed_day"
    with pytest.raises(lcp.LifecycleCheckpointError, match="already exists"):
        save(ns, tmp_path / "ok", ns.closed)  # no overwrite
    saved2 = save(ns, tmp_path / "ok2", ns.closed)  # retry/duplicate: the same head, the same sealed digest
    assert saved2.manifest_sha256 == saved.manifest_sha256


def test_an_occurrence_and_a_pending_storm_are_not_saveable(tmp_path):
    from bridge_builders import build

    owner, bed, factory = build(tmp_path / "a", "numpy", False)
    lin, occ, _st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    with pytest.raises(lcp.LifecycleCheckpointError, match="Occurrence|saveable"):
        lcp.save_floating_lifecycle_checkpoint(tmp_path / "x", lin, factory, IDENT, original_topology=occ.event._initial_topology, available_bytes_fn=AVAIL)
    ns = closed_day_ns(tmp_path / "b", "numpy", False, days=1)
    n = ns.closed.ctx.n_cells
    pl = plan(1, 0.01, 3600.0, 0.005)  # P > T: the FULL native storm is required
    pb = br.begin_storm_from_day(ns.lin, ns.closed, pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    assert pb.cursor.pending is not None
    with pytest.raises(lcp.LifecycleCheckpointError, match="pending"):
        lcp.save_floating_lifecycle_checkpoint(tmp_path / "y", ns.lin, ns.factory, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
    assert not (tmp_path / "y").exists()


def test_the_expected_seal_is_required_and_binds_the_written_record(cday, tmp_path):
    ns, saved, _t = cday
    ck = saved.path
    for seal, rx in ((None, "REQUIRED"), ("0" * 64, "EXPECTED"), ("zz", "hex|64")):
        with pytest.raises(REFUSE, match=rx):
            lcp.load_floating_lifecycle_checkpoint(ck, ns.factory, IDENT, seal, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL)
    with pytest.raises(REFUSE, match="identity"):
        lcp.load_floating_lifecycle_checkpoint(ck, ns.factory, {"case": "other"}, saved.manifest_sha256, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL)
    # a resealed file carries its OWN valid seal but is not the one the writer returned
    d, new_seal = _forge(ck, tmp_path, "seq", lambda m: m["lifecycle"]["head"].update(seq=m["lifecycle"]["head"]["seq"] + 5))
    with pytest.raises(REFUSE, match="EXPECTED"):
        lcp.load_floating_lifecycle_checkpoint(d, ns.factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL)
    # a partition/boundary mismatch of the caller's arguments is refused before any payload work
    with pytest.raises(REFUSE, match="partition"):
        lcp.load_floating_lifecycle_checkpoint(ck, ns.factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=None, available_bytes_fn=AVAIL)
    assert new_seal != saved.manifest_sha256


def test_resealed_lifecycle_tampers_are_refused_with_their_own_valid_seal(cday, tmp_path):
    ns, saved, _t = cday
    ck = saved.path

    def daily(**kw):
        return lambda m: m["lifecycle"]["daily"].update(**kw)

    D = json.loads((ck / lcp.MANIFEST_NAME).read_text())["lifecycle"]["daily"]
    cases = {
        "rev": (daily(revision=D["revision"] + 1), "digest"),
        "recv": (lambda m: m["lifecycle"]["daily"]["receipt"].update(kind="daily"), "digest|receipt kind"),
        "ledger": (lambda m: m["lifecycle"]["daily"]["ledger"].update(days_applied=D["ledger"]["days_applied"] + 1), "digest"),
        "cursor": (lambda m: m["lifecycle"]["daily"]["cursor"].update(ordinal=D["cursor"]["ordinal"] + 1), "digest|cursor"),
        "prev": (daily(prev_execution="1" * 64), "digest"),
        "parent": (daily(parent_digest="2" * 64), "parent digest"),
        "kind_origin": (lambda m: m["lifecycle"]["daily"]["receipt"].update(kind="bridge_start"), "origin|digest"),
        "clock": (daily(t_s=D["t_s"] + 1.0), "clock"),
        "rain": (lambda m: m["lifecycle"]["head"]["ledger"].update(rain_m3=m["lifecycle"]["head"]["ledger"]["rain_m3"] + 1.0), "does not close|identity"),
        "ledger_keys": (lambda m: m["lifecycle"]["head"]["ledger"].update(extra=1.0), "keys"),
        "kind": (lambda m: m["lifecycle"].update(kind="boundary"), "boundary record|keys|kind"),
        "route": (lambda m: m["lifecycle"]["routing"].update(generation=m["lifecycle"]["routing"]["generation"] + 1), "routing"),
        "bridge": (lambda m: m["lifecycle"]["bridge"].update(stable_sha256="3" * 64), "differs|bridge"),
        "bridge_gen": (lambda m: m["lifecycle"]["bridge"].update(property_generation=99), "bridge|generation"),
        "prov": (lambda m: m["lifecycle"]["head"]["provenance"].update(cold_anchor_witness="4" * 8), "provenance"),
    }
    for name, (edit, rx) in cases.items():
        d, seal = _forge(ck, tmp_path, name, edit)
        with pytest.raises(REFUSE, match=rx):
            lcp.load_floating_lifecycle_checkpoint(d, ns.factory, IDENT, seal, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL)
    # a changed caller partition is another case: the recomputed context identity differs from the written one
    other = se.EvaporationPartition(1.3, 0.5, 1.0, 0.125, 0.75)
    with pytest.raises(REFUSE, match="differs"):
        lcp.load_floating_lifecycle_checkpoint(ck, ns.factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=other, available_bytes_fn=AVAIL)
    r = load(ns, ck, saved, partition=PART)  # the untouched file still restores after every refusal
    assert r.closed.execution_digest == ns.closed.execution_digest


def test_payload_tampers_extra_files_and_wrong_original_are_refused_before_adoption(cday, tmp_path):
    ns, saved, _t = cday
    ck = saved.path
    for rel in ("context_arrays.npz", "child/pond_continuation.npz", "child/" + fc.MANIFEST_NAME):
        d = tmp_path / rel.replace("/", "_")
        shutil.copytree(ck, d)
        raw = bytearray((d / rel).read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        (d / rel).write_bytes(bytes(raw))
        with pytest.raises(REFUSE):
            lcp.load_floating_lifecycle_checkpoint(d, ns.factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL)
    d = tmp_path / "extra"
    shutil.copytree(ck, d)
    (d / "stray").write_text("x")
    with pytest.raises(REFUSE, match="exactly"):
        lcp.load_floating_lifecycle_checkpoint(d, ns.factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL)
    other = dataclasses.replace(ns.original, source_z_full=np.array(ns.original.source_z_full) + 1e-3, source_sha256="0" * 64)
    with pytest.raises(REFUSE):
        lcp.load_floating_lifecycle_checkpoint(ck, ns.factory, IDENT, saved.manifest_sha256, original_topology=other, partition=PART, available_bytes_fn=AVAIL)
    with pytest.raises(REFUSE, match="source pair"):
        mp = pytest.MonkeyPatch()
        try:
            mp.setattr(pc, "pond_checkpoint_source_pair", lambda: {"maple_syrup": "0" * 64, "maple": "1" * 64})
            lcp.load_floating_lifecycle_checkpoint(ck, ns.factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL)
        finally:
            mp.undo()


def test_composed_admission_precedes_every_payload_read_and_save_is_refused_before_the_carrier(cday, tmp_path, monkeypatch):
    ns, saved, _t = cday
    need = lcp.lifecycle_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=AVAIL)
    assert need > 0

    def trip(*a, **k):
        raise AssertionError("a payload was read before the composed admission")

    monkeypatch.setattr(pc, "_hash_file", trip)
    monkeypatch.setattr(np, "load", trip)
    with pytest.raises(REFUSE):
        lcp.load_floating_lifecycle_checkpoint(saved.path, ns.factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=PART, available_bytes_fn=lambda: need // 2)
    monkeypatch.undo()
    # save: the stage-0 reader refuses before any vertical; a reader that is fine until the vertical exists refuses the COLD carrier before `continued_state` allocates it
    state = {"after": False}
    real_from = fp.FloatingPondVertical.from_boundary.__func__

    def spy(cls, *a, **k):
        v = real_from(cls, *a, **k)
        state["after"] = True
        return v

    monkeypatch.setattr(fp.FloatingPondVertical, "from_boundary", classmethod(spy))

    def reader():
        return 1 << 40 if not state["after"] else 1 << 30  # plenty for the vertical/H0; the carrier + child model needs more than 1 GiB only if the bed is large

    def carrier_trip(self, *a, **k):
        raise AssertionError("the carrier was built after a failed admission")

    monkeypatch.setattr(PondEvent, "continued_state", carrier_trip)
    state["after"] = False
    with pytest.raises((AssertionError, *REFUSE)):
        lcp.save_floating_lifecycle_checkpoint(tmp_path / "s1", ns.lin, ns.factory, IDENT, closed=ns.closed, original_topology=ns.original, available_bytes_fn=lambda: 1 if state["after"] else 1 << 40)
    assert not (tmp_path / "s1").exists() and not list(tmp_path.glob(".s1.partial-*"))
    monkeypatch.undo()
    with pytest.raises(REFUSE):
        save(ns, tmp_path / "s0", None) if False else lcp.save_floating_lifecycle_checkpoint(tmp_path / "s0", ns.lin, ns.factory, IDENT, closed=ns.closed, original_topology=ns.original, available_bytes_fn=lambda: 1)
    assert not (tmp_path / "s0").exists() and not list(tmp_path.glob(".s0.partial-*"))


def test_unpublished_windowed_bed_is_refused_before_the_carrier_or_any_deep_read(tmp_path, monkeypatch):
    from maple.surface.voxels.deep_store import DeepJournal
    from maple.surface.voxels.windowed_pending import attach_pending_journal

    ns = boundary_ns(tmp_path / "a", "numpy", True)
    col = ns.lin.head.bed.voxel_column
    nc = int(col.deep_store.header.n_classes)
    j = DeepJournal.from_records(np.array([0]), np.array([int(col.n_voxels) - 1]), np.zeros((1, nc)), header=col.deep_store.header)
    attach_pending_journal(col, j)  # a REAL unpublished MAPLE journal marker (restored exactly below)
    calls = []
    real_from = fp.FloatingPondVertical.from_boundary.__func__
    monkeypatch.setattr(fp.FloatingPondVertical, "from_boundary", classmethod(lambda cls, *a, **k: calls.append(1) or real_from(cls, *a, **k)))
    try:
        with pytest.raises(REFUSE, match="published|journal"):
            save(ns, tmp_path / "ck")
        assert not calls and not (tmp_path / "ck").exists() and not list(tmp_path.glob(".ck.partial-*"))
    finally:
        object.__setattr__(col, "pending_deep", None)
    assert save(ns, tmp_path / "ck2").kind == "boundary"  # published again: the same save works


def test_parent_boundary_may_be_released_and_a_restored_closed_day_is_saveable_again(tmp_path):
    ns = closed_day_ns(tmp_path / "a", "numpy", False, days=1)
    ref = weakref.ref(ns.b)
    ns.b = None
    del ns.origin
    gc.collect()
    assert ref() is None  # the old parent Boundary is garbage: a save does not need it (nor retain it)
    saved = save(ns, tmp_path / "ck", ns.closed)
    r = load(ns, tmp_path / "ck", saved, partition=PART)
    again = lcp.save_floating_lifecycle_checkpoint(tmp_path / "ck2", r.lineage, ns.factory, IDENT, closed=r.closed, original_topology=ns.original, available_bytes_fn=AVAIL)  # a restored hold is saveable (dead parent slot)
    r2 = lcp.load_floating_lifecycle_checkpoint(tmp_path / "ck2", ns.factory, IDENT, again.manifest_sha256, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL)
    assert (r2.closed.execution_digest, r2.closed.digest) == (ns.closed.execution_digest, ns.closed.digest)
    assert run_day is not None


def test_the_published_need_counts_the_reserve_once_and_includes_the_windowed_native_cache_bound(tmp_path):
    ns = closed_day_ns(tmp_path / "a", "numpy", True, days=1)
    saved = save(ns, tmp_path / "ck", ns.closed)
    n0 = lcp.lifecycle_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=AVAIL)
    n1 = lcp.lifecycle_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=AVAIL, reserve_bytes=12345)
    assert n1 - n0 == 12345  # the caller's reserve is counted exactly once in the composed model
    cman = json.loads((saved.path / "child" / fc.MANIFEST_NAME).read_text())
    meta = json.loads((saved.path / lcp.MANIFEST_NAME).read_text())["context_meta"]
    bound = fc.deep_cache_bound_bytes(cman["deep_storage"], meta["n_classes"])
    assert bound > 0 and n0 >= bound  # the ACTUAL native mapping bound of the deep store is inside the composed need (never a compressed-file proxy)
    assert cman.get("deep_storage") and cman.get("native_dense") is None


def test_boundary_restore_continues_the_actual_bed_physics_like_the_uninterrupted_chain(tmp_path):
    from lcp_builders import storm

    ns = boundary_ns(tmp_path / "a", "numpy", False)
    head = ns.lin.head
    saved = save(ns, tmp_path / "ck")
    r = load(ns, tmp_path / "ck", saved)
    n1, n2 = storm(ns.lin, head, ns.factory), storm(r.lineage, r.head, ns.factory)
    c1, c2 = n1.bed.voxel_column, n2.bed.voxel_column
    assert np.array_equal(np.asarray(c1.mass_kg), np.asarray(c2.mass_kg)) and np.array_equal(np.asarray(n1.bed.water.depth_m), np.asarray(n2.bed.water.depth_m))
    assert np.array_equal(np.asarray(n1.root.history.cumulative_intake_m), np.asarray(n2.root.history.cumulative_intake_m)) and n1.ledger == n2.ledger


def _spy_load(monkeypatch, ns, ck, saved, partition):
    """Load while recording WEAK references to the transient carrier (event, state, phase arrays, ledgers) and the token."""
    seen = {}
    real = lc.verified_restore

    def spy(event, state, seal):
        tok = real(event, state, seal)
        seen.update(event=weakref.ref(event), state=weakref.ref(state), frac=weakref.ref(state.phase_fraction), pos=weakref.ref(state.phase_position_m), water=weakref.ref(state.water),
                    sediment=weakref.ref(state.sediment), token=weakref.ref(tok), vert=weakref.ref(event.vertical))
        seen["_install"] = tok  # a strong ref only to prove one-shot; dropped by the caller
        return tok

    monkeypatch.setattr(lc, "verified_restore", spy)
    r = load(ns, ck, saved, partition=partition)
    monkeypatch.undo()
    return r, seen


@pytest.mark.parametrize("kind", ["boundary", "closed_day"])
def test_the_transient_carrier_is_released_after_load_while_the_result_stays_alive_and_works(tmp_path, monkeypatch, kind):
    from lcp_builders import storm

    ns = boundary_ns(tmp_path / "a", "numpy", False) if kind == "boundary" else closed_day_ns(tmp_path / "a", "numpy", False, days=2)
    closed = None if kind == "boundary" else ns.closed
    saved = save(ns, tmp_path / "ck", closed)
    r, seen = _spy_load(monkeypatch, ns, tmp_path / "ck", saved, PART if closed is not None else None)
    tok = seen.pop("_install")
    with pytest.raises(lc.LifecycleError, match="one-shot|already"):
        lc.install_restored_head(tok, lc.FloatingEventLineage(), r.lineage.head)  # a consumed token cannot open a second lineage
    del tok
    gc.collect()
    dead = {k: v() is None for k, v in seen.items()}
    assert all(dead.values()), dead  # event, state, phase arrays, ledgers, token and the transient vertical are all garbage; the RESULT (lineage/head/closed) is still held here
    assert set(RestoredFields()) == set(r.__dataclass_fields__) and not hasattr(r, "event") and not hasattr(r, "state")
    # the current physical state is intact and continues: the digest/frame identity (closed) and the next native event / next day
    if kind == "closed_day":
        assert (r.closed.execution_digest, r.closed.digest, r.closed.frame_identity()) == (ns.closed.execution_digest, ns.closed.digest, ns.closed.frame_identity())
        b = br.hand_back(r.lineage, r.closed)
        s2 = br.open_day(r.lineage, b, partition=PART, available_bytes_fn=AVAIL)
        c2 = run_day(s2, 2).state
        assert c2.t_s > b.t_s
        b = br.hand_back(r.lineage, c2)
    else:
        b = r.head
    n1 = storm(r.lineage, b, ns.factory)  # the next native event opens and completes on the restored head alone
    assert n1.ledger["n_occurrences"] >= 2 and abs(n1.ledger["residual_m3"]) <= n1.ledger["residual_bound_m3"]


def RestoredFields():
    return ("lineage", "head", "closed", "kind", "manifest_sha256")
