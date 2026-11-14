"""Vegetation S1 controls: owning save eligibility, no change on save, expected seal, resealed tampers, caller LEAF pins, composed admission, retry/duplicate/stale installs, weak lifetimes and the MUTUAL refusal with the old
S1 profile. REAL native lifecycle states and the ACTUAL LEAF projection only (no fake source, producer, owner, projection or head)."""
from __future__ import annotations

import copy
import gc
import json
import shutil
import weakref
from pathlib import Path

import numpy as np
import pytest
from vck_builders import (
    AVAIL,
    BIG,
    IDENT,
    PART,
    boundary_vns,
    closed_day_vns,
    load,
    pins,
    save,
)

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as fcc
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_lifecycle_checkpoint as lcp
from maple_syrup import floating_lifecycle_checkpoint_vegetation as lcv
from maple_syrup import native_floating_soil as nfs
from maple_syrup import pond_checkpoint as pc
from maple_syrup import vegetation_lifecycle as vl

REFUSE = (lcv.VegetationLifecycleCheckpointError, lcp.LifecycleCheckpointError, fc.FloatingCheckpointError, pc.PondCheckpointError, nfs.NativeFloatingMemoryError)


def _forge(src: Path, tmp: Path, name: str, edit) -> tuple:
    """Copy the container, edit the OUTER manifest, RESEAL it (a recomputed seal is not a proof): `(dir, new_seal)`."""
    d = tmp / name
    shutil.copytree(src, d)
    m = json.loads((d / lcv.MANIFEST_NAME).read_text())
    edit(m)
    sealed = fcc._seal(m)
    (d / lcv.MANIFEST_NAME).write_text(pc._json_dumps(sealed))
    return d, sealed["manifest_sha256"]


@pytest.fixture(scope="module")
def cday(tmp_path_factory, leaf):
    tmp = tmp_path_factory.mktemp("cday")
    ns = closed_day_vns(tmp / "a", leaf, "numpy", False, days=2)
    saved = save(ns, tmp / "ck", ns.closed)
    return ns, saved, tmp


@pytest.fixture(scope="module")
def bnd(tmp_path_factory, leaf):
    tmp = tmp_path_factory.mktemp("bnd")
    ns = boundary_vns(tmp / "a", leaf, "numpy", False)
    saved = save(ns, tmp / "ck")
    return ns, saved, tmp


def test_old_save_refuses_a_vegetation_head_and_the_new_save_refuses_a_plain_head(tmp_path, leaf):
    ns = boundary_vns(tmp_path / "a", leaf, "numpy", False)
    head = ns.lin.head
    with pytest.raises(lcp.LifecycleCheckpointError, match="vegetation-bound head is not saveable"):
        lcp.save_floating_lifecycle_checkpoint(tmp_path / "old", ns.lin, ns.factory, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
    assert not (tmp_path / "old").exists() and ns.lin.head is head
    with pytest.raises(lc.LifecycleError, match="does not support a vegetation-bound boundary"):
        lc.export_boundary_held_context(head, available_bytes_fn=AVAIL)  # the old H0 export stays strict and LEAF-free
    from maple_syrup import floating_held_context as fh
    from maple_syrup import floating_pond as fp

    vert = fp.FloatingPondVertical.from_boundary(head.root, available_bytes_fn=AVAIL)
    with pytest.raises(fh.HeldContextError, match="per-cell vegetation factor"):
        fh.export_held_context(vert, None, available_bytes_fn=AVAIL)
    assert ns.lin.head is head
    plain = boundary_vns(tmp_path / "b", leaf, "numpy", False, apply=False)
    with pytest.raises(lcv.VegetationLifecycleCheckpointError, match="vegetation-bound head only"):
        save(plain, tmp_path / "new")
    assert not (tmp_path / "new").exists() and not list(tmp_path.glob(".new.partial-*"))
    with pytest.raises(lc.LifecycleError, match="bool"):
        lc.saveable_head(ns.lin, vegetation=1)


def test_save_authority_refusals_retry_and_no_overwrite(tmp_path, leaf):
    ns = closed_day_vns(tmp_path / "a", leaf, "numpy", False, days=1)
    head, rec = ns.lin.head, vl.current_record(ns.lin.head)
    other = closed_day_vns(tmp_path / "b", leaf, "numpy", False, days=1)
    for closed, rx in ((None, "closed"), (other.closed, "genuine closed day")):
        with pytest.raises(lcv.VegetationLifecycleCheckpointError, match=rx):
            save(ns, tmp_path / "x", closed)
    assert not (tmp_path / "x").exists() and not list(tmp_path.glob(".x.partial-*")) and ns.lin.head is head
    saved = save(ns, tmp_path / "ok", ns.closed)
    assert ns.lin.head is head and vl.current_record(ns.lin.head) is rec and saved.kind == "closed_day" and saved.generation == 1 and saved.recipe_id == rec.recipe_id
    with pytest.raises(lcv.VegetationLifecycleCheckpointError, match="already exists"):
        save(ns, tmp_path / "ok", ns.closed)
    assert save(ns, tmp_path / "ok2", ns.closed).manifest_sha256 == saved.manifest_sha256  # a retry of the same head is the same sealed file


def test_mutual_refusal_between_the_old_and_the_vegetation_profiles(cday, tmp_path, leaf):
    ns, saved, _t = cday
    with pytest.raises(lcp.LifecycleCheckpointError, match="exactly the manifest"):
        lcp.load_floating_lifecycle_checkpoint(saved.path, ns.factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL)
    from lcp_builders import closed_day_ns
    from lcp_builders import save as old_save

    old = closed_day_ns(tmp_path / "o", "numpy", False, days=1)
    osaved = old_save(old, tmp_path / "oldck", old.closed)
    with pytest.raises(lcv.VegetationLifecycleCheckpointError, match="exactly the vegetation manifest"):
        load(ns, osaved.path, osaved, partition=PART)
    from maple_syrup import floating_held_context as fh
    from maple_syrup import floating_held_context_vegetation as fhv

    assert fh.SCHEMA == "maple_syrup.floating_held_context/1" and fhv.SCHEMA3 == "maple_syrup.floating_held_context/3"


def test_the_expected_seal_is_required_and_resealed_data_is_not_a_proof(cday, tmp_path):
    ns, saved, _t = cday
    ck = saved.path
    for seal, rx in ((None, "REQUIRED"), ("0" * 64, "EXPECTED"), ("zz", "hex|64")):
        with pytest.raises(REFUSE, match=rx):
            lcv.load_floating_lifecycle_vegetation_checkpoint(ck, ns.factory, IDENT, seal, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL, memory_budget_bytes=BIG, **pins(ns))
    with pytest.raises(REFUSE, match="identity"):
        lcv.load_floating_lifecycle_vegetation_checkpoint(ck, ns.factory, {"case": "other"}, saved.manifest_sha256, original_topology=ns.original, partition=PART, available_bytes_fn=AVAIL,
                                                          memory_budget_bytes=BIG, **pins(ns))
    d, new_seal = _forge(ck, tmp_path, "gen", lambda m: m["vegetation"].update(generation=m["vegetation"]["generation"] + 1))
    with pytest.raises(REFUSE, match="EXPECTED"):
        load(ns, d, saved, partition=PART)  # the writer's seal is expected: a re-sealed file is a different file
    with pytest.raises(REFUSE, match="recipe id|parent"):
        load(ns, d, type(saved)(d, new_seal, saved.kind, 2, "x"), partition=PART)  # even with its OWN seal presented, the stable counter / recipe id must reproduce
    with pytest.raises(REFUSE, match="partition"):
        load(ns, saved.path, saved, partition=None)


def test_resealed_vegetation_block_and_lifecycle_tampers_are_refused(cday, tmp_path):
    ns, saved, _t = cday
    ck = saved.path
    M = json.loads((ck / lcv.MANIFEST_NAME).read_text())
    V = M["vegetation"]
    cases = {
        "recipe": (lambda m: m["vegetation"].update(recipe_id="1" * 64), "recipe id"),
        "parent": (lambda m: m["vegetation"].update(parent_recipe_id="2" * 64), "parent"),
        "binding": (lambda m: m["vegetation"].update(binding_sha256="3" * 64), "differs from the recipe|recipe id"),
        "projection": (lambda m: m["vegetation"].update(projection_sha256="4" * 64), "differs from the recipe"),
        "cached_rec": (lambda m: m["vegetation"].update(cached_record_sha256="5" * 64), "differs from the recipe"),
        "controls": (lambda m: m["vegetation"].update(controls_sha256="6" * 64), "differs from the recipe"),
        "mask": (lambda m: m["vegetation"].update(mask_sha256="7" * 64), "differs from the recipe"),
        "surface": (lambda m: m["vegetation"].update(surface_sha256="8" * 64), "differs from the recipe"),
        "source": (lambda m: m["vegetation"]["source"].update(state_sha256="9" * 64), "differs from the recipe"),
        "helpers": (lambda m: m["vegetation"].update(external_helpers=[]), "differs from the recipe"),
        "pkg": (lambda m: m["vegetation"].update(leaf_package_sha256="a" * 64), "differs from the recipe"),
        "gen0": (lambda m: m["vegetation"].update(generation=0), "positive int"),
        "genbool": (lambda m: m["vegetation"].update(generation=True), "positive int"),
        "genfloat": (lambda m: m["vegetation"].update(generation=1.0), "positive int"),
        "gen1parent": (lambda m: m["vegetation"].update(parent_recipe_id="b" * 64), "no parent"),
        "keys": (lambda m: m["vegetation"].update(extra=1), "keys"),
        "applied_keys": (lambda m: m["vegetation"]["applied"].update(extra="c" * 64), "applied"),
        "applied_hex": (lambda m: m["vegetation"]["applied"].update(law_sha256="zz"), "hex|64"),
        "origin": (lambda m: m["vegetation"].update(origin_at_save="foreign"), "origin"),
        "no_veg": (lambda m: m.pop("vegetation"), "keys"),
        "schema_old": (lambda m: m.update(context_schema="maple_syrup.floating_held_context/1"), "unsupported"),
        "schema_cuda": (lambda m: m.update(context_schema="maple_syrup.floating_held_context/4"), "unsupported"),
        "profile": (lambda m: m.update(profile=lcp.PROFILE), "unknown vegetation lifecycle profile"),
        "source_pair": (lambda m: m["source"].update(maple_syrup="0" * 64) if isinstance(m["source"], dict) else None, "source pair|keys"),
        "kernel": (lambda m: m["lifecycle"]["head"]["provenance"].update(cold_anchor_witness="4" * 8), "provenance"),
        "routing": (lambda m: m["lifecycle"]["routing"].update(generation=m["lifecycle"]["routing"]["generation"] + 1), "routing"),
        "ledger": (lambda m: m["lifecycle"]["head"]["ledger"].update(rain_m3=m["lifecycle"]["head"]["ledger"]["rain_m3"] + 1.0), "does not close|identity"),
        "cursor": (lambda m: m["lifecycle"]["daily"]["cursor"].update(ordinal=m["lifecycle"]["daily"]["cursor"]["ordinal"] + 1), "digest|cursor"),
        "veg_array_shape": (lambda m: m["context_arrays"]["veg.splash_cover"].update(shape=[1]), "declared dtype/shape|array table"),
        "veg_array_dtype": (lambda m: m["context_arrays"]["veg.splash_cover"].update(dtype="<f4"), "declared dtype/shape|array table"),
        "veg_array_drop": (lambda m: m["context_arrays"].pop("veg.splash_cover"), "names differ|array table"),
        "child_schema": (lambda m: m["child"].update(schema="x"), "child schema"),
        "meta_cached_record": (lambda m: m["context_meta"]["vegetation"]["cached"].update(record_sha256="d" * 64), "differs from the recipe|invalid"),
        "meta_law": (lambda m: m["context_meta"]["vegetation"]["law"].update(law_sha256="e" * 64), "law|identit|cannot be restored"),
    }
    assert V["generation"] == 1
    for name, (edit, rx) in cases.items():
        d, seal = _forge(ck, tmp_path, name, edit)
        with pytest.raises(REFUSE, match=rx):
            load(ns, d, type(saved)(d, seal, saved.kind, 1, V["recipe_id"]), partition=PART)
        assert not list(Path(d).glob(".*partial*"))


def test_tampered_payload_bytes_and_child_files_are_refused(cday, tmp_path):
    ns, saved, _t = cday
    for rel in ("context_arrays.npz", "child/" + fc.MANIFEST_NAME):
        d = tmp_path / ("t_" + rel.replace("/", "_"))
        shutil.copytree(saved.path, d)
        p = d / rel
        raw = bytearray(p.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        p.write_bytes(bytes(raw))
        with pytest.raises(REFUSE):
            load(ns, d, saved, partition=PART)
    d = tmp_path / "extra"
    shutil.copytree(saved.path, d)
    (d / "stray").write_text("x")
    with pytest.raises(REFUSE, match="exactly"):
        load(ns, d, saved, partition=PART)


def test_the_callers_leaf_pins_are_external_and_checked(cday):
    ns, saved, _t = cday
    good = pins(ns)
    with pytest.raises(REFUSE, match="pins differ"):
        load(ns, saved.path, saved, partition=PART, leaf_package_sha256="0" * 64, external_helpers=good["external_helpers"])
    wrong = [{**good["external_helpers"][0], "sha256": "sha256:" + "0" * 64}]
    with pytest.raises(REFUSE, match="pins differ"):
        load(ns, saved.path, saved, partition=PART, leaf_package_sha256=good["leaf_package_sha256"], external_helpers=wrong)
    with pytest.raises(REFUSE, match="pins are malformed"):
        load(ns, saved.path, saved, partition=PART, leaf_package_sha256=good["leaf_package_sha256"], external_helpers=[])
    with pytest.raises(REFUSE, match="pins are malformed"):
        load(ns, saved.path, saved, partition=PART, leaf_package_sha256="zz", external_helpers=good["external_helpers"])
    with pytest.raises(REFUSE, match="pins are malformed"):
        load(ns, saved.path, saved, partition=PART, leaf_package_sha256=good["leaf_package_sha256"], external_helpers="x")


def test_composed_admission_reserve_budget_and_readers(cday):
    ns, saved, _t = cday
    need = lcv.vegetation_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=AVAIL)
    assert need > 0 and lcv.vegetation_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=AVAIL, reserve_bytes=1000) == need + 1000  # the reserve is counted ONCE
    with pytest.raises(nfs.NativeFloatingMemoryError):
        load(ns, saved.path, saved, partition=PART, available_bytes_fn=lambda: need - 1)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        load(ns, saved.path, saved, partition=PART, available_bytes_fn=lambda: need - 1001, reserve_bytes=1000)
    with pytest.raises(REFUSE):
        load(ns, saved.path, saved, partition=PART, memory_budget_bytes=1024)
    with pytest.raises(REFUSE, match="memory_budget_bytes"):
        load(ns, saved.path, saved, partition=PART, memory_budget_bytes=None)
    with pytest.raises(REFUSE, match="memory_budget_bytes"):
        load(ns, saved.path, saved, partition=PART, memory_budget_bytes=True)
    r = load(ns, saved.path, saved, partition=PART, available_bytes_fn=lambda: need)  # the published need reaches and finishes the load
    assert r.kind == "closed_day"
    with pytest.raises(nfs.NativeFloatingMemoryError):
        save(ns, saved.path.parent / "q", ns.closed, available_bytes_fn=lambda: 1)
    assert not (saved.path.parent / "q").exists() and not list(saved.path.parent.glob(".q.partial-*"))


def test_save_is_bit_exact_non_mutating(tmp_path, leaf):
    ns = closed_day_vns(tmp_path / "a", leaf, "numpy", False, days=1)
    h, c = ns.lin.head, ns.closed
    snap = (np.array(c.water.water_m), np.array(c.history.wetting_m), np.array(c.history.cumulative_intake_m), dict(h.ledger), h.t_s, h.seq, c.digest, c.execution_digest, c.cursor)
    save(ns, tmp_path / "ck", c)
    save(ns, tmp_path / "ck2", c)
    assert ns.lin.head is h and (np.array_equal(snap[0], c.water.water_m) and np.array_equal(snap[1], c.history.wetting_m) and np.array_equal(snap[2], c.history.cumulative_intake_m)
                                  and snap[3] == h.ledger and snap[4] == h.t_s and snap[5] == h.seq and snap[6] == c.digest and snap[7] == c.execution_digest and snap[8] == c.cursor)


def test_duplicate_stale_and_forged_restores_are_refused_before_any_head_moves(bnd, tmp_path):
    ns, saved, _t = bnd
    r = load(ns, saved.path, saved)
    # a restored head cannot be re-installed on another lineage; the restored vertical opened exactly one lineage
    with pytest.raises(lc.LifecycleError, match="not a genuine VerifiedRestore"):
        lc.install_restored_head(object(), lc.FloatingEventLineage(), r.head)
    rec = vl.current_record(r.head)
    assert rec.binding.origin == "cached_derived" and vl.verify_record(rec) is rec
    # a record of one restore is not accepted by another restore's held recipe (stale/foreign record)
    seen = {}
    orig = lc.stage_restored_head

    def spy(token, kind, head_rec, hold_rec=None, **kw):
        seen["rec"] = kw.get("vegetation")
        return orig(token, kind, head_rec, hold_rec, **kw)

    lc.stage_restored_head = spy
    try:
        load(ns, saved.path, saved)
    finally:
        lc.stage_restored_head = orig
    stale = seen["rec"]
    other_held = {}
    orig_v = lc.verified_restore

    def spy2(event, state, seal):
        tok = orig_v(event, state, seal)
        other_held["tok"] = tok
        return tok

    lc.verified_restore = spy2
    orig_stage = lc.stage_restored_head

    def refuse_stage(token, kind, head_rec, hold_rec=None, **kw):
        with pytest.raises(lc.LifecycleError, match="cached-derived record of this restore|not genuine"):
            orig_stage(token, kind, head_rec, hold_rec, vegetation=stale)
        with pytest.raises(lc.LifecycleError, match="needs the vegetation record"):
            orig_stage(token, kind, head_rec, hold_rec)
        return orig_stage(token, kind, head_rec, hold_rec, **kw)

    lc.stage_restored_head = refuse_stage
    try:
        r3 = load(ns, saved.path, saved)
    finally:
        lc.verified_restore = orig_v
        lc.stage_restored_head = orig_stage
    assert r3.head is not None
    # forged records: a copy / replace / hand-built record is refused by the owning verifier
    for bad in (copy.copy, copy.deepcopy):
        with pytest.raises(vl.VegetationLifecycleError):
            bad(rec)
    with pytest.raises(vl.VegetationLifecycleError, match="minted only"):
        type(rec)(*[getattr(rec, f.name) for f in __import__("dataclasses").fields(rec)])


def test_a_restore_factory_refuses_a_foreign_token_and_wrong_counters(bnd):
    ns, saved, _t = bnd
    with pytest.raises(vl.VegetationLifecycleError, match="unconsumed verified-restore"):
        vl.restore_vegetation_record(object(), generation=1, recipe_id="a" * 64, parent_recipe_id=None, applied={}, max_bytes=BIG)
    V = json.loads((saved.path / lcv.MANIFEST_NAME).read_text())["vegetation"]
    good = {"generation": V["generation"], "recipe_id": V["recipe_id"], "parent_recipe_id": V["parent_recipe_id"], "applied": dict(V["applied"]), "max_bytes": BIG}
    seen = []
    orig = lc.stage_restored_head

    def spy(token, kind, head_rec, hold_rec=None, **kw):  # the genuine token is still UNCONSUMED here: the strict counter checks are reachable
        for patch, rx in (({"generation": 0}, "positive int"), ({"generation": True}, "positive int"), ({"generation": 2}, "parent|reproduced"), ({"recipe_id": "0" * 64}, "recipe id"),
                          ({"recipe_id": "zz"}, "64-hex"), ({"parent_recipe_id": "a" * 64}, "no parent"), ({"applied": {}}, "closed key set"),
                          ({"applied": {**V["applied"], "g_sha256": "zz"}}, "64-hex")):
            with pytest.raises(vl.VegetationLifecycleError, match=rx):
                vl.restore_vegetation_record(token, **{**good, **patch})
        seen.append(1)
        return orig(token, kind, head_rec, hold_rec, **kw)

    lc.stage_restored_head = spy
    try:
        load(ns, saved.path, saved)
    finally:
        lc.stage_restored_head = orig
    assert seen == [1]


def test_weak_lifetimes_the_restored_head_needs_no_original_objects(tmp_path, leaf):
    ns = boundary_vns(tmp_path / "a", leaf, "numpy", False)
    saved = save(ns, tmp_path / "ck")
    r = load(ns, saved.path, saved)
    rec_ref, lin_ref = weakref.ref(vl.current_record(ns.lin.head)), weakref.ref(ns.lin)
    keep = r.head
    ns.lin = ns.b = None
    ns.stack = ns.binding = None
    gc.collect()
    assert rec_ref() is None and lin_ref() is None  # the original lineage / record / binding / projection are gone
    assert vl.verify_record(vl.current_record(keep)) is not None and np.isfinite(vl.current_record(keep).splash_cover).all()
    assert r.lineage.head is keep


def test_a_duplicate_install_of_one_restore_and_a_second_restore_of_the_same_vertical_are_refused(bnd):
    ns, saved, _t = bnd
    seen = {}
    orig = lc.install_restored_head

    def spy(token, lineage, head):
        orig(token, lineage, head)
        seen.update(token=token, lineage=lineage, head=head)
        with pytest.raises(lc.LifecycleError, match="fresh lineage|one-shot"):
            orig(token, lineage, head)  # the SAME token / head again (the lineage head is no longer None; the vertical is consumed)
        with pytest.raises(lc.LifecycleError, match="fresh lineage|one-shot"):
            orig(token, lc.FloatingEventLineage(), head)  # ... or on another fresh lineage

    lc.install_restored_head = spy
    try:
        r = load(ns, saved.path, saved)
    finally:
        lc.install_restored_head = orig
    assert seen["lineage"] is r.lineage and r.lineage.head is r.head
    # a second restore of the same vertical cannot stage another head
    with pytest.raises(lc.LifecycleError, match="one-shot"):
        lc.stage_restored_head(seen["token"], "boundary", lc.head_record(r.head), vegetation=vl.current_record(r.head))
