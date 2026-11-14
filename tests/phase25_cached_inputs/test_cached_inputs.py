"""Compact cached vegetation inputs: capture from genuine live bindings, cold staging from a closed record, strict tamper refusals, labelled origin, current-column preparation with EXTERNAL expected
operator ids. LEAF is the ACTUAL frozen H1 package (real PlantState / PlantSession / projection / H1-H2-H3): the consumer stays UNQUALIFIED until Root reconciles it."""
from __future__ import annotations

import copy
import gc
import json
import pickle
import weakref

import numpy as np
import pytest
from cached_support import (
    ARR,
    LEAF_PACKAGE,
    MAPLE_PACKAGE,
    NATIVE,
    collect,
    controls,
    expected,
    fingerprint,
    live,
    loadable,
    make_column,
    native_env,
    prepare,
    stage,
    wh,
    wvb,
)
from veg_binding_support import BIG

ERR = wvb.VegetationBindingError
MEM = wvb.VegetationBindingMemoryError


# ---------------------------------------------------------------- capture: genuine, immutable, lifetime
def test_capture_borrows_immutable_bytes_and_outlives_the_original_objects(leaf):
    p, b, surf, _m = live(leaf)
    wref = weakref.ref(p)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    before = {k: v.tobytes() for k, v in rec.arrays}
    sha, content = rec.record_sha256, rec.content_sha256
    del p, b, surf
    collect()
    assert wref.__call__() is None or gc.collect() >= 0                                  # the projection may be freed ...
    for k, v in rec.arrays:
        assert not v.flags.writeable and v.tobytes() == before[k]                         # ... the record keeps its own immutable bytes
        node = v
        while isinstance(node, np.ndarray):
            with pytest.raises(ValueError):
                node.setflags(write=True)
            node = node.base
        assert type(node) is bytes
    assert wh.verify_cached_inputs(rec) is rec and rec.record_sha256 == sha and rec.content_sha256 == content
    assert rec.origin == "cached_derived" and rec.live_session_proof is False


def test_capture_estimate_borrow_vs_copy_and_admission(leaf):
    p, b, _surf, _m = live(leaf)
    est0 = wh.estimate_capture_bytes_from_binding(b)
    estc = wh.estimate_capture_bytes(9, p.n_entries, 2, 3, 0, copy_arrays=True, meta_json_bytes=est0["meta_json_bytes"])
    assert est0["borrowed_bytes"] > 0 and estc["peak_bytes"] > est0["peak_bytes"] and est0["peak_bytes"] >= 3 * est0["meta_json_bytes"]
    with pytest.raises(MEM):
        wh.capture_cached_inputs(b, max_bytes=est0["peak_bytes"] - 1)
    with pytest.raises(MEM):
        wh.capture_cached_inputs(b, max_bytes=est0["peak_bytes"] + 4, reserve_bytes=5)
    with pytest.raises(MEM):
        wh.capture_cached_inputs(b, max_bytes=BIG, available_bytes_fn=lambda: est0["peak_bytes"] - 1)
    assert wh.capture_cached_inputs(b, max_bytes=est0["peak_bytes"]).n_cells == 9


def test_record_is_compact_no_s_by_n_or_p_by_n(leaf):
    from cached_support import catalog, plants_state, project
    cat = catalog(leaf, 3, [0, 1, 1], [0, 1, 1])
    rng = np.random.default_rng(3)
    pl = [(i, "grass" if i % 2 else "shrub", float(rng.uniform(0.5, 99.5)), float(rng.uniform(0.5, 99.5)), float(rng.uniform(0.2, 0.6)), "live") for i in range(1, 301)]
    st = plants_state(leaf, cat, pl, nx=100, ny=100)
    p, surf = project(leaf, st, cat)
    b = wvb.bind_projection(p, controls(n=10000), expected_leaf_package_sha256=__import__("veg_binding_support").leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"],
                            expected_source=__import__("veg_binding_support").full_source(p), expected_external_helpers=__import__("veg_binding_support").helper_pin(),
                            active_mask=np.ones((100, 100), dtype=bool), current_surface=surf, max_bytes=BIG)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    N, M, S, L = rec.n_cells, rec.n_entries, len(rec.species_ids), rec.root_layers
    assert max(a.size for _, a in rec.arrays) <= max(N + 1, M, S * L) and M < N * S                   # nothing S x N, nothing P x N (300 plants)
    assert sum(a.nbytes for _, a in rec.arrays) + rec.receiver_weights.nbytes <= rec.resident_bytes
    e = wh.export_cached_record(rec)
    assert "plant_id" not in json.dumps(e["meta"]) and "agb_dry_kg" not in json.dumps(e["meta"]) and all(a.size <= max(N + 1, M, S * L) for a in e["arrays"].values())


# ---------------------------------------------------------------- cold round trip
@pytest.mark.parametrize("layers,grass,shrub", [(3, (0, 1, 1), (0, 1, 1)), (2, (0, 1), (0, 0)), (1, (1,), (1,)), (3, (0, 0, 0), (0, 0, 0))])
def test_cold_roundtrip_reproduces_hash_fields_streams_and_operator_ids(leaf, layers, grass, shrub):
    p, b, _surf, m = live(leaf, layers=layers, grass=grass, shrub=shrub)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    snap = fingerprint(meta, arrays)
    ex = expected(p, b, rec)
    rec2 = stage(rec, p, b, meta, arrays, m)
    assert fingerprint(meta, arrays) == snap                                              # inputs never touched
    assert rec2.record_sha256 == rec.record_sha256 and rec2.binding_sha256 == b.sha256 and rec2.content_sha256 == ex["expected_projection_sha256"]
    assert rec2.mask_entry_guard_checked is True and rec.mask_entry_guard_checked is False
    for (k1, a1), (k2, a2) in zip(rec.arrays, rec2.arrays, strict=True):
        assert k1 == k2 and a1.tobytes() == a2.tobytes() and not a2.flags.writeable
    arrays["splash_cover"][:] = 0.5                                                       # mutating the loader array afterwards cannot reach the record
    assert dict(rec2.arrays)["splash_cover"].tobytes() == dict(rec.arrays)["splash_cover"].tobytes()
    col = make_column(layers)
    live_law = prepare(col, b)
    cb = wh.cached_binding(rec2, expected_binding_sha256=b.sha256, max_bytes=BIG)
    assert cb.origin == "cached_derived" and cb.sha256 == b.sha256
    law = prepare(col, cb, expected_matrix_op_sha256=live_law.matrix_sha256, expected_preferential_op_sha256=live_law.preferential_sha256)
    assert (law.law_sha256, law.g_sha256, law.matrix_sha256, law.preferential_sha256, law.n_contributions) == (live_law.law_sha256, live_law.g_sha256, live_law.matrix_sha256,
                                                                                                             live_law.preferential_sha256, live_law.n_contributions)
    assert law.binding_origin == "cached_derived" and live_law.binding_origin == "live_projection"
    assert wvb.verify_binding(cb) is cb and wvb.verify_prepared_law(law) is law


def test_cached_stream_equals_the_live_stream_and_numba_matches_numpy(leaf):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    ids_l, w_l, g_l = p.preferential_inputs(max_bytes=BIG)
    ids_c, w_c, g_c = wh.cached_preferential_inputs(rec, BIG, 0, None)
    assert ids_l == ids_c and w_l.tobytes() == w_c.tobytes()
    assert [x.tobytes() for x in g_l] == [x.tobytes() for x in g_c]
    meta, arrays = loadable(rec)
    pytest.importorskip("numba")
    r_np = stage(rec, p, b, meta, arrays, m, backend="numpy")
    r_nb = stage(rec, p, b, meta, arrays, m, backend="numba")
    assert r_np.record_sha256 == r_nb.record_sha256
    cb = wh.cached_binding(r_nb, expected_binding_sha256=b.sha256, max_bytes=BIG)
    assert prepare(make_column(backend="numba"), cb).law_sha256 == prepare(make_column(backend="numba"), b).law_sha256
    assert prepare(make_column(), cb).g_sha256 == prepare(make_column(backend="numba"), cb).g_sha256


def test_genuine_plant_session_binding_roundtrip(leaf):
    from test_seals_source_session import make_session, session_col, session_ctl
    from veg_binding_support import full_source, helper_pin, leaf_pin
    sess, _ = make_session()
    surf = np.zeros(sess.snapshot_state().domain.shape)
    p = leaf["wv"].project_from_plant_session(sess, surf, max_edges=1 << 16, max_bytes=BIG)
    n = surf.size
    b = wvb.bind_projection(p, session_ctl(n), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                            expected_external_helpers=helper_pin(), active_mask=np.ones(surf.shape, dtype=bool), current_surface=surf, session=sess, max_bytes=BIG)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    assert rec.source["kind"] == "plant_session" and rec.source["sequence"] == sess.sequence
    meta, arrays = loadable(rec)
    rec2 = stage(rec, p, b, meta, arrays, np.ones(surf.shape, dtype=bool))
    del p, b, sess
    collect()
    cb = wh.cached_binding(rec2, expected_binding_sha256=rec.binding_sha256, max_bytes=BIG)
    law = prepare(session_col(n), cb)
    assert law.n_contributions == 2 and law.binding_origin == "cached_derived"
    assert rec2.live_session_proof is False                                               # the recorded HELD identity is not a live PlantSession proof


# ---------------------------------------------------------------- tamper: every array, metadata key, pin, seal
def test_tamper_every_array_is_refused_and_the_input_is_unchanged_and_retry_works(leaf):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    for name in list(arrays):
        a = arrays[name]
        if a.size == 0:
            continue
        orig = a.copy()
        flat = a.reshape(-1)
        if a.dtype == np.float64:
            flat[0] = flat[0] + 1e-9 if flat[0] != 0 else 1e-9
        else:
            flat[0] = flat[0] + 1
        snap = fingerprint(meta, arrays)
        with pytest.raises(ERR):
            stage(rec, p, b, meta, arrays, m)
        assert fingerprint(meta, arrays) == snap, name
        arrays[name][...] = orig
    assert stage(rec, p, b, meta, arrays, m).record_sha256 == rec.record_sha256           # retry after the refusals succeeds


def _mut(meta, path, fn):
    node = meta
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = fn(node.get(path[-1]) if isinstance(node, dict) else node[path[-1]])


HEXFLIP = lambda s: s[:-1] + ("0" if s[-1] != "0" else "1")
META_TAMPER = [
    (("tag",), lambda v: v + "x"), (("version",), lambda v: "1"), (("version",), lambda v: 2), (("version",), lambda v: True), (("content_sha256",), HEXFLIP), (("binding_sha256",), HEXFLIP), (("record_sha256",), HEXFLIP),
    (("mask_sha256",), HEXFLIP), (("surface_sha256",), HEXFLIP), (("receiver_ids",), lambda v: v[:-1]), (("receiver_ids",), lambda v: [*v, "grass"]), (("receiver_ids",), lambda v: ["nope"]),
    (("compact", "schema"), lambda v: v + "x"), (("compact", "version"), lambda v: v + 1), (("compact", "field_order"), lambda v: v[::-1]), (("compact", "species_ids"), lambda v: v[::-1]),
    (("compact", "n_cells"), lambda v: v + 1), (("compact", "shape"), lambda v: v[::-1] if v[0] != v[1] else [1, 9]), (("compact", "cell_area_hex"), lambda v: (2.0).hex()),
    (("compact", "n_entries"), lambda v: v + 1), (("compact", "n_plants"), lambda v: v + 1), (("compact", "root_layers"), lambda v: v + 1), (("compact", "n_omitted"), lambda v: v + 1),
    (("controls", "a1"), lambda v: (0.02).hex()), (("controls", "a2"), lambda v: (0.9).hex()), (("controls", "a3"), lambda v: (0.9).hex()), (("controls", "a3_layout"), lambda v: "cell"),
    (("controls", "max_fraction_per_hour"), lambda v: (0.1).hex()), (("controls", "max_intake_rate_m_s"), lambda v: (2e-4).hex()), (("controls", "has_background"), lambda v: False),
    (("controls", "n_layers"), lambda v: v + 1), (("controls", "sha256"), HEXFLIP),
    (("source", "kind"), lambda v: "plant_session"), (("source", "state_sha256"), HEXFLIP), (("source", "catalog_hash"), HEXFLIP), (("source", "note"), lambda v: v + "x"),
    (("package", "digest"), lambda v: v[:-1] + ("0" if v[-1] != "0" else "1")), (("package", "file_count"), lambda v: v + 1),
    (("external_helpers",), lambda v: []), (("external_helpers", 0, "sha256"), HEXFLIP), (("external_helpers", 0, "bytes"), lambda v: v + 1),
    (("identity", "surface_sha256"), HEXFLIP), (("identity", "root_layers"), lambda v: v + 1), (("identity", "species_ids"), lambda v: ["grass", "shrub"]),
]


@pytest.mark.parametrize("path,fn", META_TAMPER, ids=["/".join(map(str, p)) for p, _ in META_TAMPER])
def test_tamper_every_metadata_field_is_refused(leaf, path, fn):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    _mut(meta, path, fn)
    snap = fingerprint(meta, arrays)
    with pytest.raises(ERR):
        stage(rec, p, b, meta, arrays, m)
    assert fingerprint(meta, arrays) == snap


def test_unknown_missing_wrong_type_keys_and_arrays_are_refused_before_anything_is_copied(leaf, monkeypatch):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    seals = []
    real = wh._seal_any
    monkeypatch.setattr(wh, "_seal_any", lambda a: seals.append(1) or real(a))
    meta, arrays = loadable(rec)
    cases = []
    for k in meta:
        d = copy.deepcopy(meta)
        del d[k]
        cases.append((d, arrays))
    d = copy.deepcopy(meta)
    d["extra"] = 1
    cases.append((d, arrays))
    d = copy.deepcopy(meta)
    d["compact"]["extra"] = 1
    cases.append((d, arrays))
    d = copy.deepcopy(meta)
    d["controls"]["extra"] = 1
    cases.append((d, arrays))
    for name in list(arrays):
        a2 = {k: v for k, v in arrays.items() if k != name}
        cases.append((meta, a2))
        a3 = dict(arrays)
        a3[name] = arrays[name].astype(np.float32 if arrays[name].dtype != np.float32 else np.float64)
        cases.append((meta, a3))
        a4 = dict(arrays)
        a4[name] = np.asfortranarray(arrays[name]) if arrays[name].ndim == 2 else arrays[name][::2] if arrays[name].size > 1 else arrays[name].reshape(-1, 1)
        cases.append((meta, a4))
    cases.append((meta, {**arrays, "extra": np.zeros(1)}))
    cases.append(([], arrays))
    cases.append((meta, [arrays]))
    for mt, ar in cases:
        with pytest.raises(ERR):
            stage(rec, p, b, mt, ar, m)
    assert seals == []                                                                    # the metadata gate precedes every copy


def test_nan_negative_weights_csr_and_root_fractions_refused_even_with_a_recomputed_content_hash(leaf):
    """The independent value checks hold when the tamperer also recomputes the content / binding / record hashes (honest H1 on altered data)."""
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    lw = leaf["wv"]
    for name, edit in (("root", lambda a: a.__setitem__((0, 0), a[0, 0] + 0.01)), ("weights", lambda a: a.__setitem__((0, 1), -1.0)), ("splash_cover", lambda a: a.__setitem__(0, float("nan"))),
                       ("cell_offsets", lambda a: a.__setitem__(1, a[1] + 1)), ("entry_species", lambda a: a.__setitem__(0, 7))):
        meta, arrays = loadable(rec)
        edit(arrays[name])
        compact = {"schema": meta["compact"]["schema"], "version": meta["compact"]["version"], "field_order": tuple(meta["compact"]["field_order"]), "arrays": {k: arrays[k] for k in ARR},
                   "species_ids": tuple(meta["compact"]["species_ids"]), "n_cells": 9, "shape": (3, 3), "cell_area_m2": float.fromhex(meta["compact"]["cell_area_hex"]),
                   "n_entries": meta["compact"]["n_entries"], "n_plants": meta["compact"]["n_plants"], "root_layers": 3}
        try:
            new = lw.compact_content_sha256(compact, max_bytes=BIG)
        except lw.VegetationProjectionError:
            new = meta["content_sha256"]                                                  # LEAF's H1 itself refuses the values: the stage must too
        meta["content_sha256"] = new
        meta["identity"]["content_sha256"] = new
        with pytest.raises(ERR):
            stage(rec, p, b, meta, arrays, m, expected_projection_sha256=new)
    # a stored cover that differs from its re-derivation is caught by H3 even when its hash is made consistent
    meta, arrays = loadable(rec)
    arrays["splash_cover"][0] = min(1.0, float(arrays["splash_cover"][0]) + 0.25)
    compact = {"schema": meta["compact"]["schema"], "version": meta["compact"]["version"], "field_order": tuple(meta["compact"]["field_order"]), "arrays": {k: arrays[k] for k in ARR},
               "species_ids": tuple(meta["compact"]["species_ids"]), "n_cells": 9, "shape": (3, 3), "cell_area_m2": float.fromhex(meta["compact"]["cell_area_hex"]),
               "n_entries": meta["compact"]["n_entries"], "n_plants": meta["compact"]["n_plants"], "root_layers": 3}
    new = lw.compact_content_sha256(compact, max_bytes=BIG)
    meta["content_sha256"] = new
    meta["identity"]["content_sha256"] = new
    with pytest.raises(ERR, match="re-derivation|differs"):
        stage(rec, p, b, meta, arrays, m, expected_projection_sha256=new)


def test_wrong_external_expectations_pins_and_source_keys_are_refused(leaf):
    from veg_binding_support import helper_pin
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    bad = "0" * 64
    for key in ("expected_record_sha256", "expected_binding_sha256", "expected_projection_sha256", "expected_leaf_package_sha256"):
        with pytest.raises(ERR):
            stage(rec, p, b, meta, arrays, m, **{key: bad})
    good = helper_pin()
    for hp in ([{**good[0], "sha256": "sha256:" + bad}], [{**good[0], "bytes": good[0]["bytes"] + 1}], [], [*good, {"module": "x.y", "sha256": good[0]["sha256"], "bytes": 1}]):
        with pytest.raises(ERR):
            stage(rec, p, b, meta, arrays, m, expected_external_helpers=hp)
    src = expected(p, b, rec)["expected_source"]
    for es in ({}, {"kind": "initial_state"}, {**src, "state_sha256": bad}, {**src, "catalog_hash": bad}, {k: v for k, v in src.items() if k != "catalog_hash"}, {**src, "token": "t"}):
        with pytest.raises(ERR):
            stage(rec, p, b, meta, arrays, m, expected_source=es)
    with pytest.raises(TypeError):
        wh.stage_cached_inputs(meta, arrays, active_mask=m, max_bytes=BIG)                # every external expectation is mandatory
    with pytest.raises(ERR):
        stage(rec, p, b, meta, arrays, m, backend="cuda")


def test_recorded_only_source_proof_is_not_enough_a_foreign_runtime_package_refuses(leaf, monkeypatch):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    real = wvb._runtime_leaf_digest
    monkeypatch.setattr(wvb, "_runtime_leaf_digest", lambda lw: dict(real(lw), digest_sha256="1" * 64))
    with pytest.raises(ERR, match="package digest"):
        stage(rec, p, b, meta, arrays, m)
    monkeypatch.setattr(wvb, "_runtime_leaf_digest", real)
    real_h = wvb._runtime_helper
    monkeypatch.setattr(wvb, "_runtime_helper", lambda mod: (mod, "0" * 64, real_h(mod)[2]))
    with pytest.raises(ERR, match="changed or foreign helper"):
        stage(rec, p, b, meta, arrays, m)


# ---------------------------------------------------------------- masks / held surface / current column
def test_mask_guard_is_explicit_never_inferred_and_a_wrong_mask_is_refused(leaf):
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    p, b, _surf, _m = live(leaf, mask=mask)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    assert meta["mask_sha256"] == b.mask_sha256
    with_mask = stage(rec, p, b, meta, arrays, mask)
    no_mask = stage(rec, p, b, meta, arrays, None)
    assert with_mask.mask_entry_guard_checked is True and no_mask.mask_entry_guard_checked is False and with_mask.record_sha256 == no_mask.record_sha256 == rec.record_sha256
    other = mask.copy()
    other[0, 0] = False
    for bad in (other, np.ones((3, 3), dtype=bool), mask.astype(np.uint8), mask.reshape(-1), np.asfortranarray(mask)):
        with pytest.raises(ERR):
            stage(rec, p, b, meta, arrays, bad)


def test_held_surface_may_predate_the_current_bed_and_properties_change_legally(leaf):
    from veg_binding_support import column
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    rec2 = stage(rec, p, b, meta, arrays, m)
    cb = wh.cached_binding(rec2, expected_binding_sha256=b.sha256, max_bytes=BIG)
    assert cb.surface_sha256 == b.surface_sha256                                         # the HELD surface provenance is preserved, never replaced by the current bed
    base = make_column()
    edges = [0.0, 0.1, 0.3, 0.6]
    refreshed = column(edges, [0.45] * 9, tr=0.12, tf=0.28, ts=0.5, backend="numpy", active=None, stress=0.2)           # a property / depth refresh: same recipe, new column identity
    assert refreshed.sha256 != base.sha256
    old = prepare(base, b)
    cur = prepare(refreshed, b)
    assert old.matrix_sha256 != cur.matrix_sha256 and old.law_sha256 != cur.law_sha256
    ok = prepare(refreshed, cb, expected_matrix_op_sha256=cur.matrix_sha256, expected_preferential_op_sha256=cur.preferential_sha256)
    assert ok.matrix_sha256 == cur.matrix_sha256 and ok.binding_sha256 == b.sha256       # same binding recipe, new CURRENT operator ids
    with pytest.raises(ERR, match="CURRENT operator identity"):
        prepare(refreshed, cb, expected_matrix_op_sha256=old.matrix_sha256)              # the OLD operator ids of the earlier column are not the expectation after a refresh
    with pytest.raises(ERR, match="CURRENT operator identity"):
        prepare(refreshed, cb, expected_preferential_op_sha256=old.preferential_sha256)
    with pytest.raises(ERR):
        prepare(refreshed, cb, expected_matrix_op_sha256="xyz")


# ---------------------------------------------------------------- admission
def test_stage_admission_need_minus_one_exact_reserve_readers_before_any_copy(leaf, monkeypatch):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    stage(rec, p, b, meta, arrays, m)                                                     # warm the cold source digest / helper hash
    est = wh.estimate_cached_stage_bytes_from_meta(meta, source_cold=False, with_mask=True)
    seals = []
    real = wh._seal_any
    monkeypatch.setattr(wh, "_seal_any", lambda a: seals.append(1) or real(a))
    need = est["peak_bytes"]
    with pytest.raises(MEM):
        stage(rec, p, b, meta, arrays, m, max_bytes=need - 1)
    with pytest.raises(MEM):
        stage(rec, p, b, meta, arrays, m, max_bytes=need + 9, reserve_bytes=10)
    with pytest.raises(MEM):
        stage(rec, p, b, meta, arrays, m, available_bytes_fn=lambda: need - 1)
    assert seals == []                                                                    # every refusal precedes the first copy
    assert stage(rec, p, b, meta, arrays, m, max_bytes=need).record_sha256 == rec.record_sha256
    assert stage(rec, p, b, meta, arrays, m, max_bytes=need + 10, reserve_bytes=10).record_sha256 == rec.record_sha256


def test_moderate_host_trace_stays_inside_the_declared_model_and_is_not_a_tautology(leaf):
    import tracemalloc

    from cached_support import catalog, plants_state, project
    from veg_binding_support import full_source, helper_pin, leaf_pin
    cat = catalog(leaf, 3, [0, 1, 1], [0, 1, 1])
    rng = np.random.default_rng(5)
    pl = [(i, "grass" if i % 2 else "shrub", float(rng.uniform(0.5, 99.5)), float(rng.uniform(0.5, 99.5)), float(rng.uniform(0.2, 0.6)), "live") for i in range(1, 301)]
    st = plants_state(leaf, cat, pl, nx=100, ny=100)
    p, surf = project(leaf, st, cat)
    mask = np.ones((100, 100), dtype=bool)
    b = wvb.bind_projection(p, controls(n=10000), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                            expected_external_helpers=helper_pin(), active_mask=mask, current_surface=surf, max_bytes=BIG)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    stage(rec, p, b, meta, arrays, mask)                                                  # warm: cold digest / JIT / imports excluded from the steady-state trace
    est = wh.estimate_cached_stage_bytes_from_meta(meta, source_cold=False, with_mask=True)
    args = {**expected(p, b, rec), "active_mask": mask, "max_bytes": BIG}                 # (the helper hashes the LEAF files: built OUTSIDE the traced region)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    out = wh.stage_cached_inputs(meta, arrays, **args)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert out.record_sha256 == rec.record_sha256
    assert peak <= est["peak_bytes"] and peak >= est["resident_bytes"] // 2              # bounded by the model, and of the order of the resident copies (not a pad)


# ---------------------------------------------------------------- honesty, sealed type
def test_origin_is_labelled_strict_bind_refuses_a_cached_record_and_capture_roundtrips(leaf):
    from veg_binding_support import full_source, helper_pin, leaf_pin
    p, b, surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    with pytest.raises(ERR, match="genuine leaf.water_vegetation.VegetationProjection"):
        wvb.bind_projection(rec, controls(), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=rec.content_sha256, expected_source=full_source(p),
                            expected_external_helpers=helper_pin(), active_mask=m, current_surface=surf, max_bytes=BIG)
    cb = wh.cached_binding(rec, expected_binding_sha256=b.sha256, max_bytes=BIG)
    assert wh.capture_cached_inputs(cb, max_bytes=BIG).record_sha256 == rec.record_sha256
    assert cb.export_metadata()["origin"] == "cached_derived" and b.export_metadata()["origin"] == "live_projection"
    with pytest.raises(ERR):
        wh.cached_binding(rec, expected_binding_sha256="0" * 64, max_bytes=BIG)
    full = cb.export_compact()
    assert full["projection"]["content_sha256"] == rec.content_sha256 and np.shares_memory(cb.export_arrays()["projection.matrix_root_index"], dict(rec.arrays)["matrix_root_index"])


FIELDS = ("species_ids", "n_cells", "shape", "cell_area_m2", "n_entries", "n_plants", "root_layers", "n_omitted", "content_sha256", "receiver_ids", "source_json", "package_json",
          "helpers_json", "identity_json", "mask_sha256", "surface_sha256", "mask_entry_guard_checked", "binding_sha256", "record_sha256", "resident_bytes", "origin", "tag", "live_session_proof")
SWAP = {"species_ids": ("x",), "n_cells": 1, "shape": (1, 9), "cell_area_m2": 2.0, "n_entries": 0, "n_plants": 0, "root_layers": 1, "n_omitted": 1, "content_sha256": "sha256:" + "0" * 64,
        "receiver_ids": (), "source_json": "{}", "package_json": "{}", "helpers_json": "[]", "identity_json": "{}", "mask_sha256": "0" * 64, "surface_sha256": "0" * 64,
        "mask_entry_guard_checked": True, "binding_sha256": "0" * 64, "record_sha256": "0" * 64, "resident_bytes": 1, "origin": "live_projection", "tag": "x", "live_session_proof": True}


@pytest.mark.parametrize("field", FIELDS)
def test_every_record_field_is_sealed(leaf, field):
    _p, b, _surf, _m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    assert wh.verify_cached_inputs(rec) is rec
    object.__setattr__(rec, field, SWAP[field])
    with pytest.raises(ERR):
        wh.verify_cached_inputs(rec)


def test_record_arrays_controls_and_table_replacement_are_detected(leaf):
    _p, b, _surf, _m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    orig_arrays = rec.arrays
    object.__setattr__(rec, "arrays", tuple((k, np.array(v)) for k, v in orig_arrays))            # equal values, writable storage
    with pytest.raises(ERR):
        wh.verify_cached_inputs(rec)
    object.__setattr__(rec, "arrays", orig_arrays)
    wh.verify_cached_inputs(rec)
    object.__setattr__(rec, "receiver_weights", np.array(rec.receiver_weights))
    with pytest.raises(ERR):
        wh.verify_cached_inputs(rec)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    object.__setattr__(rec, "controls", controls(a3=0.3))
    with pytest.raises(ERR):
        wh.verify_cached_inputs(rec)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    object.__setattr__(rec.controls, "a3", 0.9)                                                          # nested forgery inside the controls
    with pytest.raises(ERR):
        wh.verify_cached_inputs(rec)


def test_no_copy_pickle_subclass_direct_construction_uninitialised_or_foreign(leaf):
    p, b, _surf, _m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    for f in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(ERR):
            f(rec)
    with pytest.raises(TypeError):
        type("Sub", (wh.CachedVegetationInputs,), {})
    with pytest.raises(ERR):
        wh.CachedVegetationInputs((), (), 1, (1, 1), 1.0, 0, 0, 1, 0, "s", 3, "", None, (), None, "", "", "", "", "", "", False, "", "", 0)
    with pytest.raises(ERR):
        wh.verify_cached_inputs(object.__new__(wh.CachedVegetationInputs))                              # uninitialised
    for bad in (None, object(), {"arrays": ()}, p):
        with pytest.raises(ERR):
            wh.verify_cached_inputs(bad)
    with pytest.raises(ERR):
        wvb._mint_cached_binding(origin="live_projection")


def test_cached_binding_forgery_is_refused_by_the_public_verifier_and_the_law(leaf):
    _p, b, _surf, _m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    cb = wh.cached_binding(rec, expected_binding_sha256=b.sha256, max_bytes=BIG)
    object.__setattr__(cb, "origin", "live_projection")
    with pytest.raises(ERR):
        wvb.verify_binding(cb)
    cb = wh.cached_binding(rec, expected_binding_sha256=b.sha256, max_bytes=BIG)
    object.__setattr__(rec, "n_cells", 4)
    with pytest.raises(ERR):
        prepare(make_column(), cb)


# ---------------------------------------------------------------- old sources
def _historical_169_guard():
    import hashlib

    from veg_binding_support import CANDIDATE, TASK
    pkg = CANDIDATE / "src" / "maple_syrup"
    manifest = json.loads((TASK / "syrup_preedit_manifest169.json").read_text())
    recorded = dict(manifest["files"])
    present = {q.relative_to(pkg).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in pkg.rglob("*") if q.is_file() and "__pycache__" not in q.parts and q.suffix not in (".pyc", ".pyo")}
    assert len(recorded) == 169 and {k: v for k, v in present.items() if k in recorded} == recorded
    assert set(present) - set(recorded) == {"water_vegetation_binding.py", "water_vegetation_held.py"}


def test_the_169_are_unchanged_and_only_the_two_known_modules_were_added():
    """Branch chosen ONLY by the literal-pinned package: the historical cache-C3 package runs the original task-folder proof; the Root-locked final package runs the portable whole-source proof (no task path, no Git)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase25_source_profile"))
    import p25_source_profile as p25
    branch = p25.require_branch(Path(__file__).resolve().parents[2] / "src" / "maple_syrup", historical=("cache_c3",))
    if branch.kind == "historical":
        _historical_169_guard()
    else:
        v = branch.verified
        assert {"water_vegetation_binding.py", "water_vegetation_held.py"} <= v.added and not {"water_vegetation_binding.py", "water_vegetation_held.py"} & set(v.base)
        assert set(v.base) <= set(v.files) and {k for k in v.base if v.files[k] != v.base[k]} == v.changed and v.unchanged == set(v.base) - v.changed
    import subprocess

    out = subprocess.run([sys.executable, "-B", "-c", "import sys, maple_syrup.water_vegetation_held; print('leaf' in sys.modules)"], capture_output=True, text=True, env=native_env(), check=False)
    assert out.stdout.split() == ["False"], out.stderr                                                    # ordinary imports stay LEAF-free


def test_fresh_process_stage_from_a_written_record_with_the_pinned_native_path(leaf, tmp_path):
    """A FRESH child on the explicit pinned prefix: loads ONLY the written record (npz + json), no original binding / projection / plants / bed, stages it, prepares the law and reports hashes."""
    import subprocess
    import sys

    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    np.savez(tmp_path / "arrays.npz", **arrays)
    pins = {**{k: v for k, v in expected(p, b, rec).items()}, "mask": m.tolist()}
    (tmp_path / "meta.json").write_text(json.dumps({"meta": meta, "pins": pins}))
    live_law = prepare(make_column(), b)
    script = f"""
import json, sys
from pathlib import Path
import numpy as np
NATIVE = Path({str(NATIVE)!r})
import maple
assert Path(maple.__file__).resolve().is_relative_to(NATIVE / "source" / "src"), maple.__file__
from maple_syrup.provenance import source_tree_digest
assert source_tree_digest(Path(maple.__file__).resolve().parent).digest_sha256 == {MAPLE_PACKAGE!r}
import veg_binding_support  # noqa  (actual LEAF root on sys.path)
import leaf
assert source_tree_digest(Path(leaf.__file__).resolve().parent).digest_sha256 == {LEAF_PACKAGE!r}, leaf.__file__
from maple_syrup import water_vegetation_binding as wvb, water_vegetation_held as wh
d = json.loads(Path({str(tmp_path / "meta.json")!r}).read_text())
z = np.load({str(tmp_path / "arrays.npz")!r})
arrays = {{k: np.array(z[k]) for k in z.files}}
pins = d["pins"]
mask = np.array(pins.pop("mask"), dtype=bool)
rec = wh.stage_cached_inputs(d["meta"], arrays, active_mask=mask, max_bytes=1 << 30, **pins)
cb = wh.cached_binding(rec, expected_binding_sha256=pins["expected_binding_sha256"], max_bytes=1 << 30)
from veg_binding_support import make_column
law = wvb.prepare_vegetation_law(make_column(), cb, conductivity_m_s=2e-6, suction_m=0.3, memory_budget_bytes=1 << 30)
assert Path(maple.__file__).resolve().is_relative_to(NATIVE / "source" / "src")
assert source_tree_digest(Path(maple.__file__).resolve().parent).digest_sha256 == {MAPLE_PACKAGE!r}
assert source_tree_digest(Path(leaf.__file__).resolve().parent).digest_sha256 == {LEAF_PACKAGE!r}
print(json.dumps({{"record": rec.record_sha256, "law": law.law_sha256, "matrix": law.matrix_sha256, "pref": law.preferential_sha256, "origin": law.binding_origin}}))
"""
    out = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, env=native_env(), check=False)
    assert out.returncode == 0, out.stderr[-3000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {"record": rec.record_sha256, "law": live_law.law_sha256, "matrix": live_law.matrix_sha256, "pref": live_law.preferential_sha256, "origin": "cached_derived"}
    # the same child with one tampered byte of the written record refuses (no original objects exist there either)
    z = dict(np.load(tmp_path / "arrays.npz"))
    z["entry_area_all_m2"] = z["entry_area_all_m2"].copy()
    z["entry_area_all_m2"][0] += 1e-9
    np.savez(tmp_path / "arrays.npz", **z)
    bad = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, env=native_env(), check=False)
    assert bad.returncode != 0 and "VegetationBindingError" in bad.stderr
