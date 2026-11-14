"""Correction01 regressions (D1-D5 of the independent review), REAL actual probes: pre-quota gate cost, identifier-sized copies, native cell-area compatibility, the composed gates against the FINAL
public LEAF (H3) models, identity cross-checks. LEAF = Root's frozen H3 package; native MAPLE pinned; nothing faked."""
from __future__ import annotations

import copy
import json
import time
import tracemalloc

import numpy as np
import pytest
from cached_support import (
    collect,
    controls,
    expected,
    fingerprint,
    live,
    loadable,
    make_column,
    plants_state,
    prepare,
    project,
    stage,
    wh,
    wvb,
)
from veg_binding_support import BIG, full_source, helper_pin, leaf_pin

ERR = wvb.VegetationBindingError
MEM = wvb.VegetationBindingMemoryError
ZERO = "0" * 64


def traced(fn):
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    try:
        try:
            out = ("ok", fn())
        except Exception as exc:  # noqa: BLE001
            out = ("err", exc)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    return out, peak


def pure_meta(ids, *, n=9, layers=2, recv=()):
    """A syntactically valid closed record for `ids` (zero arrays): only the gates and the quota are exercised."""
    from leaf import water_vegetation as wv
    S, N = len(ids), n
    arrays = {"splash_cover": np.zeros(N), "shading_cover": np.zeros(N), "matrix_root_index": np.zeros(N), "cell_offsets": np.zeros(N + 1, np.int64), "entry_cell": np.zeros(0, np.int64),
              "entry_species": np.zeros(0, np.int32), "entry_area_all_m2": np.zeros(0), "entry_area_live_m2": np.zeros(0), "root": np.tile([1.0] + [0.0] * (layers - 1), (S, 1)),
              "lower": np.zeros(S), "weights": np.zeros((S, layers)), "has_w": np.zeros(S, np.uint8), "omitted_ids": np.zeros(0, np.int64), "receiver_weights_enabled": np.zeros((len(recv), layers))}
    meta = {"tag": wh.CACHED_TAG, "version": wh.CACHED_VERSION,
            "compact": {"schema": wv.SCHEMA, "version": wv.VERSION, "field_order": list(wh._FIELD_ORDER), "species_ids": ids, "n_cells": N, "shape": [3, N // 3], "cell_area_hex": (1.0).hex(),
                        "n_entries": 0, "n_plants": 0, "root_layers": layers, "n_omitted": 0},
            "content_sha256": "sha256:" + ZERO,
            "controls": {"a1": (0.013).hex(), "a2": (0.067).hex(), "a3_layout": "scalar", "a3": (0.4).hex(), "max_fraction_per_hour": (0.7).hex(), "max_rate_layout": "scalar",
                         "max_intake_rate_m_s": (1e-5).hex(), "has_background": False, "n_layers": layers, "n_cells": None, "sha256": ZERO},
            "receiver_ids": list(recv), "source": {}, "package": {}, "external_helpers": [], "identity": {}, "mask_sha256": ZERO, "surface_sha256": ZERO, "binding_sha256": ZERO,
            "record_sha256": ZERO}
    return meta, arrays


def hostile_kw(**kw):
    args = {"expected_record_sha256": ZERO, "expected_binding_sha256": ZERO, "expected_projection_sha256": ZERO, "expected_leaf_package_sha256": ZERO,
            "expected_external_helpers": [{"module": "benchmarks.spatial_mapping", "sha256": "sha256:" + ZERO, "bytes": 1}], "expected_source": {"kind": "x"}, "active_mask": None,
            "max_bytes": 1}
    args.update(kw)
    return args


# ---------------------------------------------------------------- D1: pre-quota work is O(1) owned memory and linear time
def test_d1_gate_and_max_bytes_1_refusal_own_o1_memory_even_for_100000_species(leaf):
    S = 100000
    ids = [f"s{i:06d}" for i in range(S)]
    meta, arrays = pure_meta(ids, n=9)
    r, peak_gate = traced(lambda: wh._gate_meta(meta))
    assert r[0] == "ok" and peak_gate < 32 * 1024, peak_gate                           # the review measured 6,291,912 B here
    r, peak = traced(lambda: wh.stage_cached_inputs(meta, arrays, **hostile_kw()))
    assert r[0] == "err" and isinstance(r[1], MEM) and "nothing allocated" in str(r[1])
    assert peak < 64 * 1024, peak                                                       # ... and 6,292,056 B before the refusal in the review


def test_d1_receiver_subset_and_uniqueness_are_not_quadratic_before_any_quota(leaf):
    def gate_time(n):
        ids = [f"s{i:06d}" for i in range(n)]
        meta, _ = pure_meta(ids, recv=ids[::-1])
        t = time.perf_counter()
        wh._gate_meta(meta)
        return time.perf_counter() - t
    gate_time(500)                                                                       # warm
    t1, t4 = min(gate_time(4000) for _ in range(3)), min(gate_time(16000) for _ in range(3))
    assert t4 / max(t1, 1e-9) < 9.0, (t1, t4)                                           # 4x the identifiers: linear is ~4x, the old membership scan was ~16x


def test_d1_hostile_identifiers_and_documents_are_refused_before_any_owned_work(leaf):
    long_ids = ["x" * 2000 + f"{i:04d}" for i in range(2000)]                           # 4 MB of hostile identifier bytes
    meta, arrays = pure_meta(long_ids)
    r, peak = traced(lambda: wh.stage_cached_inputs(meta, arrays, **hostile_kw(max_bytes=1 << 40)))
    assert r[0] == "err" and isinstance(r[1], ERR) and not isinstance(r[1], MEM) and "1..64" in str(r[1]) and peak < 64 * 1024
    ok_ids = [f"s{i:03d}" for i in range(4)]
    for mut in (lambda m: m["compact"].__setitem__("species_ids", ok_ids[:3] + ["y" * 65]), lambda m: m.__setitem__("receiver_ids", ["z" * 65]),
                lambda m: m["compact"].__setitem__("species_ids", ok_ids[:3] + [""]), lambda m: m["compact"].__setitem__("species_ids", [*ok_ids[:3], 5]),
                lambda m: m.__setitem__("identity", {"x": "a" * (2 << 20)}), lambda m: m["source"].__setitem__("k", [{"a": "b" * 600000}, {"a": "c" * 600000}]),
                lambda m: m.__setitem__("external_helpers", [{}] * 65), lambda m: m["compact"].__setitem__("species_ids", tuple(ok_ids))):
        m2, a2 = pure_meta(ok_ids, recv=ok_ids[:1])
        mut(m2)
        r, peak = traced(lambda m2=m2, a2=a2: wh.stage_cached_inputs(m2, a2, **hostile_kw(max_bytes=1 << 40)))
        assert r[0] == "err" and isinstance(r[1], ERR) and not isinstance(r[1], MEM) and peak < 64 * 1024, (r, peak)


def test_d1_version_and_count_types_are_strict(leaf):
    meta, _arrays = pure_meta([f"s{i}" for i in range(3)])
    assert wh.CACHED_VERSION == "2" and isinstance(wh.CACHED_VERSION, str)             # explicitly a STRING (the suspected bool/int leniency does not exist)
    for bad in (1, 2, True, 2.0, None, "1", "02", ""):
        m = copy.deepcopy(meta)
        m["version"] = bad
        with pytest.raises(ERR):
            wh._gate_meta(m)
    for bad in (True, 3.0, "3", None):
        m = copy.deepcopy(meta)
        m["compact"]["version"] = bad
        with pytest.raises(ERR):
            wh._gate_meta(m)
    for key in ("n_cells", "n_entries", "n_plants", "root_layers", "n_omitted"):
        for bad in (True, 9.0, "9", None):
            m = copy.deepcopy(meta)
            m["compact"][key] = bad
            with pytest.raises(ERR):
                wh._gate_meta(m)


# ---------------------------------------------------------------- D2: identifier-sized copies are modelled or streamed
def _big_species_case(leaf, S=400, layers=3):
    """A genuine P1 projection from a REAL catalog of S species with 64-character ids (the maximum), two of them populated."""
    import json as _json

    from veg_binding_support import LEAF_ROOT, fractions
    doc = _json.loads((LEAF_ROOT / "examples" / "phase0" / "species.json").read_text())
    doc["root_layers"] = layers
    doc["schema_version"] = 2
    base = doc["species"]
    names = [("g" * 4 + f"{i:04d}").ljust(64, "x") for i in range(S - 2)] + ["grass", "shrub"]
    sp = {}
    for i, n in enumerate(names):
        rec = copy.deepcopy(base["grass" if i % 2 == 0 else "shrub"])
        rec["roots"]["layer_fractions"] = list(fractions(layers)["grass" if i % 2 == 0 else "shrub"])
        rec["hydrology"] = {"preferential_infiltration_receiver_weights": [0.0] + [1.0] * (layers - 1)}
        sp[n] = rec
    doc["species"] = sp
    cat = leaf["traits"].SpeciesCatalog(doc)
    st = plants_state(leaf, cat, [(1, "grass", 0.5, 0.5, 0.4, "live"), (2, "shrub", 1.5, 1.5, 0.35, "live")])
    p, surf = project(leaf, st, cat)
    mask = np.ones((3, 3), dtype=bool)
    ctl = controls(L=layers)
    b = wvb.bind_projection(p, ctl, expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                            expected_external_helpers=helper_pin(), active_mask=mask, current_surface=surf, max_bytes=BIG)
    return p, b, surf, mask


def test_d2_real_catalog_with_64_char_ids_exact_admitted_control_and_traced_peak(leaf):
    p, b, _surf, m = _big_species_case(leaf, S=400)
    assert len(p.species_ids) == 400 and max(len(x) for x in p.species_ids) == 64
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    assert "species_ids" not in json.loads(rec.identity_json)                          # the identifiers are stored ONCE (the record field)
    meta, arrays = loadable(rec)
    stage(rec, p, b, meta, arrays, m)                                                   # warm the cold source digest / imports
    est = wh.estimate_cached_stage_bytes_from_meta(meta, source_cold=False, with_mask=True)
    need = est["peak_bytes"]
    with pytest.raises(MEM):
        stage(rec, p, b, meta, arrays, m, max_bytes=need - 1)
    args = {**expected(p, b, rec), "active_mask": m}                                      # (the test helper hashes the LEAF files: built OUTSIDE the traced region)
    r, peak = traced(lambda: wh.stage_cached_inputs(meta, arrays, max_bytes=need, **args))   # the EXACT admitted valid producer control
    assert r[0] == "ok" and r[1].record_sha256 == rec.record_sha256
    assert peak <= need and peak >= est["resident_bytes"] // 2, (peak, est)
    r, peak1 = traced(lambda: wh.stage_cached_inputs(meta, arrays, max_bytes=1, **args))   # max_bytes=1: refused with O(1) owned memory
    assert r[0] == "err" and isinstance(r[1], MEM) and peak1 < 64 * 1024
    r, peak_sha = traced(lambda: wh._record_sha(rec.content_sha256, rec.controls.sha256, rec.binding_sha256, rec.source_json, rec.package_json, rec.helpers_json, rec.identity_json,
                                                rec.mask_sha256, rec.surface_sha256, rec.receiver_ids, rec.receiver_weights, rec.species_ids, rec.shape, rec.cell_area_m2, rec.schema,
                                                rec.compact_version))
    assert r[1] == rec.record_sha256 and peak_sha < 16 * 1024                           # streamed: no identifier-sized JSON copy (the review measured 16 MB for 4 MB of ids)
    ident = rec.identity
    assert ident["species_ids"] == list(rec.species_ids)                                # the accessor returns detached, truthful content
    ident["species_ids"].append("x")
    ident["domain"]["dx_m"] = 99.0
    assert rec.identity["species_ids"] == list(rec.species_ids) and rec.identity["domain"]["dx_m"] == 1.0


def test_d2_model_terms_scale_with_the_longest_identifier_and_the_metadata_bytes(leaf):
    a = wh.estimate_cached_stage_bytes(9, 0, 100, 2, 0, max_id_chars=1, meta_json_bytes=0)
    b = wh.estimate_cached_stage_bytes(9, 0, 100, 2, 0, max_id_chars=64, meta_json_bytes=0)
    c = wh.estimate_cached_stage_bytes(9, 0, 100, 2, 0, max_id_chars=64, meta_json_bytes=10 << 20)
    assert b["stages"]["h1"] - a["stages"]["h1"] == 3 * 12 * 63                          # LEAF's literal 3 x (12 C + 4) header transient
    assert c["resident_bytes"] - b["resident_bytes"] == 10 << 20 and c["stages"]["record"] - b["stages"]["record"] == 3 * (10 << 20)


# ---------------------------------------------------------------- D3: exact cell area against the native column
def test_d3_native_column_cell_area_must_equal_the_binding_area_live_and_cached(leaf, monkeypatch):
    from veg_binding_support import column
    _p, b, _surf, _m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    cb = wh.cached_binding(rec, expected_binding_sha256=b.sha256, max_bytes=BIG)
    assert b.cell_area_m2 == rec.cell_area_m2 == cb.cell_area_m2 == 1.0 and b.export_metadata()["cell_area_m2_hex"] == (1.0).hex()
    edges = [0.0, 0.1, 0.3, 0.6]
    from maple_syrup import floating_matrix_infiltration as fm
    calls = []
    real = fm.prepare_floating_matrix
    monkeypatch.setattr(fm, "prepare_floating_matrix", lambda *a, **k: calls.append(1) or real(*a, **k))
    for area in (0.01, 1.0000001, 4.0):
        bad = column(edges, [0.6] * 9, tr=0.1, tf=0.3, ts=0.5, backend="numpy", active=None, stress=0.2, area=area)
        for binding in (b, cb):
            with pytest.raises(ERR, match="cell_area_m2"):
                prepare(bad, binding)
    assert calls == []                                                                   # refused BEFORE the native preparation / any allocation
    good = column(edges, [0.6] * 9, tr=0.1, tf=0.3, ts=0.5, backend="numpy", active=None, stress=0.2, area=1.0)
    assert prepare(good, cb).law_sha256 == prepare(good, b).law_sha256 and calls


def test_d3_the_binding_sha_is_version_2_and_covers_the_exact_area(leaf):
    _p, b, _surf, _m = live(leaf)
    assert wvb.BINDING_VERSION == "2" and wvb.BINDING_TAG.endswith(".v2") and b.export_metadata()["version"] == "2"
    args = (b.controls.sha256, b.leaf_package_sha256, b.projection_sha256, b.source, b.mask_sha256, b.surface_sha256, b.shape)
    helpers = tuple(sorted((h["module"], h["sha256"].removeprefix("sha256:"), h["bytes"]) for h in json.loads(b.external_helpers_json)))
    base = wvb._binding_sha(*args, 1.0, b.species_ids, b.enabled_species, b.n_layers, helpers, b.receiver_weights)
    assert base == b.sha256                                                              # reproducible from the public parts
    assert wvb._binding_sha(*args, 1.0000000000000002, b.species_ids, b.enabled_species, b.n_layers, helpers, b.receiver_weights) != base


def test_d3_and_d5_live_bind_checks_the_identity_domain_and_square_cells(leaf, monkeypatch):
    p, _b, surf, m = live(leaf)
    real = type(p).identity_record

    def forged(self):
        d = real(self)
        d["domain"]["dx_m"] = 0.5
        return d
    monkeypatch.setattr(type(p), "identity_record", forged)
    with pytest.raises(ERR, match="cell area|square"):
        wvb.bind_projection(p, controls(), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                            expected_external_helpers=helper_pin(), active_mask=m, current_surface=surf, max_bytes=BIG)


# ---------------------------------------------------------------- D4: composed gates equal the inner (final public LEAF) gates
@pytest.mark.parametrize("layers", [1, 2, 3])
def test_d4_stage_exact_cap_need_minus_one_and_reserve_for_one_two_three_layers(leaf, layers):
    grass = (1,) if layers == 1 else (0,) + (1,) * (layers - 1)
    p, b, _surf, m = live(leaf, layers=layers, grass=grass, shrub=grass)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    stage(rec, p, b, meta, arrays, m)
    est = wh.estimate_cached_stage_bytes_from_meta(meta, source_cold=False, with_mask=True)
    lw = leaf["wv"]
    assert est["h2_weights_stage_bytes"] == lw.estimate_stream_bytes(9, rec.n_entries, len(rec.receiver_ids), layers)["weights_stage_bytes"]    # the PUBLIC weights stage
    need = est["peak_bytes"]
    with pytest.raises(MEM):
        stage(rec, p, b, meta, arrays, m, max_bytes=need - 1)
    with pytest.raises(MEM):
        stage(rec, p, b, meta, arrays, m, max_bytes=need + 4, reserve_bytes=5)
    with pytest.raises(MEM):
        stage(rec, p, b, meta, arrays, m, available_bytes_fn=lambda: need - 1)
    assert stage(rec, p, b, meta, arrays, m, max_bytes=need).record_sha256 == rec.record_sha256                  # exact: no inner gate refuses after the copies
    assert stage(rec, p, b, meta, arrays, m, max_bytes=need + 5, reserve_bytes=5).record_sha256 == rec.record_sha256
    assert stage(rec, p, b, meta, arrays, m, available_bytes_fn=lambda: need).record_sha256 == rec.record_sha256


def test_d4_refusals_at_the_exact_gate_never_come_after_the_copies(leaf, monkeypatch):
    """At need-1 nothing is copied; at need no inner (H1 / H3 / H2) refusal may occur: the outer model is >= every inner admission."""
    p, b, _surf, m = _big_species_case(leaf, S=64)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    stage(rec, p, b, meta, arrays, m)
    seals = []
    real = wh._seal_any
    monkeypatch.setattr(wh, "_seal_any", lambda a: seals.append(1) or real(a))
    need = wh.estimate_cached_stage_bytes_from_meta(meta, source_cold=False, with_mask=True)["peak_bytes"]
    with pytest.raises(MEM):
        stage(rec, p, b, meta, arrays, m, max_bytes=need - 1)
    assert seals == []
    assert stage(rec, p, b, meta, arrays, m, max_bytes=need).record_sha256 == rec.record_sha256 and seals


@pytest.mark.parametrize("layers,S", [(1, 2), (3, 2), (3, 300)])
def test_d4_s0_bind_exact_cap_and_law_composition_with_the_public_models(leaf, monkeypatch, layers, S):
    if S == 2:
        grass = (1,) if layers == 1 else (0, 1, 1)
        _cat, _st, p, surf = __import__("veg_binding_support").two_species(leaf, layers, grass_w=list(grass), shrub_w=list(grass))
    else:
        p, _b, surf, _m = _big_species_case(leaf, S=S, layers=layers)
    mask = np.ones((3, 3), dtype=bool)
    pub = p.stream_estimate(previous_held=False)
    wvb.bind_projection(p, controls(L=layers), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                        expected_external_helpers=helper_pin(), active_mask=mask, current_surface=surf, max_bytes=BIG)           # warm cold digest
    est = wvb.estimate_binding_bytes(layers, len(p.species_ids), weights_stage_bytes=pub["weights_stage_bytes"], retained_metadata_bytes=pub["retained_metadata_bytes"], source_cold=False,
                                     borrowed_bytes=p.resident_bytes)
    kw = {"expected_leaf_package_sha256": leaf_pin(), "expected_projection_sha256": p.identity_record()["content_sha256"], "expected_source": full_source(p),
          "expected_external_helpers": helper_pin(), "active_mask": mask, "current_surface": surf}
    calls = []
    real = type(p).identity_record
    monkeypatch.setattr(type(p), "identity_record", lambda self: calls.append(1) or real(self))
    with pytest.raises(MEM):
        wvb.bind_projection(p, controls(L=layers), max_bytes=est["peak_bytes"] - 1, **kw)
    assert calls == []                                                                   # the identity copy (8 S pointers) is admitted BEFORE it exists
    b = wvb.bind_projection(p, controls(L=layers), max_bytes=est["peak_bytes"], **kw)    # exact: LEAF's own weights-stage gate does not refuse after my admission
    assert calls
    # law composition at the exact cap (final public stream models, weights stage included)
    col = make_column(layers)
    held = p.stream_estimate(previous_held=True)
    free = p.stream_estimate(previous_held=False)
    le = wvb.estimate_law_bytes(layers, 9, len(b.enabled_species), binding_resident_bytes=b.resident_bytes + b.borrowed_bytes, stream_per_next_held_bytes=held["per_next_bytes"],
                                stream_per_next_free_bytes=free["per_next_bytes"], stream_weights_stage_bytes=free["weights_stage_bytes"], backend="numpy")
    with pytest.raises(MEM):
        prepare(col, b, memory_budget_bytes=le["peak_bytes"] - 1)
    assert prepare(col, b, memory_budget_bytes=le["peak_bytes"]).n_contributions == len(b.enabled_species)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    cb = wh.cached_binding(rec, expected_binding_sha256=b.sha256, max_bytes=BIG)
    need_c = prepare(col, cb, memory_budget_bytes=BIG).memory["declared"]["peak_bytes"]      # the cached origin OWNS its record (borrowed bytes differ): its own exact cap
    with pytest.raises(MEM):
        prepare(col, cb, memory_budget_bytes=need_c - 1)
    assert prepare(col, cb, memory_budget_bytes=need_c).law_sha256 == prepare(col, b).law_sha256


# ---------------------------------------------------------------- D5: the recorded identity is cross-checked against the compact record
IDENT_FORGERIES = [
    ("domain.dx_m", lambda i: i["domain"].__setitem__("dx_m", 0.1)), ("domain.dy_m", lambda i: i["domain"].__setitem__("dy_m", 2.0)), ("domain.nx", lambda i: i["domain"].__setitem__("nx", 4)),
    ("domain.ny", lambda i: i["domain"].__setitem__("ny", 4)), ("species_ids", lambda i: i.__setitem__("species_ids", ["shrub", "grass"])), ("root_layers", lambda i: i.__setitem__("root_layers", 7)),
    ("catalog_hash", lambda i: i.__setitem__("catalog_hash", "x")), ("version", lambda i: i.__setitem__("version", 99)), ("version_bool", lambda i: i.__setitem__("version", True)),
    ("schema", lambda i: i.__setitem__("schema", "other")), ("content_sha256", lambda i: i.__setitem__("content_sha256", "sha256:" + ZERO)),
    ("surface_sha256", lambda i: i.__setitem__("surface_sha256", ZERO)), ("active_mask_sha256", lambda i: i.__setitem__("active_mask_sha256", ZERO)),
    ("source", lambda i: i["source"].__setitem__("kind", "plant_session")), ("package", lambda i: i["package"].__setitem__("digest", "sha256:" + ZERO)),
]


@pytest.mark.parametrize("name,fn", IDENT_FORGERIES, ids=[n for n, _ in IDENT_FORGERIES])
def test_d5_forged_identity_is_refused_at_stage_even_when_the_external_hashes_match(leaf, name, fn):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    fn(meta["identity"])
    # the forger also recomputes the record SHA over the forged identity and passes it as the external expectation: the cross-check, not the hash, must refuse
    ident_json = wvb._frozen_json(meta["identity"]) if "species_ids" not in meta["identity"] else None
    if ident_json is not None:
        meta["record_sha256"] = wh._record_sha(rec.content_sha256, rec.controls.sha256, rec.binding_sha256, rec.source_json, rec.package_json, rec.helpers_json, ident_json, rec.mask_sha256,
                                               rec.surface_sha256, rec.receiver_ids, rec.receiver_weights, rec.species_ids, rec.shape, rec.cell_area_m2, rec.schema, rec.compact_version)
    snap = fingerprint(meta, arrays)
    with pytest.raises(ERR):
        stage(rec, p, b, meta, arrays, m, expected_record_sha256=meta["record_sha256"])
    assert fingerprint(meta, arrays) == snap
    collect()


def test_d5_the_compact_record_itself_is_cross_checked_square_cells_and_grid(leaf):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    for path, fn in ((("compact", "cell_area_hex"), lambda v: (2.0).hex()), (("compact", "shape"), lambda v: [1, 9]), (("compact", "shape"), lambda v: [9, 1])):
        meta, arrays = loadable(rec)
        node = meta
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = fn(node[path[-1]])
        with pytest.raises(ERR):
            stage(rec, p, b, meta, arrays, m)
