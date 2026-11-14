"""Seal coverage of EVERY field (object.__setattr__ forgeries), strict per-kind source keys, schema/version, and GENUINE PlantSession bindings (LEAF's own real-session fixture, no fake). The frozen
C2 LEAF is an independent control only: the consumer stays UNQUALIFIED until a compatible P1 is accepted."""
from __future__ import annotations

import json

import numpy as np
import pytest
from veg_binding_support import (
    A1,
    A2,
    BIG,
    EDGES,
    KEY,
    column,
    full_source,
    helper_pin,
    leaf_pin,
    load_leaf_fixture,
    make_column,
    two_species,
)

from maple_syrup import water_vegetation_binding as wvb

ERR = wvb.VegetationBindingError
K, PSI = 2e-6, 0.3


def ctl(**kw):
    base = {"a1": A1, "a2": A2, "a3": 0.2, "max_fraction_per_hour": 0.7, "max_intake_rate_m_s": 1e-4, "background_receiver_weights": [0.0, 1.0, 1.0], "n_layers": 3, "n_cells": 9, "max_bytes": BIG}
    base.update(kw)
    return wvb.site_controls(**base)


def bind(p, surf, c=None, mask=None, **kw):
    args = {"expected_leaf_package_sha256": leaf_pin(), "expected_projection_sha256": p.identity_record()["content_sha256"], "expected_source": full_source(p), "expected_external_helpers": helper_pin(),
            "active_mask": np.ones(p.shape, dtype=bool) if mask is None else mask, "current_surface": surf, "max_bytes": BIG}
    args.update(kw)
    return wvb.bind_projection(p, ctl() if c is None else c, **args)


def law_of(leaf, **kw):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    b = bind(p, surf)
    return wvb.prepare_vegetation_law(make_column(), b, conductivity_m_s=K, suction_m=PSI, memory_budget_bytes=BIG, **kw), b, p, surf


@pytest.mark.parametrize("field,value", [("a3", 0.9), ("max_intake_rate_m_s", 2e-4), ("max_fraction_per_hour", 0.1), ("a1", 0.02), ("a2", 0.5), ("n_layers", 4), ("n_cells", 5),
                                         ("resident_bytes", 7), ("sha256", "0" * 64), ("tag", "x"), ("a3_layout", "cell"), ("max_rate_layout", "cell"), ("background_receiver_weights", None)])
def test_every_controls_field_is_sealed(field, value):
    c = ctl()
    assert wvb.verify_site_controls(c) is c
    object.__setattr__(c, field, value)
    with pytest.raises(ERR):
        wvb.verify_site_controls(c)


def test_replaced_background_array_with_same_values_is_detected():
    c = ctl()
    object.__setattr__(c, "background_receiver_weights", np.array([0.0, 1.0, 1.0]))  # equal values, different (writeable, non-bytes) storage
    with pytest.raises(ERR):
        wvb._check_controls(c)


@pytest.mark.parametrize("field,value", [("n_entries", 99), ("shape", (9, 1)), ("resident_bytes", 1), ("borrowed_bytes", 1), ("enabled_species", ()), ("source_kind", "plant_session"),
                                         ("leaf_package_file_count", 1), ("external_helpers_json", "[]"), ("species_ids", ("x",)), ("n_cells", 4), ("n_layers", 2), ("tag", "x"), ("sha256", "0" * 64),
                                         ("surface_sha256", "0" * 64), ("source_json", "{}"), ("projection_sha256", "0" * 64), ("leaf_package_sha256", "0" * 64)])
def test_every_binding_field_is_sealed(leaf, field, value):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    b = bind(p, surf)
    assert wvb.verify_binding(b) is b
    object.__setattr__(b, field, value)
    with pytest.raises(ERR):
        wvb.verify_binding(b)


def test_binding_projection_and_controls_replacement_detected(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    _cat2, _st2, p2, _surf2 = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    b = bind(p, surf)
    object.__setattr__(b, "projection", p2)
    with pytest.raises(ERR):
        wvb._check_binding(b)
    b = bind(p, surf)
    object.__setattr__(b, "controls", ctl(a3=0.3))
    with pytest.raises(ERR):
        wvb._check_binding(b)
    b = bind(p, surf)
    object.__setattr__(b.controls, "a3", 0.9)  # a nested forgery inside the borrowed controls is also caught at the binding gate
    with pytest.raises(ERR):
        wvb._check_binding(b)


@pytest.mark.parametrize("field,value", [("g_min", 0.0), ("g_max", 2.0), ("n_contributions", 9), ("law_sha256", "0" * 64), ("memory_json", "{}"), ("backend", "cuda"), ("device_id", 3),
                                         ("tag", "x"), ("water_head_authority", True), ("step_taken", True), ("binding_sha256", "0" * 64), ("column_sha256", "0" * 64),
                                         ("g_sha256", "0" * 64), ("matrix_sha256", "0" * 64), ("preferential_sha256", "0" * 64)])
def test_every_prepared_law_field_is_sealed(leaf, field, value):
    law, *_ = law_of(leaf)
    assert wvb.verify_prepared_law(law) is law
    object.__setattr__(law, field, value)
    with pytest.raises(ERR):
        wvb.verify_prepared_law(law)


def test_prepared_law_operator_swap_direct_construction_and_immutable_memory(leaf):
    law, *_ = law_of(leaf)
    with pytest.raises(ERR):
        wvb.PreparedVegetationLaw(law.matrix_op, law.preferential_op, "", "", "numpy", "live_projection", None, "", 0.0, 0.0, 0, "", "", "", "{}")
    m = law.memory
    m["declared"]["peak_bytes"] = 0
    assert law.memory["declared"]["peak_bytes"] > 0 and json.loads(law.memory_json)["declared"]["peak_bytes"] > 0
    other, *_ = law_of(leaf)
    swapped = law_of(leaf)[0]
    object.__setattr__(swapped, "matrix_op", other.preferential_op)
    with pytest.raises(ERR):
        wvb.verify_prepared_law(swapped)
    with pytest.raises(ERR):
        wvb.verify_prepared_law(object())


def test_source_keys_are_mandatory_per_kind_and_exact(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    full = full_source(p)
    assert set(full) == set(wvb.SOURCE_KEYS["initial_state"])
    for bad in ({"kind": "initial_state"}, {k: v for k, v in full.items() if k != "state_sha256"}, {k: v for k, v in full.items() if k != "catalog_hash"}, {**full, "token": "x"},
                {**full, "catalog_hash": "0" * 64}, {**full, "state_sha256": "0" * 64}, {**full, "kind": "plant_session"}):
        with pytest.raises(ERR):
            bind(p, surf, expected_source=bad)
    assert bind(p, surf).source_kind == "initial_state"


def test_unknown_projection_schema_version_and_source_layouts_refused(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    wv = leaf["wv"]
    real_v = wv.VERSION
    wv.VERSION = real_v + 1
    try:
        with pytest.raises(ERR, match="schema/version"):
            bind(p, surf)
    finally:
        wv.VERSION = real_v
    with pytest.raises(ERR):
        wvb._check_source({"kind": "mystery"}, {"kind": "mystery"})
    with pytest.raises(ERR):
        wvb._check_source({"kind": "initial_state", "state_sha256": "a", "catalog_hash": "b", "extra": 1}, {"kind": "initial_state", "state_sha256": "a", "catalog_hash": "b"})


# ---------------------------------------------------------------- genuine PlantSession (LEAF's own fixture; no fake session)
def make_session(plants=None):
    nfx = load_leaf_fixture()
    from leaf import state, traits
    doc = json.loads(nfx.FIXTURES["phase1"].read_text())
    nfx.simple_edit(doc)
    doc["schema_version"] = 2
    for sid in doc["species"]:
        doc["species"][sid]["hydrology"] = {KEY: [0.0, 1.0]}
    cat = traits.SpeciesCatalog(doc)
    st = nfx.make_state(cat, plants=plants)
    sess = nfx.start_session(st, nfx.make_config(cat), nfx.anchor(), bound=True)
    return sess, state


def session_ctl(n):
    return wvb.site_controls(a1=A1, a2=A2, a3=0.3, max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0], n_layers=2, n_cells=n, max_bytes=BIG)


def session_col(n):
    return column(EDGES[:3], [EDGES[2]] * n, tr=0.1, tf=0.3, ts=0.5, backend="numpy", active=None, stress=0.2)


def test_genuine_session_binding_complete_expected_source_and_law(leaf):
    sess, _ = make_session()
    surf = np.zeros(sess.snapshot_state().domain.shape)
    p = leaf["wv"].project_from_plant_session(sess, surf, max_edges=1 << 16, max_bytes=BIG)
    full = full_source(p)
    assert set(full) == set(wvb.SOURCE_KEYS["plant_session"])
    assert (full["token"], full["sequence"], full["pop_sha256"], full["geometry_sha256"]) == (sess.token, sess.sequence, sess.pop_sha256, sess.geometry_sha256)
    n = surf.size
    c = session_ctl(n)
    b = bind(p, surf, c, expected_source=full, session=sess)
    assert b.source_kind == "plant_session" and b.source["session_id"] == sess.session_id
    law = wvb.prepare_vegetation_law(session_col(n), b, conductivity_m_s=K, suction_m=PSI, memory_budget_bytes=BIG)
    assert law.n_contributions == 2 and law.g_max > law.g_min          # the real plants raise g above a1/a2 in their cells
    for key in ("token", "sequence", "pop_sha256", "geometry_sha256", "session_id", "lineage", "catalog_hash"):
        bad = {**full, key: ("0" * 64 if isinstance(full[key], str) else full[key] + 1)}
        with pytest.raises(ERR):
            bind(p, surf, c, expected_source=bad)
    with pytest.raises(ERR):
        bind(p, surf, c, expected_source={"kind": "plant_session"})   # a bare kind cannot skip the generation
    other, _ = make_session(plants=[(11, 3.5, 3.5, 0.1), (12, 4.2, 3.5, 0.3)])
    with pytest.raises(ERR, match="check_session"):
        bind(p, surf, c, expected_source=full, session=other)         # a different accepted generation


def test_same_biology_new_surface_is_a_new_binding_without_a_sequence_increment(leaf):
    sess, _ = make_session()
    surf = np.zeros(sess.snapshot_state().domain.shape)
    p1 = leaf["wv"].project_from_plant_session(sess, surf, max_edges=1 << 16, max_bytes=BIG)
    surf2 = surf + 0.001                                              # the authoritative surface moved; the plants (and the session sequence) did not
    p2 = leaf["wv"].project_from_plant_session(sess, surf2, max_edges=1 << 16, max_bytes=BIG)
    c = session_ctl(surf.size)
    b1, b2 = bind(p1, surf, c, session=sess), bind(p2, surf2, c, session=sess)
    assert b1.source == b2.source and b1.surface_sha256 != b2.surface_sha256 and b1.sha256 != b2.sha256
    with pytest.raises(ERR):                                          # stale biology/surface is still refused: the old projection against the new surface
        bind(p1, surf2, c, session=sess)
    again = bind(p1, surf, c, session=sess)
    assert again.sha256 == b1.sha256                                  # an identical binding is deterministic (idempotent identity)


def test_initial_state_projection_cannot_pass_as_a_session(leaf):
    sess, _state = make_session()
    cat = sess.config.catalog
    st = sess.snapshot_state()
    surf = np.zeros(st.domain.shape)
    p = leaf["wv"].project_from_initial_state(st, cat, surf, max_edges=1 << 16, max_bytes=BIG)
    c = session_ctl(surf.size)
    assert bind(p, surf, c).source_kind == "initial_state"
    with pytest.raises(ERR, match="check_session"):
        bind(p, surf, c, session=sess)


def test_public_verifiers_refuse_foreign_objects(leaf):
    for f in (wvb.verify_binding, wvb.verify_site_controls, wvb.verify_prepared_law):
        with pytest.raises(ERR):
            f(object())
        with pytest.raises(ERR):
            f(None)
