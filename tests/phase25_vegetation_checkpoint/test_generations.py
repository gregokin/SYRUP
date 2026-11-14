"""Vegetation S1: the STABLE generation counter / recipe id / parent survive the file, a restored cached-derived generation is NOT a new live biology generation, and a subsequent NEW live generation (actual LEAF PlantSession) still needs the
producer proof and ordering. REAL native boundaries, the LEAF candidate's own native session fixtures; the consumers skip (no claim) if those fixtures cannot be loaded."""
from __future__ import annotations

import numpy as np
import pytest
import veg_lc_builders as vb
from vck_builders import AVAIL, BIG, boundary_vns, load, save, storm

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_held as wvh


def _session_lineage(tmp, leaf, backend="numpy", windowed=False):
    nfx = vb.leaf_fixture(leaf)
    ns = boundary_vns(tmp, leaf, backend, windowed, apply=False)
    b = ns.b
    surf = vb.surface_of(b)
    dx = float(b.routing[0].dx_m)
    mask = np.ones(surf.shape, dtype=bool)
    cat = vb.session_catalog(leaf, nfx, b.root.column.n_layers)
    cfg = nfx.make_config(cat)
    sess0 = nfx.start_session(vb.session_state(leaf, cat, surf, dx), cfg, nfx.anchor(), bound=True)
    ctl = vb.controls(b)
    bd0, _p0 = vb.session_binding(leaf, b, sess0, surf, mask, ctl)
    ns.b = vl.apply_vegetation_generation(ns.lin, b, bd0, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    sess1 = nfx.run_day(sess0, nfx.anchor(), 0, t=False, g=False)[0]
    ns.sess = (nfx, cat, cfg, sess0, sess1, surf, dx, mask, ctl)
    ns.bd0 = bd0
    return ns


def test_generation_counter_parent_and_recipe_survive_and_a_new_live_generation_orders_after_the_restore(tmp_path, leaf):
    ns = _session_lineage(tmp_path / "a", leaf)
    nfx, cat, cfg, _sess0, sess1, surf, dx, mask, ctl = ns.sess
    # generation 2 BEFORE the save (a two-link chain: the file must carry the counter, the parent recipe and the child id)
    bd1, _ = vb.session_binding(leaf, ns.b, sess1, surf, mask, ctl)
    b2 = vl.apply_vegetation_generation(ns.lin, ns.b, bd1, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    rec2 = vl.current_record(b2)
    assert rec2.generation == 2 and rec2.parent_recipe_id is not None
    saved = save(ns, tmp_path / "ck")
    assert saved.generation == 2 and saved.recipe_id == rec2.recipe_id
    r = load(ns, saved.path, saved)
    rr = vl.current_record(r.head)
    assert (rr.generation, rr.recipe_id, rr.parent_recipe_id) == (2, rec2.recipe_id, rec2.parent_recipe_id) and rr.binding.origin == "cached_derived"
    assert rr.source == rec2.source and rr.external_helpers == rec2.external_helpers
    # a cached-derived record is not a new live generation: the LIVE apply refuses it (no producer proof), the head stays
    cached = rr.binding.projection
    cb = wvh.cached_binding(cached, expected_binding_sha256=rr.binding_sha256, max_bytes=BIG)
    with pytest.raises(vl.VegetationLifecycleError, match="LEAF|live|cached"):
        vl.apply_vegetation_generation(r.lineage, r.head, cb, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    assert r.lineage.head is r.head
    # the genuine LIVE generation 2 again is idempotent against the restored record; the OLDER generation 1 is stale; a foreign session is refused
    assert vl.apply_vegetation_generation(r.lineage, r.head, bd1, memory_budget_bytes=BIG, available_bytes_fn=AVAIL) is r.head
    with pytest.raises(vl.VegetationLifecycleError, match="OLDER biological generation"):
        vl.apply_vegetation_generation(r.lineage, r.head, ns.bd0, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    other = nfx.start_session(vb.session_state(leaf, cat, surf, dx), cfg, nfx.anchor(), bound=True)
    bdx, _ = vb.session_binding(leaf, r.head, other, surf, mask, ctl)
    with pytest.raises(vl.VegetationLifecycleError, match="foreign session|PlantSession / lineage"):
        vl.apply_vegetation_generation(r.lineage, r.head, bdx, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    assert r.lineage.head is r.head
    # a genuine NEXT session generation applies on BOTH the uninterrupted and the restored lineage with the same result and a continued parent chain
    sess2 = nfx.run_day(sess1, nfx.anchor(), 1, t=False, g=False)[0]
    bd2a, _ = vb.session_binding(leaf, b2, sess2, surf, mask, ctl)
    bd2b, _ = vb.session_binding(leaf, r.head, sess2, surf, mask, ctl)
    n_a = vl.apply_vegetation_generation(ns.lin, b2, bd2a, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    n_b = vl.apply_vegetation_generation(r.lineage, r.head, bd2b, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    ra, rb = vl.current_record(n_a), vl.current_record(n_b)
    assert (ra.generation, ra.recipe_id, ra.parent_recipe_id) == (rb.generation, rb.recipe_id, rb.parent_recipe_id) and ra.generation == 3 and ra.parent_recipe_id == rec2.recipe_id
    assert rb.binding.origin != "cached_derived" and ra.applied_law_sha256 == rb.applied_law_sha256
    assert n_a.root.matrix_op.sha256 == n_b.root.matrix_op.sha256 and n_a.root.pref_op.sha256 == n_b.root.pref_op.sha256
    assert np.array_equal(np.asarray(fm.held_cell_vegetation_factor(n_a.root.matrix_op)), np.asarray(fm.held_cell_vegetation_factor(n_b.root.matrix_op)))
    assert np.array_equal(n_a.root.history.wetting_m, n_b.root.history.wetting_m) and np.array_equal(n_a.root.history.cumulative_intake_m, n_b.root.history.cumulative_intake_m)
    # and the restored lineage can run a genuine storm next with the new bound cover and refreshes with the carried law
    s_a, s_b = storm(ns.lin, n_a, ns.factory), storm(r.lineage, n_b, ns.factory)
    assert np.array_equal(s_a.root.water_state.water_m, s_b.root.water_state.water_m) and s_a.ledger == s_b.ledger and vl.current_record(s_a).recipe_id == vl.current_record(s_b).recipe_id
    assert s_a.root.matrix_op.sha256 == s_b.root.matrix_op.sha256


def test_the_saved_recipe_of_a_property_refreshed_head_rebuilds_the_current_law_not_the_applied_one(tmp_path, leaf):
    ns = boundary_vns(tmp_path / "a", leaf, "numpy", False)
    rec0 = vl.current_record(ns.b)
    after = storm(ns.lin, ns.b, ns.factory)  # a genuine storm -> completion -> property refresh keeps the recipe, the CURRENT law may differ from the applied one
    rec1 = vl.current_record(after)
    assert rec1.recipe_id == rec0.recipe_id
    saved = save(ns, tmp_path / "ck")
    r = load(ns, saved.path, saved)
    rr = vl.current_record(r.head)
    cur_a, cur_b = vl.vegetation_attribution(after), vl.vegetation_attribution(r.head)
    assert cur_a["current_matrix_op_sha256"] == after.root.matrix_op.sha256 == r.head.root.matrix_op.sha256 == cur_b["current_matrix_op_sha256"]
    assert cur_a["current_property_generation"] == cur_b["current_property_generation"] == after.root.generation and cur_a["current_law_id"] == cur_b["current_law_id"]
    assert rr.applied_column_sha256 == rec0.applied_column_sha256 and rr.applied_matrix_op_sha256 == rec0.applied_matrix_op_sha256  # historical apply-time provenance is carried, labelled and never the current law
    assert after.root.column.sha256 == r.head.root.column.sha256
