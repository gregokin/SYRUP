"""Vegetation S1 eligibility: genuine completion only. A running Occurrence, a pending storm (P > T), an UNCLOSED / mid-day trial, an unpublished Windowed bed and a head without a vegetation generation never qualify; a restored head is saveable
again (chained re-save, dead parent slot) and the parent Boundary may be released. REAL native lifecycle states and the ACTUAL LEAF projection only."""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from vck_builders import (
    AVAIL,
    IDENT,
    PART,
    boundary_vns,
    closed_day_vns,
    load,
    plan,
    save,
)

from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_lifecycle_checkpoint_vegetation as lcv
from maple_syrup import floating_pond as fp
from maple_syrup import pond_checkpoint as pc
from maple_syrup import recharge as rc
from maple_syrup import vegetation_lifecycle as vl


def test_an_occurrence_a_pending_storm_and_an_unclosed_trial_are_not_saveable(tmp_path, leaf):
    ns = boundary_vns(tmp_path / "a", leaf, "numpy", False)
    n = None
    # a running Occurrence (a reopened storm) is refused: the head is not a Boundary / DailyHold
    occ, _st = lc.begin_occurrence(ns.lin, ns.b, vl.event_factory_for(ns.b, lambda vert, topology, cover: ns.factory(vert, topology, cover)), available_bytes_fn=AVAIL)
    assert ns.lin.head is occ
    with pytest.raises(lcv.VegetationLifecycleCheckpointError, match="Occurrence|saveable"):
        save(ns, tmp_path / "x")
    assert not (tmp_path / "x").exists() and not list(tmp_path.glob(".x.partial-*"))
    d = closed_day_vns(tmp_path / "b", leaf, "numpy", False, days=1)
    n = d.closed.ctx.n_cells
    pl = rc.plan_daily_rain(0.01, 3600.0, 0.005, run_digest="ab" * 32, ordinal=1)  # P > T: the FULL native storm is required
    pb = br.begin_storm_from_day(d.lin, d.closed, pl, pet_terrain_m_s=np.full(n, 3e-8), cover=None, available_bytes_fn=AVAIL)
    assert pb.cursor.pending is not None
    with pytest.raises(lcv.VegetationLifecycleCheckpointError, match="pending|saveable"):
        save(d, tmp_path / "y")
    assert not (tmp_path / "y").exists()
    # an UNCLOSED mid-day trial (a day_action state, T not yet applied) is not a genuine closed day
    e = closed_day_vns(tmp_path / "c", leaf, "numpy", False, days=1)
    n = e.closed.ctx.n_cells
    trial = br.day_action(e.closed, plan=plan(1), pet_terrain_m_s=np.full(n, 3e-8), cover=None, available_bytes_fn=AVAIL).state
    with pytest.raises(lcv.VegetationLifecycleCheckpointError, match="genuine closed day"):
        save(e, tmp_path / "z", trial)
    assert not (tmp_path / "z").exists()


def test_a_head_without_a_vegetation_generation_is_not_saved_by_this_profile_and_a_vegetation_head_is_not_saved_by_the_old_one(tmp_path, leaf):
    from maple_syrup import floating_lifecycle_checkpoint as lcp

    plain = boundary_vns(tmp_path / "a", leaf, "numpy", False, apply=False)
    with pytest.raises(lcv.VegetationLifecycleCheckpointError, match="vegetation-bound head only"):
        save(plain, tmp_path / "x")
    # the OLD profile still saves a plain head (Leaf-free) and the vegetation loader refuses its directory
    saved = lcp.save_floating_lifecycle_checkpoint(tmp_path / "old", plain.lin, lambda v, t=None, s=None: plain.factory(v, t, None, s), IDENT, original_topology=plain.original, available_bytes_fn=AVAIL)
    assert saved.kind == "boundary" and plain.lin.head is plain.b


def test_unpublished_windowed_bed_is_refused_before_the_carrier_or_any_deep_read(tmp_path, leaf, monkeypatch):
    from maple.surface.voxels.deep_store import DeepJournal
    from maple.surface.voxels.windowed_pending import attach_pending_journal

    ns = boundary_vns(tmp_path / "a", leaf, "numpy", True)
    col = ns.lin.head.bed.voxel_column
    nc = int(col.deep_store.header.n_classes)
    j = DeepJournal.from_records(np.array([0]), np.array([int(col.n_voxels) - 1]), np.zeros((1, nc)), header=col.deep_store.header)
    attach_pending_journal(col, j)  # a REAL unpublished MAPLE journal marker (restored exactly below)
    calls = []
    real_from = fp.FloatingPondVertical.from_boundary.__func__
    monkeypatch.setattr(fp.FloatingPondVertical, "from_boundary", classmethod(lambda cls, *a, **k: calls.append(1) or real_from(cls, *a, **k)))
    try:
        with pytest.raises((lcv.VegetationLifecycleCheckpointError, pc.PondCheckpointError), match="published|journal"):  # the same typed native refusal as the S1 profile
            save(ns, tmp_path / "ck")
        assert not calls and not (tmp_path / "ck").exists() and not list(tmp_path.glob(".ck.partial-*"))
    finally:
        object.__setattr__(col, "pending_deep", None)
    assert save(ns, tmp_path / "ck2").kind == "boundary"  # published again: the same save works


def test_the_parent_may_be_released_and_a_restored_closed_day_is_saveable_again_with_the_same_recipe(tmp_path, leaf):
    ns = closed_day_vns(tmp_path / "a", leaf, "numpy", False, days=1)
    ref = weakref.ref(ns.b)
    rec0 = vl.current_record(ns.lin.head)
    ns.b = None
    del ns.origin
    gc.collect()
    assert ref() is None  # the old parent Boundary is garbage: a save does not need it (nor retain it)
    saved = save(ns, tmp_path / "ck", ns.closed)
    r = load(ns, tmp_path / "ck", saved, partition=PART)
    again = save(_ns(r, ns), tmp_path / "ck2", r.closed)  # a restored cached-derived record saves ITS OWN recipe
    assert again.recipe_id == rec0.recipe_id and again.generation == rec0.generation
    r2 = load(ns, tmp_path / "ck2", again, partition=PART)
    assert (r2.closed.execution_digest, r2.closed.digest) == (ns.closed.execution_digest, ns.closed.digest)
    assert vl.current_record(r2.lineage.head).binding.origin == "cached_derived" and vl.current_record(r2.lineage.head).recipe_id == rec0.recipe_id


def _ns(r, ns):
    from types import SimpleNamespace

    return SimpleNamespace(lin=r.lineage, factory=ns.factory, original=ns.original)
