# ruff: noqa: RUF059
"""Phase 2.5 vegetation lifecycle: the owned zero-time `apply_vegetation_generation` on REAL native lifecycle boundaries (CPU NumPy/Numba x Dense/published Windowed) with the ACTUAL LEAF P1 projection and the S0 binding. Independent
expectations: closed-form circle areas for R1 / g / contributions (rtol 1e-9), explicit prescribed-map native preparations for the bit-level operator identities, and independent before/after water inventories. LEAF is never faked."""
from __future__ import annotations

import copy
import gc
import weakref

import numpy as np
import pytest
import veg_lc_builders as vb
from bridge_builders import AVAIL, dry_boundary

from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import matrix_infiltration as mi
from maple_syrup import soil_evaporation as se
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_binding as wvb

PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
BIG = vb.BIG


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def apply(lin, b, bd, **kw):
    args = {"memory_budget_bytes": BIG, "available_bytes_fn": AVAIL}
    args.update(kw)
    return vl.apply_vegetation_generation(lin, b, bd, **args)


def apply_estimate(b, bd):
    S = len(bd.enabled_species)
    held, free, wstage = vl.stream_numbers(bd)  # LEAF's PUBLIC numbers (per-next held/free and the separate weights stage)
    return vl.estimate_apply_bytes(b.root.column.n_layers, b.root.column.n_cells, S, binding_resident_bytes=bd.resident_bytes, borrowed_bytes=bd.borrowed_bytes,
                                   old_retained_bytes=int(b.root.matrix_op.retained_bytes) + (0 if b.root.pref_op is None else int(b.root.pref_op.retained_bytes)),
                                   stream_per_next_held_bytes=held, stream_per_next_free_bytes=free, stream_weights_stage_bytes=wstage)


def inventory(b):
    return float(fsl.floating_storage_m3(b.root.column, b.root.water_state)) + float(np.asarray(b.surface_depth_m).sum()) * float(b.root.column.cell_area_m2)


@pytest.mark.parametrize("windowed", [False, True])
def test_zero_time_apply_preserves_everything_and_builds_the_native_law(tmp_path, leaf, backend, windowed):
    lin, b, factory = dry_boundary(tmp_path / "a", backend, windowed)
    ny, nx = b.surface_depth_m.shape
    dx = float(b.routing[0].dx_m)
    s = vb.stack(leaf, b)
    bd = vb.binding(leaf, b, s)
    snap = {"w": np.array(b.root.water_state.water_m, copy=True), "j": np.array(b.root.history.wetting_m, copy=True), "i": np.array(b.root.history.cumulative_intake_m, copy=True),
            "inv": inventory(b), "depth": np.array(b.surface_depth_m, copy=True), "qx": np.array(b.qx_m2_s, copy=True), "led": dict(b.ledger), "old_op": b.root.matrix_op.sha256}
    nb = apply(lin, b, bd)
    assert lin.head is nb and nb is not b and nb.role == "boundary"
    # nothing physical moved: the SAME current objects / bit-identical values, no new generation, no event, no refresh, no time
    assert nb.root.water_state is b.root.water_state and nb.bed is b.bed and nb.surface_depth_m is b.surface_depth_m and nb.routing is b.routing and nb.provenance is b.provenance
    assert (nb.t_s, nb.seq, nb.cursor, nb.root.generation, nb.root.source_witness, nb.root.column) == (b.t_s, b.seq, b.cursor, b.root.generation, b.root.source_witness, b.root.column)
    assert np.array_equal(np.asarray(nb.root.water_state.water_m), snap["w"]) and np.array_equal(np.asarray(nb.root.history.wetting_m), snap["j"]) and np.array_equal(np.asarray(nb.root.history.cumulative_intake_m), snap["i"])
    assert nb.ledger == snap["led"] and inventory(nb) == snap["inv"]  # independent water inventory identical (exactly: nothing was touched)
    assert nb.root.history is not b.root.history and nb.root.history.operator_sha256 == nb.root.matrix_op.sha256  # J/I rebound (new operator identity), values bit for bit
    # the NEW matrix law: per-cell g from the actual LEAF R1 through the existing function; independent closed form
    g = np.asarray(fm.held_cell_vegetation_factor(nb.root.matrix_op))
    r1 = vb.closed_form_r1(dx, ny, nx)
    assert np.allclose(np.asarray(s["proj"].matrix_root_index), r1, rtol=1e-9, atol=1e-15)
    assert np.allclose(g, (vb.A1 + r1) / (vb.A2 + r1), rtol=1e-9, atol=1e-15) and nb.root.matrix_op.sha256 != snap["old_op"]
    assert np.array_equal(g, mi.reference_vegetation_function(np.asarray(s["proj"].matrix_root_index), vb.A1, vb.A2))  # the existing function, called once, bit for bit
    # the NEW preferential law = the canonical derivation fed the LEAF contributions (explicit prescribed-map native preparation), shrub (all-zero weights) disabled
    ids, w, gen = s["proj"].preferential_inputs(max_bytes=BIG)
    contrib = [np.array(c) for c in gen]
    assert tuple(ids) == ("grass",) and len(contrib) == 1 and np.allclose(contrib[0], vb.closed_form_pref_grass(dx, ny, nx), rtol=1e-9, atol=1e-15)
    ref = fpi.prepare_floating_preferential(b.root.column, a3=0.4, max_fraction_per_hour=0.7, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0, 1.0], species_receiver_weights=list(w),
                                            species_contributions=contrib, available_bytes_fn=AVAIL)
    assert nb.root.pref_op.sha256 == ref.sha256
    rec = vl.current_record(nb)
    assert rec.generation == 1 and rec.parent_recipe_id is None and rec.applied_matrix_op_sha256 == nb.root.matrix_op.sha256 and rec.applied_preferential_sha256 == nb.root.pref_op.sha256 and rec.applied_column_sha256 == nb.root.column.sha256
    assert np.allclose(rec.splash_cover, _total(dx, ny, nx), rtol=1e-9, atol=1e-15) and np.array_equal(rec.shade_cover, rec.splash_cover)  # all plants live: visible live+dead == live
    assert rec.splash_cover.flags.writeable is False and rec.shade_cover.flags.writeable is False and rec.splash_cover_2d.base is not None
    # the SAME binding is idempotent; the old head can no longer be used (stale)
    assert apply(lin, nb, bd) is nb and lin.head is nb
    with pytest.raises(lc.LifecycleError, match="stale|accepted head"):
        apply(lin, b, bd)


def _total(dx, ny, nx):
    import math

    f = lambda r: math.pi * (r / dx) ** 2
    t = np.zeros(ny * nx)
    t[0] = f(0.4 * dx)
    t[nx + 1] = min(f(0.3 * dx) + f(0.35 * dx), 1.0)
    return t


def test_refusals_leave_the_head_untouched_and_a_retry_works(tmp_path, leaf, monkeypatch):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    s = vb.stack(leaf, b)
    bd = vb.binding(leaf, b, s)
    head0, root0, ops0 = lin.head, b.root, (b.root.matrix_op, b.root.pref_op, b.root.history)
    calls = []
    real = wvb.prepare_vegetation_law
    monkeypatch.setattr(wvb, "prepare_vegetation_law", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    def untouched():
        assert lin.head is head0 and b.root is root0 and (b.root.matrix_op, b.root.pref_op, b.root.history) == ops0 and b.root.vegetation is None

    # stale surface: a binding built on another surface than the bed's CURRENT committed surface
    s2 = vb.stack(leaf, b, surface=s["surface"] + 1.0e-3)
    with pytest.raises(vl.VegetationLifecycleError, match="CURRENT committed surface"):
        apply(lin, b, vb.binding(leaf, b, s2))
    # wrong mask (one inactive cell)
    m = np.ones(s["surface"].shape, dtype=bool)
    m[2, 3] = False
    with pytest.raises(vl.VegetationLifecycleError, match="mask"):
        apply(lin, b, vb.binding(leaf, b, vb.stack(leaf, b, mask=m)))
    # forged / copied / non-binding inputs
    with pytest.raises(wvb.VegetationBindingError):
        copy.copy(bd)
    with pytest.raises(vl.VegetationLifecycleError, match="VegetationBinding"):
        apply(lin, b, object())
    with pytest.raises(vl.VegetationLifecycleError, match="genuine sealed|uninitialised"):
        apply(lin, b, object.__new__(wvb.VegetationBinding))
    # wrong / forged / non-boundary expected heads
    with pytest.raises(lc.LifecycleError):
        apply(lin, object(), bd)
    with pytest.raises(lc.LifecycleError):
        apply(lin, lin.head.root, bd)
    # memory: need - 1 refuses BEFORE any allocation (the S0 law preparation is never reached), the exact need accepts
    est = apply_estimate(b, bd)
    with pytest.raises(vl.VegetationLifecycleError, match="nothing allocated"):
        apply(lin, b, bd, memory_budget_bytes=est["peak_bytes"] - 1)
    with pytest.raises(vl.VegetationLifecycleError, match="reader"):
        apply(lin, b, bd, available_bytes_fn=lambda: est["peak_bytes"] - 1)
    assert calls == []
    untouched()
    nb = apply(lin, b, bd, memory_budget_bytes=est["peak_bytes"], available_bytes_fn=lambda: est["peak_bytes"])  # the exact modelled need is accepted (the head was retryable)
    assert lin.head is nb and calls == [1]
    # eligibility: only a COMPLETE boundary head; a DailyHold head is refused and a pending-recession / pending-storm boundary is refused
    br.open_day(lin, nb, partition=PART, run_digest="ab" * 32, available_bytes_fn=AVAIL)  # installs the DailyHold as the head
    assert isinstance(lin.head, lc.DailyHold)
    with pytest.raises(lc.LifecycleError):
        apply(lin, lin.head, bd)


def test_a_boundary_with_a_pending_state_is_not_eligible(tmp_path, leaf):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    bd = vb.binding(leaf, b, vb.stack(leaf, b))
    for field, value in (("role", "pending_recession"),):
        import dataclasses

        with lc._minting():
            fake = dataclasses.replace(b, **{field: value})  # a hand-minted lookalike is not a genuine record
        with pytest.raises(lc.LifecycleError):
            apply(lin, fake, bd)
    assert lin.head is b


def test_the_record_borrows_the_binding_but_retains_no_plant_population_owner_or_superseded_root(tmp_path, leaf):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    s = vb.stack(leaf, b)
    bd = vb.binding(leaf, b, s)
    refs = {"state": weakref.ref(s["state"]), "old_root": weakref.ref(b.root), "old_matrix": weakref.ref(b.root.matrix_op), "old_history": weakref.ref(b.root.history)}
    nb = apply(lin, b, bd)
    rec = vl.current_record(nb)
    assert rec.binding is bd and rec.splash_cover is not None and np.shares_memory(rec.splash_cover, s["proj"].splash_cover) and np.shares_memory(rec.shade_cover, s["proj"].shading_cover)  # borrowed, not copied
    assert rec.owned_bytes < 16384 and rec.borrowed_bytes == bd.resident_bytes + bd.borrowed_bytes
    pref = weakref.ref(s["proj"])
    del s, bd, b
    for _ in range(3):
        gc.collect()
    assert {k: r() is None for k, r in refs.items()} == dict.fromkeys(refs, True)  # no plant population, superseded root, old operator or old history
    assert pref() is not None and vl.verify_record(rec) is rec and lin.head is nb  # the borrowed derived projection (no plants) stays alive through the one binding


def test_the_record_is_fingerprinted_forgery_replacement_and_hand_building_refuse(tmp_path, leaf):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    s = vb.stack(leaf, b)
    nb = apply(lin, b, vb.binding(leaf, b, s))
    rec = vl.current_record(nb)
    assert vl.verify_record(rec) is rec
    for field, value in (("recipe_id", "0" * 64), ("generation", 7), ("surface_sha256", "1" * 64), ("binding", object()), ("splash_cover", np.zeros(rec.n_cells)), ("shade_cover", np.array(rec.shade_cover))):
        old = getattr(rec, field)
        object.__setattr__(rec, field, value)  # a frozen dataclass does not stop this
        with pytest.raises(vl.VegetationLifecycleError):
            vl.verify_record(rec)
        with pytest.raises(vl.VegetationLifecycleError):
            vl.current_record(nb)
        object.__setattr__(rec, field, old)
        assert vl.verify_record(rec) is rec
    with pytest.raises(vl.VegetationLifecycleError):
        vl.verify_record(object.__new__(vl.VegetationRecord))
    with pytest.raises(vl.VegetationLifecycleError, match="minted only"):
        vl.VegetationRecord(**{k: getattr(rec, k) for k in rec.__dataclass_fields__})
    with pytest.raises(vl.VegetationLifecycleError):
        copy.copy(rec)
    # the borrowed covers are immutable (the chain ends in bytes) and cannot be unlocked
    for a in (rec.splash_cover, rec.shade_cover):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    # the identities are the binding's, kept explicitly (source / helper pins / package)
    assert rec.source["kind"] == "initial_state" and rec.external_helpers == vb.helper_pin() and rec.leaf_package_sha256 == vb.leaf_pin()
    # a forged record in a head is refused by every consumer
    att = vl.vegetation_attribution(nb)
    assert att["generation"] == 1 and att["recipe_id"] == rec.recipe_id and att["binding_sha256"] == rec.binding_sha256 and vl.vegetation_attribution(b) is None


def test_dead_standing_counts_for_splash_only_and_the_live_laws_use_live_plants(tmp_path, leaf):
    import math

    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    ny, nx = b.surface_depth_m.shape
    dx = float(b.routing[0].dx_m)
    layout = vb.plant_layout(ny, nx, dx)
    layout[2] = (3, "grass", layout[2][2], layout[2][3], layout[2][4], "dead_standing")  # the second grass in cell (1,1) is a standing dead plant
    s = vb.stack(leaf, b, plants=layout)
    nb = apply(lin, b, vb.binding(leaf, b, s))
    rec = vl.current_record(nb)
    f = lambda r: math.pi * (r / dx) ** 2
    c = nx + 1
    assert rec.splash_cover[c] == pytest.approx(f(0.3 * dx) + f(0.35 * dx), rel=1e-9)  # visible live + dead_standing add (explicit add-and-cap)
    assert rec.shade_cover[c] == pytest.approx(f(0.35 * dx), rel=1e-9)  # live only: evaporation shading ignores the dead plant
    g = np.asarray(fm.held_cell_vegetation_factor(nb.root.matrix_op))
    r1 = f(0.35 * dx) * vb.FR3["shrub"][0]  # only the live shrub's top-layer roots
    assert g[c] == pytest.approx((vb.A1 + r1) / (vb.A2 + r1), rel=1e-9)


def test_a_live_application_requires_the_genuine_leaf_projection_before_any_water_or_head_work(tmp_path, leaf, monkeypatch):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    bd = vb.binding(leaf, b, vb.stack(leaf, b))
    forged = object.__new__(wvb.VegetationBinding)  # the right CLASS, a non-LEAF projection (what a cached derived binding of a later slice looks like)
    object.__setattr__(forged, "projection", type("CachedInputs", (), {})())
    calls = []
    monkeypatch.setattr(wvb, "prepare_vegetation_law", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(wvb, "verify_binding", lambda x: calls.append(2) or x)  # even a permissive binding check must not be what stops it
    with pytest.raises(vl.VegetationLifecycleError, match="genuine leaf.water_vegetation.VegetationProjection"):
        apply(lin, b, forged)
    assert calls == [] and lin.head is b and b.root.vegetation is None  # refused before any binding check, preparation, water or head work
    monkeypatch.undo()
    assert apply(lin, b, bd) is lin.head


def test_the_old_h0_export_refuses_a_per_cell_matrix_law_and_stays_unchanged_for_a_scalar_law(tmp_path, leaf):
    from maple_syrup import floating_held_context as fhc
    from maple_syrup import floating_pond as fp

    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    v0 = fp.FloatingPondVertical.from_boundary(b.root, available_bytes_fn=AVAIL)
    assert fhc.export_held_context(v0, available_bytes_fn=AVAIL, verify=False) is not None  # the old scalar record: unchanged
    nb = apply(lin, b, vb.binding(leaf, b, vb.stack(leaf, b)))
    v1 = fp.FloatingPondVertical.from_boundary(nb.root, available_bytes_fn=AVAIL)
    for verify in (False, True):
        with pytest.raises(fhc.HeldContextError, match="per-cell vegetation factor"):
            fhc.export_held_context(v1, available_bytes_fn=AVAIL, verify=verify)  # before any record is minted, verify or not


def test_cuda_arguments_on_a_cpu_boundary_are_refused_and_the_stable_recipe_id_is_reproducible(tmp_path, leaf):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    bd = vb.binding(leaf, b, vb.stack(leaf, b))
    for kw in ({"host_memory_budget_bytes": 1 << 20}, {"host_reserve_bytes": 5}, {"host_available_bytes_fn": AVAIL}):
        with pytest.raises(vl.VegetationLifecycleError, match="only to a CUDA boundary"):
            apply(lin, b, bd, **kw)
    assert lin.head is b
    nb = apply(lin, b, bd)
    rec = vl.current_record(nb)
    assert rec.recipe_id == vl.recipe_id_of(bd.sha256, 1, None) and rec.generation == 1


def test_a_genuinely_captured_and_staged_cached_binding_cannot_apply_a_live_generation(tmp_path, leaf, monkeypatch):
    """Negative control with the REAL cached path (capture -> export -> cold stage with external pins -> cached binding), not a forged object: the cached binding is a restored recipe whose projection is the typed cached record, never a LEAF
    projection, so a LIVE application refuses it before binding checks, preparation, water or head work; the head stays retryable."""
    import copy

    from maple_syrup import water_vegetation_held as wh

    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    s = vb.stack(leaf, b)
    live_b = vb.binding(leaf, b, s)
    rec = wh.capture_cached_inputs(live_b, max_bytes=BIG)
    exported = wh.export_cached_record(rec)
    meta, arrays = copy.deepcopy(exported["meta"]), {k: np.array(v) for k, v in exported["arrays"].items()}
    ident = s["proj"].identity_record()
    staged = wh.stage_cached_inputs(meta, arrays, expected_record_sha256=rec.record_sha256, expected_binding_sha256=live_b.sha256, expected_projection_sha256=ident["content_sha256"],
                                    expected_leaf_package_sha256=vb.leaf_pin(), expected_external_helpers=vb.helper_pin(), expected_source={k: ident["source"][k] for k in ("kind", "state_sha256", "catalog_hash")},
                                    active_mask=s["mask"], max_bytes=BIG)
    cached = wh.cached_binding(staged, expected_binding_sha256=live_b.sha256, max_bytes=BIG)
    assert cached.origin == wvb.ORIGIN_CACHED and cached.sha256 == live_b.sha256 and type(cached.projection) is not type(s["proj"])
    calls = []
    monkeypatch.setattr(wvb, "prepare_vegetation_law", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(wvb, "verify_binding", lambda x: calls.append(2) or x)
    with pytest.raises(vl.VegetationLifecycleError, match="genuine leaf.water_vegetation.VegetationProjection"):
        apply(lin, b, cached)
    assert calls == [] and lin.head is b and b.root.vegetation is None
    monkeypatch.undo()
    nb = apply(lin, b, live_b)  # the head was retryable with the genuine live binding
    assert lin.head is nb


def test_the_cell_area_of_the_binding_is_compared_exactly_before_any_hash_or_allocation(tmp_path, leaf):
    import inspect

    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    bd = vb.binding(leaf, b, vb.stack(leaf, b))
    assert float(bd.cell_area_m2) == float(b.root.column.cell_area_m2)  # the matched binding passes the exact comparison
    src = inspect.getsource(vl.apply_vegetation_generation)
    assert "binding.cell_area_m2" in src and src.index("column.cell_area_m2) != float(binding.cell_area_m2") < src.index("_hash_chunked")  # compared BEFORE the first hash
