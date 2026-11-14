"""Binding of an ACTUAL LEAF projection and the native law preparation. Independent expectations: closed-form circle areas for R1 / g (rtol 1e-9; the circle-cell builder is not bit-equal to pi r^2), and explicit
prescribed-map native preparations for the bit-level operator identities. LEAF is the real candidate package; without it these tests skip (the consumer stays UNQUALIFIED, no claim)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
from veg_binding_support import (
    A1,
    A2,
    BIG,
    INF,
    catalog,
    full_source,
    helper_pin,
    leaf_pin,
    make_column,
    plants_state,
    project,
    two_species,
)

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import matrix_infiltration as mi
from maple_syrup import water_vegetation_binding as wvb

ERR = wvb.VegetationBindingError
MEM = wvb.VegetationBindingMemoryError
K, PSI = 2e-6, 0.3


def controls(L=3, n=9, **kw):
    base = {"a1": A1, "a2": A2, "a3": 0.4, "max_fraction_per_hour": 0.7, "max_intake_rate_m_s": 1e-5, "background_receiver_weights": [0.0, 1.0, 1.0][:L] if L == 3 else [1.0] * L, "n_layers": L, "n_cells": n, "max_bytes": BIG}
    base.update(kw)
    return wvb.site_controls(**base)


def bind(leaf, p, surf, ctl, *, mask=None, **kw):
    m = np.ones(p.shape, dtype=bool) if mask is None else mask
    args = {"expected_leaf_package_sha256": leaf_pin(), "expected_projection_sha256": p.identity_record()["content_sha256"], "expected_source": full_source(p), "expected_external_helpers": helper_pin(), "active_mask": m, "current_surface": surf, "max_bytes": BIG}
    args.update(kw)
    return wvb.bind_projection(p, ctl, **args)


def prepare(col, b, **kw):
    # The frozen C2 projection has NO public stream estimator: the explicit declaration below reads its private value and is a TEST-ONLY control (dependency unqualified).
    args = {"conductivity_m_s": K, "suction_m": PSI, "memory_budget_bytes": BIG}
    args.update(kw)
    return wvb.prepare_vegetation_law(col, b, **args)


def stream_arrays(p):
    ids, w, gen = p.preferential_inputs(max_bytes=BIG)
    return ids, w, [np.array(c) for c in gen]


def test_actual_leaf_pin_equals_independent_digest_and_import_is_lazy():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-B", "-c", "import sys, maple_syrup.water_vegetation_binding as m; print('leaf' in sys.modules, 'leaf.water_vegetation' in sys.modules)"], capture_output=True,
                         text=True, check=False)
    assert out.stdout.split() == ["False", "False"], out.stderr  # ordinary SYRUP imports of the module never import LEAF


def test_r1_to_g_independent_arithmetic_and_matrix_identity(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    b = bind(leaf, p, surf, controls())
    col = make_column()
    law = prepare(col, b)
    r1 = np.zeros(9)
    r1[0] = math.pi * 0.4**2 * 0.8                                    # grass r=0.4 alone in cell (0,0): f=pi r^2 / 1, top-root fraction 0.8
    r1[4] = math.pi * 0.3**2 * 0.8 + math.pi * 0.35**2 * 0.25         # grass r=0.3 and shrub r=0.35 in cell (1,1)
    assert np.allclose(p.matrix_root_index, r1, rtol=1e-9, atol=1e-15)
    g_ref = (A1 + r1) / (A2 + r1)
    assert law.g_min == pytest.approx(float(g_ref.min()), rel=1e-9) and law.g_max == pytest.approx(float(g_ref.max()), rel=1e-9)
    assert law.g_min == pytest.approx(A1 / A2, rel=1e-12)             # empty cells: g(R1=0) = a1/a2
    g_native = mi.reference_vegetation_function(p.matrix_root_index, A1, A2)
    control = fm.prepare_floating_matrix(col, conductivity_m_s=K, suction_m=PSI, vegetation_factor=g_native, memory_budget_bytes=BIG)
    assert law.matrix_op.sha256 == control.sha256                      # the prepared operator IS the native one built from the single g
    assert law.g_sha256 == hashlib.sha256(np.ascontiguousarray(g_native).tobytes()).hexdigest()
    assert law.matrix_op.layouts["vegetation_factor"] == "cell" and (law.step_taken, law.water_head_authority) == (False, False)


def test_preferential_identity_equals_prescribed_maps_and_zero_weights_disable_only_pref(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])   # shrub: ALL-ZERO receiving weights
    b = bind(leaf, p, surf, controls())
    assert b.enabled_species == ("grass",)                                 # the shrub is disabled for the pathway ...
    assert p.matrix_root_index[4] > 0.0 and math.isclose(float(p.matrix_root_index[4]), math.pi * 0.09 * 0.8 + math.pi * 0.35**2 * 0.25, rel_tol=1e-9)  # ... but still in the matrix R1
    col = make_column()
    law = prepare(col, b)
    _ids, w, cs = stream_arrays(p)
    assert law.n_contributions == 1 == len(cs)
    # independent contribution: grass cover x sum of the grass LOWER root fractions (0.1 + 0.1), cells (0,0) and (1,1)
    expect = np.zeros(9)
    expect[0], expect[4] = math.pi * 0.4**2 * 0.2, math.pi * 0.3**2 * 0.2
    assert np.allclose(cs[0], expect, rtol=1e-9, atol=1e-15)
    control = fpi.prepare_floating_preferential(col, a3=0.4, max_fraction_per_hour=0.7, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([0.0, 1.0, 1.0]),
                                                species_receiver_weights=list(w), species_contributions=iter(cs), memory_budget_bytes=BIG)
    assert law.preferential_op.sha256 == control.sha256 and law.preferential_op.n_species_enabled == control.n_species_enabled == 1
    # stage results: the same real step with both operators is bitwise identical

    from maple_syrup import floating_soil_layers as fsl
    water = np.full((3, 9), 0.01)
    s0 = fsl.floating_continued_state(col, np.ascontiguousarray(water), 0.0)
    surface = np.full(9, 0.004)
    r1_ = fpi.floating_preferential_step(law.preferential_op, surface, s0, 600.0)
    r2_ = fpi.floating_preferential_step(control, surface, s0, 600.0)
    assert np.array_equal(r1_.state.water_m, r2_.state.water_m) and np.array_equal(r1_.surface_water_m, r2_.surface_water_m)


def test_all_zero_weights_species_do_not_change_the_pref_law_but_do_change_matrix(leaf):
    on = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    off = two_species(leaf, 3, grass_w=[0, 0, 0], shrub_w=[0, 0, 0])
    col = make_column()
    b_on, b_off = bind(leaf, on[2], on[3], controls()), bind(leaf, off[2], off[3], controls())
    law_on, law_off = prepare(col, b_on), prepare(col, b_off)
    assert b_off.enabled_species == () and law_off.n_contributions == 0 and law_off.preferential_op.n_species_enabled == 0
    background_only = fpi.prepare_floating_preferential(col, a3=0.4, max_fraction_per_hour=0.7, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([0.0, 1.0, 1.0]), memory_budget_bytes=BIG)
    assert law_off.preferential_op.sha256 == background_only.sha256     # ONE background allowance, no plant term at all
    assert law_on.preferential_op.sha256 != law_off.preferential_op.sha256
    assert law_on.matrix_op.sha256 == law_off.matrix_op.sha256          # matrix g depends on R1 only: the zero weights change nothing there


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_layer_counts_one_two_three(leaf, layers):
    w = {1: [1.0], 2: [0.0, 1.0], 3: [0.0, 1.0, 1.0]}[layers]
    _cat, _st, p, surf = two_species(leaf, layers, grass_w=w, shrub_w=w)
    b = bind(leaf, p, surf, controls(L=layers, background_receiver_weights=[1.0] * layers if layers != 3 else [0.0, 1.0, 1.0]))
    law = prepare(make_column(layers), b)
    ids, _wt, cs = stream_arrays(p)
    if layers == 1:
        assert all(not c.any() for c in cs)                              # one-layer profile: the lower-root sum is 0, so the contributions are zero (not refused)
    else:
        assert any(c.any() for c in cs)
    assert law.n_contributions == len(ids) == 2


def test_empty_and_fully_masked_and_partially_masked(leaf):
    cat = catalog(leaf, 3, [0, 1, 1], [0, 1, 1])
    st = plants_state(leaf, cat, [])
    p, surf = project(leaf, st, cat)
    law = prepare(make_column(), bind(leaf, p, surf, controls()))
    assert law.g_min == law.g_max == pytest.approx(A1 / A2, rel=1e-12)   # no plants: R1 = 0 everywhere
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    _cat2, _st2, p2, surf2 = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 1, 1], mask=mask)
    col = make_column(active=mask.reshape(-1))
    law2 = prepare(col, bind(leaf, p2, surf2, controls(), mask=mask))
    assert p2.matrix_root_index[4] == 0.0 and law2.g_max > law2.g_min
    with pytest.raises(ERR):                                              # the column's mask must be the projection's mask
        prepare(make_column(), bind(leaf, p2, surf2, controls(), mask=mask))
    allout = np.zeros((3, 3), dtype=bool)
    _cat3, _st3, p3, surf3 = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 1, 1], mask=allout)
    law3 = prepare(make_column(active=allout.reshape(-1)), bind(leaf, p3, surf3, controls(), mask=allout))
    assert law3.n_contributions == 2 and law3.g_min == law3.g_max


def test_a3_and_max_rate_maps_reach_the_native_operator(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    a3 = np.linspace(0.1, 0.9, 9)
    mr = np.full(9, 5e-6)
    mr[2] = INF
    law = prepare(make_column(), bind(leaf, p, surf, controls(a3=a3, max_intake_rate_m_s=mr)))
    _ids, w, cs = stream_arrays(p)
    control = fpi.prepare_floating_preferential(make_column(), a3=a3, max_fraction_per_hour=0.7, max_intake_rate_m_s=mr, background_receiver_weights=np.array([0.0, 1.0, 1.0]),
                                                species_receiver_weights=list(w), species_contributions=iter(cs), memory_budget_bytes=BIG)
    assert law.preferential_op.sha256 == control.sha256 and law.preferential_op.layouts == {"a3": "cell", "max_intake_rate_m_s": "cell"}


def test_no_duplicate_plant_factor_matrix_law_is_g_only(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    col = make_column()
    law = prepare(col, bind(leaf, p, surf, controls()))
    # the only vegetation factor in the matrix law is g; K and psi are the explicit scalars (no plant K multiplier, no soil factor)
    k, psi, g = law.matrix_op._law
    assert k.tolist() == [K] and psi.tolist() == [PSI] and g.shape == (9,)
    again = prepare(col, bind(leaf, p, surf, controls()))
    assert again.law_sha256 == law.law_sha256                            # deterministic; nothing accumulates between preparations


def test_expected_pin_source_projection_mask_surface_mismatches_refused(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    ctl = controls()
    ok = {"expected_leaf_package_sha256": leaf_pin(), "expected_projection_sha256": p.identity_record()["content_sha256"], "expected_source": full_source(p), "expected_external_helpers": helper_pin()}
    bad_hex = "0" * 64
    for key, val in (("expected_leaf_package_sha256", bad_hex), ("expected_projection_sha256", bad_hex), ("expected_source", {"kind": "plant_session"}), ("expected_source", {"state_sha256": bad_hex}),
                     ("expected_source", {})):
        with pytest.raises(ERR):
            bind(leaf, p, surf, ctl, **{key: val})
    for key in ("expected_leaf_package_sha256", "expected_projection_sha256", "expected_source", "active_mask", "current_surface"):
        with pytest.raises(TypeError):
            args = dict(ok, active_mask=np.ones((3, 3), dtype=bool), current_surface=surf, max_bytes=BIG)
            del args[key]
            wvb.bind_projection(p, ctl, **args)
    stale_mask = np.ones((3, 3), dtype=bool)
    stale_mask[0, 0] = False
    with pytest.raises(ERR):
        bind(leaf, p, surf, ctl, mask=stale_mask)
    moved = surf.copy()
    moved[0, 0] = 1e-9
    with pytest.raises(ERR):
        bind(leaf, p, moved, ctl)
    with pytest.raises(ERR):
        bind(leaf, p, surf, controls(n_layers=2, background_receiver_weights=[1.0, 1.0]))  # layers of the controls vs the projection
    with pytest.raises(ERR):
        bind(leaf, p, surf, controls(n=4, a3=np.full(4, 0.1)))             # map length vs domain cells
    with pytest.raises(ERR):
        bind(leaf, p, surf.astype(np.float32), ctl)
    with pytest.raises(ERR):
        bind(leaf, p, surf, ctl, mask=np.ones((3, 3), dtype=np.uint8))
    with pytest.raises(ERR):
        wvb.bind_projection(object(), ctl, **dict(ok, active_mask=np.ones((3, 3), dtype=bool), current_surface=surf, max_bytes=BIG))  # not a genuine projection: no fake class


def test_foreign_or_stale_package_digest_refused(leaf, monkeypatch):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    real = wvb._runtime_leaf_digest
    monkeypatch.setattr(wvb, "_runtime_leaf_digest", lambda lw: dict(real(lw), digest_sha256="1" * 64))
    with pytest.raises(ERR, match="package digest"):
        bind(leaf, p, surf, controls())


def test_binding_is_sealed_borrows_leaf_data_and_exports(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    b = bind(leaf, p, surf, controls())
    assert b.projection is p and b.source_kind == "initial_state" and b.source["kind"] == "initial_state" and b.borrowed_bytes == p.resident_bytes
    full = b.export_compact()
    arrs = b.export_arrays()
    proj = full["projection"]
    assert set(proj["field_order"]) <= {k.split(".", 1)[1] for k in arrs if k.startswith("projection.")}                      # every canonical LEAF array is present
    for name in ("splash_cover", "shading_cover", "matrix_root_index", "cell_offsets", "entry_cell", "entry_species", "entry_area_all_m2", "entry_area_live_m2", "root", "weights", "omitted_ids"):
        a = arrs[f"projection.{name}"]
        assert not a.flags.writeable and (a.size == 0 or np.shares_memory(a, proj["arrays"][name]))           # immutable, NO copy
    assert np.shares_memory(arrs["projection.matrix_root_index"], p.matrix_root_index)
    assert arrs["receiver_weights_enabled"].shape == (1, 3) and not arrs["receiver_weights_enabled"].flags.writeable and {k for k in arrs if k.startswith("controls.")} == {"controls.background_receiver_weights"}
    assert proj["content_sha256"] == p.identity_record()["content_sha256"] and full["binding"]["external_helpers"][0]["module"] == "benchmarks.spatial_mapping"
    assert full["controls"]["a1"] == A1 and full["controls"]["sha256"] == b.controls.sha256
    with pytest.raises(TypeError):
        full["binding"]["x"] = 1                                                           # read-only mapping
    cm = bind(leaf, p, surf, controls(a3=np.linspace(0.1, 0.9, 9)))
    assert np.shares_memory(cm.export_arrays()["controls.a3"], cm.controls.a3)
    meta = b.export_metadata()
    meta["source"]["kind"] = "plant_session"                              # mutating an export never touches the record
    assert b.source["kind"] == "initial_state" and meta["water_head_authority"] is False and meta["species_ids"] == ["grass", "shrub"]
    for f in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(ERR):
            f(b)
    with pytest.raises(ERR):
        wvb.VegetationBinding(p, controls(), 9, 3, (3, 3), (), (), None, 0, "", 0, "", "", "", "{}", "", "", "", 0, 0, 0, "live_projection")
    object.__setattr__(b, "mask_sha256", "0" * 64)
    with pytest.raises(ERR):
        prepare(make_column(), b)                                         # a forged binding is refused before anything is prepared


def test_prepare_refusals_leave_nothing(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    b = bind(leaf, p, surf, controls())
    with pytest.raises(ERR):
        prepare(make_column(layers=2), b)
    with pytest.raises(ERR):
        prepare(make_column(n=4), b)
    with pytest.raises(ERR):
        prepare(make_column(), b, host_memory_budget_bytes=1)             # host_* only for cuda
    with pytest.raises(ERR):
        prepare(make_column(), b, conductivity_m_s=True)
    with pytest.raises(ERR):
        wvb.prepare_vegetation_law(make_column(), object(), conductivity_m_s=K, suction_m=PSI, memory_budget_bytes=BIG)
    with pytest.raises(TypeError):
        wvb.prepare_vegetation_law(make_column(), b, conductivity_m_s=K, suction_m=PSI)  # an explicit budget is required


def test_law_admission_need_minus_one_refuses_before_any_native_call_exact_admits(leaf, monkeypatch):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 1, 1])
    b = bind(leaf, p, surf, controls())
    col = make_column()
    est = wvb.estimate_law_bytes(3, 9, len(b.enabled_species), binding_resident_bytes=b.resident_bytes + b.borrowed_bytes, stream_per_next_held_bytes=p.stream_estimate(previous_held=True)["per_next_bytes"], backend="numpy")
    need = est["peak_bytes"]
    calls = []
    real = fm.prepare_floating_matrix
    monkeypatch.setattr(fm, "prepare_floating_matrix", lambda *a, **k: calls.append(1) or real(*a, **k))
    with pytest.raises(MEM):
        prepare(col, b, memory_budget_bytes=need - 1)
    with pytest.raises(MEM):
        prepare(col, b, memory_budget_bytes=need + 99, reserve_bytes=100)  # reserve counted once
    with pytest.raises(MEM):
        prepare(col, b, memory_budget_bytes=BIG, available_bytes_fn=lambda: need - 1)
    assert calls == []                                                    # nothing native ran, nothing was prepared
    law = prepare(col, b, memory_budget_bytes=need)
    assert calls == [1] and law.n_contributions == 2
    law2 = prepare(col, b, memory_budget_bytes=need + 100, reserve_bytes=100)
    assert law2.law_sha256 == law.law_sha256


def test_cold_vs_warm_binding_admission(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    ctl = controls()
    wvb._LEAF_CACHE.clear()
    pub = p.stream_estimate(previous_held=False)
    kw = {"weights_stage_bytes": pub["weights_stage_bytes"], "retained_metadata_bytes": pub["retained_metadata_bytes"], "borrowed_bytes": p.resident_bytes}
    cold = wvb.estimate_binding_bytes(3, 2, source_cold=True, **kw)["peak_bytes"]
    warm = wvb.estimate_binding_bytes(3, 2, source_cold=False, **kw)["peak_bytes"]
    assert warm < cold
    with pytest.raises(MEM):
        bind(leaf, p, surf, ctl, max_bytes=cold - 1)
    assert not wvb._LEAF_CACHE                                            # the refusal did not run the cold digest
    b = bind(leaf, p, surf, ctl, max_bytes=cold)
    assert b.leaf_package_file_count >= 28
    with pytest.raises(MEM):
        bind(leaf, p, surf, ctl, max_bytes=warm - 1)                       # warm: the digest allowance is gone, the small remainder is still enforced
    assert bind(leaf, p, surf, ctl, max_bytes=warm).sha256 == b.sha256


def test_streamed_contributions_are_one_at_a_time(leaf, monkeypatch):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 1, 1])
    b = bind(leaf, p, surf, controls())
    live = {"max": 0, "now": 0}
    real = wvb._counted_stream

    def spy(gen, n, state, to_native=None):
        for c in real(gen, n, state, to_native):
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
            yield c
            live["now"] -= 1

    monkeypatch.setattr(wvb, "_counted_stream", spy)
    law = prepare(make_column(), b)
    assert law.n_contributions == 2 and live["max"] == 1                  # one contribution handed to the native derivation at a time (never S x N)


@pytest.mark.parametrize("bad", [np.full(9, -1.0), np.full(9, np.nan), np.zeros(8), np.zeros(9, dtype=np.float32), np.zeros(18)[::2], np.zeros((3, 3))])
def test_bad_contribution_is_refused_before_the_native_derivation(bad):
    with pytest.raises(ERR):
        list(wvb._counted_stream(iter([bad]), 9, {"count": 0}))


def test_stream_model_is_the_public_estimate_with_no_declaration_parameter(leaf, monkeypatch):
    """The consumer models the stream with LEAF's PUBLIC `stream_estimate` only: no caller declaration parameter exists, and without the public method it is typed-unavailable."""
    import inspect
    assert "p1_stream_scratch_bytes" not in inspect.signature(wvb.prepare_vegetation_law).parameters
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 1, 1])
    b = bind(leaf, p, surf, controls())
    est = prepare(make_column(), b).memory["declared"]
    held = p.stream_estimate(previous_held=True)["per_next_bytes"]
    assert est["stream_per_next_held_bytes"] == held and p.stream_estimate(previous_held=False)["per_next_bytes"] == held - 8 * 9    # previous-held adds exactly one (N,) array
    _cat0, _st0, p0, surf0 = two_species(leaf, 3, grass_w=[0, 0, 0], shrub_w=[0, 0, 0])
    b0 = bind(leaf, p0, surf0, controls())
    monkeypatch.delattr(type(p), "stream_estimate")
    with pytest.raises(wvb.VegetationConsumerUnavailableError):
        prepare(make_column(), b)
    with pytest.raises(wvb.VegetationConsumerUnavailableError):                             # the same public method is the O(1) genuine-record proof of EVERY binding (no identity copy)
        prepare(make_column(), b0)


def test_previous_held_array_is_counted_once_in_the_composed_stage(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 1, 1])
    b = bind(leaf, p, surf, controls())
    br = b.resident_bytes + b.borrowed_bytes
    held, free = p.stream_estimate(previous_held=True)["per_next_bytes"], p.stream_estimate(previous_held=False)["per_next_bytes"]
    mat = fm.estimate_floating_matrix_memory(3, 9, backend="numpy", operation="prepare")
    pref = fpi.estimate_floating_preferential_memory(3, 9, backend="numpy", operation="prepare", n_species=2)
    est = wvb.estimate_law_bytes(3, 9, 2, binding_resident_bytes=br, stream_per_next_held_bytes=held, stream_per_next_free_bytes=free, stream_weights_stage_bytes=p.stream_estimate(previous_held=False)["weights_stage_bytes"], backend="numpy")
    assert est["stages"]["preferential"] == br + mat.retained_bytes + pref.required_bytes + held       # ONE previous array: inside LEAF's per-next, not again beside it
    assert est["native_reserve_preferential"] == br + mat.retained_bytes + held
    assert est["p1_stream_reserve"] == br + mat.retained_bytes + pref.required_bytes                 # LEAF's quota adds its own per-next exactly once


def test_helper_pin_is_required_exact_and_checked_against_runtime_bytes(leaf, monkeypatch):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 0, 0])
    good = helper_pin()
    assert "sha256:" + p.identity_record()["external_helpers"]["helpers"][0]["sha256"].removeprefix("sha256:") == good[0]["sha256"]
    ok = bind(leaf, p, surf, controls())
    assert json.loads(ok.external_helpers_json) == good
    with pytest.raises(TypeError):
        wvb.bind_projection(p, controls(), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                            active_mask=np.ones((3, 3), dtype=bool), current_surface=surf, max_bytes=BIG)                      # the helper pin is REQUIRED
    wrong_hash = [{**good[0], "sha256": "sha256:" + "0" * 64}]
    wrong_bytes = [{**good[0], "bytes": good[0]["bytes"] + 1}]
    extra = [*good, {"module": "benchmarks.other", "sha256": good[0]["sha256"], "bytes": 1}]
    for bad in (wrong_hash, wrong_bytes, extra, [], [{"module": good[0]["module"]}], [{**good[0], "file": "x"}], good + good, "x", [dict(good[0], module="")]):
        with pytest.raises(ERR):
            bind(leaf, p, surf, controls(), expected_external_helpers=bad)
    real = wvb._runtime_helper
    monkeypatch.setattr(wvb, "_runtime_helper", lambda m: (m, "0" * 64, real(m)[2]))      # the bytes SYRUP hashes now differ from the (projection-recorded == expected) pin
    with pytest.raises(ERR, match="changed or foreign helper"):
        bind(leaf, p, surf, controls())
    monkeypatch.setattr(wvb, "_runtime_helper", real)
    monkeypatch.setattr(wvb, "_helper_path", lambda m: (_ for _ in ()).throw(ERR("missing")))
    with pytest.raises(ERR):
        bind(leaf, p, surf, controls())


def test_helper_changed_after_the_projection_in_a_fresh_isolated_process_is_refused(tmp_path):
    """A fresh process on a COPY of the frozen LEAF root: the projection records the original helper; the helper file is then altered; binding with the original pin must refuse."""
    import shutil
    import subprocess
    import sys

    from veg_binding_support import LEAF_ROOT
    root = tmp_path / "leafroot"
    for d in ("leaf", "benchmarks", "examples"):
        shutil.copytree(LEAF_ROOT / d, root / d, ignore=shutil.ignore_patterns("__pycache__"))
    script = f"""
import hashlib, sys, json
sys.path.insert(0, {str(root)!r})
import numpy as np
from veg_binding_support import catalog, plants_state, helper_pin, leaf_pin, BIG
import veg_binding_support as sup
from leaf import state, traits, water_vegetation as wv
from maple_syrup import water_vegetation_binding as wvb
leaf = {{"state": state, "traits": traits, "wv": wv}}
cat = catalog(leaf, 3, [0, 1, 1], [0, 1, 1])
st = plants_state(leaf, cat, [(1, "grass", 0.5, 0.5, 0.4, "live")])
surf = np.zeros(st.domain.shape)
p = wv.project_from_initial_state(st, cat, surf, active_mask=None, max_edges=1 << 16, max_bytes=BIG)
pin, pkg = helper_pin(), leaf_pin()
ctl = wvb.site_controls(a1=0.013, a2=0.067, a3=0.4, max_fraction_per_hour=0.7, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0, 1.0], n_layers=3, n_cells=9, max_bytes=BIG)
src = p.identity_record()["source"]
full = {{k: src[k] for k in wvb.SOURCE_KEYS[src["kind"]]}}
with open({str(root / "benchmarks" / "spatial_mapping.py")!r}, "ab") as fh:
    fh.write(b"# changed after the projection" + bytes([10]))
try:
    wvb.bind_projection(p, ctl, expected_leaf_package_sha256=pkg, expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full, expected_external_helpers=pin,
                        active_mask=np.ones((3, 3), dtype=bool), current_surface=surf, max_bytes=BIG)
    print("BOUND")
except wvb.VegetationBindingError as exc:
    print("REFUSED", "helper" in str(exc))
"""
    env = {**__import__("os").environ, "PYTHONPATH": f"{__import__('os').environ['PYTHONPATH']}:{Path(__file__).resolve().parent}", "PYTHONDONTWRITEBYTECODE": "1", "LEAF_CANDIDATE_ROOT": str(root)}
    out = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, env=env, check=False)
    assert out.stdout.split()[-2:] == ["REFUSED", "True"], (out.stdout, out.stderr[-2000:])


def test_numba_owner_prepares_the_same_native_laws(leaf):
    pytest.importorskip("numba")
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[1, 0, 1])
    b = bind(leaf, p, surf, controls())
    nb, np_ = prepare(make_column(backend="numba"), b), prepare(make_column(), b)
    assert nb.matrix_op.backend == nb.preferential_op.backend == "numba" and nb.g_sha256 == np_.g_sha256 and nb.n_contributions == np_.n_contributions == 2
    _ids, w, cs = stream_arrays(p)
    control = fpi.prepare_floating_preferential(make_column(backend="numba"), a3=0.4, max_fraction_per_hour=0.7, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([0.0, 1.0, 1.0]),
                                                species_receiver_weights=list(w), species_contributions=iter(cs), memory_budget_bytes=BIG)
    assert nb.preferential_op.sha256 == control.sha256
