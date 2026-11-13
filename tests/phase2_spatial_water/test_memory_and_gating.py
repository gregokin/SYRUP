"""Memory model/admission for EVERY allocating spatial entry point, warm peaks, no permanent derived (L, N) maps, lazy row lifetimes, one-layer columns, and the gating of the older APIs. Author-written."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from conftest import BIG, make_spec, params, prepare, water_kind

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss


def build(L=3, N=20000, mode="constant", bottom="open", mixed=False, seed=1):
    p = params(np.random.default_rng(seed), L, N, mode=mode)
    if mixed:
        spec = sp.spatial_layer_spec(p["edges"][:, 0].copy(), theta_residual=0.03, theta_field_capacity=p["fc"], theta_saturation=p["sat"], theta_evaporation_stress=0.5, diffusion_mode=mode,
                                     bottom_drainage=bottom, n_cells=N, **(dict(diffusivity_m2_s=2e-7) if mode == "constant" else dict(interface_conductivity_m_s=2e-6, diffusion_coefficient=0.3)))
        p = {"edges": np.repeat(p["edges"][:, :1], N, 1), "res": np.full((L, N), 0.03), "fc": p["fc"], "sat": p["sat"], "iface": p["iface"]}
        return p, prepare(spec)
    return p, prepare(make_spec(p, mode=mode, bottom=bottom))


def inputs(p, seed=3):
    L, N = p["res"].shape
    rng = np.random.default_rng(seed)
    sat = p["sat"] * np.diff(p["edges"], axis=0)
    return sat * rng.uniform(0, 1, size=(L, N)), rng.uniform(0, 1e-3, size=(L, N)) * 0.3, rng.uniform(0, 2e-7, size=N)


# ---- estimator ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_step_estimate_formulas(L):
    N = 1000
    est = sp.estimate_spatial_operation_memory("step", L, N, with_evaporation_share=True, reserve_bytes=17)
    out = (L * N + 2 * (L - 1) * N + 4 * N + ss.ROW_COUNT * N) * 8
    scratch = (ss.NUMPY_SCRATCH_DOUBLES_PER_CELL + sp.SPATIAL_ROW_TEMP_DOUBLES) * N * 8 + ss.SCRATCH_MASK_BYTES_PER_CELL * N + ss._PACKET_SCRATCH_BYTES + sp.PY_ALLOWANCE_BYTES
    assert (est.output_bytes, est.scratch_bytes, est.reserve_bytes, est.required_bytes) == (out, scratch, 17, out + scratch + 17)
    assert dict(est.input_bytes) == {"state.water_m": 8 * L * N, "potential_evaporation_m_s": 8 * N, "added_water_m": 8 * L * N, "evaporation_share": 8 * N}
    assert est.terms["no_permanent_LN_derived_maps"] and est.terms["no_internal_zero_forcing_tensor"] and est.terms["no_history"]
    pulse = sp.estimate_spatial_operation_memory("step_with_recharge", L, N)
    assert "added_water_m" not in pulse.input_bytes and pulse.required_bytes == out + scratch - 0 and pulse.input_bytes["state.water_m"] == 8 * L * N
    for op, expected_out in (("state", 8 * L * N), ("uniform_state", 8 * L * N), ("moisture", 8 * L * N), ("storage", 0)):
        assert sp.estimate_spatial_operation_memory(op, L, N).output_bytes == expected_out


def test_estimate_scales_linearly_in_cells_and_refuses_bad_arguments():
    a, b = sp.estimate_spatial_operation_memory("step", 3, 1000), sp.estimate_spatial_operation_memory("step", 3, 2000)
    assert b.required_bytes - sp.PY_ALLOWANCE_BYTES - ss._PACKET_SCRATCH_BYTES == 2 * (a.required_bytes - sp.PY_ALLOWANCE_BYTES - ss._PACKET_SCRATCH_BYTES)
    for bad in (("nope", 3, 10), ("step", 0, 10), ("step", 3, 0), ("step", True, 10)):
        with pytest.raises(sp.SpatialLayerError):
            sp.estimate_spatial_operation_memory(*bad)
    with pytest.raises(sp.SpatialLayerError):
        sp.estimate_spatial_operation_memory("step", 3, 10, with_evaporation_share=1)
    with pytest.raises(sp.SpatialLayerError):
        sp.estimate_spatial_operation_memory("step", 3, 10, reserve_bytes=-1)


def test_no_permanent_derived_maps_exist_on_the_owner_or_the_reader():
    p, col = build(N=64)
    assert set(col.retained_names()) == {"theta_residual", "theta_field_capacity", "theta_saturation", "dz", "pair_inverse_thickness", "interface_conductance_down"}  # the A1 set, nothing added
    params_ = sp._SpatialStepParams(col)
    for name, value in vars(params_).items():
        assert not isinstance(value, np.ndarray) or value.shape == (64,), name  # only the layer-0 residual row may be an (N,) view of a retained map; no (L, N) array
    assert isinstance(params_.sat_m, sp._Rows) and not isinstance(params_.cap_tol, np.ndarray)
    assert params_.sat_m[1].shape == (64,) and params_.sat_m[1] is not params_.sat_m[1]  # a fresh lazy row each time: no cache


# ---- admission for every allocating entry --------------------------------------------------------------------------------------------------
def entries(col, p, state):
    water, added, pet = inputs(p)
    L, N = water.shape
    return {
        "state": ("state", lambda **kw: sp.spatial_column_state(col, water, 0.0, **kw)),
        "uniform_state": ("uniform_state", lambda **kw: sp.spatial_uniform_state(col, [0.01] * L, 0.0, **kw)),
        "moisture": ("moisture", lambda **kw: sp.spatial_moisture(col, state, **kw)),
        "storage": ("storage", lambda **kw: sp.spatial_storage_m3(col, state, **kw)),
        "step": ("step", lambda **kw: sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, **kw)),
        "step_with_recharge": ("step_with_recharge", lambda **kw: sp.spatial_step_with_recharge(col, state, 3600.0, recharge_depth_m=1e-4, potential_evaporation_m_s=pet, **kw)),
    }


@pytest.mark.parametrize("which", ["state", "uniform_state", "moisture", "storage", "step", "step_with_recharge"])
def test_every_entry_is_admitted_before_it_allocates(which):
    p, col = build(N=3000)
    water, _, _ = inputs(p)
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    op, call = entries(col, p, state)[which]
    need = sp.estimate_spatial_operation_memory(op, 3, 3000).required_bytes
    tracemalloc.start()
    with pytest.raises(sp.SpatialLayerMemoryError):
        call(memory_budget_bytes=need - 1)
    with pytest.raises(sp.SpatialLayerMemoryError):
        call(memory_budget_bytes=need + 10, reserve_bytes=11)  # the reserve is inside the requirement, counted once
    with pytest.raises(sp.SpatialLayerMemoryError):
        call(memory_budget_bytes=BIG, available_bytes_fn=lambda: need - 1)
    for bad in (lambda: -1, lambda: 1.5, lambda: True):
        with pytest.raises(sp.SpatialLayerError):
            call(memory_budget_bytes=BIG, available_bytes_fn=bad)
    for kw in (dict(memory_budget_bytes=-1), dict(memory_budget_bytes=True), dict(memory_budget_bytes=BIG, reserve_bytes=-1), dict(memory_budget_bytes=BIG, available_bytes_fn=3)):
        with pytest.raises(sp.SpatialLayerError):
            call(**kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 200_000  # none of the refusals allocated an array of the call
    call(memory_budget_bytes=need, available_bytes_fn=lambda: need)  # exactly enough is admitted


# ---- warm peak against the modelled requirement ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("layout", ["cell", "mixed"])
@pytest.mark.parametrize("variant", ["step", "share", "pulse", "pulse_share"])
@pytest.mark.parametrize("L", [1, 3])
def test_warm_peak_of_a_step_is_within_the_modelled_requirement(mode, layout, variant, L):
    N = 40000
    p, col = build(L=L, N=N, mode=mode, bottom="closed", mixed=layout == "mixed", seed=7)
    water, added, pet = inputs(p)
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    share = np.full(N, 0.5) if "share" in variant else None
    pulse = "pulse" in variant

    def go():
        if pulse:
            return sp.spatial_step_with_recharge(col, state, 3600.0, recharge_depth_m=0.0, potential_evaporation_m_s=pet, evaporation_share=share, memory_budget_bytes=BIG)
        return sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, evaporation_share=share, memory_budget_bytes=BIG)

    go()  # warm: imports and first-call costs outside the measurement (state, forcing and the owner already exist)
    est = sp.estimate_spatial_operation_memory("step_with_recharge" if pulse else "step", L, N, with_evaporation_share=share is not None)
    tracemalloc.start()
    out = go()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    kept = sum(a.nbytes for a in (out.state.water_m, out.surface_return_m, out.drainage_m, out.evaporation_m, out.evaporation_unmet_m, out.percolation_m, out.diffusion_m))
    assert peak <= est.required_bytes, (peak, est.required_bytes, est.terms)
    assert kept <= est.output_bytes


@pytest.mark.parametrize("which", ["state", "uniform_state", "moisture", "storage"])
def test_warm_peaks_of_the_state_helpers(which):
    N = 40000
    p, col = build(N=N, seed=5)
    water, _, _ = inputs(p)
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    op, call = entries(col, p, state)[which]
    call(memory_budget_bytes=BIG)
    est = sp.estimate_spatial_operation_memory(op, 3, N)
    tracemalloc.start()
    call(memory_budget_bytes=BIG)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= est.required_bytes, (which, peak, est.required_bytes)


def test_a_one_layer_column_steps_without_interface_maps():
    N = 50
    p = params(np.random.default_rng(1), 1, N)
    col = prepare(make_spec(p, iface=2e-7))
    water, added, pet = inputs(p)
    out = sp.spatial_step(col, sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG), 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    assert out.percolation_m.shape == (0, N) and out.diffusion_m.shape == (0, N) and abs(out.budget.residual_m3) <= out.budget.bound_m3


# ---- gating of the older APIs ---------------------------------------------------------------------------------------------------------------
def test_old_entry_points_still_refuse_spatial_columns_and_states_typed():
    p, col = build(N=20)
    state = sp.spatial_uniform_state(col, [0.01] * 3, 0.0, memory_budget_bytes=BIG)
    import importlib
    import inspect

    n = 0
    for module, name in [("subsurface", n_) for n_ in ("admit_step", "column_state", "initialize_profile", "moisture", "reference_runner", "reference_step", "step", "step_memory",
                                                       "step_with_recharge", "storage_m3", "summary", "uniform_state")] + [
            ("matrix_infiltration", "prepare_matrix"), ("preferential_infiltration", "prepare_preferential"), ("soil_evaporation", "prepare_evaporation"),
            ("subsurface_transpiration", "estimate_combined_memory"), ("subsurface_transpiration", "prepare_transpiration"), ("layered_storm", "prepare_layered_storm"),
            ("water_lifecycle", "prepare_lifecycle")]:
        fn = getattr(importlib.import_module("maple_syrup." + module), name)
        req = [q for q in inspect.signature(fn).parameters.values() if q.default is q.empty and q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
        kwreq = {q.name: None for q in inspect.signature(fn).parameters.values() if q.default is q.empty and q.kind is q.KEYWORD_ONLY}
        with pytest.raises(ss.SubsurfaceError):
            fn(col, *[state if q.name == "state" else None for q in req[1:]], **kwreq)  # a spatial STATE is offered where the old API expects a state
        n += 1
    assert n == 19
