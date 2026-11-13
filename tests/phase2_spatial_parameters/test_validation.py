"""Metadata and value refusals, flag hierarchy, first-offender reports, parity of refusal with the real `soil_layers`, validation of every cell (inactive included). Author-written."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import BIG, common_cell, make_spec, params, prepare

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss


def base(L=3, N=6, mode="constant", seed=1):
    return params(np.random.default_rng(seed), L, N, mode=mode)


# ---- metadata ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mutate", [
    lambda p: p.__setitem__("edges", p["edges"].astype(np.float32)),
    lambda p: p.__setitem__("res", p["res"].astype(np.int64)),
    lambda p: p.__setitem__("edges", np.asfortranarray(p["edges"])),
    lambda p: p.__setitem__("sat", p["sat"][:-1]),
    lambda p: p.__setitem__("sat", p["sat"][:, :-1]),
    lambda p: p.__setitem__("edges", p["edges"][:-1]),
    lambda p: p.__setitem__("fc", np.ma.masked_array(p["fc"])),
    lambda p: p.__setitem__("res", p["res"].tolist()),
    lambda p: p.__setitem__("edges", 0.5),
    lambda p: p.__setitem__("res", True),
    lambda p: p.__setitem__("iface", p["iface"][:, :-1]),
    lambda p: p.__setitem__("iface", np.zeros((5, 6))),
], ids=["edges_f32", "res_int", "edges_fortran", "sat_short", "sat_cols", "edges_short", "masked", "list", "edges_scalar", "bool_theta", "iface_cols", "iface_rows"])
def test_metadata_refusals(mutate):
    p = base()
    mutate(p)
    with pytest.raises(sp.SpatialLayerError):
        make_spec(p, N=6)


def test_non_contiguous_and_n_cells_mismatch():
    p = base(N=4)
    big = np.zeros((p["sat"].shape[0], 8))
    big[:, ::2] = p["sat"]
    with pytest.raises(sp.SpatialLayerError):
        make_spec({**p, "sat": big[:, ::2]})
    with pytest.raises(sp.SpatialLayerError):
        make_spec(p, N=5)  # per-cell arrays disagree with n_cells
    with pytest.raises(sp.SpatialLayerError):
        sp.spatial_layer_spec(p["edges"], theta_residual=p["res"], theta_field_capacity=p["fc"], theta_saturation=p["sat"], theta_evaporation_stress=0.5, diffusivity_m2_s=p["iface"], n_cells=0)


@pytest.mark.parametrize("kw", [dict(diffusivity_m2_s=None), dict(interface_conductivity_m_s=1e-6), dict(diffusion_coefficient=0.3)])
def test_constant_mode_input_rules(kw):
    p = base()
    args = dict(theta_residual=p["res"], theta_field_capacity=p["fc"], theta_saturation=p["sat"], theta_evaporation_stress=0.5, diffusivity_m2_s=p["iface"], n_cells=6)
    args.update(kw)
    with pytest.raises(sp.SpatialLayerError):
        sp.spatial_layer_spec(p["edges"], **args)


@pytest.mark.parametrize("kw", [dict(diffusivity_m2_s=1e-7), dict(interface_conductivity_m_s=None), dict(diffusion_coefficient=None), dict(diffusion_coefficient=-0.1),
                                dict(diffusion_coefficient=float("nan")), dict(diffusion_coefficient=True)])
def test_donor_mode_input_rules(kw):
    p = base(mode="tietjen_donor_depth")
    args = dict(theta_residual=p["res"], theta_field_capacity=p["fc"], theta_saturation=p["sat"], theta_evaporation_stress=0.5, interface_conductivity_m_s=p["iface"],
                diffusion_coefficient=0.3, diffusion_mode="tietjen_donor_depth", n_cells=6)
    args.update(kw)
    with pytest.raises(ss.SubsurfaceError):
        sp.spatial_layer_spec(p["edges"], **args)


@pytest.mark.parametrize("kw", [dict(diffusion_mode="x"), dict(bottom_drainage="x"), dict(gravity_drainage=1), dict(theta_evaporation_stress=float("inf")),
                                dict(evaporation_factor=-1.0), dict(evaporation_factor=True), dict(n_cells=1.5), dict(n_cells=True)])
def test_option_refusals(kw):
    p = base()
    args = dict(theta_residual=p["res"], theta_field_capacity=p["fc"], theta_saturation=p["sat"], theta_evaporation_stress=0.5, diffusivity_m2_s=p["iface"], n_cells=6)
    args.update(kw)
    with pytest.raises(ss.SubsurfaceError):
        sp.spatial_layer_spec(p["edges"], **args)


def test_single_layer_interface_input_rules():
    p = base(L=1)
    for good in (1e-7, np.zeros(0), np.zeros((0, 6))):
        make_spec(p, iface=good)
    for bad in (np.zeros(1), np.zeros((1, 6)), True):
        with pytest.raises(sp.SpatialLayerError):
            make_spec(p, iface=bad)


@pytest.mark.parametrize("mask", [np.ones(6, dtype=np.uint8), np.ones(5, dtype=bool), np.ones((6, 1), dtype=bool), [True] * 6, np.ones(12, dtype=bool)[::2]])
def test_mask_refusals(mask):
    with pytest.raises(sp.SpatialLayerError):
        prepare(make_spec(base()), mask=mask)


@pytest.mark.parametrize("area", [0.0, -1.0, float("nan"), float("inf"), True, "1", None])
def test_area_refusals(area):
    with pytest.raises(sp.SpatialLayerError):
        prepare(make_spec(base()), area=area)


@pytest.mark.parametrize("kw", [dict(backend="cupy"), dict(memory_budget_bytes=-1), dict(memory_budget_bytes=1.5), dict(reserve_bytes=-1), dict(chunk_columns=0),
                                dict(hash_chunk_bytes=4), dict(available_bytes_fn=3), dict(available_bytes_fn=lambda: -1), dict(available_bytes_fn=lambda: True),
                                dict(host_reserve_bytes=1), dict(host_memory_budget_bytes=10**9), dict(host_available_bytes_fn=lambda: 10**9)])
def test_option_refusals_of_prepare(cpu_backend, kw):
    kw = dict(kw)
    backend = kw.pop("backend", cpu_backend)
    budget = kw.pop("memory_budget_bytes", BIG)
    spec = make_spec(base())
    with pytest.raises(sp.SpatialLayerError if backend != "cupy" else sp.SpatialLayerError):
        sp.prepare_spatial_column(spec, np.ones(6, dtype=bool), 1.0, backend=backend, memory_budget_bytes=budget, **kw)


# ---- values ---------------------------------------------------------------------------------------------------------------------------
def set_cell(p, key, idx, value):
    q = {k: v.copy() for k, v in p.items()}
    q[key][idx] = value
    return q


@pytest.mark.parametrize("name,key,idx,value,bit", [
    ("edge_nan", "edges", (1, 2), np.nan, sp.FLAG_EDGES), ("edge_inf", "edges", (3, 2), np.inf, sp.FLAG_EDGES), ("edge0_not_zero", "edges", (0, 2), 0.1, sp.FLAG_EDGES),
    ("collapsed", "edges", (2, 2), "same_as_prev", sp.FLAG_EDGES), ("decreasing", "edges", (2, 2), "below_prev", sp.FLAG_EDGES), ("width_overflow", "edges", (1, 2), 1.7e308, sp.FLAG_EDGES),
    ("res_nan", "res", (1, 2), np.nan, sp.FLAG_THETA), ("sat_inf", "sat", (0, 2), np.inf, sp.FLAG_THETA),
    ("res_negative", "res", (1, 2), -1e-9, sp.FLAG_REL), ("fc_below_res", "fc", (1, 2), "below_res", sp.FLAG_REL), ("sat_below_fc", "sat", (1, 2), "below_fc", sp.FLAG_REL),
    ("sat_above_one", "sat", (2, 2), 1.0000001, sp.FLAG_REL), ("sat_equals_res", "sat", (1, 2), "res_eq", sp.FLAG_REL),
    ("iface_negative", "iface", (0, 2), -1e-12, sp.FLAG_IFACE), ("iface_nan", "iface", (1, 2), np.nan, sp.FLAG_IFACE), ("iface_inf", "iface", (0, 2), np.inf, sp.FLAG_IFACE)])
def test_value_refusals_report_the_first_offending_cell_and_the_lowest_flag(cpu_backend, name, key, idx, value, bit):
    p = base(L=3, N=6)
    k, c = idx
    if value == "same_as_prev":
        value = p["edges"][k - 1, c]
    elif value == "below_prev":
        value = p["edges"][k - 1, c] - 0.01
    elif value == "below_res":
        value = p["res"][k, c] - 0.001
    elif value == "below_fc":
        value = p["fc"][k, c] - 0.001
    elif value == "res_eq":
        value = p["res"][k, c]
    q = set_cell(p, key, idx, value)
    if name == "width_overflow":
        q["edges"][0, c] = -1.7e308
        q["edges"][1:, c] = np.array([1.7e308, 1.7e308, 1.7e308])
    with pytest.raises(sp.SpatialLayerError) as info:
        prepare(make_spec(q), backend=cpu_backend)
    assert info.value.first_cell == c and info.value.n_flagged == 1
    assert info.value.bit == bit, (name, info.value.bit)


def test_stress_flag_and_every_cell_is_validated_including_inactive_ones(cpu_backend):
    p = base()
    p["res"][0, 4] = 0.6 - 0.01  # stress 0.5 is not above this layer-0 residual in cell 4 (and sat is raised to keep the thetas ordered)
    p["fc"][0, 4], p["sat"][0, 4] = 0.7, 0.8
    mask = np.ones(6, dtype=bool)
    mask[4] = False  # the offending cell is INACTIVE: it still must carry valid placeholders
    with pytest.raises(sp.SpatialLayerError) as info:
        prepare(make_spec(p), backend=cpu_backend, mask=mask)
    assert info.value.bit == sp.FLAG_STRESS and info.value.first_cell == 4


def test_derived_unrepresentable_values_are_refused_after_input_checks_pass(cpu_backend):
    p = base(L=3, N=4)
    q = {k: v.copy() for k, v in p.items()}
    q["iface"][0, 1] = 1e308  # finite input, but D / separation overflows when the separation is small: derived flag only
    q["edges"][:, 1] = np.array([0.0, 1e-3, 2e-3, 3e-3])
    with pytest.raises(sp.SpatialLayerError) as info:
        prepare(make_spec(q), backend=cpu_backend)
    assert info.value.bit == sp.FLAG_DERIVED and info.value.first_cell == 1
    tiny = {k: v.copy() for k, v in p.items()}
    tiny["edges"][:, 2] = np.array([0.0, 1e-320, 2e-320, 3e-320])  # subnormal widths: the inverse overflows
    with pytest.raises(sp.SpatialLayerError) as info:
        prepare(make_spec(tiny), backend=cpu_backend)
    assert info.value.bit == sp.FLAG_DERIVED and info.value.first_cell == 2


def test_input_flags_take_precedence_over_derived_flags(cpu_backend):
    p = base(L=3, N=3)
    p["edges"][:, 1] = np.array([0.0, 1e-320, 2e-320, 3e-320])  # would also be derived-unrepresentable...
    p["res"][1, 1] = -0.5  # ...but the input check fails first and is the ONLY reported class for that cell
    with pytest.raises(sp.SpatialLayerError) as info:
        prepare(make_spec(p), backend=cpu_backend)
    assert info.value.bit == sp.FLAG_REL and info.value.n_flagged == 1


def test_several_flagged_cells_are_counted_and_the_first_is_reported(cpu_backend):
    p = base(L=2, N=9)
    for c in (7, 3, 5):
        p["res"][0, c] = -1.0
    p["edges"][1, 8] = np.nan
    with pytest.raises(sp.SpatialLayerError) as info:
        prepare(make_spec(p), backend=cpu_backend, chunk_columns=2)
    assert (info.value.first_cell, info.value.n_flagged) == (3, 4)


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
def test_refusal_parity_with_the_real_soil_layers_on_random_corruptions(cpu_backend, mode):
    """Single-cell random corruptions (including extreme magnitudes): the spatial preparation refuses exactly when the accepted `soil_layers` refuses that cell's parameters."""
    rng = np.random.default_rng(2024)
    agree = refuse_n = accept_n = 0
    for trial in range(250):
        L = int(rng.integers(1, 5))
        p = params(rng, L, 3, mode=mode)
        c = int(rng.integers(0, 3))
        key = ("edges", "res", "fc", "sat", "iface")[int(rng.integers(0, 5))]
        if key == "iface" and L == 1:
            key = "res"
        shape = p[key].shape
        idx = (int(rng.integers(0, shape[0])), c)
        choice = rng.integers(0, 7)
        p[key] = p[key].copy()
        p[key][idx] = [np.nan, np.inf, -1.0, 0.0, 1e308, 1e-320, p[key][idx] * rng.uniform(0.0, 3.0)][int(choice)]
        try:
            common_cell(p, c, mode)
            common = True
        except ss.SubsurfaceError:
            common = False
        try:
            prepare(make_spec(p, mode=mode), backend=cpu_backend)
            spatial = True
        except sp.SpatialLayerError as exc:
            spatial = False
            cell = exc.first_cell
        if spatial and not common:
            raise AssertionError(f"spatial accepted what soil_layers refused: trial {trial} key {key} idx {idx} L {L}")
        if common:
            # cell c alone is acceptable to soil_layers; the spatial run may refuse only because ANOTHER cell is unchanged-valid? it is: other cells are valid, so it must accept
            assert spatial, f"spatial refused what soil_layers accepted: trial {trial} key {key} idx {idx}"
            accept_n += 1
        else:
            refuse_n += 1
        agree += 1
    assert agree == 250 and refuse_n > 40 and accept_n > 20
