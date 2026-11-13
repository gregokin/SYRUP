"""Equal-column spatial mode reproduces the common solver; SHARED/SCALAR layouts equal their CELL expansions bit for bit; cell permutations permute the results. Author-written."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import AREA, BIG, make_spec, prepare, water_kind

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss


def common_params(L, mode):
    edges = np.concatenate([[0.0], np.cumsum(np.linspace(0.07, 0.31, L))])
    res, fc, sat = np.linspace(0.01, 0.05, L), np.linspace(0.16, 0.26, L), np.linspace(0.38, 0.52, L)
    iface = np.linspace(2e-7, 9e-7, L - 1) if mode == "constant" else np.linspace(1e-6, 8e-6, L - 1)
    return edges, res, fc, sat, iface


@pytest.mark.parametrize("L", [2, 3, 5])
def test_equal_columns_reproduce_the_common_solver_for_every_layout(mode, bottom, L):
    N = 19
    edges, res, fc, sat, iface = common_params(L, mode)
    kw = dict(diffusivity_m2_s=iface) if mode == "constant" else dict(interface_conductivity_m_s=iface, diffusion_coefficient=0.25)
    lay = ss.soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat, theta_evaporation_stress=0.5, diffusion_mode=mode, bottom_drainage=bottom, **kw)
    mask = np.arange(N) % 5 != 0
    ccol = ss.prepare_column(lay, mask, AREA)
    rng = np.random.default_rng(17)
    water = np.asarray(lay.saturation_m)[:, None] * rng.uniform(0.0, 1.0, size=(L, N))
    added, pet = rng.uniform(0, 1e-3, size=(L, N)) * 0.4, rng.uniform(0, 3e-7, size=N)
    added[:, ~mask] = 0.0
    pet[~mask] = 0.0
    ref = ss.step(ccol, ss.column_state(ccol, water.copy(), 3.0), 3600.0, added_water_m=added, potential_evaporation_m_s=pet)
    layouts = {
        "all_cell": dict(edges=np.repeat(edges[:, None], N, 1), res=np.repeat(res[:, None], N, 1), fc=np.repeat(fc[:, None], N, 1), sat=np.repeat(sat[:, None], N, 1),
                         iface=np.repeat(iface[:, None], N, 1)),
        "all_shared": dict(edges=edges, res=res, fc=fc, sat=sat, iface=iface),
        "mixed": dict(edges=edges, res=np.repeat(res[:, None], N, 1), fc=fc, sat=np.repeat(sat[:, None], N, 1), iface=iface),
    }
    for name, a in layouts.items():
        extra = dict(diffusivity_m2_s=a["iface"]) if mode == "constant" else dict(interface_conductivity_m_s=a["iface"], diffusion_coefficient=0.25)
        spec = sp.spatial_layer_spec(a["edges"], theta_residual=a["res"], theta_field_capacity=a["fc"], theta_saturation=a["sat"], theta_evaporation_stress=0.5, diffusion_mode=mode,
                                     bottom_drainage=bottom, n_cells=N, **extra)
        col = sp.prepare_spatial_column(spec, mask, AREA, backend="numpy", memory_budget_bytes=BIG)
        out = sp.spatial_step(col, sp.spatial_column_state(col, water, 3.0, memory_budget_bytes=BIG), 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
        np.testing.assert_array_equal(out.state.water_m.view(np.uint64), ref.state.water_m.view(np.uint64), err_msg=name)
        for f in ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "percolation_m", "diffusion_m"):
            np.testing.assert_array_equal(getattr(out, f).view(np.uint64), getattr(ref, f).view(np.uint64), err_msg=f"{name} {f}")
        assert dataclasses.asdict(out.budget) == dataclasses.asdict(ref.budget), name
        assert out.state.t_s == ref.state.t_s == 3600.0 + 3.0


def run(col, p, water, added, pet, share=None):
    st = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    return sp.spatial_step(col, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, evaporation_share=share, memory_budget_bytes=BIG)


def test_shared_scalar_and_cell_expansions_are_bit_identical(mode, bottom):
    L, N = 3, 12
    rng = np.random.default_rng(2)
    edges, res, fc, sat, iface = common_params(L, mode)
    water = np.asarray(sat)[:, None] * np.repeat(np.diff(edges)[:, None], N, 1) * rng.uniform(0.0, 1.0, size=(L, N))
    added, pet = rng.uniform(0, 1e-3, size=(L, N)) * 0.4, rng.uniform(0, 3e-7, size=N)
    base = dict(edges=edges, res=0.03, fc=fc, sat=sat, iface=iface)
    p_cell = dict(edges=np.repeat(edges[:, None], N, 1), res=np.full((L, N), 0.03), fc=np.repeat(fc[:, None], N, 1), sat=np.repeat(sat[:, None], N, 1), iface=np.repeat(iface[:, None], N, 1))
    a = prepare(sp.spatial_layer_spec(edges, theta_residual=0.03, theta_field_capacity=fc, theta_saturation=sat, theta_evaporation_stress=0.5, diffusion_mode=mode, bottom_drainage=bottom,
                                      n_cells=N, **(dict(diffusivity_m2_s=iface) if mode == "constant" else dict(interface_conductivity_m_s=iface, diffusion_coefficient=0.3))))
    b = prepare(make_spec(p_cell, mode=mode, bottom=bottom))
    ra, rb = run(a, base, water, added, pet), run(b, p_cell, water, added, pet)
    np.testing.assert_array_equal(ra.state.water_m.view(np.uint64), rb.state.water_m.view(np.uint64))
    assert dataclasses.asdict(ra.budget) == dataclasses.asdict(rb.budget)


def test_cell_permutation_permutes_the_results_bitwise(mode, bottom):
    from conftest import params

    L, N = 3, 26
    rng = np.random.default_rng(4)
    p = params(rng, L, N, mode=mode)
    perm = rng.permutation(N)
    q = {k: (np.ascontiguousarray(v[..., perm]) if v.ndim == 2 else v) for k, v in p.items()}
    water = water_kind(rng, p, "above_fc")
    added, pet = rng.uniform(0, 1e-3, size=(L, N)) * 0.3, rng.uniform(0, 3e-7, size=N)
    r1 = run(prepare(make_spec(p, mode=mode, bottom=bottom)), p, water, added, pet)
    r2 = run(prepare(make_spec(q, mode=mode, bottom=bottom)), q, water[:, perm].copy(), added[:, perm].copy(), pet[perm].copy())
    np.testing.assert_array_equal(r1.state.water_m[:, perm].view(np.uint64), r2.state.water_m.view(np.uint64))
    np.testing.assert_array_equal(r1.percolation_m[:, perm], r2.percolation_m)
    np.testing.assert_array_equal(r1.diffusion_m[:, perm], r2.diffusion_m)
    np.testing.assert_array_equal(r1.drainage_m[perm], r2.drainage_m)
