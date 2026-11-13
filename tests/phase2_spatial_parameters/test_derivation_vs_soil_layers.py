"""Per-cell derived/retained maps against the REAL accepted `subsurface.soil_layers` evaluated cell by cell (bit for bit), for 1/2/3/5 layers, both diffusion modes, dry thetas, mixed layouts."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import RETAINED_BY_MODE, common_cell, host, make_spec, params, prepare

from maple_syrup import spatial_soil_layers as sp


def expected_maps(lay, mode):
    out = {"dz": lay.thickness_m, "theta_residual": lay.theta_residual, "theta_field_capacity": lay.theta_field_capacity, "theta_saturation": lay.theta_saturation}
    if lay.n_layers > 1:
        out["pair_inverse_thickness"] = lay.pair_inverse_thickness
        out["interface_conductance_down"] = lay.interface_conductance_down_m_s
        if mode != "constant":
            out["interface_conductance_up"] = lay.interface_conductance_up_m_s
    return out


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_all_cell_layouts_equal_soil_layers_bitwise(cpu_backend, mode, L):
    N = 37
    p = params(np.random.default_rng(100 + L), L, N, mode=mode)
    col = prepare(make_spec(p, mode=mode), backend=cpu_backend)
    assert set(col.retained_names()) == set(n for n in RETAINED_BY_MODE[mode] if L > 1 or n not in ("pair_inverse_thickness", "interface_conductance_down", "interface_conductance_up"))
    for c in range(N):
        lay = common_cell(p, c, mode)
        for name, want in expected_maps(lay, mode).items():
            got = host(col, name)
            got = got[:, c] if got.ndim == 2 else got
            np.testing.assert_array_equal(got.view(np.uint64), np.asarray(want).view(np.uint64), err_msg=f"{name} cell {c}")


def test_constant_mode_up_conductance_aliases_down_and_donor_has_its_own(cpu_backend):
    p = params(np.random.default_rng(1), 3, 5)
    c = prepare(make_spec(p), backend=cpu_backend)
    assert "interface_conductance_up" not in c.retained_names()
    pd = params(np.random.default_rng(1), 3, 5, mode="tietjen_donor_depth")
    d = prepare(make_spec(pd, mode="tietjen_donor_depth"), backend=cpu_backend)
    assert "interface_conductance_up" in d.retained_names()
    assert not np.array_equal(host(d, "interface_conductance_down"), host(d, "interface_conductance_up"))


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
def test_dry_thetas_are_allowed_and_retained_exactly(cpu_backend, mode):
    p = params(np.random.default_rng(2), 3, 9, mode=mode, dry=True)
    assert not p["res"].any()
    col = prepare(make_spec(p, mode=mode), backend=cpu_backend)
    np.testing.assert_array_equal(host(col, "theta_residual"), p["res"])  # residual is a withdrawal floor, not a minimum storage: zero is retained as zero


@pytest.mark.parametrize("edges_shared,res_scalar,fc_shared,sat_cell,iface_kind", [
    (True, True, True, False, "scalar"), (True, False, False, True, "shared"), (False, True, True, True, "scalar"), (False, False, True, False, "cell"),
    (True, True, True, True, "cell"), (False, False, False, False, "shared")])
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
def test_mixed_shared_scalar_and_cell_layouts(cpu_backend, mode, edges_shared, res_scalar, fc_shared, sat_cell, iface_kind):
    L, N = 3, 11
    rng = np.random.default_rng(7)
    p = params(rng, L, N, mode=mode)
    edges = p["edges"][:, 0].copy() if edges_shared else p["edges"]
    res = 0.04 if res_scalar else p["res"]
    fc = (p["res"][:, 0] * 0 + 0.15) if fc_shared else p["fc"]
    sat = p["sat"] if sat_cell else np.full(L, 0.5)
    sat = np.maximum(sat, 0.5) if sat_cell else sat  # keep sat >= fc with the shared fc
    if iface_kind == "scalar":
        iface = 3e-7
    elif iface_kind == "shared":
        iface = p["iface"][:, 0].copy()
    else:
        iface = p["iface"]
    q = {"edges": edges, "res": res if not res_scalar else np.full((L, N), 0.04), "fc": fc if fc_shared else p["fc"], "sat": sat, "iface": iface}
    spec = make_spec({"edges": edges, "res": res, "fc": fc, "sat": sat, "iface": iface}, mode=mode, N=N, iface=iface)
    col = prepare(spec, backend=cpu_backend)
    kinds = col.summary()["input_layouts"]
    assert kinds["depth_edges_m"] == ("shared" if edges_shared else "cell")
    assert kinds["theta_residual"] == ("scalar" if res_scalar else "cell")
    # every cell reproduces the accepted common layer built from that cell's explicit (broadcast) parameters
    full = {"edges": np.broadcast_to(edges.reshape(L + 1, -1), (L + 1, N)), "res": np.broadcast_to(np.asarray(res, dtype=float).reshape(-1, 1) if np.ndim(res) < 2 else res, (L, N)),
            "fc": np.broadcast_to(np.asarray(fc).reshape(L, -1), (L, N)), "sat": np.broadcast_to(np.asarray(sat).reshape(L, -1), (L, N)),
            "iface": np.broadcast_to(np.asarray(iface, dtype=float).reshape(-1, 1) if np.ndim(iface) < 2 else iface, (L - 1, N)) if np.ndim(iface) else np.full((L - 1, N), iface)}
    for c in range(N):
        lay = common_cell(full, c, mode)
        for name, want in expected_maps(lay, mode).items():
            got = host(col, name)
            got = got[:, c] if got.ndim == 2 else got
            got = np.broadcast_to(got, np.asarray(want).shape) if got.shape == (1,) else got  # a SCALAR-layout map is a 1-element array
            np.testing.assert_array_equal(np.ascontiguousarray(got).view(np.uint64), np.asarray(want).view(np.uint64), err_msg=f"{name} cell {c} layouts {kinds}")


def test_shared_layouts_cost_independent_of_n(cpu_backend):
    L = 4
    e = np.cumsum(np.full((L + 1,), 0.1)) - 0.1
    sa = lambda N: make_spec({"edges": e, "res": 0.02, "fc": 0.2, "sat": 0.4, "iface": 1e-7}, N=N, iface=1e-7)  # noqa: E731
    a, b = prepare(sa(10), backend=cpu_backend), prepare(sa(100_000), backend=cpu_backend)
    assert a.retained_bytes == b.retained_bytes  # SHARED/SCALAR arrays: no broadcast materialisation
    assert b.estimate.mask_bytes == 100_000


def test_single_layer_has_no_interface_maps_and_no_fake_values(cpu_backend):
    N = 6
    p = params(np.random.default_rng(3), 1, N)
    for iface in (2e-7, np.zeros(0), np.zeros((0, N))):
        col = prepare(make_spec(p, iface=iface), backend=cpu_backend)
        assert set(col.retained_names()) == {"theta_residual", "theta_field_capacity", "theta_saturation", "dz"}
        np.testing.assert_array_equal(host(col, "dz")[0], p["edges"][1])


def test_chunk_size_invariance_of_retained_maps(cpu_backend):
    p = params(np.random.default_rng(4), 4, 53, mode="tietjen_donor_depth")
    ref = prepare(make_spec(p, mode="tietjen_donor_depth"), backend=cpu_backend, chunk_columns=53)
    for chunk in (1, 7, 52, 1000):
        other = prepare(make_spec(p, mode="tietjen_donor_depth"), backend=cpu_backend, chunk_columns=chunk)
        for name in ref.retained_names():
            np.testing.assert_array_equal(host(ref, name).view(np.uint64), host(other, name).view(np.uint64), err_msg=f"{name} chunk {chunk}")
