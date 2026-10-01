"""Benchmark-only legacy MAHLERAN transport port: walk fractions, Crank-Nicolson identity, ordering, clipping."""
from __future__ import annotations

import math

import numpy as np
import pytest

from maple_syrup import legacy_transport as L
from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_physics import REGIME_CODES


def chain_graph(ny=6, nx=3, dx=0.5):
    """Plane sloping to the south (row 0) with an export ring: every column drains to row 0's outlet."""
    z = np.zeros((ny + 2, nx + 2))
    for r in range(ny + 2):
        z[r, :] = 0.1 * r
    export = np.zeros_like(z, dtype=bool)
    export[0, :] = True
    export[-1, :] = True
    export[:, 0] = True
    export[:, -1] = True
    return build_routing_graph(z, export, np.full((ny, nx), 21.45), dx)


def test_walk_limits_match_legacy_initialisation():
    assert L.walk_limit_cells(10.0, 0.5) == 20 and L.walk_limit_cells(100.0, 0.5) == 200
    assert L.walk_limit_cells(500.0, 0.5) == 1000 and L.walk_limit_cells(10.0, 10.0) == 2
    limits = L.walk_limits_by_regime(0.5)
    assert limits[REGIME_CODES["diffuse"]] == 20 and limits[REGIME_CODES["suspended"]] == 1000
    assert limits[REGIME_CODES["dry"]] == 0 and limits[REGIME_CODES["wet_no_law"]] == 0


def test_network_order_puts_donors_before_receivers():
    g = chain_graph()
    net = L.legacy_network(g)
    position = {int(c): k for k, c in enumerate(net.order)}
    for i in np.flatnonzero(net.active & ~net.outlet):
        assert position[int(i)] < position[int(net.receiver[i])]
    assert set(net.order.tolist()) == set(np.flatnonzero(net.active).tolist())
    # donors table inverse of receivers
    for i in range(net.receiver.size):
        for d in net.donors[i]:
            if d >= 0:
                assert net.receiver[d] == i


def test_walk_fractions_reproduce_flow_distrib_bins_and_ring():
    g = chain_graph(ny=6, nx=1)
    net = L.legacy_network(g)
    n, nc = net.receiver.size, 2
    det = np.zeros((n, nc))
    source = 5  # top of the column (row 5), 5 cells above the outlet row
    det[source, 0] = 1.0
    Lm = 0.8
    inv = np.full((n, nc), 1.0 / Lm)
    law = np.ones((n, nc), bool)
    regime = np.full((n, nc), REGIME_CODES["concentrated"], np.int8)
    step = L.legacy_transport_step(net, det, inv, law, regime, np.zeros((n, nc)), np.zeros((n, nc)),
                                   np.zeros((n, nc)), np.zeros((n, nc)), 1.0)
    dx = net.dx_m
    expected = [1 - math.exp(-dx / Lm)] + [math.exp(-(k + 1) * dx / Lm) - math.exp(-(k + 2) * dx / Lm) for k in range(5)]
    path = [source] + [int(c) for c in [4, 3, 2, 1, 0]]
    for cell, frac in zip(path, expected):
        assert step.deposition_rate_kg_s[cell, 0] == pytest.approx(frac, rel=1e-14)
    ring = math.exp(-6 * dx / Lm) - math.exp(-7 * dx / Lm)  # bin n=5 (after the outlet) lands in the ring, then exits
    assert step.ring_deposition_rate_kg_s[0] == pytest.approx(ring, rel=1e-14)
    assert step.deposition_rate_kg_s[:, 1].sum() == 0.0 and step.ring_deposition_rate_kg_s[1] == 0.0


def test_walk_limit_and_no_capacity_local_deposit():
    g = chain_graph(ny=6, nx=1)
    net = L.legacy_network(g)
    n, nc = net.receiver.size, 1
    det = np.zeros((n, nc)); det[5, 0] = 2.0
    inv = np.full((n, nc), 1.0 / 100.0)  # long travel: the limit, not vge, stops the walk
    law = np.ones((n, nc), bool)
    regime = np.full((n, nc), REGIME_CODES["diffuse"], np.int8)
    limited = L.legacy_transport_step(net, det, inv, law, regime, np.zeros((n, nc)), np.zeros((n, nc)),
                                      np.zeros((n, nc)), np.zeros((n, nc)), 1.0)
    # diffuse limit at dx = 0.5 is 20 cells, but the column has only 5 downstream cells + ring
    assert limited.deposition_rate_kg_s[:, 0].sum() + limited.ring_deposition_rate_kg_s[0] < 2.0
    law[:] = False  # no excess capacity: everything deposits locally
    local = L.legacy_transport_step(net, det, inv, law, regime, np.zeros((n, nc)), np.zeros((n, nc)),
                                    np.zeros((n, nc)), np.zeros((n, nc)), 1.0)
    assert local.deposition_rate_kg_s[5, 0] == 2.0 and local.deposition_rate_kg_s.sum() == 2.0


def test_crank_nicolson_identity_with_clipping_source():
    g = chain_graph(ny=8, nx=4)
    net = L.legacy_network(g)
    rng = np.random.default_rng(3)
    n, nc = net.receiver.size, 3
    M = np.zeros((n, nc)); Q = np.zeros((n, nc)); Qin = np.zeros((n, nc))
    clip_total = 0.0
    for step in range(40):
        det = rng.random((n, nc)) * 1e-3 * (rng.random((n, 1)) < 0.6)
        det[~net.active] = 0.0
        inv = 1.0 / rng.uniform(0.05, 3.0, (n, nc))
        law = rng.random((n, nc)) < 0.8
        regime = np.where(law, REGIME_CODES["diffuse"], REGIME_CODES["wet_no_law"]).astype(np.int8)
        v = rng.random((n, nc)) * 0.2
        v[~net.active] = 0.0
        out = L.legacy_transport_step(net, det, inv, law, regime, v, M, Q, Qin, 1.0)
        residual = out.identity_residual_kg()
        scale = max(out.mobile_before_kg.sum(), out.detachment_rate_kg_s.sum(), 1e-3)
        assert np.all(np.abs(residual) <= 64 * np.finfo(float).eps * scale * n), residual
        assert np.all(out.mobile_after_kg >= 0.0) and np.all(out.clipping_source_kg >= 0.0)
        assert np.all(out.flux_after_kg_s == out.mobile_after_kg * v / net.dx_m)
        clip_total += out.clipping_source_kg.sum()
        M, Q, Qin = out.mobile_after_kg, out.flux_after_kg_s, out.inflow_after_kg_s
    assert clip_total > 0.0  # source-based deposition drives receiving pools negative: the legacy creates mass


def test_python_and_compiled_kernels_agree():
    if L.KERNEL_IMPLEMENTATION != "numba":
        pytest.skip("Numba kernels not compiled in this environment")
    g = chain_graph(ny=5, nx=3)
    net = L.legacy_network(g)
    rng = np.random.default_rng(9)
    n, nc = net.receiver.size, 2
    det = rng.random((n, nc)) * 1e-3; det[~net.active] = 0.0
    inv = 1.0 / rng.uniform(0.1, 2.0, (n, nc)); law = rng.random((n, nc)) < 0.7
    nsteps = L.walk_limits_by_regime(net.dx_m)[np.full((n, nc), REGIME_CODES["diffuse"])]
    v = rng.random((n, nc)) * 0.1; v[~net.active] = 0.0
    M1 = rng.random((n, nc)) * 1e-3; M1[~net.active] = 0.0
    Q1 = M1 * v / net.dx_m; Qin1 = rng.random((n, nc)) * 1e-4
    out = {}
    for name, walk, cn in (("py", L._walk_py, L._cn_py), ("nb", L._walk, L._cn)):
        depos = np.zeros((n, nc)); ring = np.zeros(nc)
        walk(det, inv, law, nsteps, net.receiver, net.dx_m, 1.0, depos, ring)
        M2 = np.zeros((n, nc)); Q2 = np.zeros((n, nc)); Qin2 = np.zeros((n, nc)); clip = np.zeros((n, nc))
        ce = np.zeros(nc); ee = np.zeros(nc)
        cn(net.order, net.donors, net.receiver, net.outlet, M1, Q1, Qin1, det, depos, v, 1.0, net.dx_m, M2, Q2, Qin2, clip, ce, ee)
        out[name] = (depos, ring, M2, Q2, Qin2, clip, ce, ee)
    for a, b in zip(out["py"], out["nb"]):
        assert np.allclose(a, b, rtol=1e-13, atol=1e-18)


# --- input validation happens BEFORE any kernel runs ----------------------------------------------------
@pytest.fixture
def guarded_kernels(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("kernel entered with invalid inputs")
    monkeypatch.setattr(L, "_walk", boom)
    monkeypatch.setattr(L, "_cn", boom)


def _valid_inputs():
    g = chain_graph(ny=4, nx=2)
    net = L.legacy_network(g)
    n, nc = net.receiver.size, 3
    det = np.zeros((n, nc)); det[net.active] = 1e-3
    inv = np.ones((n, nc)); law = np.ones((n, nc), bool)
    regime = np.full((n, nc), REGIME_CODES["diffuse"], np.int8)
    v = np.zeros((n, nc)); v[net.active] = 0.05
    z = np.zeros((n, nc))
    return net, {"detachment_rate_kg_s": det, "inverse_travel_distance_per_m": inv, "law_applies": law, "regime": regime,
                 "sediment_velocity_m_s": v, "mobile_before_kg": z, "flux_before_kg_s": z.copy(),
                 "inflow_before_kg_s": z.copy(), "dt_s": 1.0}


@pytest.mark.parametrize("corrupt", [
    ("dt_s", math.inf), ("dt_s", 0.0), ("dt_s", -1.0), ("dt_s", True),
    ("sediment_velocity_m_s", "fewer_classes"), ("mobile_before_kg", "more_classes"), ("flux_before_kg_s", "fewer_cells"),
    ("inflow_before_kg_s", "negative"), ("sediment_velocity_m_s", "negative"), ("sediment_velocity_m_s", "nan"),
    ("inverse_travel_distance_per_m", "negative"), ("inverse_travel_distance_per_m", "zero_with_law"),
    ("regime", "negative_code"), ("regime", "too_large"), ("regime", "float_dtype"), ("law_applies", "int_dtype"),
    ("detachment_rate_kg_s", "wrong_grid"), ("mobile_before_kg", "inactive_pool"),
])
def test_invalid_inputs_are_refused_before_kernels(guarded_kernels, corrupt):
    net, kw = _valid_inputs()
    name, how = corrupt
    n, nc = kw["detachment_rate_kg_s"].shape
    if name == "dt_s":
        kw["dt_s"] = how
    elif how == "fewer_classes":
        kw[name] = np.zeros((n, nc - 1))
    elif how == "more_classes":
        kw[name] = np.zeros((n, nc + 1))
    elif how == "fewer_cells":
        kw[name] = np.zeros((n - 1, nc))
    elif how == "negative":
        a = kw[name].copy(); a[net.order[0], 0] = -1e-9; kw[name] = a
    elif how == "nan":
        a = kw[name].copy(); a[net.order[0], 0] = np.nan; kw[name] = a
    elif how == "zero_with_law":
        a = kw[name].copy(); a[net.order[0], 0] = 0.0; kw[name] = a
    elif how == "negative_code":
        a = kw[name].astype(np.int64); a[net.order[0], 0] = -1; kw[name] = a
    elif how == "too_large":
        a = kw[name].astype(np.int64); a[net.order[0], 0] = 99; kw[name] = a
    elif how == "float_dtype":
        kw[name] = kw[name].astype(np.float64)
    elif how == "int_dtype":
        kw[name] = kw[name].astype(np.int64)
    elif how == "wrong_grid":
        kw[name] = np.zeros((n + 2, nc))
    elif how == "inactive_pool":
        # emulate an inactive interior cell on the chain network (host-side table edit, kernels never reached)
        import dataclasses
        inactive_cell = int(net.order[0])
        active = net.active.copy(); active[inactive_cell] = False
        net = dataclasses.replace(net, active=active)
        a = kw[name].copy(); a[inactive_cell, 0] = 1.0; kw[name] = a
    with pytest.raises(L.LegacyTransportError):
        L.legacy_transport_step(net, **kw)


def test_valid_inputs_reach_kernels_and_grid_shapes_accepted():
    net, kw = _valid_inputs()
    out = L.legacy_transport_step(net, **kw)
    ny, nx = net.shape
    grid = {k: (v.reshape(ny, nx, -1) if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
    out2 = L.legacy_transport_step(net, **grid)
    assert np.array_equal(out.mobile_after_kg, out2.mobile_after_kg)
    assert out.mobile_after_kg.sum() > 0


def test_single_outlet_cell_analytic_at_half_second_step():
    """One active outlet cell: no inflow, the walk sends the downstream bins straight to the ring, and the
    Crank-Nicolson pool has the closed form M2 = (M1/dt - 0.5 Q1 + Det - Dep) / (1/dt + 0.5 v/dx). Checked at dt = 0.5 s,
    with the ring returned as a RATE (kg/s) like the other deposition terms."""
    net = L.LegacyNetwork((1, 1), 0.5, np.array([L.EXPORT], dtype=np.int64),
                          np.full((1, 4), -1, dtype=np.int64), np.array([0], dtype=np.int64),
                          np.array([True]), np.array([True]), "analytic-outlet")
    assert net.outlet.sum() == 1 and net.receiver[0] == L.EXPORT
    dx, dt, Lm, det, v, M1 = net.dx_m, 0.5, 0.7, 2.0, 0.3, 0.4
    out = L.legacy_transport_step(net, np.array([[det]]), np.array([[1.0 / Lm]]), np.array([[True]]),
                                  np.array([[REGIME_CODES["concentrated"]]], np.int8), np.array([[v]]), np.array([[M1]]),
                                  np.array([[M1 * v / dx]]), np.array([[0.0]]), dt)
    source_fraction = 1 - math.exp(-dx / Lm)
    ring_fraction = math.exp(-dx / Lm) - math.exp(-2 * dx / Lm)  # bin n = 0 lands in the ring and the walk exits
    assert out.deposition_rate_kg_s[0, 0] == pytest.approx(det * source_fraction, rel=1e-14)
    assert out.ring_deposition_rate_kg_s[0] == pytest.approx(det * ring_fraction, rel=1e-14)
    expected_m2 = (M1 / dt - 0.5 * M1 * v / dx + det - det * source_fraction) / (1 / dt + 0.5 * v / dx)
    assert out.mobile_after_kg[0, 0] == pytest.approx(expected_m2, rel=1e-14)
    assert out.flux_after_kg_s[0, 0] == pytest.approx(expected_m2 * v / dx, rel=1e-14)
    assert out.cn_export_kg[0] == pytest.approx(0.5 * (M1 * v / dx + expected_m2 * v / dx) * dt, rel=1e-14)
    assert out.endpoint_export_kg[0] == pytest.approx(expected_m2 * v / dx * dt, rel=1e-14)
    assert abs(out.identity_residual_kg()[0]) <= 1e-15
