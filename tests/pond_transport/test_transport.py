"""Multi-face characteristic pond transport (`maple_syrup.pond_transport`). NOT RUN by their author (file-only tools).

Independent references used here (none of them calls the code under test):
- `ref_packet`: exact continuous-time characteristic of ONE packet through a chain of cells with piecewise
  constant (v, r, settle). It is event based (face crossing times `(dx - x) / v`), not substepped.
- closed-form `exp(-r s)` survival / `m (1 - exp(-r s))` deposition arithmetic and the logarithmic mean
  `L log(sum m exp(x / L) / sum m)` evaluated with `np.logaddexp`;
- the existing single-receiver `characteristic_transport.characteristic_step` on a graph-derived D4 network
  (water volumes and outlets are derived from the graph's own face tables);
- budget identities computed from the returned MAPLE-layout face arrays and cumulative sums.
Physics tolerances are explicit FP64 values; the operator's own `budget_*` diagnostics are only inspected in
`test_diagnostics_are_separate`.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("maple")

from maple.surface.voxels._numerics import summation_error_bound_kg  # noqa: E402

from maple_syrup.characteristic_transport import (  # noqa: E402
    PhaseState,
    bin_index,
    characteristic_step,
    fraction_sum_tolerance,
)
from maple_syrup.characteristic_transport import empty_phase_state as single_empty_phase_state  # noqa: E402
from maple_syrup.pond_transport import (  # noqa: E402
    FaceNetwork,
    empty_phase_state,
    face_network,
    face_network_from_geometry,
    face_network_from_pond_topology,
    operation_count,
    outflow_weights,
    pond_transport_step,
    validate_phase,
)
from maple_syrup.routing import build_routing_graph  # noqa: E402
from maple_syrup.sediment_transport import (  # noqa: E402
    TransportError,
    TransportStepRejected,
    transport_network,
)

DX = 0.5


# --- helpers -----------------------------------------------------------------------------------
def cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device visible to CuPy")
    except Exception as exc:  # noqa: BLE001 - any driver failure is an honest skip
        pytest.skip(f"CUDA unavailable: {exc}")
    return cp


def row_net(n, *, dx=DX, outlet_east=True, outlet_west=False):
    """1 x n row; x faces 0..n. Flow east when `row_volumes` is used."""
    active = np.ones((1, n), dtype=bool)
    ox = np.zeros((1, n + 1), dtype=bool)
    ox[0, n], ox[0, 0] = outlet_east, outlet_west
    return face_network(active, dx, ox, None)


def row_volumes(n, value=1.0):
    vx = np.zeros((1, n + 1))
    vx[0, 1:] = value
    return vx, np.zeros((2, n))


def fld(shape, nc, v=0.0, r=0.0, s=False):
    ny, nx = shape
    return (np.full((ny, nx, nc), float(v)), np.full((ny, nx, nc), float(r)), np.full((ny, nx, nc), bool(s)))


def zeros(shape, nc):
    return np.zeros((*shape, nc))


def step(net, M, phase, P, V, R, S, vx, vy, dt, nsub, **kw):
    return pond_transport_step(net, M, phase, P, V, R, S, vx, vy, dt, n_substeps=nsub, **kw)


def ref_packet(v, r, settle, horizon, dx, mass=1.0):
    """Exact event-based trajectory of one packet injected at x = 0 of cell 0 of a chain.
    Returns (deposition per cell, final cell or None if exported, x, remaining mass, exported mass)."""
    k = len(v)
    dep = [0.0] * k
    x, i, t, m = 0.0, 0, 0.0, mass
    while True:
        if settle[i]:
            dep[i] += m
            return dep, i, 0.0, 0.0, 0.0
        if v[i] == 0.0:
            return dep, i, x, m, 0.0
        tau = (dx - x) / v[i]
        if t + tau <= horizon:
            d = dx - x
            dep[i] += m * -math.expm1(-r[i] * d)
            m *= math.exp(-r[i] * d)
            t, x, i = t + tau, 0.0, i + 1
            if i == k:
                return dep, None, 0.0, 0.0, m
        else:
            s = v[i] * (horizon - t)
            dep[i] += m * -math.expm1(-r[i] * s)
            m *= math.exp(-r[i] * s)
            return dep, i, x + s, m, 0.0


def phase_position(phase, row, col, cls=0):
    """The representative position(s) of the occupied bins of one cell/class."""
    f = np.asarray(phase.fraction)[row, col, cls]
    x = np.asarray(phase.position_m)[row, col, cls]
    return x[f > 0.0]


def assert_phase_canonical(phase, remaining, dx):
    f, x = np.asarray(phase.fraction), np.asarray(phase.position_m)
    nb = f.shape[-1]
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(x)) and np.all(f >= 0.0)
    assert np.all(x >= 0.0) and np.all(x < dx)
    occupied = np.asarray(remaining) > 0.0
    assert np.all(np.abs(f.sum(-1) - 1.0)[occupied] <= fraction_sum_tolerance(nb))
    canonical = np.zeros(nb)
    canonical[0] = 1.0
    assert np.all(f[~occupied] == canonical) and np.all(x[~occupied] == 0.0)
    assert np.all(x[f == 0.0] == 0.0)
    inside = bin_index(x, dx, nb, np) == np.arange(nb)
    assert np.all(inside[f > 0.0])


# --- single packet: analytic travel, deposition, timing -------------------------------------------
def test_single_packet_travel_deposition_and_position():
    n, nb = 4, 8
    net = row_net(n)
    vx, vy = row_volumes(n)
    V, R, S = fld((1, n), 1, v=0.1, r=2.0)
    P = zeros((1, n), 1)
    P[0, 0, 0] = 1.0
    res = step(net, zeros((1, n), 1), empty_phase_state((1, n), 1, nb, DX), P, V, R, S, vx, vy, dt=12.0, nsub=8)
    # Hand numbers: 1.2 m of travel = cells 0 and 1 in full, 0.2 m into cell 2.
    e = math.exp
    dep = [1 - e(-1), e(-1) * (1 - e(-1)), e(-2) * (1 - e(-0.4)), 0.0]
    np.testing.assert_allclose(res.transport.deposition_request_kg[0, :, 0], dep, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(res.mobile_remaining_kg[0, :, 0], [0, 0, e(-2.4), 0], rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(phase_position(res.phase_after, 0, 2), [0.2], atol=1e-12)
    np.testing.assert_allclose(res.transport.x_face_gross_kg[0, :, 0], [0, e(-1), e(-2), 0, 0], rtol=1e-12, atol=1e-16)
    assert np.all(res.transport.export_request_kg == 0.0)
    # The independent event-based reference agrees.
    d, i, x, m, _ = ref_packet([0.1] * n, [2.0] * n, [False] * n, 12.0, DX)
    assert i == 2 and abs(x - 0.2) < 1e-12 and abs(m - e(-2.4)) < 1e-15
    np.testing.assert_allclose(res.transport.deposition_request_kg[0, :, 0], d, rtol=1e-12, atol=1e-16)


@pytest.mark.parametrize("nsub", [3, 30])
def test_survival_to_face_is_exp_minus_dx_over_L_for_L_much_smaller_than_dx(nsub):
    r, v = 100.0, 0.2  # L = 0.01 m, dx = 0.5 m: dx / L = 50
    net = row_net(2)
    vx, vy = row_volumes(2)
    V, R, S = fld((1, 2), 1, v=v, r=r)
    P = zeros((1, 2), 1)
    P[0, 0, 0] = 1.0
    res = step(net, zeros((1, 2), 1), empty_phase_state((1, 2), 1, 4, DX), P, V, R, S, vx, vy, dt=3.0, nsub=nsub)
    crossing = float(res.transport.x_face_gross_kg[0, 1, 0])
    np.testing.assert_allclose(crossing, math.exp(-50.0), rtol=1e-9)
    assert crossing < 1e-20  # the corrected-away well-mixed L / (L + dx) = 0.0196 would be ~18 orders larger
    assert res.transport.deposition_request_kg[0, 0, 0] == pytest.approx(1.0, abs=1e-15)
    d, *_ = ref_packet([v, v], [r, r], [False, False], 3.0, DX)
    np.testing.assert_allclose(res.transport.deposition_request_kg[0, :, 0], d, rtol=1e-9, atol=1e-30)


def test_extreme_decay_deposits_in_source_without_nan():
    net = row_net(2)
    vx, vy = row_volumes(2)
    V, R, S = fld((1, 2), 1, v=0.2, r=1.0e300)
    P = zeros((1, 2), 1)
    P[0, 0, 0] = 1.0
    res = step(net, zeros((1, 2), 1), empty_phase_state((1, 2), 1, 4, DX), P, V, R, S, vx, vy, dt=3.0, nsub=3)
    assert res.transport.deposition_request_kg[0, 0, 0] == 1.0
    assert np.all(res.mobile_remaining_kg == 0.0) and np.all(np.isfinite(res.transport.x_face_gross_kg))


# --- reduction to the existing D4 operator -----------------------------------------------------------
def chain_full(n, dz=0.015625, walls=1.0):
    z = np.repeat(np.arange(n + 2, dtype=np.float64)[:, None] * dz, 3, axis=1)
    z[:, 0] += walls
    z[:, 2] += walls
    return z


def valley_full(ny, nx, sy=0.015625, sx=0.03125):
    rows, cols = np.indices((ny + 2, nx + 2)).astype(np.float64)
    return rows * sy + np.abs(cols - (nx + 1) / 2) * sx


def graph_network(z):
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    export = np.zeros(z.shape, dtype=bool)
    export[0, :] = True
    graph = build_routing_graph(z, export, np.full((ny, nx), 1.0), DX)
    return transport_network(graph)


def pond_inputs_from_graph(tn):
    """Water volumes (+-1 on each cell's single out-face) and outlet booleans from the graph's own face tables."""
    ny, nx = tn.shape
    outlet = np.asarray(tn.outlet_host).reshape(-1)
    cx, rx, fx, sx = (np.asarray(a) for a in (tn.cells_x, tn.rows_x, tn.faces_x, tn.sign_x))
    cy, fy, qy, sy = (np.asarray(a) for a in (tn.cells_y, tn.faces_y, tn.cols_y, tn.sign_y))
    vx, vy = np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
    ox, oy = np.zeros((ny, nx + 1), dtype=bool), np.zeros((ny + 1, nx), dtype=bool)
    vx[rx, fx] = sx
    vy[fy, qy] = sy
    ox[rx[outlet[cx]], fx[outlet[cx]]] = True
    oy[fy[outlet[cy]], qy[outlet[cy]]] = True
    assert np.all((fx[outlet[cx]] == 0) | (fx[outlet[cx]] == nx))  # outlets leave the interior
    assert np.all((fy[outlet[cy]] == 0) | (fy[outlet[cy]] == ny))
    return vx, vy, ox, oy


@pytest.mark.parametrize("terrain", ["chain", "valley"])
def test_fixed_single_direction_reduces_to_existing_d4_operator(terrain):
    z = chain_full(6) if terrain == "chain" else valley_full(5, 4)
    tn = graph_network(z)
    ny, nx = tn.shape
    vx, vy, ox, oy = pond_inputs_from_graph(tn)
    net = face_network(np.asarray(tn.active_host), DX, ox, oy)
    w = np.asarray(outflow_weights(net, vx, vy))
    assert np.all(np.sort(w, axis=-1)[..., :3] == 0.0) and np.all(w.max(-1) == 1.0)
    rng = np.random.default_rng(11)
    nc, nb, dt, nsub = 2, 8, 4.0, 8
    V = rng.uniform(0.0, 0.2, (ny, nx, nc))
    R = rng.uniform(0.0, 6.0, (ny, nx, nc))
    S = rng.random((ny, nx, nc)) < 0.05
    M_ref = M_new = zeros((ny, nx), nc)
    ph_ref = single_empty_phase_state((ny, nx), nc, nb, DX)
    ph_new = empty_phase_state((ny, nx), nc, nb, DX)
    for _ in range(4):
        P = np.where(rng.random((ny, nx, nc)) < 0.4, rng.uniform(0.0, 1.0, (ny, nx, nc)), 0.0)
        old = characteristic_step(tn, M_ref, ph_ref, P, V, R, S, dt, n_substeps=nsub)
        new = step(net, M_new, ph_new, P, V, R, S, vx, vy, dt, nsub)
        a, b = old.transport, new.transport
        for name in ("mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg",
                     "internal_transfer_in_kg", "internal_transfer_out_kg", "x_face_gross_kg", "y_face_gross_kg",
                     "x_face_net_kg", "y_face_net_kg", "settled_kg", "decay_deposition_kg"):
            np.testing.assert_allclose(getattr(b, name), getattr(a, name), rtol=1e-10, atol=1e-14, err_msg=name)
        np.testing.assert_allclose(new.phase_after.fraction, old.phase_after.fraction, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(new.phase_after.position_m, old.phase_after.position_m, rtol=1e-10, atol=1e-11)
        M_ref, ph_ref = old.mobile_remaining_kg, old.phase_after
        M_new, ph_new = new.mobile_remaining_kg, new.phase_after


# --- splitting, convergence, blocked cells, settling ---------------------------------------------------
def three_row(vx_mid_left, vx_mid_right):
    """1 x 3 row, outlets at both ends (each end cell sends out of its own end), signed middle faces."""
    active = np.ones((1, 3), dtype=bool)
    ox = np.zeros((1, 4), dtype=bool)
    ox[0, 0] = ox[0, 3] = True
    vx = np.array([[-1.0, vx_mid_left, vx_mid_right, 1.0]])
    return face_network(active, DX, ox, None), vx, np.zeros((2, 3))


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (1.0, 3.0), (7.0, 0.5)])
def test_two_way_split_by_water_volume_weight(a, b):
    net, vx, vy = three_row(-a, b)  # middle cell: west volume a, east volume b
    V, R, S = fld((1, 3), 1, v=0.4, r=0.5)
    P = zeros((1, 3), 1)
    P[0, 1, 0] = 1.0
    res = step(net, zeros((1, 3), 1), empty_phase_state((1, 3), 1, 4, DX), P, V, R, S, vx, vy, dt=2.0, nsub=4)
    dep, i, x, m, _ = ref_packet([0.4, 0.4], [0.5, 0.5], [False, False], 2.0, DX)
    w_west, w_east = a / (a + b), b / (a + b)
    np.testing.assert_allclose(res.mobile_remaining_kg[0, :, 0], [w_west * m, 0.0, w_east * m], rtol=1e-12)
    np.testing.assert_allclose(res.transport.deposition_request_kg[0, :, 0],
                               [w_west * dep[1], dep[0], w_east * dep[1]], rtol=1e-12)
    np.testing.assert_allclose(phase_position(res.phase_after, 0, 0), [x], atol=1e-12)
    np.testing.assert_allclose(phase_position(res.phase_after, 0, 2), [x], atol=1e-12)
    surv = math.exp(-0.5 * DX)
    np.testing.assert_allclose(res.transport.x_face_gross_kg[0, 1, 0], w_west * surv, rtol=1e-12)
    np.testing.assert_allclose(res.transport.x_face_gross_kg[0, 2, 0], w_east * surv, rtol=1e-12)
    np.testing.assert_allclose(res.transport.x_face_net_kg[0, 1, 0], -w_west * surv, rtol=1e-12)  # westward
    np.testing.assert_allclose(res.transport.x_face_net_kg[0, 2, 0], w_east * surv, rtol=1e-12)


def test_four_way_biased_split():
    active = np.ones((3, 3), dtype=bool)
    ox = np.zeros((3, 4), dtype=bool)
    oy = np.zeros((4, 3), dtype=bool)
    ox[1, 0] = ox[1, 3] = oy[0, 1] = oy[3, 1] = True  # each neighbour of the centre leaves outward
    net = face_network(active, DX, ox, oy)
    vx, vy = np.zeros((3, 4)), np.zeros((4, 3))
    vx[1, 2], vy[2, 1], vx[1, 1], vy[1, 1] = 1.0, 2.0, -3.0, -4.0  # centre outward: E 1, N 2, W 3, S 4
    vx[1, 0], vx[1, 3], vy[0, 1], vy[3, 1] = -1.0, 1.0, -1.0, 1.0  # neighbours' own outlets
    V, R, S = fld((3, 3), 1, v=0.4, r=0.5)
    P = zeros((3, 3), 1)
    P[1, 1, 0] = 1.0
    res = step(net, zeros((3, 3), 1), empty_phase_state((3, 3), 1, 4, DX), P, V, R, S, vx, vy, dt=2.0, nsub=4)
    _, _, x, m, _ = ref_packet([0.4, 0.4], [0.5, 0.5], [False, False], 2.0, DX)
    for (row, col), w in {(1, 2): 0.1, (2, 1): 0.2, (1, 0): 0.3, (0, 1): 0.4}.items():
        np.testing.assert_allclose(res.mobile_remaining_kg[row, col, 0], w * m, rtol=1e-12)
        np.testing.assert_allclose(phase_position(res.phase_after, row, col), [x], atol=1e-12)
    np.testing.assert_allclose(np.asarray(res.outflow_weight)[1, 1], [0.1, 0.2, 0.3, 0.4], rtol=1e-14)
    assert res.mobile_remaining_kg[1, 1, 0] == 0.0


def convergent_setup():
    """Row 0: donors (0,0) -> east, (0,2) -> west into the middle (0,1), which leaves through a south outlet."""
    active = np.ones((2, 3), dtype=bool)
    oy = np.zeros((3, 3), dtype=bool)
    oy[0, 1] = True
    net = face_network(active, DX, None, oy)
    vx, vy = np.zeros((2, 4)), np.zeros((3, 3))
    vx[0, 1], vx[0, 2], vy[0, 1] = 1.0, -1.0, -1.0
    return net, vx, vy


@pytest.mark.parametrize("m2", [1.0, 1.0e-200])
def test_convergent_donors_merge_to_hazard_preserving_log_mean(m2):
    net, vx, vy = convergent_setup()
    r = 20.0
    V, R, S = fld((2, 3), 1, v=0.0, r=r)
    V[0, 0, 0], V[0, 2, 0], V[0, 1, 0] = 0.4, 0.3, 0.1
    P = zeros((2, 3), 1)
    P[0, 0, 0], P[0, 2, 0] = 1.0, m2
    res = step(net, zeros((2, 3), 1), empty_phase_state((2, 3), 1, 1, DX), P, V, R, S, vx, vy, dt=2.0, nsub=4)
    out = [ref_packet([vv, 0.1], [r, r], [False, False], 2.0, DX, mass=mm) for vv, mm in ((0.4, 1.0), (0.3, m2))]
    (_, _, xa, ma, _), (_, _, xb, mb, _) = out
    assert xa != xb
    np.testing.assert_allclose(res.mobile_remaining_kg[0, 1, 0], ma + mb, rtol=1e-12)
    L = 1.0 / r
    expected = L * (np.logaddexp(np.log(ma) + xa / L, np.log(mb) + xb / L) - np.log(ma + mb))
    (got,) = phase_position(res.phase_after, 0, 1)
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    assert min(xa, xb) - 1e-12 <= got <= max(xa, xb) + 1e-12  # inside the constituents' range
    # Mass that crossed into the middle cell: the pickups' survivors after one full dx, exp(-r dx) each.
    np.testing.assert_allclose(res.transport.internal_transfer_in_kg[0, 1, 0], (1.0 + m2) * math.exp(-r * DX),
                               rtol=1e-9)


def blocked_row():
    active = np.ones((1, 3), dtype=bool)
    vx = np.array([[0.0, 1.0, -1.0, 0.0]])  # both donors push into the middle cell, which has no outflow
    return face_network(active, DX), vx, np.zeros((2, 3))


def test_cell_without_outflow_preserves_inventory_and_phase():
    net, vx, vy = blocked_row()
    nb = 4
    phase = empty_phase_state((1, 3), 1, nb, DX)
    fraction, position = np.array(phase.fraction), np.array(phase.position_m)
    fraction[0, 1, 0], position[0, 1, 0] = [0, 0, 1, 0], [0, 0, 0.3, 0]  # 1 kg sitting at x = 0.3 (bin 2)
    phase = PhaseState(fraction=fraction, position_m=position, dx_m=DX, n_bins=nb, xp=np)
    M = zeros((1, 3), 1)
    M[0, 1, 0] = 1.0
    V, R, S = fld((1, 3), 1, v=0.4, r=3.0)
    P = zeros((1, 3), 1)
    P[0, 0, 0] = P[0, 2, 0] = 1.0
    res = step(net, M, phase, P, V, R, S, vx, vy, dt=2.0, nsub=4)
    arrived = math.exp(-3.0 * DX)
    assert bool(res.blocked_cell[0, 1]) and not bool(res.blocked_cell[0, 0])
    np.testing.assert_allclose(res.mobile_remaining_kg[0, 1, 0], 1.0 + 2.0 * arrived, rtol=1e-12)
    assert res.transport.deposition_request_kg[0, 1, 0] == 0.0 and np.all(res.transport.export_request_kg == 0.0)
    np.testing.assert_allclose(res.transport.deposition_request_kg[0, [0, 2], 0], 1.0 - arrived, rtol=1e-12)
    pos = np.asarray(res.phase_after.position_m)[0, 1, 0]
    assert pos[2] == 0.3 and pos[0] == 0.0  # old mass did not move; arrivals stay at the upstream face
    np.testing.assert_allclose(np.asarray(res.phase_after.fraction)[0, 1, 0, [0, 2]],
                               [2.0 * arrived / (1 + 2 * arrived), 1 / (1 + 2 * arrived)], rtol=1e-12)
    # explicit settle mask: the whole pool, arrivals included, becomes a deposition request. Never an export.
    S2 = S.copy()
    S2[0, 1, 0] = True
    res2 = step(net, M, phase, P, V, R, S2, vx, vy, dt=2.0, nsub=4)
    np.testing.assert_allclose(res2.transport.deposition_request_kg[0, 1, 0], 1.0 + 2.0 * arrived, rtol=1e-12)
    assert res2.mobile_remaining_kg[0, 1, 0] == 0.0 and np.all(res2.transport.export_request_kg == 0.0)
    np.testing.assert_allclose(res2.arrival_deposition_kg[0, 1, 0], 2.0 * arrived, rtol=1e-12)


def test_pickup_in_a_closed_cell_stays_at_the_face():
    net = face_network(np.ones((1, 1), dtype=bool), DX)
    V, R, S = fld((1, 1), 1, v=1.0, r=5.0)
    P = np.full((1, 1, 1), 2.0)
    res = step(net, zeros((1, 1), 1), empty_phase_state((1, 1), 1, 4, DX), P, V, R, S, np.zeros((1, 2)),
               np.zeros((2, 1)), 10.0, 1)
    assert res.mobile_remaining_kg[0, 0, 0] == 2.0 and res.transport.deposition_request_kg[0, 0, 0] == 0.0
    assert phase_position(res.phase_after, 0, 0).tolist() == [0.0]


def test_settling_receiver_takes_arrivals_and_matches_reference():
    n = 3
    net = row_net(n)
    vx, vy = row_volumes(n)
    V, R, S = fld((1, n), 1, v=0.4, r=0.7)
    S[0, 1, 0] = True
    P = zeros((1, n), 1)
    P[0, 0, 0] = 1.0
    res = step(net, zeros((1, n), 1), empty_phase_state((1, n), 1, 4, DX), P, V, R, S, vx, vy, dt=2.0, nsub=4)
    dep, i, x, m, _ = ref_packet([0.4] * n, [0.7] * n, [False, True, False], 2.0, DX)
    assert i == 1 and m == 0.0
    np.testing.assert_allclose(res.transport.deposition_request_kg[0, :, 0], dep, rtol=1e-12)
    np.testing.assert_allclose(res.transport.settled_kg[0, 1, 0], math.exp(-0.7 * DX), rtol=1e-12)
    assert np.all(res.mobile_remaining_kg == 0.0)


def test_classes_use_actual_pickup_and_their_own_velocities():
    n = 4
    net = row_net(n)
    vx, vy = row_volumes(n)
    V, R, S = fld((1, n), 2)
    V[..., 0], V[..., 1] = 0.1, 0.2
    R[..., 0], R[..., 1] = 1.0, 0.0
    P = zeros((1, n), 2)
    P[0, 0, 0], P[0, 0, 1] = 0.3, 0.0  # class 1 has NO pickup this step
    res = step(net, zeros((1, n), 2), empty_phase_state((1, n), 2, 4, DX), P, V, R, S, vx, vy, dt=12.0, nsub=12)
    dep, i, x, m, _ = ref_packet([0.1] * n, [1.0] * n, [False] * n, 12.0, DX, mass=0.3)
    np.testing.assert_allclose(res.transport.deposition_request_kg[0, :, 0], dep, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(res.mobile_remaining_kg[0, i, 0], m, rtol=1e-12)
    assert np.all(res.mobile_remaining_kg[..., 1] == 0.0) and np.all(res.transport.x_face_gross_kg[..., 1] == 0.0)
    np.testing.assert_allclose(res.transport.mobile_before_by_class_kg, [0.3, 0.0])


def test_zero_rate_and_zero_velocity():
    n = 4
    net = row_net(n)
    vx, vy = row_volumes(n)
    V, R, S = fld((1, n), 1, v=0.1, r=0.0)
    P = zeros((1, n), 1)
    P[0, 0, 0] = 1.0
    res = step(net, zeros((1, n), 1), empty_phase_state((1, n), 1, 4, DX), P, V, R, S, vx, vy, dt=3.0, nsub=2)
    assert np.all(res.transport.deposition_request_kg == 0.0)  # r = 0: no deposition law
    np.testing.assert_allclose(phase_position(res.phase_after, 0, 0), [0.3], atol=1e-14)  # first moment M v dt
    assert res.mobile_remaining_kg[0, 0, 0] == 1.0
    V0, R0, _ = fld((1, n), 1, v=0.0, r=1.0e300)  # v = 0 with a huge rate: no motion, no overflow, no deposition
    res0 = step(net, zeros((1, n), 1), empty_phase_state((1, n), 1, 4, DX), P, V0, R0, S, vx, vy, dt=3.0, nsub=1)
    assert res0.mobile_remaining_kg[0, 0, 0] == 1.0 and np.all(res0.transport.deposition_request_kg == 0.0)
    assert phase_position(res0.phase_after, 0, 0).tolist() == [0.0]


# --- domain: closed faces, holes, single outlet, adapters ---------------------------------------------
def test_single_open_face_exports_only_there_and_closed_domain_exports_nothing():
    n = 3
    net = row_net(n)
    vx, vy = row_volumes(n)
    V, R, S = fld((1, n), 1, v=0.4, r=0.0)
    P = zeros((1, n), 1)
    P[0, :, 0] = 1.0
    res = step(net, zeros((1, n), 1), empty_phase_state((1, n), 1, 4, DX), P, V, R, S, vx, vy, dt=10.0, nsub=20)
    assert res.transport.export_request_kg[0, 2, 0] > 0.0
    assert np.all(res.transport.export_request_kg[0, :2, 0] == 0.0)
    sent = np.asarray(res.face_crossing_by_direction_kg)
    assert np.all(sent[..., [1, 2, 3]] == 0.0) and sent[0, 2, 0, 0] == res.transport.export_request_kg[0, 2, 0]
    assert res.transport.x_face_gross_kg[0, 0, 0] == 0.0
    closed = face_network(np.ones((2, 2), dtype=bool), DX)  # no outlets: all mass stays
    vxc = np.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0]])
    Vc, Rc, Sc = fld((2, 2), 1, v=0.4, r=0.3)
    Pc = np.full((2, 2, 1), 1.0)
    resc = step(closed, zeros((2, 2), 1), empty_phase_state((2, 2), 1, 4, DX), Pc, Vc, Rc, Sc, vxc, np.zeros((3, 2)),
                10.0, 20)
    assert np.all(resc.transport.export_request_kg == 0.0)
    total = resc.mobile_remaining_kg.sum() + resc.transport.deposition_request_kg.sum()
    assert total == pytest.approx(4.0, rel=1e-13)


def test_masked_hole_and_closed_faces_carry_no_flux():
    active = np.ones((1, 3), dtype=bool)
    active[0, 1] = False
    net = face_network(active, DX)
    assert np.all(net.fx_type_host == 0)  # nothing across the hole, no implied open sides
    V, R, S = fld((1, 3), 1, v=0.2)
    V[0, 1, 0] = 0.0
    args = (net, zeros((1, 3), 1), empty_phase_state((1, 3), 1, 4, DX), zeros((1, 3), 1), V, R, S)
    for vx in (np.array([[0.0, 1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0, 1.0]])):  # across hole / closed ring
        with pytest.raises(TransportError, match="closed faces"):
            step(*args, vx, np.zeros((2, 3)), 1.0, 1)
    inactive_mobile = zeros((1, 3), 1)
    inactive_mobile[0, 1, 0] = 1.0
    with pytest.raises(TransportError, match="inactive"):
        step(net, inactive_mobile, empty_phase_state((1, 3), 1, 4, DX), zeros((1, 3), 1), V, R, S,
             np.zeros((1, 4)), np.zeros((2, 3)), 1.0, 1)
    with pytest.raises(TransportError, match="inactive"):
        face_network_from_geometry(active, np.array([[0, 1, 1, 0]], dtype=np.int8), np.zeros((2, 3), dtype=np.int8), DX)


def test_construction_refusals():
    active = np.ones((2, 2), dtype=bool)
    ox = np.zeros((2, 3), dtype=bool)
    ox[0, 1] = True  # interior face as an outlet: interior open holes are unsupported
    with pytest.raises(TransportError, match="EXTERIOR"):
        face_network(active, DX, ox, None)
    with pytest.raises(TransportError, match="periodic"):
        face_network(active, DX, periodic=True)
    with pytest.raises(TransportError, match="external"):
        face_network(active, DX, external_inflow=True)
    masked = active.copy()
    masked[0, 0] = False
    oy = np.zeros((3, 2), dtype=bool)
    oy[0, 0] = True  # outlet of an inactive cell
    with pytest.raises(TransportError, match="EXTERIOR|ACTIVE"):
        face_network(masked, DX, None, oy)
    with pytest.raises(TransportError):
        face_network(np.zeros((2, 2), dtype=bool), DX)
    with pytest.raises(TransportError):
        face_network(active, -1.0)


def test_pond_topology_adapter_matches_direct_construction():
    active = np.ones((2, 3), dtype=bool)
    active[1, 2] = False
    ox = np.zeros((2, 4), dtype=bool)
    ox[0, 3] = True
    oy = np.zeros((3, 3), dtype=bool)
    oy[0, 0] = True
    direct = face_network(active, DX, ox, oy)
    fx, fy = direct.fx_type_host.copy(), direct.fy_type_host.copy()
    sx = np.where(fx == 2, np.where(np.arange(4)[None, :] == 0, -1, 1), 0).astype(np.int8)
    sy = np.where(fy == 2, np.where(np.arange(3)[:, None] == 0, -1, 1), 0).astype(np.int8)
    geometry = SimpleNamespace(active=active, fx_type=fx, fy_type=fy, fx_sign=sx, fy_sign=sy)
    adapted = face_network_from_pond_topology(SimpleNamespace(active=active, dx_m=DX, geometry=geometry))
    assert isinstance(adapted, FaceNetwork) and adapted.input_sha256 == direct.input_sha256
    bad = SimpleNamespace(active=active, dx_m=DX, geometry=SimpleNamespace(**{**vars(geometry), "fx_sign": -sx}))
    with pytest.raises(TransportError, match="fx_sign"):
        face_network_from_pond_topology(bad)


def test_water_volume_validation_and_nonmutation():
    n = 3
    net = row_net(n)
    vx, vy = row_volumes(n)
    V, R, S = fld((1, n), 1, v=0.2, r=1.0)
    P = zeros((1, n), 1)
    P[0, 0, 0] = 1.0
    M = zeros((1, n), 1)
    phase = empty_phase_state((1, n), 1, 4, DX)
    snapshot = [a.copy() for a in (M, P, V, R, S, vx, vy, phase.fraction, phase.position_m)]
    cases = []
    for bad, match in ((np.nan, "finite"), (np.inf, "finite")):
        v2 = vx.copy()
        v2[0, 1] = bad
        cases.append((v2, vy, match))
    v_in = vx.copy()
    v_in[0, 3] = -1.0  # water flowing INTO the domain through the outlet
    cases.append((v_in, vy, "INTO"))
    v_closed = vx.copy()
    v_closed[0, 0] = 0.5  # closed exterior face
    cases.append((v_closed, vy, "closed"))
    vy_closed = vy.copy()
    vy_closed[1, 1] = 1.0
    cases.append((vx, vy_closed, "closed"))
    for vxb, vyb, match in cases:
        with pytest.raises(TransportError, match=match):
            step(net, M, phase, P, V, R, S, vxb, vyb, 3.0, 3)
    with pytest.raises(TransportError, match="float64"):
        step(net, M, phase, P, V, R, S, vx.astype(np.float32), vy, 3.0, 3)
    with pytest.raises(TransportError, match="numba|implementation"):
        step(net, M, phase, P, V, R, S, vx, vy, 3.0, 3, implementation="numba")
    with pytest.raises(TransportError, match="external"):
        step(net, M, phase, P, V, R, S, vx, vy, 3.0, 3, external_inflow_kg=np.zeros((1, n, 1)))
    with pytest.raises(TransportStepRejected):  # Courant 0.2 * 3 / 0.5 = 1.2 > 0.5 with one substep
        step(net, M, phase, P, V, R, S, vx, vy, 3.0, 1)
    for before, after in zip(snapshot, (M, P, V, R, S, vx, vy, phase.fraction, phase.position_m)):
        np.testing.assert_array_equal(before, after)


def test_extreme_weights_and_hazard_overflow_refusal():
    active = np.ones((1, 2), dtype=bool)
    net = face_network(active, DX)
    zero_y = np.zeros((2, 2))
    # one positive outflow face only for a cell of a 1x2 row: weight 1 whatever the magnitude
    for mag in (1.0e-300, 1.0e300, np.nextafter(0.0, 1.0)):
        w = np.asarray(outflow_weights(net, np.array([[0.0, mag, 0.0]]), zero_y))
        assert w[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0] and w[0, 1].sum() == 0.0
    four = face_network(np.ones((3, 3), dtype=bool), DX)
    tiny = np.nextafter(0.0, 1.0)  # exact multiples of the smallest subnormal keep an exact 1 : 3 ratio
    vx, vy = np.zeros((3, 4)), np.zeros((4, 3))
    vx[1, 2], vx[1, 1] = tiny * 1000.0, -tiny * 3000.0
    np.testing.assert_allclose(np.asarray(outflow_weights(four, vx, vy))[1, 1], [0.25, 0.0, 0.75, 0.0], rtol=1e-12)
    vx[1, 2], vx[1, 1] = 1.0e308, -5.0e307
    np.testing.assert_allclose(np.asarray(outflow_weights(four, vx, vy))[1, 1], [2 / 3, 0.0, 1 / 3, 0.0], rtol=1e-12)
    vx[1, 2], vx[1, 1] = 1.0e300, -1.0e-300  # 1e-600 underflows to weight 0; the mass goes to the other face
    assert np.asarray(outflow_weights(four, vx, vy))[1, 1].tolist() == [1.0, 0.0, 0.0, 0.0]
    big = face_network(np.ones((1, 2), dtype=bool), 10.0)  # hazard (v dt) r = 5 * 1.7e308 overflows
    vxb, vyb = np.array([[0.0, 1.0, 0.0]]), np.zeros((2, 2))
    V, R, S = fld((1, 2), 1, v=1.0, r=1.7e308)
    with pytest.raises(TransportError, match="overflow") as info:
        step(big, zeros((1, 2), 1), empty_phase_state((1, 2), 1, 4, 10.0), zeros((1, 2), 1), V, R, S, vxb, vyb,
             5.0, 1)
    assert not isinstance(info.value, TransportStepRejected)


# --- global closure, face bookkeeping, changing directions -------------------------------------------------
def scenario(backend, seed=7, steps=6, nb=8):
    """Several steps with fresh random signed volumes every step on a 3 x 4 grid with a hole and three outlets."""
    if backend == "cupy":
        xp = cupy_or_skip()
        put, get = xp.asarray, xp.asnumpy
    else:
        xp, put, get = np, (lambda a: a), (lambda a: a)
    rng = np.random.default_rng(seed)
    ny, nx, nc, dt, nsub = 3, 4, 2, 4.0, 8
    active = np.ones((ny, nx), dtype=bool)
    active[1, 1] = False
    ox = np.zeros((ny, nx + 1), dtype=bool)
    oy = np.zeros((ny + 1, nx), dtype=bool)
    ox[2, 0] = ox[0, nx] = oy[0, 3] = True
    net = face_network(active, DX, ox, oy, xp=xp)
    sign_x = np.zeros((ny, nx + 1))
    sign_x[:, 0], sign_x[:, nx] = -1.0, 1.0
    sign_y = np.zeros((ny + 1, nx))
    sign_y[0, :], sign_y[ny, :] = -1.0, 1.0
    V = np.where(active[..., None], rng.uniform(0.0, 0.2, (ny, nx, nc)), 0.0)
    R = rng.uniform(0.0, 5.0, (ny, nx, nc))
    S = rng.random((ny, nx, nc)) < 0.05
    M = put(zeros((ny, nx), nc))
    phase = empty_phase_state((ny, nx), nc, nb, DX, xp=xp)
    log = []
    flips = 0
    previous = None
    for _ in range(steps):
        tx, ty = net.fx_type_host, net.fy_type_host
        vx = np.where(tx == 1, rng.normal(size=tx.shape), 0.0)
        vx = np.where(tx == 2, rng.uniform(0.1, 1.0, tx.shape) * sign_x, vx)
        vy = np.where(ty == 1, rng.normal(size=ty.shape), 0.0)
        vy = np.where(ty == 2, rng.uniform(0.1, 1.0, ty.shape) * sign_y, vy)
        if previous is not None:
            flips += int(np.count_nonzero(np.sign(previous[0]) * np.sign(vx) < 0))
        previous = (vx, vy)
        P = np.where(active[..., None] & (rng.random((ny, nx, nc)) < 0.5), rng.uniform(0.0, 1.0, (ny, nx, nc)), 0.0)
        res = step(net, M, phase, put(P), put(V), put(R), put(S), put(vx), put(vy), dt, nsub)
        t = res.transport
        log.append({"P": P, "before": get(res.mobile_before_kg), "T": get(t.mobile_after_transfer_kg),
                    "dep": get(t.deposition_request_kg), "exp": get(t.export_request_kg),
                    "xg": get(t.x_face_gross_kg), "yg": get(t.y_face_gross_kg), "xn": get(t.x_face_net_kg),
                    "yn": get(t.y_face_net_kg), "rem": get(res.mobile_remaining_kg),
                    "frac": get(res.phase_after.fraction), "pos": get(res.phase_after.position_m),
                    "in": get(t.internal_transfer_in_kg), "out": get(t.internal_transfer_out_kg),
                    "crossing": get(res.crossing_kg), "n_ops": operation_count(nsub, nb)})
        validate_phase(net, res.mobile_remaining_kg, res.phase_after)
        M, phase = res.mobile_remaining_kg, res.phase_after
    return log, flips, nb


def test_global_class_and_face_closure_with_changing_flow_directions():
    log, flips, nb = scenario("numpy")
    assert flips > 0  # the water really did change face directions between steps
    picked = sum(s["P"].sum(axis=(0, 1)) for s in log)
    deposited = sum(s["dep"].sum(axis=(0, 1)) for s in log)
    exported = sum(s["exp"].sum(axis=(0, 1)) for s in log)
    np.testing.assert_allclose(log[-1]["rem"].sum(axis=(0, 1)) + deposited + exported, picked, rtol=1e-12)
    outlet_cells = {(2, 0), (0, 3)}
    for s in log:
        np.testing.assert_allclose(s["T"].sum(axis=(0, 1)), s["before"].sum(axis=(0, 1)), rtol=1e-12)
        # FULL face divergence includes export E, which T still holds: D = (M + P) - (T - E), and
        # after native removal remaining = (M + P) - D - Dep. Pre-native pool: T - (M + P) = in - out (internal).
        div = (s["xn"][:, 1:] - s["xn"][:, :-1]) + (s["yn"][1:] - s["yn"][:-1])
        scale = np.maximum(s["before"], s["T"])
        tol = summation_error_bound_kg(s["n_ops"], scale)
        assert np.all(np.abs(div - (s["before"] - (s["T"] - s["exp"]))) <= tol)
        assert np.all(np.abs(s["rem"] - (s["before"] - div - s["dep"])) <= tol)
        assert np.all(np.abs((s["T"] - s["before"]) - (s["in"] - s["out"])) <= tol)
        assert np.all(s["xg"] >= np.abs(s["xn"])) and np.all(s["yg"] >= np.abs(s["yn"]))
        np.testing.assert_allclose(s["xg"].sum() + s["yg"].sum(), s["crossing"].sum(), rtol=1e-12)
        assert np.all(s["T"][1, 1] == 0.0) and np.all(s["rem"][1, 1] == 0.0)  # the hole holds nothing
        for r, c in np.ndindex(3, 4):
            if (r, c) not in outlet_cells:
                assert np.all(s["exp"][r, c] == 0.0)
        assert np.all(s["dep"] + s["exp"] <= s["T"] * (1 + 4 * np.finfo(float).eps))
        assert_phase_canonical(PhaseState(s["frac"], s["pos"], DX, nb, np), s["rem"], DX)


def test_export_only_regression_pool_and_post_export_relations():
    """Pickup in the last cell of a 1 x 2 row; the packet crosses the outlet at t = 1.25 s of 2 s. Known amounts:
    E = exp(-r dx), Dep = 1 - E, nothing remains, no internal transfer."""
    net = row_net(2)
    vx, vy = row_volumes(2)
    r = 0.5
    V, R, S = fld((1, 2), 1, v=0.4, r=r)
    P = zeros((1, 2), 1)
    P[0, 1, 0] = 1.0
    res = step(net, zeros((1, 2), 1), empty_phase_state((1, 2), 1, 4, DX), P, V, R, S, vx, vy, dt=2.0, nsub=4)
    t = res.transport
    E = math.exp(-r * DX)
    np.testing.assert_allclose(t.export_request_kg[0, :, 0], [0.0, E], rtol=1e-12)
    np.testing.assert_allclose(t.deposition_request_kg[0, :, 0], [0.0, 1.0 - E], rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(t.mobile_after_transfer_kg[0, :, 0], [0.0, 1.0], rtol=1e-12)  # T keeps E and Dep
    assert np.all(res.mobile_remaining_kg == 0.0)
    assert np.all(t.internal_transfer_in_kg == 0.0) and np.all(t.internal_transfer_out_kg == 0.0)
    # Pre-native pool: T - (M + P) = in - out = 0 although the full face divergence is E.
    np.testing.assert_allclose(t.mobile_after_transfer_kg - res.mobile_before_kg, 0.0, atol=1e-15)
    np.testing.assert_allclose(t.x_face_gross_kg[0, :, 0], [0.0, 0.0, E], rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(t.x_face_net_kg[0, :, 0], [0.0, 0.0, E], rtol=1e-12, atol=1e-16)
    div = t.x_face_net_kg[0, 1:, 0] - t.x_face_net_kg[0, :-1, 0]
    np.testing.assert_allclose(div, [0.0, E], rtol=1e-12, atol=1e-16)
    # Full face divergence = (M + P) - (T - E); post-native pool = (M + P) - D - Dep.
    before, T = res.mobile_before_kg[0, :, 0], t.mobile_after_transfer_kg[0, :, 0]
    np.testing.assert_allclose(div, before - (T - t.export_request_kg[0, :, 0]), atol=1e-15)
    native_after = T - t.deposition_request_kg[0, :, 0] - t.export_request_kg[0, :, 0]
    np.testing.assert_allclose(native_after, before - div - t.deposition_request_kg[0, :, 0], atol=1e-15)
    np.testing.assert_allclose(native_after, res.mobile_remaining_kg[0, :, 0], atol=1e-15)


def test_invalid_volume_validation_emits_no_warning():
    import warnings

    net = row_net(3)
    vx, vy = row_volumes(3)
    vx[0, 1] = np.inf  # a closed-adjacent/zero-sign face elsewhere must not trigger inf * 0
    V, R, S = fld((1, 3), 1, v=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TransportError, match="finite"):
            step(net, zeros((1, 3), 1), empty_phase_state((1, 3), 1, 4, DX), zeros((1, 3), 1), V, R, S, vx, vy,
                 3.0, 3)
        with pytest.raises(TransportError, match="finite"):
            outflow_weights(net, vx, vy)


def test_diagnostics_are_separate_from_physics_tolerances():
    n = 4
    net = row_net(n)
    vx, vy = row_volumes(n)
    V, R, S = fld((1, n), 1, v=0.1, r=2.0)
    P = zeros((1, n), 1)
    P[0, 0, 0] = 1.0
    t = step(net, zeros((1, n), 1), empty_phase_state((1, n), 1, 8, DX), P, V, R, S, vx, vy, 12.0, 8).transport
    assert abs(float(t.budget_residual_by_class_kg[0])) <= float(t.budget_tolerance_by_class_kg[0])
    assert float(t.max_courant) == pytest.approx(0.1 * 1.5 / DX)
    assert float(t.max_cell_balance_residual_kg) <= summation_error_bound_kg(operation_count(8, 8), 1.0)


# --- CPU / GPU parity ----------------------------------------------------------------------------------
def test_cupy_matches_numpy_reference():
    cupy_or_skip()
    cpu, _, _ = scenario("numpy")
    gpu, _, _ = scenario("cupy")
    for a, b in zip(cpu, gpu):
        for key in ("T", "dep", "exp", "xg", "yg", "xn", "yn", "rem", "frac"):
            np.testing.assert_allclose(b[key], a[key], rtol=1e-10, atol=1e-13, err_msg=key)
        np.testing.assert_allclose(b["pos"], a["pos"], rtol=1e-10, atol=1e-11, err_msg="pos")


def test_cupy_network_refuses_host_arrays():
    cp = cupy_or_skip()
    net = face_network(np.ones((1, 2), dtype=bool), DX, xp=cp)
    host_phase = empty_phase_state((1, 2), 1, 4, DX)
    V, R, S = fld((1, 2), 1)
    with pytest.raises(TransportError):  # host arrays against a device network: no silent transfer
        step(net, zeros((1, 2), 1), host_phase, zeros((1, 2), 1), V, R, S, np.zeros((1, 3)), np.zeros((2, 2)), 1.0, 1)
