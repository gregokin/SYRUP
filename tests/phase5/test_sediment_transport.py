"""Phase 5a transport operator (`maple_syrup.sediment_transport`) on small
controlled graphs.

Independent references: exact exponential survival `exp(-v t / L)` of a
constant-coefficient pool, the analytic mean deposition lifetime `L / v`
and mean advective transit `n dx / v` (geometric residence), a dense
continuous-time generator integrated with `scipy.linalg.expm`, an
explicitly assembled one-substep matrix, and the continuum
exponential-in-distance deposition pattern that the discretization must
approach under grid refinement. These are equation-level checks, not
MAHLERAN executions. One test applies a transport step through MAPLE's
real `apply_water_process_demand` on the Phase 1 probe bed; one runs the
operator on CuPy when a GPU is available (skipped otherwise).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_transport import (
    CELL_BALANCE_RTOL,
    TransportError,
    TransportStepRejected,
    reaction_survival,
    transport_network,
    transport_step,
    water_demand_from_transport,
)

EPS = np.finfo(np.float64).eps


# --- synthetic terrain (south-first full grids with a ring) --------------------------------
def chain_full(n: int, dz: float = 0.015625, walls: float = 1.0) -> np.ndarray:
    """n cells in one column draining south (row 0 = outlet); E/W walls."""
    z = np.repeat(np.arange(n + 2, dtype=np.float64)[:, None] * dz, 3, axis=1)
    z[:, 0] += walls
    z[:, 2] += walls
    return z


def valley_full(ny: int, nx: int, sy: float = 0.015625, sx: float = 0.03125) -> np.ndarray:
    """V valley (nx odd): side cells drain to the centre column, which drains south."""
    rows, cols = np.indices((ny + 2, nx + 2)).astype(np.float64)
    return rows * sy + np.abs(cols - (nx + 1) / 2) * sx


def random_full(rng, ny: int, nx: int) -> np.ndarray:
    rows = np.indices((ny + 2, nx + 2))[0].astype(np.float64)
    z = 0.025 * rows + rng.uniform(0.0, 0.02, rows.shape)
    z[:, 0] += 10.0
    z[:, -1] += 10.0
    return z


def south_export(z: np.ndarray) -> np.ndarray:
    export = np.zeros(z.shape, dtype=bool)
    export[0, :] = True
    return export


def make_graph(z, *, ff=1.0, dx=0.5, active=None, export=None, xp=None):
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    export = south_export(z) if export is None else export
    return build_routing_graph(z, export, np.full((ny, nx), float(ff)), dx, active_mask=active, xp=xp)


def chain(n, *, dx=0.5):
    graph = make_graph(chain_full(n), dx=dx)
    return graph, transport_network(graph)


def fields(shape, nc, *, v=0.0, rate=0.0, settle=False):
    ny, nx = shape
    return (np.full((ny, nx, nc), float(v)), np.full((ny, nx, nc), float(rate)),
            np.full((ny, nx, nc), bool(settle)))


def run_to_completion(network, pool, v, rate, settle, dt, *, n_substeps=1, max_steps=50000, tol=1e-15):
    """Repeat transport steps, removing the requests as MAPLE would, until
    the TOTAL remaining pool is at most `tol` (the callers assert on the
    sum, so the termination criterion is the sum, not the largest cell).
    Returns cumulative deposition, per-step deposition and export totals,
    and the remaining pool."""
    dep_total = np.zeros_like(pool)
    dep_steps, exports = [], []
    for _ in range(max_steps):
        step = transport_step(network, pool, v, rate, settle, dt, n_substeps=n_substeps)
        pool = step.mobile_after_transfer_kg - step.deposition_request_kg - step.export_request_kg
        dep_total += step.deposition_request_kg
        dep_steps.append(step.deposition_request_kg.sum())
        exports.append(step.export_request_kg.sum())
        if pool.sum() <= tol:
            break
    else:
        raise AssertionError("pool did not empty")
    return dep_total, np.array(dep_steps), np.array(exports), pool


# --- deposition timescale: exact exponential hazard v / L ---------------------------------------
@pytest.mark.parametrize("a", [1.0, 0.5, 0.2, 0.002])
def test_uniform_hazard_total_survival_is_exact_for_any_courant_number(a):
    """Constant v and L in a domain long enough that nothing exports: the
    in-domain pool after time T is exactly exp(-v T / L) M0, whatever the
    Courant number (advection conserves, the reaction is exact)."""
    n, dx, v_s, L = 60, 0.5, 0.8, 0.6
    graph, net = chain(n, dx=dx)
    v, rate, settle = fields(graph.shape, 1, v=v_s, rate=1.0 / L)
    dt = a * dx / v_s
    total_time = 2.5  # 4 cells of travel at most; the outlet is 59 cells away
    n_steps = round(total_time / dt)
    assert abs(n_steps * dt - total_time) < 1e-12
    pool = pool_start(n)
    dep = 0.0
    for _ in range(n_steps):
        step = transport_step(net, pool, v, rate, settle, dt)
        # Upwind can move a vanishing fraction many cells; nothing measurable reaches the outlet.
        assert step.export_request_kg.sum() <= 1e-30
        pool = step.mobile_after_transfer_kg - step.deposition_request_kg - step.export_request_kg
        dep = dep + step.deposition_request_kg.sum()
    survival = math.exp(-v_s * total_time / L)
    assert pool.sum() == pytest.approx(survival, rel=1e-11)
    assert dep == pytest.approx(1.0 - survival, rel=1e-11)


def test_short_travel_distance_deposits_on_the_physical_timescale():
    """Codex's reproducer: 1 kg, L = 0.01 m, v = 0.001 m/s, dx = 0.5 m, 100
    steps of 1 s. Physical survival exp(-10); the earlier crossing-only
    scheme kept 0.82 kg. Mass deposits in its source cell up to the upwind
    leak bound; the mean deposition time is L / v."""
    n, dx, L, v_s, dt = 8, 0.5, 0.01, 0.001, 1.0
    graph, net = chain(n, dx=dx)
    v, rate, settle = fields(graph.shape, 1, v=v_s, rate=1.0 / L)
    pool = pool_start(n)
    dep_total = np.zeros_like(pool)
    for _ in range(100):
        step = transport_step(net, pool, v, rate, settle, dt)
        pool = step.mobile_after_transfer_kg - step.deposition_request_kg - step.export_request_kg
        dep_total += step.deposition_request_kg
    assert pool.sum() == pytest.approx(math.exp(-10.0), rel=1e-9)
    assert pool.sum() + dep_total.sum() == pytest.approx(1.0, rel=1e-12)
    # Fraction of a source pool that ever leaves its cell (documented upwind
    # leak): per substep survive the first half-step, then cross with
    # probability a; s_h = exp(-v r dt / 2), s2 = s_h^2.
    a = v_s * dt / dx
    s_h = math.exp(-0.5 * v_s * dt / L)
    leak = a * s_h / (1.0 - (1.0 - a) * s_h ** 2)
    assert leak < 0.02
    assert dep_total[n - 1, 0, 0] >= (1.0 - leak) * (1.0 - math.exp(-10.0))
    assert 0.0 < dep_total[n - 2, 0, 0] <= leak
    assert dep_total[: n - 2].sum() <= leak * (1.0 + 1e-12)
    assert dep_total[: n - 3].sum() <= 2.0 * leak ** 2
    # Mean deposition time from a finer run to completion: exact exponential
    # sampling gives dt/(1 - e^{-k dt}) - dt/2 = 1/k + k dt^2/12 + O(dt^3).
    dt = 0.1
    _dep, dep_steps, _exports, remaining = run_to_completion(net, pool_start(n), v, rate, settle, dt, tol=1e-13)
    times = dt * np.arange(1, dep_steps.size + 1) - 0.5 * dt
    k = v_s / L
    mean_time = float((times * dep_steps).sum() / dep_steps.sum())
    assert abs(mean_time - L / v_s) <= 1.05 * k * dt ** 2 / 12.0 + 1e-9
    assert remaining.sum() <= 1e-13


def pool_start(n):
    pool = np.zeros((n, 1, 1))
    pool[n - 1] = 1.0
    return pool


# --- advective timing (no deposition) ------------------------------------------------------------
@pytest.mark.parametrize("a", [1.0, 0.5, 0.25])
def test_mean_arrival_time_without_deposition_is_the_physical_transit_time(a):
    n, dx, v_s = 5, 0.5, 0.8
    graph, net = chain(n, dx=dx)
    v, rate, settle = fields(graph.shape, 1, v=v_s)
    dt = a * dx / v_s
    _dep, _steps, exports, _ = run_to_completion(net, pool_start(n), v, rate, settle, dt, tol=1e-16)
    times = dt * np.arange(1, exports.size + 1)
    assert exports.sum() == pytest.approx(1.0, abs=1e-14)
    mean_time = float((times * exports).sum())
    assert mean_time == pytest.approx(n * dx / v_s, rel=1e-9)
    spread = float(((times - mean_time) ** 2 * exports).sum())
    if a == 1.0:
        assert spread == 0.0 and exports.size == n  # exact translation
    else:
        # geometric residence in each of n cells: variance n (1 - a) / a^2 dt^2
        assert spread == pytest.approx(n * (1.0 - a) / a ** 2 * dt ** 2, rel=1e-8)


# --- one-substep matrix reference --------------------------------------------------------------------
def explicit_operator(graph, v, rate, dt, nc):
    """Independent assembly of the Strang substep: S = diag(exp(-v r dt/2)),
    upwind A (A[i,i] = 1 - a_i, A[r(i), i] = a_i for internal cells), export
    Eo = diag(a_i at outlets). T = S A S + (I - S) + (I - S) A S + Eo S."""
    ny, nx = graph.shape
    n = ny * nx
    dx = graph.dx_m
    receiver = graph.receiver.reshape(-1)
    outlet = graph.outlet.reshape(-1)
    T = np.zeros((nc, n, n))
    D = np.zeros((nc, n, n))
    E = np.zeros((nc, n, n))
    identity = np.eye(n)
    for k in range(nc):
        vk = v.reshape(n, nc)[:, k]
        rk = rate.reshape(n, nc)[:, k]
        S = np.diag(np.exp(-0.5 * vk * rk * dt))
        A = np.zeros((n, n))
        Eo = np.zeros((n, n))
        for i in range(n):
            a = vk[i] * dt / dx
            A[i, i] = 1.0 - a
            if outlet[i]:
                Eo[i, i] = a
            elif receiver[i] >= 0:
                A[receiver[i], i] = a
        D[k] = (identity - S) + (identity - S) @ A @ S
        E[k] = Eo @ S
        T[k] = S @ A @ S + D[k] + E[k]
    return T, D, E


@pytest.mark.parametrize("terrain", ["valley", "random"])
def test_operator_matches_explicit_matrix_on_branching_network(terrain):
    rng = np.random.default_rng(7)
    z = valley_full(4, 5) if terrain == "valley" else random_full(rng, 5, 4)
    graph = make_graph(z, dx=0.5)
    net = transport_network(graph)
    ny, nx = graph.shape
    n, nc, dt = ny * nx, 2, 0.4
    v = rng.uniform(0.0, 1.25, (ny, nx, nc))  # a <= 1 at dt 0.4, dx 0.5
    rate = rng.uniform(0.0, 4.0, (ny, nx, nc))
    settle = np.zeros((ny, nx, nc), dtype=bool)
    T_ref, D_ref, E_ref = explicit_operator(graph, v, rate, dt, nc)
    for k in range(nc):
        for i in range(n):
            basis = np.zeros((ny, nx, nc))
            basis.reshape(n, nc)[i, k] = 1.0
            step = transport_step(net, basis, v, rate, settle, dt)
            np.testing.assert_allclose(step.mobile_after_transfer_kg.reshape(n, nc)[:, k], T_ref[k][:, i],
                                       rtol=0, atol=16 * EPS)
            np.testing.assert_allclose(step.deposition_request_kg.reshape(n, nc)[:, k], D_ref[k][:, i],
                                       rtol=0, atol=16 * EPS)
            np.testing.assert_allclose(step.export_request_kg.reshape(n, nc)[:, k], E_ref[k][:, i],
                                       rtol=0, atol=16 * EPS)
            assert not np.any(step.mobile_after_transfer_kg.reshape(n, nc)[:, 1 - k])
    pool = rng.uniform(0.0, 3.0, (ny, nx, nc))
    step = transport_step(net, pool, v, rate, settle, dt)
    for k in range(nc):
        expected = T_ref[k] @ pool.reshape(n, nc)[:, k]
        np.testing.assert_allclose(step.mobile_after_transfer_kg.reshape(n, nc)[:, k], expected, rtol=64 * EPS)
    assert np.all(np.abs(step.budget_residual_by_class_kg) <= step.budget_tolerance_by_class_kg)
    assert np.all(step.budget_tolerance_by_class_kg < 1e-12 * pool.sum())
    assert float(step.max_decay_exponent) == pytest.approx(float((v * rate).max()) * dt)


# --- continuous-time generator reference and temporal convergence -----------------------------------------
def generator_reference(graph, v, rate, pool, total_time):
    """Dense generator of the continuum law on the D4 graph: advection at
    rate v_i / dx to the receiver (or to an export state at outlets) and
    deposition at rate v_i r_i into a per-cell absorbing state. Returns
    (remaining pool, deposited per cell, exported) at `total_time`, one
    class."""
    expm = pytest.importorskip("scipy.linalg").expm
    ny, nx = graph.shape
    n = ny * nx
    dx = graph.dx_m
    receiver = graph.receiver.reshape(-1)
    outlet = graph.outlet.reshape(-1)
    Q = np.zeros((2 * n + 1, 2 * n + 1))
    for i in range(n):
        adv, dep = v[i] / dx, v[i] * rate[i]
        Q[i, i] = -(adv + dep)
        Q[n + i, i] = dep
        if outlet[i]:
            Q[2 * n, i] = adv
        elif receiver[i] >= 0:
            Q[receiver[i], i] = adv
    y0 = np.concatenate([pool, np.zeros(n + 1)])
    y = expm(Q * total_time) @ y0
    return y[:n], y[n:2 * n], y[2 * n]


def test_step_converges_to_the_continuous_time_generator_with_substeps():
    rng = np.random.default_rng(13)
    graph = make_graph(valley_full(3, 3), dx=0.5)
    net = transport_network(graph)
    ny, nx = graph.shape
    n = ny * nx
    v = rng.uniform(0.2, 0.6, n)
    rate = rng.uniform(0.0, 3.0, n)
    pool = rng.uniform(0.0, 1.0, n)
    total_time = 1.0
    ref_pool, ref_dep, ref_exp = generator_reference(graph, v, rate, pool, total_time)
    assert ref_pool.sum() + ref_dep.sum() + ref_exp == pytest.approx(pool.sum(), rel=1e-12)
    settle = np.zeros((ny, nx, 1), dtype=bool)
    errors = []
    for n_sub in (4, 8, 16, 32, 64):
        step = transport_step(net, pool.reshape(ny, nx, 1), v.reshape(ny, nx, 1), rate.reshape(ny, nx, 1), settle,
                              total_time, n_substeps=n_sub)
        remaining = (step.mobile_after_transfer_kg - step.deposition_request_kg - step.export_request_kg).reshape(-1)
        err = (np.abs(remaining - ref_pool).sum() + np.abs(step.deposition_request_kg.reshape(-1) - ref_dep).sum()
               + abs(step.export_request_kg.sum() - ref_exp))
        errors.append(err)
    errors = np.array(errors)
    assert np.all(np.diff(errors) < 0.0)  # monotone convergence
    ratios = errors[:-1] / errors[1:]
    assert np.all(ratios[1:] >= 1.5)  # first order (explicit upwind advection), asymptotically 2
    assert errors[-1] < errors[1] / 4.0  # three halvings from n = 8 to 64
    assert errors[-1] < 0.05 * pool.sum()


def test_internal_substeps_match_separate_calls():
    n, dx = 4, 0.5
    _graph, net = chain(n, dx=dx)
    rng = np.random.default_rng(3)
    nc = 2
    pool = rng.uniform(0.0, 1.0, (n, 1, nc))
    v = rng.uniform(0.0, 1.0, (n, 1, nc))
    rate = rng.uniform(0.0, 3.0, (n, 1, nc))
    settle = np.zeros((n, 1, nc), dtype=bool)
    one = transport_step(net, pool, v, rate, settle, 0.4, n_substeps=4)
    p, dep, exp_ = pool, 0.0, 0.0
    for _ in range(4):
        s = transport_step(net, p, v, rate, settle, 0.1)
        p = s.mobile_after_transfer_kg - s.deposition_request_kg - s.export_request_kg
        dep = dep + s.deposition_request_kg
        exp_ = exp_ + s.export_request_kg
    np.testing.assert_allclose(one.deposition_request_kg, dep, rtol=0, atol=1e-14)
    np.testing.assert_allclose(one.export_request_kg, exp_, rtol=0, atol=1e-14)
    np.testing.assert_allclose(one.mobile_after_transfer_kg - one.deposition_request_kg - one.export_request_kg,
                               p, rtol=0, atol=1e-14)
    assert one.n_substeps == 4 and float(one.max_courant) == pytest.approx(float(v.max()) * 0.1 / dx)


# --- spatial pattern: grid convergence to the exponential distance law ---------------------------------------
def test_deposition_pattern_converges_to_exponential_in_distance_with_grid_refinement():
    """At fixed Courant number the cumulative deposition over the first
    1 m from the source converges to 1 - exp(-1 m / L) as dx -> 0 (first
    order: the upwind residence per cell is exponential, not fixed)."""
    L, a, v_s, span = 0.6, 0.5, 1.0, 1.0
    exact = 1.0 - math.exp(-span / L)
    errors = []
    for dx in (0.5, 0.25, 0.125, 0.0625):
        m = round(span / dx)
        n = m + 6
        graph, net = chain(n, dx=dx)
        v, rate, settle = fields(graph.shape, 1, v=v_s, rate=1.0 / L)
        dep_total, _steps, _exports, _ = run_to_completion(net, pool_start(n), v, rate, settle, a * dx / v_s,
                                                           tol=1e-14)
        within = dep_total[n - m:, 0, 0].sum()  # the source cell and the m - 1 cells below it
        errors.append(abs(within - exact))
    errors = np.array(errors)
    assert np.all(np.diff(errors) < 0.0) and np.all(errors[:-1] / errors[1:] >= 1.5)
    # Expected from the jump-count distribution P(J >= j) = q (s_h q)^(j-1),
    # q = a s_h / (1 - (1 - a) s_h^2): about 0.11, 0.053, 0.027, 0.013.
    assert errors[0] > 0.05 and errors[-1] < 0.02


# --- requests, junctions, faces, divergence -------------------------------------------------------------------
def test_requests_stay_in_pool_and_are_bounded_by_it():
    graph, net = chain(3)
    v, rate, settle = fields(graph.shape, 1, v=0.6, rate=2.0)
    pool = np.array([[[1.0]], [[2.0]], [[3.0]]])
    step = transport_step(net, pool, v, rate, settle, 0.5)
    T, dep, exp_ = step.mobile_after_transfer_kg, step.deposition_request_kg, step.export_request_kg
    assert np.all(dep + exp_ <= T)
    assert exp_[0, 0, 0] > 0.0 and np.all(exp_[1:] == 0.0)  # only the outlet exports
    assert T[0, 0, 0] >= exp_[0, 0, 0] + dep[0, 0, 0]
    np.testing.assert_allclose(T.sum(axis=(0, 1)), pool.sum(axis=(0, 1)), rtol=16 * EPS)
    np.testing.assert_allclose(T - pool, step.divergence_kg, atol=CELL_BALANCE_RTOL * 2 * 6.0)
    assert float(step.max_courant) == pytest.approx(0.6)
    assert np.all(step.decay_deposition_kg > 0.0) and not np.any(step.settled_kg)


def face_divergence(step):
    xn, yn = step.x_face_net_kg, step.y_face_net_kg
    return xn[:, :-1] - xn[:, 1:] + yn[:-1, :] - yn[1:, :]


def test_junction_face_flux_and_divergence_identities():
    rng = np.random.default_rng(11)
    graph = make_graph(valley_full(4, 5), dx=0.5)
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 3
    pool = rng.uniform(0.0, 2.0, (ny, nx, nc))
    v = rng.uniform(0.2, 1.0, (ny, nx, nc))
    rate = rng.uniform(0.0, 2.0, (ny, nx, nc))
    settle = np.zeros((ny, nx, nc), dtype=bool)
    step = transport_step(net, pool, v, rate, settle, 0.5)
    T, E = step.mobile_after_transfer_kg, step.export_request_kg
    inflow, outflow = step.internal_transfer_in_kg, step.internal_transfer_out_kg
    scale = pool + inflow + outflow
    np.testing.assert_allclose(T - pool, inflow - outflow, rtol=0, atol=(CELL_BALANCE_RTOL * 2 * scale).max())
    np.testing.assert_allclose(inflow.sum(axis=(0, 1)), outflow.sum(axis=(0, 1)), rtol=32 * EPS)
    centre = nx // 2
    assert np.all(inflow[:, centre] > 0.0)  # both sides feed the centre column, plus upslope
    assert np.all(inflow[:, [0, nx - 1]] == 0.0)  # ridge cells have no donors
    gross = step.x_face_gross_kg.sum(axis=(0, 1)) + step.y_face_gross_kg.sum(axis=(0, 1))
    np.testing.assert_allclose(gross, (outflow + E).sum(axis=(0, 1)), rtol=32 * EPS)
    np.testing.assert_allclose(face_divergence(step) + E, T - pool, rtol=0, atol=(CELL_BALANCE_RTOL * 4 * scale).max())
    # Directions: side cells cross x faces, the centre column crosses y faces (south, negative).
    assert np.all(step.x_face_net_kg[:, centre, :] > 0.0)  # east-flowing from the west side into the centre
    assert np.all(step.x_face_net_kg[:, centre + 1, :] < 0.0)  # west-flowing from the east side
    assert np.all(step.y_face_net_kg[:ny, centre, :] < 0.0) and not np.any(step.y_face_net_kg[ny])
    assert np.all(step.y_face_net_kg[0, centre, :] == -E[0, centre, :])  # export crosses the south boundary face
    assert not np.any(step.y_face_gross_kg[:, [c for c in range(nx) if c != centre]])
    assert E[0, centre].sum() > 0.0 and E[1:].sum() == 0.0 and E[0, [c for c in range(nx) if c != centre]].sum() == 0.0
    assert np.all(T[0, centre] >= E[0, centre] + step.deposition_request_kg[0, centre])
    assert net.n_active == ny * nx and net.graph_input_sha256 == graph.input_sha256


def test_north_draining_rectangular_grid_faces_and_chain_shapes():
    """nx != ny with a north outlet (the y-face index bug of the first
    version was hidden on square grids): index (face row, column), sign
    +1 for a north crossing, export through the north boundary face."""
    ny, nx, nc = 3, 5, 2
    rows, cols = np.indices((ny + 2, nx + 2)).astype(np.float64)
    z = -rows * 0.015625 + np.abs(cols - (nx + 1) / 2) * 0.03125 + 1.0
    export = np.zeros(z.shape, dtype=bool)
    export[-1, :] = True
    graph = make_graph(z, dx=0.5, export=export)
    centre = nx // 2
    assert np.all(graph.aspect[:, centre] == 1) and int(graph.outlet.sum()) == 1 and graph.outlet[ny - 1, centre]
    net = transport_network(graph)
    rng = np.random.default_rng(2)
    pool = rng.uniform(0.5, 1.0, (ny, nx, nc))
    v = np.full((ny, nx, nc), 0.5)
    rate = np.full((ny, nx, nc), 1.0)
    settle = np.zeros((ny, nx, nc), dtype=bool)
    step = transport_step(net, pool, v, rate, settle, 0.5)
    T, E = step.mobile_after_transfer_kg, step.export_request_kg
    assert step.y_face_gross_kg.shape == (ny + 1, nx, nc) and step.x_face_gross_kg.shape == (ny, nx + 1, nc)
    assert np.all(step.y_face_net_kg[1:, centre, :] > 0.0) and not np.any(step.y_face_net_kg[0])
    np.testing.assert_array_equal(step.y_face_net_kg[ny, centre], E[ny - 1, centre])
    assert not np.any(step.y_face_gross_kg[:, [c for c in range(nx) if c != centre]])
    scale = pool + step.internal_transfer_in_kg + step.internal_transfer_out_kg
    np.testing.assert_allclose(face_divergence(step) + E, T - pool, rtol=0, atol=(CELL_BALANCE_RTOL * 4 * scale).max())
    # Tall and wide single chains build and step without index errors.
    for shape_z in (chain_full(8), chain_full(2)):
        g = make_graph(shape_z)
        s = transport_step(transport_network(g), np.ones(g.shape + (1,)), *fields(g.shape, 1, v=0.5, rate=1.0), 0.5)
        assert s.y_face_gross_kg.shape == (g.shape[0] + 1, 1, 1)


# --- settling ---------------------------------------------------------------------------------------------
def test_settled_cells_request_their_whole_pool_including_arrivals():
    graph, net = chain(3)
    nc = 2
    v, rate, settle = fields(graph.shape, nc, v=1.0, rate=0.5)
    settle[1, 0, :] = True  # the middle cell is dry / has no capacity
    pool = np.array([[[1.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]]])
    dt = 0.5
    step = transport_step(net, pool, v, rate, settle, dt)  # a = 1
    T, dep, exp_ = step.mobile_after_transfer_kg, step.deposition_request_kg, step.export_request_kg
    s_half = reaction_survival(1.0, 0.5, 0.5 * dt)
    arrival = pool[2] * s_half  # top cell: half reaction, then the whole pool crosses
    np.testing.assert_allclose(T[1], pool[1] + arrival, rtol=4 * EPS)
    np.testing.assert_array_equal(dep[1], T[1])  # everything there settles
    np.testing.assert_allclose(step.settled_kg[1], pool[1] + arrival, rtol=4 * EPS)
    assert not np.any(step.decay_deposition_kg[1])  # no separate decay on a settled cell
    assert np.all(step.internal_transfer_out_kg[1] == 0.0)
    assert np.all(step.internal_transfer_in_kg[0] == 0.0)  # nothing passes through the settled cell
    np.testing.assert_allclose(exp_[0], pool[0] * s_half, rtol=4 * EPS)
    np.testing.assert_allclose(step.decay_deposition_kg[0], pool[0] * (1.0 - s_half), rtol=4 * EPS)
    np.testing.assert_allclose(T.sum(axis=(0, 1)), pool.sum(axis=(0, 1)), rtol=16 * EPS)


def test_all_dry_domain_settles_everything_without_erasing():
    graph, net = chain(4)
    v, rate, settle = fields(graph.shape, 2, v=0.0, rate=0.0, settle=True)
    pool = np.arange(8, dtype=np.float64).reshape(4, 1, 2) + 1.0
    step = transport_step(net, pool, v, rate, settle, 1.0)
    np.testing.assert_array_equal(step.mobile_after_transfer_kg, pool)
    np.testing.assert_array_equal(step.deposition_request_kg, pool)
    assert not np.any(step.export_request_kg) and not np.any(step.internal_transfer_in_kg)
    np.testing.assert_array_equal(step.budget_residual_by_class_kg, 0.0)


# --- trivial and invalid inputs -------------------------------------------------------------------------------
def test_zero_mass_and_zero_velocity_are_identities():
    graph, net = chain(3)
    v, rate, settle = fields(graph.shape, 2, v=0.9, rate=1.0)
    zero = np.zeros((3, 1, 2))
    step = transport_step(net, zero, v, rate, settle, 0.5)
    for array in (step.mobile_after_transfer_kg, step.deposition_request_kg, step.export_request_kg,
                  step.x_face_gross_kg, step.y_face_net_kg):
        assert not np.any(array)
    pool = np.full((3, 1, 2), 2.0)
    still = transport_step(net, pool, np.zeros_like(v), rate, settle, 0.5)
    np.testing.assert_array_equal(still.mobile_after_transfer_kg, pool)  # v = 0: no hazard, no advection
    assert not np.any(still.deposition_request_kg) and not np.any(still.export_request_kg)
    assert float(still.max_courant) == 0.0 and float(still.max_decay_exponent) == 0.0
    moving = transport_step(net, pool, v, np.zeros_like(rate), settle, 0.5)  # r = 0: pure advection
    assert not np.any(moving.deposition_request_kg) and moving.export_request_kg[0].sum() > 0.0


def test_rejections_leave_inputs_untouched():
    graph, net = chain(4)
    nc = 2
    v, rate, settle = fields(graph.shape, nc, v=1.0, rate=1.0)
    pool = np.full((4, 1, nc), 1.0)
    inputs = [a.copy() for a in (pool, v, rate, settle)]
    with pytest.raises(TransportStepRejected, match="Courant"):
        transport_step(net, pool, v, rate, settle, 0.75)  # a = 1.5
    assert issubclass(TransportStepRejected, TransportError)
    transport_step(net, pool, v, rate, settle, 0.75, n_substeps=2)  # a = 0.75 per substep: accepted
    with pytest.raises(TransportStepRejected):
        transport_step(net, pool, v, rate, settle, 0.5, courant_max=0.5)
    # The Courant check is on the SEDIMENT velocity supplied, not inferred
    # from any water step: a recession velocity above the water's is caught here.
    with pytest.raises(TransportStepRejected):
        transport_step(net, pool, np.full_like(v, 1.2), rate, settle, 0.5)
    bad_cases = [
        ({"mobile_kg": np.where(np.arange(4)[:, None, None] == 1, -1e-9, pool)}, ">= 0"),
        ({"mobile_kg": np.where(np.arange(4)[:, None, None] == 2, np.nan, pool)}, "finite"),
        ({"sediment_velocity_m_s": np.full_like(v, -0.1)}, ">= 0"),
        ({"deposition_rate_per_m": np.full_like(rate, np.inf)}, "finite"),
        ({"settle_mask": settle.astype(np.float64)}, "bool"),
        ({"mobile_kg": pool[:, :, :1]}, "shape"),
        ({"dt_s": 0.0}, "dt_s"),
        ({"n_substeps": 0}, "n_substeps"),
        ({"n_substeps": 2.0}, "n_substeps"),
        ({"courant_max": 1.5}, "courant_max"),
    ]
    for override, match in bad_cases:
        kwargs = {"mobile_kg": pool, "sediment_velocity_m_s": v, "deposition_rate_per_m": rate,
                  "settle_mask": settle, "dt_s": 0.5}
        kwargs.update(override)
        with pytest.raises(TransportError, match=match) as info:
            transport_step(net, **kwargs)
        assert not isinstance(info.value, TransportStepRejected)
    for before, after in zip(inputs, (pool, v, rate, settle), strict=True):
        assert np.array_equal(before, after)


def test_mobile_mass_on_inactive_cells_is_refused_not_erased():
    active = np.ones((4, 1), dtype=bool)
    active[3, 0] = False
    graph = make_graph(chain_full(4), active=active)
    net = transport_network(graph)
    v, rate, settle = fields(graph.shape, 1, v=0.5, rate=1.0)
    v[3] = 0.0
    pool = np.ones((4, 1, 1))
    with pytest.raises(TransportError, match="inactive"):
        transport_step(net, pool, v, rate, settle, 0.5)
    pool[3] = 0.0
    step = transport_step(net, pool, v, rate, settle, 0.5)
    assert step.mobile_after_transfer_kg[3, 0, 0] == 0.0 and net.n_active == 3
    v[3] = 0.2
    with pytest.raises(TransportError, match="inactive"):
        transport_step(net, pool, v, rate, settle, 0.5)


# --- MAPLE integration through the real water step ------------------------------------------------------------
def probe_graph_and_state():
    from maple_syrup.probe import build_minimal_state

    state = build_minimal_state()
    ny, nx = state.surface_elevation_m.shape
    z = np.full((ny + 2, nx + 2), 10.0)
    z[1:-1, 1:-1] = state.surface_elevation_m
    z[0, 1:-1] = state.surface_elevation_m[0] - 0.01  # south ring lower than row 0: outlets
    export = np.zeros(z.shape, dtype=bool)
    export[0, :] = True
    graph = build_routing_graph(z, export, np.full((ny, nx), 5.0), state.geometry.dx_m)
    assert np.all(graph.aspect == 3) and int(graph.outlet.sum()) == nx  # every cell drains south
    return graph, state


def test_transport_step_applies_through_maple_water_step():
    from maple.water import apply_water_process_demand

    graph, state = probe_graph_and_state()
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = state.active_layer.mass_kg.shape[-1]
    rng = np.random.default_rng(5)
    mobile = rng.uniform(0.0, 0.01, (ny, nx, nc))
    v = np.full((ny, nx, nc), 0.4)
    rate = np.full((ny, nx, nc), 1.0)
    settle = np.zeros((ny, nx, nc), dtype=bool)
    settle[2, 1, :] = True
    step = transport_step(net, mobile, v, rate, settle, 1.0)
    pickup = np.zeros((ny, nx, nc))
    pickup[1, 2, 0] = 0.002
    demand = water_demand_from_transport(step, pickup)
    water = dataclasses.replace(state.water, mobile_mass_by_cell_class_kg=step.mobile_after_transfer_kg)
    result = apply_water_process_demand(
        state.voxel_column, state.active_layer, water, state.ledger, demand, state.geometry, state.grain_classes,
        state.mass_resolution_kg, adapter_name="maple_syrup/phase5a_test",
    )
    expected_mobile = step.mobile_after_transfer_kg + result.actual_removal_by_cell_class_kg \
        - step.deposition_request_kg - step.export_request_kg
    np.testing.assert_allclose(result.new_water.mobile_mass_by_cell_class_kg, expected_mobile, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.boundary_export_by_class_kg, step.export_request_kg.sum(axis=(0, 1)),
                               rtol=16 * EPS)
    resolution = 4.0 * state.mass_resolution_kg + 1e-15  # MAPLE's bed routines may leave a sub-resolution residual
    np.testing.assert_allclose(result.deposition_by_cell_class_kg, step.deposition_request_kg, rtol=0, atol=resolution)
    np.testing.assert_allclose(result.actual_removal_by_cell_class_kg, pickup, rtol=0, atol=resolution)
    before = step.mobile_after_transfer_kg.sum(axis=(0, 1))
    np.testing.assert_allclose(result.mobile_mass_before_by_class_kg, before, rtol=16 * EPS)
    np.testing.assert_allclose(result.mobile_mass_after_by_class_kg,
                               before + pickup.sum(axis=(0, 1)) - step.deposition_request_kg.sum(axis=(0, 1))
                               - step.export_request_kg.sum(axis=(0, 1)), rtol=64 * EPS, atol=1e-15)
    np.testing.assert_allclose(before, mobile.sum(axis=(0, 1)), rtol=16 * EPS)  # internal transfer is invisible
    assert result.face_flux.x_face_crossing_mass_kg is step.x_face_gross_kg
    assert not np.any(state.water.mobile_mass_by_cell_class_kg)  # MAPLE-owned inputs unchanged


def test_water_demand_from_transport_validates_pickup():
    graph, net = chain(2)
    v, rate, settle = fields(graph.shape, 1, v=0.5, rate=1.0)
    step = transport_step(net, np.ones((2, 1, 1)), v, rate, settle, 0.5)
    with pytest.raises(TransportError, match="shape"):
        water_demand_from_transport(step, np.ones((2, 1, 2)))
    with pytest.raises(TransportError, match=">= 0"):
        water_demand_from_transport(step, -np.ones((2, 1, 1)))
    with pytest.raises(TransportError, match="finite"):
        water_demand_from_transport(step, np.full((2, 1, 1), np.nan))
    demand = water_demand_from_transport(step, np.zeros((2, 1, 1)))
    assert demand.boundary_export_by_cell_class_kg is step.export_request_kg
    assert demand.face_flux.y_face_net_crossing_mass_kg.shape == (3, 1, 1)


# --- optional GPU execution -----------------------------------------------------------------------------------
def test_cupy_backend_matches_numpy_when_a_gpu_is_available():
    from maple.core.backend import cupy_module, gpu_execution_available, to_host

    if not gpu_execution_available():
        pytest.skip("CuPy with a CUDA device is not available; GPU path not exercised, no claim made")
    cp = cupy_module()
    rng = np.random.default_rng(17)
    z = valley_full(4, 5)
    graph_np = make_graph(z, dx=0.5)
    graph_cp = make_graph(z, dx=0.5, xp=cp)
    ny, nx, nc = 4, 5, 3
    pool = rng.uniform(0.0, 2.0, (ny, nx, nc))
    v = rng.uniform(0.1, 0.9, (ny, nx, nc))
    rate = rng.uniform(0.0, 2.0, (ny, nx, nc))
    settle = np.zeros((ny, nx, nc), dtype=bool)
    settle[3, 0, :] = True
    host = transport_step(transport_network(graph_np), pool, v, rate, settle, 0.5, n_substeps=3)
    device = transport_step(transport_network(graph_cp), cp.asarray(pool), cp.asarray(v), cp.asarray(rate),
                            cp.asarray(settle), 0.5, n_substeps=3)
    for name in ("mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg", "x_face_net_kg",
                 "y_face_net_kg", "divergence_kg"):
        np.testing.assert_allclose(to_host(getattr(device, name)), getattr(host, name), rtol=1e-12, atol=1e-15)
