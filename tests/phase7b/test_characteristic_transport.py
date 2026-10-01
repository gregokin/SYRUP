"""Phase 7b candidate kernel `maple_syrup.characteristic_transport` on small
controlled graphs.

Independent references: the legacy distance convention `exp(-k dx / L)`
for face crossings of an impulse injected at the upstream face; exact
two-cell characteristic arithmetic with differing receiver laws; the
no-deposition first moment `sum m x` advancing by `M v dt`; an UNMERGED
packet reference for a continuous source (exact delay `dx / v`, every
packet kept separately), against which the bounded-bin merge is measured
with predeclared, explicit pulsation limits. These are equation-level
checks, not MAHLERAN executions; no event, MAPLE bed transaction or
checkpoint is exercised here.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX,
    CharacteristicStep,
    bin_index,
    characteristic_step,
    empty_phase_state,
    fraction_sum_tolerance,
    validate_phase_state,
)
from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_transport import (
    TransportError,
    TransportStepRejected,
    transport_network,
)

EPS = np.finfo(np.float64).eps
DX = 0.5


def maybe_skip(implementation):
    if implementation == "numba":
        pytest.importorskip("numba")


# --- terrain -------------------------------------------------------------------------------------
def chain_full(n: int, dz: float = 0.015625, walls: float = 1.0) -> np.ndarray:
    """n cells in one column draining south (row 0 = outlet); E/W walls."""
    z = np.repeat(np.arange(n + 2, dtype=np.float64)[:, None] * dz, 3, axis=1)
    z[:, 0] += walls
    z[:, 2] += walls
    return z


def valley_full(ny: int, nx: int, sy: float = 0.015625, sx: float = 0.03125) -> np.ndarray:
    rows, cols = np.indices((ny + 2, nx + 2)).astype(np.float64)
    return rows * sy + np.abs(cols - (nx + 1) / 2) * sx


def west_plane_full(ny: int, nx: int, dz: float = 0.015, wall: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    z = np.repeat(np.arange(nx + 2, dtype=np.float64)[None, :] * dz, ny + 2, axis=0)
    z[[0, -1], :] += wall
    export = np.zeros(z.shape, dtype=bool)
    export[:, 0] = True
    return z, export


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


def make_graph(z, *, export=None, dx=DX, xp=None, active=None):
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    export = south_export(z) if export is None else export
    return build_routing_graph(z, export, np.full((ny, nx), 1.0), dx, xp=xp, active_mask=active)


def chain(n, *, dx=DX, xp=None):
    graph = make_graph(chain_full(n), dx=dx, xp=xp)
    return graph, transport_network(graph)


def fields(shape, nc, *, v=0.0, rate=0.0, settle=False):
    ny, nx = shape
    return (np.full((ny, nx, nc), float(v)), np.full((ny, nx, nc), float(rate)),
            np.full((ny, nx, nc), bool(settle)))


def zeros(shape, nc):
    return np.zeros((*shape, nc), dtype=np.float64)


class Runner:
    """Repeat characteristic steps the way a caller would: the remaining
    pool and its phase become the next step's pre-pickup inputs."""

    def __init__(self, network, n_bins, nc, implementation="array"):
        self.network = network
        self.shape = network.shape
        self.nc = nc
        self.mobile = zeros(self.shape, nc)
        self.phase = empty_phase_state(self.shape, nc, n_bins, network.dx_m)
        self.implementation = implementation
        self.dep = zeros(self.shape, nc)
        self.export = zeros(self.shape, nc)
        self.crossing = zeros(self.shape, nc)
        self.steps: list[CharacteristicStep] = []

    def step(self, pickup, v, rate, settle, dt, **kwargs):
        result = characteristic_step(self.network, self.mobile, self.phase, pickup, v, rate, settle, dt,
                                     implementation=self.implementation, **kwargs)
        self.mobile = result.mobile_remaining_kg
        self.phase = result.phase_after
        self.dep = self.dep + result.transport.deposition_request_kg
        self.export = self.export + result.transport.export_request_kg
        self.crossing = self.crossing + result.crossing_kg
        self.steps.append(result)
        return result

    def total(self):
        return float(self.dep.sum() + self.export.sum() + self.mobile.sum())


# --- T01: constant-law impulse on a chain: legacy distance convention exactly ----------------------
@pytest.mark.parametrize("n_bins", [8, 32])
@pytest.mark.parametrize("implementation", ["array", "numba"])
@pytest.mark.parametrize("v", [0.01, 1.0 / 64.0])
def test_impulse_face_crossings_follow_exp_minus_k_dx_over_L(n_bins, implementation, v):
    maybe_skip(implementation)
    n, L = 6, 0.05
    graph, net = chain(n)
    vel, rate, settle = fields(graph.shape, 1, v=v, rate=1.0 / L)
    run = Runner(net, n_bins, 1, implementation)
    pickup = zeros(graph.shape, 1)
    pickup[n - 1, 0, 0] = 1.0
    first = None
    for t in range(1, 601):
        result = run.step(pickup, vel, rate, settle, 1.0)
        pickup = zeros(graph.shape, 1)
        if first is None and float(result.transport.export_request_kg.sum()) > 0.0:
            first = t
    target = np.exp(-np.arange(1, n + 1) * DX / L)
    np.testing.assert_allclose(run.crossing[::-1, 0, 0], target, rtol=1e-12, atol=1e-35)
    assert first == round(n * DX / v)  # finite arrival: n cells at v, no early leak
    assert abs(run.total() - 1.0) <= 1e-13
    assert float(run.mobile.sum()) <= 1e-30
    # deposition in the top cell is 1 - exp(-dx / L); in cell k below it the legacy bin
    exact_dep = np.exp(-np.arange(0, n) * DX / L) - np.exp(-np.arange(1, n + 1) * DX / L)
    np.testing.assert_allclose(run.dep[::-1, 0, 0], exact_dep, rtol=1e-12, atol=1e-35)


# --- T02: no deposition: finite front, exact first moment ------------------------------------------
@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_no_deposition_front_and_first_moment(implementation):
    maybe_skip(implementation)
    v = 1.0 / 64.0  # binary-exact: 32 steps per cell
    graph, net = chain(3)
    vel, rate, settle = fields(graph.shape, 1, v=v, rate=0.0)
    run = Runner(net, 8, 1, implementation)
    pickup = zeros(graph.shape, 1)
    pickup[2, 0, 0] = 2.0
    exported_at = None
    for t in range(1, 200):
        result = run.step(pickup, vel, rate, settle, 1.0)
        pickup = zeros(graph.shape, 1)
        if exported_at is None and float(result.transport.export_request_kg.sum()) > 0.0:
            exported_at = t
        assert float(result.transport.deposition_request_kg.sum()) == 0.0
    assert exported_at == 3 * 32
    assert run.export[0, 0, 0] == 2.0 and float(run.mobile.sum()) == 0.0
    # continuous source at r = 0: sum m x advances by M v dt per step (mass-weighted merge);
    # the pickup packet enters at x = 0 and travels in its injection step like every packet
    run = Runner(net, 4, 1, implementation)
    moment = 0.0
    mass = 0.0
    for t in range(1, 25):  # 24 steps: nothing reaches a face (24 / 64 < 0.5)
        pickup = zeros(graph.shape, 1)
        pickup[2, 0, 0] = 0.1 * t
        run.step(pickup, vel, rate, settle, 1.0)
        mass += 0.1 * t
        moment += mass * v  # every unit of mass present (including the new packet) moves v dt
        w = run.mobile[..., None] * run.phase.fraction
        np.testing.assert_allclose(float((w * run.phase.position_m).sum()), moment, rtol=1e-12)
        np.testing.assert_allclose(float(w.sum()), mass, rtol=1e-13)


# --- T03: differing receiver v and r: exact two-cell path split ------------------------------------
@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_two_cell_path_with_differing_receiver_laws_is_exact(implementation):
    maybe_skip(implementation)
    graph, net = chain(2)
    ny, nx = graph.shape
    v1, r1, v2, r2 = 0.2, 3.0, 0.1, 7.0
    dt = 1.0
    vel = zeros(graph.shape, 1)
    rate = zeros(graph.shape, 1)
    vel[1, 0, 0], rate[1, 0, 0] = v1, r1  # upstream (row 1)
    vel[0, 0, 0], rate[0, 0, 0] = v2, r2  # outlet (row 0)
    settle = np.zeros((ny, nx, 1), dtype=bool)
    # existing packet at x0 in the upstream cell: place it through two r = 0 steps
    # (binary-exact 0.1875 + 0.1875 = 0.375; Courant 0.375 each)
    x0 = 0.375
    run = Runner(net, 8, 1, implementation)
    pickup = zeros(graph.shape, 1)
    pickup[1, 0, 0] = 1.0
    place = zeros(graph.shape, 1)
    place[1, 0, 0] = 0.1875
    run.step(pickup, place, zeros(graph.shape, 1), settle, dt)
    run.step(zeros(graph.shape, 1), place, zeros(graph.shape, 1), settle, dt)
    assert run.phase.position_m[1, 0, 0, int(bin_index(np.array(x0), DX, 8, np))] == x0
    result = run.step(zeros(graph.shape, 1), vel, rate, settle, dt)
    tau = (DX - x0) / v1
    s1 = math.exp(-r1 * (DX - x0))
    d = v2 * (dt - tau)
    s2 = s1 * math.exp(-r2 * d)
    tr = result.transport
    np.testing.assert_allclose(tr.deposition_request_kg[1, 0, 0], 1.0 - s1, rtol=1e-13)
    np.testing.assert_allclose(tr.internal_transfer_out_kg[1, 0, 0], s1, rtol=1e-13)
    np.testing.assert_allclose(tr.internal_transfer_in_kg[0, 0, 0], s1, rtol=1e-13)
    np.testing.assert_allclose(tr.deposition_request_kg[0, 0, 0], s1 - s2, rtol=1e-13)
    np.testing.assert_allclose(result.arrival_deposition_kg[0, 0, 0], s1 - s2, rtol=1e-13)
    np.testing.assert_allclose(result.mobile_remaining_kg[0, 0, 0], s2, rtol=1e-13)
    assert result.mobile_remaining_kg[1, 0, 0] == 0.0
    b = int(bin_index(np.array(d), DX, 8, np))
    np.testing.assert_allclose(result.phase_after.position_m[0, 0, 0, b], d, rtol=1e-13)
    assert result.phase_after.fraction[0, 0, 0, b] == 1.0
    assert float(tr.export_request_kg.sum()) == 0.0
    np.testing.assert_allclose(tr.mobile_after_transfer_kg.sum(), 1.0, rtol=1e-14)


# --- T04: v -> 0, r -> 0, settle toggles ------------------------------------------------------------
@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_zero_velocity_zero_rate_and_settle_transitions(implementation):
    maybe_skip(implementation)
    graph, net = chain(3)
    vel, rate, settle = fields(graph.shape, 2, v=0.05, rate=4.0)
    run = Runner(net, 8, 2, implementation)
    pickup = zeros(graph.shape, 2)
    pickup[2, 0, :] = [1.0, 3.0]
    run.step(pickup, vel, rate, settle, 1.0)
    before_phase = run.phase
    before_mobile = run.mobile.copy()
    # v = 0: nothing moves, nothing deposits even with r > 0; phase unchanged
    result = run.step(zeros(graph.shape, 2), vel * 0.0, rate, settle, 1.0)
    assert float(result.transport.deposition_request_kg.sum()) == 0.0
    np.testing.assert_allclose(run.mobile, before_mobile, rtol=4 * EPS, atol=0)
    np.testing.assert_allclose(run.phase.fraction, before_phase.fraction, rtol=4 * EPS, atol=0)
    np.testing.assert_array_equal(run.phase.position_m, before_phase.position_m)
    # r = 0: moves without depositing
    result = run.step(zeros(graph.shape, 2), vel, rate * 0.0, settle, 1.0)
    assert float(result.transport.deposition_request_kg.sum()) == 0.0
    np.testing.assert_allclose(run.mobile.sum(), before_mobile.sum(), rtol=1e-14)
    assert float((run.phase.position_m * run.phase.fraction).sum()) > float(
        (before_phase.position_m * before_phase.fraction).sum())
    # settle everywhere: the whole pool plus new pickup is a deposition request; canonical phase
    result = run.step(pickup, vel, rate, ~settle, 1.0)
    np.testing.assert_allclose(result.transport.settled_kg.sum(), before_mobile.sum() + pickup.sum(), rtol=1e-14)
    assert float(run.mobile.sum()) == 0.0
    assert np.all(run.phase.fraction[..., 0] == 1.0) and not np.any(run.phase.fraction[..., 1:])
    assert not np.any(run.phase.position_m)


# --- T05: source injection beside an existing pool ---------------------------------------------------
@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_pickup_is_a_new_packet_at_the_upstream_face_not_merged_with_older_mass(implementation):
    maybe_skip(implementation)
    graph, net = chain(2)
    nb = 8
    run = Runner(net, nb, 1, implementation)
    vel, rate, settle = fields(graph.shape, 1, v=0.1, rate=0.0)
    pickup = zeros(graph.shape, 1)
    pickup[1, 0, 0] = 1.0
    for _ in range(3):  # the 1 kg packet (injected once) travels to x = 0.3
        run.step(pickup, vel, rate, settle, 1.0)
        pickup = zeros(graph.shape, 1)
    assert run.mobile[1, 0, 0] == 1.0
    pickup[1, 0, 0] = 2.0
    result = run.step(pickup, vel, rate * 0.0 + 2.0, settle, 1.0)  # r = 2: both packets decay identically
    w = run.mobile[1, 0, 0] * run.phase.fraction[1, 0, 0]
    x = run.phase.position_m[1, 0, 0]
    s = math.exp(-2.0 * 0.1)
    old_bin = int(bin_index(np.array(0.4), DX, nb, np))
    new_bin = int(bin_index(np.array(0.1), DX, nb, np))
    assert old_bin != new_bin
    np.testing.assert_allclose(w[old_bin], 1.0 * s, rtol=1e-14)
    np.testing.assert_allclose(w[new_bin], 2.0 * s, rtol=1e-14)
    np.testing.assert_allclose(x[old_bin], 0.4, rtol=1e-14)
    np.testing.assert_allclose(x[new_bin], 0.1, rtol=1e-14)
    assert w[[b for b in range(nb) if b not in (old_bin, new_bin)]].sum() == 0.0
    np.testing.assert_allclose(result.transport.deposition_request_kg[1, 0, 0], 3.0 * (1.0 - s), rtol=1e-14)


# --- T06: dry arrival settles through the receiver's request -----------------------------------------
@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_dry_receiver_settles_arrivals_immediately(implementation):
    maybe_skip(implementation)
    graph, net = chain(2)
    vel, rate, settle = fields(graph.shape, 1, v=0.2, rate=1.0)
    settle[0] = True  # outlet cell dry
    run = Runner(net, 8, 1, implementation)
    pickup = zeros(graph.shape, 1)
    pickup[1, 0, 0] = 1.0
    run.step(pickup, vel, rate, settle, 1.0)  # x = 0.2
    run.step(zeros(graph.shape, 1), vel, rate, settle, 1.0)  # x = 0.4
    result = run.step(zeros(graph.shape, 1), vel, rate, settle, 1.0)  # crosses after 0.1 m
    s = math.exp(-1.0 * DX)
    tr = result.transport
    np.testing.assert_allclose(tr.settled_kg[0, 0, 0], s, rtol=1e-13)
    np.testing.assert_allclose(result.arrival_deposition_kg[0, 0, 0], s, rtol=1e-13)
    np.testing.assert_allclose(tr.internal_transfer_in_kg[0, 0, 0], s, rtol=1e-13)
    assert float(tr.export_request_kg.sum()) == 0.0 and float(result.mobile_remaining_kg.sum()) == 0.0
    np.testing.assert_allclose(run.dep.sum(), 1.0, rtol=1e-14)


# --- T07: converging graph, pure-orientation planes, boundary signs and face identities --------------
def _identity_checks(result: CharacteristicStep, network, before, dx):
    tr = result.transport
    ny, nx = network.shape
    assert tr.x_face_gross_kg.shape == (ny, nx + 1, before.shape[-1])
    assert tr.y_face_gross_kg.shape == (ny + 1, nx, before.shape[-1])
    np.testing.assert_allclose(tr.mobile_after_transfer_kg - before, tr.divergence_kg, rtol=0,
                               atol=1e-12 * max(1.0, float(before.max())))
    assert np.all(tr.export_request_kg[~network.outlet_host] == 0.0)
    assert np.all(tr.deposition_request_kg + tr.export_request_kg <= tr.mobile_after_transfer_kg * (1 + 4 * EPS))
    gross = tr.x_face_gross_kg.sum(axis=(0, 1)) + tr.y_face_gross_kg.sum(axis=(0, 1))
    np.testing.assert_allclose(gross, (tr.internal_transfer_out_kg + tr.export_request_kg).sum(axis=(0, 1)),
                               rtol=1e-12, atol=1e-15)
    assert np.all(np.abs(tr.budget_residual_by_class_kg) <= tr.budget_tolerance_by_class_kg)
    np.testing.assert_allclose(tr.mobile_before_by_class_kg, before.sum(axis=(0, 1)), rtol=1e-13)
    ph = result.phase_after
    occ = result.mobile_remaining_kg > 0
    assert np.all(np.abs(ph.fraction.sum(-1)[occ] - 1.0) <= fraction_sum_tolerance(ph.n_bins))
    assert np.all(ph.fraction[~occ][:, 0] == 1.0) and not np.any(ph.fraction[~occ][:, 1:])
    assert np.all(ph.position_m >= 0.0) and np.all(ph.position_m < dx)
    bins = np.arange(ph.n_bins)
    assert np.all(bin_index(ph.position_m, dx, ph.n_bins, np)[ph.fraction > 0] == np.broadcast_to(
        bins, ph.fraction.shape)[ph.fraction > 0])


@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_converging_valley_and_pure_planes_keep_identities_and_face_signs(implementation):
    maybe_skip(implementation)
    rng = np.random.default_rng(11)
    nc = 3
    cases = [("valley", make_graph(valley_full(6, 5))), ("south", make_graph(chain_full(6)))]
    z, export = west_plane_full(5, 6)
    cases.append(("west", make_graph(z, export=export)))
    for name, graph in cases:
        net = transport_network(graph)
        ny, nx = graph.shape
        mobile = rng.uniform(0.0, 2.0, (ny, nx, nc))
        pickup = rng.uniform(0.0, 1.0, (ny, nx, nc))
        v = rng.uniform(0.0, 0.2, (ny, nx, nc))
        rate = rng.uniform(0.0, 4.0, (ny, nx, nc))
        settle = np.zeros((ny, nx, nc), dtype=bool)
        settle[ny // 2, nx // 2, 0] = True
        run = Runner(net, 8, nc, implementation)
        # a first step makes the phase non-trivial; later steps cross faces
        # (v dt <= 0.2 per step, so crossings begin from the third step)
        run.step(mobile, v, rate, settle, 1.0)
        crossed = zeros(graph.shape, nc)
        arrived = zeros(graph.shape, nc)
        for _ in range(5):
            before = run.mobile + pickup
            result = run.step(pickup, v, rate, settle, 1.0)
            _identity_checks(result, net, before, DX)
            tr = result.transport
            crossed += tr.internal_transfer_out_kg + tr.export_request_kg
            arrived += tr.internal_transfer_in_kg
            if name == "south":
                assert not np.any(tr.x_face_gross_kg) and np.all(tr.y_face_net_kg <= 0.0)  # +y north; all south
                np.testing.assert_array_equal(tr.y_face_gross_kg[0], tr.export_request_kg[0])
            if name == "west":
                assert not np.any(tr.y_face_gross_kg) and np.all(tr.x_face_net_kg <= 0.0)  # +x east; all west
                np.testing.assert_array_equal(tr.x_face_gross_kg[:, 0], tr.export_request_kg[:, 0])
        assert float(crossed.sum()) > 0.0 and float(run.export.sum()) > 0.0
        if name == "valley":
            assert float(arrived[:, nx // 2].sum()) > 0.0  # side cells drain into the centre column
            assert float(arrived[:, 0].sum()) == 0.0 and float(arrived[:, nx - 1].sum()) == 0.0


# --- T08: conservation on a random network, several classes and substeps ----------------------------
@pytest.mark.parametrize("n_substeps", [1, 3])
@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_random_network_conserves_per_class_and_per_cell(implementation, n_substeps):
    maybe_skip(implementation)
    rng = np.random.default_rng(5)
    graph = make_graph(random_full(rng, 7, 6))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 4
    run = Runner(net, 8, nc, implementation)
    total_pickup = 0.0
    for _ in range(12):
        pickup = rng.uniform(0.0, 1.0, (ny, nx, nc))
        v = rng.uniform(0.0, 0.25 * n_substeps, (ny, nx, nc))  # a = v dt / (n_sub dx) <= 0.5
        rate = rng.uniform(0.0, 20.0, (ny, nx, nc)) * (rng.uniform(size=(ny, nx, nc)) > 0.2)
        settle = rng.uniform(size=(ny, nx, nc)) < 0.1
        before = run.mobile + pickup
        total_pickup += float(pickup.sum())
        result = run.step(pickup, v, rate, settle, 1.0, n_substeps=n_substeps)
        _identity_checks(result, net, before, DX)
        tr = result.transport
        assert float(tr.max_courant) <= DEFAULT_COURANT_MAX
        assert tr.n_substeps == n_substeps
    # global closure: every kg injected is deposited, exported or still mobile
    np.testing.assert_allclose(run.dep.sum() + run.export.sum() + run.mobile.sum(), total_pickup, rtol=1e-12)


# --- T09: immutability and refusals -------------------------------------------------------------------
def test_inputs_are_not_modified_and_outputs_are_frozen():
    rng = np.random.default_rng(2)
    graph = make_graph(valley_full(4, 3))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 2
    phase = empty_phase_state(graph.shape, nc, 8, DX)
    first = characteristic_step(net, zeros(graph.shape, nc), phase, rng.uniform(0, 1, (ny, nx, nc)),
                                rng.uniform(0, 0.2, (ny, nx, nc)), rng.uniform(0, 3, (ny, nx, nc)),
                                np.zeros((ny, nx, nc), bool), 1.0)
    inputs = {"mobile": first.mobile_remaining_kg.copy(), "pickup": rng.uniform(0, 1, (ny, nx, nc)), "v": rng.uniform(0, 0.2, (ny, nx, nc)), "rate": rng.uniform(0, 3, (ny, nx, nc)), "settle": rng.uniform(size=(ny, nx, nc)) < 0.2}
    copies = {k: v.copy() for k, v in inputs.items()}
    fraction, position = first.phase_after.fraction.copy(), first.phase_after.position_m.copy()
    result = characteristic_step(net, inputs["mobile"], first.phase_after, inputs["pickup"], inputs["v"],
                                 inputs["rate"], inputs["settle"], 1.0)
    for k, value in inputs.items():
        np.testing.assert_array_equal(value, copies[k])
    np.testing.assert_array_equal(first.phase_after.fraction, fraction)
    np.testing.assert_array_equal(first.phase_after.position_m, position)
    assert not result.phase_after.fraction.flags.writeable and not result.phase_after.position_m.flags.writeable
    assert not phase.fraction.flags.writeable


def test_refusals_leave_nothing_behind():
    graph = make_graph(valley_full(4, 3))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 2
    good = {"mobile": np.full((ny, nx, nc), 0.5), "pickup": np.full((ny, nx, nc), 0.1), "v": np.full((ny, nx, nc), 0.1), "rate": np.full((ny, nx, nc), 2.0), "settle": np.zeros((ny, nx, nc), bool)}
    phase = empty_phase_state(graph.shape, nc, 8, DX)
    # a valid non-trivial phase to corrupt
    ok = characteristic_step(net, good["mobile"] * 0.0, phase, good["mobile"], good["v"], good["rate"],
                             good["settle"], 1.0)
    base = {"mobile": ok.mobile_remaining_kg, "phase": ok.phase_after, **{k: good[k] for k in ("pickup", "v", "rate", "settle")}}

    def call(**changes):
        args = dict(base)
        args.update(changes)
        return characteristic_step(net, args["mobile"], args["phase"], args["pickup"], args["v"], args["rate"],
                                   args["settle"], args.get("dt", 1.0), n_substeps=args.get("n_substeps", 1),
                                   courant_max=args.get("courant_max", DEFAULT_COURANT_MAX),
                                   implementation=args.get("implementation", "array"))

    call()  # baseline accepted

    def corrupted(field, mutate):
        p = base["phase"]
        f, x = p.fraction.copy(), p.position_m.copy()
        mutate(f, x)
        return dataclasses.replace(p, fraction=f, position_m=x)

    occupied = np.argwhere(base["mobile"] > 0)[0]
    r, c, k = occupied
    b = int(np.argmax(base["phase"].fraction[r, c, k]))
    empty_b = int(np.argmin(base["phase"].fraction[r, c, k]))
    assert base["phase"].fraction[r, c, k, empty_b] == 0.0
    bad_cases = {
        "nonfinite mobile": {"mobile": np.where(np.arange(nc) == 0, np.nan, base["mobile"])},
        "negative pickup": {"pickup": -base["pickup"]},
        "negative velocity": {"v": -base["v"]},
        "nonfinite rate": {"rate": base["rate"] * np.inf},
        "settle dtype": {"settle": base["settle"].astype(np.float64)},
        "shape": {"pickup": base["pickup"][:, :, :1]},
        "not an array": {"pickup": list(base["pickup"])},
        "fractions do not sum": {"phase": corrupted("f", lambda f, x: f.__setitem__((r, c, k, b), f[r, c, k, b] * 0.5))},
        "negative fraction": {"phase": corrupted("f", lambda f, x: (f.__setitem__((r, c, k, b), f[r, c, k, b] + 0.5),
                                                                       f.__setitem__((r, c, k, (b + 1) % 8), -0.5)))},
        "position beyond dx": {"phase": corrupted("x", lambda f, x: x.__setitem__((r, c, k, b), DX))},
        "negative position": {"phase": corrupted("x", lambda f, x: x.__setitem__((r, c, k, b), -1e-3))},
        "position outside its bin": {"phase": corrupted("x", lambda f, x: x.__setitem__(
            (r, c, k, b), (b + 1.5) * DX / 8 if b < 7 else (b - 0.5) * DX / 8))},
        "position on an empty bin": {"phase": corrupted("x", lambda f, x: x.__setitem__(
            (r, c, k, empty_b), empty_b * DX / 8 + 1e-3))},
        "non-canonical empty cell": {"mobile": base["mobile"] * 0.0},
        "wrong dx": {"phase": empty_phase_state(graph.shape, nc, 8, 0.25), "mobile": base["mobile"] * 0.0},
        "dt": {"dt": 0.0},
        "substeps": {"n_substeps": 0},
        "courant_max": {"courant_max": 0.75},
        "implementation": {"implementation": "loop"},
    }
    for changes in bad_cases.values():
        with pytest.raises(TransportError):
            call(**changes)
    with pytest.raises(TransportStepRejected):
        call(v=np.full((ny, nx, nc), 0.3))  # a = 0.6 > 0.5
    call(v=np.full((ny, nx, nc), 0.3), n_substeps=2)  # recoverable with substeps
    # mobile mass, pickup or velocity on an inactive cell is refused, not erased
    active = np.ones((4, 1), dtype=bool)
    active[3, 0] = False
    graph2 = make_graph(chain_full(4), active=active)
    net2 = transport_network(graph2)
    assert net2.n_active == 3
    phase2 = empty_phase_state(graph2.shape, 1, 8, DX)
    empty = np.zeros((4, 1, 1))
    for field in ("mobile", "pickup", "v"):
        bad = {"mobile": empty, "pickup": empty, "v": empty}
        bad[field] = np.where(np.arange(4)[:, None, None] == 3, 1.0, 0.0)
        with pytest.raises(TransportError, match="inactive"):
            characteristic_step(net2, bad["mobile"], phase2, bad["pickup"], bad["v"], empty + 1.0,
                                np.zeros((4, 1, 1), bool), 1.0)
    ok2 = characteristic_step(net2, empty, phase2, empty + np.where(np.arange(4)[:, None, None] == 2, 1.0, 0.0),
                              empty + np.where(np.arange(4)[:, None, None] < 3, 0.1, 0.0), empty + 1.0,
                              np.zeros((4, 1, 1), bool), 1.0)
    assert ok2.mobile_remaining_kg[3, 0, 0] == 0.0 and ok2.transport.mobile_after_transfer_kg[3, 0, 0] == 0.0


def test_validate_phase_state_and_empty_state_are_canonical():
    graph = make_graph(chain_full(3))
    net = transport_network(graph)
    phase = empty_phase_state(graph.shape, 2, 16, DX)
    assert phase.shape == graph.shape and phase.n_classes == 2 and phase.n_bins == 16
    assert np.all(phase.fraction[..., 0] == 1.0) and not np.any(phase.position_m)
    validate_phase_state(phase, zeros(graph.shape, 2), net)
    validate_phase_state(phase, zeros(graph.shape, 2) + 1.0, net)  # a positive pool wholly at x = 0 is valid
    with pytest.raises(TransportError):
        validate_phase_state(empty_phase_state(graph.shape, 2, 16, 0.25), zeros(graph.shape, 2), net)
    # the public validator checks the mobile field itself, not only its shape
    for bad in (zeros(graph.shape, 2) - 1.0, zeros(graph.shape, 2) + np.nan, zeros(graph.shape, 2)[..., :1],
                zeros(graph.shape, 2).astype(np.float32), list(zeros(graph.shape, 2))):
        with pytest.raises(TransportError):
            validate_phase_state(phase, bad, net)
    shifted = dataclasses.replace(phase, position_m=phase.position_m + 0.01)
    with pytest.raises(TransportError):
        validate_phase_state(shifted, zeros(graph.shape, 2) + 1.0, net)  # positions on empty bins
    with pytest.raises(TransportError):
        empty_phase_state(graph.shape, 2, 0, DX)
    with pytest.raises(TransportError):
        empty_phase_state(graph.shape, 2, 129, DX)


# --- T10: internal substeps equal repeated calls ------------------------------------------------------
@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_internal_substeps_match_separate_calls(implementation):
    maybe_skip(implementation)
    rng = np.random.default_rng(9)
    graph = make_graph(valley_full(5, 3))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 2
    v = rng.uniform(0.0, 0.1, (ny, nx, nc))
    rate = rng.uniform(0.0, 5.0, (ny, nx, nc))
    settle = rng.uniform(size=(ny, nx, nc)) < 0.1
    pickup = rng.uniform(0.0, 1.0, (ny, nx, nc))
    phase = empty_phase_state(graph.shape, nc, 8, DX)
    prior = characteristic_step(net, zeros(graph.shape, nc), phase, rng.uniform(0, 1, (ny, nx, nc)), v, rate,
                                settle, 0.7, implementation=implementation)
    one = characteristic_step(net, prior.mobile_remaining_kg, prior.phase_after, pickup, v, rate, settle, 1.0,
                              n_substeps=2, implementation=implementation)
    a = characteristic_step(net, prior.mobile_remaining_kg, prior.phase_after, pickup, v, rate, settle, 0.5,
                            implementation=implementation)
    b = characteristic_step(net, a.mobile_remaining_kg, a.phase_after, zeros(graph.shape, nc), v, rate, settle, 0.5,
                            implementation=implementation)
    np.testing.assert_allclose(one.mobile_remaining_kg, b.mobile_remaining_kg, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(one.transport.deposition_request_kg,
                               a.transport.deposition_request_kg + b.transport.deposition_request_kg, rtol=1e-13,
                               atol=1e-16)
    np.testing.assert_allclose(one.transport.export_request_kg,
                               a.transport.export_request_kg + b.transport.export_request_kg, rtol=1e-13, atol=1e-16)
    w1 = one.mobile_remaining_kg[..., None] * one.phase_after.fraction
    w2 = b.mobile_remaining_kg[..., None] * b.phase_after.fraction
    np.testing.assert_allclose(w1, w2, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(one.phase_after.position_m, b.phase_after.position_m, rtol=0, atol=1e-12)
    assert one.transport.n_substeps == 2


# --- T11: CuPy parity -----------------------------------------------------------------------------------
def test_cupy_backend_matches_numpy_when_a_gpu_is_available():
    from maple.core.backend import cupy_module, gpu_execution_available, to_host

    if not gpu_execution_available():
        pytest.skip("CuPy with a CUDA device is not available; GPU path not exercised, no claim made")
    cp = cupy_module()
    rng = np.random.default_rng(21)
    z_w, e_w = west_plane_full(5, 6)
    for z, export in ((chain_full(6), None), (z_w, e_w), (valley_full(6, 5), None)):
        graph_np = make_graph(z, export=export)
        graph_cp = make_graph(z, export=export, xp=cp)
        net_np, net_cp = transport_network(graph_np), transport_network(graph_cp)
        ny, nx = graph_np.shape
        nc = 3
        v = rng.uniform(0.0, 0.2, (ny, nx, nc))
        rate = rng.uniform(0.0, 6.0, (ny, nx, nc)) * (rng.uniform(size=(ny, nx, nc)) > 0.3)
        settle = rng.uniform(size=(ny, nx, nc)) < 0.1
        p0 = rng.uniform(0.0, 1.0, (ny, nx, nc))
        p1 = rng.uniform(0.0, 1.0, (ny, nx, nc))
        host0 = characteristic_step(net_np, zeros(graph_np.shape, nc), empty_phase_state(graph_np.shape, nc, 8, DX),
                                    p0, v, rate, settle, 1.0)
        host = characteristic_step(net_np, host0.mobile_remaining_kg, host0.phase_after, p1, v, rate, settle, 1.0,
                                   n_substeps=2)
        dev0 = characteristic_step(net_cp, cp.zeros((ny, nx, nc)), empty_phase_state(graph_cp.shape, nc, 8, DX, xp=cp),
                                   cp.asarray(p0), cp.asarray(v), cp.asarray(rate), cp.asarray(settle), 1.0)
        dev = characteristic_step(net_cp, dev0.mobile_remaining_kg, dev0.phase_after, cp.asarray(p1), cp.asarray(v),
                                  cp.asarray(rate), cp.asarray(settle), 1.0, n_substeps=2)
        for name in ("mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg", "decay_deposition_kg",
                     "settled_kg", "internal_transfer_in_kg", "internal_transfer_out_kg", "divergence_kg",
                     "x_face_gross_kg", "y_face_gross_kg", "x_face_net_kg", "y_face_net_kg",
                     "mobile_before_by_class_kg", "mobile_after_by_class_kg", "deposition_request_by_class_kg",
                     "export_request_by_class_kg"):
            np.testing.assert_allclose(to_host(getattr(dev.transport, name)), getattr(host.transport, name),
                                       rtol=1e-10, atol=1e-15, err_msg=name)
        np.testing.assert_allclose(to_host(dev.mobile_remaining_kg), host.mobile_remaining_kg, rtol=1e-10, atol=1e-15)
        wd = to_host(dev.mobile_remaining_kg)[..., None] * to_host(dev.phase_after.fraction)
        wh = host.mobile_remaining_kg[..., None] * host.phase_after.fraction
        np.testing.assert_allclose(wd, wh, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(to_host(dev.phase_after.position_m), host.phase_after.position_m, rtol=0, atol=1e-9)
        # each backend independently satisfies the unchanged MAPLE bound
        for step in (host, dev):
            residual = np.abs(to_host(step.transport.budget_residual_by_class_kg))
            assert np.all(residual <= to_host(step.transport.budget_tolerance_by_class_kg))
    with pytest.raises(TransportError):  # mixed namespaces are refused, never transferred silently
        characteristic_step(net_cp, cp.zeros((ny, nx, nc)), empty_phase_state(graph_cp.shape, nc, 8, DX, xp=cp),
                            p1, cp.asarray(v), cp.asarray(rate), cp.asarray(settle), 1.0)


# --- T12: continuous source, bounded-bin merge versus the unmerged reference ----------------------------
def unmerged_reference(births, delay_steps, survival):
    """Every packet kept separately: exact delay `dx / v`, exact survival."""
    ref = np.zeros(len(births))
    for step, born in enumerate(births):
        if step + delay_steps < len(ref):
            ref[step + delay_steps] = born * survival
    return ref


# Predeclared limits from the Codex probe (same arithmetic, v = 0.01, dx = 0.5,
# L = 0.05, dt = 1): rate relative L2 6.73 / 3.34 / 1.73 / 1.000 / 3e-15,
# centroid offsets 13.1 / 1.6 / 0.50 / 0.50 / 0 s, first export 203 / 59 / 52 /
# 51 / 50 s (reference 50 s). The export TOTAL is exact for every B; the
# rate pulsation is real and is not promised away.
PULSATION_LIMITS = {4: (7.5, 15.0, 160), 8: (3.7, 2.5, 12), 16: (1.9, 1.0, 3), 32: (1.1, 1.0, 2), 64: (1e-9, 1e-6, 0)}


@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_continuous_source_total_exact_and_pulsation_within_declared_limits(implementation):
    maybe_skip(implementation)
    v, L, dt, end = 0.01, 0.05, 1.0, 800
    # A packet injected at x = 0 travels v dt in its injection step, so it crosses
    # in the step in which its accumulated travel reaches dx: dx / (v dt) steps
    # counting the injection step, i.e. a delay of dx / (v dt) - 1 later steps.
    delay = round(DX / v / dt) - 1
    survival = math.exp(-DX / L)
    graph, net = chain(2)  # a one-cell chain is not a valid legacy graph (edge rule); inject at the outlet cell
    vel, rate, settle = fields(graph.shape, 1, v=v, rate=1.0 / L)
    births = np.array([dt * math.exp(-((t * dt - 250.0) / 100.0) ** 2) if t * dt < 600.0 else 0.0
                       for t in range(end)])
    ref = unmerged_reference(births, delay, survival)
    ref_centroid = float(np.sum((np.arange(end) + 1) * dt * ref) / ref.sum())
    ref_first = float(np.flatnonzero(ref)[0] + 1)
    previous_l2 = np.inf
    for n_bins, (l2_limit, centroid_limit, first_limit) in PULSATION_LIMITS.items():
        run = Runner(net, n_bins, 1, implementation)
        rates = np.zeros(end)
        for t in range(end):
            pickup = zeros(graph.shape, 1)
            pickup[0, 0, 0] = births[t]
            result = run.step(pickup, vel, rate, settle, dt)
            rates[t] = float(result.transport.export_request_kg.sum())
        np.testing.assert_allclose(run.export.sum(), births.sum() * survival, rtol=1e-12)
        assert abs(run.total() - births.sum()) <= 1e-12 * births.sum()
        l2 = float(np.linalg.norm(rates - ref) / np.linalg.norm(ref))
        centroid = float(np.sum((np.arange(end) + 1) * dt * rates) / rates.sum())
        first = float(np.flatnonzero(rates)[0] + 1)
        assert l2 <= l2_limit, (n_bins, l2)
        assert abs(centroid - ref_centroid) <= centroid_limit, (n_bins, centroid, ref_centroid)
        assert abs(first - ref_first) <= first_limit, (n_bins, first, ref_first)
        assert l2 <= previous_l2 * (1 + 1e-9), "pulsation must not grow with bin refinement"
        previous_l2 = l2


# --- T12b: logarithmic-mean merge under extreme dynamic range (Codex reproducer) ------------------------
@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_log_mean_merge_survives_extreme_mass_and_position_spread(implementation):
    """A 1e-20 kg packet at x = 0.009 (near the face) merged with 1 kg at
    x = 0 in a 0.01 m cell with L = 1e-4: the representative must be the
    exact logarithmic mean 0.0043948298140119085 m (a subtractive expm1
    sum shifted by the light constituent's position rounds to -M and gave
    0.005395634661088284), and the merged bin must later transmit exactly
    the eventual survival sum m_i exp(-(dx - x_i)/L) = 4.54e-25 kg."""
    maybe_skip(implementation)
    dx, L = 0.01, 1.0e-4
    graph = make_graph(chain_full(2), dx=dx)
    net = transport_network(graph)
    old_mass, old_x, new_mass = 1.0e-20, 0.009, 1.0
    mobile = zeros(graph.shape, 1)
    mobile[1, 0, 0] = old_mass
    fraction = np.zeros((*graph.shape, 1, 1))
    fraction[..., 0] = 1.0
    position = np.zeros((*graph.shape, 1, 1))
    position[1, 0, 0, 0] = old_x
    phase = dataclasses.replace(empty_phase_state(graph.shape, 1, 1, dx), fraction=fraction, position_m=position)
    validate_phase_state(phase, mobile, net)
    pickup = zeros(graph.shape, 1)
    pickup[1, 0, 0] = new_mass
    vel, rate, settle = fields(graph.shape, 1, v=0.0, rate=1.0 / L)
    merged = characteristic_step(net, mobile, phase, pickup, vel, rate, settle, 1.0, implementation=implementation)
    exact = L * math.log((old_mass * math.exp(old_x / L) + new_mass) / (old_mass + new_mass))
    np.testing.assert_allclose(exact, 0.0043948298140119085, rtol=1e-15)
    np.testing.assert_allclose(merged.phase_after.position_m[1, 0, 0, 0], exact, rtol=1e-12)
    assert float(merged.transport.deposition_request_kg.sum()) == 0.0
    # eventual survival across the cell's face under the same constant law
    eventual = old_mass * math.exp(-(dx - old_x) / L) + new_mass * math.exp(-dx / L)
    run = Runner(net, 1, 1, implementation)
    run.mobile, run.phase = merged.mobile_remaining_kg, merged.phase_after
    vel, rate, settle = fields(graph.shape, 1, v=0.005, rate=1.0 / L)  # Courant 0.5
    for _ in range(4):
        run.step(zeros(graph.shape, 1), vel, rate, settle, 1.0)
    np.testing.assert_allclose(run.crossing[1, 0, 0], eventual, rtol=1e-9)
    assert run.mobile[1, 0, 0] == 0.0
    # small-r regime keeps its accuracy: the narrow branch is a mass-weighted mean to first order
    small = characteristic_step(net, mobile * 0.0 + 1.0, dataclasses.replace(phase, position_m=position * 0.5),
                                pickup, vel * 0.0, rate * 0.0 + 1e-6, settle, 1.0, implementation=implementation)
    expected_mean = (1.0 * 0.0045 + 1.0 * 0.0) / 2.0
    np.testing.assert_allclose(small.phase_after.position_m[1, 0, 0, 0], expected_mean, rtol=0, atol=1e-11)


# --- T13: compiled host kernel against the array path -------------------------------------------------
def test_numba_kernel_matches_array_path():
    pytest.importorskip("numba")
    rng = np.random.default_rng(31)
    graph = make_graph(random_full(rng, 6, 5))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 3
    runs = {impl: Runner(net, 16, nc, impl) for impl in ("array", "numba")}
    for _ in range(8):
        pickup = rng.uniform(0.0, 1.0, (ny, nx, nc))
        v = rng.uniform(0.0, 0.2, (ny, nx, nc))
        rate = rng.uniform(0.0, 8.0, (ny, nx, nc)) * (rng.uniform(size=(ny, nx, nc)) > 0.25)
        settle = rng.uniform(size=(ny, nx, nc)) < 0.1
        results = {impl: run.step(pickup, v, rate, settle, 1.0, n_substeps=2) for impl, run in runs.items()}
        a, b = results["array"], results["numba"]
        for name in ("mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg", "settled_kg",
                     "internal_transfer_in_kg", "internal_transfer_out_kg", "y_face_net_kg", "x_face_net_kg"):
            np.testing.assert_allclose(getattr(b.transport, name), getattr(a.transport, name), rtol=1e-12, atol=1e-16,
                                       err_msg=name)
        np.testing.assert_allclose(b.mobile_remaining_kg[..., None] * b.phase_after.fraction,
                                   a.mobile_remaining_kg[..., None] * a.phase_after.fraction, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(b.phase_after.position_m, a.phase_after.position_m, rtol=0, atol=1e-12)


def test_numba_unavailable_is_an_explicit_error(monkeypatch):
    import builtins

    from maple_syrup import characteristic_numba

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numba":
            raise ImportError("no numba")
        return real_import(name, *args, **kwargs)

    characteristic_numba.reset_compiled()
    monkeypatch.setattr(builtins, "__import__", fake_import)
    graph, net = chain(2)
    with pytest.raises(characteristic_numba.NumbaUnavailableError):
        characteristic_step(net, zeros(graph.shape, 1), empty_phase_state(graph.shape, 1, 8, DX),
                            zeros(graph.shape, 1), zeros(graph.shape, 1), zeros(graph.shape, 1),
                            np.zeros((*graph.shape, 1), bool), 1.0, implementation="numba")
    monkeypatch.undo()
    characteristic_numba.reset_compiled()
