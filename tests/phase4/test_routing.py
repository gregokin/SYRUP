"""Phase 4b routing graph and method-5 step.

Independent references used here:
- a scalar transcription of topog_attrib.for (aspect 94-117, masking
  190-197, cap 218-224, edge rule 265-293) on the legacy north-first grid;
- a sequential pure-Python sweep with `scipy.optimize.brentq` roots;
- analytic single-cell recession `h(t) = (h0^-1/2 + k t / (2 dx))^-2`;
- the discrete uniform-plane steady state `q_i = e dx (i + 1)`.
All are equation-level checks, not executions of MAHLERAN (see
test_fortran_routing.py for the executed original routine).
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from maple_syrup import routing_numba
from maple_syrup.routing import (
    DONOR_SLOTS,
    EXPORT,
    GRAVITY_M_S2,
    IMPLEMENTATIONS,
    LEGACY_FRICTION_FLOOR,
    RoutingError,
    RoutingGraphError,
    _bisect,
    _donor_sum,
    build_routing_graph,
    legacy_stale_inflow_step,
    plot1_routing_graph,
    route_step,
)

# tests/phase4/<file>.py: parents[0] = tests/phase4, [1] = tests, [2] = the repository root.
REPO = Path(__file__).resolve().parents[2]
RECIPE_PATH = REPO / "cases" / "plot1" / "recipe.yaml"
MAHLERAN_ROOT = Path(os.environ.get("MAPLE_SYRUP_MAHLERAN_ROOT", "/home/okin/MAHLERAN"))
PLOT1_DIR = MAHLERAN_ROOT / "Input" / "input_p1"
EPS = np.finfo(np.float64).eps

pytest.importorskip("maple")

NUMBA_SKIP = pytest.mark.skipif(not routing_numba.numba_available(),
                                reason="Numba not installed (optional extra maple-syrup[numba]); "
                                       "compiled sweep not exercised, no claim made")


def implementations():
    """Both ordered sweeps; the compiled one skips honestly without Numba."""
    return ["array", pytest.param("numba", marks=NUMBA_SKIP)]


def test_repo_root_resolves_from_tests_phase4():
    assert (REPO / "pyproject.toml").is_file() and (REPO / "src" / "maple_syrup" / "routing.py").is_file()
    assert REPO.name == "SYRUP" or (REPO / "AGENTS.md").is_file()


# --- synthetic terrain (south-first full grids with a ring; exact binary steps) ----------
def chain_full(n: int, dz: float = 0.015625, walls: float = 1.0) -> np.ndarray:
    """n cells in one column draining south; E/W ring walls; south ring lowest."""
    z = np.repeat(np.arange(n + 2, dtype=np.float64)[:, None] * dz, 3, axis=1)
    z[:, 0] += walls
    z[:, 2] += walls
    return z


def valley_full(ny: int, nx: int, sy: float = 0.015625, sx: float = 0.03125) -> np.ndarray:
    """V valley (nx odd): side cells drain to the centre column, which drains south."""
    rows, cols = np.indices((ny + 2, nx + 2)).astype(np.float64)
    return rows * sy + np.abs(cols - (nx + 1) / 2) * sx


def random_full(rng, ny: int, nx: int) -> np.ndarray:
    """South tilt 0.025 m per cell plus noise < 0.02 m: every cell has a strictly
    lower south neighbour (no sinks); E/W walls; random E/W/S receivers."""
    rows = np.indices((ny + 2, nx + 2))[0].astype(np.float64)
    z = 0.025 * rows + rng.uniform(0.0, 0.02, rows.shape)
    z[:, 0] += 10.0
    z[:, -1] += 10.0
    return z


def south_export(z: np.ndarray) -> np.ndarray:
    export = np.zeros(z.shape, dtype=bool)
    export[0, :] = True
    return export


def make_graph(z, *, export=None, ff=1.0, dx=0.5, active=None, xp=None):
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    friction = ff if isinstance(ff, np.ndarray) else np.full((ny, nx), float(ff))
    return build_routing_graph(z, south_export(z) if export is None else export, friction, dx,
                               active_mask=active, xp=xp)


def read_legacy_grid(path: Path) -> np.ndarray:
    """Independent ESRI ASCII read, north-first; ignores bytes after the rows."""
    lines = Path(path).read_bytes().splitlines()
    header = {key.lower(): value for key, value in (line.decode("ascii").split() for line in lines[:6])}
    nrows = int(header["nrows"])
    return np.array([[float(t) for t in line.split()] for line in lines[6:6 + nrows]], dtype=np.float64)


def legacy_topog(z_nf_m: np.ndarray, rmask_nf: np.ndarray, dx_m: float):
    """Scalar transcription of topog_attrib.for on the full north-first grid
    (1-based legacy i = index + 1). Returns aspect, slope and edge-rule flags."""
    zf = z_nf_m * 1000.0
    nr2, nc2 = zf.shape
    sdir = ((-1, 0), (0, 1), (1, 0), (0, -1))
    aspect = np.zeros((nr2, nc2), dtype=np.int64)
    slope = np.zeros((nr2, nc2))
    for i in range(1, nr2 - 1):
        for k in range(1, nc2 - 1):
            zmin, asp = zf[i, k], 0
            for code, (di, dk) in enumerate(sdir, start=1):
                if zf[i + di, k + dk] < zmin:
                    asp, zmin = code, zf[i + di, k + dk]
            aspect[i, k] = asp
            slope[i, k] = (zf[i, k] - zmin) / (dx_m * 1000.0)
    slope[rmask_nf < 0.0] = 0.0
    slope[slope > 1000.0] = 1.0
    applied = np.zeros((nr2, nc2), dtype=bool)
    for i in range(1, nr2 - 1):
        for j in range(1, nc2 - 1):
            a = aspect[i, j]
            if rmask_nf[i, j] < 0.0 or a == 0:
                continue
            di, dj = sdir[a - 1]
            if rmask_nf[i + di, j + dj] < 0.0:
                oi, oj = i - di, j - dj
                if slope[i, j] > slope[oi, oj] or slope[i, j] == 0.0:
                    slope[i, j] = slope[oi, oj]
                    applied[i, j] = True
    return aspect, slope, applied


def sf_interior(full_nf: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(full_nf[1:-1, 1:-1][::-1])


def sequential_oracle(graph, z_full, h_start, h_old, dt):
    """Independent sequential sweep: receivers from graph.receiver, order by
    descending elevation, k recomputed, scipy brentq roots, scatter sums."""
    brentq = pytest.importorskip("scipy.optimize").brentq
    ny, nx = graph.shape
    c = dt / (2.0 * graph.dx_m)
    act = graph.active.reshape(-1)
    recv = graph.receiver.reshape(-1)
    k = np.sqrt(8.0 * 9.81 * graph.slope / np.where(graph.active, graph.friction_factor, 1.0)).reshape(-1)
    k[~act] = 0.0
    hs, ho = h_start.reshape(-1), h_old.reshape(-1)
    q_old = np.where(act, k * ho ** 1.5, 0.0)
    qin_old = np.zeros(ny * nx)
    for cell in np.flatnonzero(act):
        if recv[cell] >= 0:
            qin_old[recv[cell]] += q_old[cell]
    elevation = z_full[1:-1, 1:-1].reshape(-1)
    qin_new, q_new, h_new = np.zeros(ny * nx), np.zeros(ny * nx), hs.copy()
    for cell in sorted(np.flatnonzero(act), key=lambda i: -elevation[i]):
        rhs = hs[cell] + c * (qin_old[cell] + qin_new[cell] - q_old[cell])
        assert rhs >= 0.0
        root = 0.0 if rhs == 0.0 else brentq(lambda h, r=rhs, kc=k[cell]: h + c * kc * h ** 1.5 - r,
                                              0.0, rhs, xtol=1e-18, rtol=1e-15, maxiter=500)
        h_new[cell], q_new[cell] = root, k[cell] * root ** 1.5
        if recv[cell] >= 0:
            qin_new[recv[cell]] += q_new[cell]
    shape = graph.shape
    return h_new.reshape(shape), q_new.reshape(shape), qin_old.reshape(shape), qin_new.reshape(shape)


def random_state(rng, graph, scale=3e-3):
    shape = graph.shape
    h_old = rng.uniform(0.0, scale, shape) * (rng.uniform(size=shape) > 0.2)
    h_start = h_old + rng.uniform(0.0, 1e-4, shape)
    return np.where(graph.active, h_start, 0.0), np.where(graph.active, h_old, 0.0)


# --- graph ------------------------------------------------------------------------------
def test_chain_graph_structure():
    g = make_graph(chain_full(5))
    assert g.shape == (5, 1) and g.n_active == 5 and g.n_levels == 5 and g.max_level_width == 1
    assert np.all(g.aspect == 3)
    assert g.receiver[0, 0] == EXPORT and list(g.receiver[1:, 0]) == [0, 1, 2, 3]
    assert list(np.flatnonzero(g.outlet)) == [0]
    assert list(g.level[:, 0]) == [4, 3, 2, 1, 0]
    assert np.all(g.slope == 0.03125) and not g.edge_rule_applied.any()
    np.testing.assert_array_equal(g.conveyance, np.sqrt(8.0 * GRAVITY_M_S2 * 0.03125 / 1.0))
    assert len(g.input_sha256) == 64


def test_valley_donor_slots_levels_and_order():
    g = make_graph(valley_full(4, 5))
    centre = 2
    assert np.all(g.aspect[:, centre] == 3)
    assert np.all(g.aspect[:, :centre] == 2) and np.all(g.aspect[:, centre + 1:] == 4)
    assert list(zip(*np.nonzero(g.outlet), strict=True)) == [(0, centre)]
    # Donor slots of the bottom centre cell in legacy sdirin order (S, W, N, E).
    order = g.level_order_host
    pos = int(np.flatnonzero(order == 0 * 5 + centre)[0])
    assert list(np.asarray(g.donor_mask)[:, pos]) == [False, True, True, True]
    donors = [order[p] for p, m in zip(np.asarray(g.donor_position)[:, pos], np.asarray(g.donor_mask)[:, pos],
                                        strict=True) if m]
    assert donors == [0 * 5 + 1, 1 * 5 + 2, 0 * 5 + 3]
    assert [(dr, dc) for dr, dc in DONOR_SLOTS] == [(-1, 0), (0, -1), (1, 0), (0, 1)]
    # Levels: independent longest path, topological and contiguous.
    recv = g.receiver.reshape(-1)

    def depth(cell, memo={}):  # noqa: B006 - local memo
        if cell not in memo:
            ups = [d for d in range(recv.size) if recv[d] == cell]
            memo[cell] = 0 if not ups else 1 + max(depth(d) for d in ups)
        return memo[cell]

    levels = np.array([depth(i) for i in range(recv.size)]).reshape(g.shape)
    np.testing.assert_array_equal(g.level, levels)
    assert g.n_levels == levels.max() + 1
    assert np.all(np.diff(g.level.reshape(-1)[order]) >= 0)


def test_donor_sum_is_the_legacy_sequential_sum():
    rng = np.random.default_rng(3)
    z = random_full(rng, 9, 8)
    g = make_graph(z)
    q_lo = rng.uniform(0.0, 1.0, g.n_active) * 10.0 ** rng.integers(-12, 0, g.n_active)
    got = _donor_sum(q_lo, g.donor_position, g.donor_mask, np)
    for p in range(g.n_active):
        expected = 0.0
        for s in range(4):  # route_water.for 537-544: qin = qin + q(donor), slot order
            if g.donor_mask[s, p]:
                expected = expected + q_lo[g.donor_position[s, p]]
        assert got[p] == expected


def test_edge_rule_and_slopes_match_topog_attrib_transcription():
    rng = np.random.default_rng(11)
    z = random_full(rng, 8, 7)
    z[0, 2:6] -= 0.2  # steep drops to part of the south ring: outlet slopes exceed their upslope neighbour's
    g = make_graph(z)
    rmask_nf = np.where(south_export(z), -9999.0, 1.0)[::-1]
    aspect_nf, slope_nf, applied_nf = legacy_topog(z[::-1], rmask_nf, 0.5)
    np.testing.assert_array_equal(g.aspect, sf_interior(aspect_nf))
    np.testing.assert_array_equal(g.slope, sf_interior(slope_nf))
    np.testing.assert_array_equal(g.edge_rule_applied, sf_interior(applied_nf))
    assert g.edge_rule_applied.any(), "fixture should trigger the legacy edge rule"


def _reject(match, **kwargs):
    z = kwargs.pop("z", chain_full(3))
    args = {"export": south_export(z), "friction": np.ones((z.shape[0] - 2, z.shape[1] - 2)), "dx": 0.5}
    args.update(kwargs)
    copies = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    z_copy = z.copy()
    with pytest.raises(RoutingGraphError, match=match):
        build_routing_graph(z, args["export"], args["friction"], args["dx"],
                            dy_m=args.get("dy"), active_mask=args.get("active"), nodata_value=args.get("nodata"))
    np.testing.assert_array_equal(z, z_copy)
    for k, v in copies.items():
        np.testing.assert_array_equal(args[k], v)


def test_graph_rejects_sinks_flats_and_silent_losses():
    flat = np.zeros((5, 5))
    flat[0, :] = -1.0
    _reject("sinks", z=flat)  # interior plateau: no strictly lower neighbour
    pit = valley_full(3, 3)
    pit[2, 2] -= 1.0
    _reject("strict pits", z=pit)
    z = chain_full(3)
    _reject("neither an active cell nor an export-flagged", z=z, export=np.zeros(z.shape, dtype=bool))
    active = np.ones((3, 1), dtype=bool)
    active[0, 0] = False  # interior receiver of cell 1 is inactive and not flagged
    _reject("neither an active cell nor an export-flagged", z=z, active=active)
    export = south_export(z)
    export[1, 1] = True
    _reject("cannot be an export receiver", z=z, export=export)
    _reject("edge rule", z=chain_full(1))  # opposite neighbour is the north ring
    active = np.ones((3, 1), dtype=bool)
    active[1, 0] = False
    export = south_export(z)
    export[2, 1] = True  # inactive cell flagged as export: cell 2 exports, its opposite...
    _reject("edge rule|zero slope", z=z, active=active, export=export)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dx": 0.0}, "dx_m"),
        ({"dx": float("nan")}, "dx_m"),
        ({"dx": True}, "dx_m"),
        ({"dx": 1e160}, "overflows"),  # dx^2 = inf: face volumes could never be finite
        ({"dx": 1e-170}, "overflows"),  # dx^2 = 0
        ({"dy": 0.25}, "non-square"),
        ({"friction": np.zeros((3, 1))}, "friction_factor"),
        ({"friction": np.full((3, 1), 0.5 * LEGACY_FRICTION_FLOOR)}, "916-920"),  # legacy floors ff after the root
        ({"friction": np.ones((3, 2))}, "shape"),
        ({"friction": np.ones((3, 1), dtype=np.float32)}, "dtype"),
        ({"export": np.zeros((5, 3))}, "dtype"),
        ({"active": np.zeros((3, 1), dtype=bool)}, "no active cell"),
        ({"z": np.full((5, 3), np.nan)}, "finite"),
        ({"z": np.zeros((2, 3))}, "ring"),
    ],
)
def test_graph_rejects_invalid_inputs(kwargs, match):
    _reject(match, **kwargs)


def test_graph_rejects_device_or_list_inputs():
    z = chain_full(2)
    with pytest.raises(RoutingGraphError, match="host NumPy"):
        build_routing_graph(z.tolist(), south_export(z), np.ones((2, 1)), 0.5)


def test_graph_rejects_a_finite_nodata_sentinel_when_declared():
    """-9999 is finite and would pass the finiteness check as a very low
    receiver; with the declared nodata value it is rejected instead."""
    z = chain_full(3)
    z[0, 1] = -9999.0  # south ring cell below the outlet: would be a false low receiver
    _reject("nodata", z=z, nodata=-9999.0)
    with pytest.raises(RoutingGraphError, match="nodata_value"):
        build_routing_graph(chain_full(3), south_export(z), np.ones((3, 1)), 0.5, nodata_value="x")
    make_graph(chain_full(3))  # undeclared: only finiteness is enforced (documented restriction)


# --- root solver --------------------------------------------------------------------------
def _solve(rhs, k, c, iterations=40):
    m = rhs.size
    lo = np.zeros(m)
    w, mid, t = (np.empty(m) for _ in range(3))
    below = np.empty(m, dtype=bool)
    _bisect(lo, rhs, k, c, iterations, w, mid, t, below, np)
    return lo


def test_bisection_brackets_and_matches_brentq():
    brentq = pytest.importorskip("scipy.optimize").brentq
    rhs_values = [0.0, 1e-300, 1e-12, 1e-6, 1e-3, 0.05, 1.0, 10.0]
    k_values = [0.0, 1e-3, 1.0, 5.0]
    rhs, k = (np.array(v, dtype=np.float64) for v in zip(*itertools.product(rhs_values, k_values), strict=True))
    for c in (0.5, 2.0):
        lo = _solve(rhs, k, c)
        q = (np.sqrt(lo) * lo) * k
        h_new = rhs - q * c
        assert np.all(lo >= 0.0) and np.all(h_new >= lo)
        for r, kk, low, h in zip(rhs, k, lo, h_new, strict=True):
            if r == 0.0:
                assert low == 0.0 and h == 0.0
                continue
            root = brentq(lambda x, c=c, kk=kk, r=r: x + c * kk * x ** 1.5 - r,
                          0.0, r, xtol=1e-300, rtol=1e-15, maxiter=1000)
            assert root - low <= r * 2.0 ** -40 + 8 * EPS * root
            assert low <= root * (1 + 8 * EPS)
            assert h - low <= r * 2.0 ** -40 * (1.0 + 1.5 * c * kk * np.sqrt(r)) + 8 * EPS * r


# --- step: references ------------------------------------------------------------------------
@pytest.mark.parametrize("impl", implementations())
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_route_step_matches_sequential_brentq_oracle(seed, impl):
    rng = np.random.default_rng(seed)
    z = random_full(rng, 9, 7)
    g = make_graph(z, ff=rng.uniform(5.0, 30.0, (9, 7)))
    h_start, h_old = random_state(rng, g)
    step = route_step(g, h_start, h_old, 1.0, implementation=impl)
    assert step.implementation == impl
    h_ref, q_ref, qin_old_ref, qin_new_ref = sequential_oracle(g, z, h_start, h_old, 1.0)
    np.testing.assert_allclose(step.depth_m, h_ref, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(step.flow_depth_m, h_ref, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(step.discharge_m2_s, q_ref, rtol=1e-9, atol=1e-16)
    np.testing.assert_allclose(step.old_inflow_m2_s, qin_old_ref, rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(step.inflow_m2_s, qin_new_ref, rtol=1e-9, atol=1e-16)
    assert abs(float(step.budget_residual_m3)) <= 1e-15
    assert float(step.max_constitutive_residual_m) <= 1e-11
    assert step.conservative and step.stale_inflow_gain_m3 is None


@pytest.mark.parametrize("impl", implementations())
def test_same_face_cancellation_and_export_on_branching_valley(impl):
    g = make_graph(valley_full(6, 5), ff=5.0)
    rng = np.random.default_rng(5)
    h_start, h_old = random_state(rng, g)
    s = route_step(g, h_start, h_old, 1.0, implementation=impl)
    area, c = 0.25, 1.0 / (2 * 0.5)
    face = s.face_volume_m3.reshape(-1)
    recv = g.receiver.reshape(-1)
    received = np.zeros(recv.size)
    for cell in range(recv.size):
        if recv[cell] >= 0:
            received[recv[cell]] += face[cell]
    inflow_volume = area * c * (s.old_inflow_m2_s + s.inflow_m2_s).reshape(-1)
    np.testing.assert_allclose(inflow_volume, received, rtol=1e-14, atol=1e-20)
    np.testing.assert_allclose((s.depth_m - h_start) * area,
                               (received - face).reshape(g.shape), rtol=1e-12, atol=1e-18)
    assert float(s.export_m3) == pytest.approx(face[recv == EXPORT].sum(), rel=1e-15)
    assert float(s.outlet_discharge_m3_s) == pytest.approx(0.5 * s.discharge_m2_s[g.outlet].sum(), rel=1e-15)
    storage = area * float((s.depth_m - h_start).sum())
    assert abs(storage + float(s.export_m3)) <= 1e-16
    assert float(s.storage_change_m3) == pytest.approx(storage, abs=1e-18)


def _recession(dt, impl="array", t_end=64.0, h0=0.004):
    g = make_graph(chain_full(2), ff=10.0)
    h = np.array([[h0], [0.0]])
    for _ in range(round(t_end / dt)):
        h = route_step(g, h, h, dt, implementation=impl).depth_m
    k = float(g.conveyance[0])
    exact = (h0 ** -0.5 + k * t_end / (2 * 0.5)) ** -2
    return float(h[0, 0]), exact


@pytest.mark.parametrize("impl", implementations())
def test_single_cell_recession_second_order_in_time(impl):
    errors = []
    for dt in (8.0, 4.0, 2.0, 1.0):
        h, exact = _recession(dt, impl)
        errors.append(abs(h - exact))
    assert errors[-1] < 1e-2 * exact
    for a, b in itertools.pairwise(errors):
        assert a / b > 3.0  # ~4 for the Crank-Nicolson (trapezoidal) update


def _plane(n=5, e=1e-4, ff=1.0):
    g = make_graph(chain_full(n), ff=ff)
    k = float(g.conveyance[0])
    distance = np.arange(n, 0, -1, dtype=np.float64)[:, None]  # cells from the top, inclusive
    q = e * 0.5 * distance
    return g, e, q, (q / k) ** (2.0 / 3.0)


@pytest.mark.parametrize("impl", implementations())
def test_discrete_uniform_plane_steady_state_is_a_fixed_point(impl):
    g, e, q, h = _plane()
    s = route_step(g, h + e * 1.0, h, 1.0, implementation=impl)
    np.testing.assert_allclose(s.depth_m, h, rtol=1e-11)
    np.testing.assert_allclose(s.discharge_m2_s, q, rtol=1e-10)
    assert float(s.outlet_discharge_m3_s) == pytest.approx(e * 0.5 * 5 * 0.5, rel=1e-10)


@pytest.mark.parametrize("impl", implementations())
def test_uniform_plane_converges_to_steady_state_from_dry(impl):
    g, e, q, h_steady = _plane()
    h = np.zeros(g.shape)
    for _ in range(400):
        s = route_step(g, h + e * 1.0, h, 1.0, implementation=impl)
        h = s.depth_m
    np.testing.assert_allclose(s.discharge_m2_s, q, rtol=1e-6)
    np.testing.assert_allclose(h, h_steady, rtol=1e-6)


# --- step: behaviour and refusals ----------------------------------------------------------
@pytest.mark.parametrize("impl", implementations())
def test_dry_domain_is_exactly_dry_and_wetting_happens_in_the_same_sweep(impl):
    g = make_graph(valley_full(4, 5))
    zero = np.zeros(g.shape)
    s = route_step(g, zero, zero, 1.0, implementation=impl)
    for name in ("depth_m", "discharge_m2_s", "velocity_m_s", "face_volume_m3", "inflow_m2_s"):
        assert not np.any(getattr(s, name))
    assert float(s.export_m3) == 0.0
    h = zero.copy()
    h[3, 0] = 2e-3  # top west corner only
    s = route_step(g, h, h, 1.0, implementation=impl)
    assert s.depth_m[3, 1] > 0.0 and s.depth_m[3, 2] > 0.0  # downstream cells wetted this step
    assert s.depth_m[0, 2] > 0.0  # ... down to the outlet cell in the same sweep


@pytest.mark.parametrize("impl", implementations())
def test_inactive_cells_keep_their_inventory_exactly(impl):
    z = valley_full(4, 5)
    active = np.ones((4, 5), dtype=bool)
    active[3, 0] = False  # top west corner: nothing drains into it
    g = make_graph(z, active=active)
    h = np.full(g.shape, 1e-3)
    h[3, 0] = 0.3
    s = route_step(g, h, np.where(active, h, 0.0), 1.0, implementation=impl)
    assert s.depth_m[3, 0] == 0.3
    assert s.discharge_m2_s[3, 0] == 0.0 and s.face_volume_m3[3, 0] == 0.0
    q_bad = np.where(active, 0.0, 1e-6)
    with pytest.raises(RoutingError, match="inactive"):
        route_step(g, np.zeros(g.shape), np.zeros(g.shape), 1.0, old_discharge_m2_s=q_bad, implementation=impl)


def test_previous_step_discharge_is_accepted_as_old_flux():
    g = make_graph(valley_full(4, 5), ff=5.0)
    h0 = np.full(g.shape, 2e-3)
    a = route_step(g, h0, h0, 1.0)
    b = route_step(g, a.depth_m, a.depth_m, 1.0, old_discharge_m2_s=a.discharge_m2_s)
    c = route_step(g, a.depth_m, a.depth_m, 1.0)
    np.testing.assert_allclose(b.depth_m, c.depth_m, rtol=0, atol=1e-13)


@pytest.mark.parametrize("impl", implementations())
def test_courant_rejection_leaves_inputs_unchanged(impl):
    g = make_graph(chain_full(3), ff=1.0)
    h = np.full(g.shape, 0.05)
    before = h.copy()
    with pytest.raises(RoutingError, match="Courant"):
        route_step(g, h, h, 16.0, implementation=impl)
    np.testing.assert_array_equal(h, before)
    with pytest.raises(RoutingError, match="courant_max"):
        route_step(g, h, h, 1.0, courant_max=2.5, implementation=impl)


@pytest.mark.parametrize("impl", implementations())
def test_nonconvergence_is_reported_not_hidden(impl):
    g = make_graph(chain_full(3))
    h = np.full(g.shape, 2e-3)
    with pytest.raises(RoutingError, match="bisection did not reach"):
        route_step(g, h, h, 1.0, bisection_iterations=5, implementation=impl)


@pytest.mark.parametrize("impl", implementations())
def test_overflowing_states_are_rejected_not_returned_as_nan(impl):
    """Finite-output holes (Codex review): every grid and scalar output is
    flag-checked for finiteness before any tolerance comparison, so an
    overflow raises instead of a NaN residual passing `> tol` as False."""
    g = make_graph(chain_full(3), ff=1.0)
    huge = np.full(g.shape, 1e300)  # k h^{3/2} overflows: q_old = inf
    before = huge.copy()
    with pytest.raises(RoutingError, match="overflowed"):
        route_step(g, huge, huge, 1.0, implementation=impl)
    np.testing.assert_array_equal(huge, before)
    # Finite inputs but overflowing balance scale: the storage is 1e308
    # with dry old flow. Reject the non-finite diagnostic, not return NaN.
    zero = np.zeros(g.shape)
    with pytest.raises(RoutingError, match="non-finite balance scale"):
        route_step(g, np.full(g.shape, 1e308), zero, 1.0, implementation=impl)
    stale = np.array([[1e308], [0.0], [0.0]])
    with pytest.raises(RoutingError, match="right-hand side overflowed"):
        legacy_stale_inflow_step(g, np.full(g.shape, 1e308), zero, 1.0, stale_old_inflow_m2_s=stale,
                                 implementation=impl)
    with pytest.raises(RoutingError, match="finite"):
        legacy_stale_inflow_step(g, zero, zero, 1.0, stale_old_inflow_m2_s=np.full(g.shape, np.inf),
                                 implementation=impl)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"dt": 0.0}, "dt_s"),
        ({"dt": -1.0}, "dt_s"),
        ({"dt": float("nan")}, "dt_s"),
        ({"iterations": 0}, "bisection_iterations"),
        ({"iterations": True}, "bisection_iterations"),
        ({"root_tol": 0.0}, "root_tolerance_m"),
        ({"start": np.zeros((2, 1))}, "shape"),
        ({"start": np.zeros((3, 1), dtype=np.float32)}, "float64"),
        ({"start": [[0.0], [0.0], [0.0]]}, "array"),
        ({"start": np.full((3, 1), -1e-3)}, ">= 0"),
        ({"start": np.full((3, 1), np.nan)}, "finite"),
        ({"old": np.full((3, 1), 5e-3)}, "exceeds depth_start_m"),
        ({"q_old": np.full((3, 1), 1.0)}, "does not satisfy"),
        ({"impl": "fortran"}, "implementation"),
        ({"impl": None}, "implementation"),
    ],
)
@pytest.mark.parametrize("impl", implementations())
def test_route_step_refusals_do_not_mutate(change, match, impl):
    g = make_graph(chain_full(3))
    args = {"start": np.full((3, 1), 2e-3), "old": np.full((3, 1), 1e-3), "dt": 1.0, "iterations": 40,
            "root_tol": 1e-11, "q_old": None, "impl": impl}
    args.update(change)
    copies = {k: np.array(v, copy=True) for k, v in args.items() if isinstance(v, np.ndarray)}
    with pytest.raises(RoutingError, match=match):
        route_step(g, args["start"], args["old"], args["dt"], old_discharge_m2_s=args["q_old"],
                   bisection_iterations=args["iterations"], root_tolerance_m=args["root_tol"],
                   implementation=args["impl"])
    for k, v in copies.items():
        np.testing.assert_array_equal(args[k], v)
    with pytest.raises(RoutingError, match="RoutingGraph"):
        route_step(object(), np.zeros((3, 1)), np.zeros((3, 1)), 1.0, implementation=impl)


def test_missing_numba_is_an_explicit_error_without_fallback(monkeypatch):
    """Whatever is installed: with `numba` unimportable and no compiled
    dispatcher cached, the compiled selector fails loudly before anything is
    computed or mutated, and never falls back to the array sweep."""
    g = make_graph(chain_full(3))
    h = np.full(g.shape, 2e-3)
    before = h.copy()
    monkeypatch.setattr(routing_numba, "_SWEEP", None)
    monkeypatch.setitem(sys.modules, "numba", None)  # `import numba` -> ImportError
    assert not routing_numba.numba_available() and routing_numba.numba_versions() is None
    with pytest.raises(routing_numba.NumbaUnavailableError, match="not installed") as info:
        route_step(g, h, h, 1.0, implementation="numba")
    assert isinstance(info.value, RoutingError)
    np.testing.assert_array_equal(h, before)
    assert IMPLEMENTATIONS == ("array", "numba")
    route_step(g, h, h, 1.0)  # the default array path is unaffected


@NUMBA_SKIP
def test_numba_sweep_matches_array_sweep_bitwise():
    """Same operation sequence, no fastmath: the compiled sweep must reproduce
    the array sweep exactly. (An ulp-level difference would indicate
    floating-point contraction and must be reported, not tolerated.)"""
    rng = np.random.default_rng(29)
    for z, ff in ((random_full(rng, 11, 8), rng.uniform(5.0, 30.0, (11, 8))), (valley_full(6, 5), 5.0)):
        g = make_graph(z, ff=ff)
        h_start, h_old = random_state(rng, g)
        a = route_step(g, h_start, h_old, 1.0)
        b = route_step(g, h_start, h_old, 1.0, implementation="numba")
        for name in ("depth_m", "flow_depth_m", "discharge_m2_s", "velocity_m_s", "inflow_m2_s",
                     "old_inflow_m2_s", "face_volume_m3"):
            np.testing.assert_array_equal(getattr(b, name), getattr(a, name), err_msg=name)
        for name in ("export_m3", "budget_residual_m3", "outlet_discharge_m3_s", "max_constitutive_residual_m"):
            assert float(getattr(b, name)) == float(getattr(a, name)), name
        assert (a.implementation, b.implementation) == ("array", "numba")


@NUMBA_SKIP
def test_numba_cold_then_warm_calls_reuse_one_compilation():
    routing_numba.reset_compiled()
    g = make_graph(valley_full(5, 5), ff=8.0)
    h = np.full(g.shape, 1.5e-3)
    cold = route_step(g, h, h, 1.0, implementation="numba")  # compiles here
    dispatcher = routing_numba.compiled_sweep()
    assert len(dispatcher.signatures) == 1
    warm = route_step(g, cold.depth_m, cold.depth_m, 0.5, implementation="numba")
    other = route_step(make_graph(chain_full(4)), np.full((4, 1), 1e-3), np.full((4, 1), 1e-3), 2.0,
                       implementation="numba")
    assert len(dispatcher.signatures) == 1, "a second compilation would mean the argument types changed"
    reference = route_step(g, cold.depth_m, cold.depth_m, 0.5)
    np.testing.assert_array_equal(warm.depth_m, reference.depth_m)
    assert abs(float(other.budget_residual_m3)) < 1e-18
    versions = routing_numba.numba_versions()
    assert versions is not None and set(versions) == {"numba", "llvmlite"}
    assert dispatcher.targetoptions.get("fastmath", False) is False


def test_numba_rejects_device_graphs_without_transfer():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; device-rejection path not exercised")
    cp = backend.cupy_module()
    g = make_graph(chain_full(3), xp=cp)
    h = backend.to_device(np.full(g.shape, 1e-3), cp)
    with pytest.raises(RoutingError, match="host NumPy"):
        route_step(g, h, h, 1.0, implementation="numba")


def test_outputs_do_not_alias_inputs():
    g = make_graph(chain_full(3))
    h = np.full(g.shape, 2e-3)
    s = route_step(g, h, h, 1.0)
    for name in ("depth_m", "old_discharge_m2_s", "flow_depth_m"):
        assert not np.shares_memory(getattr(s, name), h)


# --- literal legacy stale inflow (Codex reproducer, SI) ------------------------------------------
@pytest.mark.parametrize("impl", implementations())
def test_stale_old_inflow_creates_water_only_in_the_literal_mode(impl):
    """Two serial cells after complete upstream infiltration: h = q_old = 0,
    but the receiver's legacy qin(1) still holds 100 mm2/s. dt 1 s, dx 0.5 m.
    Expected spurious gain dx^2 c qin = 0.25 * 1 * 1e-4 = 2.5e-5 m3."""
    g = make_graph(chain_full(2), ff=21.45, dx=0.5)
    zero = np.zeros(g.shape)
    stale = np.array([[1e-4], [0.0]])
    literal = legacy_stale_inflow_step(g, zero, zero, 1.0, stale_old_inflow_m2_s=stale, implementation=impl)
    assert not literal.conservative
    assert float(literal.stale_inflow_gain_m3) == pytest.approx(2.5e-5, rel=1e-14)
    assert float(literal.budget_residual_m3) == pytest.approx(2.5e-5, rel=1e-12)
    stored_plus_exported = 0.25 * float(literal.depth_m.sum()) + float(literal.export_m3)
    assert stored_plus_exported == pytest.approx(2.5e-5, rel=1e-12)
    coherent = route_step(g, zero, zero, 1.0, implementation=impl)
    assert float(coherent.budget_residual_m3) == 0.0 and not np.any(coherent.depth_m)


# --- Plot 1 ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plot1_audit(tmp_path_factory):
    if not (MAHLERAN_ROOT / "mahleran_input.xml").is_file() or not PLOT1_DIR.is_dir():
        pytest.skip(f"MAHLERAN reference not available at {MAHLERAN_ROOT}")
    from maple_syrup.case_import import audit_plot1, load_recipe

    return audit_plot1(load_recipe(RECIPE_PATH, mahleran_root=MAHLERAN_ROOT), tmp_path_factory.mktemp("plot1_audit"))


def _map_file(report, key):
    return next(m["file"] for m in report["legacy_options"]["maps"] if m["xml_key"] == key)


def test_plot1_graph_matches_independent_legacy_transcription(plot1_audit):
    fields, report = plot1_audit.fields, plot1_audit.report
    g = plot1_routing_graph(fields, report)
    dem_nf = read_legacy_grid(PLOT1_DIR / _map_file(report, "dem"))
    rm_nf = read_legacy_grid(PLOT1_DIR / _map_file(report, "rainfall-scaling_map"))
    # The sidecar full grids are already south-first: one flip of the file, no double flip.
    np.testing.assert_array_equal(fields["legacy_full_elevation_m"], dem_nf[::-1])
    np.testing.assert_array_equal(fields["legacy_full_rainfall_scaling"], rm_nf[::-1])
    aspect_nf, slope_nf, applied_nf = legacy_topog(dem_nf, rm_nf, float(report["grid"]["cellsize_m"]))
    np.testing.assert_array_equal(g.aspect, sf_interior(aspect_nf))
    np.testing.assert_array_equal(g.slope, sf_interior(slope_nf))
    np.testing.assert_array_equal(g.edge_rule_applied, sf_interior(applied_nf))
    np.testing.assert_array_equal(g.friction_factor, 21.45)

    # Independent receivers, outlets and longest paths from the transcription.
    ny, nx = g.shape
    sdir = ((-1, 0), (0, 1), (1, 0), (0, -1))
    receiver, outlets = {}, set()
    for i in range(1, ny + 1):
        for j in range(1, nx + 1):
            assert aspect_nf[i, j] != 0, "Plot 1 was audited sink-free"
            di, dj = sdir[aspect_nf[i, j] - 1]
            ti, tj = i + di, j + dj
            if rm_nf[ti, tj] < 0.0:
                outlets.add((i, j))
            else:
                assert 1 <= ti <= ny and 1 <= tj <= nx, "a receiver on an unmasked ring cell"
                receiver[(i, j)] = (ti, tj)
    donors = {}
    for cell, target in receiver.items():
        donors.setdefault(target, []).append(cell)
    memo = {}

    def depth(cell):
        if cell not in memo:
            memo[cell] = 1 + max((depth(d) for d in donors.get(cell, [])), default=-1)
        return memo[cell]

    n_levels = 1 + max(depth((i, j)) for i in range(1, ny + 1) for j in range(1, nx + 1))
    expected_outlet = np.zeros((ny + 2, nx + 2), dtype=bool)
    for i, j in outlets:
        expected_outlet[i, j] = True
    np.testing.assert_array_equal(g.outlet, sf_interior(expected_outlet))
    summary = g.summary()
    assert g.n_active == ny * nx == 1200
    assert g.n_levels == n_levels
    assert all(o["maple_row"] == 0 and o["aspect"] == 3 for o in summary["outlets"])
    print("plot1 routing graph:", {k: v for k, v in summary.items() if k not in ("level_widths", "outlets")})


@pytest.mark.parametrize("impl", implementations())
def test_plot1_single_step_closes(plot1_audit, impl):
    g = plot1_routing_graph(plot1_audit.fields, plot1_audit.report)
    rng = np.random.default_rng(17)
    h_start, h_old = random_state(rng, g, scale=3.5e-3)
    s = route_step(g, h_start, h_old, 1.0, implementation=impl)
    assert abs(float(s.budget_residual_m3)) <= 1e-13
    assert float(s.export_m3) > 0.0 and float(s.max_courant_old) < 1.0
    if impl == "numba":
        reference = route_step(g, h_start, h_old, 1.0)
        np.testing.assert_array_equal(s.depth_m, reference.depth_m)
        np.testing.assert_array_equal(s.discharge_m2_s, reference.discharge_m2_s)


def test_plot1_graph_declares_the_header_nodata_value(plot1_audit):
    """The Plot 1 builder passes the DEM header nodata value; a sentinel
    elevation would then be refused rather than routed to."""
    fields = dict(plot1_audit.fields)
    z = np.array(fields["legacy_full_elevation_m"], copy=True)
    z[0, 5] = float(plot1_audit.report["grid"]["header"]["nodata_value"])
    fields["legacy_full_elevation_m"] = z
    with pytest.raises(RoutingGraphError, match="nodata"):
        plot1_routing_graph(fields, plot1_audit.report)


# --- optional CuPy parity ------------------------------------------------------------------------
def test_cupy_matches_numpy():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; GPU path not exercised (no GPU claim)")
    cp = backend.cupy_module()
    rng = np.random.default_rng(23)
    z = random_full(rng, 12, 9)
    host_graph = make_graph(z, ff=7.0)
    dev_graph = make_graph(z, ff=7.0, xp=cp)
    h_start, h_old = random_state(rng, host_graph)
    a = route_step(host_graph, h_start, h_old, 1.0)
    b = route_step(dev_graph, backend.to_device(h_start, cp), backend.to_device(h_old, cp), 1.0)
    assert backend.is_device_array(b.depth_m)
    for name in ("depth_m", "discharge_m2_s", "inflow_m2_s", "face_volume_m3"):
        np.testing.assert_allclose(backend.to_host(getattr(b, name)), getattr(a, name), rtol=1e-13, atol=1e-18)
