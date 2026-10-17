"""Direct rain-splash core: weights, exact bins and tails, obstructions, closure, dry demand, MAPLE exchange.

The brute-force `oracle` walks every ray cell by cell for every source (loops over cells are fine in a test) with
the unlimited bins of splash_transport.for 119-239 and its own copy of the Fortran slot mapping 66-103. It is an
independent re-derivation, NOT an execution of the original Fortran. CuPy cases skip when no device exists.
"""
from __future__ import annotations

import math
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from maple_syrup.sediment_physics import (
    LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM,
    SedimentPhysicsError,
    dry_raindrop_pickup_demand,
    physics_grid,
    plot1_sediment_parameters,
    sediment_physics_step,
)
from maple_syrup.splash import (
    SplashError,
    apply_splash_exchange,
    ray_weights,
    resolve_implementation,
    splash_grid,
    splash_grid_from_graph,
    splash_parameters,
    splash_redistribute,
)

MV_STEPS = ((0, 1), (-1, 0), (0, -1), (1, 0))  # (d_row, d_col) of splash_mv(1..4): east, south, west, north (row 0 = south)


@pytest.fixture(params=["numpy", "cupy"])
def xp(request):
    if request.param == "numpy":
        return np
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; no GPU claim is made")
    return backend.cupy_module()


def host(xp, array):
    return array if xp is np else xp.asnumpy(array)


def fortran_weights(slope, aspect):
    """splash_transport.for 52-103 for one cell: (mv1, mv2, mv3, mv4)."""
    across = 0.25
    if slope == 0.0:
        return [across] * 4
    down = 0.5 * (1.0 - 0.5 * math.exp(-2.42 * slope))
    up = 0.5 - down
    mv = [across] * 4
    if aspect == 2:
        mv[0], mv[2] = down, up
    elif aspect == 3:
        mv[1], mv[3] = down, up
    elif aspect == 4:
        mv[2], mv[0] = down, up
    else:
        mv[3], mv[1] = down, up
    return mv


def faces_from_sides(active, sides):
    """Independent derivation of the open-side shorthand: only faces of ACTIVE edge cells open."""
    ny, nx = active.shape
    west, east, south, north = sides
    open_x = np.zeros((ny, nx + 1), bool)
    open_y = np.zeros((ny + 1, nx), bool)
    for r in range(ny):
        open_x[r, 0] = west and active[r, 0]
        open_x[r, nx] = east and active[r, nx - 1]
    for c in range(nx):
        open_y[0, c] = south and active[0, c]
        open_y[ny, c] = north and active[ny - 1, c]
    return open_x, open_y


def oracle(pickup, slope, aspect, active, dx, spd, faces):
    """`faces` = (open_x_faces, open_y_faces) exterior openness in MAPLE's face layout."""
    open_x, open_y = faces
    ny, nx, nc = pickup.shape
    dep = np.zeros_like(pickup)
    exp_edge = np.zeros_like(pickup)
    exp_src = np.zeros_like(pickup)
    xg = np.zeros((ny, nx + 1, nc))
    xn = np.zeros((ny, nx + 1, nc))
    yg = np.zeros((ny + 1, nx, nc))
    yn = np.zeros((ny + 1, nx, nc))
    for r in range(ny):
        for c in range(nx):
            if not active[r, c]:
                continue
            w = fortran_weights(slope[r, c], aspect[r, c])
            for k in range(nc):
                p = pickup[r, c, k]
                a = spd[k] * dx
                g = math.exp(-a)
                dep[r, c, k] += p * -math.expm1(-a)
                for m, (dr, dc) in enumerate(MV_STEPS):
                    n = 1
                    while True:
                        rr, cc = r + n * dr, c + n * dc
                        if not (0 <= rr < ny and 0 <= cc < nx and active[rr, cc]):
                            break  # first inactive cell or grid edge ends the ray; nothing jumps a hole
                        dep[rr, cc, k] += p * w[m] * g ** n * (1.0 - g)
                        # the face between cell n - 1 and n is crossed by all mass landing at cell n or beyond
                        _face(xg, xn, yg, yn, r, c, m, n, k, p * w[m] * g ** n)
                        n += 1
                    reached = n - 1
                    tail = p * w[m] * g ** (reached + 1)
                    lr, lc = r + reached * dr, c + reached * dc
                    rr, cc = r + n * dr, c + n * dc
                    face_open = (open_x[lr, nx], open_y[0, lc], open_x[lr, 0], open_y[ny, lc])[m]
                    if not (0 <= rr < ny and 0 <= cc < nx) and face_open:
                        exp_edge[lr, lc, k] += tail
                        exp_src[r, c, k] += tail
                        _face(xg, xn, yg, yn, r, c, m, reached + 1, k, tail)
                    else:
                        dep[lr, lc, k] += tail
    return dep, exp_edge, exp_src, xg, yg, xn, yn


def _face(xg, xn, yg, yn, r, c, m, k, ci, value):
    """Record the crossing of the k-th face from source (r, c) along mv slot m."""
    if m == 0:
        xg[r, c + k, ci] += value
        xn[r, c + k, ci] += value
    elif m == 2:
        xg[r, c - k + 1, ci] += value
        xn[r, c - k + 1, ci] -= value
    elif m == 3:
        yg[r + k, c, ci] += value
        yn[r + k, c, ci] += value
    else:
        yg[r - k + 1, c, ci] += value
        yn[r - k + 1, c, ci] -= value


def run(xp, pickup, slope, aspect, active, dx, spd, open_sides=(True, True, True, True), faces=None,
        implementation="array"):
    params = splash_parameters(spd, xp=xp)
    if faces is None:
        grid = splash_grid(slope, aspect, active, dx, open_sides=open_sides, xp=xp)
    else:
        grid = splash_grid(slope, aspect, active, dx, open_x_faces=faces[0], open_y_faces=faces[1], xp=xp)
    return splash_redistribute(params, grid, xp.asarray(pickup), implementation=implementation), params, grid


def oracle_for_sides(pickup, slope, aspect, active, dx, spd, open_sides=(True, True, True, True)):
    return oracle(pickup, slope, aspect, active, dx, spd, faces_from_sides(active, open_sides))


def assert_matches_oracle(xp, result, expected, rtol=1e-12, atol=1e-15):
    names = ("deposition_kg", "export_edge_cell_kg", "export_source_cell_kg", "x_face_gross_kg", "y_face_gross_kg",
             "x_face_net_kg", "y_face_net_kg")
    for name, want in zip(names, expected, strict=True):
        np.testing.assert_allclose(host(xp, getattr(result, name)), want, rtol=rtol, atol=atol, err_msg=name)


def assert_closed(xp, result):
    assert np.all(np.abs(host(xp, result.cell_balance_residual_kg)) <= host(xp, result.cell_balance_tolerance_kg))
    assert np.all(np.abs(host(xp, result.class_budget_residual_kg)) <= host(xp, result.class_budget_tolerance_kg))
    assert np.all(np.abs(host(xp, result.source_export_residual_kg)) <= host(xp, result.source_export_tolerance_kg))


# --- weights -----------------------------------------------------------------------------
def test_flat_cells_split_equally_for_every_aspect():
    for aspect in (0, 1, 2, 3, 4):
        np.testing.assert_array_equal(ray_weights(0.0, aspect), np.full(4, 0.25))


@pytest.mark.parametrize("aspect, slot", [(2, 0), (3, 1), (4, 2), (1, 3)])
def test_down_ray_follows_aspect_south_first_and_up_is_opposite(aspect, slot):
    s = 0.37
    w = ray_weights(s, aspect)
    down = 0.5 * (1.0 - 0.5 * math.exp(-2.42 * s))
    assert w[slot] == pytest.approx(down, rel=1e-15)
    assert w[(slot + 2) % 4] == pytest.approx(0.5 - down, rel=1e-13)
    assert w[(slot + 1) % 4] == 0.25 and w[(slot + 3) % 4] == 0.25
    assert w.sum() == pytest.approx(1.0, abs=2e-16)
    np.testing.assert_allclose(w, fortran_weights(s, aspect), rtol=1e-15)


def test_weights_are_continuous_with_flat_and_never_negative():
    for aspect in (1, 2, 3, 4):
        np.testing.assert_allclose(ray_weights(1e-14, aspect), 0.25, atol=1e-13)
        for s in (0.0, 1e-6, 0.1, 1.0, 50.0, 1000.0):
            w = ray_weights(s, aspect)
            assert np.all(w >= 0.0) and w.sum() == pytest.approx(1.0, abs=4e-16)


def test_grid_weights_match_and_vanish_on_inactive(xp):
    active = np.ones((2, 3), bool)
    active[0, 1] = False
    aspect = np.array([[3, 3, 2], [4, 1, 0]], np.int8)
    slope = np.array([[0.2, 0.2, 0.1], [0.3, 0.0, 0.0]])
    grid = splash_grid(slope, aspect, active, 1.0, xp=xp)
    w = host(xp, grid.weights)
    assert w.shape == (4, 2, 3)
    np.testing.assert_array_equal(w[:, 0, 1], 0.0)
    np.testing.assert_allclose(w[:, 0, 0], fortran_weights(0.2, 3), rtol=1e-15)
    np.testing.assert_allclose(w[:, 1, 0], fortran_weights(0.3, 4), rtol=1e-15)
    np.testing.assert_array_equal(w[:, 1, 1], 0.25)


# --- exact bins and tails on an open grid ----------------------------------------------------
@pytest.mark.parametrize("aspect", [1, 2, 3, 4])
def test_single_source_analytic_bins_tails_and_exports(xp, aspect):
    n, dx, spd, slope, mass = 9, 1.0, 0.7, 0.4, 3.0
    c = n // 2
    pickup = np.zeros((n, n, 1))
    pickup[c, c, 0] = mass
    res, _, _ = run(xp, pickup, np.full((n, n), slope), np.full((n, n), aspect, np.int8), np.ones((n, n), bool), dx,
                    [spd])
    w = fortran_weights(slope, aspect)
    g = math.exp(-spd * dx)
    dep = np.zeros((n, n, 1))
    exp_edge = np.zeros((n, n, 1))
    dep[c, c, 0] = mass * (1.0 - g)
    for m, (dr, dc) in enumerate(MV_STEPS):
        for k in range(1, c + 1):
            dep[c + k * dr, c + k * dc, 0] += mass * w[m] * g ** k * (1.0 - g)
        exp_edge[c + c * dr, c + c * dc, 0] += mass * w[m] * g ** (c + 1)  # tail beyond the last of c cells
    np.testing.assert_allclose(host(xp, res.deposition_kg), dep, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(host(xp, res.export_edge_cell_kg), exp_edge, rtol=1e-13, atol=1e-16)
    src = np.zeros((n, n, 1))
    src[c, c, 0] = exp_edge.sum()
    np.testing.assert_allclose(host(xp, res.export_source_cell_kg), src, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(host(xp, res.local_deposition_kg)[c, c, 0], mass * (1.0 - g), rtol=1e-15)
    assert float(host(xp, res.class_pickup_kg)[0]) == mass
    assert_closed(xp, res)
    assert not np.any(host(xp, res.obstruction_tail_deposition_kg))


def test_flat_source_pattern_is_fourfold_symmetric(xp):
    n = 7
    pickup = np.zeros((n, n, 1))
    pickup[3, 3, 0] = 1.0
    res, _, _ = run(xp, pickup, np.zeros((n, n)), np.zeros((n, n), np.int8), np.ones((n, n), bool), 0.5, [1.3])
    d = host(xp, res.deposition_kg)[..., 0]
    for k in (1, 2, 3):
        np.testing.assert_allclose(np.rot90(d, k), d, rtol=1e-14, atol=1e-18)
    xn, yn = host(xp, res.x_face_net_kg), host(xp, res.y_face_net_kg)
    np.testing.assert_allclose(xn, -xn[:, ::-1], atol=1e-16)
    np.testing.assert_allclose(yn, -yn[::-1], atol=1e-16)
    assert_closed(xp, res)


def test_face_layout_and_signs_on_a_one_by_three_strip(xp):
    spd, mass = 0.9, 2.0
    g = math.exp(-spd)
    pickup = np.zeros((1, 3, 1))
    pickup[0, 0, 0] = mass
    res, _, _ = run(xp, pickup, np.zeros((1, 3)), np.zeros((1, 3), np.int8), np.ones((1, 3), bool), 1.0, [spd])
    q = 0.25 * mass
    xg = host(xp, res.x_face_gross_kg)[0, :, 0]
    xn = host(xp, res.x_face_net_kg)[0, :, 0]
    # east ray crosses faces 1, 2 and the open east boundary face 3; the west ray exits at once through face 0
    np.testing.assert_allclose(xg, [q * g, q * g, q * g ** 2, q * g ** 3], rtol=1e-14)
    np.testing.assert_allclose(xn, [-q * g, q * g, q * g ** 2, q * g ** 3], rtol=1e-14)
    yg = host(xp, res.y_face_gross_kg)[:, 0, 0]
    yn = host(xp, res.y_face_net_kg)[:, 0, 0]
    np.testing.assert_allclose(yg, [q * g, q * g], rtol=1e-14)  # south exit face 0, north exit face 1
    np.testing.assert_allclose(yn, [-q * g, q * g], rtol=1e-14)
    assert host(xp, res.x_face_gross_kg).shape == (1, 4, 1) and host(xp, res.y_face_gross_kg).shape == (2, 3, 1)
    # edge-cell export: west, north and south tails leave at the source cell (also an edge); east at the far cell
    e = host(xp, res.export_edge_cell_kg)[0, :, 0]
    np.testing.assert_allclose(e, [3 * q * g, 0.0, q * g ** 3], rtol=1e-14)
    np.testing.assert_allclose(host(xp, res.export_source_cell_kg)[0, 0, 0], 3 * q * g + q * g ** 3, rtol=1e-14)
    assert_matches_oracle(xp, res, oracle_for_sides(pickup, np.zeros((1, 3)), np.zeros((1, 3), int),
                                                    np.ones((1, 3), bool), 1.0, [spd]))


# --- oracle comparison on heterogeneous masked grids -------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("open_sides", [(True, True, True, True), (False, True, False, True), (False,) * 4])
def test_random_masked_heterogeneous_grid_matches_oracle(xp, seed, open_sides):
    rng = np.random.default_rng(seed)
    ny, nx, nc = 6, 8, 3
    active = rng.random((ny, nx)) > 0.15
    slope = np.where(rng.random((ny, nx)) > 0.2, rng.random((ny, nx)) * 0.6, 0.0)
    aspect = rng.integers(1, 5, (ny, nx)).astype(np.int8)
    pickup = rng.random((ny, nx, nc)) * 2.0 * active[..., None]
    pickup[rng.random((ny, nx, nc)) > 0.6] = 0.0
    spd = [0.3, 2.0, 0.05]
    dx = 0.5
    res, _, _ = run(xp, pickup, slope, aspect, active, dx, spd, open_sides)
    assert_matches_oracle(xp, res, oracle_for_sides(pickup, slope, aspect, active, dx, spd, open_sides))
    assert_closed(xp, res)
    if not any(open_sides):
        assert not np.any(host(xp, res.export_edge_cell_kg)) and not np.any(host(xp, res.export_source_cell_kg))
        np.testing.assert_allclose(host(xp, res.class_deposition_kg), pickup.sum((0, 1)), rtol=1e-13)


# --- obstructions -------------------------------------------------------------------------------
def test_masked_hole_stops_the_ray_and_tail_lands_at_the_last_active_cell(xp):
    spd, mass = 0.5, 1.0
    g = math.exp(-spd)
    active = np.ones((1, 6), bool)
    active[0, 3] = False  # masked hole; cells 4 and 5 lie beyond it
    pickup = np.zeros((1, 6, 1))
    pickup[0, 0, 0] = mass
    res, _, _ = run(xp, pickup, np.zeros((1, 6)), np.zeros((1, 6), np.int8), active, 1.0, [spd],
                    open_sides=(True, True, False, False))
    d = host(xp, res.deposition_kg)[0, :, 0]
    q = 0.25 * mass
    # east ray reaches cells 1, 2 (m = 2); its tail q g^3 stops at cell 2; cells 3..5 get nothing
    assert d[3] == 0.0 and d[4] == 0.0 and d[5] == 0.0
    expected2 = q * g ** 2 * (1 - g) + q * g ** 3
    assert d[2] == pytest.approx(expected2, rel=1e-13)
    assert d[1] == pytest.approx(q * g * (1 - g), rel=1e-13)
    # blocked south and north rays return to the source (closed top/bottom edges): q g each; west ray exits
    assert d[0] == pytest.approx(mass * (1 - g) + 2 * q * g, rel=1e-13)
    assert host(xp, res.export_edge_cell_kg)[0, 0, 0] == pytest.approx(q * g, rel=1e-13)  # west exit at cell 0
    assert float(host(xp, res.class_export_kg)[0]) == pytest.approx(q * g, rel=1e-13)
    assert host(xp, res.obstruction_tail_deposition_kg)[0, 2, 0] == pytest.approx(q * g ** 3, rel=1e-13)
    assert_closed(xp, res)
    assert_matches_oracle(xp, res, oracle_for_sides(pickup, np.zeros((1, 6)), np.zeros((1, 6), int), active, 1.0,
                                                    [spd], (True, True, False, False)))


def test_source_walled_in_on_all_sides_keeps_everything(xp):
    active = np.zeros((3, 3), bool)
    active[1, 1] = True
    pickup = np.zeros((3, 3, 2))
    pickup[1, 1] = [1.5, 0.25]
    res, _, _ = run(xp, pickup, np.full((3, 3), 0.2), np.full((3, 3), 3, np.int8), active, 2.0, [0.1, 7.0])
    np.testing.assert_allclose(host(xp, res.deposition_kg), pickup, rtol=1e-14)
    assert not np.any(host(xp, res.export_edge_cell_kg))
    assert not np.any(host(xp, res.x_face_gross_kg)) and not np.any(host(xp, res.y_face_gross_kg))
    assert_closed(xp, res)


def test_closed_edge_reflects_nothing_and_deposits_at_edge_cell(xp):
    spd, mass = 0.4, 1.0
    g = math.exp(-spd)
    pickup = np.zeros((1, 3, 1))
    pickup[0, 1, 0] = mass
    res, _, _ = run(xp, pickup, np.zeros((1, 3)), np.zeros((1, 3), np.int8), np.ones((1, 3), bool), 1.0, [spd],
                    open_sides=(False,) * 4)
    q = 0.25 * mass
    d = host(xp, res.deposition_kg)[0, :, 0]
    # east: one cell then wall: bin q g (1-g) + tail q g^2 at cell 2; west mirrored; north/south tails return to source
    assert d[2] == pytest.approx(q * g * (1 - g) + q * g ** 2, rel=1e-13)
    assert d[0] == pytest.approx(d[2], rel=1e-14)
    assert d[1] == pytest.approx(mass * (1 - g) + 2 * q * g, rel=1e-13)
    assert d.sum() == pytest.approx(mass, rel=1e-14)


# --- extreme decay, no truncation, no distance-sized arrays ---------------------------------------
def test_huge_decay_deposits_everything_locally(xp):
    pickup = np.zeros((3, 3, 1))
    pickup[1, 1, 0] = 4.0
    res, _, _ = run(xp, pickup, np.full((3, 3), 0.1), np.full((3, 3), 2, np.int8), np.ones((3, 3), bool), 1.0, [1e300])
    got = host(xp, res.deposition_kg)
    np.testing.assert_array_equal(got, pickup)
    assert not np.any(host(xp, res.export_edge_cell_kg))


def test_tiny_decay_exports_nearly_all_with_no_truncation_and_long_domain_is_cheap(xp):
    nx = 400
    pickup = np.zeros((1, nx, 1))
    pickup[0, 200, 0] = 1.0
    spd = 1e-14
    res, _, _ = run(xp, pickup, np.zeros((1, nx)), np.zeros((1, nx), np.int8), np.ones((1, nx), bool), 1.0, [spd])
    exported = float(host(xp, res.class_export_kg)[0])
    deposited = float(host(xp, res.class_deposition_kg)[0])
    # all four rays leave through open faces; the whole thing is exported but for the (1 - g^k) deposited en route
    assert exported + deposited == pytest.approx(1.0, abs=1e-12)
    assert deposited < 1e-9 and exported > 1.0 - 1e-9
    assert deposited > 0.0  # expm1 keeps the tiny local deposit exactly representable
    assert res.deposition_kg.shape == (1, nx, 1)  # nothing sized by 1 / spd
    assert_closed(xp, res)


def test_zero_decay_sends_the_whole_mass_to_boundaries(xp):
    pickup = np.zeros((3, 3, 1))
    pickup[1, 1, 0] = 2.0
    res, _, _ = run(xp, pickup, np.zeros((3, 3)), np.zeros((3, 3), np.int8), np.ones((3, 3), bool), 1.0, [0.0])
    assert not np.any(host(xp, res.deposition_kg))
    assert float(host(xp, res.class_export_kg)[0]) == pytest.approx(2.0, rel=1e-14)
    np.testing.assert_allclose(host(xp, res.export_edge_cell_kg)[..., 0].sum(), 2.0, rtol=1e-14)
    assert_closed(xp, res)


def test_spd_dx_overflow_is_refused():
    params = splash_parameters([1e308])
    grid = splash_grid(np.zeros((2, 2)), np.zeros((2, 2), np.int8), np.ones((2, 2), bool), 10.0)
    with pytest.raises(SplashError, match="overflow"):
        splash_redistribute(params, grid, np.zeros((2, 2, 1)))


def test_no_pickup_gives_exact_zeros(xp):
    res, _, _ = run(xp, np.zeros((3, 4, 2)), np.full((3, 4), 0.3), np.full((3, 4), 1, np.int8), np.ones((3, 4), bool),
                    1.0, [0.5, 5.0])
    for name in ("deposition_kg", "export_edge_cell_kg", "export_source_cell_kg", "x_face_gross_kg", "y_face_gross_kg",
                 "x_face_net_kg", "y_face_net_kg", "local_deposition_kg"):
        assert not np.any(host(xp, getattr(res, name))), name


# --- validation and non-mutation --------------------------------------------------------------------
def base_objects():
    params = splash_parameters([0.5, 1.0])
    grid = splash_grid(np.full((3, 3), 0.1), np.full((3, 3), 1, np.int8), np.ones((3, 3), bool), 1.0)
    return params, grid


@pytest.mark.parametrize("mutate, match", [
    (lambda p: p.__setitem__((1, 1, 0), -1.0), ">= 0"),
    (lambda p: p.__setitem__((1, 1, 0), np.nan), "finite"),
    (lambda p: p.__setitem__((1, 1, 0), np.inf), "finite"),
])
def test_bad_pickup_values_are_refused_and_input_is_not_modified(mutate, match):
    params, grid = base_objects()
    pickup = np.ones((3, 3, 2))
    mutate(pickup)
    before = pickup.copy()
    with pytest.raises(SplashError, match=match):
        splash_redistribute(params, grid, pickup)
    np.testing.assert_array_equal(pickup, before)


def test_pickup_on_inactive_cell_shape_dtype_and_type_are_refused():
    params = splash_parameters([0.5])
    active = np.ones((2, 2), bool)
    active[0, 0] = False
    grid = splash_grid(np.zeros((2, 2)), np.zeros((2, 2), np.int8), active, 1.0)
    bad = np.zeros((2, 2, 1))
    bad[0, 0, 0] = 1e-9
    with pytest.raises(SplashError, match="inactive"):
        splash_redistribute(params, grid, bad)
    with pytest.raises(SplashError, match="shape"):
        splash_redistribute(params, grid, np.zeros((2, 2, 2)))
    with pytest.raises(SplashError, match="float64"):
        splash_redistribute(params, grid, np.zeros((2, 2, 1), np.float32))
    with pytest.raises(SplashError, match="array"):
        splash_redistribute(params, grid, [[[0.0]]])


@pytest.mark.parametrize("bad", [[], [[0.1]], [-1.0], [np.nan], [np.inf], [True], ["a"], 0.5])
def test_parameters_require_an_explicit_valid_vector(bad):
    with pytest.raises(SplashError):
        splash_parameters(bad)


def test_grid_validation():
    ok = (np.zeros((2, 2)), np.zeros((2, 2), np.int8), np.ones((2, 2), bool), 1.0)
    for index, value in ((3, 0.0), (3, -1.0), (3, np.nan), (3, True)):
        args = list(ok)
        args[index] = value
        with pytest.raises(SplashError):
            splash_grid(*args)
    with pytest.raises(SplashError, match="aspect"):
        splash_grid(np.full((2, 2), 0.1), np.zeros((2, 2), np.int8), np.ones((2, 2), bool), 1.0)
    with pytest.raises(SplashError, match="aspect"):
        splash_grid(np.full((2, 2), 0.1), np.full((2, 2), 5, np.int8), np.ones((2, 2), bool), 1.0)
    with pytest.raises(SplashError, match="slope"):
        splash_grid(np.full((2, 2), -0.1), np.ones((2, 2), np.int8), np.ones((2, 2), bool), 1.0)
    with pytest.raises(SplashError, match="open_sides"):
        splash_grid(*ok, open_sides=(True, True))
    with pytest.raises(SplashError, match="bool"):
        splash_grid(np.zeros((2, 2)), np.zeros((2, 2), np.int8), np.ones((2, 2), int), 1.0)
    with pytest.raises(SplashError, match="shape"):
        splash_grid(np.zeros((2, 2)), np.zeros((2, 3), np.int8), np.ones((2, 2), bool), 1.0)


def test_class_count_mismatch_is_refused():
    params, grid = base_objects()
    with pytest.raises(SplashError, match="shape"):
        splash_redistribute(params, grid, np.zeros((3, 3, 3)))


def test_parameters_are_frozen_copies():
    spd = np.array([0.5, 1.0])
    params = splash_parameters(spd)
    spd[0] = 99.0
    assert params.inverse_distance_per_m[0] == 0.5
    with pytest.raises(ValueError):
        params.inverse_distance_per_m[0] = 1.0


# --- dry raindrop demand: reuse of the wet law, exclusion, rate scaling -----------------------------
RAIN = 5.0e-6  # m/s = 18 mm/h
SLOPE = 0.1
VEG = 0.2


def expected_dry_mass(params, dt, area, rain=RAIN, veg=VEG, slope=SLOPE):
    r2 = rain * 1000.0
    factor = 1.0 - 8.1e-3 * veg * 100.0
    ke = 29.0 - 20.88 * math.exp(-180.0 * r2) * factor  # KE model 2, literal Fortran precedence 33-34
    x = (params.raindrop_a / 1200.0) * (ke * r2 * 1200.0) ** params.raindrop_b * (slope * 100.0) ** params.raindrop_c
    rate = 2.0 * x / params.particle_density_kg_m3 / params.reference_interval_s
    return rate * area * params.particle_density_kg_m3 * dt


def dry_inputs(xp, depth, rain):
    ny, nx = depth.shape
    params = plot1_sediment_parameters(xp=xp)
    grid = physics_grid(np.full((ny, nx), SLOPE), np.ones((ny, nx), bool), 4.0, xp=xp)
    mass = xp.asarray(np.ones((ny, nx, params.n_classes)))
    return params, grid, xp.asarray(depth), xp.asarray(rain), xp.asarray(np.full((ny, nx), VEG)), mass


def test_dry_demand_equals_the_wet_law_formula_only_on_dry_raining_cells(xp):
    depth = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.0]])
    rain = np.array([[RAIN, 0.0, RAIN], [RAIN, RAIN, RAIN]])
    params, grid, d, r, v, m = dry_inputs(xp, depth, rain)
    out = dry_raindrop_pickup_demand(params, grid, d, r, v, m, 1.0)
    cells = host(xp, out.dry_cells)
    np.testing.assert_array_equal(cells, [[True, False, False], [True, True, True]])
    want = expected_dry_mass(params, 1.0, 4.0)
    got = host(xp, out.requested_pickup_kg)
    for rr, cc in np.argwhere(cells):
        np.testing.assert_allclose(got[rr, cc], want, rtol=1e-12)
    assert not np.any(got[0, 1]) and not np.any(got[0, 2])  # no rain / wet: no dry splash demand
    assert np.all(want > 0.0)


def test_dry_demand_scales_with_dt_and_uses_the_one_second_reference(xp):
    depth = np.zeros((2, 2))
    rain = np.full((2, 2), RAIN)
    params, grid, d, r, v, m = dry_inputs(xp, depth, rain)
    one = host(xp, dry_raindrop_pickup_demand(params, grid, d, r, v, m, 1.0).requested_pickup_kg)
    five = host(xp, dry_raindrop_pickup_demand(params, grid, d, r, v, m, 5.0).requested_pickup_kg)
    half = host(xp, dry_raindrop_pickup_demand(params, grid, d, r, v, m, 0.5).requested_pickup_kg)
    np.testing.assert_allclose(five, 5.0 * one, rtol=1e-14)
    np.testing.assert_allclose(half, 0.5 * one, rtol=1e-14)
    # a different declared reference interval halves the per-second rate: the dt scaling is rate x dt
    p2 = plot1_sediment_parameters(xp=xp, reference_interval_s=2.0)
    two = host(xp, dry_raindrop_pickup_demand(p2, grid, d, r, v, m, 1.0).requested_pickup_kg)
    np.testing.assert_allclose(two, 0.5 * one, rtol=1e-14)


def test_dry_demand_is_the_wet_step_law_without_depth_attenuation(xp):
    depth = np.full((2, 2), 0.01)
    rain = np.full((2, 2), RAIN)
    params, grid, _, r, v, m = dry_inputs(xp, depth, rain)
    zeros = xp.asarray(np.zeros((2, 2)))
    wet = sediment_physics_step(params, grid, xp.asarray(depth), zeros, r, v, m, xp.zeros_like(m), 1.0)
    dry = dry_raindrop_pickup_demand(params, grid, zeros, r, v, m, 1.0)
    attenuation = np.exp(-np.array(LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM) * 1.0)  # 0.01 m = 1 cm
    np.testing.assert_allclose(host(xp, wet.raindrop_pickup_kg), host(xp, dry.requested_pickup_kg) * attenuation,
                               rtol=1e-12)
    # the same wet cells are excluded from the dry law
    assert not np.any(host(xp, dry_raindrop_pickup_demand(params, grid, xp.asarray(depth), r, v, m, 1.0)
                           .requested_pickup_kg))


def test_dry_demand_no_rain_and_validation(xp):
    depth = np.zeros((2, 2))
    params, grid, d, r, v, m = dry_inputs(xp, depth, np.zeros((2, 2)))
    out = dry_raindrop_pickup_demand(params, grid, d, r, v, m, 1.0)
    assert not np.any(host(xp, out.requested_pickup_kg)) and not np.any(host(xp, out.dry_cells))
    with pytest.raises(SedimentPhysicsError):
        dry_raindrop_pickup_demand(params, grid, d, r, v, m, 0.0)
    with pytest.raises(SedimentPhysicsError, match=">= 0"):
        dry_raindrop_pickup_demand(params, grid, xp.asarray(-np.ones((2, 2))), r, v, m, 1.0)
    before = host(xp, r).copy()
    dry_raindrop_pickup_demand(params, grid, d, r, v, m, 1.0)
    np.testing.assert_array_equal(host(xp, r), before)


# --- real MAPLE exchange -------------------------------------------------------------------------------
def arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value):
        for f in fields(value):
            yield from arrays(getattr(value, f.name))


@pytest.fixture
def bed():
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_availability import (
        sediment_availability_state_from_active_layer,
    )
    from maple.surface.topographic_commit import (
        initial_committed_topography_state_from_physical_state,
    )

    from maple_syrup.probe import build_minimal_state
    from maple_syrup.sediment_bed import BedContext, BedState

    p = build_minimal_state()
    committed = initial_committed_topography_state_from_physical_state(
        p.voxel_column, p.active_layer, p.geometry, p.mass_resolution_kg)
    available = sediment_availability_state_from_active_layer(p.active_layer.mass_kg,
                                                              np.ones_like(p.active_layer.mass_kg))
    state = BedState(p.voxel_column, p.active_layer, p.water, p.ledger, committed, available)
    ctx = BedContext(p.geometry, p.grain_classes, p.mass_resolution_kg, TopographicCommitSpec(),
                     AvalanchingSpec(enabled=False), np.ones_like(p.active_layer.mass_kg))
    return state, ctx


def test_native_exchange_caps_pickup_and_preserves_preexisting_mobile(bed):
    from maple_syrup.sediment_bed import bed_inventory, with_water

    state, ctx = bed
    nc = len(ctx.grain_classes.classes)
    ny, nx = 3, 4
    mobile = np.zeros((ny, nx, nc))
    mobile[0, 3] = 0.01  # preexisting water-mobile sediment that the splash law must not move
    dry = with_water(state, ctx, np.zeros((ny, nx)), mobile)
    requested = np.zeros((ny, nx, nc))
    requested[1, 1] = 10.0  # far above holdings: MAPLE caps it
    requested[1, 2] = np.arange(1, nc + 1) * 1e-4  # heterogeneous small pickup
    requested[2, 0, 2] = 5e-4
    params = splash_parameters(np.linspace(0.4, 3.0, nc))
    grid = splash_grid(np.full((ny, nx), 0.2), np.full((ny, nx), 3, np.int8), np.ones((ny, nx), bool), 0.5,
                       open_sides=(True,) * 4)
    before = [a.copy() for a in arrays(dry)] + [requested.copy()]

    out = apply_splash_exchange(dry, ctx, params, grid, requested)

    for saved, original in zip(before, [*arrays(dry), requested], strict=True):
        np.testing.assert_array_equal(saved, original)  # caller state and demand untouched
    actual = out.pickup.actual_removal_by_cell_class_kg
    assert np.all(actual <= requested + 1e-15)
    np.testing.assert_allclose(actual[1, 1], dry.active_layer.mass_kg[1, 1], atol=1e-10)  # MAPLE holdings cap
    assert actual[1, 1].sum() < requested[1, 1].sum()
    np.testing.assert_allclose(out.splash.class_pickup_kg, actual.sum((0, 1)), rtol=1e-13)
    # splash redistributed the ACTUAL pickup: the closure is on actual, not requested
    np.testing.assert_allclose(out.splash.class_deposition_kg + out.splash.class_export_kg, actual.sum((0, 1)),
                               rtol=1e-12)
    audit = out.audit
    np.testing.assert_allclose(audit.native_deposited_by_class_kg + audit.unapplied_deposit_by_class_kg,
                               out.splash.class_deposition_kg, rtol=1e-12, atol=1e-15)
    # native deposit/export equal the request up to MAPLE's declared sub-resolution burial residual
    slack = nc * ny * nx * ctx.mass_resolution_kg + 1e-12
    np.testing.assert_allclose(audit.native_deposited_by_class_kg, out.splash.class_deposition_kg, rtol=0, atol=slack)
    np.testing.assert_allclose(audit.native_export_by_class_kg, out.splash.class_export_kg, rtol=0, atol=1e-12)
    # preexisting water-mobile sediment: the pool returns to what it was before the exchange
    change = out.bed.water.mobile_mass_by_cell_class_kg - mobile
    np.testing.assert_allclose(change, 0.0, atol=1e-12)
    np.testing.assert_allclose(out.bed.water.mobile_mass_by_cell_class_kg[0, 3], mobile[0, 3], atol=1e-12)
    np.testing.assert_allclose(audit.mobile_change_by_cell_class_kg, change, atol=1e-15)
    np.testing.assert_array_equal(out.bed.water.depth_m, dry.water.depth_m)
    assert out.bed.sediment_availability is not None
    # inventory: bed + mobile before == bed + mobile after + native boundary export
    lhs = bed_inventory(dry) + mobile.sum((0, 1))
    rhs = bed_inventory(out.bed) + out.bed.water.mobile_mass_by_cell_class_kg.sum((0, 1)) \
        + out.deposit.boundary_export_by_class_kg
    np.testing.assert_allclose(lhs, rhs, atol=1e-10, rtol=0)
    np.testing.assert_allclose(audit.conservation_residual_by_class_kg, 0.0, atol=1e-10)
    # the native deposit saw the splash faces, export and deposition request
    np.testing.assert_array_equal(out.deposit.face_flux.x_face_crossing_mass_kg, out.splash.x_face_gross_kg)
    np.testing.assert_array_equal(out.deposit.face_flux.y_face_net_crossing_mass_kg, out.splash.y_face_net_kg)
    totals = audit.scalar_totals()
    assert all(math.isfinite(v) for v in totals.values()) and totals["actual_pickup_by_class_kg"] > 0.0


def test_native_exchange_refuses_wet_and_inactive_pickup(bed):
    from maple_syrup.sediment_bed import with_water

    state, ctx = bed
    nc = len(ctx.grain_classes.classes)
    params = splash_parameters(np.full(nc, 1.0))
    active = np.ones((3, 4), bool)
    active[2, 3] = False
    grid = splash_grid(np.full((3, 4), 0.2), np.full((3, 4), 3, np.int8), active, 0.5)
    depth = np.zeros((3, 4))
    depth[1, 1] = 1e-3
    wet = with_water(state, ctx, depth)
    requested = np.zeros((3, 4, nc))
    requested[1, 1] = 1e-4
    with pytest.raises(SplashError, match="wet cell"):
        apply_splash_exchange(wet, ctx, params, grid, requested)
    dry = with_water(state, ctx, np.zeros((3, 4)))
    requested = np.zeros((3, 4, nc))
    requested[2, 3] = 1e-4
    with pytest.raises(SplashError, match="inactive"):
        apply_splash_exchange(dry, ctx, params, grid, requested)
    with pytest.raises(SplashError, match="classes"):
        apply_splash_exchange(dry, ctx, splash_parameters([1.0, 2.0]), grid, np.zeros((3, 4, 2)))


# --- C1: exact per-face exterior openness ----------------------------------------------------------------------
RESULT_FIELDS = ("deposition_kg", "local_deposition_kg", "ray_bin_deposition_kg", "obstruction_tail_deposition_kg",
                 "export_edge_cell_kg", "export_source_cell_kg", "x_face_gross_kg", "y_face_gross_kg",
                 "x_face_net_kg", "y_face_net_kg", "class_pickup_kg", "class_deposition_kg", "class_export_kg",
                 "class_export_source_kg")


def flat_pit_graph(xp):
    """The confirmed defect case: flat 3 x 4 interior, closed export flags, pit + flat storage -> 0 outlets."""
    from maple_syrup.routing import build_routing_graph

    return build_routing_graph(np.zeros((5, 6)), np.zeros((5, 6), bool), np.full((3, 4), 21.45), 1.0, xp=xp,
                               allow_pit_storage=True, allow_flat_storage=True)


def plane_graph(xp):
    """3 x 4 interior draining south (aspect 3) into an export-flagged south ring: 4 outlets, all on the south face."""
    from maple_syrup.routing import build_routing_graph

    z = np.zeros((5, 6)) + (0.1 + 0.01 * np.arange(5))[:, None]
    exports = np.zeros((5, 6), bool)
    exports[0, 1:-1] = True
    return build_routing_graph(z, exports, np.full((3, 4), 21.45), 1.0, xp=xp)


def test_graph_without_outlets_is_closed_and_exports_nothing(xp):
    graph = flat_pit_graph(xp)
    assert int(graph.outlet.sum()) == 0
    grid = splash_grid_from_graph(graph)
    assert grid.n_open_faces == 0 and not grid.open_x_faces.any() and not grid.open_y_faces.any()
    pickup = np.zeros((3, 4, 1))
    pickup[1, 1, 0] = 1.0
    res = splash_redistribute(splash_parameters([0.5], xp=xp), grid, xp.asarray(pickup))
    assert not np.any(host(xp, res.export_edge_cell_kg)) and not np.any(host(xp, res.export_source_cell_kg))
    assert float(host(xp, res.class_export_kg)[0]) == 0.0
    assert float(host(xp, res.class_deposition_kg)[0]) == pytest.approx(1.0, rel=1e-14)
    # no mass crosses any outer face
    xg, yg = host(xp, res.x_face_gross_kg), host(xp, res.y_face_gross_kg)
    assert not xg[:, 0].any() and not xg[:, -1].any() and not yg[0].any() and not yg[-1].any()
    assert_closed(xp, res)


def test_graph_outlets_open_exactly_their_aspect_faces(xp):
    graph = plane_graph(xp)
    assert int(graph.outlet.sum()) == 4 and np.all(graph.aspect == 3)
    grid = splash_grid_from_graph(graph)
    assert grid.open_y_faces[0].all() and grid.n_open_faces == 4 and grid.ignored_interior_outlets == 0
    assert not grid.open_x_faces.any() and not grid.open_y_faces[1:].any()
    pickup = np.zeros((3, 4, 1))
    pickup[1, 1, 0] = 1.0
    spd = 0.5
    res = splash_redistribute(splash_parameters([spd], xp=xp), grid, xp.asarray(pickup))
    w = fortran_weights(float(graph.slope[1, 1]), 3)
    g = math.exp(-spd * 1.0)
    # only the south ray (m = 1 cell, then the open south face) exports; east/west/north rays end on closed faces
    expected = w[1] * g ** 2
    assert float(host(xp, res.class_export_kg)[0]) == pytest.approx(expected, rel=1e-13)
    edge = host(xp, res.export_edge_cell_kg)[..., 0]
    assert edge[0, 1] == pytest.approx(expected, rel=1e-13) and edge.sum() == pytest.approx(expected, rel=1e-13)
    # the same grid from explicit faces and from the shorthand give identical results
    other = splash_grid(graph.slope, graph.aspect, graph.active, graph.dx_m, open_sides=(False, False, True, False), xp=xp)
    res2 = splash_redistribute(splash_parameters([spd], xp=xp), other, xp.asarray(pickup))
    for name in RESULT_FIELDS:
        np.testing.assert_array_equal(host(xp, getattr(res, name)), host(xp, getattr(res2, name)), err_msg=name)
    assert_closed(xp, res)


def random_case(seed, ny=6, nx=8, nc=3):
    rng = np.random.default_rng(seed)
    active = rng.random((ny, nx)) > 0.15
    slope = np.where(rng.random((ny, nx)) > 0.2, rng.random((ny, nx)) * 0.6, 0.0)
    aspect = rng.integers(1, 5, (ny, nx)).astype(np.int8)
    pickup = rng.random((ny, nx, nc)) * 2.0 * active[..., None]
    pickup[rng.random((ny, nx, nc)) > 0.6] = 0.0
    return pickup, slope, aspect, active


def test_all_faces_closed_one_open_face_and_partially_open_edge(xp):
    pickup, slope, aspect, active = random_case(5)
    active[:] = True  # every edge face adjoins an active cell so each mask is honoured literally
    pickup = np.where(active[..., None], pickup, 0.0)
    ny, nx = active.shape
    spd = [0.3, 2.0, 0.05]
    closed = (np.zeros((ny, nx + 1), bool), np.zeros((ny + 1, nx), bool))
    one = (np.zeros((ny, nx + 1), bool), np.zeros((ny + 1, nx), bool))
    one[0][2, nx] = True  # the single east face of row 2
    part = (np.zeros((ny, nx + 1), bool), np.zeros((ny + 1, nx), bool))
    part[1][ny, 1:4] = True  # north edge open for columns 1..3 only
    part[0][0, 0] = True  # and the west face of row 0
    for faces in (closed, one, part):
        res, _, grid = run(xp, pickup, slope, aspect, active, 0.5, spd, faces=faces)
        assert_matches_oracle(xp, res, oracle(pickup, slope, aspect, active, 0.5, spd, faces))
        assert_closed(xp, res)
        assert grid.n_open_faces == int(faces[0].sum() + faces[1].sum())
    res, _, _ = run(xp, pickup, slope, aspect, active, 0.5, spd, faces=closed)
    assert not np.any(host(xp, res.export_edge_cell_kg))
    np.testing.assert_allclose(host(xp, res.class_deposition_kg), pickup.sum((0, 1)), rtol=1e-13)
    res, _, _ = run(xp, pickup, slope, aspect, active, 0.5, spd, faces=one)
    edge = host(xp, res.export_edge_cell_kg)
    assert edge.sum() > 0.0 and np.count_nonzero(edge.sum(-1)) == 1
    assert edge[2, nx - 1].sum() == pytest.approx(edge.sum(), rel=1e-14)
    res, _, _ = run(xp, pickup, slope, aspect, active, 0.5, spd, faces=part)
    edge = host(xp, res.export_edge_cell_kg).sum(-1)
    assert np.count_nonzero(edge[:, 1:-1]) <= 3 and not edge[1:ny - 1, 1:nx - 1].any()  # north cells 1..3 only
    assert not edge[ny - 1, [0, 4, 5, 6, 7]].any()


def test_inactive_edge_cells_cannot_be_opened_but_shorthand_skips_them():
    active = np.ones((3, 4), bool)
    active[0, 0] = False
    z = (np.zeros((3, 4)), np.zeros((3, 4), np.int8))
    bad = np.zeros((3, 5), bool)
    bad[0, 0] = True  # the west face of the inactive cell (0, 0)
    with pytest.raises(SplashError, match="inactive"):
        splash_grid(*z, active, 1.0, open_x_faces=bad)
    bad = np.zeros((4, 4), bool)
    bad[0, 0] = True
    with pytest.raises(SplashError, match="inactive"):
        splash_grid(*z, active, 1.0, open_y_faces=bad)
    grid = splash_grid(*z, active, 1.0, open_sides=(True, False, True, False))
    assert not grid.open_x_faces[0, 0] and grid.open_x_faces[1:, 0].all() and grid.open_x_faces[:, 0].sum() == 2
    assert not grid.open_y_faces[0, 0] and grid.open_y_faces[0, 1:].all()


def test_face_mask_validation_and_default_closed():
    z = (np.zeros((3, 4)), np.zeros((3, 4), np.int8), np.ones((3, 4), bool), 1.0)
    assert splash_grid(*z).n_open_faces == 0  # nothing specified: closed
    interior = np.zeros((3, 5), bool)
    interior[1, 2] = True
    with pytest.raises(SplashError, match="OUTER"):
        splash_grid(*z, open_x_faces=interior)
    interior = np.zeros((4, 4), bool)
    interior[2, 1] = True
    with pytest.raises(SplashError, match="OUTER"):
        splash_grid(*z, open_y_faces=interior)
    with pytest.raises(SplashError, match="bool"):
        splash_grid(*z, open_x_faces=np.zeros((3, 5), int))
    with pytest.raises(SplashError, match="shape"):
        splash_grid(*z, open_x_faces=np.zeros((3, 4), bool))
    with pytest.raises(SplashError, match="shape"):
        splash_grid(*z, open_y_faces=np.zeros((3, 4), bool))
    with pytest.raises(SplashError, match="not both"):
        splash_grid(*z, open_sides=(True,) * 4, open_x_faces=np.zeros((3, 5), bool))


def test_face_inputs_are_copied_not_aliased_and_grid_is_read_only():
    z = (np.zeros((3, 4)), np.zeros((3, 4), np.int8), np.ones((3, 4), bool), 1.0)
    ox = np.zeros((3, 5), bool)
    ox[1, 4] = True
    keep = ox.copy()
    grid = splash_grid(*z, open_x_faces=ox)
    np.testing.assert_array_equal(ox, keep)  # input untouched
    ox[:] = False  # later caller mutation does not reach the grid
    assert grid.open_x_faces[1, 4] and grid.n_open_faces == 1
    with pytest.raises(ValueError):
        grid.open_x_faces[0, 0] = True


def test_corrupt_grid_is_refused_before_any_output():
    from dataclasses import replace

    params = splash_parameters([0.5])
    grid = splash_grid(np.zeros((3, 4)), np.zeros((3, 4), np.int8), np.ones((3, 4), bool), 1.0, open_sides=(True,) * 4)
    pickup = np.zeros((3, 4, 1))
    for broken in (replace(grid, weights=grid.weights[:3]), replace(grid, ray_continues=grid.ray_continues.astype(int)),
                   replace(grid, reach_cells=grid.reach_cells.astype(np.float64)),
                   replace(grid, open_x_faces=np.ones((3, 5), bool)),
                   replace(grid, open_y_faces=np.zeros((3, 4), bool))):
        with pytest.raises(SplashError, match="corrupt"):
            splash_redistribute(params, broken, pickup)


def test_implementation_selection_is_explicit_and_refuses_unsupported():
    params = splash_parameters([0.5])
    grid = splash_grid(np.zeros((2, 2)), np.zeros((2, 2), np.int8), np.ones((2, 2), bool), 1.0, open_sides=(True,) * 4)
    pickup = np.zeros((2, 2, 1))
    pickup[0, 0, 0] = 1.0
    with pytest.raises(SplashError, match="CuPy"):
        splash_redistribute(params, grid, pickup, implementation="cuda")
    with pytest.raises(SplashError, match="implementation"):
        splash_redistribute(params, grid, pickup, implementation="numba")
    assert resolve_implementation("auto", np) == "array" and resolve_implementation("array", np) == "array"
    a = splash_redistribute(params, grid, pickup)
    b = splash_redistribute(params, grid, pickup, implementation="auto")
    np.testing.assert_array_equal(a.deposition_kg, b.deposition_kg)


# --- C1: CUDA sweep versus the array reference (device required; skips honestly) -------------------------------
@pytest.fixture
def gpu():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA sweep is not exercised, no claim made")
    return backend.cupy_module()


def both(gpu, pickup, slope, aspect, active, dx, spd, **kwargs):
    array, _, _ = run(gpu, pickup, slope, aspect, active, dx, spd, implementation="array", **kwargs)
    cuda, _, _ = run(gpu, pickup, slope, aspect, active, dx, spd, implementation="cuda", **kwargs)
    return array, cuda


def assert_cuda_matches_array(gpu, array, cuda):
    """Declared bound: every element within rtol 1e-13 (atol 1e-300) of the array path; the class budgets of BOTH
    close inside their own declared tolerances, and the class sums agree inside the larger of the two."""
    for name in RESULT_FIELDS:
        np.testing.assert_allclose(host(gpu, getattr(cuda, name)), host(gpu, getattr(array, name)), rtol=1e-13,
                                   atol=1e-300, err_msg=name)
    assert_closed(gpu, array)
    assert_closed(gpu, cuda)
    tol = np.maximum(host(gpu, array.class_budget_tolerance_kg), host(gpu, cuda.class_budget_tolerance_kg))
    assert np.all(np.abs(host(gpu, cuda.class_deposition_kg) - host(gpu, array.class_deposition_kg)) <= tol)
    assert np.all(np.abs(host(gpu, cuda.class_export_kg) - host(gpu, array.class_export_kg)) <= tol)
    for name in ("deposition_kg", "export_edge_cell_kg", "x_face_net_kg", "y_face_net_kg"):
        assert np.all(np.isfinite(host(gpu, getattr(cuda, name))))
    assert np.all(host(gpu, cuda.deposition_kg) >= 0.0) and np.all(host(gpu, cuda.export_edge_cell_kg) >= 0.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("open_sides", [(True,) * 4, (False, True, False, True), (False,) * 4])
def test_cuda_random_masked_grids_match_array_and_oracle(gpu, seed, open_sides):
    pickup, slope, aspect, active = random_case(seed)
    spd = [0.3, 2.0, 0.05]
    array, cuda = both(gpu, pickup, slope, aspect, active, 0.5, spd, open_sides=open_sides)
    assert_cuda_matches_array(gpu, array, cuda)
    assert_matches_oracle(gpu, cuda, oracle_for_sides(pickup, slope, aspect, active, 0.5, spd, open_sides))


@pytest.mark.parametrize("seed", [0, 1])
def test_cuda_is_bitwise_identical_to_array_hypothesis(gpu, seed):
    """Same operations in the same order with FMA disabled: bitwise equality is the INTENT, tested as a hypothesis."""
    pickup, slope, aspect, active = random_case(seed)
    array, cuda = both(gpu, pickup, slope, aspect, active, 0.5, [0.3, 2.0, 0.05])
    for name in RESULT_FIELDS:
        np.testing.assert_array_equal(host(gpu, getattr(cuda, name)), host(gpu, getattr(array, name)), err_msg=name)


@pytest.mark.parametrize("aspect", [1, 2, 3, 4])
def test_cuda_all_four_orientations_single_source(gpu, aspect):
    n, c = 9, 4
    pickup = np.zeros((n, n, 1))
    pickup[c, c, 0] = 3.0
    slope, asp, active = np.full((n, n), 0.4), np.full((n, n), aspect, np.int8), np.ones((n, n), bool)
    array, cuda = both(gpu, pickup, slope, asp, active, 1.0, [0.7])
    assert_cuda_matches_array(gpu, array, cuda)
    assert_matches_oracle(gpu, cuda, oracle_for_sides(pickup, slope, asp, active, 1.0, [0.7]))


@pytest.mark.parametrize("shape", [(1, 37), (37, 1), (1, 1)])
def test_cuda_degenerate_strips_and_single_cell(gpu, shape):
    rng = np.random.default_rng(7)
    pickup = rng.random(shape + (2,))
    slope, asp, active = np.zeros(shape), np.zeros(shape, np.int8), np.ones(shape, bool)
    array, cuda = both(gpu, pickup, slope, asp, active, 1.0, [0.2, 3.0])
    assert_cuda_matches_array(gpu, array, cuda)
    assert_matches_oracle(gpu, cuda, oracle_for_sides(pickup, slope, asp, active, 1.0, [0.2, 3.0]))


def test_cuda_fully_masked_blocks_and_all_inactive(gpu):
    rng = np.random.default_rng(3)
    ny, nx = 8, 9
    active = np.ones((ny, nx), bool)
    active[2:6, 3:7] = False  # interior block hole
    active[:, 0] = False  # fully masked west edge column
    pickup = rng.random((ny, nx, 2)) * active[..., None]
    slope, asp = rng.random((ny, nx)) * 0.5, rng.integers(1, 5, (ny, nx)).astype(np.int8)
    array, cuda = both(gpu, pickup, slope, asp, active, 1.0, [0.4, 1.5], open_sides=(True,) * 4)
    assert_cuda_matches_array(gpu, array, cuda)
    assert_matches_oracle(gpu, cuda, oracle_for_sides(pickup, slope, asp, active, 1.0, [0.4, 1.5]))
    none = np.zeros((ny, nx), bool)
    array, cuda = both(gpu, np.zeros((ny, nx, 2)), slope, asp, none, 1.0, [0.4, 1.5], open_sides=(True,) * 4)
    for name in RESULT_FIELDS:
        assert not np.any(host(gpu, getattr(cuda, name))), name


@pytest.mark.parametrize("spd", [[1e300, 1e-14, 0.0], [0.0, 0.0, 0.0], [1e-300, 5.0, 1e300]])
def test_cuda_extreme_decay(gpu, spd):
    pickup, slope, aspect, active = random_case(11)
    array, cuda = both(gpu, pickup, slope, aspect, active, 1.0, spd, open_sides=(True,) * 4)
    assert_cuda_matches_array(gpu, array, cuda)
    assert_matches_oracle(gpu, cuda, oracle_for_sides(pickup, slope, aspect, active, 1.0, spd))


def test_cuda_per_face_masks_and_graph_grid(gpu):
    graph = plane_graph(gpu)
    grid = splash_grid_from_graph(graph)
    params = splash_parameters([0.5, 2.0], xp=gpu)
    pickup = np.random.default_rng(2).random((3, 4, 2))
    a = splash_redistribute(params, grid, gpu.asarray(pickup), implementation="array")
    c = splash_redistribute(params, grid, gpu.asarray(pickup), implementation="cuda")
    assert_cuda_matches_array(gpu, a, c)
    assert float(host(gpu, c.class_export_kg).sum()) > 0.0
    closed = splash_grid_from_graph(flat_pit_graph(gpu))
    z = splash_redistribute(params, closed, gpu.asarray(pickup), implementation="cuda")
    assert not np.any(host(gpu, z.export_edge_cell_kg))


def test_cuda_does_not_mutate_inputs_and_refuses_bad_ones(gpu):
    pickup, slope, aspect, active = random_case(4)
    params = splash_parameters([0.3, 2.0, 0.05], xp=gpu)
    grid = splash_grid(slope, aspect, active, 0.5, open_sides=(True,) * 4, xp=gpu)
    dev = gpu.asarray(pickup)
    before = host(gpu, dev).copy()
    weights = host(gpu, grid.weights).copy()
    splash_redistribute(params, grid, dev, implementation="cuda")
    np.testing.assert_array_equal(host(gpu, dev), before)
    np.testing.assert_array_equal(host(gpu, grid.weights), weights)
    noncontig = gpu.asarray(np.asfortranarray(pickup))  # a non-C-contiguous caller array is read, never written
    res = splash_redistribute(params, grid, noncontig, implementation="cuda")
    ref = splash_redistribute(params, grid, dev, implementation="array")
    np.testing.assert_allclose(host(gpu, res.deposition_kg), host(gpu, ref.deposition_kg), rtol=1e-13, atol=1e-300)
    bad = dev.copy()
    bad[0, 0, 0] = -1.0
    with pytest.raises(SplashError, match=">= 0"):
        splash_redistribute(params, grid, bad, implementation="cuda")


def test_cuda_kernel_provenance_is_reported(gpu):
    from maple_syrup import splash_cuda

    info = splash_cuda.kernel_provenance()
    assert info["fastmath"] is False and "--fmad=false" in info["compile_options"]
    assert "__dmul_rn" in splash_cuda.kernel_source() and "fast" not in splash_cuda.kernel_source()
    assert splash_cuda.CUDA_QUALIFIED is False  # "auto" stays on the array path until a qualified run sets it


# --- C1: actual-MAPLE exchange on a device bed ---------------------------------------------------------------------
def leaves(value):
    from maple.core.backend import is_array

    if is_array(value):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            yield from leaves(getattr(value, f.name))


def test_native_exchange_on_device_matches_cpu_and_conserves(bed, gpu):
    from dataclasses import replace

    from maple.core.backend import to_device, to_device_tree, to_host

    from maple_syrup.sediment_bed import bed_inventory, with_water

    def H(value):
        return np.asarray(to_host(value))

    state, ctx = bed
    nc = len(ctx.grain_classes.classes)
    ny, nx = 3, 4
    mobile = np.zeros((ny, nx, nc))
    mobile[0, 3] = 0.01  # preexisting water-mobile sediment
    requested = np.zeros((ny, nx, nc))
    requested[1, 1] = 10.0  # above holdings: MAPLE must cap it
    requested[1, 2] = np.arange(1, nc + 1) * 1e-4
    requested[2, 0, 2] = 5e-4
    raw_spd = np.linspace(0.4, 3.0, nc)
    terrain = (np.full((ny, nx), 0.2), np.full((ny, nx), 3, np.int8), np.ones((ny, nx), bool), 0.5)

    cpu_dry = with_water(state, ctx, np.zeros((ny, nx)), mobile)
    cpu = apply_splash_exchange(cpu_dry, ctx, splash_parameters(raw_spd),
                                splash_grid(*terrain, open_sides=(True,) * 4), requested)

    dstate = to_device_tree(state, gpu)
    dctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, gpu))
    dry = with_water(dstate, dctx, gpu.zeros((ny, nx)), gpu.asarray(mobile))
    params = splash_parameters(raw_spd, xp=gpu)
    grid = splash_grid(*terrain, open_sides=(True,) * 4, xp=gpu)
    dev_requested = gpu.asarray(requested)
    saved = [H(a).copy() for a in leaves(dry)] + [requested.copy()]

    out = apply_splash_exchange(dry, dctx, params, grid, dev_requested, implementation="cuda")

    for before, after in zip(saved, [*leaves(dry), dev_requested], strict=True):
        np.testing.assert_array_equal(before, H(after))  # caller bed, mobile, depth and demand untouched
    actual = H(out.pickup.actual_removal_by_cell_class_kg)
    holdings = H(dry.active_layer.mass_kg)
    assert np.all(actual <= requested + 1e-15)
    np.testing.assert_allclose(actual[1, 1], holdings[1, 1], atol=1e-10)  # MAPLE holdings cap, not the demand
    assert actual[1, 1].sum() < requested[1, 1].sum()
    np.testing.assert_allclose(H(out.splash.class_pickup_kg), actual.sum((0, 1)), rtol=1e-13)
    np.testing.assert_allclose(H(out.splash.class_deposition_kg) + H(out.splash.class_export_kg), actual.sum((0, 1)),
                               rtol=1e-12)
    # native deposit / export equal the splash request at the native bound (resolution burial residual)
    audit = out.audit
    slack = nc * ny * nx * ctx.mass_resolution_kg + 1e-12
    np.testing.assert_allclose(H(audit.native_deposited_by_class_kg), H(out.splash.class_deposition_kg), rtol=0,
                               atol=slack)
    np.testing.assert_allclose(H(audit.native_export_by_class_kg), H(out.splash.class_export_kg), rtol=0, atol=1e-12)
    # preexisting mobile and depth preserved
    after_mobile = H(out.bed.water.mobile_mass_by_cell_class_kg)
    np.testing.assert_allclose(after_mobile, mobile, atol=1e-12)
    np.testing.assert_array_equal(H(out.bed.water.depth_m), np.zeros((ny, nx)))
    # bed + mobile + export conservation, and MAPLE's own independent residual
    lhs = H(bed_inventory(dry)) + mobile.sum((0, 1))
    rhs = H(bed_inventory(out.bed)) + after_mobile.sum((0, 1)) + H(out.deposit.boundary_export_by_class_kg)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10, rtol=0)
    np.testing.assert_allclose(H(audit.conservation_residual_by_class_kg), 0.0, atol=1e-10)
    # every native residual stays reported (never erased)
    totals = audit.scalar_totals()
    assert math.isfinite(totals["numerical_residual_scalar_kg"])
    assert H(audit.pickup_numerical_residual_by_class_kg).shape == (nc,)
    # declared cross-backend sediment bound (rtol 1e-10, atol 1e-14) against the CPU exchange
    np.testing.assert_allclose(actual, cpu.pickup.actual_removal_by_cell_class_kg, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(H(out.splash.deposition_kg), cpu.splash.deposition_kg, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(H(out.splash.export_edge_cell_kg), cpu.splash.export_edge_cell_kg, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(H(audit.native_deposited_by_class_kg), cpu.audit.native_deposited_by_class_kg,
                               rtol=1e-10, atol=1e-14)
