"""Pond network (topology) tests. Nothing here was run by its author (file-only tools); Codex records results.

Run explicitly (`pytest tests/pond`), not together with the suites whose non-package conftest files share a module name. The
module also holds the small terrain builders that `test_hydraulics.py` imports (`from test_network import ...`).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from maple_syrup.pond_network import (
    OutletFace,
    PondNetworkError,
    PondTopology,
    build_pond_topology,
    update_pond_topology,
)
from maple_syrup.routing import GRAVITY_M_S2

DX = 0.5
NODATA = -9999.0
D4 = ((1, 0), (0, 1), (-1, 0), (0, -1))


def topo_from(interior, outlets=(), *, drop=1.0, ring=None, active=None, friction=1.0, allow_closed=None, nodata=None,
              dx=DX) -> PondTopology:
    """Topology of an interior bed. Each outlet `(row, col, d_row, d_col)` gets a ring cell `drop` m below its cell; every other
    ring cell is higher than the whole bed. `ring` overrides `{(full_row, full_col): z}`."""
    z = np.asarray(interior, dtype=np.float64)
    ny, nx = z.shape
    full = np.full((ny + 2, nx + 2), float(z.max()) + 10.0)
    full[1:-1, 1:-1] = z
    faces = [o if isinstance(o, OutletFace) else OutletFace(*o) for o in outlets]
    for o in faces:  # malformed outlets (refusal tests) are passed through untouched for the builder to refuse
        parts = (o.row, o.col, o.d_row, o.d_col)
        well_formed = all(type(v) is int for v in parts)
        if well_formed and 0 <= o.row < ny and 0 <= o.col < nx and 0 <= o.row + o.d_row + 1 < ny + 2 \
                and 0 <= o.col + o.d_col + 1 < nx + 2:
            full[o.row + o.d_row + 1, o.col + o.d_col + 1] = z[o.row, o.col] - drop
    for (r, c), value in (ring or {}).items():
        full[r, c] = value
    if active is None:
        active = np.ones((ny, nx), dtype=bool) if nodata is None else z != nodata
    if allow_closed is None:
        allow_closed = not faces
    return build_pond_topology(full, np.array(active, dtype=bool), np.full((ny, nx), float(friction)), dx, faces,
                               nodata_value=nodata, allow_closed_components=allow_closed)


def bruteforce_minimax(z, active, cells):
    """Independent Bellman-Ford style node-weighted minimax to the outlet cells."""
    ny, nx = z.shape
    level = np.full(z.shape, np.inf)
    for r, c in cells:
        level[r, c] = z[r, c]
    changed = True
    while changed:
        changed = False
        for r in range(ny):
            for c in range(nx):
                if not active[r, c]:
                    continue
                for dr, dc in D4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < ny and 0 <= cc < nx and active[rr, cc] and np.isfinite(level[rr, cc]):
                        cand = max(z[r, c], level[rr, cc])
                        if cand < level[r, c]:
                            level[r, c], changed = cand, True
    return level


# --- geometry, units, immutability ------------------------------------------------------------------------------------
def test_geometry_shapes_units_and_face_counts():
    t = topo_from(np.ones((3, 4)), [(1, 3, 0, 1)], drop=0.5)
    g = t.geometry
    assert t.shape == (3, 4) and t.dx_m == DX
    assert g.fx_type.shape == (3, 5) and g.fy_type.shape == (4, 4)
    assert int(np.count_nonzero(g.fx_type == 1)) == 3 * 3 and int(np.count_nonzero(g.fy_type == 1)) == 2 * 4
    assert int(np.count_nonzero(g.fx_type == 2)) == 1 and int(np.count_nonzero(g.fy_type == 2)) == 0
    assert g.fx_type[1, 4] == 2 and g.fx_sign[1, 4] == 1
    slope = 0.5 / DX  # m/m
    assert g.fx_kb[1, 4] == pytest.approx(math.sqrt(8.0 * GRAVITY_M_S2 * slope / 1.0), rel=1e-14)
    faces = t.interior_faces()
    assert faces["x"][0].size == 9 and faces["y"][0].size == 8
    assert np.all(faces["x"][1] - faces["x"][0] == 1) and np.all(faces["y"][1] - faces["y"][0] == 4)
    assert t.summary()["n_interior_faces"] == 17 and t.summary()["n_outlets"] == 1


def test_arrays_are_read_only_and_not_aliased():
    interior = np.arange(12, dtype=float).reshape(3, 4) / 8.0
    t = topo_from(interior, [(0, 0, 0, -1)])
    for array in (t.source_z_full, t.active, t.friction, t.escape_level_m, t.escape_path_next, t.outlet_connected,
                  t.closed_component_label, t.geometry.z, t.geometry.fx_type, t.source_bed_m, t.fill_depth_m):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[(0,) * array.ndim] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.dx_m = 2.0
    full = np.array(t.source_z_full)
    active, friction = np.ones((3, 4), bool), np.ones((3, 4))
    t2 = build_pond_topology(full, active, friction, DX, [(0, 0, 0, -1)])
    digest = t2.topology_sha256
    full[2, 2] += 5.0
    active[0, 0] = False
    friction[1, 1] = 9.0
    assert t2.topology_sha256 == digest and t2.source_z_full[2, 2] != full[2, 2] and t2.active[0, 0]


def test_digests_are_deterministic_and_sensitive():
    interior = np.arange(12, dtype=float).reshape(3, 4) / 8.0
    a, b = topo_from(interior, [(0, 0, 0, -1)]), topo_from(interior, [(0, 0, 0, -1)])
    assert a.topology_sha256 == b.topology_sha256 and a.identity_sha256 == b.identity_sha256
    moved = interior.copy()
    moved[2, 2] += 0.125
    c = topo_from(moved, [(0, 0, 0, -1)])
    assert c.source_sha256 != a.source_sha256 and c.topology_sha256 != a.topology_sha256
    assert c.identity_sha256 == a.identity_sha256  # the bed is not part of the identity
    assert topo_from(interior, [(0, 1, -1, 0)]).identity_sha256 != a.identity_sha256


# --- nodata and inactive cells ------------------------------------------------------------------------------------------
def test_no_open_face_through_nodata_or_inactive():
    t = topo_from([[1.0, 2.0, NODATA, 2.0, 1.0]], nodata=NODATA)
    g = t.geometry
    assert not t.active[0, 2]
    assert g.fx_type[0, 2] == 0 and g.fx_type[0, 3] == 0  # both faces of the nodata cell
    assert g.fx_type[0, 1] == 1 and g.fx_type[0, 4] == 1
    assert t.n_closed_components == 2 and set(np.unique(t.closed_component_label[t.active])) == {0, 1}
    assert np.all(np.isinf(t.escape_level_m)) and t.closed_component_label[0, 2] == -1
    inactive = topo_from(np.ones((3, 3)), [(0, 0, 0, -1)], active=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool))
    types = np.concatenate([inactive.geometry.fx_type[1, 1:3], inactive.geometry.fy_type[1:3, 1]])
    assert np.all(types == 0)


def test_nodata_refusals():
    z = np.array([[1.0, NODATA, 1.0]])
    with pytest.raises(PondNetworkError, match="nodata"):
        topo_from(z, nodata=NODATA, active=np.ones((1, 3), bool))
    with pytest.raises(PondNetworkError, match="nodata"):
        topo_from([[1.0, 2.0]], [(0, 1, 0, 1)], ring={(1, 3): NODATA}, nodata=NODATA)


# --- explicit outlets ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("outlet, ring, match", [
    ((0, 1, 0, 1), {(1, 3): 1.0}, "drop"),          # zero drop: ring == cell bed
    ((0, 1, 0, 1), {(1, 3): 2.0}, "drop"),          # uphill / backwater
    ((0, 0, 0, 1), None, "interior"),                # leads into the interior
    ((0, 1, 0, 2), None, "D4"),                      # not a unit D4 step
    ((0, 1, 1, 1), None, "D4"),
    ((0, 5, 0, 1), None, "active interior"),         # out of range
])
def test_outlet_refusals(outlet, ring, match):
    with pytest.raises(PondNetworkError, match=match):
        topo_from(np.ones((1, 2)), [outlet], ring=ring)


def test_outlet_entry_types_duplicates_and_inactive():
    with pytest.raises(PondNetworkError, match="int"):
        topo_from(np.ones((1, 2)), [(0, True, 0, 1)])
    with pytest.raises(PondNetworkError, match="int"):
        topo_from(np.ones((1, 2)), [(0.0, 1, 0, 1)])
    with pytest.raises(PondNetworkError, match="twice"):
        topo_from(np.ones((1, 2)), [(0, 1, 0, 1), (0, 1, 0, 1)])
    with pytest.raises(PondNetworkError, match="active interior"):
        topo_from(np.ones((1, 2)), [(0, 1, 0, 1)], active=np.array([[True, False]]))
    with pytest.raises(PondNetworkError):
        build_pond_topology(np.zeros((3, 4)), np.ones((1, 2), bool), np.ones((1, 2)), DX, "east")


def test_flat_interior_with_stepped_outlet_has_no_legacy_direction_need():
    t = topo_from(np.full((3, 3), 2.0), [(1, 2, 0, 1)], drop=0.25)
    assert t.n_closed_components == 0 and np.all(t.escape_level_m == 2.0) and np.all(t.outlet_connected)
    assert np.all(t.fill_depth_m == 0.0)
    assert t.escape_path_next[1, 2] == -1 and t.escape_path_next[1, 1] == 1 * 3 + 2


# --- escape levels -------------------------------------------------------------------------------------------------------
def test_single_cell_lip_is_not_a_multicell_spill_level():
    t = topo_from([[0.5, 3.0, 1.0, 1.0]], [(0, 0, 0, -1)])
    assert np.array_equal(t.escape_level_m[0], [0.5, 3.0, 3.0, 3.0])
    naive = max(1.0, 1.0)  # the neighbour-lip of the last cell
    assert t.escape_level_m[0, 3] != naive
    assert np.array_equal(t.fill_depth_m[0], [0.0, 0.0, 2.0, 2.0])


def test_connected_basins_with_distinct_levels():
    t = topo_from([[0.5, 1.0, 2.0, 0.8, 3.0, 0.1, 0.2]], [(0, 0, 0, -1)])
    assert np.array_equal(t.escape_level_m[0], [0.5, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0])


def test_nested_bowl_levels():
    z = np.full((5, 5), 3.0)
    z[1:4, 1:4] = 2.0
    z[2, 2] = 0.0
    z[0, 2] = 1.0
    t = topo_from(z, [(0, 2, -1, 0)])
    assert t.escape_level_m[0, 2] == 1.0
    assert t.escape_level_m[2, 2] == 2.0 and t.fill_depth_m[2, 2] == 2.0
    assert np.all(t.escape_level_m[1:4, 1:4] == 2.0)
    assert t.escape_level_m[0, 1] == 3.0 and t.escape_level_m[4, 4] == 3.0


@pytest.mark.parametrize("seed", range(5))
def test_escape_matches_independent_bruteforce(seed):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 40, size=(6, 7)) / 8.0
    active = rng.random(z.shape) > 0.15
    outlets = [(0, 0, -1, 0), (5, 6, 0, 1)]
    for r, c, _, _ in outlets:
        active[r, c] = True
    t = topo_from(z, outlets, active=active, allow_closed=True)
    expected = bruteforce_minimax(z, active, [(0, 0), (5, 6)])
    assert np.array_equal(t.escape_level_m, np.where(active, expected, np.inf))
    assert np.array_equal(t.outlet_connected, np.isfinite(t.escape_level_m))
    reached = t.outlet_connected | ~active
    assert np.all((t.closed_component_label >= 0) == ~reached)
    assert np.all(np.isinf(t.escape_level_m[t.active & (t.closed_component_label >= 0)]))


def test_closed_components_need_explicit_designation_and_stay_infinite():
    z = np.array([[1.0, 2.0, 1.0, 5.0, 1.0, 1.0]])
    active = np.array([[1, 1, 1, 0, 1, 1]], dtype=bool)
    with pytest.raises(PondNetworkError, match="closed component"):
        topo_from(z, [(0, 0, 0, -1)], active=active, allow_closed=False)
    t = topo_from(z, [(0, 0, 0, -1)], active=active, allow_closed=True)
    assert t.n_closed_components == 1
    assert np.array_equal(t.escape_level_m[0, :3], [1.0, 2.0, 2.0])
    assert np.all(np.isinf(t.escape_level_m[0, 3:])) and np.all(t.closed_component_label[0, 4:] == 0)
    assert np.all(t.closed_component_label[0, :3] == -1) and np.all(np.isinf(t.fill_depth_m[0, 4:]))
    assert np.all(t.escape_path_next[0, 4:] == -2)
    with pytest.raises(PondNetworkError, match="closed component"):
        topo_from(np.ones((2, 2)), allow_closed=False)  # no outlet at all
    assert topo_from(np.ones((2, 2))).n_closed_components == 1  # default designates a purely closed lake


def test_bed_and_mask_validation():
    ok_active, ok_f = np.ones((2, 2), bool), np.ones((2, 2))
    good = np.zeros((4, 4))
    for bad in (good.astype(np.float32), good.tolist(), np.zeros((2, 4)), np.where(np.eye(4) > 0, np.nan, good),
                np.where(np.eye(4) > 0, np.inf, good)):
        with pytest.raises(PondNetworkError):
            build_pond_topology(bad, ok_active, ok_f, DX, [], allow_closed_components=True)
    for bad_active in (np.ones((2, 2), int), np.ones((3, 3), bool), np.zeros((2, 2), bool)):
        with pytest.raises(PondNetworkError):
            build_pond_topology(good, bad_active, ok_f, DX, [], allow_closed_components=True)
    for bad_f in (np.zeros((2, 2)), -np.ones((2, 2)), np.full((2, 2), np.nan), np.ones((2, 2), np.float32)):
        with pytest.raises(PondNetworkError):
            build_pond_topology(good, ok_active, bad_f, DX, [], allow_closed_components=True)
    for bad_dx in (0.0, -1.0, math.nan, math.inf, True, "1"):
        with pytest.raises(PondNetworkError):
            build_pond_topology(good, ok_active, ok_f, bad_dx, [], allow_closed_components=True)


# --- net-elevation refresh ----------------------------------------------------------------------------------------------
def _base():
    z = np.array([[1.0, 1.5, 2.0], [0.75, 1.25, 1.75], [0.5, 1.0, 1.5]])
    return topo_from(z, [(1, 2, 0, 1)], drop=0.25)


def _bump(t, cell, amount):
    new = np.array(t.source_z_full)
    new[cell[0] + 1, cell[1] + 1] += amount
    return new


def test_below_threshold_returns_the_exact_old_topology():
    t = _base()
    result = update_pond_topology(t, _bump(t, (0, 1), 0.25), 0.5)
    assert not result.refreshed and result.topology is t
    assert result.max_abs_delta_m == 0.25 and result.n_cells_at_or_above == 0
    same = update_pond_topology(t, np.array(t.source_z_full), 0.5)
    assert same.topology is t and same.max_abs_delta_m == 0.0


def test_threshold_is_inclusive_and_is_a_net_per_cell_gate():
    t = _base()
    hit = update_pond_topology(t, _bump(t, (0, 1), 0.5), 0.5)
    assert hit.refreshed and hit.topology is not t and hit.n_cells_at_or_above == 1
    # same-cell deltas that cancel never trigger (the gate sees only the net versus the held source)
    up = update_pond_topology(t, _bump(t, (0, 1), 0.25), 0.5)
    back = update_pond_topology(up.topology, _bump(t, (0, 1), 0.25 - 0.25), 0.5)
    assert not up.refreshed and not back.refreshed and back.max_abs_delta_m == 0.0
    # deltas in DIFFERENT cells neither add nor cancel: only the largest per-cell net counts
    two = _bump(t, (0, 1), 0.375)
    two[3, 3] -= 0.375
    mixed = update_pond_topology(t, two, 0.5)
    assert not mixed.refreshed and mixed.max_abs_delta_m == 0.375
    both = _bump(t, (0, 1), 0.5)
    both[3, 3] -= 0.5
    assert update_pond_topology(t, both, 0.5).n_cells_at_or_above == 2


def test_slow_drift_accumulates_against_the_held_source():
    t = _base()
    flags = [update_pond_topology(t, _bump(t, (0, 0), 0.125 * k), 0.5).refreshed for k in (1, 2, 3, 4)]
    assert flags == [False, False, False, True]


def test_refresh_keeps_identity_and_updates_levels():
    z = np.array([[0.5, 1.0, 0.2, 0.1]])
    t = topo_from(z, [(0, 0, 0, -1)])
    assert np.array_equal(t.escape_level_m[0], [0.5, 1.0, 1.0, 1.0])
    before = (t.topology_sha256, t.source_sha256, np.array(t.geometry.z), np.array(t.escape_level_m))
    result = update_pond_topology(t, _bump(t, (0, 1), 1.0), 0.5)
    n = result.topology
    assert result.refreshed and n.generation == 1 and t.generation == 0
    assert n.identity_sha256 == t.identity_sha256 and n.outlets == t.outlets
    assert np.array_equal(n.active, t.active) and np.array_equal(n.friction, t.friction)
    assert n.source_sha256 != t.source_sha256 and n.topology_sha256 != t.topology_sha256
    assert np.array_equal(n.escape_level_m[0], [0.5, 2.0, 2.0, 2.0])
    assert n.geometry.fx_zmax[0, 1] == 2.0 and t.geometry.fx_zmax[0, 1] == 1.0
    assert (t.topology_sha256, t.source_sha256) == before[:2]
    assert np.array_equal(t.geometry.z, before[2]) and np.array_equal(t.escape_level_m, before[3])
    assert np.array_equal(n.geometry.fx_type, t.geometry.fx_type) and np.array_equal(n.geometry.fx_sign, t.geometry.fx_sign)


def test_refresh_refusals_publish_nothing():
    t = _base()
    digest = t.topology_sha256
    nan = np.array(t.source_z_full)
    nan[2, 2] = np.nan
    inf = np.array(t.source_z_full)
    inf[2, 2] = np.inf
    ring = np.array(t.source_z_full)
    ring[2, 4] += 1.0  # the outlet ring cell
    drop_gone = _bump(t, (1, 2), -0.5)  # outlet cell lowered below its ring cell: no positive drop remains
    for bad in (nan, inf, ring, drop_gone, t.source_z_full[:-1], t.source_z_full.astype(np.float32), t.source_z_full.tolist()):
        with pytest.raises(PondNetworkError):
            update_pond_topology(t, bad, 0.1)
    for bad_threshold in (0.0, -1.0, math.nan, math.inf, True, "0.5", None):
        with pytest.raises(PondNetworkError):
            update_pond_topology(t, np.array(t.source_z_full), bad_threshold)
    with pytest.raises(PondNetworkError):
        update_pond_topology("topology", np.array(t.source_z_full), 0.5)
    assert t.topology_sha256 == digest and t.generation == 0


def test_nonfinite_bed_is_refused_even_below_threshold():
    t = _base()
    bad = np.array(t.source_z_full)
    bad[1, 1] = np.nan
    with pytest.raises(PondNetworkError, match="finite"):
        update_pond_topology(t, bad, 1e6)


def test_inactive_cell_change_never_triggers():
    z = np.array([[1.0, 1.0, 1.0]])
    t = topo_from(z, [(0, 2, 0, 1)], active=np.array([[1, 0, 1]], bool), allow_closed=True)
    new = np.array(t.source_z_full)
    new[1, 2] += 100.0  # inactive cell: not part of the gate
    result = update_pond_topology(t, new, 0.5)
    assert not result.refreshed and result.max_abs_delta_m == 0.0
