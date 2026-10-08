"""Native-walk semantics on tiny networks (pure-Python kernels; every expected value is an independent closed form)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from maple_syrup import legacy_native as N
from maple_syrup.sediment_physics import REGIME_CODES

from .helpers import (
    DX,
    NODATA,
    arrays,
    channel,
    converging,
    make_graph,
    pit_channel,
    run_walk,
)


def only(net, a, cell, nc=2):
    keep = np.zeros_like(a["det"])
    keep[cell] = a["det"][cell]
    return {**a, "det": keep}


def test_network_views_are_separate_and_classified():
    net = N.native_network(pit_channel())
    assert net.terminal_idx.tolist() == [2]
    assert net.walk_next[2] == N.STOP_TERMINAL
    assert net.walk_first[2] == 1  # aspect 0 source starts WEST
    assert net.aspect0[2] and not net.aspect0[1]
    assert net.outlet_idx.size == 0
    # CN donor view: the pit has two donors, neither the pit itself
    assert sorted(net.donors[2][net.donors[2] >= 0].tolist()) == [1, 3]
    assert not np.any(net.donors == 2)
    pos = np.empty(net.active.size, dtype=int)
    pos[net.order] = np.arange(net.order.size)
    assert pos[1] < pos[2] and pos[3] < pos[2]


def test_donor_slot_order_is_south_west_north_east():
    net = N.native_network(converging())
    # the outlet (row 0, col 2) = flat 2 receives from its west (flat 1) and north (flat 5) neighbours
    row = net.donors[2]
    assert row[1] == 1 and row[2] == 5 and row[0] == -1 and row[3] == -1


def test_ordinary_walk_to_ring_closed_form():
    net = N.native_network(channel(5))
    a = only(net, arrays(net, par=1.0), 0)
    depos, ring, inactive_dep, _, counts = run_walk(net, a)
    d = a["det"][0, 0]
    total = depos[:, 0].sum() + ring[0]
    assert total == pytest.approx(d * (1.0 - math.exp(-6.0 * DX)), rel=1e-14)  # source + 4 cells + ring telescopes
    assert depos[0, 0] == pytest.approx(d * (1.0 - math.exp(-1.0)), rel=1e-15)
    assert depos[3, 0] == pytest.approx(d * (math.exp(-3.0) - math.exp(-4.0)), rel=1e-14)
    assert counts[N.C_RING] == 2 and counts[N.C_WALKS] == 2  # two classes, each ends in the ring once
    assert inactive_dep.sum() == 0.0


def test_truncated_tail_and_vge_stops_are_counted():
    net = N.native_network(channel(5))
    a = only(net, arrays(net, par=1.0), 0)
    depos, ring, _, _, counts = run_walk(net, a, limits=np.full(8, 2))
    assert counts[N.C_LIMIT] == 2 and ring.sum() == 0.0 and depos[3].sum() == 0.0 and depos[2].sum() > 0.0
    big = only(net, arrays(net, par=100.0), 0)
    _, ring, _, _, counts = run_walk(net, big)
    assert counts[N.C_VGE] == 2 and ring.sum() == 0.0


def test_terminal_pit_is_credited_once_then_the_walk_stops():
    net = N.native_network(pit_channel())
    a = only(net, arrays(net, par=1.0), 0)
    depos, ring, _, _, counts = run_walk(net, a)
    d = a["det"][0, 0]
    assert depos[1, 0] == pytest.approx(d * (math.exp(-1.0) - math.exp(-2.0)), rel=1e-14)
    assert depos[2, 0] == pytest.approx(d * (math.exp(-2.0) - math.exp(-3.0)), rel=1e-14)
    assert depos[3, 0] == 0.0 and depos[4, 0] == 0.0 and ring.sum() == 0.0
    assert counts[N.C_TERMINAL] == 2 and counts[N.C_RING] == 0 and counts[N.C_ASPECT0] == 0


def test_source_with_aspect_zero_starts_west():
    net = N.native_network(pit_channel())
    a = only(net, arrays(net, par=1.0), 2)  # forced detachment at the pit (zero in a real storm)
    depos, _, _, _, counts = run_walk(net, a)
    d = a["det"][2, 0]
    assert depos[2, 0] == pytest.approx(d * (1.0 - math.exp(-1.0)) + d * (math.exp(-2.0) - math.exp(-3.0)), rel=1e-14)
    assert depos[1, 0] == pytest.approx(d * (math.exp(-1.0) - math.exp(-2.0)), rel=1e-14)
    assert depos[3, 0] == 0.0
    assert counts[N.C_ASPECT0] == 2 and counts[N.C_TERMINAL] == 2


def test_aspect_zero_source_in_column_zero_credits_the_ring_west():
    net = N.native_network(make_graph([[1, 2, 3, 4, 5]]))
    assert net.terminal_idx.tolist() == [0] and net.walk_first[0] == N.RING
    a = only(net, arrays(net, par=1.0), 0)
    _, ring, _, _, counts = run_walk(net, a)
    d = a["det"][0, 0]
    assert ring[0] == pytest.approx(d * (math.exp(-1.0) - math.exp(-2.0)), rel=1e-14)
    assert counts[N.C_RING] == 2 and counts[N.C_ASPECT0] == 2


def test_inactive_cell_is_credited_once_outside_the_active_pools():
    net = N.native_network(make_graph([[NODATA, 1, 2, 3]]))
    assert net.inactive[0] and net.terminal_idx.tolist() == [1] and net.walk_first[1] == 0
    a = only(net, arrays(net, par=1.0), 1)
    depos, ring, inactive_dep, _, counts = run_walk(net, a)
    d = a["det"][1, 0]
    assert inactive_dep[0] == pytest.approx(d * (math.exp(-1.0) - math.exp(-2.0)), rel=1e-14)
    assert depos[0].sum() == 0.0 and ring.sum() == 0.0 and counts[N.C_INACTIVE] == 2


def test_no_law_cases_zero_slope_diffuse_versus_local_deposit():
    net = N.native_network(pit_channel())
    base = arrays(net, par=1.0, law=False, regime="diffuse")
    a = only(net, base, 2)
    depos, _, _, _, counts = run_walk(net, a)
    assert depos.sum() == 0.0 and counts[N.C_ZERO_SLOPE] == 2 and counts[N.C_LOCAL] == 0  # nothing deposited; pool keeps it
    conc = only(net, arrays(net, par=1.0, law=False, regime="concentrated"), 2)
    depos, _, _, _, counts = run_walk(net, conc)
    assert depos[2, 0] == conc["det"][2, 0] and counts[N.C_LOCAL] == 2  # conc_flow_transport 49-54
    sloped = only(net, arrays(net, par=1.0, law=False, regime="diffuse"), 1)
    depos, _, _, _, counts = run_walk(net, sloped)
    assert depos[1, 0] == sloped["det"][1, 0] and counts[N.C_LOCAL] == 2  # sloped cell without a law deposits locally


def test_legacy_order_erasure_discards_earlier_credits_only():
    net = N.native_network(channel(5))
    n = net.active.size
    a = arrays(net, par=1.0)
    a["det"][:] = 0.0
    a["det"][0] = 1.0
    erase = np.zeros(n, dtype=bool)
    erase[2] = True
    legacy = net.source_order_legacy
    assert legacy.tolist() == [0, 1, 2, 3, 4]  # one row: west to east
    kept, _, _, erased0, c0 = run_walk(net, a, erase=erase, erase_on=False, order=legacy)
    cut, _, _, erased1, _ = run_walk(net, a, erase=erase, erase_on=True, order=legacy)
    assert erased0.sum() == 0.0 and kept[2].sum() > 0.0 and c0[N.C_ERASE_CELLS] == 1
    assert cut[2].sum() == 0.0 and erased1[0] == pytest.approx(kept[2, 0], rel=1e-15)
    # reversed order: the erase cell is processed BEFORE the source credits it, so nothing is lost
    rev, _, _, erased2, _ = run_walk(net, a, erase=erase, erase_on=True, order=legacy[::-1])
    assert erased2.sum() == 0.0 and rev[2, 0] == pytest.approx(kept[2, 0], rel=1e-15)


def test_source_order_changes_only_summation_order():
    net = N.native_network(converging())
    a = arrays(net, par=0.7)
    d1 = run_walk(net, a)[0]
    d2 = run_walk(net, a, order=net.source_order_legacy)[0]
    np.testing.assert_allclose(d1, d2, rtol=1e-14, atol=0.0)


def test_mixed_regime_limits_follow_the_source_regime():
    lim = N.walk_limits(1.0)
    assert lim[REGIME_CODES["diffuse"]] == 10 and lim[REGIME_CODES["concentrated"]] == 100 and lim[REGIME_CODES["suspended"]] == 500
