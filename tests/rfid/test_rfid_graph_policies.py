"""Opt-in graph policies (masked nodata, terminal pit storage): strict defaults are preserved, the policies behave as documented,
and the prepared/CUDA validators accept the new terminal receiver only in its safe form. Nothing here was run by its author."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("maple")

from rfid_helpers import (
    DX,
    NODATA,
    build,
    chain_full,
    pit_chain,
    south_export,
)

from maple_syrup import routing
from maple_syrup.routing import (
    EXPORT,
    PIT_STORAGE,
    RoutingGraphError,
    build_routing_graph,
)


def test_strict_default_still_refuses_a_pit_with_the_historical_message():
    with pytest.raises(RoutingGraphError, match=r"unsupported sinks \(no strictly lower D4 neighbour; no filling, carving or pit storage\)"):
        build(pit_chain())


def test_strict_default_still_refuses_a_nodata_sentinel():
    z = chain_full(5)
    z[0, 0] = NODATA  # a ring corner
    with pytest.raises(RoutingGraphError, match="nodata"):
        build(z, nodata_value=NODATA)


def test_policy_flags_default_off_give_the_identical_graph_and_digest():
    z = chain_full(5)
    a, b = build(z), build(z, allow_masked_nodata=False, allow_pit_storage=False)
    assert a.input_sha256 == b.input_sha256 and a.policy == b.policy == "strict"
    assert not np.any(a.pit_storage) and "policy" not in a.summary()
    c = build(z, allow_pit_storage=True)  # same terrain, other policy: the digest must say so
    assert c.input_sha256 != a.input_sha256 and c.policy != "strict"
    assert np.array_equal(a.receiver, c.receiver) and np.array_equal(a.slope, c.slope)


def test_pit_storage_is_a_terminal_receiver_distinct_from_export_and_inactive():
    g = build(pit_chain(), allow_pit_storage=True)
    pit = np.asarray(g.pit_storage)
    assert pit.sum() == 1
    r, c = map(int, np.argwhere(pit)[0])
    assert (r, c) == (2, 0)  # interior row index 2 = third cell from the south
    assert g.receiver[r, c] == PIT_STORAGE and PIT_STORAGE not in (EXPORT, routing.INACTIVE)
    assert g.aspect[r, c] == 0 and g.slope[r, c] == 0.0 and float(np.asarray(g.conveyance).reshape(g.shape)[r, c]) == 0.0
    assert not g.outlet[r, c] and g.active[r, c]
    assert g.summary()["n_pit_storage"] == 1
    # it receives the donor above it and the cell below it still drains to the export
    flat = r * g.shape[1] + c
    assert np.asarray(g.receiver)[r + 1, c] == flat
    assert g.receiver[0, 0] == EXPORT


def test_flat_sinks_are_still_refused_even_with_pit_storage():
    z = pit_chain()
    z[2, 1] = z[3, 1]  # an EQUAL neighbour above the pit: a flat sink
    with pytest.raises(RoutingGraphError, match="flat-sink storage"):
        build(z, allow_pit_storage=True)


def test_masked_nodata_neighbour_is_never_a_receiver():
    z = chain_full(5)
    z[3, 2] = NODATA  # the east ring neighbour of interior row 3 (wall, inactive-like): would be a FALSE LOW receiver
    with pytest.raises(RoutingGraphError, match="nodata"):
        build(z, nodata_value=NODATA)  # strict: refused
    g = build(z, nodata_value=NODATA, allow_masked_nodata=True)
    assert g.aspect[2, 0] == 3 and g.receiver[2, 0] == 1 * g.shape[1]  # still drains south, not into the sentinel


def test_masked_nodata_needs_the_sentinel_and_refuses_an_active_nodata_cell():
    z = chain_full(5)
    with pytest.raises(RoutingGraphError, match="needs the nodata_value"):
        build(z, allow_masked_nodata=True)
    z[2, 1] = NODATA
    with pytest.raises(RoutingGraphError, match="ACTIVE cell holds the nodata"):
        build(z, nodata_value=NODATA, allow_masked_nodata=True)


def test_inactive_masked_cells_are_not_donors_or_receivers():
    z = chain_full(5)
    # 2 interior columns; the second is inactive nodata, the first a normal chain with high walls
    wide = np.full((7, 4), NODATA)
    wide[:, 0] = z[:, 0] + 1.0
    wide[:, 1] = z[:, 1]
    wide[:, 3] = z[:, 2] + 1.0
    active2 = np.zeros((5, 2), dtype=bool)
    active2[:, 0] = True
    g = build_routing_graph(wide, south_export(wide), np.ones((5, 2)), DX, active_mask=active2, nodata_value=NODATA,
                            allow_masked_nodata=True)
    assert g.n_active == 5 and not g.active[:, 1].any() and np.all(g.receiver[:, 1] == routing.INACTIVE)
    assert np.all(g.aspect[:, 0] == 3) and g.outlet[0, 0] and g.outlet.sum() == 1


def _forge(graph, **changes):
    return dataclasses.replace(graph, **changes)


def test_prepared_hydrology_accepts_pit_storage_and_refuses_unsafe_terminals():
    from maple_syrup import hydrology_numba as hn

    if not hn.numba_available():
        pytest.skip("Numba not installed; prepared hydrology not exercised")
    from cand_cases import make_params, param_arrays

    g = build(pit_chain(), allow_pit_storage=True)
    params = make_params(param_arrays(g.shape, ksat=1e-6), np.asarray(g.active))
    ctx = hn.prepare_hydrology(g, params)
    assert ctx.n_active == 6
    bad = np.array(g.receiver, copy=True)
    bad[0, 0] = -4  # neither a cell, EXPORT nor PIT_STORAGE
    bad.flags.writeable = False
    with pytest.raises(hn.HydrologyPreparationError, match="no receiver"):
        hn.prepare_hydrology(_forge(g, receiver=bad), params)
    k = np.array(g.conveyance, copy=True)
    k[2 * g.shape[1]] = 1.0  # the pit gets a non-zero conveyance
    klo = np.array(g.conveyance_lo, copy=True)
    klo[np.flatnonzero(np.asarray(g.level_order) == 2 * g.shape[1])] = 1.0
    with pytest.raises(hn.HydrologyPreparationError, match="zero conveyance"):
        hn.prepare_hydrology(_forge(g, conveyance=k, conveyance_lo=klo), params)
    # the terminal code must agree with the declared pit mask, the policy and aspect 0 / slope 0
    with pytest.raises(hn.HydrologyPreparationError, match="inconsistent"):
        hn.prepare_hydrology(_forge(g, policy="strict"), params)
    wrong_mask = np.zeros(g.shape, dtype=bool)
    wrong_mask.flags.writeable = False
    with pytest.raises(hn.HydrologyPreparationError, match="inconsistent"):
        hn.prepare_hydrology(_forge(g, pit_storage=wrong_mask), params)
    aspect = np.array(g.aspect, copy=True)
    aspect[2, 0] = 3
    aspect.flags.writeable = False
    with pytest.raises(hn.HydrologyPreparationError, match="inconsistent"):
        hn.prepare_hydrology(_forge(g, aspect=aspect), params)
    slope = np.array(g.slope, copy=True)
    slope[2, 0] = 0.1
    slope.flags.writeable = False
    with pytest.raises(hn.HydrologyPreparationError, match="inconsistent"):
        hn.prepare_hydrology(_forge(g, slope=slope), params)
