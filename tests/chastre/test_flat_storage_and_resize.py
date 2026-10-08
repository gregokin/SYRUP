"""Flat-sink storage option of the routing builder, the RunGuard bed-digest hook and the deterministic rainfall-scaling gap
fill / resize. Pure host tests (no MAPLE compile). Nothing here was run by its author (file-only tools)."""
from types import SimpleNamespace

import numpy as np
import pytest

from maple_syrup.routing import RoutingGraphError, build_routing_graph

NODATA = -9999.0


def _graph(interior, **kwargs):
    ny, nx = interior.shape
    full = np.full((ny + 2, nx + 2), NODATA)
    full[1:-1, 1:-1] = interior
    ring = np.ones(full.shape, dtype=np.bool_)
    ring[1:-1, 1:-1] = False
    return build_routing_graph(full, ring, np.full((ny, nx), 40.0), 1.0, active_mask=np.ones((ny, nx), dtype=np.bool_),
                               nodata_value=NODATA, allow_masked_nodata=True, **kwargs)


# 4 flat sinks (the 2 x 2 plateau at the low corner) + 1 strict pit (2, 2); nothing reaches the nodata ring
INTERIOR = np.array([[10.0, 10.0, 13.0], [10.0, 10.0, 12.0], [14.0, 13.0, 11.5]])


def test_default_and_pit_only_still_refuse_flat_sinks_with_the_historical_error():
    for kwargs in ({}, {"allow_pit_storage": True}):
        with pytest.raises(RoutingGraphError, match="unsupported sinks") as info:
            _graph(INTERIOR, **kwargs)
        assert "flats (r=0, c=0" in str(info.value)


def test_flat_flag_requires_pit_storage():
    with pytest.raises(RoutingGraphError, match="allow_flat_storage"):
        _graph(INTERIOR, allow_flat_storage=True)


def test_flat_storage_keeps_flat_and_strict_sinks_as_zero_conveyance_storage_without_outlets():
    graph = _graph(INTERIOR, allow_pit_storage=True, allow_flat_storage=True)
    assert graph.policy == "masked_nodata=True;pit_storage=True;flat_storage=True"
    assert int(graph.pit_storage.sum()) == 5 and int(graph.flat_storage.sum()) == 4
    assert graph.pit_storage[2, 2] and not graph.flat_storage[2, 2]  # the strict pit
    assert int(graph.outlet.sum()) == 0
    assert np.all(graph.aspect[graph.pit_storage] == 0) and np.all(graph.slope[graph.pit_storage] == 0.0)
    assert np.all(np.asarray(graph.conveyance).reshape(3, 3)[graph.pit_storage] == 0.0)
    summary = graph.summary()
    assert summary["n_pit_storage"] == 5 and summary["n_flat_storage"] == 4 and summary["n_strict_pit_storage"] == 1
    # a non-sink cell still routes to its strictly lower neighbour (cell (0, 2) drains west into the plateau)
    assert graph.aspect[0, 2] != 0 and not graph.pit_storage[0, 2]


def test_default_policy_string_and_summary_keys_are_unchanged_without_the_new_flag():
    strict_pit_only = np.array([[10.5, 11.0, 12.0], [11.0, 11.5, 12.5], [12.0, 12.5, 13.0]])
    graph = _graph(strict_pit_only, allow_pit_storage=True)
    assert graph.policy == "masked_nodata=True;pit_storage=True" and graph.flat_storage is None
    assert "n_flat_storage" not in graph.summary()
    flagged = _graph(strict_pit_only, allow_pit_storage=True, allow_flat_storage=True)
    assert flagged.input_sha256 != graph.input_sha256  # the new policy changes the digest ONLY when requested
    assert np.array_equal(flagged.aspect, graph.aspect) and np.array_equal(flagged.pit_storage, graph.pit_storage)


def test_flat_cell_with_a_lower_neighbour_is_not_a_sink():
    interior = np.array([[10.0, 11.0, 11.5], [11.0, 12.0, 12.0], [12.0, 12.5, 12.5]])
    graph = _graph(interior, allow_pit_storage=True, allow_flat_storage=True)
    assert int(graph.pit_storage.sum()) == 1 and graph.pit_storage[0, 0]  # only the strict pit
    # (1, 1) and (1, 2) are equal neighbours of each other but each has a strictly lower neighbour: they route, not store
    assert not graph.pit_storage[1, 1] and graph.aspect[1, 1] != 0
    assert not graph.pit_storage[1, 2] and graph.aspect[1, 2] != 0


def test_runguard_bed_digest_hook_is_additive(monkeypatch):
    import compare_plot1 as cmp

    from maple_syrup import column_experiment as ce

    monkeypatch.setattr(ce, "_bed_digest", lambda case: "default-digest")
    monkeypatch.setattr(ce, "_source_digests", lambda *dirs: {"s": "x"})
    inputs = SimpleNamespace(case=object())
    default = cmp.RunGuard(inputs, ("a", "b"), "default")
    assert default.bed_before == "default-digest" and default._custom_bed is False
    custom = cmp.RunGuard(inputs, ("a", "b"), "custom", bed_digest=lambda case: "persisted-digest")
    assert custom.bed_before == "persisted-digest" and custom._custom_bed is True


# --------------------------------------------------------------------------------------------------------------------
# gap fill and nearest-pixel-centre resize
# --------------------------------------------------------------------------------------------------------------------
def test_nearest_index_is_pixel_centre_exact_and_in_range():
    from maple_syrup.chastre_case import nearest_index

    assert nearest_index(4, 8).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    # target centres 1/4 and 3/4 of the extent are EXACT source boundaries 1 and 3 (4 source pixels): the higher index is taken
    assert nearest_index(4, 2).tolist() == [1, 3]
    # the RFID rows 104 -> 1393 have exactly one exact tie (the centre row 696); the columns 58 -> 1604 have none
    rows, cols = nearest_index(104, 1393), nearest_index(58, 1604)
    assert np.count_nonzero((2 * np.arange(1393) + 1) * 104 % (2 * 1393) == 0) == 1 and rows[696] == 52
    assert np.count_nonzero((2 * np.arange(1604) + 1) * 58 % (2 * 1604) == 0) == 0 and cols.max() == 57
    for n_src, n_dst in ((104, 1393), (58, 1604), (5, 3), (3, 5), (1, 7)):
        index = nearest_index(n_src, n_dst)
        assert index.min() >= 0 and index.max() <= n_src - 1 and np.all(np.diff(index) >= 0)


def test_resize_uses_only_source_values_and_is_deterministic():
    from maple_syrup.chastre_case import nearest_resize

    source = np.arange(1.0, 13.0).reshape(3, 4)
    resized, rows, cols = nearest_resize(source, (7, 9))
    assert resized.shape == (7, 9) and set(np.unique(resized)) <= set(source.ravel())
    assert np.array_equal(resized, source[rows][:, cols])
    # rows 3 -> 7 have no exact tie, so this particular map is mirror-symmetric (a tie, e.g. 4 -> 2, would break it)
    flipped, _, _ = nearest_resize(source[::-1], (7, 9))
    assert np.array_equal(flipped, resized[::-1])
    again, _, _ = nearest_resize(source, (7, 9))
    assert np.array_equal(again, resized)


def test_gap_fill_nearest_valid_with_deterministic_row_major_ties():
    from maple_syrup.chastre_case import fill_source_gaps

    scale = np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0], [7.0, 0.0, 9.0]])
    valid = scale > 0.0
    filled, record = fill_source_gaps(scale, valid)
    assert np.array_equal(filled[valid], scale[valid])  # valid cells untouched
    assert filled[0, 1] == 1.0  # equidistant from (0,0)=1 and (0,2)=3: the first in row-major order wins
    assert filled[1, 0] == 1.0  # equidistant from (0,0)=1 and (2,0)=7: first in row-major order
    assert filled[1, 1] == 1.0  # all four valid cells are at squared distance 2: the first in row-major order, (0, 0)
    assert filled[1, 2] == 3.0  # (0,2) d2=1, (2,2) d2=1: first in row-major order
    assert filled[2, 1] == 7.0
    assert record["n_filled"] == 5 and record["n_filled_with_distance_ties"] == 5
    assert set(np.unique(filled)) <= set(scale[valid])  # new values come only from valid cells
    again, _ = fill_source_gaps(scale, valid)
    assert np.array_equal(again, filled)


def test_gap_fill_refuses_without_any_valid_cell():
    from maple_syrup.case_import import Plot1ImportError
    from maple_syrup.chastre_case import fill_source_gaps

    with pytest.raises(Plot1ImportError):
        fill_source_gaps(np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))
