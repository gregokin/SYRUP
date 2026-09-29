"""Fast Phase 2 checks that need neither MAHLERAN data nor a MAPLE compile."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import RECIPE_PATH

from maple_syrup.case_import import (
    Plot1ImportError,
    apply_legacy_pavement_rescaling,
    legacy_d4_audit,
    load_recipe,
    normalize_closure_roundoff,
    plan_bed,
    resolve_particle_size_maps,
    stage_legacy_ascii,
)


def _fractions(*rows):
    return np.array(rows, dtype=np.float64).reshape(1, len(rows), 6)


# --- closure ------------------------------------------------------------------
def test_roundoff_is_normalized_and_reported():
    raw = _fractions([0.2, 0.2, 0.2, 0.2, 0.1, 0.1 + 4e-7], [0.1, 0.1, 0.1, 0.1, 0.3, 0.3 - 3e-7])
    normalized, stats = normalize_closure_roundoff(raw, 1e-6)
    assert np.abs(normalized.sum(axis=-1) - 1.0).max() <= 1e-15
    np.testing.assert_allclose(normalized, raw / raw.sum(axis=-1, keepdims=True), rtol=0, atol=0)
    assert 3e-7 <= stats["max_abs_sum_deviation"] <= 4.1e-7
    assert max(stats["max_abs_change_by_class"]) < 1e-6


@pytest.mark.parametrize(
    "row",
    [
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],  # sums to 1.2: material nonclosure
        [0.2, 0.2, 0.2, 0.2, 0.1, 0.1 + 2e-6],  # just outside the tolerance
        [0.3, 0.3, 0.2, 0.2, 0.1, -0.1],  # closes, but negative
        [0.2, 0.2, 0.2, 0.2, 0.1, np.nan],
    ],
)
def test_nonclosing_negative_or_nan_fractions_are_rejected(row):
    with pytest.raises(Plot1ImportError):
        normalize_closure_roundoff(_fractions(row), 1e-6)


# --- legacy pavement rescaling --------------------------------------------------
def _legacy_cell(ps, pave_percent):
    """Direct transcription of MAHLERAN_storm_setting_xml.f90 352-369 for one cell."""
    grav = ps[4] + ps[5]
    fines = 1.0 - grav
    p = pave_percent / 100.0
    if p <= 0.0 or grav == 0.0:
        return list(ps)
    return [ps[k] * ((1.0 - p) / fines) if k < 4 else ps[k] * (p / grav) for k in range(6)]


def test_pavement_rescaling_matches_legacy_formula_and_closes():
    closed = normalize_closure_roundoff(
        _fractions([0.2, 0.25, 0.15, 0.1, 0.2, 0.1], [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
                   [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]), 1e-6)[0]
    pave_percent = np.array([[35.0, 0.0, 80.0]])
    out, stats = apply_legacy_pavement_rescaling(closed, pave_percent / 100.0)
    for c in range(3):
        expected = _legacy_cell(closed[0, c], pave_percent[0, c])
        np.testing.assert_allclose(out[0, c], expected, rtol=1e-15, atol=1e-17)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=0, atol=1e-15)
    assert out[0, 0, 4] + out[0, 0, 5] == pytest.approx(0.35, abs=1e-15)  # gravel = cover
    np.testing.assert_array_equal(out[0, 1], closed[0, 1])  # pave = 0: unchanged
    np.testing.assert_array_equal(out[0, 2], closed[0, 2])  # grav = 0: unchanged
    assert stats["n_cells_rescaled"] == 1


def test_pavement_rescaling_rejects_legacy_division_by_zero_and_bad_cover():
    all_gravel = _fractions([0, 0, 0, 0, 0.5, 0.5])
    with pytest.raises(Plot1ImportError, match="divide by zero"):
        apply_legacy_pavement_rescaling(all_gravel, np.array([[0.2]]))
    ordinary = _fractions([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    with pytest.raises(Plot1ImportError):
        apply_legacy_pavement_rescaling(ordinary, np.array([[1.2]]))
    with pytest.raises(Plot1ImportError):
        apply_legacy_pavement_rescaling(ordinary, np.array([[-0.1]]))


# --- grain-map selection ----------------------------------------------------------
def test_repeated_xml_maps_are_reported_and_a_repeated_selection_is_rejected():
    repeated = ("plot1_phi1.asc",) * 6
    distinct = tuple(f"plot1_phi{k}.asc" for k in range(1, 7))
    record = resolve_particle_size_maps(repeated, distinct)
    assert record["xml_references_distinct"] is False
    assert record["xml_reference_counts"] == {"plot1_phi1.asc": 6}
    assert record["correction_applied"] is True
    with pytest.raises(Plot1ImportError, match="six distinct"):
        resolve_particle_size_maps(repeated, repeated)


# --- terrain audit ----------------------------------------------------------------
def _d4_grid(corner):
    z = np.array(
        [
            [9, 9, 9, 9, 9],
            [9, corner, 5, 9, 9],
            [9, 6, 4, 6, 9],
            [9, 7, 3, 8, 9],
            [9, 9, 1, 9, 9],
        ],
        dtype=np.float64,
    )
    rmask = np.ones_like(z)
    rmask[-1, :] = -9999.0
    return z, rmask


def test_d4_audit_follows_legacy_rules():
    z, rmask = _d4_grid(corner=5.0)  # corner equals its east neighbour: flat sink
    arrays, summary = legacy_d4_audit(z, rmask, -9999.0)
    expected_aspect = np.array([[0, 3, 4], [2, 3, 4], [2, 3, 4]])
    np.testing.assert_array_equal(arrays["aspect"], expected_aspect)
    assert arrays["flat_sink"][0, 0] and not arrays["strict_pit"][0, 0]
    assert summary["n_edge_outflow_cells_by_side"] == {"N": 0, "E": 0, "S": 1, "W": 0}
    assert summary["n_cells_ending_in_ring_by_side"]["S"] == 8
    assert summary["n_cells_ending_in_masked_ring"] == 8
    assert summary["n_cells_ending_in_sinks"] == 1
    assert arrays["edge_outflow_side"][2, 1] == 3


def test_d4_audit_strict_pit():
    z, rmask = _d4_grid(corner=4.5)
    arrays, summary = legacy_d4_audit(z, rmask, -9999.0)
    assert arrays["strict_pit"][0, 0] and not arrays["flat_sink"][0, 0]
    assert summary["sinks"][0]["kind"] == "strict_pit"
    assert summary["sinks"][0]["legacy_row_1based"] == 2
    assert summary["sinks"][0]["cells_draining_here"] == 1


# --- staging ------------------------------------------------------------------------
_HEADER = b"ncols 3\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 0.5\nnodata_value -9999\n"
_ROWS = b"1 2 3\n4 5 6\n"


def test_staging_drops_a_binary_trailer_and_reports_it(tmp_path):
    source = tmp_path / "dem.asc"
    source.write_bytes(_HEADER + _ROWS + b"\x00\xff\x10 dem.asc application/octet-stream")
    record = stage_legacy_ascii(source, tmp_path / "out" / "dem.asc")
    assert (tmp_path / "out" / "dem.asc").read_bytes() == _HEADER + _ROWS
    assert record["trailer"]["length_bytes"] == 36
    assert record["trailer"]["mentions_own_file_name"] is True
    assert record["staged_is_prefix_of_original"] and not record["staged_is_byte_identical"]


def test_staging_copies_a_clean_file_verbatim(tmp_path):
    source = tmp_path / "phi.asc"
    source.write_bytes(_HEADER + _ROWS + b"\n")
    record = stage_legacy_ascii(source, tmp_path / "out" / "phi.asc")
    assert record["trailer"] is None and record["staged_is_byte_identical"]


@pytest.mark.parametrize("body", [b"1 2 3\n", b"1 2\n3 4 5 6\n"])
def test_staging_rejects_short_or_wrapped_grids(tmp_path, body):
    source = tmp_path / "bad.asc"
    source.write_bytes(_HEADER + body)
    with pytest.raises(Plot1ImportError):
        stage_legacy_ascii(source, tmp_path / "out" / "bad.asc")


def test_staging_never_overwrites(tmp_path):
    source = tmp_path / "phi.asc"
    source.write_bytes(_HEADER + _ROWS)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "phi.asc").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        stage_legacy_ascii(source, tmp_path / "out" / "phi.asc")
    assert (tmp_path / "out" / "phi.asc").read_bytes() == b"existing"


# --- recipe and bed plan --------------------------------------------------------------
def test_bed_plan_declares_offset_and_allocation():
    recipe = load_recipe(RECIPE_PATH)
    z = np.array([[0.04, 1.5], [0.7, 1.0]])
    plan = plan_bed(z, recipe)
    assert plan.datum_offset_m == pytest.approx(0.26, abs=1e-12)
    assert plan.elevation_m.min() >= recipe.minimum_fill_depth_m - 1e-12
    np.testing.assert_array_equal(plan.elevation_m, z + plan.datum_offset_m)
    assert plan.nz == 20 and plan.vertical_extent_m == pytest.approx(2.0)
    assert plan.headroom_range_m[0] >= recipe.headroom_m - 1e-9


@pytest.mark.parametrize(
    "old,new",
    [
        ("headroom_m: 0.2", "headroom_m: 0.2\n  extra_key: 1"),
        ("interior_cols_1based: [2, 21]", "interior_cols_1based: [1, 22]"),
        ("closure_roundoff_tolerance: 1.0e-6", "closure_roundoff_tolerance: 0.01"),
        ("pavement_rescaling: legacy_storm_setting", "pavement_rescaling: something_else"),
    ],
)
def test_recipe_is_validated_strictly(tmp_path, old, new):
    text = RECIPE_PATH.read_text(encoding="utf-8")
    assert old in text
    bad = tmp_path / "recipe.yaml"
    bad.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(Plot1ImportError):
        load_recipe(bad)


def test_recipe_round_trip():
    recipe = load_recipe(RECIPE_PATH)
    assert recipe.interior_shape == (60, 20)
    assert recipe.corrected_maps == tuple(f"plot1_phi{k}.asc" for k in range(1, 7))
    assert dataclasses.replace(recipe, headroom_m=0.3).headroom_m == 0.3
