"""Plot 1 audit against the real MAHLERAN inputs (read-only), through MAPLE's
importer, checked with an independent test-only parse. No MAPLE compile."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
from conftest import (
    PLOT1_DIR,
    RECIPE_PATH,
    independent_fractions,
    interior_maple,
    read_legacy_grid,
    write_legacy_grid,
)

from maple_syrup.case_import import Plot1ImportError, audit_plot1, load_recipe


@pytest.fixture(scope="module")
def audit(tmp_path_factory, mahleran_root):
    recipe = load_recipe(RECIPE_PATH, mahleran_root=mahleran_root)
    case_dir = tmp_path_factory.mktemp("plot1_audit") / "case"
    case_dir.mkdir()
    return audit_plot1(recipe, case_dir)


def _brute_force_legacy_aspect(z_m: np.ndarray) -> np.ndarray:
    """Cell-by-cell transcription of topog_attrib.for 94-114 (ndirn = 4)."""
    z = z_m * 1000.0
    nr, nc = z.shape
    aspect = np.zeros((nr - 2, nc - 2), dtype=int)
    for i in range(1, nr - 1):
        for k in range(1, nc - 1):
            zmin, code = z[i, k], 0
            for j, (di, dk) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1)), start=1):
                if z[i + di, k + dk] < zmin:
                    code, zmin = j, z[i + di, k + dk]
            aspect[i - 1, k - 1] = code
    return aspect


def test_dem_crop_and_orientation(audit):
    dem = read_legacy_grid(PLOT1_DIR / "p1dem.asc")
    assert dem.shape == (62, 22)
    np.testing.assert_array_equal(audit.elevation_source_m, interior_maple(dem))
    # MAPLE row 0 (south) is legacy row 61; the plot descends from north to south.
    np.testing.assert_array_equal(audit.elevation_source_m[0], dem[60, 1:21])
    np.testing.assert_array_equal(audit.elevation_source_m[-1], dem[1, 1:21])
    assert audit.elevation_source_m[-1].mean() > audit.elevation_source_m[0].mean()


def test_surface_fields_share_the_crop(audit):
    f = audit.fields
    for name, file, scale in (
        ("vegetation_cover_fraction", "p1vegcover.asc", 0.01),
        ("pavement_cover_fraction", "p1pavcoverveg.asc", 0.01),
        ("saturated_soil_moisture", "thetasat39.asc", 1.0),
        ("rainfall_scaling", "rm_new.asc", 1.0),
    ):
        expected = interior_maple(read_legacy_grid(PLOT1_DIR / file)) * scale
        assert f[name].shape == (60, 20)
        np.testing.assert_allclose(f[name], expected, rtol=1e-15, atol=0)
    stype = interior_maple(read_legacy_grid(PLOT1_DIR / "p1_surface_type.asc"))
    np.testing.assert_array_equal(f["surface_type_raw"], stype)
    assert set(np.unique(f["surface_type_resolved"])) == {1}  # one surface type configured


def test_ring_cover_reports_preserve_percent_units(audit):
    for field, filename in (
        ("pavement_percent", "p1pavcoverveg.asc"),
        ("vegetation_percent", "p1vegcover.asc"),
    ):
        raw = read_legacy_grid(PLOT1_DIR / filename)
        ring = audit.report["grid"]["ring_evidence"][field]
        for side, values in (
            ("N", raw[0, 1:-1]), ("S", raw[-1, 1:-1]),
            ("W", raw[1:-1, 0]), ("E", raw[1:-1, -1]),
        ):
            np.testing.assert_allclose(
                ring[side]["ring_value_range"], [values.min(), values.max()],
                rtol=1e-14, atol=0,
            )


def test_staged_sources_are_hash_bound(audit):
    staged = audit.report["staged_sources"]
    for name, record in staged.items():
        original = (PLOT1_DIR / name).read_bytes()
        assert record["original_sha256"] == hashlib.sha256(original).hexdigest()
        staged_bytes = (audit.case_dir / record["relpath"]).read_bytes()
        assert record["staged_sha256"] == hashlib.sha256(staged_bytes).hexdigest()
        assert original.startswith(staged_bytes)
        assert record["staged_is_byte_identical"] == (record["trailer"] is None)
    dem = staged["p1dem.asc"]
    assert dem["trailer"] is not None, "p1dem.asc is known to carry bytes after its 62 rows"
    rain = audit.report["rainfall"]
    assert rain["sha256"] == hashlib.sha256((PLOT1_DIR / rain["file"]).read_bytes()).hexdigest()


def test_xml_grain_map_defect_is_reported(audit):
    gm = audit.report["grain_maps"]
    assert gm["xml_references"] == ["plot1_phi1.asc"] * 6
    assert gm["xml_references_distinct"] is False and gm["correction_applied"] is True
    assert gm["selected_maps"] == [f"plot1_phi{k}.asc" for k in range(1, 7)]
    phi1 = interior_maple(read_legacy_grid(PLOT1_DIR / "plot1_phi1.asc"))
    low, high = gm["xml_as_configured"]["raw_sum_range"]
    assert low == pytest.approx(6.0 * phi1.min(), rel=1e-14)
    assert high == pytest.approx(6.0 * phi1.max(), rel=1e-14)
    assert 1.14 < low < high < 1.40
    assert gm["xml_as_configured"]["n_cells_post_setup_not_closed_1e-6"] > 0


def test_composition_matches_independent_source_formula(audit):
    raw, normalized, final = independent_fractions(PLOT1_DIR)
    sums = raw.sum(axis=-1)
    # Roundoff only (Codex's independent check: 0.99999992549..1.00000007078).
    assert np.abs(sums - 1.0).max() < 1e-7
    assert sums.min() == pytest.approx(0.9999999254941929, abs=1e-12)
    assert sums.max() == pytest.approx(1.0000000707805157, abs=1e-12)
    np.testing.assert_allclose(audit.fields["phi_raw_selected"], raw, rtol=0, atol=0)
    np.testing.assert_allclose(audit.fields["phi_normalized"], normalized, rtol=0, atol=1e-16)
    np.testing.assert_allclose(audit.final_fractions, final, rtol=0, atol=1e-15)
    # Normalization roundoff and the pavement model choice are reported apart.
    gm = audit.report["grain_maps"]
    assert max(gm["roundoff_normalization"]["max_abs_change_by_class"]) < 1e-7
    assert max(gm["pavement_rescaling"]["max_abs_change_by_class"]) > 1e-3
    changed = np.any(final != normalized, axis=-1)
    assert gm["pavement_rescaling"]["n_cells_rescaled"] >= int(changed.sum())
    assert gm["final_max_abs_sum_deviation"] <= 1e-12
    table, codes = audit.composition_table, audit.composition_codes
    np.testing.assert_array_equal(table[codes - 1], audit.final_fractions)


def test_terrain_audit_matches_brute_force_legacy_d4(audit):
    dem = read_legacy_grid(PLOT1_DIR / "p1dem.asc")
    aspect = _brute_force_legacy_aspect(dem)
    np.testing.assert_array_equal(audit.fields["legacy_d4_aspect"], aspect[::-1])
    summary = audit.report["terrain"]["d4_audit"]
    assert summary["n_sinks"] == int((aspect == 0).sum())
    edge = {
        "N": int((aspect[0] == 1).sum()), "S": int((aspect[-1] == 3).sum()),
        "W": int((aspect[:, 0] == 4).sum()), "E": int((aspect[:, -1] == 2).sum()),
    }
    assert summary["n_edge_outflow_cells_by_side"] == edge
    ending = summary["n_cells_ending_in_sinks"] + sum(summary["n_cells_ending_in_ring_by_side"].values())
    assert ending == 1200
    ring = audit.report["grid"]["ring_evidence"]["dem_m"]
    np.testing.assert_allclose(ring["N"]["ring_minus_adjacent_interior_range"],
                               [np.min(dem[0, 1:-1] - dem[1, 1:-1]), np.max(dem[0, 1:-1] - dem[1, 1:-1])])
    assert np.all(dem[-1, 1:-1] < dem[-2, 1:-1]), "south ring lower than the last interior row"
    rm = read_legacy_grid(PLOT1_DIR / "rm_new.asc")
    assert audit.report["grid"]["ring_nodata"]["rainfall_scaling_nodata_cells"] == int((rm == -9999).sum())


def test_legacy_options_resolution(audit):
    maps = {m["xml_key"]: m for m in audit.report["legacy_options"]["maps"]}
    assert maps["pavement_map"]["read_by_storm_setup"] is True
    assert maps["saturated_soil-moisture_map"]["read_by_storm_setup"] is True
    assert maps["final_infiltration_map"]["read_by_storm_setup"] is False
    assert maps["initial_soil-moisture_map"]["read_by_storm_setup"] is False
    settings = audit.report["legacy_options"]["settings"]
    assert settings["update_topography"] is False
    assert settings["particle_density_occurrences"] == 2
    assert audit.particle_density_kg_m3 == 2650.0
    inactive = {r["file"] for r in audit.report["inactive_referenced_maps"]}
    assert {"p1ksat290806.asc", "p1sm290905.asc"} <= inactive


def test_bed_plan(audit):
    bed, recipe = audit.bed, audit.recipe
    zmin = float(audit.elevation_source_m.min())
    expected_offset = math.ceil((recipe.minimum_fill_depth_m - zmin) / 0.001 - 1e-9) * 0.001
    assert bed.datum_offset_m == pytest.approx(expected_offset, abs=1e-12)
    assert bed.elevation_m.min() >= recipe.minimum_fill_depth_m - 1e-12
    assert bed.nz * recipe.voxel_dz_m >= bed.elevation_m.max() + recipe.headroom_m - 1e-9
    total = float((bed.elevation_m * 0.25 * recipe.bulk_density_kg_m3).sum())
    assert audit.report["bed"]["total_mass_kg"] == pytest.approx(total, rel=1e-12)


# --- rejection paths, on a disposable copy of the inputs ---------------------------------
def _audit_copy(root, tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    return audit_plot1(load_recipe(RECIPE_PATH, mahleran_root=root), case_dir)


def test_rejects_a_map_with_a_different_shape(fake_mahleran, tmp_path):
    path = fake_mahleran / "Input" / "input_p1" / "plot1_phi3.asc"
    write_legacy_grid(path, read_legacy_grid(path)[:, :-1])
    with pytest.raises(Plot1ImportError, match="ncols"):
        _audit_copy(fake_mahleran, tmp_path)


def test_rejects_interior_nodata(fake_mahleran, tmp_path):
    path = fake_mahleran / "Input" / "input_p1" / "plot1_phi2.asc"
    grid = read_legacy_grid(path)
    grid[10, 5] = -9999.0
    write_legacy_grid(path, grid)
    with pytest.raises(Plot1ImportError, match="nodata"):
        _audit_copy(fake_mahleran, tmp_path)


def test_rejects_material_nonclosure(fake_mahleran, tmp_path):
    path = fake_mahleran / "Input" / "input_p1" / "plot1_phi2.asc"
    write_legacy_grid(path, read_legacy_grid(path) * 1.2)
    with pytest.raises(Plot1ImportError, match="nonclosure"):
        _audit_copy(fake_mahleran, tmp_path)


def test_rejects_a_missing_map(fake_mahleran, tmp_path):
    (fake_mahleran / "Input" / "input_p1" / "plot1_phi4.asc").unlink()
    with pytest.raises(Plot1ImportError, match="not found"):
        _audit_copy(fake_mahleran, tmp_path)
