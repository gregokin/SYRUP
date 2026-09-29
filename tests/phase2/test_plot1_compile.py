"""End-to-end: generate the Plot 1 case, compile and reload it with MAPLE's
own compile_case / load_compiled_case, and check the result against an
independent reading of the MAHLERAN inputs. No wind run is launched."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import yaml
from conftest import (
    PLOT1_DIR,
    RECIPE_PATH,
    independent_fractions,
    interior_maple,
    read_legacy_grid,
)

from maple_syrup.case_import import (
    Plot1ImportError,
    generate_plot1_case,
    load_recipe,
    main,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generated(tmp_path_factory, mahleran_root):
    recipe = load_recipe(RECIPE_PATH, mahleran_root=mahleran_root)
    out = tmp_path_factory.mktemp("plot1_compile") / "case"
    summary = generate_plot1_case(recipe, out)
    return recipe, out, summary


@pytest.fixture(scope="module")
def reloaded(generated):
    from maple.case_tools.compilers.case_compiler import load_compiled_case

    return load_compiled_case(generated[1])


def test_generation_is_bound_to_the_compiled_identity(generated):
    _, out, summary = generated
    assert summary["status"] == "ok"
    binding = json.loads((out / "syrup" / "plot1_binding.json").read_text(encoding="utf-8"))
    provenance = yaml.safe_load((out / "provenance.yaml").read_text(encoding="utf-8"))
    assert binding["maple_case_identity_sha256"] == provenance["case_identity_sha256"]
    assert binding["maple_artifact_sha256"] == provenance["artifact_sha256"]
    assert binding["provenance_yaml_sha256"] == _sha(out / "provenance.yaml")
    assert binding["case_yaml_sha256"] == _sha(out / "case.yaml")
    assert binding["syrup_fields_sha256"] == _sha(out / "syrup" / "plot1_fields.npz")
    assert binding["syrup_report_sha256"] == _sha(out / "syrup" / "plot1_import_report.json")
    assert binding["maple_source_stable"] is True
    assert not (out / "syrup" / "FAILED.json").exists()
    assert not (out / "outputs").exists(), "no MAPLE run may be launched"


def test_case_yaml_uses_maple_import_paths(generated):
    _, out, summary = generated
    case = yaml.safe_load((out / "case.yaml").read_text(encoding="utf-8"))
    elevation = case["import"]["elevation"]
    assert elevation["source"]["format"] == "ascii_grid"
    assert [t["name"] for t in elevation["transforms"]] == ["crop", "datum_offset"]
    assert elevation["transforms"][1]["reference"] == "none"
    assert case["topography"]["base"] == "imported"
    assert case["topography"]["perturbation"]["relief_m"] == 0.0
    sediment = case["import"]["sediment"]
    assert sediment["mode"] == "categorical_map"
    assert len(sediment["categorical_map"]["profiles"]) == summary["n_composition_codes"]
    for profile in sediment["categorical_map"]["profiles"].values():
        (interval,) = profile["intervals"]
        assert abs(sum(interval["fractions"].values()) - 1.0) <= 1e-12
    assert case["geometry"]["boundary_x"]["kind"] == "prescribed"
    assert case["topographic_wind"]["enabled"] is False


def test_reloaded_bed_matches_independent_inventory(generated, reloaded):
    _recipe, out, _ = generated
    case_yaml = yaml.safe_load((out / "case.yaml").read_text(encoding="utf-8"))
    offset = case_yaml["import"]["elevation"]["transforms"][1]["offset_m"]
    g = reloaded.config.geometry
    assert (g.ny, g.nx, g.dx_m, g.dy_m) == (60, 20, 0.5, 0.5)
    assert (g.voxel_dz_m, g.bulk_density_kg_m3, g.active_layer_thickness_m) == (0.1, 1250.0, 0.002)

    z = interior_maple(read_legacy_grid(PLOT1_DIR / "p1dem.asc")) + offset
    np.testing.assert_allclose(reloaded.topography_result.elevation_m, z, rtol=0, atol=1e-12)
    assert z.min() >= 0.3 - 1e-12

    _, _, final = independent_fractions(PLOT1_DIR)
    mass_per_m = 1250.0 * 0.25
    expected = (z * mass_per_m)[..., None] * final
    voxel = np.asarray(reloaded.voxel_column.mass_kg)
    active = np.asarray(reloaded.active_layer.mass_kg)
    bed = voxel.sum(axis=2) + active
    np.testing.assert_allclose(bed, expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(bed.sum(axis=(0, 1)), expected.sum(axis=(0, 1)), rtol=1e-12)

    # Active layer: 2 mm of the cell's own composition.
    np.testing.assert_allclose(active.sum(axis=-1), 0.002 * mass_per_m, rtol=0, atol=1e-9)
    np.testing.assert_allclose(active / active.sum(axis=-1, keepdims=True), final, rtol=0, atol=1e-9)
    # Voxels: full from the base up to the fill depth minus the active layer.
    levels = np.arange(voxel.shape[2]) * 0.1
    expected_levels = np.clip((z - 0.002)[..., None] - levels, 0.0, 0.1) * mass_per_m
    np.testing.assert_allclose(voxel.sum(axis=3), expected_levels, rtol=0, atol=1e-7)
    assert voxel.shape[2] * 0.1 >= z.max() + 0.2 - 1e-9  # declared headroom


def test_initial_water_mobile_and_ledger_are_empty(reloaded):
    assert not np.any(np.asarray(reloaded.water.depth_m))
    assert not np.any(np.asarray(reloaded.water.mobile_mass_by_cell_class_kg))
    assert reloaded.initial_mobile_state is None
    ledger = reloaded.sediment_ledger
    assert not np.any(np.asarray(ledger.pending_bed_mass_change_kg))
    assert not np.any(np.asarray(ledger.process_totals_kg))
    active = np.asarray(reloaded.active_layer.mass_kg)
    np.testing.assert_array_equal(np.asarray(reloaded.sediment_availability.available_mass_kg), active)


def test_existing_output_is_never_overwritten(generated):
    recipe, out, _ = generated
    before = _sha(out / "case.yaml")
    with pytest.raises(Plot1ImportError, match="existing"):
        generate_plot1_case(recipe, out)
    assert _sha(out / "case.yaml") == before


def test_cli_writes_package_without_compiling(tmp_path, mahleran_root):
    out = tmp_path / "no_compile"
    code = main([
        "--recipe", str(RECIPE_PATH), "--output-dir", str(out),
        "--mahleran-root", str(mahleran_root), "--no-compile",
    ])
    assert code == 0
    assert (out / "case.yaml").is_file() and (out / "syrup" / "plot1_import_report.json").is_file()
    assert not (out / "provenance.yaml").exists() and not (out / "processed").exists()
