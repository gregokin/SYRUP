"""CW-R1c2: the nested (constant-preserving) bilinear elevation.

EXACT constant reproduction (array_equal / bytes) including the root's 155.54 counterexample, varied positive constants
and positions; the defect of the C1 weighted sum is shown on the same points (independent re-implementation); planar,
integer-centre and saddle oracles; support/fallback/nearest paths unchanged; the real stable inputs: every constant-support
target cell equals its source value exactly, the root counterexample record pinned, the C1 weighted recipe now refused."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from conftest import SYRUP

from maple_syrup import chastre_case as cc
from maple_syrup.case_import import Plot1ImportError
from maple_syrup.rfid_case import _to_maple, read_legacy_grid

NODATA = -9999.0
EPS = np.finfo(np.float64).eps
REPRO = SYRUP / "agent_handoffs/tasks/conservative_water/cw_r1_root_constant_surface_repro.json"
STABLE = SYRUP / "outputs/conservative_water/cw_r1_rotated_chastre_inputs_bilinear_stable"
WEIGHTED = SYRUP / "outputs/conservative_water/cw_r1_rotated_chastre_inputs_bilinear"


def _bilinear(src, x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    nearest = cc.sample_source_cells(src, np.floor(y).astype(np.int64), np.floor(x).astype(np.int64), NODATA)[0]
    return cc.bilinear_source_elevation(src, x, y, nearest, NODATA)


def _weighted_c1(src, x, y):
    """Independent re-implementation of the SUPERSEDED C1 weighted-sum arithmetic (evidence of the defect only)."""
    px, py = np.asarray(x) - 0.5, np.asarray(y) - 0.5
    j0, i0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    fx, fy = px - j0, py - i0
    z00, z01, z10, z11 = src[i0, j0], src[i0, j0 + 1], src[i0 + 1, j0], src[i0 + 1, j0 + 1]
    return (((1.0 - fx) * (1.0 - fy) * z00 + fx * (1.0 - fy) * z01) + (1.0 - fx) * fy * z10) + fx * fy * z11


def test_root_counterexample_is_now_exact_and_was_wrong_under_the_weighted_sum():
    repro = json.loads(REPRO.read_text())
    assert hashlib.sha256(REPRO.read_bytes()).hexdigest() == \
        "44ca75d967520fd8570e5acc957b027f88d77d55f604a0aab36ff5f36707f063"
    c = repro["constant_source_elevation_m"]
    src = np.full((6, 6), c)
    x, y = np.array(repro["counterexample_sample_x"]), np.array(repro["counterexample_sample_y"])
    z, used = _bilinear(src, x, y)
    assert used.all() and np.array_equal(z, np.full(4, c)) and z.tobytes() == np.full(4, c).tobytes()
    old = _weighted_c1(src, x, y)
    offsets = [int(round((v - c) / np.spacing(c))) for v in old]
    assert offsets == repro["counterexample_weighted_sum_ulp_offsets"] == [0, 1, -1, 0]   # the defect, reproduced


@pytest.mark.parametrize("c", [155.54, 130.68, 160.76, 0.1, 1.0, 147.18, 1e-7, 12345.678, 2.0 ** 0.5])
def test_varied_constants_and_positions_are_reproduced_bit_exactly(c):
    rng = np.random.default_rng(int(c * 1000) % 2 ** 32)
    src = np.full((9, 11), c)
    x, y = rng.uniform(0.5, 10.5 - 1e-9, 2000), rng.uniform(0.5, 8.5 - 1e-9, 2000)
    z, used = _bilinear(src, x, y)
    assert used.all() and z.tobytes() == np.full(2000, c).tobytes()


def test_constant_patch_inside_a_varying_surface_is_exact_and_neighbours_are_untouched():
    src = np.round(np.random.default_rng(9).uniform(130, 160, (10, 12)), 2)
    src[3:6, 4:8] = 155.54                     # constant patch: centres rows 3..5, cols 4..7
    rng = np.random.default_rng(10)
    x, y = rng.uniform(4.5, 7.5, 500), rng.uniform(3.5, 5.5, 500)   # all four centres inside the patch
    z, used = _bilinear(src, x, y)
    assert used.all() and np.array_equal(z, np.full(500, 155.54))


def test_planar_and_saddle_surfaces_match_exact_rational_oracles():
    r, c = np.indices((8, 9), dtype=np.float64)
    rng = np.random.default_rng(11)
    x, y = rng.uniform(0.5, 8.5 - 1e-9, 400), rng.uniform(0.5, 7.5 - 1e-9, 400)
    for name, f in (("planar", lambda r, c: 120.0 + 0.37 * c + 0.11 * r),
                    ("saddle", lambda r, c: 140.0 + 0.05 * (c - 4) * (r - 3))):
        src = f(r, c)
        z, used = _bilinear(src, x, y)
        assert used.all(), name
        for zi, xi, yi in zip(z, x, y, strict=True):
            px, py = Fraction(float(xi)) - Fraction(1, 2), Fraction(float(yi)) - Fraction(1, 2)
            i0, j0 = math.floor(py), math.floor(px)
            fy, fx = py - i0, px - j0
            exact = sum(Fraction(float(src[i0 + di, j0 + dj])) * (fx if dj else 1 - fx) * (fy if di else 1 - fy)
                        for di in (0, 1) for dj in (0, 1))
            assert abs(float(exact) - zi) <= 8 * EPS * 160.0, (name, zi, float(exact))


def test_integer_centres_and_unit_fractions_return_the_corner_values_exactly():
    src = np.round(np.random.default_rng(12).uniform(130, 160, (8, 9)), 2)
    rr, cc_ = np.indices((6, 7))
    z, used = _bilinear(src, cc_ + 1.5, rr + 1.5)          # fx = fy = 0 exactly
    assert used.all() and np.array_equal(z, src[rr + 1, cc_ + 1])


def test_support_fallback_and_nearest_paths_are_unchanged_by_the_nested_form():
    src = np.round(np.random.default_rng(13).uniform(130, 160, (6, 7)), 2)
    src[2, 3] = NODATA
    x = np.array([3.9, 3.6, 6.9, 5.2])
    y = np.array([3.1, 2.5, 3.4, 4.6])
    z, used = _bilinear(src, x, y)
    assert used.tolist() == [False, False, False, True]
    assert (z[0], z[1], z[2]) == (src[3, 3], NODATA, src[3, 6])
    assert src[4:6, 4:6].min() <= z[3] <= src[4:6, 4:6].max()


# ------------------------------------------------------------------------------------- the real stable inputs
@pytest.fixture(scope="module")
def real():
    recipe_path = STABLE / "recipe_r1c2_rotated_bilinear_stable.yaml"
    if not recipe_path.is_file():
        pytest.skip("stable inputs absent")
    recipe = cc.load_chastre_recipe(recipe_path)
    rot = recipe.raw["rainfall_scaling"]["rotation"]
    original = cc.load_chastre_recipe(rot["original_recipe"]["path"])
    full = _to_maple(read_legacy_grid(Path(original.raw["terrain"]["path"]))[1])
    stored = _to_maple(read_legacy_grid(Path(recipe.raw["terrain"]["path"]))[1])
    return recipe, rot, full, stored


def test_every_constant_support_target_cell_of_the_real_stable_terrain_is_its_source_value(real):
    recipe, rot, full, stored = real
    shape = (stored.shape[0] - 2, stored.shape[1] - 2)
    x, y = cc.rotated_source_coordinates(rot["axis_vector_cells"], rot["interior_origin_numerators"], shape)
    rows, cols, _ = cc.rotated_source_indices(rot["axis_vector_cells"], rot["interior_origin_numerators"], shape)
    nearest, _ = cc.sample_source_cells(full, rows, cols, NODATA)
    _z, used = cc.bilinear_source_elevation(full, x, y, nearest, NODATA)
    i0, j0 = np.floor(y - 0.5).astype(np.int64), np.floor(x - 0.5).astype(np.int64)
    ic, jc = np.clip(i0, 0, full.shape[0] - 2), np.clip(j0, 0, full.shape[1] - 2)
    z00 = full[ic, jc]
    constant = used & (z00 == full[ic, jc + 1]) & (z00 == full[ic + 1, jc]) & (z00 == full[ic + 1, jc + 1])
    assert constant.sum() > 0
    assert np.array_equal(stored[1:-1, 1:-1][constant], z00[constant])
    assert int(constant.sum()) == json.loads(REPRO.read_text())["constant_support_target_cells"]   # root's 431


def test_the_superseded_c1_weighted_recipe_is_refused_by_the_nested_code(real):
    weighted = cc.load_chastre_recipe(WEIGHTED / "recipe_r1c1_rotated_bilinear.yaml")
    terrain = _to_maple(read_legacy_grid(Path(weighted.raw["terrain"]["path"]))[1])[1:-1, 1:-1]
    with pytest.raises(Plot1ImportError, match="not the bilinear_full_valid_support_nearest_fallback sampling"):
        cc.rotation_mapping(weighted, terrain)
    # and the stable inputs pass, recording the nested definition
    recipe, _rot, _full, stored = real
    m = cc.rotation_mapping(recipe, stored[1:-1, 1:-1])
    assert "NESTED" in m["record"]["elevation_sampling"]["definition"]


def test_stable_ascii_header_is_the_declared_local_frame(real):
    recipe, *_ = real
    header = Path(recipe.raw["terrain"]["path"]).read_text().splitlines()[:6]
    assert header == ["ncols 1782", "nrows 888", "xllcorner 0", "yllcorner 0", "cellsize 1", "NODATA_value -9999"]
