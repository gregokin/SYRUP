"""CW-R1: rotated Chastre grid (opt-in recipe descriptor), inverse-nearest terrain/mask/rainfall sampling, native path unchanged.

CPU only, small synthetic terrains plus read-only landmarks of the real Chastre DTM; no MAPLE tile compilation, INITIAL, GPU
or model run. Independent references: exact rational (Fraction) geometry, hand landmarks, brute-force angle scan, the native
audit of the SAME synthetic original (same-index comparison), and the frozen B93 module (AST)."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from conftest import B93_CHASTRE, SYRUP, load_prep

from maple_syrup import chastre_case as cc
from maple_syrup.case_import import Plot1ImportError
from maple_syrup.rfid_case import _to_maple, read_legacy_grid

NODATA = -9999.0
RFID_CASE = SYRUP / "outputs/rfid/case"
REAL_RECIPE = SYRUP / "cases/chastre/recipe.yaml"
PREP = load_prep()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -------------------------------------------------------------------------------------------------- native path unchanged
def _top_level(path):
    out = {}
    for n in ast.parse(Path(path).read_text()).body:
        if hasattr(n, "name"):
            key = n.name
        elif isinstance(n, ast.Assign):
            key = ",".join(ast.unparse(t) for t in n.targets)
        else:
            key = ast.unparse(n)[:80]
        assert key not in out, key
        out[key] = ast.dump(n)
    return out


def test_only_the_declared_top_level_definitions_differ_from_frozen_b93():
    old, new = _top_level(B93_CHASTRE), _top_level(cc.__file__)
    changed = {k for k in old if new.get(k) != old[k]}
    added = set(new) - set(old)
    assert not set(old) - set(new)
    assert changed == {"load_chastre_recipe", "audit_chastre", "__all__"}, changed
    assert added == {"_is_int", "_validate_rotation", "rotated_source_indices", "sample_source_cells", "rotation_mapping",
                     "ROTATION_SCHEMA", "ROTATION_SAMPLING", "_ROTATION_KEYS", "_HEX64",
                     # CW-R1c1: the opt-in bilinear elevation (mode tuple, coordinates, bilinear sampler)
                     "ELEVATION_SAMPLING_MODES", "rotated_source_coordinates", "bilinear_source_elevation"}, added
    # the native branch statement of load_chastre_recipe and the native rainfall statements are kept verbatim
    text = Path(cc.__file__).read_text()
    assert 'scaling = _keys(raw["rainfall_scaling"], {"resize", "gap_fill"}, "rainfall_scaling")' in text
    assert "resized, row_index, col_index = nearest_resize(filled, (ny, nx))\n        scale = np.where(active, resized, 0.0)" \
        in text


def test_real_native_recipe_loads_exactly_as_authored():
    recipe = cc.load_chastre_recipe(REAL_RECIPE)
    assert recipe.raw == yaml.safe_load(REAL_RECIPE.read_text())
    assert "rotation" not in recipe.raw["rainfall_scaling"]
    assert recipe.sha256 == _sha(REAL_RECIPE)


# ------------------------------------------------------------------------------------------------- descriptor validation
def _descriptor(**over):
    d = {"schema": cc.ROTATION_SCHEMA, "original_recipe": {"path": "/x/recipe.yaml", "sha256": "a" * 64},
         "original_terrain_sha256": "b" * 64, "axis_vector_cells": [16, 15], "interior_origin_numerators": [3, -7],
         "sampling": cc.ROTATION_SAMPLING}
    d.update(over)
    return d


@pytest.mark.parametrize("over, message", [
    ({"schema": "other"}, "schema"),
    ({"sampling": "bilinear"}, "sampling"),
    ({"axis_vector_cells": [2, 0]}, "PRIMITIVE"),
    ({"axis_vector_cells": [1, 2]}, "PRIMITIVE"),        # angle > 45 degrees
    ({"axis_vector_cells": [1, -1]}, "PRIMITIVE"),       # exactly -45 degrees is outside (-45, 45]
    ({"axis_vector_cells": [0, 1]}, "PRIMITIVE"),
    ({"axis_vector_cells": [True, 0]}, "two integers"),
    ({"axis_vector_cells": [1.0, 0]}, "two integers"),
    ({"axis_vector_cells": [1, 0, 0]}, "two integers"),
    ({"interior_origin_numerators": [1]}, "two integers"),
    ({"original_terrain_sha256": "B" * 64}, "hexadecimal"),
    ({"original_recipe": {"path": "/x", "sha256": "z" * 64}}, "hexadecimal"),
    ({"original_recipe": {"path": "", "sha256": "a" * 64}}, "non-empty"),
    ({"original_recipe": {"path": "/x"}}, "exactly the keys"),
    ({"extra": 1}, "exactly the keys"),
])
def test_malformed_rotation_descriptors_are_refused(over, message):
    with pytest.raises(Plot1ImportError, match=message):
        cc._validate_rotation(_descriptor(**over))


def test_valid_descriptor_and_the_boundary_45_degree_axis_are_accepted():
    assert cc._validate_rotation(_descriptor())["axis_vector_cells"] == [16, 15]
    assert cc._validate_rotation(_descriptor(axis_vector_cells=[1, 1]))["axis_vector_cells"] == [1, 1]
    assert cc._validate_rotation(_descriptor(axis_vector_cells=[1, 0]))["axis_vector_cells"] == [1, 0]


# ---------------------------------------------------------------------------------------------- geometry (independent)
def test_identity_axis_maps_each_interior_cell_to_itself():
    rows, cols, edge = cc.rotated_source_indices((1, 0), (1, 1), (7, 9))
    r, c = np.indices((7, 9))
    assert np.array_equal(rows, r + 1) and np.array_equal(cols, c + 1)
    assert edge["minimum_distance_to_a_source_edge_cells"] == 0.5


def _exact_floor_reference(axis, origin, shape):
    """Exact rational floor for a Pythagorean axis (L rational): the independent reference."""
    a, b = axis
    length = Fraction(math.isqrt(a * a + b * b))
    assert length * length == a * a + b * b
    nu, nv = origin
    rows = np.empty(shape, dtype=np.int64)
    cols = np.empty(shape, dtype=np.int64)
    for r in range(shape[0]):
        for c in range(shape[1]):
            u = Fraction(nu) / length + c + Fraction(1, 2)
            v = Fraction(nv) / length + r + Fraction(1, 2)
            x, y = (a * u - b * v) / length, (b * u + a * v) / length
            rows[r, c], cols[r, c] = math.floor(y), math.floor(x)  # floor: an exact edge goes to the HIGHER index
    return rows, cols


@pytest.mark.parametrize("axis, origin", [((4, 3), (13, -21)), ((12, 5), (40, 7)), ((24, -7), (5, 9))])
def test_pythagorean_axes_equal_the_exact_rational_floor(axis, origin):
    rows, cols, _ = cc.rotated_source_indices(axis, origin, (23, 31))
    ref_rows, ref_cols = _exact_floor_reference(axis, origin, (23, 31))
    assert np.array_equal(rows, ref_rows) and np.array_equal(cols, ref_cols)


def test_hand_landmarks_of_the_4_3_axis():
    # axis (4, 3): L = 5, +u = (0.8, 0.6), +v = (-0.6, 0.8). With (nu, nv) = (25, 0): u0 = 5, v0 = 0.
    # cell (0, 0): u = 5.5, v = 0.5 -> x = 4.4 - 0.3 = 4.1, y = 3.3 + 0.4 = 3.7 -> source (row 3, col 4)
    # cell (2, 1): u = 6.5, v = 2.5 -> x = 5.2 - 1.5 = 3.7, y = 3.9 + 2.0 = 5.9 -> source (row 5, col 3)
    rows, cols, _ = cc.rotated_source_indices((4, 3), (25, 0), (3, 2))
    assert (rows[0, 0], cols[0, 0]) == (3, 4)
    assert (rows[2, 1], cols[2, 1]) == (5, 3)


def test_affine_inverse_recovers_the_target_centres():
    a, b, nu, nv = 16, 15, 1234, -5678
    length = math.sqrt(a * a + b * b)
    ny, nx = 11, 17
    u = nu / length + np.arange(nx) + 0.5
    v = nv / length + np.arange(ny) + 0.5
    uu, vv = np.meshgrid(u, v)
    x, y = (a * uu - b * vv) / length, (b * uu + a * vv) / length
    assert np.allclose((a * x + b * y) / length, uu, atol=1e-9) and np.allclose((-b * x + a * y) / length, vv, atol=1e-9)
    rows, cols, _ = cc.rotated_source_indices((a, b), (nu, nv), (ny, nx))
    assert np.array_equal(rows, np.floor(y).astype(np.int64)) and np.array_equal(cols, np.floor(x).astype(np.int64))


def test_sampling_is_the_exact_source_value_or_nodata_outside_and_never_interpolated():
    rng = np.random.default_rng(5)
    src = np.round(rng.uniform(100, 200, (9, 12)), 3)
    src[0, :] = src[-1, :] = src[:, 0] = src[:, -1] = NODATA
    src[4, 5] = src[4, 6] = NODATA
    rows = np.array([[-1, 0, 4, 4, 5, 8, 9]])
    cols = np.array([[3, 3, 5, 6, 7, 11, 2]])
    out, inside = cc.sample_source_cells(src, rows, cols, NODATA)
    assert inside.tolist() == [[False, True, True, True, True, True, False]]
    assert out[0, 0] == NODATA and out[0, 6] == NODATA          # outside the grid
    assert out[0, 1] == NODATA and out[0, 5] == NODATA          # ring
    assert out[0, 2] == NODATA and out[0, 3] == NODATA          # interior holes stay holes (no fill)
    assert out[0, 4] == src[5, 7]                               # exact value at the exact index


# ------------------------------------------------------------------------------------------------ rectangle (independent)
def test_axis_aligned_block_gives_the_identity_rectangle():
    act = np.zeros((10, 12), dtype=bool)
    act[2:7, 3:11] = True
    best, _ = PREP.min_area_rectangle(PREP.convex_hull(PREP.footprint_hull_points(act)))
    assert best["axis"] == (1, 0) and (best["nrows"], best["ncols"]) == (5, 8) and (best["nu"], best["nv"]) == (3, 2)


def test_rectangle_is_not_beaten_by_a_brute_force_angle_scan():
    rng = np.random.default_rng(1)
    yy, xx = np.indices((60, 80))
    t = math.radians(33.0)
    act = (((xx - 40) * math.cos(t) + (yy - 30) * math.sin(t)) / 30) ** 2 + \
          ((-(xx - 40) * math.sin(t) + (yy - 30) * math.cos(t)) / 9) ** 2 < 1
    act &= rng.random(act.shape) < 0.97
    hull = PREP.convex_hull(PREP.footprint_hull_points(act))
    best, _ = PREP.min_area_rectangle(hull)
    pts = np.array(hull, dtype=float)
    brute = min((np.ptp(pts @ [math.cos(a), math.sin(a)]) * np.ptp(pts @ [-math.sin(a), math.cos(a)]))
                for a in np.radians(np.arange(0, 90, 0.005)))
    assert float(best["area"]) <= brute + 1e-9
    corners = [(x, y) for r, c in zip(*np.nonzero(act), strict=True) for x, y in ((c, r), (c + 1, r), (c, r + 1),
                                                                                    (c + 1, r + 1))]
    a, b = best["axis"]
    length = math.sqrt(a * a + b * b)
    pu = [(a * x + b * y) for x, y in corners]
    pv = [(-b * x + a * y) for x, y in corners]
    assert min(pu) == best["nu"] and min(pv) == best["nv"]       # hull from row extremes = hull of ALL corners
    assert best["ncols"] >= (max(pu) - min(pu)) / length and best["ncols"] - 1 < (max(pu) - min(pu)) / length


def test_real_chastre_footprint_rectangle_landmark():
    """The prompt's independent expectation: ~43.1556926555 degrees CCW, ~1780 columns x 886 rows (full squares)."""
    header, body = read_legacy_grid(Path(cc.load_chastre_recipe(REAL_RECIPE).raw["terrain"]["path"]))
    act = _to_maple(body) != NODATA
    best, _ = PREP.min_area_rectangle(PREP.convex_hull(PREP.footprint_hull_points(act)))
    a, b = best["axis"]
    assert abs(math.degrees(math.atan2(b, a)) - 43.1556926555) < 1e-6
    assert abs(best["ncols"] - 1780) <= 2 and abs(best["nrows"] - 886) <= 2


# --------------------------------------------------------------------------------------------------- ESRI text
def test_esri_orientation_header_and_exact_roundtrip(tmp_path):
    full = np.full((5, 6), NODATA)
    full[1:-1, 1:-1] = [[0.1, 1 / 3, 1e-7, 123456.789], [-0.0 + 2.5, 160.12, 160.111, 9.999999999999999e2],
                        [7.0, 1e300, 5e-324, 2 ** 0.5]]
    text = PREP.esri_text(full, NODATA)
    lines = text.splitlines()
    assert lines[:6] == ["ncols 6", "nrows 5", "xllcorner 0", "yllcorner 0", "cellsize 1", "NODATA_value -9999"]
    assert lines[7].split()[1:-1] == [repr(float(v)) for v in full[3, 1:-1]]   # first interior file line = NORTHERNMOST row
    path = tmp_path / "t.asc"
    path.write_text(text)
    _header, body = read_legacy_grid(path)
    assert np.array_equal(_to_maple(body), full)
    assert PREP.esri_text(full, NODATA) == text                          # deterministic


def test_index_mapping_is_deterministic():
    one = cc.rotated_source_indices((1003, 940), (12345, -6789), (40, 50))
    two = cc.rotated_source_indices((1003, 940), (12345, -6789), (40, 50))
    assert np.array_equal(one[0], two[0]) and np.array_equal(one[1], two[1]) and one[2] == two[2]


# ------------------------------------------------------------------------------- synthetic original + rotated audits
def _write_native(tmp, name, full):
    dtm = tmp / f"{name}.asc"
    dtm.write_text(PREP.esri_text(full, NODATA))
    return dtm


def _recipe_yaml(path, case_name, dtm, full, counts, rotation=None):
    raw = {"schema": cc.RECIPE_SCHEMA, "case_name": case_name, "source_rfid": {"case_dir": str(RFID_CASE)},
           "terrain": {"path": str(dtm), "sha256": _sha(dtm), "nrows": full.shape[0], "ncols": full.shape[1],
                       "cellsize_m": 1.0, "nodata_value": -9999, "expected_n_active": counts["n_active"],
                       "expected_n_sinks": counts["n_sinks"], "expected_n_outlets": counts["n_outlets"],
                       "expected_nz": counts["nz"]},
           "tiles": {"rows": 8, "max_rss_gib": 4},
           "rainfall_scaling": {"resize": "nearest_pixel_centre", "gap_fill": "nearest_valid_euclid_first_row_major"}}
    if rotation is not None:
        raw["rainfall_scaling"]["rotation"] = rotation
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return cc.load_chastre_recipe(path)


@pytest.fixture(scope="module")
def rfid_source():
    """The real verified RFID sidecar read-only (no MAPLE verification here): what `audit_chastre` reads from `source`."""
    side = RFID_CASE / "syrup"
    report = json.loads((side / "rfid_import_report.json").read_text())
    binding = json.loads((side / "rfid_binding.json").read_text())
    with np.load(side / "rfid_fields.npz", allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}
    return SimpleNamespace(report=report, fields=fields, binding=binding, case_dir=RFID_CASE,
                           checks={"binding_sha256": _sha(side / "rfid_binding.json")})


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory, rfid_source):
    tmp = tmp_path_factory.mktemp("r1")
    yy, xx = np.indices((40, 52))
    t = math.radians(40.0)
    inside = (((xx - 26) * math.cos(t) + (yy - 20) * math.sin(t)) / 22) ** 2 + \
             ((-(xx - 26) * math.sin(t) + (yy - 20) * math.cos(t)) / 9) ** 2 < 1
    z = np.round(150.0 + 0.031 * xx + 0.047 * yy + 0.4 * np.sin(xx / 3.0) * np.cos(yy / 4.0), 2)
    full = np.where(inside, z, NODATA)
    full[0, :] = full[-1, :] = full[:, 0] = full[:, -1] = NODATA
    full[20, 26] = NODATA  # an interior hole
    odtm = _write_native(tmp, "original", full)
    ocounts = PREP.expected_counts(full, NODATA, rfid_source.report)
    original = _recipe_yaml(tmp / "original.yaml", "synthetic_original", odtm, full, ocounts)
    hull = PREP.convex_hull(PREP.footprint_hull_points(full != NODATA))
    best, _ = PREP.min_area_rectangle(hull)
    rows, cols, _ = cc.rotated_source_indices(best["axis"], (best["nu"], best["nv"]), (best["nrows"], best["ncols"]))
    interior, _ = cc.sample_source_cells(full, rows, cols, NODATA)
    tfull = np.full((best["nrows"] + 2, best["ncols"] + 2), NODATA)
    tfull[1:-1, 1:-1] = interior
    tdtm = _write_native(tmp, "rotated", tfull)
    rotation = {"schema": cc.ROTATION_SCHEMA, "original_recipe": {"path": str(original.path), "sha256": original.sha256},
                "original_terrain_sha256": _sha(odtm), "axis_vector_cells": list(best["axis"]),
                "interior_origin_numerators": [best["nu"], best["nv"]], "sampling": cc.ROTATION_SAMPLING}
    tcounts = PREP.expected_counts(tfull, NODATA, rfid_source.report)
    rotated = _recipe_yaml(tmp / "rotated.yaml", "synthetic_rotated", tdtm, tfull, tcounts, rotation)
    return SimpleNamespace(tmp=tmp, full=full, tfull=tfull, odtm=odtm, tdtm=tdtm, original=original, rotated=rotated,
                           rotation=rotation, best=best, rows=rows, cols=cols, tcounts=tcounts)


def test_rotated_audit_carries_terrain_mask_and_rainfall_with_the_same_source_cell(synthetic, rfid_source):
    native = cc.audit_chastre(synthetic.original, synthetic.odtm, rfid_source)
    rotated = cc.audit_chastre(synthetic.rotated, synthetic.tdtm, rfid_source)
    act = rotated.fields["active"]
    r, c = rotated.fields["rotation_source_row"], rotated.fields["rotation_source_col"]
    assert np.array_equal(r, synthetic.rows) and np.array_equal(c, synthetic.cols)
    n_elev = native.fields["legacy_full_elevation_m"]
    t_elev = rotated.fields["legacy_full_elevation_m"][1:-1, 1:-1]
    assert np.array_equal(t_elev[act], n_elev[r[act], c[act]])                          # same cell: elevation
    assert np.array_equal(native.fields["active"][r[act] - 1, c[act] - 1], np.ones(int(act.sum()), dtype=bool))  # mask
    assert np.array_equal(rotated.fields["rainfall_scaling"][act],
                          native.fields["rainfall_scaling"][r[act] - 1, c[act] - 1])   # same cell: rainfall (native map)
    assert np.all(rotated.fields["rainfall_scaling"][~act] == 0.0)
    # the native resize is the ORIGINAL-shape resize (no new stretching to the rotated box)
    assert np.array_equal(rotated.fields["resize_row_index"], native.fields["resize_row_index"])
    assert np.array_equal(rotated.fields["resize_col_index"], native.fields["resize_col_index"])
    rec = rotated.report["rainfall_scaling"]["rotation"]
    assert rec["original_terrain"]["sha256"] == _sha(synthetic.odtm) and rec["original_recipe"]["sha256"] == \
        synthetic.original.sha256
    assert "rotation" not in native.report["rainfall_scaling"] and "rotated_frame" not in native.report["grid"]
    assert "rotation_source_row" not in native.fields
    assert rotated.report["terrain"]["n_outlets"] == 0 and rotated.report["grid"]["n_active"] == synthetic.tcounts["n_active"]


def test_rotated_audit_is_deterministic(synthetic, rfid_source):
    one = cc.audit_chastre(synthetic.rotated, synthetic.tdtm, rfid_source)
    two = cc.audit_chastre(synthetic.rotated, synthetic.tdtm, rfid_source)
    assert json.dumps(one.report, sort_keys=True, default=str) == json.dumps(two.report, sort_keys=True, default=str)
    assert all(np.array_equal(one.fields[k], two.fields[k]) for k in one.fields)


def _rotated_variant(synthetic, name, *, rotation=None, tfull=None):
    tfull = synthetic.tfull if tfull is None else tfull
    dtm = _write_native(synthetic.tmp, name, tfull)
    return _recipe_yaml(synthetic.tmp / f"{name}.yaml", name, dtm, tfull, synthetic.tcounts,
                        rotation or synthetic.rotation), dtm


def test_tampered_target_terrain_is_refused(synthetic, rfid_source):
    tfull = synthetic.tfull.copy()
    i, j = np.argwhere(tfull[1:-1, 1:-1] != NODATA)[7] + 1
    tfull[i, j] += 0.01
    recipe, dtm = _rotated_variant(synthetic, "tampered", tfull=tfull)
    with pytest.raises(Plot1ImportError, match="not the inverse-nearest sampling"):
        cc.rotation_mapping(recipe, _to_maple(read_legacy_grid(dtm)[1])[1:-1, 1:-1])


@pytest.mark.parametrize("change, message", [
    ("recipe_sha", "original recipe sha256"),
    ("terrain_pin", "terrain sha256 differs"),
    ("shifted_origin", "not the inverse-nearest sampling"),
    ("other_axis", "not the inverse-nearest sampling"),
])
def test_pin_and_descriptor_mismatches_are_refused(synthetic, change, message):
    rotation = json.loads(json.dumps(synthetic.rotation))
    if change == "recipe_sha":
        rotation["original_recipe"]["sha256"] = "0" * 64
    elif change == "terrain_pin":
        rotation["original_terrain_sha256"] = "1" * 64
    elif change == "shifted_origin":
        rotation["interior_origin_numerators"][0] += 3
    else:
        a, b = rotation["axis_vector_cells"]
        rotation["axis_vector_cells"] = [a + 1, b] if math.gcd(a + 1, b) == 1 else [a + 2, b]
    recipe, _ = _rotated_variant(synthetic, f"mismatch_{change}", rotation=rotation)
    with pytest.raises(Plot1ImportError, match=message):
        cc.rotation_mapping(recipe, synthetic.tfull[1:-1, 1:-1])


def test_nested_rotation_and_modified_original_terrain_are_refused(synthetic):
    # nested: the "original" is itself a rotated recipe
    rotation = json.loads(json.dumps(synthetic.rotation))
    rotation["original_recipe"] = {"path": str(synthetic.rotated.path), "sha256": synthetic.rotated.sha256}
    rotation["original_terrain_sha256"] = synthetic.rotated.raw["terrain"]["sha256"]
    recipe, _ = _rotated_variant(synthetic, "nested", rotation=rotation)
    with pytest.raises(Plot1ImportError, match="nested rotations are refused"):
        cc.rotation_mapping(recipe, synthetic.tfull[1:-1, 1:-1])
    # the original terrain file no longer matches its pin
    tmp = synthetic.tmp / "moved"
    tmp.mkdir()
    full = synthetic.full.copy()
    full[5, 5] = 150.0 if full[5, 5] == NODATA else NODATA
    dtm = _write_native(tmp, "original_changed", full)
    raw = dict(synthetic.original.raw)
    raw["terrain"] = dict(raw["terrain"], path=str(dtm))   # keep the OLD pin
    (tmp / "original.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    o2 = cc.load_chastre_recipe(tmp / "original.yaml")
    rotation = json.loads(json.dumps(synthetic.rotation))
    rotation["original_recipe"] = {"path": str(o2.path), "sha256": o2.sha256}
    recipe, _ = _rotated_variant(synthetic, "changed_original", rotation=rotation)
    with pytest.raises(Plot1ImportError, match="original terrain sha256"):
        cc.rotation_mapping(recipe, synthetic.tfull[1:-1, 1:-1])
