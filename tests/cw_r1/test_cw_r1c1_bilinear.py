"""CW-R1c1: opt-in BILINEAR rotated elevation (full valid 4-centre support, else nearest); nearest mask and rainfall unchanged.

CPU only; independent oracles (planar / constant / integer-centre / exact Fraction bilinear, hand landmarks), nodata
support fallback, same-index mask/rainfall through the actual audit, tamper / pin refusals, nearest/native unchanged, and
the real generated inputs vs SciPy order-1 with a quantified FP64 bound plus byte-exact reproduction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from conftest import SYRUP, load_prep

from maple_syrup import chastre_case as cc
from maple_syrup.case_import import Plot1ImportError
from maple_syrup.rfid_case import _to_maple, read_legacy_grid

NODATA = -9999.0
EPS = np.finfo(np.float64).eps
MODE = "bilinear_full_valid_support_nearest_fallback"
RFID_CASE = SYRUP / "outputs/rfid/case"
R1_INPUTS = SYRUP / "outputs/conservative_water/cw_r1_rotated_chastre_inputs"
# CW-R1c2: the real-input checks bind the STABLE (nested) inputs and the C2 preprocessing (same `derive`); the C1 weighted
# inputs are superseded evidence (snapshot cw_r1c1_weighted_pre_correction_snapshot).
BILINEAR_INPUTS = SYRUP / "outputs/conservative_water/cw_r1_rotated_chastre_inputs_bilinear_stable"
C1_PREP = SYRUP / "agent_handoffs/tasks/conservative_water/run_cw_r1c2_prepare_inputs.py"
PREP = load_prep()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bilinear(src, x, y, nearest=None):
    nearest = cc.sample_source_cells(src, np.floor(y).astype(np.int64), np.floor(x).astype(np.int64), NODATA)[0] \
        if nearest is None else nearest
    return cc.bilinear_source_elevation(src, np.asarray(x, dtype=float), np.asarray(y, dtype=float), nearest, NODATA)


# ------------------------------------------------------------------------------------------------------- descriptor
def test_elevation_sampling_is_optional_and_validated():
    base = {"schema": cc.ROTATION_SCHEMA, "original_recipe": {"path": "/x", "sha256": "a" * 64},
            "original_terrain_sha256": "b" * 64, "axis_vector_cells": [577, 541], "interior_origin_numerators": [1, 2],
            "sampling": cc.ROTATION_SAMPLING}
    assert "elevation_sampling" not in cc._validate_rotation(dict(base))
    for mode in cc.ELEVATION_SAMPLING_MODES:
        assert cc._validate_rotation(dict(base, elevation_sampling=mode))["elevation_sampling"] == mode
    for bad in ("bilinear", "cubic", None, 1):
        with pytest.raises(Plot1ImportError, match="elevation_sampling must be one of"):
            cc._validate_rotation(dict(base, elevation_sampling=bad))


def test_r1_nearest_recipe_has_no_mode_and_is_unchanged():
    recipe = cc.load_chastre_recipe(R1_INPUTS / "recipe_r1_rotated.yaml")
    assert recipe.sha256 == "6c4fc606d31834c2dae9620ec5ee8cd5e5ab8eaecb591f74e258cb077187b8dc"
    assert "elevation_sampling" not in recipe.raw["rainfall_scaling"]["rotation"]
    assert _sha(R1_INPUTS / "DTM_Chastre_PD4way_R1_rotated.asc") == \
        "b9f09dd1d6051123e7273074faecfcfeaa1f9ec5007cd1967ae1c5ef0f901ac9"


# ----------------------------------------------------------------------------------------- interpolation oracles
def _grid(f, shape=(8, 9)):
    r, c = np.indices(shape, dtype=np.float64)
    return f(r, c)


def test_constant_field_is_reproduced_exactly():
    src = np.full((8, 9), 151.37)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(1.0, 8.0, 200), rng.uniform(1.0, 7.0, 200)
    z, used = _bilinear(src, x, y)
    assert used.all() and np.array_equal(z, np.full(200, 151.37))   # CW-R1c2: nested form is exact on constants


def test_planar_field_is_reproduced_and_matches_an_exact_rational_oracle():
    # z = 120 + 0.37 c + 0.11 r at source centres (c + 1/2, r + 1/2): bilinear is exact for planes in exact arithmetic
    src = _grid(lambda r, c: 120.0 + 0.37 * c + 0.11 * r)
    rng = np.random.default_rng(4)
    x, y = rng.uniform(0.5, 8.5, 300), rng.uniform(0.5, 7.5, 300)
    z, used = _bilinear(src, x, y)
    assert used.all()
    for zi, xi, yi in zip(z, x, y, strict=True):
        px, py = Fraction(float(xi)) - Fraction(1, 2), Fraction(float(yi)) - Fraction(1, 2)
        i0, j0 = math.floor(py), math.floor(px)
        fy, fx = py - i0, px - j0
        exact = sum(Fraction(float(src[i0 + di, j0 + dj])) * (fx if dj else 1 - fx) * (fy if di else 1 - fy)
                    for di in (0, 1) for dj in (0, 1))
        assert abs(float(exact) - zi) <= 8 * EPS * 130.0              # independent exact bilinear oracle
        assert abs(float(120 + Fraction(37, 100) * px + Fraction(11, 100) * py) - zi) <= 1e-12  # the plane itself


def test_integer_centres_return_the_source_value_bit_exactly():
    src = np.round(np.random.default_rng(5).uniform(130, 160, (8, 9)), 2)
    rr, cc_ = np.indices((6, 7))
    z, used = _bilinear(src, cc_ + 1.5, rr + 1.5)          # exactly on source centres (c+1, r+1)
    assert used.all() and np.array_equal(z, src[rr + 1, cc_ + 1])


def test_identity_axis_landmark_equals_the_nearest_value():
    # axis (1, 0), origin (1, 1): target centre (r, c) lands exactly on source centre (r + 1, c + 1)
    src = np.round(np.random.default_rng(6).uniform(130, 160, (9, 11)), 2)
    x, y = cc.rotated_source_coordinates((1, 0), (1, 1), (7, 9))
    rows, cols, _ = cc.rotated_source_indices((1, 0), (1, 1), (7, 9))
    nearest, _ = cc.sample_source_cells(src, rows, cols, NODATA)
    z, used = cc.bilinear_source_elevation(src, x, y, nearest, NODATA)
    assert np.array_equal(z, nearest) and used.all()


def test_hand_landmark_of_the_4_3_axis():
    # axis (4, 3), (nu, nv) = (25, 0): cell (0, 0) -> x = 4.1, y = 3.7 -> px = 3.6, py = 3.2 -> j0 = 3, fx = 0.6,
    # i0 = 3, fy = 0.2. On the plane z = 10 + 2 c + 3 r (values at centres) bilinear gives 10 + 2*3.6 + 3*3.2 = 26.8.
    src = _grid(lambda r, c: 10.0 + 2.0 * c + 3.0 * r, (8, 8))
    x, y = cc.rotated_source_coordinates((4, 3), (25, 0), (2, 2))
    assert abs(x[0, 0] - 4.1) < 1e-12 and abs(y[0, 0] - 3.7) < 1e-12
    rows, cols, _ = cc.rotated_source_indices((4, 3), (25, 0), (2, 2))
    nearest, _ = cc.sample_source_cells(src, rows, cols, NODATA)
    z, used = cc.bilinear_source_elevation(src, x, y, nearest, NODATA)
    assert used[0, 0] and abs(z[0, 0] - 26.8) < 1e-12
    assert np.array_equal(np.floor(x).astype(np.int64), cols) and np.array_equal(np.floor(y).astype(np.int64), rows)


def test_nodata_in_the_support_or_outside_the_grid_falls_back_to_nearest_and_never_blends():
    src = np.round(np.random.default_rng(7).uniform(130, 160, (6, 7)), 2)
    src[2, 3] = NODATA
    #                  (a) support has nodata      (b) nearest is nodata  (c) beyond the last centre  (d) full support
    x = np.array([3.9, 3.6, 6.9, 5.2])                 # px = 3.4 (j0 = 3), 3.1, 6.4 (j0 = 6: j0+1 outside), 4.7
    y = np.array([3.1, 2.5, 3.4, 4.6])                 # py = 2.6 (i0 = 2 -> contains (2,3)), 2.0, 2.9, 4.1
    z, used = _bilinear(src, x, y)
    assert used.tolist() == [False, False, False, True]
    assert z[0] == src[3, 3]                             # nearest containing cell (floor y = 3, floor x = 3)
    assert z[1] == NODATA                                # the nearest mask is kept: nodata stays nodata
    assert z[2] == src[3, 6]                             # edge: nearest value
    support = src[4:6, 4:6]
    assert support.min() <= z[3] <= support.max()       # a convex blend of four valid values only (no sentinel)


# --------------------------------------------------------------------------- synthetic original -> audits (same index)
def _write(tmp, name, full):
    path = tmp / f"{name}.asc"
    path.write_text(PREP.esri_text(full, NODATA))
    return path


def _recipe(path, name, dtm, full, counts, rotation=None):
    raw = {"schema": cc.RECIPE_SCHEMA, "case_name": name, "source_rfid": {"case_dir": str(RFID_CASE)},
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
    side = RFID_CASE / "syrup"
    with np.load(side / "rfid_fields.npz", allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}
    return SimpleNamespace(report=json.loads((side / "rfid_import_report.json").read_text()), fields=fields,
                           binding=json.loads((side / "rfid_binding.json").read_text()), case_dir=RFID_CASE,
                           checks={"binding_sha256": _sha(side / "rfid_binding.json")})


@pytest.fixture(scope="module")
def synth(tmp_path_factory, rfid_source):
    tmp = tmp_path_factory.mktemp("r1c1")
    yy, xx = np.indices((40, 52))
    t = math.radians(40.0)
    inside = (((xx - 26) * math.cos(t) + (yy - 20) * math.sin(t)) / 22) ** 2 + \
             ((-(xx - 26) * math.sin(t) + (yy - 20) * math.cos(t)) / 9) ** 2 < 1
    z = np.round(150.0 + 0.031 * xx + 0.047 * yy + 0.4 * np.sin(xx / 3.0) * np.cos(yy / 4.0), 2)
    full = np.where(inside, z, NODATA)
    full[0, :] = full[-1, :] = full[:, 0] = full[:, -1] = NODATA
    full[20, 26] = NODATA
    odtm = _write(tmp, "original", full)
    original = _recipe(tmp / "original.yaml", "orig", odtm, full, PREP.expected_counts(full, NODATA, rfid_source.report))
    best, _ = PREP.min_area_rectangle(PREP.convex_hull(PREP.footprint_hull_points(full != NODATA)))
    geom = (best["axis"], (best["nu"], best["nv"]), (best["nrows"], best["ncols"]))
    spec = importlib.util.spec_from_file_location("_c1prep", C1_PREP)
    c1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c1)
    tfull, interior, nearest, used, rows, cols, _, _ = c1.derive(full, NODATA, *geom)
    nfull = np.full_like(tfull, NODATA)
    nfull[1:-1, 1:-1] = nearest
    rot = {"schema": cc.ROTATION_SCHEMA, "original_recipe": {"path": str(original.path), "sha256": original.sha256},
           "original_terrain_sha256": _sha(odtm), "axis_vector_cells": list(geom[0]),
           "interior_origin_numerators": list(geom[1]), "sampling": cc.ROTATION_SAMPLING}
    bdtm, ndtm = _write(tmp, "bilinear", tfull), _write(tmp, "nearest", nfull)
    bil = _recipe(tmp / "bilinear.yaml", "bil", bdtm, tfull, PREP.expected_counts(tfull, NODATA, rfid_source.report),
                  dict(rot, elevation_sampling=MODE))
    near = _recipe(tmp / "nearest.yaml", "near", ndtm, nfull, PREP.expected_counts(nfull, NODATA, rfid_source.report), rot)
    return SimpleNamespace(tmp=tmp, full=full, tfull=tfull, nfull=nfull, odtm=odtm, bdtm=bdtm, ndtm=ndtm, bil=bil,
                           near=near, rot=rot, used=used, rows=rows, cols=cols, nearest=nearest, interior=interior,
                           original=original)


def test_bilinear_audit_keeps_the_nearest_mask_and_rainfall_and_records_the_mode(synth, rfid_source):
    near = cc.audit_chastre(synth.near, synth.ndtm, rfid_source)
    bil = cc.audit_chastre(synth.bil, synth.bdtm, rfid_source)
    assert np.array_equal(bil.fields["active"], near.fields["active"])                      # nearest mask
    assert np.array_equal(bil.fields["rainfall_scaling"], near.fields["rainfall_scaling"])  # nearest-index rainfall
    assert np.array_equal(bil.fields["rotation_source_row"], near.fields["rotation_source_row"])
    assert np.array_equal(bil.fields["rotation_source_col"], near.fields["rotation_source_col"])
    act = bil.fields["active"]
    belev = bil.fields["legacy_full_elevation_m"][1:-1, 1:-1]
    nelev = near.fields["legacy_full_elevation_m"][1:-1, 1:-1]
    assert np.array_equal(belev[act & ~synth.used], nelev[act & ~synth.used])              # fallback = nearest
    assert np.array_equal(belev, synth.interior)
    assert synth.used.sum() > 0 and (act & ~synth.used).sum() > 0                           # both branches exercised
    rec = bil.report["rainfall_scaling"]["rotation"]["elevation_sampling"]
    assert rec["mode"] == MODE and rec["n_bilinear"] == int(synth.used.sum())
    assert rec["n_nearest_fallback"] == int((act & ~synth.used).sum())
    assert "elevation_sampling" not in near.report["rainfall_scaling"]["rotation"]          # R1 record unchanged


def test_explicit_nearest_mode_gives_the_nearest_terrain(synth):
    recipe = _recipe(synth.tmp / "explicit_nearest.yaml", "en", synth.ndtm, synth.nfull,
                     {"n_active": 0, "n_sinks": 0, "n_outlets": 0, "nz": 1}, dict(synth.rot, elevation_sampling="nearest"))
    m = cc.rotation_mapping(recipe, synth.nfull[1:-1, 1:-1])
    assert m["record"]["elevation_sampling"] == {"mode": "nearest", "mask": "nearest source cell (unchanged)",
                                                 "rainfall": "nearest source cell (unchanged)"}


@pytest.mark.parametrize("case", ["one_ulp", "nearest_terrain_under_bilinear", "bilinear_terrain_under_nearest",
                                  "wrong_original_pin"])
def test_tampered_or_mismatched_bilinear_is_refused(synth, case):
    rot, target = dict(synth.rot, elevation_sampling=MODE), synth.tfull[1:-1, 1:-1].copy()
    message = "not the bilinear_full_valid_support_nearest_fallback sampling"
    if case == "one_ulp":
        i, j = np.argwhere(synth.used)[3]
        target[i, j] = np.nextafter(target[i, j], np.inf)
    elif case == "nearest_terrain_under_bilinear":
        target = synth.nfull[1:-1, 1:-1].copy()
    elif case == "bilinear_terrain_under_nearest":
        rot = dict(synth.rot)
        message = "not the inverse-nearest sampling"
    else:
        rot["original_terrain_sha256"] = "2" * 64
        message = "terrain sha256 differs"
    recipe = _recipe(synth.tmp / f"t_{case}.yaml", case, synth.bdtm, synth.tfull,
                     {"n_active": 0, "n_sinks": 0, "n_outlets": 0, "nz": 1}, rot)
    with pytest.raises(Plot1ImportError, match=message):
        cc.rotation_mapping(recipe, target)


# ------------------------------------------------------------------------------------- the real generated inputs
@pytest.fixture(scope="module")
def real():
    recipe_path = BILINEAR_INPUTS / "recipe_r1c2_rotated_bilinear_stable.yaml"
    if not recipe_path.is_file():
        pytest.skip("generated bilinear inputs absent")
    recipe = cc.load_chastre_recipe(recipe_path)
    rot = recipe.raw["rainfall_scaling"]["rotation"]
    original = cc.load_chastre_recipe(rot["original_recipe"]["path"])
    full = _to_maple(read_legacy_grid(Path(original.raw["terrain"]["path"]))[1])
    stored = _to_maple(read_legacy_grid(Path(recipe.raw["terrain"]["path"]))[1])
    return SimpleNamespace(recipe=recipe, rot=rot, full=full, stored=stored)


def test_real_bilinear_terrain_matches_scipy_order1_within_a_quantified_fp64_bound(real):
    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    shape = (real.stored.shape[0] - 2, real.stored.shape[1] - 2)
    x, y = cc.rotated_source_coordinates(real.rot["axis_vector_cells"], real.rot["interior_origin_numerators"], shape)
    rows, cols, _ = cc.rotated_source_indices(real.rot["axis_vector_cells"], real.rot["interior_origin_numerators"], shape)
    nearest, _ = cc.sample_source_cells(real.full, rows, cols, NODATA)
    _z, used = cc.bilinear_source_elevation(real.full, x, y, nearest, NODATA)
    stored = real.stored[1:-1, 1:-1]
    sci = scipy_ndimage.map_coordinates(real.full, [(y - 0.5)[used], (x - 0.5)[used]], order=1, mode="nearest",
                                        prefilter=False)
    zmax = float(np.abs(real.full[real.full != NODATA]).max())
    assert np.abs(stored[used] - sci).max() <= 16 * EPS * zmax     # quantified bound, not bit identity
    act = nearest != NODATA
    assert np.array_equal(stored[act & ~used], nearest[act & ~used]) and np.array_equal(stored != NODATA, act)
    assert int(used.sum()) == 1136609 and int((act & ~used).sum()) == 3580   # root investigation's counts


def test_real_bilinear_terrain_is_byte_exact_from_the_declared_code(real):
    spec = importlib.util.spec_from_file_location("_c1prep_real", C1_PREP)
    c1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c1)
    shape = (real.stored.shape[0] - 2, real.stored.shape[1] - 2)
    target_full, *_ = c1.derive(real.full, NODATA, tuple(real.rot["axis_vector_cells"]),
                                tuple(real.rot["interior_origin_numerators"]), shape)
    text = PREP.esri_text(target_full, NODATA).encode()
    assert hashlib.sha256(text).hexdigest() == real.recipe.raw["terrain"]["sha256"] == \
        _sha(real.recipe.raw["terrain"]["path"])
    # and the importer's own re-derivation accepts it
    m = cc.rotation_mapping(real.recipe, real.stored[1:-1, 1:-1])
    assert m["record"]["elevation_sampling"]["n_bilinear"] == 1136609
