"""Phase 2.5 native Python example (NOT a command): species trait validation -> LEAF canopy/root projection -> explicit site controls -> strict S0 binding -> compact cached recipe (capture, export,
cold stage) -> the native vegetation law on a prescribed floating column (live and cached bindings give the SAME law). Documentation example; CPU only; no GPU.

What it does NOT do: it does not apply a vegetation generation to a lifecycle Boundary (that needs a genuinely completed native event and is exercised by the test suite
`tests/phase25_vegetation_lifecycle`; see section 9 of docs/subsurface_hydrology/phase25_vegetation.md), it writes no frame and no physical checkpoint, and it calibrates nothing: every
number below is a declared FIXTURE value.

Runtime (nothing is installed by this script): put on PYTHONPATH, in this order, the pinned native MAPLE `<MAPLE>/editable` and `<MAPLE>/source/src`, the SYRUP `src` of the source under test, and the LEAF
ROOT that supplies BOTH the `leaf` package and the `benchmarks` helper directory. The script asserts the actual imported package digests against the pins below, prints every imported path, and exits
non-zero on any mismatch. Example (paths are the reference runtime of the qualification records; adjust to your own checkout):

    export MAPLE=.../maple-materialized/5d850789f9e82dc456227cb1eb4dfcc255b3ee4dfb0ed74883813c93e98d0ab3
    export PYTHONPATH=$MAPLE/editable:$MAPLE/source/src:/path/to/SYRUP/src:/path/to/LEAF_ROOT
    CUDA_VISIBLE_DEVICES= NUMBA_CACHE_DIR=/tmp/phase25_example_numba PYTHONDONTWRITEBYTECODE=1 \
        /path/to/MAPLE/.venv/bin/python -B phase25_projection_example.py [--out result.json]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# External expected pins (from the Root acceptance records of the release; a production run takes them from ITS release record, never from the files being checked).
PINS = {"syrup_package": "37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0", "syrup_files": 180,   # the accepted Phase 2.5 source package (SYRUP commit bb28af6c58c2892cb3c065d73d33b88285c498f0)
        "maple_package": "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e", "maple_files": 412,
        "leaf_package": "38009872f8969f86b56c720adc1b5d9ffefc554fb9cacf90855d999db604530f", "leaf_files": 28,
        "leaf_helper": "333921bb645d984fa8e7449619d03114262ee333c7fea61a22493a878fd9ccc5"}
BIG = 1 << 30
A1, A2 = 0.013, 0.067  # the reference pair of the descendant source, passed EXPLICITLY as site controls (never a default of the API)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="optional JSON result file")
    out_path = ap.parse_args().out

    import maple
    import maple_syrup
    from maple_syrup.provenance import source_tree_digest

    import leaf
    from leaf import state, traits
    from leaf import water_vegetation as wv

    # ---- 1. the actual runtime: paths, package digests, helper (nothing is assumed installed)
    runtime = {}
    for name, mod, want, files in (("maple", maple, PINS["maple_package"], PINS["maple_files"]), ("leaf", leaf, PINS["leaf_package"], PINS["leaf_files"]),
                                   ("syrup", maple_syrup, PINS["syrup_package"], PINS["syrup_files"])):
        d = source_tree_digest(Path(mod.__file__).resolve().parent)
        runtime[name] = {"file": str(mod.__file__), "package_sha256": d.digest_sha256, "files": d.file_count}
        if d.digest_sha256 != want or d.file_count != files:
            print(f"PIN MISMATCH for {name}: {d.digest_sha256} ({d.file_count} files)", file=sys.stderr)
            return 2
    helper = wv.external_source_record()["helpers"][0]
    spec = importlib.util.find_spec("benchmarks.spatial_mapping")
    runtime["helper"] = {"module": helper["module"], "origin": spec.origin, "sha256": helper["sha256"], "bytes": helper["bytes"]}
    if helper["sha256"] != "sha256:" + PINS["leaf_helper"]:
        print("PIN MISMATCH for the external helper benchmarks.spatial_mapping", file=sys.stderr)
        return 2
    import benchmarks
    runtime["benchmarks_namespace_path"] = [str(p) for p in benchmarks.__path__]

    # ---- 2. species trait validation (catalog schema_version 2, per-species receiving weights, fixed roots)
    doc = json.loads((HERE / "species_v2.json").read_text())
    cat = traits.SpeciesCatalog(doc)
    weights = {sid: cat.preferential_receiver_weights(sid).tolist() for sid in cat.species_ids}
    roots = {sid: list(doc["species"][sid]["roots"]["layer_fractions"]) for sid in cat.species_ids}
    trait = {"schema_version": cat.schema_version, "root_layers": cat.root_layers, "receiving_weights": weights, "root_fractions": roots, "catalog_hash": cat.hash}

    # ---- 3. LEAF canopy/root projection on a declared 3 x 3 fixture (dx = 1 m; every circle lies inside its cell)
    dom = state.Domain(3, 3, 1.0, origin_x_m=0.0, origin_y_m=0.0, periodic=(False, False))
    layout = [(1, "grass", 0.5, 0.5, 0.4, "live"), (2, "shrub", 1.5, 1.5, 0.35, "live"), (3, "shrub", 2.5, 2.5, 0.30, "dead_standing")]
    recs = [{"plant_id": pid, "species": sp, "x_m": x, "y_m": y, "establishment_ground_m": 0.0, "agb_dry_kg": float(cat.biomass_from_diameter(sp, 2.0 * r)), "seasonal_agb_dry_kg": 0.0,
             "age_days": 10.0, "status": status} for pid, sp, x, y, r, status in layout]
    st = state.PlantState.from_records(cat, dom, recs)
    surface, mask = np.zeros((3, 3)), np.ones((3, 3), dtype=bool)
    proj = wv.project_from_initial_state(st, cat, surface, active_mask=mask, max_edges=1 << 16, max_bytes=BIG)
    f = lambda r: float(np.pi * r * r)  # cell area 1 m2
    check = {"splash_cell(0,0)": (float(proj.splash_cover[0]), f(0.4)), "R1_cell(0,0)": (float(proj.matrix_root_index[0]), f(0.4) * roots["grass"][0]),
             "splash_cell(2,2) dead standing": (float(proj.splash_cover[8]), f(0.30)), "shading_cell(2,2) live only": (float(proj.shading_cover[8]), 0.0),
             "R1_cell(2,2) live only": (float(proj.matrix_root_index[8]), 0.0)}
    for k, (got, want) in check.items():
        assert abs(got - want) <= 1e-12 * max(1.0, abs(want)), (k, got, want)  # closed-form geometry (pi r^2 / cell area)

    # ---- 4. explicit site controls and the strict binding (every pin is an external expectation)
    from maple_syrup import water_vegetation_binding as wvb
    ctl = wvb.site_controls(a1=A1, a2=A2, a3=0.4, max_fraction_per_hour=0.7, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0, 1.0], n_layers=3, n_cells=9, max_bytes=BIG)
    ident = proj.identity_record()
    source = {k: ident["source"][k] for k in ("kind", "state_sha256", "catalog_hash")}
    binding = wvb.bind_projection(proj, ctl, expected_leaf_package_sha256=PINS["leaf_package"], expected_projection_sha256=ident["content_sha256"], expected_source=source,
                                  expected_external_helpers=[{"module": helper["module"], "sha256": helper["sha256"], "bytes": helper["bytes"]}], active_mask=mask, current_surface=surface, max_bytes=BIG)

    # ---- 5. the compact cached recipe: capture -> closed record -> cold stage with external pins -> cached binding
    from maple_syrup import water_vegetation_held as wh
    rec = wh.capture_cached_inputs(binding, max_bytes=BIG)
    export = wh.export_cached_record(rec)
    meta = json.loads(json.dumps(export["meta"]))
    arrays = {k: np.array(v) for k, v in export["arrays"].items()}
    staged = wh.stage_cached_inputs(meta, arrays, expected_record_sha256=rec.record_sha256, expected_binding_sha256=binding.sha256, expected_projection_sha256=ident["content_sha256"],
                                    expected_leaf_package_sha256=PINS["leaf_package"], expected_external_helpers=[{"module": helper["module"], "sha256": helper["sha256"], "bytes": helper["bytes"]}],
                                    expected_source=source, active_mask=mask, max_bytes=BIG)
    assert staged.record_sha256 == rec.record_sha256 and staged.origin == "cached_derived" and staged.live_session_proof is False
    cbind = wh.cached_binding(staged, expected_binding_sha256=binding.sha256, max_bytes=BIG)

    # ---- 6. the native law on a PRESCRIBED floating column (cell area must equal the LEAF cell area exactly)
    from maple_syrup import floating_soil_layers as fsl
    col = fsl.prepare_floating_column(np.array([0.0, 0.1, 0.3, 0.6]), theta_residual=0.1, theta_field_capacity=0.3, theta_saturation=0.5, diffusivity_m2_s=np.zeros(2), theta_evaporation_stress=0.2,
                                      diffusion_mode="constant", bottom_drainage="open", soil_depth_m=np.full(9, 0.6), active_mask=np.ones(9, dtype=bool), cell_area_m2=1.0, backend="numpy",
                                      evaporation_factor=1.0)
    law_live = wvb.prepare_vegetation_law(col, binding, conductivity_m_s=2e-6, suction_m=0.3, memory_budget_bytes=BIG)
    law_cached = wvb.prepare_vegetation_law(col, cbind, conductivity_m_s=2e-6, suction_m=0.3, memory_budget_bytes=BIG)
    assert law_live.law_sha256 == law_cached.law_sha256 and law_live.g_sha256 == law_cached.g_sha256 and law_cached.binding_origin == "cached_derived"
    refused = None
    col_bad = fsl.prepare_floating_column(np.array([0.0, 0.1, 0.3, 0.6]), theta_residual=0.1, theta_field_capacity=0.3, theta_saturation=0.5, diffusivity_m2_s=np.zeros(2), theta_evaporation_stress=0.2,
                                          diffusion_mode="constant", bottom_drainage="open", soil_depth_m=np.full(9, 0.6), active_mask=np.ones(9, dtype=bool), cell_area_m2=0.01, backend="numpy",
                                          evaporation_factor=1.0)
    try:
        wvb.prepare_vegetation_law(col_bad, binding, conductivity_m_s=2e-6, suction_m=0.3, memory_budget_bytes=BIG)
    except wvb.VegetationBindingError as exc:
        refused = str(exc)[:110]
    assert refused is not None, "a column with another cell area must be refused"
    g_expected = (A1 + np.asarray(proj.matrix_root_index)) / (A2 + np.asarray(proj.matrix_root_index))  # Tietjen-inspired g = (a1 + R1) / (a2 + R1), declared adaptation
    # ---- the three package digests and the helper again AFTER the work: nothing changed while the example ran (cold re-read of the actual files)
    after = {}
    for name, mod in (("maple", maple), ("leaf", leaf), ("syrup", maple_syrup)):
        d = source_tree_digest(Path(mod.__file__).resolve().parent)
        after[name] = {"package_sha256": d.digest_sha256, "files": d.file_count}
        assert after[name] == {"package_sha256": runtime[name]["package_sha256"], "files": runtime[name]["files"]}, f"{name} changed during the run"
    helper_after = wv.external_source_record()["helpers"][0]
    assert helper_after["sha256"] == helper["sha256"], "the external helper changed during the run"
    runtime["after"] = {**after, "helper_sha256": helper_after["sha256"]}
    result = {"runtime": runtime, "trait": trait,
              "projection": {"splash": np.asarray(proj.splash_cover).round(6).tolist(), "shading_live_only": np.asarray(proj.shading_cover).round(6).tolist(),
                             "R1": np.asarray(proj.matrix_root_index).round(6).tolist(), "g_expected": g_expected.round(6).tolist(),
                             "species_enabled_for_preferential": list(binding.enabled_species), "content_sha256": ident["content_sha256"], "omitted_ids": [int(v) for v in proj.omitted_ids]},
              "binding_sha256": binding.sha256, "record_sha256": rec.record_sha256, "cached_origin": staged.origin, "law_sha256": law_live.law_sha256, "g_sha256": law_live.g_sha256,
              "g_min_max": [law_live.g_min, law_live.g_max], "wrong_area_column_refused": refused}
    text = json.dumps(result, indent=1)
    print(text)
    if out_path:
        Path(out_path).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
