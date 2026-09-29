# Phase 2: MAHLERAN Plot 1 audit and import into an actual MAPLE case

Status: implemented 2026-09-29 for task `phase2_plot1_import`; **not yet executed or reviewed**. Claude wrote the code, tests and this document but ran nothing, because Codex runs all scripts, tests and generation for this task. Every number below that is not quoted from a source file is either computed by the importer (and recorded in its report) or an expectation from reading the inputs. Expectations are labelled as such.

This is an initialization fixture only. There is no rainfall forcing, infiltration, routing, detachment or deposition physics, no wind run, and no GPU claim. It does not establish hydraulic or model acceptance.

Code: `src/maple_syrup/case_import.py`. Recipe: `cases/plot1/recipe.yaml`. Tests: `tests/phase2/`.

## 1. What the importer does

`python -m maple_syrup.case_import --recipe cases/plot1/recipe.yaml --output-dir NEW_DIR`

1. It resolves MAPLE through Phase 1 `resolve_maple_dependency` and checks `PHASE2_REQUIRED_MAPLE_API` (Phase 1 code is unchanged). It then refuses an existing output path, or a path inside the MAPLE tree, the MAHLERAN tree or the recipe directory.
2. It stages the Plot 1 rasters into `NEW_DIR/source/mahleran_input_p1/` (see §2) and copies the rainfall record into `NEW_DIR/syrup/rainfall/`.
3. It reads every raster with MAPLE's `read_source` and `build_imported_field` and cross-checks the two reads.
4. It resolves options, grain maps, surface fields, the terrain audit and the bed plan.
5. It writes `NEW_DIR/case.yaml` and `source/syrup_derived/composition_codes.npy`, the sidecar `syrup/plot1_fields.npz` and the report `syrup/plot1_import_report.json`.
6. It calls MAPLE `compile_case(NEW_DIR)` and then `load_compiled_case(NEW_DIR)`, and runs `check_compiled_plot1` on both. It requires the compiled and reloaded states to be array-identical and MAPLE's source digest to be unchanged.
7. It writes `syrup/plot1_binding.json`. Only a directory that contains this file is a completed import. On any failure after the directory exists, it writes `syrup/FAILED.json` with the traceback and preserves the partial directory as evidence.

`--no-compile` stops after step 5. `GIT_OPTIONAL_LOCKS=0` is set around `compile_case`: MAPLE's `git_commit_hash` runs `git rev-parse`/`git status --porcelain` in the MAPLE checkout, and the setting stops that from refreshing MAPLE's git index.

## 2. Sources and provenance

For the root `mahleran_input.xml`, the report records its path, SHA-256, version (`1.2.3`, checked against the recipe) and MAHLERAN git HEAD (read from files, without running git). The XML's `input_folder` `.\Input\input_p1\` must resolve to the recipe's `Input/input_p1`.

For each staged raster, the report records the original path, its SHA-256 and size, the staged SHA-256, the six header records and any trailer. Headers must agree across files (`ncols 22`, `nrows 62`, `cellsize 0.5`, `xllcorner`/`yllcorner 0`, `nodata_value -9999`), otherwise the import stops.

**DEM trailer.** From reading the file: `p1dem.asc` carries non-text bytes after its 62 declared rows. They mention `p1dem.asc` and `application/octet-stream` and look like an appended attachment record. The Fortran reader (`read_spatial_data.f90` 114-118) reads exactly `nrows` records and never sees them. MAPLE's reader rejects a body that holds anything but the grid. The staged file is therefore the byte-identical prefix of six header lines plus 62 rows. The trailer's length, SHA-256 and first bytes are reported. A clean file is copied byte-for-byte. A short, wrapped or non-ASCII grid is refused.

Maps that the XML references but the legacy setup does not read (`p1ksat290806.asc`, `p1sm290905.asc`) are hashed only. The rainfall series `p1_01_08_06.dat` is copied verbatim with its hash, and nothing interprets it: `rain_type 2`, `stormlength 5400 s`, forcing semantics are Phase 3.

## 3. Legacy options, raw to resolved

Source: `MAHLERAN_storm_setting_xml.f90` (line numbers as read 2026-09-29). The complete record is `report.legacy_options`.

| XML map / option | Read by the storm setup? | Resolution in this fixture |
|---|---|---|
| `dem` p1dem.asc | always (161-189, m to mm) | MAPLE surface after crop and constant datum offset |
| `vegetation-cover_map` p1vegcover.asc | always (226-227) | sidecar `vegetation_cover_fraction`; legacy consumer `raindrop_detachment.for` 24-34 (`KE (1 - 8.1e-3 veg%)`); **not** MAPLE vegetation |
| `surface-type_map` | always (240-270), clamped to [1, `number_of_surface_types`=1] | every cell resolves to type 1; raw and resolved both kept |
| `rainfall-scaling_map` rm_new.asc | always (275-276) as `rmask` | interior all 1.0; the south ring row is -9999 (`rmask < 0` excludes cells in topog_attrib and routing); rainfall scaling itself belongs to Phase 3 |
| `pavement_map` p1pavcoverveg.asc | name length > 5 and file exists (279-297) | grain rescaling (352-369); sidecar `pavement_cover_fraction`; `pave x 1e-4` infiltration use (375-379) left to Phase 3 |
| `particle_size_map` phi_1..6 | `use_map_phi = true` (299-311) | **defect**, corrected (§5) |
| `final_infiltration_map` p1ksat290806.asc | no (`use_final_infiltration_map=false`, type 2) | ksat comes from surface types: mean 0.00025, std 0.001, `normal`, which is stochastic and can go negative; unresolved, Phase 3 |
| `initial_soil-moisture_map` p1sm290905.asc | no | theta_0 = 0.25 deterministic |
| `saturated_soil-moisture_map` thetasat39.asc | yes (736-740) | sidecar `saturated_soil_moisture` |
| suction / drainage / friction maps | no (empty names, flags false) | psi 46.6 x `psi_mod` (from `calibration_xml`, not traced); drainage 0.05; friction factor type 1 = 21.45 |
| `soil_thickness` 0.3 | Phase 3 source correction: `initialize_values_xml.f90` 228–229 initializes retained and maximum soil water from thickness | a soil-water depth, **not** sediment inventory |
| `particle_density` 2.65 (twice, identical) | yes | 2650 kg/m3 for every MAPLE grain class |
| `active_layer_sensitivity` 1.52e-6 | yes | a detachment coefficient, **not** an active-layer thickness |
| `update_topography` "n" | `initialize_values_xml.f90` 138-142 | false |
| routing 5, sediment routing 2, `flow_direction` 4, dt 1 s, infiltration model 2 | recorded | Phases 3-5 |
| nutrients, markers, continuous section | n/a | excluded |

## 4. Grid, crop, orientation and ring

The legacy grid is 62 x 22 at 0.5 m. The computed loops are rows `i = 2..61` and columns `k = 2..21` (`nr = n_rows - 1`, `nc = n_cols - 1`, storm_setting 204-205; `topog_attrib.for` 44-45). The MAPLE grid is the 60 x 20 interior at 0.5 m (30 m x 10 m).

Orientation: the ESRI ASCII first data row is north, and MAHLERAN `i = 1` is that row. MAPLE row 0 is south. MAPLE's `ascii_grid` reader flips the rows. The crop transform `{crop: rows [1,61), cols [1,21)}` is the same window in either orientation because the ring is symmetric. The index map is MAPLE `(r, c)` = legacy 1-based `(i = 61 - r, k = c + 2)`. Every field is cropped with the same MAPLE transform, and each read is compared with MAPLE's own full-grid read. The tests compare them again with an independent parse.

Ring evidence (`report.grid.ring_evidence`) gives, per field and side, the range of ring minus adjacent interior. Expected from reading the DEM:

- the north ring is the adjacent row + 0.01 m;
- the south ring is the adjacent row - 0.01 m;
- the west and east rings equal the adjacent columns;
- the grain, cover and theta_sat ring cells duplicate their neighbours;
- `rm_new` marks only the south ring row -9999.

No ring cell enters MAPLE state. The ring is legacy boundary scaffolding, not plot sediment, so no ring mass is created and none is lost.

## 5. Grain composition (scientific decision)

**Defect.** The root XML maps `phi_1..phi_6` all to `plot1_phi1.asc`. As configured, the six raw "fractions" are six copies of phi_1, with sums of about 1.146-1.398 (Codex check; recomputed in `grain_maps.xml_as_configured`). The post-setup sums after the legacy pavement rescaling are also recorded. This is a diagnostic only and is never used as a composition.

**Decision (user/Codex, applied by the recipe).** This fixture is a documented **corrected fixture**, not an exact reproduction of the root XML:

1. It uses the six distinct supplied maps `plot1_phi1..6.asc`, in MAHLERAN order phi_1 (finest) .. phi_6 (coarsest).
2. It normalizes float-storage roundoff only. Each cell's six values must sum to 1 within `closure_roundoff_tolerance` = 1e-6, and are then divided by that sum. A larger departure, a negative value or NaN is rejected, never rescaled. Codex measured interior sums of 0.9999999254941929..1.0000000707805157, and a test checks this independently.
3. It applies the legacy pavement rescaling (storm_setting 352-369), faithfully: `grav = phi5 + phi6`, `fines = 1 - grav`, `p = pave/100`. Where `p > 0` and `grav != 0`, `phi1..4 *= (1-p)/fines` and `phi5..6 *= p/grav`; elsewhere the fractions are unchanged. With closed input this sets the gravel share equal to pavement cover and keeps closure. Where the legacy code would divide by zero (`fines == 0`, `p > 0`), or where the cover lies outside 0-100 %, the importer rejects the input. The legacy loops cover `i, k = 1..n-1` and never touch the far ring, which is excluded anyway.

   A known consequence of this legacy convention is that a cell with pavement exactly 0 keeps its mapped gravel share, while a cell with 0.1 % cover gets gravel = 0.001.

Normalization change (below 1e-7 per class) and rescaling change (a model choice, of order 0.1) are reported separately under `grain_maps.roundoff_normalization` and `grain_maps.pavement_rescaling`. The pavement fraction comes from MAPLE's percent conversion (`x 0.01`) rather than `/100`. The two can differ by 1 ulp.

## 6. Terrain audit (diagnostic, no routing)

`legacy_d4_audit` reproduces the `topog_attrib.for` aspect rule (85-117, `ndirn = 4`). It compares elevations in mm with a strict `<` in N, E, S, W order, so ties are never receivers and aspect 0 marks a sink. It then follows each interior cell's descent path to its end, which is either a sink or a ring cell (side and `rmask`). Outputs, on the MAPLE grid:

- `legacy_d4_aspect`, where 1 = N (+y), 2 = E, 3 = S (-y), 4 = W;
- `legacy_d4_sink`, split into `legacy_d4_flat_sink` (an equal neighbour exists) and `legacy_d4_strict_pit`;
- `hydraulic_edge_outflow_side`, and `hydraulic_candidate_outlet` (the receiver is a ring cell);
- `hydraulic_drains_to_ring_side` and `hydraulic_drains_to_masked_ring`.

Counts and a sink list with legacy and MAPLE indices are in `report.terrain.d4_audit`.

Expectation from the DEM: north and side rings never receive, because they are higher or equal. The south ring is lower than every adjacent cell, so the south edge is the candidate outlet.

These masks are SYRUP data kept separate from MAPLE's boundary metadata. They are not an outlet decision, a pit policy or a routing graph. Nothing is filled or edited. Measuring how much legacy ring deposition or loss would be counted as export, and deciding the supported pit behaviour, remain Phase 4 tasks.

## 7. The MAPLE case

`case.yaml` goes through MAPLE's unmodified compiler, with no parallel state builder:

- **`geometry`**:
  - nx 20, ny 60, 0.5 m cells;
  - `voxel_dz_m` 0.1, `bulk_density_kg_m3` 1250, `active_layer_thickness_m` 0.002;
  - both axes `prescribed` with zero inflow for every class. This is MAPLE sediment and wind boundary metadata for a non-periodic plot. It is not a hydraulic wall or outlet.
- **`topographic_wind: {enabled: false}`**: MAPLE requires periodic axes for its FFT wind operator. A future wind event on this plot needs its own boundary and wind decision (Phase 8).
- **`grain_classes`** `phi_1..phi_6`: diameters from MAPLE `MAHLERAN_1_2_1_CLASS_DIAMETERS_M`, which equals MAHLERAN 1.2.3 `shared_data.f90`:179 radii x 2. Particle density 2650.
- **`topography`**: `base: imported`, `nz` from the bed plan, `perturbation.relief_m: 0.0`.
- **`import.elevation`**: the staged `p1dem.asc` (`ascii_grid`, metres), transforms `crop`, then `datum_offset {reference: none, offset_m: <declared constant>}`. `import.domain.label: empirical_transformed`.
- **`import.sediment`**: `mode: categorical_map`.
  - The category raster is `source/syrup_derived/composition_codes.npy`, south-to-north, one integer code per distinct resolved composition vector. The profile table gives each code one depth interval `[0, nz*dz]` with that exact composition, so MAPLE fills every column with a homogeneous mixture through its own `build_layered_requests`, `deposit_surface_mixture_batch` and `initialize_active_layer_from_voxels`.
  - **Seam:** MAPLE has no per-cell grain-fraction-map import. The categorical path reproduces the per-cell fractions exactly (checked: the table reproduces every cell). It is a lookup, not a soil-unit classification. A narrow MAPLE extension (a `fraction_maps` sediment mode, one raster per class, closure validated) would be the cleaner seam. It is proposed, not required.
- **`sediment_availability.global_available_fraction: 1.0`**: stated explicitly. MAPLE availability is the aeolian available/bound split. Legacy pavement and vegetation effects stay SYRUP sidecar fields for the later water laws. MAPLE moisture, vegetation and threshold-modifier fields are left at their neutral or empty defaults and do **not** represent legacy water effects.
- There is no `water_initial_depth`, so the water state is dry, mobile mass is zero and the ledger is empty. No water coupling is enabled.

## 8. Bed assumptions and vertical datum

None of these has a MAHLERAN value. All are configurable in the recipe.

- **Voxel `dz` 0.1 m and active layer 0.002 m** are MAPLE-style defaults. The legacy `active_layer_sensitivity` is unrelated.
- **Bulk density 1250 kg/m3** is an assumption, the MAPLE default. The MAHLERAN source and XML contain no bulk density or porosity. One alternative was investigated: if `theta_sat` (0.39) were total porosity, the bulk density would be about `(1 - 0.39) x 2650 = 1616 kg/m3`. It was not adopted, because theta_sat is a hydraulic parameter and need not equal total porosity. The report records both values.
- **Datum.** MAPLE elevation = legacy DEM elevation + `datum_offset_m`, and MAPLE z = 0 is the base of the erodible column. The offset lifts the lowest interior cell to at least `minimum_fill_depth_m` = 0.3 m, rounded **up** to 1 mm. The same constant applies everywhere, so DEM slopes are preserved exactly; a check requires this. This finite erodible depth is an assumption, not an equivalence with a legacy inventory. It is unrelated to `soil_thickness`, although both happen to be 0.3.
  - Expectation from the DEM: interior range about 0.040-1.537 m, so the offset is 0.26 m and fill depths are about 0.300-1.797 m.
- **Allocation.** `nz = ceil((max fill + headroom 0.2 m) / dz)`. The expected value is `nz = 20` (2.0 m column, at least 0.2 m of empty headroom), about 1.2 MB of voxel mass at FP64 for 60 x 20 x 20 x 6. The report gives the actual per-cell fill range and mean, the offset, `nz`, the headroom range and the highest occupied level.
- **Stratigraphy.** Each column is homogeneous, using the cell's resolved surface composition. No measured stratigraphy is invented.

## 9. Sidecar and binding

`syrup/plot1_fields.npz` holds MAPLE-grid arrays (row 0 = south, `(60, 20[, 6])`):

- the source elevation, and the MAPLE elevation;
- vegetation, pavement, surface type (raw and resolved), rainfall scaling and theta_sat;
- the raw, normalized and final fractions, and the XML-as-configured sums;
- the composition codes and table;
- the expected mass by cell and class;
- the D4 audit masks;
- the 62 x 22 legacy DEM and `rmask` (also row 0 = south) as ring evidence.

`syrup/plot1_import_report.json` holds the full audit and the field manifest, including the npz SHA-256. `syrup/plot1_binding.json` binds everything:

- MAPLE `case_identity_sha256`, `code_version` and the artifact SHA-256 map;
- the SHA-256 of `provenance.yaml` and `case.yaml`;
- the MAPLE source inventory;
- the report and npz SHA-256;
- the grid and class ids;
- compiled and reloaded check results;
- MAPLE and SYRUP provenance.

## 10. Checks

`check_compiled_plot1` runs on both the compiled and the reloaded case. It compares them with values computed from the source arrays and the declared assumptions. It raises on failure.

- Geometry, class order, diameters and density.
- Elevation equals the DEM plus offset, the perturbation is zero, and slopes are unchanged.
- Per-cell and per-class bed mass equals `cell area x 1250 x fill depth x final fraction`. The tolerance is `(2 nz + 4) x mass_resolution` plus FP64 roundoff. Domain class totals and MAPLE's own expected totals are checked as well.
- Voxel levels are full from the base up to `fill - 0.002` m, the highest occupied level is as predicted, and the declared headroom holds.
- The active layer is 2 mm of the cell's own composition.
- MAPLE's `check_voxel_column_state`, `check_active_layer_voxel_partition` and `diagnose_combined_surface_state` agree.
- Water depth, mobile mass, mobile seed and every numeric ledger field are zero. Availability matches the declared fraction. MAPLE vegetation is absent.
- There are no synthetic, excluded or gap-filled cells, and every cell is empirical core.

Tests in `tests/phase2`, which read the MAHLERAN tree only:

- `test_plot1_units.py` (fast, no data): closure and roundoff; the legacy rescaling against a per-cell transcription; the division-by-zero and cover-range rejections; the repeated-map rejection; D4 on hand-built grids; trailer, clean, short and wrapped staging; no overwrite; the bed plan; strict recipe validation.
- `test_plot1_audit.py` (real data, no compile):
  - crop and orientation, and shared crop for all fields;
  - source hashes and the DEM trailer;
  - the XML defect, and composition against an independent parse plus formula, with roundoff reported separately from rescaling;
  - D4 against a brute-force loop transcription, and ring facts;
  - option resolution and the bed plan;
  - rejection of a mismatched shape, interior nodata, material nonclosure and a missing map, each on a disposable copy.
- `test_plot1_compile.py` (one real compile per module):
  - binding hashes and identity;
  - `case.yaml` import paths;
  - reload through MAPLE `load_compiled_case`, and inventory, voxel and active-layer checks against an independent parse;
  - empty water, mobile and ledger;
  - no overwrite of an existing output;
  - `--no-compile` CLI;
  - no `outputs/` (no run launched).

## 11. Commands (for Codex)

```
cd /home/okin/SYRUP
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m pytest -p no:cacheprovider tests/phase2/test_plot1_units.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m pytest -p no:cacheprovider tests/phase2 -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m pytest -p no:cacheprovider tests/phase1 -q
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m maple_syrup.case_import \
    --recipe cases/plot1/recipe.yaml \
    --output-dir /home/okin/SYRUP/agent_handoffs/tasks/phase2_plot1_import/plot1_case \
    --expected-maple-root /home/okin/MAPLE
```

Any new directory works as the output. Running the command twice with the same path is refused.

## 12. Departures and remaining issues

| Item | Departure / status |
|---|---|
| Grain maps | Six distinct maps instead of the XML's repeated `plot1_phi1.asc`. This is a corrected fixture, and a matched legacy benchmark must use the same correction. |
| Roundoff normalization | New step, with tolerance 1e-6; it does not occur in the legacy code. |
| DEM staging | Trailer bytes after row 62 are not staged. This is equivalent to what the Fortran reader consumes. |
| Ring | Excluded from MAPLE state. The ring is evidence and candidate-outlet data only. |
| Composition over depth | The homogeneous column and finite erodible depth are assumptions. MAPLE's evolving active-layer composition will later differ from MAHLERAN's fixed proportions by design. |
| Pavement/vegetation | Kept as SYRUP fields. Where they enter infiltration and rain detachment is Phase 3/5 work. |
| Stochastic ksat (`normal`, std > mean) | Not resolved (Phase 3). |
| Outlets and pits | The audit exists; the policy and the export measurement are Phase 4. |
| MAPLE seams | There is no per-cell fraction-map import (categorical workaround, proposed `fraction_maps` mode). MAPLE's transform log computes crop volumes with the case extent, not the 62 x 22 source extent; this is cosmetic, since placement uses array indices. `compile_case` shells out to git. |
| GPU | Not addressed. Initialization is host-side; device placement belongs to later phases through `to_device_tree`. |
| Execution | Nothing in this phase has been run by Claude. |
