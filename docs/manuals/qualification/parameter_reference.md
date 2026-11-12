# Subsurface parameter files — reference

Queued follow-up, 2026-11-12 UTC: the user added
[Phase 1.5 daily recharge](phase15_daily_recharge.md) after Phase 1 is completed,
validated, committed and pushed, and before Phase 2. It is not implemented by
this document's current interfaces. Positive P <= T attempts simplified recharge;
zero rain is a no-op; P > T uses the native full storm, with unchanged-parent
whole-occurrence fallback. Its new nominal-calendar convention must be explicit
and must not retrofit historical clocks. No Phase 1.5 parameter filenames, options
or production API are established, and nothing below accepts them.

## Current status and sources

This reference describes the files read by the scheduled water command
`masy.native_water_run` of the Phase 1 subsurface release and agrees with the
practical guide (Section 14) and the Model Structure and Physics document
(Section 13). It describes the committed sources, not the historical task
snapshots (kept in a labelled subsection at the end).

| Source | Commit | Package SHA-256 |
| --- | --- | --- |
| SYRUP (layered water) | `e40a1b8fc56a96920db4ffbd182d4068fa5de3e8`, branch `feature/subsurface-phase1` | `afcad53691b9d295ed560116d313419c08fc5ffe9b227a4bdd9c922f93c52671` (109 source files) |
| LEAF | `52f39fe06166a91e768740b19d3c26d4ffceaf6e` | `8da24af5a672138bab335d8fd1d4518c5a17ae90db9b400c5d211cc0c768fa4b` (23 source files) |
| MASY (scheduler) | `a70400230f26d7931624d49173196a1cfa05a90a` | `89dc8b392da61f7a8c4f4f5eeef46e88b524af24bfd68de3836476f00718b157` |
| Native MAPLE (historical, explicitly pinned; not the live latest) | `9f466d2cfcd52991b24496cf67f4562d6268afa9` | `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e` |

The SYRUP, LEAF and MASY commits are on the branch `feature/subsurface-phase1`
of `gregokin/SYRUP`, `LEAF` and `MASY`, pushed and verified against the remotes;
MAPLE 9f is an ancestor of the current `main` of `gregokin/MAPLE` (remote `main`
`1dca8539b87bf80e2b4bb72d4c6637d02013a305`). A later documentation commit does not
change these code identities.

Bounded qualification (exact receipts below): the committed runtime ran the full
public two-layer 4×6 dense example (interevent, a 0.25 s rain with native
recession to genuine completion, interevent) to completion on NumPy, Numba and
CUDA; same-device pause after the first interevent, resume and pause after the
completed storm, and resume to the end reproduced the uninterrupted run exactly on
the CPU and on CUDA: of the 45 stored physical entries, 29 floating-point and 15
other state arrays (44 in all) are identical (maximum relative difference 0) and the
45th is a provenance record that correctly differs by backend and source paths and
is checked separately; plant arrays are identical and the four biology files are
byte-identical. The three backends agree on the same 44 state arrays.
Six refusal controls refused before any step and left the accepted checkpoint
unchanged; 154 portable scientific controls passed. Runs keep
`production_qualified: false` and report `production_runtime_verified: true`.
This is **not** a field validation or wind, moving-soil, large-domain,
repeated-storm-recovery, vegetation-cover-feedback, performance or scaling
qualification, and existing storm defaults are unchanged.

The central public qualification record is
[Phase 1 release qualification](phase1_release_qualification.md) with its JSON
(`phase1_release_qualification.json`) and the reviewed recipe and receipt files in
`phase1_recipes/`; they record the exact commits, package digests, evidence, the
explanation of the CPU/CUDA scenario-hash comparison (both complete file sets are
byte-identical; the recorded weather digest covers the weather library file whose
path differs between the two independently sealed stores; that one field alone is
allowed in the comparison) and the limits. Machine-specific recipe pins in those
files are evidence, not reusable inputs.

Historical task receipts (kept for traceability, labelled historical; paths
relative to `agent_handoffs/tasks/`): `phase1_publication_c1_20261112/root_c1_*.json`
and the component receipts of `phase1_cuda_calendar_budget_20261112/`,
`phase1_gpu_calendar_impl_20261112/`, `phase1_leaf_cli_finalize_20261112/`,
`phase1_leaf_latest_api_20261112/` and `phase1_production_runtime_impl_20261112/`.

## The two command schemas

| `run.yaml` schema | Contents | Source modes |
| --- | --- | --- |
| `masy.native_water_run.v1` | Standalone water-only scheduled run: layered water and interevent evaporation; enabled transpiration or growth is refused. | `development_snapshot`; `production_committed` refuses (`production_recipe_unavailable`). |
| `masy.native_water_run.v2` | v1 plus one closed `leaf` block: actual LEAF plants, withdrawals and growth. A v2 file without `leaf` is refused and a v1 file with `leaf` is refused. | `development_snapshot` and `production_committed`. |

Both run on the CPU (`numpy`, `numba`) or on one CUDA device. Neither runs wind.

## File layout (public input set)

The portable input set is `examples/native_leaf_calendar_cli/` of the MASY
commit (the production template is `examples/native_production/`):

| Relative file | Contents |
| --- | --- |
| `case/case.yaml` | MAPLE terrain, dimensions, grains, voxels, active layer and `location.latitude_deg` (the only latitude source). |
| `case/model.yaml`, `case/wind.yaml` | Existing MAPLE configuration. |
| `layered_water.yaml` | Soil layers, water properties, initialization, infiltration, evaporation partition and hydraulic boundaries (below). |
| `scenario/scenario.json` | Calendar start, ordered event references with SHA-256 digests and the weather reference (`masy.scenario.v1`; `native_epoch_s` must be 0). |
| `scenario/weather.json` | Dated shared weather and the PET method (`masy.weather.v1`). |
| `scenario/events/ie_a.json`, `ie_b.json` | Interevent (`masy.interevent.v1`): `duration_s` and `processes.evaporation`, `processes.transpiration`, `processes.growth`. |
| `scenario/events/rain_a.json` | Rain (`masy.rain_event.v1`): `water_profile_id`, `intervals` of `duration_s` and `intensity_mm_h`, and `recession_guard_s`. |
| `catalog.json` | LEAF species catalog (`leaf.species_catalog`): units, `root_layers` (must equal the soil layers), per-species allometry, `roots.layer_fractions`, growth law and traits, and water-use law. |
| `plants.csv`, `plants.metadata.json` | LEAF plant table (nine columns: `plant_id, species, x_m, y_m, establishment_ground_m, agb_dry_kg, seasonal_agb_dry_kg, age_days, status`) and its required companion metadata, written together by LEAF. `establishment_ground_m` may be `SURFACE`, resolved from the retained initial bed. |
| `run.template.yaml`, `fill_config.py` | The closed run configuration template and a helper that copies the inputs outside the source tree and fills the absolute paths. |

`scenario.json` records the SHA-256 of the weather and event files it names, and
`run.yaml` is a closed schema. These values are small test settings, not
calibrated parameters, and neither water-file example is a recommended set.
Initial vegetation belongs to LEAF, separately from the temporal schedule; the
`cover` and `preferential.species` values of the water file are prescribed
hydrology inputs, not LEAF plants. Wind, rain and interevent periods do not
overlap; evaporation, transpiration and growth occur only during interevent
periods, never during rain or recession, and nothing is caught up afterwards.
The dated calendar advances through every period.

## `run.yaml` keys

| Block | Keys and units |
| --- | --- |
| top level | `schema`, `run_id`, `maple_case_dir`, `syrup_parameters`, `scenario`, `profile_id`; paths other than `source` are relative to `run.yaml`. |
| `source` (development) | `mode: development_snapshot`, `origin` (`label`, `declared_by`), `profile`, `native_target`, `masy_root`, `masy_manifest`, `syrup_root`, `syrup_manifest`. The roots are read-only sealed copies; every output is labelled `production_qualified: false`. |
| `source` (production, v2 only) | `mode: production_committed`, `profile`, `recipe`, `recipe_sha256`, `receipt`, `receipt_sha256`, `production_store`, `production_lock`, `release_masy_root`, `leaf_capsule_namespace`. The recipe identity is the SHA-256 of the recipe file. |
| `source.profile` | `native9f_layered_factory_afca_cpu_v1`, `native9f_layered_factory_afca_cuda_v1` (v1 and v2) or the earlier `native9f_layered_factory_95ef_cpu_v1` (v1 only, historical). It must match `compute.backend` of the water file. |
| `execution` | `quiet_policy: default` (60 s hold; depth change 1e-8 m; unit discharge 1e-10 m²/s; outlet discharge 1e-9 m³/s; excess depth 1e-8 m; excess volume 1e-6 m³). No override. |
| `limits` | `interevent_max_dt_s`, `interevent_min_dt_s` (s), `interevent_max_retries`; `storm`: `max_dt_s`, `min_dt_s` (s), `max_retries`, `max_steps` (native work per call), `max_recession_s` (s); `max_occurrences`, `max_dates_per_interevent`, `max_accepted_steps_per_occurrence`, `call_step_budget`, `max_record_bytes`. |
| `resources` | `caller_reserve_bytes`, `temporal_reserve_bytes`, `lifecycle_extra_reserve_bytes`, `preparation_reserve_bytes`, `host_available_cap_bytes` (null or bytes). No default reserves. Modelled memory, not RSS. |
| `output` | `run_dir` (must not exist), `step_cadence`, `max_output_bytes`, `max_step_lines`, `max_file_bytes`, `physical_checkpoint` (`none`, `terminal_standalone` for v1 CPU, or `combined_terminal`), `compile_output_reserve_bytes`. Caps are hard and never raised. |
| `cuda` | CUDA profile only (refused otherwise): `device_id`, `device_reserve_bytes`, `device_cap_bytes`, `native_reserve_bytes` (bytes). |
| `leaf` | v2 only: `leaf_variant: 8da`; `leaf_root` (development only; refused in production, where it is derived from the sealed capsule); `catalog`, `plants`; `cadence_days` (integer 1–7); `geometry` (`max_edges`, `max_bytes`, `canopy_max_edges`, `canopy_max_bytes`, `builder` `numpy` or `compiled`); `budgets` (`leaf_budget_bytes`, `checkpoint`: `max_host_bytes`, `max_device_bytes` (0 on CPU), `max_disk_bytes`, `max_biology_bytes`); `io` (`max_catalog_bytes`, `max_plants_bytes`, `max_plant_records`). |

## Command line

```
python -B -m masy.native_water_run manifest ROOT OUT
python -B -m masy.native_water_run run CONFIG
python -B -m masy.native_water_run run CONFIG --pause-after-occurrences N
python -B -m masy.native_water_run run CONFIG --resume CKPT --expected-seal-sha256 SHA [--pause-after-occurrences N]
```

Run with the interpreter of the MAPLE environment named by the runtime store,
with the release MASY `src` on `PYTHONPATH`. The options are not configuration
keys. A pause needs `output.physical_checkpoint: combined_terminal` and
writes the sealed combined artifact (`physical/`, for a v2 run also `biology/`,
`temporal.json` and a sealed manifest) at an accepted occurrence boundary; resume
goes to a fresh `run_dir`, needs the seal, re-pins every input, allows only the
three resource fields of the calendar limits (`caller_reserve_bytes`,
`temporal_reserve_bytes`, `max_record_bytes`) to differ, and refuses a changed
source, backend or artifact family. Exit codes: 0 complete, 1 internal error,
2 refused before physical work, 3 failed or incomplete, 4 paused. Old locks and
checkpoints are never rebound.

A first-time `masy.runtime.v1` runtime file needs `[sources.maple]`
`distribution_metadata`: use the four genuine metadata files of the companion
fixture `examples/native9f_metadata/distribution/` (documented in its README and
the practical guide, Section 14.10), not the output of a new `pip install -e`.
The fixture is metadata input to the source-bundle export; it installs nothing and
all source guards still run. Each device (CPU, CUDA) has its own `runtime.python`
and `runtime.store`.

The production tools (`tools/masy_runtime.py prepare|verify`,
`tools/masy_leaf_capsule.py seal|verify`, `tools/masy_production.py
draft-recipe|draft-receipt|render-config`) must be started with `-I -B`. Their
flags, the per-machine nature of the stores, locks and recipes, and the review
requirement for the receipt are in the practical guide (Section 14.10). Recorded
release-run hashes: CPU recipe
`3c81ac3282c2daacb63c0d5ecd214b836a2e7a4cc470d53aa4c8303f99f118fe` and component
review receipt `7f884b59c1b123c183093475ab158f0e00644b598206dd4f2db686ae1aec4252`;
CUDA recipe `7a9eecfad45c7b0ba920746a627f5e495289f76527a40ea0d00bcd213694ab67`
and receipt `f9345fe8b3a52f7a513492eda2b6021df5c92a4bab28b04d02912e0f406e3537`.
They document the evidence and cannot be reused on another machine.

## Water parameters (`layered_water.yaml`)

The file schema is `maple-syrup-layered-case/1`; it is read by the CPU loader
(`numpy`, `numba`) or the CUDA loader (`cuda`) of `maple_syrup.layered_case`
according to the profile. The schema requires explicit sections and rejects unknown
keys; example values are numerical test settings, not field calibration.

| Section/key | Required choice or quantity |
| --- | --- |
| `compute.backend` | `numpy`, `numba` or `cuda`: the only place the compute backend is chosen. The CPU loader refuses `cuda` and the CUDA loader refuses `numpy`/`numba`; the scheduled command selects the loader from the source profile and refuses a mismatch. Neither `run.yaml` nor the command line overrides it and there is no fallback. The CPU and CUDA files of the release run differ in this value only. |
| `soil.layers.thickness_m` **or** `soil.layers.edges_m` | Layer geometry in m, from the surface downward. Edges begin at zero. These determine the number of layers and must fit inside the actual MAPLE material. |
| `soil.theta_residual`, `soil.theta_field_capacity`, `soil.theta_saturation` | Volumetric water fractions, m³/m³; scalar or one value per layer. Require residual ≤ field capacity ≤ saturation, and residual < saturation. |
| `soil.theta_evaporation_stress` | Top-layer moisture threshold in m³/m³, above its residual threshold. |
| `soil.gravity_drainage` | Boolean enabling the gravity cascade. |
| `soil.bottom` | `open` permits bottom drainage; `closed` retains water at the bottom. This is not an upward groundwater/wicking boundary. |
| `soil.diffusion.mode` | `tietjen_donor_depth` or `constant`. |
| Tietjen-mode diffusion inputs | `soil.diffusion.interface_conductivity_m_s` (m/s, scalar or one value per interface) and `soil.diffusion.coefficient` (dimensionless). |
| Constant-mode diffusion input | `soil.diffusion.diffusivity_m2_s` (m²/s, scalar or one value per interface). |
| `matrix.conductivity_m_s`, `matrix.suction_m`, `matrix.vegetation_factor` | Matrix infiltration conductivity (m/s), suction (m), vegetation factor (dimensionless, 0–1). |
| `preferential.a3` | Reference background coefficient. A scalar must be exactly 0.1, 0.5 or 0.8; a `{file: ...}` map is accepted by the parser when every value lies in 0–1 (the example comment asks for the reference values only). |
| `preferential.max_fraction_per_hour`, `preferential.max_intake_rate_m_s` | Dimensionless reference hourly fraction of the standing water, and an additional intake-rate cap in m/s (positive number or map) or `unlimited`. The parser accepts 0 to 1.0e6 for the fraction, but the setup is refused unless the resulting per-cell hourly fraction (this value times the summed receiver map) lies in 0–1. |
| `preferential.background_receiver_weights` | Nonnegative receiving weights, one per layer. |
| `preferential.species` | List of prescribed pathways; each supplies `name`, `receiver_weights` and `contribution`. These are not plant initialization or root fractions. Positive weights are normalized; all-zero species weights disable that species' pathway. |
| `evaporation.canopy_intercept`, `canopy_slope`, `transpiration_slope`, `transpiration_offset`, `soil_factor` | Dimensionless evaporation-partition parameters. `soil_factor` is the site factor multiplying the combined soil potential (not the evaporation share; the example file's comment calling it the "post-moisture-cap soil E share" is misleading). The share is 1 − clip(`transpiration_slope`·cover − `transpiration_offset`, 0, 1), applied after the moisture cap. SYRUP debits only the resulting evaporation; it does not debit plant transpiration through this calculation. LEAF requests withdrawals separately. |
| `cover` | Prescribed canopy-cover fraction, 0–1. |
| `initial.soil_moisture.units`, `by_layer` | `theta` (m³/m³) or `inventory_depth_m` (m of water per layer), with one initial value per layer. Zero initial water is allowed; the resulting layer water must not exceed that layer's saturation depth. |
| `initial.wetting_history_m`, `initial.cumulative_intake_m` | Persistent Green–Ampt wetting history and cumulative actual matrix intake, both expressed in m. These record history; they do not add water. |
| `domain.active_mask` | `all` or a Boolean map reference. |
| `domain.friction_factor` | Positive Darcy–Weisbach friction factor. |
| `domain.closed_ring_rise_m` | Positive elevation offset used to close non-outlet boundaries. |
| `domain.outlets` | Edge faces with `row`, `col`, `d_row`, `d_col`, `drop_m`; `(d_row, d_col)` is one of (1,0), (−1,0), (0,1), (0,−1); the drop is positive and in m; duplicate faces are refused. An empty list is a closed basin. MAPLE's periodic wind boundaries do not implicitly define water outlets. |

There are L−1 interfaces for L layers. Layer hydraulic properties are explicit
inputs in Phase 1. The actual MAPLE active-layer/voxel texture is calculated, but
it does not yet automatically determine conductivity, suction or retention
thresholds; that texture-to-property method belongs to Phase 2.

Several spatial inputs accept a scalar or `{file: name.npy}` (with an optional hexadecimal `sha256`). A map is one plain file name
(no directory part, no symbolic link) in the water file's directory: an `.npy` file, format 1.0 or 2.0, C-ordered two-dimensional `(ny, nx)`
equal to the native grid, little-endian float64 (`<f8`) and finite for numeric maps, Boolean for `domain.active_mask`; no resampling is applied.
Retention thresholds vary by layer rather than by cell. The parser's default limits (`LayeredCaseLimits`) are 65 536 bytes of YAML, 5000 nodes,
depth 8, 64 layers, 16 species, 4096 outlets, and 32 map files with 256 MiB per file and in total; the Python interface lets a caller pass other positive
integers, which are honoured exactly. See [implementation.md](implementation.md) (layered case loader and file-initialized lifecycle) and `src/maple_syrup/layered_case.py` for further detail. For the daily Phase 1.5 inputs (P, I, T and the run identity, which are API arguments, not file keys) see [phase15_qualification.md](phase15_qualification.md).
YAML scientific-notation numbers need a decimal point: write `1.0e-5`.

## Weather and numerical controls

`scenario/weather.json` has closed declarations (units, `temperature.extrema` and `temperature.tmean`, `humidity.supplied`, `radiation.supplied`,
`pet.method`, provenance) and one row per consecutive calendar day. Every row has `date`; `tmean_c` when `tmean` is `supplied`; `tmin_c` and `tmax_c`
when `extrema` is true; `relative_humidity_fraction` or `vpd_kpa` as `humidity.supplied` declares; `radiation_mj_m2_day` when radiation is supplied; and
`pet_mm_day` for `pet.method: explicit`. A mean temperature is always available (supplied, or derived as the average of the extrema under a declared
convention). Extrema are required for `ecohyd_hargreaves` and for any humidity field, not for explicit PET without humidity. Latitude is not a weather field:
the integrated case always reads `location.latitude_deg` from MAPLE's `case.yaml`, with no default, even when PET is explicit. Penman–Monteith is a future hook,
not an implemented fallback.

Rain-event intervals use `duration_s` and `intensity_mm_h`. Interevent files use
`duration_s` and `processes.evaporation`, `processes.transpiration`,
`processes.growth`. In v1 only evaporation may be enabled; in v2 transpiration and
growth may be enabled, with LEAF requesting withdrawals that SYRUP checks, debits
once and reports back. In a v2 run dated instants must be exactly representable in
the microsecond LEAF clock (a 0.05 s interval is refused; 0.0625 s is accepted).

Time-step, retry, step and recession limits, memory reserves and output caps are
numerical and resource controls, not soil properties. Soil moisture and the wetting history persist
between accepted events; no automatic storm-end dry reset or general wetting-
front recovery law is implied. The approved wetting-history reset is a separate
explicit operation restricted to controlled, closed, quiescent, exactly dry
surface-water boundaries; actual soil water and physical ledgers are preserved. Here "closed" means the lifecycle is at a closed boundary between episodes (no storm, recession or interevent state open); it is unrelated to `soil.bottom: closed` and to a closed basin, and a basin with outlets is not excluded if its surface is exactly dry and its outlet discharge is within the quiet policy. No key in these parameter files requests the reset.

An incomplete native call (per-call step budget exhausted) is logged and the same occurrence continues on the next call; the run stops only at a final limit, and a run that ends with an occurrence incomplete is reported incomplete (exit 3) with no checkpoint.

## Historical examples (task snapshots; not release recipes)

Earlier development examples are kept only for traceability; they use private
absolute paths and older identities. Do not copy their source paths, hashes or
profiles into a new run and assume it is qualified:

- `agent_handoffs/tasks/phase1_native_calendar_run_20261111/claude_correction02_snapshot/examples/native_layered_calendar/` (v1, `95ef` profile, CPU only);
- `agent_handoffs/tasks/phase1_gpu_calendar_impl_20261112/claude_correction05_snapshot/examples/native_layered_calendar_cuda/` (v1 CPU109 and CUDA109 water-only development examples);
- the commented layered-case examples under `agent_handoffs/tasks/phase1_layered_case_setup_20261111/` and `phase1_layered_cuda_case_setup_20261111/` (`cases/subsurface/layered_example/layered_water.yaml` and `layered_cuda_example/layered_water_cuda.yaml`). Their comment calling `soil_factor` the "post-moisture-cap soil E share" is wrong: `soil_factor` is the site factor multiplying the combined soil potential, and the share comes from `transpiration_slope`, `transpiration_offset` and `cover` (Water parameters above).
