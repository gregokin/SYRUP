# MAPLE-SYRUP Phase 1 interface contract

Status: design contract written 2026-09-29 for task `phase1_dependency_contract`. It states how MAPLE-SYRUP depends on and calls MAPLE, which state each side owns, the actual gaps in MAPLE's current water path, and the initial choices for lateral exchange, depth ownership and restart persistence. **No SYRUP solver exists.** Only the dependency check, provenance capture and the probe in `src/maple_syrup/` are implemented; everything under sections 4–6 is a specification for later phases. Not accepted by Codex review at the time of writing.

Source references are to `/home/okin/MAPLE/src/maple/...` as read on 2026-09-29 (working tree, not a clean revision; see section 1.3). Line numbers can drift with upstream edits.

## 1. Dependency arrangement

### 1.1 What is installed

- Distribution `maple` 0.0.1, editable install in `/home/okin/MAPLE/.venv` (Python 3.12.3). `site-packages/maple-0.0.1.dist-info/direct_url.json` is `{"dir_info": {"editable": true}, "url": "file:///home/okin/MAPLE"}`; `__editable__.maple-0.0.1.pth` adds `/home/okin/MAPLE/src`.
- `/home/okin/MAPLE/pyproject.toml`: `name = "maple"`, `description = "MAPLE — Aeolian Transport Model"`, base dependencies jax, jaxlib, numpy, pyyaml, xarray, zarr, dask, tifffile. Optional `gpu` extra `cupy-cuda12x[ctk]>=14.0,<15`.
- The venv contains numpy 2.5.2, jax/jaxlib 0.11.1, pytest 9.1.1, and **no CuPy**.
- `import maple` imports jax and enables x64 (`maple/__init__.py`).

### 1.2 How MAPLE-SYRUP depends on it

`pyproject.toml` deliberately does **not** list `maple` as a requirement: the PyPI name `maple` belongs to an unrelated project and MAPLE is unpublished, so a bare requirement could install the wrong package, and a `file://` reference would pin a mutable live checkout. Instead `maple_syrup.dependency.resolve_maple_dependency()`:

1. imports `maple` from the running environment;
2. checks that distribution metadata and the imported files agree (editable root contains the imported package, or the installed `maple/__init__.py` is the imported one), so a shadowing copy is refused;
3. requires the source root's `pyproject.toml` to name project `maple`; a path import with neither metadata nor such a file is refused;
4. optionally requires an exact source root (`expected_source_root` or `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT`);
5. checks `REQUIRED_MAPLE_API`: every symbol SYRUP calls or this contract relies on, with named parameters, plus required registry values (for example `"constant_depth"` in `DEPTH_UPDATE_RULES`).

Passing step 5 means only that the specific surface exists with those parameter names. It is not a claim of general API or behavioural compatibility with any MAPLE revision; behaviour is covered by the tests in `tests/phase1/` and by later phase tests.

Development invocation (no install):

```
cd /home/okin/SYRUP
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m maple_syrup.probe
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m pytest -p no:cacheprovider tests/phase1 -q
```

`PYTHONDONTWRITEBYTECODE=1` prevents importing MAPLE from writing `__pycache__` files into the MAPLE tree.

### 1.3 Provenance and reproducibility

`maple_syrup.provenance.capture_maple_provenance` records:

- identity from section 1.2 and the source kind label; for the editable install this is `editable/development working source`;
- `package_source_digest`: SHA-256 over every file in the imported package directory (`src/maple`, 324 Python files at the time of writing), sorted relative path plus per-file SHA-256, excluding `__pycache__` and `*.pyc/*.pyo`. Only regular files and real directories are accepted: a symlink (file or directory) or special file in the scope raises rather than being followed or skipped. Limits (default 5,000 files, 64 MiB) are validated as positive integers and checked before each file is opened; each read is capped at the remaining byte budget, and growth or a size change during the read raises. This covers tracked edits and untracked new modules alike, and works without git;
- SHA-256 of `pyproject.toml`;
- git HEAD read from `.git` files without running git, plus one read-only `git --no-optional-locks status --porcelain=v1 -z --untracked-files=all -- src/maple pyproject.toml` (`GIT_OPTIONAL_LOCKS=0`, fsmonitor disabled). Result `clean`, `dirty` with entries, or `unavailable` with a reason. `head_describes_source` is true only if git positively reported the scope clean.

The probe takes a second digest after it finishes and reports `maple_source_changed_during_probe` if the two differ.

MAPLE's own `case_tools/provenance.py:capture_code_state` (lines 196–364) is not reused here: it hashes every untracked file in the whole repository (unbounded for this checkout), runs `git status` without `--no-optional-locks`, and `git_dirty_content_sha256` returns `None` both for a clean tree and for a failed query. MAPLE's orchestrator keeps using it for MAPLE runs.

Limitations: this detects source change; it does not archive the source. `agent_handoffs/upstream_maple.md` lists uncommitted edits to `surface/voxels/transfer.py` and `experiment/orchestrator.py`, both on paths SYRUP uses. A reproducible production baseline still needs an immutable snapshot or clean revision plus recorded patch; that mechanism remains a separate bounded design task. The live editable checkout is acceptable for development only.

## 2. Shared state and APIs SYRUP reuses

All class-indexed arrays use MAPLE's canonical `grain_classes.classes` order, FP64, `(ny, nx, ...)` row-major with row axis first. Masses are kg, lengths m, time s.

| Concern | MAPLE owner (module: symbol) | SYRUP use |
|---|---|---|
| Case config and compile | `case_tools/compilers/case_compiler.py: compile_case(case_path, output_path=None, *, overwrite, import_base_dir)` (337–343) | Phase 2 case import goes through this; SYRUP-only config is kept separately (§3.3). |
| Geometry, boundaries | `core/parameters/geometry.py: GeometrySpec, validate_geometry`; `core/boundaries: AxisBoundary, BoundaryKind` (periodic, prescribed only, 16–25) | Used unchanged. Boundary kinds are sediment inflow metadata, not hydraulic walls or outlets. |
| Grain classes | `core/parameters/grain_classes.py: GrainClass, GrainClassSet, validate_grain_classes`; `core/parameters/water_coupling.py: MAHLERAN_1_2_1_CLASS_DIAMETERS_M` (192–199) | Class axis and metadata; probe uses MAPLE's MAHLERAN diameter table. |
| Bed construction | `core/types/voxel.py: zeros_voxel_column_state`; `surface/voxels: deposit_surface_mixture_batch`; `surface/active_layer: initialize_active_layer_from_voxels` | The compiler's sequence (case_compiler.py 646–709). Initialization is host-only (`np.full`, initialization.py 130); device placement follows via `to_device_tree`. |
| Bed exchange | `surface/active_layer/exchange.py: update_active_layer_after_erosion/_deposition` | Only through MAPLE's water step, never called directly by SYRUP. |
| Availability | `core/types/sediment_availability.py`, `surface/availability/exchange.py: cap_demand_by_availability` | Passed through the water step (`sediment_availability`, `initial_available_fraction`). |
| Ledger | `core/types/sediment_ledger.py: zeros_sediment_ledger_state`, channel `water_erosion_deposition` (337–344) | Water step records both gross directions; SYRUP never writes the ledger directly. |
| Water state | `core/types/water.py: WaterState(depth_m (ny,nx), mobile_mass_by_cell_class_kg (ny,nx,nc))`, `water_state_from_depth`, `zeros_water_state`, `water_state_content_sha256` | Canonical depth and water-borne mobile mass. Single time level by design (10–22). |
| Water step | `water/step.py: apply_water_process_demand(voxel_column, active_layer, water, ledger, demand, geometry, grain_classes, mass_resolution_kg, *, sediment_availability, initial_available_fraction, adapter_name, detachment_integration)` (155–377) | The one conservative bed/mobile exchange SYRUP calls per accounting step. Transactional: inputs unmodified on failure. |
| Demand/result types | `water/interfaces.py: WaterProcessDemand, WaterProcessResult, HydraulicDiagnostics, zero_water_demand` | SYRUP builds demands; results carry ground-truth accounting. |
| Validation | `water/validation.py: validate_water_state, validate_water_process_result` | Used as-is; SYRUP adds its own checks for what it owns (§4). |
| Depth after commit | `water/depth.py: apply_depth_after_bed_change` (`constant_free_surface`, `constant_depth`); `water/commit_callback.py: make_water_depth_commit_callback` | See §4.2. |
| Topographic commit | `aeolian/scheduler/commit.py: maybe_commit_topography(physical, config)` (308–437); lower level `surface/topographic_commit/commit.py: commit_topography(..., water_callback, routing_callback, ...)` (1101–1117) | SYRUP commits through `maybe_commit_topography`; it does not reimplement triggers, avalanche or reconciliation. |
| Fluvial step | `coupling/water_event.py: advance_fluvial_step(..., demand=None)` (82–129) | Accepts an injected demand, but labels the result with the configured adapter name (§5, G2/G8). |
| Block accounting | `aeolian/scheduler/accounting.py: accumulate_water_step_result` (1265–1317); `block.py: water_reservoir_delta_by_class_kg` (214–249) | Reused for event accounting; the block identity uses the first step's mobile-before and last step's mobile-after domain totals per class. |
| Backend | `core/backend: resolve_backend(backend, *, device_id, deterministic_scatter)` (selection.py 291–390), `to_device_tree`, `to_host_tree`, `read_transfer_counters`, `array_namespace` | One namespace per run; explicit `cupy` never falls back silently. |
| Snapshots | `io/outputs/snapshot.py: save_state_snapshot(..., water=...)`, `load_state_snapshot`, `SNAPSHOT_SCHEMA_VERSION = 2` | Persists only water depth and mobile mass (567–578, 1136–1152). |
| Case identity | `case_tools/provenance.py: compute_case_identity_sha256` (487); water settings enter only when water is enabled | Part of the checkpoint binding (§4.3). |
| Wind | `aeolian/*`, scheduler and `experiment/orchestrator.py` | Future phases invoke these through MAPLE only. SYRUP contains and will contain no wind equations. |

## 3. Water-owned additions (future, not implemented)

### 3.1 SYRUP solver state

Held in a SYRUP-owned frozen container, same backend namespace as the MAPLE state, never pushed into `WaterState` (MAPLE forbids solver time levels there, water.py 10–22):

| Field | Shape, unit | Reconstructible? |
|---|---|---|
| Rainfall forcing cursor and integrated-forcing clock | scalars, s and mm | No; persist. |
| Cumulative infiltration, soil moisture/storage, ponding time (only what the selected infiltration model needs) | `(ny, nx)`, m or m³/m³, s | No; persist. |
| Hydraulic state required by the selected routing method (discharge, velocity, previous-level depth if the method needs it) | `(ny, nx)`, m³/s, m/s, m | No; persist. |
| Transport memory (for example legacy sediment-velocity decay) | `(ny, nx, nc)` | No; persist. |
| Routing graph (receivers, order, dependency levels) | `(ny, nx)` int / level arrays | Yes, deterministically from committed topography and outlet mask; persist only an input hash. |
| Event status (rain ended, recession hold counters, lateral-transfer residual accumulators) | scalars, `(nc,)` | No; persist. |

### 3.2 SYRUP-owned hydraulic depth

During a SYRUP event SYRUP's solver computes depth; it publishes that depth into `WaterState.depth_m` before every MAPLE water step and commit, so MAPLE snapshots see the current depth (§4.2).

### 3.3 SYRUP configuration

Rainfall, infiltration, roughness, routing method, outlet mask and stopping criteria are SYRUP config, not MAPLE `water_coupling` fields (which by design configure no hydrology, water_coupling.py 1–14). The resolved SYRUP config is hashed and recorded beside MAPLE's case identity.

## 4. Selected initial approach

### 4.1 Lateral exchange driver

Selected: a **SYRUP event driver composed from MAPLE public APIs**, without changing MAPLE initially. MAPLE's `advance_fluvial_block` is not the driver because its outer step resolves demand only from configured prescribed sources and passes no injected demand (fluvial.py 125–156).

Per accounting step of duration `dt` (SYRUP may take hydraulic substeps inside it):

1. SYRUP advances rainfall/infiltration/hydraulics and computes local detachment demand.
2. SYRUP applies its **lateral mobile-transfer operator** `T` to `WaterState.mobile_mass_by_cell_class_kg`: internal cell-to-cell transfer only, non-negative, per-class domain sum preserved. Mass that leaves across an outlet face is moved into that outlet cell's mobile pool and requested as `boundary_export_by_cell_class_kg` in step 4, so MAPLE's result records export.
3. SYRUP publishes `WaterState(depth_m=solver depth, mobile=T(mobile))`.
4. SYRUP calls `apply_water_process_demand` with removal, deposition (a request against post-transfer local mobile), boundary export, `face_flux` from `T`, and `adapter_name="maple_syrup/<algorithm version>"`.
5. SYRUP folds the result with `accumulate_water_step_result` and calls `maybe_commit_topography`.

Why this satisfies the conservation contract: MAPLE's step keeps its per-step mobile identity (`mobile_after = mobile_before + removed − deposited + input − export`, validation.py 417–440) with `mobile_before = T(mobile)`. The block identity uses first/last domain totals (block.py 238–249), which `T` leaves unchanged per class up to FP64 roundoff, so internal transfer is never disguised as boundary inflow or export.

SYRUP-side checks required before step 3 publishes (implemented with the operator, Phase 5):

- `T(mobile) >= 0` everywhere, finite;
- per-class `|Σ T(mobile) − Σ mobile| <=` a summation bound scaled from the largest operand; the residual is accumulated and reported, never absorbed;
- net face flux divergence reproduces `T(mobile) − mobile` per cell and class;
- outlet export requested only at outlet-mask cells; MAPLE's cap of export and deposition by available mobile mass is compared with the request and any refused amount reported.

A failed check raises before any new state is published. The executable demonstration that MAPLE does not move mobile mass itself is `tests/phase1/test_probe.py::test_local_pickup_conserves_mass_but_water_step_has_no_lateral_transfer`.

This is a design. `T`, detachment and deposition laws do not exist yet.

### 4.2 Depth ownership

Selected: **SYRUP owns hydraulic depth; SYRUP events run with `water_coupling.depth_update_rule: constant_depth`.**

- MAPLE builds the depth commit callback whenever the state carries water (scheduler commit.py 676–701; forced commits in orchestrator.py 424–455), and the default `constant_free_surface` rewrites depth as `max(depth − Δz_bed, 0)` (depth.py 151–169). With `constant_depth` the callback returns an unchanged copy (depth.py 139–149), so there is no double update.
- Under `constant_depth` a committed bed change of `Δz` raises the free surface by `Δz` at unchanged water volume. SYRUP's next hydraulic step works from the new committed bed plus its depth.
- Routing refresh: MAPLE's `routing_callback` is an identity stage returning only bed state, so SYRUP recomputes routing after `maybe_commit_topography` returns `did_commit=True`, from committed topography, and at event start. Hydraulics between commits use the last committed surface.

Checks: the driver refuses to start unless the resolved rule is `constant_depth`; after each commit it requires `water_commit.depth_update.rule == "constant_depth"` and the returned depth array-equal to the published depth. Any mismatch raises before the post-commit state is accepted. `tests/phase1/test_probe.py::test_selected_constant_depth_rule_leaves_solver_depth_alone` checks MAPLE's rule behaviour.

### 4.3 Checkpoint persistence

Selected: a **versioned SYRUP sidecar file** written next to each MAPLE snapshot taken during a SYRUP event. A MAPLE snapshot extension is proposed later (P3), not required now, because changing MAPLE's snapshot schema affects every MAPLE user.

Sidecar contents:

- `schema` and SYRUP solver algorithm versions;
- every non-reconstructible field in §3.1, host NumPy, with declared shape, dtype and unit;
- binding block:
  - SHA-256 of the MAPLE snapshot file bytes, plus its `time_s` and `step`;
  - `water_state_content_sha256` of the snapshot's water state, as a secondary check only;
  - MAPLE case identity (and the MAPLE run scientific identity when run through the orchestrator);
  - SHA-256 of the resolved SYRUP config;
  - MAPLE `package_source_digest` and SYRUP source digest;
  - grid, class ids and class count, and routing-input hash.

The water hash alone is never sufficient, because it does not identify bed, availability, ledger, configuration or code.

Write order: MAPLE writes its snapshot (`np.savez`, snapshot.py 725); SYRUP hashes the written file, writes the sidecar to a temporary name and renames it. A snapshot without a valid sidecar cannot be used to resume a SYRUP event.

Resume refuses (no reconstruction, no partial state) when: the sidecar is missing for a mid-event snapshot; the snapshot hash, time or step differ; the schema is unknown; shapes, class ids, grid or routing-input hash differ; any field is non-finite or out of range; or the case/config identity differs. A differing code digest also refuses unless the caller explicitly allows it with a recorded reason. A snapshot taken after the dry-again reset needs no sidecar: the reset state is MAPLE state plus the declared antecedent condition in SYRUP config.

Restart equivalence (uninterrupted versus resumed) is tested in Phase 6; Phase 1 only fixes the design.

## 5. Actual gaps in current MAPLE

| ID | Gap | Evidence | Consequence / handling |
|---|---|---|---|
| G1 | Water step is local: removal enters the same cell's mobile pool; deposition capped by that cell's mobile mass; face flux passed through unapplied. | step.py 266–300, 362–366 | SYRUP supplies `T` (§4.1). Demonstrated by test. |
| G2 | Fluvial block loop cannot inject a computed demand. | fluvial.py 136–146 | SYRUP driver (§4.1); P2 proposed. |
| G3 | Default depth rule rewrites depth at every commit, including forced commits. | commit_callback.py 164–176; orchestrator.py 424–455 | `constant_depth` (§4.2). |
| G4 | Commit lifecycle stages return only bed state; no hydraulic return channel. | commit.py 1101–1117; commit_callback.py 1–20 | Routing refresh after commit returns (§4.2). |
| G5 | Snapshots persist only depth and mobile mass. | snapshot.py 567–578, 1136–1152 | Sidecar (§4.3); P3 proposed. |
| G6 | `WaterState` forbids solver time levels. | water.py 10–22 | SYRUP container (§3.1). |
| G7 | `HydraulicDiagnostics` has five optional non-canonical fields, never persisted. | interfaces.py 103–129 | SYRUP diagnostics are separate outputs. |
| G8 | Adapter registry has only `prescribed` and `mahleran_1_2_1`. A fluvial `prescribed` event with no source is rejected; `mahleran_1_2_1` is a conversion seam for legacy fields; `advance_fluvial_step` labels results with the configured adapter name. | water_coupling.py 162, 663–683; water_event.py 122–129 | Enabling MAPLE water config for a SYRUP event currently requires either a mislabelled adapter or `water_coupling.enabled: false`, in which case MAPLE case identity omits `depth_update_rule`. **Open decision before the Phase 4/5 driver**; P5 proposed. The SYRUP driver calls `apply_water_process_demand` directly with its own `adapter_name`. |
| G9 | Boundary vocabulary is periodic/prescribed only. | core/boundaries 16–25 | Outlets and hydraulic walls are SYRUP config. |
| G10 | Active-layer initialization is host-only. | initialization.py 129–131 | Build on host, place with `to_device_tree`. |
| G11 | The water step and its validator make many counted host scalar reads per call. | step.py, validation.py (`all_finite`, `any_negative`, `max_abs`, `to_float`) | Accounting cadence and substepping must be profiled; a coarser cadence needs its own physical validation. |

## 6. Proposed narrow MAPLE extension points

Proposals only; adoption in MAPLE is a separate scoped action with its own review.

- **P2 — fluvial demand provider.** Optional `demand_provider(physical, time_s, dt_s) -> WaterProcessDemand` and adapter label for `advance_fluvial_block`/`advance_fluvial_outer_step`, so SYRUP can reuse MAPLE's block loop, accounting and commit handling instead of composing them.
- **P5 — external adapter registration.** An `external` adapter name with a caller-declared algorithm version in `WATER_ADAPTER_NAMES`/`WATER_ADAPTER_ALGORITHM_VERSIONS`, allowed for fluvial events without prescribed sources, so case identity and provenance name SYRUP honestly and include `depth_update_rule`.
- **P3 — snapshot extension payload.** A namespaced, versioned extension block written and validated inside MAPLE's snapshot transaction, replacing the sidecar once available.
- **P1 — lateral transfer in the water step (only if needed).** An optional signed, domain-sum-zero `lateral_transfer_by_cell_class_kg` applied before deposition and included in MAPLE's mobile identity. Only if SYRUP's pre-step operator proves insufficient (for example if MAPLE's validator should own the lateral identity).

## 7. Backend and performance commitments

- One array namespace per run, resolved by `resolve_backend`; state built on host at run boundaries, placed once with `to_device_tree`, returned with `to_host_tree` for snapshots and outputs. An explicit `cupy` request that cannot be honoured raises `BackendUnavailableError`.
- SYRUP arrays follow MAPLE layout (FP64, C-contiguous, `(ny, nx[, nc])`). Routing has ordered upstream dependencies; later phases must use a deliberate dependency-level, compiled or GPU algorithm and keep a CPU reference.
- Measure transfer counters, scalar reads, wall time and peak memory, separating startup from steady state. The Phase 1 probe's CuPy path only round-trips a 3×4 state and is not GPU readiness evidence; CuPy is not installed in the MAPLE venv.

## 8. What Phase 1 implements and checks

Implemented: `maple_syrup.dependency`, `maple_syrup.provenance`, `maple_syrup.probe` (CLI `python -m maple_syrup.probe`), tests in `tests/phase1/`. The probe builds a 3×4, three-voxel, six-class bed through MAPLE's public constructors, checks inventories against values computed from authored inputs, and runs one zero-demand water step. Not implemented: any water physics, SYRUP driver, `T`, sidecar I/O, MAHLERAN case import, wind invocation.
