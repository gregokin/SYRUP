# Phase 2.5 — LEAF vegetation in the water laws and closed-parent restart (native Python API)

**Status: accepted at a bounded scope** by Root's separate receipt `phase25_qualification.json` (SHA-256 `8fe92ed3f18acb5adb824bfea8bf191a8ef57035e9a5948fb262242d2eb6525b`; human-readable `phase25_qualification.md`). Root verified 714 CPU tests and 37 actual-GPU tests on the exact source, with the imported source paths, complete package hashes and the external helper checked before and after each suite (a 69-test source-guard preflight overlaps those suites and is excluded from the totals).
The source is published (SYRUP commit `bb28af6c58c2892cb3c065d73d33b88285c498f0` on `feature/subsurface-vegetation`; LEAF commit `ff127d29bd1d40f17a1800ffeef87bf080343f67` on `feature/syrup-hydrology-traits`; `phase25_publication.json`). The identity-only source lock used during verification states `whole_phase_accepted: false`; that flag means the lock alone claims no acceptance, and the acceptance is the separate receipt above.
Nothing in this page is a command, a YAML key, a CSV format or a scheduler (MASY) schema: Phase 2.5 is a set of native Python APIs used explicitly by a caller. Historical C1, N15 and Phase 2 identities, receipts and qualification statements are unchanged and are not rebound by this page.

## 1. What Phase 2.5 adds, in one paragraph

LEAF (individual plants) supplies a per-cell *projection* of canopy cover and of root-weighted indices; SYRUP uses those cell maps, together with explicit site controls and species receiving weights, to fill the
vegetation terms of its existing layered water laws — the matrix vegetation factor `g` and the preferential-infiltration contributions — exactly once, and the splash cover of the rain-impact physics. LEAF still owns plants,
SYRUP still owns all physical water, MAPLE still owns the bed. A vegetation *generation* is applied to a completed lifecycle boundary in zero time; its small compact recipe can be cached and restored; every-N-day
diagnostic frames can name the generation they were written under.

## 2. Accepted source and dependencies

| Item | Identity | Status |
|---|---|---|
| SYRUP source | commit `bb28af6c58c2892cb3c065d73d33b88285c498f0`, branch `feature/subsurface-vegetation` (https://github.com/gregokin/SYRUP.git); package SHA-256 `37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0`, 180 files; manifest `phase25_source_manifest.json` SHA-256 `a63946ddd29a5a1bd109a62de392fdb39ae2e6cad78a7085952838073ab35c63`; exact code `phase25_source_capsule.zip` | accepted at a bounded scope; published |
| LEAF source | commit `ff127d29bd1d40f17a1800ffeef87bf080343f67`, branch `feature/syrup-hydrology-traits` (https://github.com/gregokin/LEAF.git); package SHA-256 `38009872f8969f86b56c720adc1b5d9ffefc554fb9cacf90855d999db604530f`, 28 files; external helper `benchmarks/spatial_mapping.py` SHA-256 `333921bb645d984fa8e7449619d03114262ee333c7fea61a22493a878fd9ccc5` | published; unchanged by the acceptance |
| Native MAPLE (historical, unchanged) | revision `9f466d2cfcd52991b24496cf67f4562d6268afa9`; package `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`, 412 files | unchanged |
| Base | published main `953e6fdf85b73e3fdf5a1ac7b0d97dea4df7dab1` (adds the non-vegetated wind handoff WH1); Phase 2 package `3917dba4…`, 169 files, is unchanged | unchanged |
| Whole-milestone acceptance | `phase25_qualification.json`, SHA-256 `8fe92ed3f18acb5adb824bfea8bf191a8ef57035e9a5948fb262242d2eb6525b` | accepted bounded |

**Historical component compositions.** During development four component compositions were verified one by one: cache `692482499cb8d405bb1508fecea5d6570c6295a9dc842366566ef5a03001df53` (171 files), lifecycle `809cf3e98c553b0f7835ace1130dbed9e489e7df6c0debafda0832ab8c04151b` (173 files), held context `a38d4d929fb9b5b237dcdce010c72f2c2700573fa2c497473add034b12f59c53` (175 files) and frames `d5f285c8018a1c2a1fb1c945cd7a9bd525af79909deb796b628e366e9f1e9f29` (175 files). They are historical intermediate sources, kept in `release_metadata.json` for the record; none of them is the current model.

**How to reproduce.** Clone `https://github.com/gregokin/SYRUP.git` and check out commit `bb28af6c58c2892cb3c065d73d33b88285c498f0` (branch `feature/subsurface-vegetation`); clone `https://github.com/gregokin/LEAF.git` and check out commit `ff127d29bd1d40f17a1800ffeef87bf080343f67` (branch `feature/syrup-hydrology-traits`); use the pinned native MAPLE `9f466d2c…` materialization. Verify before use that SYRUP's `src/maple_syrup` digests to `37d89f23…` (180 files), LEAF's `leaf` package to `38009872…` (28 files) and `benchmarks/spatial_mapping.py` to `333921bb…`; both commits exist on the public remotes, and when this edition was prepared a `git archive` of each was extracted and digested: `src/maple_syrup` of the SYRUP commit gave `37d89f23…` with 180 files, the `leaf` package of the LEAF commit gave `38009872…` with 28 files, and `benchmarks/spatial_mapping.py` hashed to `333921bb…`. `phase25_source_capsule.zip` carries the exact accepted code as an independent copy.

## 3. Species traits: exact key, version and example

* Catalog document: `{"schema": "leaf.species_catalog", "schema_version": 2, …}`. Version 2 is an explicit opt-in (`SCHEMA_VERSION_HYDROLOGY = 2`, exact integer); version 1 is the unchanged legacy catalog and refuses the new block.
* Per species, optional: `"hydrology": {"preferential_infiltration_receiver_weights": [w_1 … w_L], "notes": "free text"}` (`leaf.traits.RECEIVER_WEIGHTS_KEY`, `HYDROLOGY_SPECIES_KEY`). `L` is the catalog's top-level `root_layers`.
* Receiving weights: exactly `L` finite numbers `>= 0` (no bool / string / null); they need not sum to 1; `[0, 0, 0]` is an explicit *disabled* declaration; `[0, 1, 1]` of three layers means equal weight on layers 2 and 3. LEAF returns the raw vector; SYRUP normalises a positive vector
  for allocation. They are **not** root fractions and are never derived from them.
* Fixed root fractions: `roots.layer_fractions`, `L` entries summing to 1 (tolerance `1e-9`), top layer first; fixed per species, no plasticity and no availability renormalisation.
* Species ids match `[A-Za-z0-9_.\-]{1,64}`. Accessors: `catalog.preferential_receiver_weights(sid)`, `catalog.has_preferential_receiver_weights(sid)`, `catalog.preferential_receiver_weight_table()`.
* Example (declared, **uncalibrated fixture** values): `docs/manuals/examples/native_phase25_example/species_v2.json` — three layers; `grass` roots `[0.8, 0.1, 0.1]` with receiving weights `[0, 0, 0]` (pathway disabled); `shrub` roots `[0.25, 0.35, 0.4]` with `[0, 1, 1]`.
  `python -m leaf validate-species --species species_v2.json` was run on it (CPU): it prints `schema_version 2`, the catalog hash and the species list; it does not print the weights.

**The one switch: `require_receiver_traits`.** `leaf.water_vegetation.project_from_initial_state(...)` and `project_from_plant_session(...)` take a keyword `require_receiver_traits`, default **`True`**. With the default, every participating *live* species must have an explicit receiving-weight vector in a `schema_version: 2` catalog; a missing vector raises, nothing is defaulted. With an explicit `require_receiver_traits=False` the
projection supports splash cover, shading and `R1` only: no preferential weights are invented for any species. This is different from an explicit all-zero vector, which is a declaration that the species' pathway is disabled. The flag is a Python argument; there is no model CLI option for it.

There is no species-trait switch in any SYRUP command. The scheduled runner (`masy.native_water_run.v1/v2`) belongs to the historical C1 sources (MASY, LEAF variant `8da`, catalog version 1) and does not read this trait. Its "v2" is the *run* schema, unrelated to catalog `schema_version: 2`.

## 4. Approved rules (user decisions of 2026-11-13/14; `user_decisions_complete_20261114.json`)

**Cover.** LEAF projects every eligible plant footprint (a circle) onto the cells it overlaps. Per species and cell, `f_s = min(Σ overlap area / cell area, 1)`; the total cover is `min(Σ_s f_s, 1)` ("add and cap", the MAHLERAN-style rule the user approved). Footprints are never scaled by an
optical-porosity or canopy-density factor: there is none in the model.

**Eligibility.**

| Use | Plants that count |
|---|---|
| Rain splash cover (`vegetation_cover_fraction` of the native event) | visible live **and** dead-standing canopies |
| Evaporation shading (daily E path) | visible **live** canopies only |
| Matrix and preferential infiltration indices | **live** plants only; no continuing dead-root enhancement is modelled |

A fully buried canopy has no visible cover. A partially buried canopy keeps its projected footprint while it remains visible. Roots are never used to infer canopy cover. Fully removed plants contribute nothing.

**Matrix law.** `R1 = Σ_s f_live,s · rootfrac_s[layer 1]` (sum over *all* live species, including one whose receiving weights are all zero). SYRUP forms `g = (a1 + R1)/(a2 + R1)` through the existing `matrix_infiltration.reference_vegetation_function`; the descendants' source
(`WaterCell.cpp`, `vegFunction`) uses the reference pair `(a1, a2) = (0.013, 0.067)`, kept only as `REFERENCE_A1_A2`: `a1, a2` are **explicit site controls**, never a default. `g` then enters the existing Tietjen-type placements once (it multiplies the ponded contribution, the pre-ponding
part is unscaled; manual section 13.4.1). There is no additional multiplier on conductivity or intake.

**Preferential law.** For each *enabled* species (receiving vector not all zero) `c_s = f_live,s · Σ rootfrac_s[layers 2..L]`. The per-layer map is `g_l = a3·b_l + (1 − a3)/(1 + C) Σ_s c_s u_sl` (manual 13.4.2) with `C = Σ c_s` over enabled species, `b` the explicit background soil vector and `u_s` the
species receiving weights. Background and species share **one** aggregate fraction (`max_fraction_per_hour`) and **one** aggregate intake cap (`max_intake_rate_m_s`); neither spends the supply or the cap twice. A species with all-zero weights contributes nothing to the preferential law but still counts in `R1`.

**What is Tietjen's and what is ours.** The hydrology is Tietjen-*inspired*: Tietjen, Zehe and Jeltsch (2009) and Tietjen et al. (2010) describe the two-layer parent model; the code follows later *descendants* of that source (Baldauf, Cantón and Tietjen 2023 and two Irob et al. packages), which are not verified copies of the 2009 program. The following are
declared adaptations of this project, not equations of the papers: the N-layer generalisation with explicit receiving weights; the per-individual live-plant projection replacing a cover-weighted two-layer root sum; the top-layer / lower-layers split of the live cover into `R1` and `c_s`; the add-and-cap species and total
cover; live-only infiltration and shading with live+dead splash; and the continuous-rate and exact Green–Ampt numerics already declared for Phase 1/2. No claim of exact reproduction of any published model is made.

## 5. LEAF side: projection, compact export, helpers

`leaf.water_vegetation` (projection version 3, schema `leaf.water_vegetation_projection`): `project_from_initial_state(state, catalog, current_surface, *, active_mask, max_edges, max_bytes, …)` and `project_from_plant_session(session, current_surface, …)` return a sealed, immutable projection
(`splash_cover`, `shading_cover`, `matrix_root_index`, `species_cell_cover`, `preferential_inputs`, `export_compact`, `check_session`). The projection stores no plants, no S×N and no P×N arrays. It records the actual surface and mask hashes, the full LEAF package digest and the external helper record.
The pure compact helpers are `compact_content_sha256`, `species_cover_from_compact`, `species_contribution_from_compact`, `preferential_inputs_from_compact` and `rederive_fields_from_compact` (CPU NumPy/Numba; CUDA is refused for the re-derivation). Every admitted operation takes an explicit host
byte budget, a reserve and an optional reader; the models are declared payload arithmetic. These are data validators, not file, head or session proof.

**The external helper.** The compiled/canopy builders import `benchmarks.spatial_mapping`, which lives *outside* the 28-file `leaf` package digest and is recorded separately (`leaf.water_vegetation.external_source_record()`: module, file, SHA-256, bytes). Do not assume a wheel installs it: LEAF's `pyproject.toml`
lists `packages = ["leaf", "benchmarks"]`, but `benchmarks/` has no `__init__.py` (a namespace directory). The reproducible runtime used by the qualification puts the pinned MAPLE `editable` and `source/src`, the SYRUP `src` under test and the LEAF **root** (supplying both `leaf` and `benchmarks`) on `PYTHONPATH`.
Audit of the actual runtime with both roots (CPU, existing venv): the SYRUP checkout also has a top-level `benchmarks/` directory without `__init__.py`; both are merged as one namespace (`benchmarks.__path__` lists both), in either `PYTHONPATH` order `benchmarks.spatial_mapping` resolved to the LEAF file and the example's law identity was identical.
The audit only shows the *current* situation: if either `benchmarks` directory ever gains an `__init__.py`, or a second `spatial_mapping.py` appears, the pin check below fails loudly rather than silently using another helper.

## 6. SYRUP side: site controls, binding, cached recipe, law

* `water_vegetation_binding.site_controls(*, a1, a2, a3, max_fraction_per_hour, max_intake_rate_m_s, background_receiver_weights, n_layers, n_cells=None, max_bytes, reserve_bytes=0, available_bytes_fn=None)` — every value required: `a1 >= 0`, `a2 > 0`, `a1 <= a2`; `a3` and `max_fraction_per_hour` in [0, 1] (scalar or exact `(N,)` for `a3` and `max_intake_rate_m_s`); `max_intake_rate_m_s >= 0` (`inf` legal);
  `background_receiver_weights` a raw nonnegative length-`n_layers` vector or an explicit `None`.
* `bind_projection(projection, controls, *, expected_leaf_package_sha256, expected_projection_sha256, expected_source, expected_external_helpers, active_mask, current_surface, session=None, max_bytes, reserve_bytes=0, available_bytes_fn=None)` binds one authentic projection to the controls. **Every pin is an external expectation:**
  the full LEAF package digest (compared with the digest SYRUP computes over the actually imported package files and with the projection's own record), the projection content hash, the complete expected source record, the expected external helper list `[{module, sha256, bytes}]`, the exact mask and the exact current surface. A record
  hash is an integrity check, never a trust or parent proof. The binding's semantic SHA is version 2 and includes the exact LEAF cell area; version-1 SHAs are historical and are never rebound.
* **Cell area.** `prepare_vegetation_law` refuses a native column whose `cell_area_m2` differs (exactly) from the projection's, before any admission or native call.
* **Compact cached recipe.** `capture_cached_inputs(binding, *, max_bytes, …)`, `export_cached_record(record)` (a closed `{meta, arrays}` pair: the 13 compact arrays, the enabled receiver table, the control maps), `stage_cached_inputs(meta, arrays, *, expected_record_sha256, expected_binding_sha256, expected_projection_sha256, expected_leaf_package_sha256, expected_external_helpers, expected_source, active_mask, backend, max_bytes, …)`
  and `cached_binding(record, *, expected_binding_sha256, max_bytes, …)`. Cold staging re-derives everything through LEAF's public helpers (H1 content hash, H3 cover fields, H2 receiver table), re-mints the controls and recomputes the binding and record SHAs. The record is labelled `origin = "cached_derived"`, `live_session_proof = False`.
  **A cached recipe is data, not a live generation:** it can feed `prepare_vegetation_law` on the current column and a held-context restore, but it can never *apply* a new vegetation generation — that always needs a genuine LEAF projection.
* `prepare_vegetation_law(column, binding, *, conductivity_m_s, suction_m, memory_budget_bytes, …, expected_matrix_op_sha256=None, expected_preferential_op_sha256=None)` prepares the real native operators (`g` once on the host, contributions streamed one species at a time) and can verify the CURRENT operator identities the caller expects.

## 7. Applying a generation, and what is current versus historical

`vegetation_lifecycle.apply_vegetation_generation(lineage, expected_head, binding, *, memory_budget_bytes, reserve_bytes=0, available_bytes_fn=None, host_memory_budget_bytes=None, host_reserve_bytes=0, host_available_bytes_fn=None)` returns the new head `Boundary`.

* **Zero time.** Nothing steps. Water `W`, the wetting history `J/I` (rebound bit for bit to the new operator), surface depth and face fluxes, the MAPLE bed, the routing reference, physical time, the nominal cursor, ledgers, sequence and the cold anchor / last actual event epoch are unchanged: the new head shares the old head's state objects and replaces only the
  operators and one sealed `VegetationRecord`. There is no event, no property refresh and no property-generation bump. Everything refusable runs before the head is replaced (last); a refusal leaves the old head retryable. The same binding is idempotent.
* **Eligibility.** Only a genuine COMPLETE `Boundary` (nothing pending). An occurrence, a daily hold, a mid-day state or a pending recession is refused. The same biology on a new legitimate surface/mask projection is accepted ("surface-only update", no sequence increase required); an older session sequence, another session / lineage / catalog / package / controls or an
  initial-state projection after a session generation is refused. CPU is run and tested; the CUDA boundary is served by `vegetation_lifecycle_cuda` (current device first, two pools).
* **Surface.** The binding's surface hash is compared with the authoritative *current committed bed surface*, never with the routing held surface. A restored recipe keeps its recorded held surface and does not compare it with a later bed (a storm may have changed the bed first).
* **Current versus apply-time ids.** `vegetation_attribution(obj)` returns the STABLE generation counter, recipe id and parent recipe id, binding / controls / projection / package / source / helper / mask / surface ids — and the CURRENT column / matrix / preferential / property-generation / `current_law_id` read from the actual holder now. The apply-time ids appear
  only under an explicit `applied` key and are labelled historical. An ordinary property refresh keeps the held `g` and the sealed preferential law (carried, nothing re-streamed) while the column and operator ids change.
* **Splash and shading.** The event factory builds the native `PondEvent` with the bound splash cover and `begin_occurrence` checks that the event's actual cover is that bound cover; the daily bridge uses the bound shading cover and refuses a caller cover. No legacy kinetic-energy formula is changed.

## 8. Water-free held context, diagnostic frames, and how they differ from physical files

| Artifact | What it is | What it is not |
|---|---|---|
| Held context `maple_syrup.floating_held_context/3` (CPU) and `/4` (CUDA) | a water-free *law* context: the true config inputs, the root column, depth `D0`, mask, Ks, the one compact recipe and the identities of the canonical law re-prepared from it; owning hook `floating_held_context.reconstruct_vegetation_held_context(record, expected, pins, …)` (CUDA: current device gated first); export always compares the vertical's CURRENT operator ids | not a checkpoint, not a head, not a source/parent proof, no water, no J/I |
| Every-N diagnostic frame, schema `maple_syrup.floating_moisture_frame_vegetation/1` | `W (L,N)`, canonical depth, mask, edges, area, nominal day, physical time, the frame identity and one per-frame `vegetation` block (generation, recipe, binding, source ids and the CURRENT law ids, or an explicit `null` before any generation) | **not** a restart: it cannot restore a run and it proves no source or parent |
| Physical file for a Boundary / closed day with a vegetation generation | the only restart file of a vegetated head: profile `maple-syrup-floating-lifecycle-checkpoint-vegetation/1` (CPU) or `/2-cuda` (section 9) | not a diagnostic frame, and it holds no every-N cadence |

**Every-N rules.** `VegetationMoistureOutput.create(directory, origin_state, *, expected_parent, every_n_days, limits, origin_date=None, caller_identity=None, …)`; `every_n_days` is a strict positive integer with **no default**, and no output object means output is off. A frame is written after a *completed nominal day* N, 2N, … counted from the bound origin ordinal,
after the actual transpiration and the audited daily close; wind or storm physical time never creates a day, and unsaved days are never reconstructed. The static sealed binding (`floating_output_vegetation.json`) deliberately holds no vegetation generation; each frame carries its own attribution. The v1 profile still refuses a vegetation-bound state and the two profiles refuse each other.
**Resume limit:** `VegetationMoistureOutput.resume(...)` re-binds an existing directory to a restored closed state by checking *manifests only* (module, profile, cadence, origin, run, source, geometry, caller identity, order bounds, the resume-day proof, and that past physical times are non-decreasing and not after the resumed state's time). It does **not** read or hash past array bytes. Full array and
seal checks happen in `read_vegetation_frame(path, *, max_return_bytes)` and when a due frame is replayed.

**Nominal calendar and event time.** A nominal day is 86400 s of calendar and is distinct from the physical time of an event. Wind, precipitation and interevent intervals never overlap; evaporation, transpiration and growth run only in interevent intervals — never during wind, rain or full storm recession — and are never caught up afterwards; calendar days still advance. The approved exception: on a *wet* interevent
day (standing surface water) the hydrology and evaporation run in short fragments that total exactly 86400 s, with transpiration and growth once at the day's end; dry days remain one update.

## 9. Physical checkpoint profiles for a vegetated head (accepted at a bounded scope)

**What exists.** Two new explicit file profiles that save and restore a head carrying one vegetation generation. They are native Python APIs (no command, YAML or CSV, no scheduler):

| | CPU | CUDA |
|---|---|---|
| Module | `maple_syrup.floating_lifecycle_checkpoint_vegetation` | `maple_syrup.floating_lifecycle_checkpoint_vegetation_cuda` |
| Profile string | `maple-syrup-floating-lifecycle-checkpoint-vegetation/1` | `maple-syrup-floating-lifecycle-checkpoint-vegetation/2-cuda` |
| Held-context schema inside | `maple_syrup.floating_held_context/3` | `maple_syrup.floating_held_context/4` |
| Backends | NumPy and Numba; Dense and published Windowed bed storage | CUDA twin, current device gated first, two memory pools |
| Public entry points | `save_floating_lifecycle_vegetation_checkpoint`, `load_floating_lifecycle_vegetation_checkpoint`, `vegetation_load_need_bytes` | the same names with the suffix `_cuda` (`load_…_cuda` and `vegetation_load_need_bytes_cuda` also take `expected_device_id`) |
| Result classes | `SavedVegetationLifecycleCheckpoint(path, manifest_sha256, kind, generation, recipe_id)`, `RestoredVegetationLifecycle(lineage, head, closed, kind, manifest_sha256, generation, recipe_id)` | the same two classes |
| Own error class | `VegetationLifecycleCheckpointError` | `VegetationLifecycleCheckpointCudaError` (a subclass of the CPU class) |

**Old profiles stay strict.** The earlier lifecycle profiles (`maple-syrup-floating-lifecycle-checkpoint/1` and `/2-cuda`) and the two new ones refuse each other: the manifest name, the profile string and the held-context schema all differ. There is no migration and no rebinding of an old file, and the old held-context `/1` and `/2` codecs stay free of LEAF. A head without a vegetation generation is saved by the old profile only; a head with one is saved by the new profile only.

**What may be saved.** Only a genuine *complete* `Boundary`, or a *closed* daily state of the head `DailyHold`, each carrying one vegetation generation (the owning check is `floating_event_lifecycle.saveable_head(lineage, vegetation=True)`). Refused: a head without a generation, a pending storm or recession, an Occurrence, a day still in progress and an unpublished Windowed bed. **Pending storm, recession and in-flight-day files are not supported.**
Saving never steps, drains, evaporates, withdraws, grows, refreshes properties or routing, or changes the head, the record or any counter.

**File layout.** One directory:

* `floating_lifecycle_checkpoint_vegetation.json` — the outer manifest, sealed last: profile, identity, source pair, backends, bed storage, the water-free held-context record meta and array table, the file table, the child digest, the closed scalar lifecycle record and the vegetation block. The vegetation block has exactly these keys: `generation`, `recipe_id`, `parent_recipe_id`, `applied` (`column_sha256`, `law_sha256`, `g_sha256`, `matrix_op_sha256`, `preferential_sha256` of the moment the generation was applied), `origin_at_save`, `binding_sha256`, `controls_sha256`, `projection_sha256`, `cached_record_sha256`, `leaf_package_sha256`, `external_helpers`, `source`, `mask_sha256`, `surface_sha256`.
* `context_arrays.npz` — the held-context arrays and the compact cached recipe (`veg.*` members); stored `.npy` members with closed names.
* `child/` — the genuine continued-origin child with the native bed and the current water, wetting history, depth, surface, routing origin and global datum (Dense schema `/1` or published Windowed `/2`).

No plant, `PlantSession`, projection or original bed is stored. If the head's record is live, the save captures its compact recipe with `capture_cached_inputs` first; a record that was itself restored from a file saves its own cached recipe.

**Saving.**

```python
from maple_syrup import floating_lifecycle_checkpoint_vegetation as vck

saved = vck.save_floating_lifecycle_vegetation_checkpoint(
    path, lineage, factory, identity,          # identity: your JSON-serialisable dict (strict, finite numbers)
    closed=closed_state_or_None,               # the genuine closed daily state for a closed-day head, else None
    original_topology=topology,                # the static generation-0 topology (not the original bed)
    available_bytes_fn=avail, memory_budget_bytes=budget, reserve_bytes=reserve)
manifest_sha = saved.manifest_sha256           # record it OUTSIDE the directory: it is the external seal
```

`factory(vertical, carried_topology, vegetation_cover_fraction) -> PondEvent` is the caller's static event recipe. It must build the native `PondEvent` with exactly the bound splash cover it is handed; the load compares it.

**Loading (a fresh process).**

```python
need = vck.vegetation_load_need_bytes(path, identity, manifest_sha, available_bytes_fn=avail)   # optional verified preflight
restored = vck.load_floating_lifecycle_vegetation_checkpoint(
    path, factory, identity, manifest_sha,
    leaf_package_sha256=ACTUAL_LEAF_PACKAGE_SHA, external_helpers=ACTUAL_HELPER_LIST,
    original_topology=topology, available_bytes_fn=avail, memory_budget_bytes=budget)
lineage, head = restored.lineage, restored.head   # a fresh lineage; the head is installed last
```

The caller supplies, from outside the file: the expected sealed manifest digest the save returned, the expected identity, and the *actual* runtime LEAF package digest and external-helper pins (never read from the file alone). A matching hash inside a JSON file never creates a head. The order is: external seal and closed headers, composed memory quota, file, header, hash and source-pair checks, the water-free held context, the static event built with the cached splash cover, the continued child (adoption last), a one-shot verified-restore token,
the cached-derived vegetation record, and the head installed last (for a closed day the genuine closed state is restored first). The original bed, owner, probe, provider, plants and `PlantSession` are not needed and are not reconstructed.

**Counters, origin and what is current.** A restored record keeps `origin = "cached_derived"`, its stable generation counter, recipe id and parent recipe id, and the historical `applied` ids; the counter and the recipe are the same from the file through the load. The *current* law and column ids are recomputed from the compact recipe on the restored column and compared with the child's, exactly as in section 7. A restored record is **not** a new live generation: a later new `apply_vegetation_generation` still needs a genuine LEAF projection. The verified-restore token is one-shot; a consumed token is refused before any array is read or uploaded.

**Errors.** Every refusal of the named older checkpoint families (the pond / floating / context / lifecycle / CPU-vegetation checkpoint errors) is re-typed by the public entry points as the profile's own class. Native memory classes (`NativeFloatingMemoryError`, `FloatingMemoryError`, and a real allocator `MemoryError`) pass through unchanged by class; nothing is classified by message text. The water-free held-context staging has no native memory class: a refusal raised there part-way through a load surfaces as the profile's generic error, not as a memory error. Caller-factory errors pass through. Invalid device ids, zero batch size and negative reserves are refused with the profile's own class.

**Memory.** Declared host arithmetic is admitted before arrays are allocated and each reserve is counted once. A newly captured live recipe adds its resident bytes once to later admissions; a restored recipe adds nothing. On load the retained cached recipe is an explicit term. These are payload models of small fixtures, not a large-grid fit.

**Not in this file.** No every-N output cadence, frame, scheduler, biology, wind state or pending event. The physical file is the only restart for a vegetated head. Diagnostic frames and the water-free held context are different artifacts (section 8): the output directory and its every-N cadence are bound to a restored closed state separately, with `VegetationMoistureOutput.resume` (manifests only), and a frame cannot restore a run.
Evidence for the day-5 case of a seven-day N = 3 run: a first interpreter ran days 1 to 3 (frame 3), applied a second generation, ran days 4 and 5 and paused at the closed day 5, copying the N = 3 directory and writing the physical file; then a second, fresh interpreter loaded from the files, the pins, the static case and the copied directory, resumed the writer at day 5 and ran days 6 and 7. The arrays, cursor, identities, vegetation attributions and the whole frame directory were identical to the uninterrupted run (NumPy and Numba, Dense and Windowed).

**Evidence.** Root's receipt records, on the exact accepted source: the checkpoint suite on CPU (63 passed, 10 skipped GPU definitions, 615.94 s) plus a separate import control for the GPU worker (1 passed); and, on the physical GPU0, the checkpoint GPU definitions (10 passed, including the Dense and Windowed fresh-process continuations), the cached inputs (5 passed, 1 skipped), the held context (5 passed, 1 skipped), the frames (3 passed) and the lifecycle (14 passed, 1 skipped); 37 GPU tests in all. The skipped tests need two visible devices and the fixture exposed one; they are reported as skipped, not passed.
**Scope of the evidence.** The seven-day, N = 3 experiment with a generation change from 1 to 2 and the closed-day-5 save and fresh load ran on CPU (NumPy and Numba, Dense and Windowed) in a separate interpreter. Its LEAF session advance used real published dry, non-growing LEAF days: it is not a plant-growth experiment. On the GPU, the physical fresh-process continuation (Dense and Windowed) and the frame and lifecycle compositions were checked as separate components; a single GPU day-5 / N = 3 fresh-process experiment was not run. The historical `applied` provenance is carried sealed in the file and is not independently re-derived on load.
The source lock used for verification is an identity-only record; it names which source was tested and is neither a parent proof of any saved file nor, by itself, an acceptance.

## 10. Backends, tested scope and limits

* Tested at a bounded scope on small fixtures: CPU NumPy and Numba, Dense and published Windowed storage, genuine completed boundaries and genuine seven-day nominal runs for the frames; actual-GPU controls by Root for the physical checkpoint, the cached inputs, the held context, the lifecycle core and the frames (counts in section 9 and in `phase25_qualification.json`).
* Not claimed: field validation, catchment or large-domain fit, memory fit beyond the declared arithmetic, scheduled MAPLE–SYRUP–LEAF acceptance, a command-line or YAML interface, general repeated-storm wetting-history recovery, a complete scheduled wind/soil clock, plant growth inside storms, vegetation feedback beyond the laws above, a physical checkpoint of a pending storm, a recession or an in-flight day, a single GPU day-5 / N = 3 fresh-process experiment, a positive vegetation-aware wind carry, GPU wind or Windowed wind, or a plant-growth experiment (the seven-day experiment used real dry, non-growing LEAF days).
* Source recipes, source guards, checkpoints and receipts of the historical releases are immutable; this edition adds identities and never edits one. Existing MAPLE environments only (`.venv`, `.venv-cupy`); no new environment is created by any procedure here.
* A helper-level or single-component test is never whole coupled acceptance.

## 11. A small runnable example (CPU)

`docs/manuals/examples/native_phase25_example/`: `species_v2.json`, `phase25_projection_example.py` and `expected_result_cpu_run.json`. The script asserts the three package pins (MAPLE `fbc504bb…`, 412 files; LEAF `38009872…`, 28 files; the reviewed SYRUP source `37d89f23…`, 180 files) and the helper hash, prints every imported path, validates the v2 catalog, projects a declared
3×3 fixture (closed-form checks of `π r²` cover, live-only shading and R1, dead-standing splash), builds site controls, binds with strict external pins, captures / exports / cold-stages the cached recipe, and prepares the native law on a prescribed floating column: the live and cached bindings give the same law identity, and a column with another cell area is refused. The three digests and the helper hash are re-read after the work and must be unchanged; the result records both the before and the after values.
The expected-result file is the actual output of a CPU run on a frozen copy of exactly the accepted SYRUP source (package `37d89f23…`, the code of commit `bb28af6c…`), the frozen LEAF tree with its helper and the pinned MAPLE materialization; the projection, binding, record, law and `g` values were identical to the earlier edition (only the runtime block changed). The same script run on the earlier 175-file held-context source stops with `PIN MISMATCH for syrup` (exit code 2).
It does **not** apply a generation to a lifecycle boundary (that needs a genuinely completed native event; see `tests/phase25_vegetation_lifecycle`), writes no frame and no checkpoint; the checkpoint procedure is the API example in section 9 and its executable evidence is `tests/phase25_vegetation_checkpoint`. All numbers are declared fixture values, not a calibration.
**Reproducing the example.** The example needs the exact accepted sources: SYRUP commit `bb28af6c…` (package `37d89f23…`, 180 files), LEAF commit `ff127d29…` (package `38009872…`, helper `333921bb…`) and the pinned MAPLE (see section 2). A package hash alone is not a Git revision: check out the commits named in section 2. The example output above was produced on frozen byte-identical copies of those sources; the documentation-only changes of this edition did not change them, so it was not run again.

## 12. Scientific references (from the existing manual edition; none added by this edition)

Tietjen, B., Zehe, E. and Jeltsch, F. (2009), *Water Resources Research* 45, W01418, doi:10.1029/2007WR006589. Tietjen, B. et al. (2010), *Ecohydrology* 3(2), 226–237. Green, W.H. and Ampt, G.A. (1911), *Journal of Agricultural Science* 4(1), 1–24. Smith, R.E. and Parlange, J.-Y. (1978), *Water Resources Research* 14(3), 533–538.
Saxton, K.E. and Rawls, W.J. (2006), *Soil Science Society of America Journal* 70(5), 1569–1578. Descendant sources: Baldauf, Cantón and Tietjen (2023), GitHub `selinaZitrone/Baldauf_et_al_Biocrust_Ecohydrology` commit `80b10e6b6b82ad16e9fd41a31901cc3157d880e9`, and the two Irob et al. Dryad packages (`ecohyd_reference_sources.md`).
**Add-and-cap is our declared choice.** The rule that species covers add and the total is capped at one was selected by the user (in the style of MAHLERAN's cell cover). It is labelled a project adaptation, not the exact published MAHLERAN canopy union or aggregation, and no external primary paper is claimed for it. The scientific background of the splash laws it feeds is the existing set cited in the Model Structure manual (Wainwright et al. 2008, doi:10.1002/esp.1624, as the framework; Wischmeier and Smith 1958; Brown and Foster 1987; Quansah 1981; Torri et al. 1987; Parsons et al. 2004).
**Verified bibliographic metadata.** Tietjen, B., Jeltsch, F., Zehe, E., Classen, N., Groengroeft, A., Schiffers, K. and Oldeland, J. (2010), *Ecohydrology* 3(2), 226–237, doi:10.1002/eco.70 (title, authors, DOI and journal volume checked against the publisher's indexed metadata page, https://onlinelibrary.wiley.com/doi/10.1002/eco.70; first published online 2009; the full text was not opened and no coefficient or equation is claimed to have been compared with it). Wainwright et al. (2008), *Earth Surface Processes and Landforms* 33(5), 813–826, doi:10.1002/esp.1624 (checked against the author's institutional publication list). No full-paper agreement with these sources is claimed.

## 13. What this edition does not establish

These tests do not establish field validation, a catchment-scale or large-grid memory fit, a complete scheduled three-model run, or general Green–Ampt drying and repeated-storm recovery. Memory bounds are declared payload models, not whole-process guarantees. Diagnostic-output resume checks manifests; the full frame reader checks arrays. Documentation checks are static (no paginated Office rendering).
Two Ruff diagnostics exist unchanged on the 953e6fd baseline (an unused `noqa` in WH1 and an unused variable in `native_source_witness`); two accepted new source modules end with one harmless blank line, kept so the qualified source identity stays exact. Root's receipt records no new lint diagnostic across the 87 explicitly checked files; it does not claim that Ruff reports zero.

## 14. Wind handoff (WH1) compatibility

Published main `953e6fdf85b73e3fdf5a1ac7b0d97dea4df7dab1` added one module, `floating_wind_handoff.py`: the Dense, NumPy, non-vegetated native wind handoff (WH1). It carries the soil water, wetting history J/I and bed state across a wind event for a head without vegetation, and that existing interface is qualified (its 68 controls pass unchanged); its publication, acceptance records and documentation are not rewritten here.
Phase 2.5 adds one bounded guard: a vegetation-bound boundary is **refused** by `stage_wind_handoff`, `adopt_wind_handoff` and `restore_wind_context` with an explicit `WindHandoffError`, before anything is read, allocated or changed; without the guard the handoff would have silently dropped the vegetation generation. Eight new controls pass beside the 68 old ones.
Not qualified: a positive vegetation-aware wind carry (there is no reset or drop fallback), GPU wind, Windowed wind and a full scheduled three-model run. Details: `phase25_wind_compatibility.md` (copied here from the reviewed source tree).
