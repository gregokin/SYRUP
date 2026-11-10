# First subsurface column qualification — 2026-10-31

Accepted scope: the isolated forced column on `feature/subsurface-hydrology`.
This is a Tietjen-inspired, configurable-layer SYRUP component, with actual
soil evaporation, conservative redistribution, gravity drainage, and streamed
post-intake profile initialization. It is not a coupled storm, vegetation,
wind, checkpoint, or evolving-ground qualification. The first usable public
API is described in [implementation.md](implementation.md).

## Ownership and science

SYRUP owns physical water; LEAF supplies vegetation, roots and potential
plant demand in the future interface. Water-equivalent depth is stored in
FP64 `[layer, cell]` arrays. Water layers are distinct from sediment voxels.
The current geometry and common per-layer properties are fixed within a run.

The two-layer physics reference is
[Tietjen, Zehe & Jeltsch (2009)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589).
This implementation extends its storage/flux concepts to configurable layers;
it does not reproduce the whole published model or the MAHLERAN daily driver.
See the implementation guide for stage order, equations, source routines,
conditional retention-curve assumption and departures. The field-capacity
cascade is instantaneous within each step. Pairwise exact diffusion is split
sequentially over interfaces. Evaporation is a bare-soil upper-layer
approximation driven by supplied potential forcing. Weather-to-PET and
transpiration/plant competition are not implemented in this milestone.

## Actual checks

Codex independently executed these against the final runtime source:

- MAPLE `.venv`: `python -m pytest -q -p no:cacheprovider tests/subsurface`:
  **153 passed, 23 skipped** (5.47 s).
- MAPLE `.venv-cupy`, physical GPU 0 only: same command:
  **175 passed, 1 skipped** (7.23 s). These include genuine CUDA execution;
  the remaining skip requires more than the allocated single visible GPU.
- Existing rainfall/infiltration baseline: 157 passed/2 skipped in `.venv`,
  159 passed in `.venv-cupy`. Existing runtime files remain byte-identical.
- Ruff passes for all three new modules and `tests/subsurface`.
- The complete example in the implementation guide ran in MAPLE `.venv`
  with `PYTHONPATH=src`. Its 24-hour quiet-period water residual was
  `4.683753385137379e-17 m3`.
- Independent analytic two-layer exchange, gravity destinations, evaporation,
  matrix-eigenvalue diffusion and continuum cosine refinement checks pass.
  Selected timestep refinement reduced maximum storage error from about
  `3.71e-4 m` (one step) to `5.96e-6 m` (64 steps). Selected joint layer/time
  refinement reduced moisture L2 error from `6.01e-4` (4 layers) to `4.14e-5`
  (16 layers). These are bounded homogeneous-column examples, not universal
  convergence or heterogeneous-soil calibration claims.
- Saved pre/post optimization results are **bitwise identical for all 320
  arrays in each compiled backend**. Cases cover 1, 2, 10 and 40 layers,
  gravity on/off, masks, varying timesteps including zero, intake and E.
- Independently reproduced guard checks pass: zero-budget refusal before setup
  and state allocation, source/candidate/cumulative time refusal, finite derived
  parameters, unchanged inputs on rejection, and release of prior stream flux
  arrays before the next step. No new conservation epsilon was introduced.

The optimization computes the cell-invariant diffusion relaxation factor once
per interface per step. Numba uses local per-call scratch; CUDA uses a device
preparation kernel and keeps scratch alive through the one host packet read.
There are no per-step field uploads or per-layer launches. No old wind/water
physics or default was changed.

## Timings and memory

Warm median milliseconds per isolated column step, three samples, allocation
and validation included. NumPy is the reference; Numba is one serial CPU thread;
CUDA uses the allocated GTX 1080 Ti (GPU 0). These are not full-storm timings,
Fortran comparisons, multi-thread CPU comparisons, or combined-model scaling
proof. Persistent compiler caches were not cleared. First column preparation
in fresh processes was approximately 1.20 s NumPy, 3.24 s Numba and 2.90 s CUDA;
this includes setup and applicable compilation/cache loading, separately from
warm step timings.

| Layers | Cells | NumPy ms | Numba ms | CUDA ms |
|---:|---:|---:|---:|---:|
| 3 | 10,000 | 1.750 | 0.984 | 0.880 |
| 3 | 100,000 | 19.104 | 12.462 | 0.745 |
| 3 | 1,000,000 | 157.858 | 92.362 | 1.800 |
| 10 | 10,000 | 2.705 | 2.842 | 0.817 |
| 10 | 100,000 | 32.371 | 33.777 | 0.862 |
| 10 | 1,000,000 | 427.155 | 347.389 | 4.680 |

The uniform benchmark uses a 1 m profile, common residual/field-capacity/
saturation values 0.02/0.25/0.45, diffusivity `2e-7 m2/s`, a 3,600 s step,
0.3 mm upper-layer intake and potential E `3e-8 m/s`. Complete forcing,
state setup, every sample and exact source paths are in `qualify_column.py`
and `qualified_{numpy,numba,cuda}.json` in the task directory.

One state field at one million cells and ten layers is 80 MB decimal.
The compiled working-array requirement is about **482 MB** with these always
returned interface diagnostics, inputs and scratch; it excludes other models,
allocator rounding/cache, context, JIT/native workspace and Python overhead.
The requirement is not a reservation or a whole-process bound. Callers must
reserve other working sets. GPU live-input and reserved-pool observations in
the raw records were taken after outputs were released; they are not peaks.

The initial NumPy prototype understated its allocations. After independent
reproduction and correction, the measured new-allocation peak for a 100,000-cell,
three-layer step fell from 38.0 MB to 16.2 MB, within the revised 25.6 MB
step-allocation bound. Host and device requirements are counted separately.
The library does not retain time histories. Further possible work includes
optional interface diagnostics and sparse/specialized forcing, after actual
storm coupling is established.

## Exact sources and preserved work

- SYRUP branch base: `deb60c29834a4aad0bf8cb4cfbb611fc28ebe090` plus the
  recorded dirty working tree. No commit, push or merge was performed.
- Actual imported MAPLE: `963b06ad1cf0f3e2c420a7d7fb50939e3bd43b7d`, with its
  actual status and 386 Python-file hashes in `maple_runtime_identity.json`.
  This is a low-level backend/column check, not adoption of this revision
  into production recipes or shared-storage qualification. Existing locks
  and provenance guards are unchanged.
- All 67 pre-existing SYRUP runtime files and all recorded MAPLE Python
  sources stayed byte-identical. Prior unrelated benchmark work is preserved.
- LEAF reference: five selected source/report files copied and hashed at the
  design inspection. LEAF subsequently changed independently; four selected
  live files now differ. `leaf_external_deltas.json` records this. Neither
  Codex nor Claude edited LEAF. Refresh its interface assessment before
  integration; no live LEAF coupling is accepted here.
- Used existing MAPLE `.venv` and `.venv-cupy`, no new environment or install.
  Observed NumPy 2.5.3, Numba 0.67.0 and CuPy 14.2.0.

Exact 70-file runtime hashes, source snapshots, CLI commands, raw stdout/stderr,
Claude session records, corrections, oracles, measurements and review
information are under
`agent_handoffs/tasks/subsurface_column_20261031/`. All implementation/correction
sessions used ordinary permissions and have closed. One optional Claude report
write was denied by its permission matcher; Codex archived the returned report
payload in the authorized task directory. No requested work remains blocked.

## Whole-suite and integration limits

Default collection has three pre-existing errors: a MAPLE
`closed_windowed_commit` import `KeyError`, and two missing
`fixture_dense_alternation` imports in older conservative-water tests. The same
three errors occur with the original testpath list (1,969 tests collected)
and the augmented list (2,145 tests collected). No guard was disabled and no
unrelated source was changed to hide these errors. Therefore this is not a
passing whole-repository regression or acceptance of a coupled simulation.

Next integration must replace or extend the authoritative event soil bucket,
feed capacity/return changes back to surface flow at accepted steps, retain
soil and ponded water physically, and version handoff/restart metadata.
A final single-bucket total cannot uniquely supply a layered profile.
Preferential intake, surface-water E, LEAF uptake, calendar coordination,
per-cell soil stacks/maps and conservative remapping after erosion/deposition
remain explicit later work. Legacy benchmark/dry-reset modes stay unchanged.
