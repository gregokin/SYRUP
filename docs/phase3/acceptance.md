# Phase 3 acceptance — 2026-09-29

Accepted as a CPU rainfall/infiltration column milestone. Claude authored the implementation and reviewed the specification against MAHLERAN; Codex independently reviewed the code and callers, ran tests and an independent mm-unit calculation, requested corrections, and verified the completed work. Phase 2 baseline commit: `951f0dbdcb9b943b503e91947e7280e441f47bde`. Validation used the uncommitted Phase 3 tree; its exact source digest is recorded below.

## Delivered

- Immutable interval-ending rainfall schedules, exact integrals across forcing boundaries, constant rainfall, spatial scales/masks using MAPLE backend helpers.
- Vectorized MAHLERAN-inspired model-2 infiltration, retained soil water, linear drainage and explicit saturation return; fixed-conductivity model for controlled checks.
- Verified Plot 1 case loader and no-routing column CLI, actual MAPLE WaterState, unchanged sediment authority. Step subdivision respects every rain knot. Final arrays and cumulative fluxes are saved once; no per-step grid archive.
- Bound case/source hashes, actual SYRUP and MAPLE source provenance including dirty state, before/after source stability checks, refusal of output paths inside reference trees.
- Source correction: soil thickness initializes antecedent and maximum soil water in initialize_values_xml.f90 228–229; earlier Phase 2 “no consumer found” wording was inaccurate and is corrected.

## Verification

Commands from SYRUP, with MAPLE's existing interpreter; no installs or reference-tree edits:

```
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m pytest -q -rs -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -m ruff check src/maple_syrup tests
git diff --check
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m maple_syrup.column_experiment --case-dir outputs/plot1 --max-dt-s 1 --output-dir outputs/plot1_columns
```

Results: **222 passed, 2 skipped in 31.59 s**, both skips explicitly CuPy/device unavailable; Ruff and diff checks passed. Tests include independent scalar mm-unit legacy formulas, supply/ponding/saturation/drainage limits, local rain branching, rejection/nonmutation/masks, timestep refinement, actual MAPLE case and sediment invariance, corruption refusal, source-change refusal, reference output protection and provenance.

Final 60×20-cell Plot 1 diagnostic: 0.5 m cells, 300 m², 1620 one-second steps over 27 rainfall intervals. Final result in `outputs/plot1_columns/column_summary.json`, arrays `final_columns.npz` (generated outputs ignored by Git).

| Water term | m³ |
|---|---:|
| Initial soil water | 22.5000000000000 |
| Rain | 2.895599999999961 |
| Intake / net infiltration | 2.671164555150836 |
| Saturation return | 0 |
| Final surface water | 0.224435444849167 |
| Final soil water | 25.167033614657363 |
| Deep drainage | 0.004130940493483 |
| Water balance residual | 5.3290705182007514e-14 |

Declared accumulated balance tolerance: 2.812325494180245e-10 m³. Maximum cell cumulative residual: 1.0685896612017132e-15 m. Maximum ponding depth 0.0035264341533438395 m. MAPLE sediment-side before/after digests identical. Package source digests stable:

- MAPLE `d3d007024ff65abc2a0ff179f0f03bcd1c3b27f091493136cce849c34f7a4264`.
- SYRUP `9fc4475520aafa2c89fdbd25ece40211313a5c2e4b3539ab33d9164b148649b2`.

An independent Codex millimetre-unit transcription gives surface 0.2244354448491624 m³, soil 25.16703361465734 m³ and drainage 0.004130940493482694 m³ at dt=1 s. At dt=2, 1, 0.5, 0.25 s, surface totals are respectively 0.2244354339513897, 0.2244354448491624, 0.22443545029250161, 0.22443545301279 m³. This is equation/numerical evidence, **not execution of MAHLERAN Fortran**.

## Initial cost observations

Actual Plot 1 loop: 0.3232 s wall; setup/verification 1.5338 s, reporting 0.0553 s. NumPy, validation on every step. No GPU measurement. These are single-run observations, not controlled performance benchmarks.

Separate warm-start NumPy kernel diagnostic (20 steps, tracemalloc enabled; initial parameter arrays excluded):

| Grid | ms/step | Peak additional traced bytes |
|---|---:|---:|
| 60×20 | 0.717 | 207282 |
| 256×256 | 6.572 | 10556914 |
| 512×512 | 27.309 | 42210586 |

Includes tracing overhead; not process RSS, device memory, or a full coupled-model measurement. Temporary allocation remains material (~161 additional traced bytes/cell at the larger sizes). GPU scalar validation synchronizations and kernel fusion/workspace reuse require later profiling; no Python cell loop or per-step grid transfers is introduced.

## Limits and retained choices

Conductivity uses the XML positive mean, replacing its potentially negative normal draw. The legacy retained-storage use in capacity is kept, including antecedent water; local-rain branching, stable limiting arithmetic, timing boundaries and explicit water accounting are documented departures. Existing compiled-case reports remain immutable.

No hydraulic routing, outlet runoff, erosion/deposition, splash, wind event, full storm ET, plant dynamics or dry reset yet. Ponded water is retained at rainfall end; this is not event completion. Full production restart, GPU execution/scaling, a pinned production MAPLE dependency and Fortran-executable comparison remain unverified or future work.

Raw prompts, Claude reports, independent scripts and command logs are archived under `agent_handoffs/tasks/phase3a_rainfall/` and `phase3b_infiltration/` (local orchestration evidence, ignored by Git). Claude sessions: rainfall `f9d1310b-817e-41cb-8114-372ee4c0669d`; infiltration and correction `cad20714-5735-4f3d-b356-6f7f254a7a1f`. Preliminary output `outputs/plot1_columns_initial` predates corrections and is not the accepted result.
