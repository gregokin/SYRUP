# Joint d + evaporation-stress calibration, E fixed at 1 — frozen protocol (stage 1, 2026-11-18)

Status: protocol and analysis helper only. No actual model result exists yet; nothing here is a finding. All three windows are used for fitting, so each fit is a **mechanism diagnostic, not independent field validation**, and not a new model default.

## Question
Which single common diffusion coefficient `d` and single common evaporation-stress threshold (volumetric fraction) best reproduce the observed JORN4_A, JORN4_B and MOAB2_C dry-down *changes*, with E = 1? This repeats the previous pooled stress calibration (`../evaporation_stress_calibration/`) but also fits `d`.

## Fixed model setup (unchanged from the previous calibration)
- E = 1 exactly, T = 0, open bottom, gravity on; the same three 10 cm layers and approved texture-derived hydraulic surrogates (same density/air-entry floors, pinned receipt), same observed weather/PET, same prepared integrated initial profiles, same dry windows JORN4_A / JORN4_B / MOAB2_C. No rain, wind, vegetation or bed.
- ALL windows are calibration windows (53 valid post-initial targets: 39 JORN, 14 MOAB); there is no held-out claim.
- ONE common `d` and ONE common stress for every window and site within a fit. Two separate fits, one per interface-conductivity version (`harmonic`, `geometric`); they are compared as equivalent physics, not site-by-site fits.
- Exact accepted sources, never a live fallback: SYRUP 180 files `37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0`, MAPLE 412 files `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`. Sources and the pinned raw/air inputs are hashed before and after (`calibrate.source_guard` / `raw_guard`, reused).

## Score (reused unchanged: `calibrate.window_score_from_run`, `calibrate.score_record`, `scoring`)
Changes from t0 with a fixed offset per (window, sensor); t0 excluded; invalid observations not filled; equal weight per valid observation (pooled SSE / pooled n). Offsets never alter the physical water; unshifted/aligned series and budgets are kept in the selected records.

## Objective evaluation
EVERY evaluation of a pair (d, stress) runs the actual native compiled Numba column through `drydown_driver.run_column` on all three windows. `d` is set in the scenario (`diffusion_coefficient`, the actual run input) and in the config; the stress is set in `layers.theta_evaporation_stress`. Inputs are prepared once per window. Only light per-window (n, SSE, zero-change SSE) are cached; full histories are kept only for selected/control reruns (which must reproduce the cached SSE exactly). Invalid or non-finite `d`/stress, or a non-finite objective, is refused BEFORE any column runs.

## Search domain and frozen grids
- `d` in [0, 1] (declared initial range; covers the previous values), `d = 0` exactly evaluated and admitted. Stress in [0.010001, 1.0] as before. Bounds are honoured and never expanded silently.
- d grid, **43 points**: exact 0; 41 log-spaced values 1e-6 … 1 (exact end points); exact 0.05.
- stress grid, **85 points**: 81 linear points including both end points; exact 0.18; the previous fitted stresses 0.21717029762566153, 0.2171797728919738, 0.20644317480382438.
- 2-D grid = 43 × 85 = **3655 pairs** per interface version (the incumbents below are grid members, so they are evaluated exactly).

## Search (`search2d.py`)
1. 2-D grid evaluation.
2. Profile over d: at every grid d (including exact d = 0 and d = 1) the best stress, refined by a bounded 1-D search (SciPy bounded, only if it reports success; otherwise deterministic golden section, labelled). The profiled best stress and MSE per d are the lowest of ALL cached stresses at that d (a sampled refinement point may beat the returned minimiser); likewise the d along each stress edge; the 2-D grid is kept for contours.
3. Boundary edges: d refined along the stress = lo and stress = hi edges (1-D in z); stress refined along d = 0 and d = 1 by step 2.
4. Local 2-D refinements in z = log1p(d / 0.001) (z = 0 ⇔ d = 0, fine resolution near 0) and the stress, in the box of the neighbouring grid cells: up to 6 separated candidates (grid local minima, profile minimisers, edge optima; separation 0.08 in the unit (z, stress) square) PLUS the previous incumbents of that method — (d = 0, 0.20644317480382438) and (d = 0.05, 0.21717029762566153 harmonic / 0.2171797728919738 geometric). SciPy Nelder–Mead is accepted only on reported success; else deterministic bounded coordinate descent, labelled. No result is clipped after the fact; a result outside its box is refused.
5. Selection = minimum over EVERY evaluated pair (grid, profile, edge and refinement points, old incumbents); exact ties go to the smaller d, then the smaller stress.

Reporting: a solution at any bound is flagged `boundary_solution`; **if the selected d is the upper bound 1 the fit is a bounded solution only: no unbounded optimum is claimed, nothing is expanded, and Root must decide a bounded extension before acceptance** (`requires_root_bounded_extension_decision`). Lower saturated-stress plateaus are possible and reported as found. A dense grid with multi-start refinement and edge checks is evidence, not proof of the global mathematical optimum. 1 % and 5 % near-optimal regions are taken against the best evaluated (refined) MSE, using grid points only (point count, d/stress extents, connected components); a coarse grid with no point in the band reports an explicitly empty region (null extents). Descriptive only, not confidence regions.

## Comparison and checks (Root execution)
- New selected objective vs the previous fixed-d fits (d = 0.05 and d = 0) and the zero-change baseline, read from `../evaporation_stress_calibration/data/actual_cpu/{summary.json,fits/*.json}` after the old artifact manifest itself matches its immutable e11 sha256 `4af7b08fc2f1234a2130dabdd82e00b9acc9e28e3ed7fbd4c85388dea3504e23` (a changed but self-consistent manifest is refused) and every `actual_cpu/` file matches the hash it lists. The previous pairs are re-evaluated (never refit) and must reproduce within a relative 1e-9; the new minimum must be no worse than each previous incumbent within 1e-9 (they are included points), else the run fails.
- Controls: baseline (d = 0.05, stress = 0.18) and the previous fitted pairs, each with a fresh actual rerun and full records.
- Verification of the selected pair per method (old `calibrate.verify_case`): actual NumPy/Numba trajectory agreement and midpoint public column-state export/reconstruction on every declared backend (not a production checkpoint claim); at d = 0 harmonic and geometric must be identical. CUDA only with the explicit `--verify-cuda` flag on Root's authorized GPU 0 (never probed by the helper); N = 1 equivalence only, not scaling.

## Provenance
New configs carry `joint_calibration` (both d and stress fitted, E = 1 fixed, all windows used, no held-out); the earlier fixed-d / site-theta1500 statements are retained only under `historical_*` keys.

## Limits
E and stress are not separately identifiable in these dry quadratic-regime windows, and fixing E = 1 makes the stress a conditional value: it neither measures theta1500 nor alters retention/Ks. `d` is an empirical model coefficient fitted to 53 shallow-sensor changes, not a measured diffusivity. This is a calibration, not field validation, not a new default, and says nothing outside the three windows.

## Counts
Axes, `n_refine` and grid sizes are validated (finite, increasing, integer) before any objective call. Reports separate `n_objective_column_runs` (= 3 × cached pairs), `n_full_rerun_column_runs` (selected/control reruns with full histories) and their total. Physical-position flags use the actual layer FC/saturation and carry the site theta1500 only as `historical_site_theta1500`.

## Cost
About 4.7 × 10³ distinct pairs per method (3655 grid + profile/edge/refinement points) × 3 windows ≈ 1.4 × 10⁴ native column runs per method (the earlier stress fit used 208 stresses per fit). The helper holds one light record per pair; Root should budget the wall time accordingly.

## Command
`python -m benchmarks.subsurface_joint_stress.joint --out NEWEMPTY [--execute] [--verify-cuda]` — without `--execute` it writes/validates configs and the protocol only (no raw data, no model run).
