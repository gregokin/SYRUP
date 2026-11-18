# Pooled evaporation-stress calibration — frozen protocol (stage 1, 2026-11-17)

Status: protocol and analysis helper only. No actual model result exists yet; nothing here is a finding. All three windows are used for fitting, so every fit is a **mechanism diagnostic, not independent field validation**.

## Question
Which single constant evaporation-stress threshold (volumetric fraction) best reproduces the observed JORN4_A, JORN4_B and MOAB2_C dry-down *changes*, with E = 1?

## Fixed model setup (same as the harmonic/geometric comparison)
- Accepted three 10 cm layers; texture-derived JORN/MOAB properties and floors from the pinned receipt `root_relative_density_probe.json`; same initial integrated profiles, weather, QA and dry periods (frozen window templates).
- E = 1, T = 0, open bottom, gravity on; no rain, wind, vegetation or bed run.
- Fits (d is never fitted; E is never fitted; ONE constant stress for all windows/sites within a fit):
  - `main`: harmonic interface mixing, d = 0.05
  - `secondary`: geometric interface mixing, d = 0.05
  - `sensitivity_d0`: harmonic, d = 0 (harmonic and geometric are identical at d = 0; verified bitwise at the selected stress).

## Score (`benchmarks/subsurface_stress_calibration/scoring.py`)
- Residual at day k ≥ 1: `(model(t_k) − model(t_0)) − (observed point(t_k) − observed point(t_0))`, equal to `aligned_model − observed`, `aligned_model = model_raw + obs(t_0) − model(t_0)`.
- Model(t_0) is the integrated initial layer mean of the layer containing the sensor; observed(t_0) is the raw point value. One constant offset per (window, sensor); never optimised, never applied to the physical state or water budget.
- t_0 is excluded. The original finite/QA mask after t_0 is kept; missing targets are counted out, never filled or dropped for sign (negative drying is kept).
- Pooled objective = Σ valid squared residuals / total valid observations (equal weight per observation; not a mean of window RMSEs). Zero valid observations is refused.
- Reported: per site/window/depth counts, RMSE, signed bias, the zero-change baseline (model change = 0), the raw and aligned model, the observations, the offsets and the unshifted RMSE.

## Search (`search.py`)
- Calibrated quantity: empirical stress threshold in [0.010001, 1.0] (native requires stress > layer-0 residual 0.01; the driver accepts ≤ 1). Not labelled the measured θ1500/wilting point.
- 199-point global grid including both endpoints; every local minimum (including plateaus and end minima) refined by bounded scalar minimisation (`scipy.optimize.minimize_scalar` bounded when importable, else golden section) inside its neighbouring grid points; global comparison over all evaluated stresses, ties to the smaller stress.
- Reported: full objective-vs-stress profile, all minima, whether the solution sits at a bound or on a flat plateau, and whether the stress is above the top-layer FC or saturation. A bound solution is reported as such; no unbounded optimum or physical meaning is claimed.
- Near-optimum interval (MSE within +1 % and +5 % of the minimum) is **descriptive, not a confidence interval**.
- Controls without refit, same E = 1 and offset rule: baseline stress 0.18 and each site's own predicted θ1500.

## Execution rules
- Every objective evaluation is the actual native column (`drydown_driver.run_column`, Numba) on three windows prepared once (pinned raw inputs, source guard before/after, exact-stress cache). Per-step and cumulative conservation checks are those of `run_column`.
- Verification of selected solutions and controls: NumPy/Numba agreement (inherited contract rtol 2e-12, atol 1e-14; bitwise at d = 0) and midpoint public-state export/reconstruction on each backend (not a production restart acceptance). CUDA only via explicit `--verify-cuda` (Root, GPU0 free); the helper never probes or falls back.
- Frozen source guards: `maple_syrup` 180 files `37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0`; `maple` 412 files `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`. Historical config defaults are not edited or rebound.
- Inputs located through `SYRUP_PHASE3_TASKS_ROOT=/home/okin/SYRUP/agent_handoffs/tasks`.

## Commands
```
# plan only (no raw data, no model): writes and validates the native configs
python -m benchmarks.subsurface_stress_calibration.calibrate --out NEWDIR
# Root only
python -m benchmarks.subsurface_stress_calibration.calibrate --out NEWDIR --execute [--verify-cuda]
```
Output (new/empty directory only): `configs/`, `prepared_inputs/<window>.json`, `fits/<fit>.json` (evaluated profile, minima, selected config, entire selected native trajectories, controls, verification), `summary.json` (source/raw identities before and after, file hashes).

## Limits
Three bounded windows with one sensor trio each; observation (point) versus layer-mean support differs; offsets absorb the initial representativeness difference but not later differences; the stress is an empirical lumped parameter that may compensate for E, d, property and PET error. Fits are descriptive and are not validation.

## Status after execution (2026-11-18)
Root executed the frozen protocol (actual CPU fits, actual CUDA verification of the selected cases). Results, figures and limits are in `report.md` and `SYRUP_JORN_MOAB_E1_stress_calibration_20261117.docx` (built by `build_report.py` from the frozen `data/`). The protocol above is unchanged; all three windows were used for fitting, so the fits remain mechanism diagnostics, not validation.
