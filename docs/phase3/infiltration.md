# Phase 3b — minimal conservative infiltration columns

Modules: `src/maple_syrup/infiltration.py` (kernel) and `src/maple_syrup/column_experiment.py` (Plot 1 no-routing diagnostic). Tests: `tests/phase3/test_infiltration.py`, `tests/phase3/test_column_experiment.py`. Controlling specification: `docs/phase3/infiltration_spec.md`.

Status: implemented, with a first correction pass (provenance, reference-root output refusal, operation-scale test tolerances). The author has not run the corrected tests; see the task report. There is no routing, sediment exchange, evapotranspiration, dry reset, plant dynamics or splash. No Fortran executable was run. This document makes no GPU or performance claim.

## Reference equations reviewed

Sources (MAHLERAN 1.2.3, read-only):

- `src/Subroutines_Water/infilt.for`, lines 38–207: the `inf_type < 5` path.
- `Program_Control/MAHLERAN_storm_setting_xml.f90`, lines 375–379, 412–555 and 801–809: pavement scaling, Ksat/psi/drain setup and `psi_mod`.
- `Subroutines_In_out/initialize_values_xml.f90`, lines 228–229 and 360–362: initial and maximum soil water.
- `Subroutines_In_out/calibration_xml.f90`: `calib.dat`.

The legacy per-cell update (mm, mm/s), for cells with `rmask ≥ 0`:

```
final_infilt = ksat * ksat_mod                                   (inf_model 1; ksat < 0 -> 1e-10)
             = lambda (1 - exp(-r2(i,j)/lambda))  if r2(i,2) > 0  (inf_model 2)
             = ksat                                otherwise
lambda       = -0.022891667 ln(pave) - 0.098575  (pave > 0),  0.16 otherwise;  pave = percent * 1e-4
c    = (psi + d(2)) (theta_sat - theta) final_infilt
def1 = cum_inf final_infilt / c;   f = final_infilt e^def1/(e^def1 - 1) if def1 <= 100 else final_infilt
water_in = r2 + d(1)/dt;  drain = (theta/theta_sat) ksat drain_par dt
f >= water_in : cum_inf += water_in dt;  d(1) = 0                       (complete run-on)
f <= r2       : cum_inf += f dt;  excess = r2 - f                       (rain excess)
otherwise     : cum_inf += f dt;  d(1) -= (f - r2) dt; v, q updated     (partial run-on)
drain limited to cum_inf, subtracted; cum_inf > stmax -> excess += (cum_inf - stmax)/dt, cum_inf = stmax
theta = cum_inf / stmax * theta_sat
```

Initialization: `ciinit = theta_0 * soil_thick * 1000` and `sminit = theta_sat * soil_thick * 1000` (mm), copied to `cum_inf` and `stmax`.

## Implemented form (SI, per cell, one step inside one constant-rain piece)

| Quantity | Meaning |
|---|---|
| `h` | ponded depth (MAPLE `WaterState.depth_m`) |
| `S` | retained soil water (legacy `cum_inf`). Includes antecedent water and has drainage removed; it is **not** cumulative infiltration. |
| `Smax` | `theta_sat · L` |
| `P` | `r · dt` |
| `A` | `h + P` |
| `K` | model 1: `Ksat`. Model 2: `λ(1 − e^{−r/λ})` for the cell's **own** `r > 0`, else `Ksat`. |
| capacity | `K / (1 − e^{−x})` with `x = S / ((ψ + h)(θs − θ))`. `1 − e^{−x}` is evaluated as `−expm1(−x)`. |
| `J` | `min(A, capacity · dt)` (intake) |
| `D` | `min((S/Smax) Ksat c_drain dt, S + J)` (drainage, leaves the column) |
| `O` | `max(S + J − D − Smax, 0)` (saturation return) |
| updates | `S' = min(S + J − D, Smax)`, `h' = (A − J) + O`, net infiltration `I = J − O` |

Limits, each written out explicitly:

- **`K = 0`:** capacity is 0.
- **`(ψ + h)(θs − θ) = 0`:** capacity is `K`. This covers no capillary term, the zero deficit, and the doubly zero `S = 0, ψ + h = 0` case.
- **`1 − e^{−x} = 0` with `K > 0`:** capacity is unbounded and `J = A`. This happens at `S = 0` with positive suction and deficit, or when `x` underflows. An overflow of `capacity · dt` resolves to `+inf`, and the `min` then selects `A`.

No NaN or negative value is cleaned up. The step validates that every output is finite and non-negative, that `S' ≤ Smax`, and that the per-cell balance `|h' + S' + D − (h + S + P)| ≤ 16 ε (h + P + S + J + D + O)` holds. Any failure raises `InfiltrationError`.

`column_step` is pure. Invalid inputs therefore cannot modify caller state: nothing is written, and the caller commits a step by adopting the returned arrays. `dt = 0` is the identity. Inactive (masked) cells keep `h` and `S` bit-for-bit and exchange nothing. Rain supplied on an inactive cell is rejected rather than dropped.

## Discrepancies from the legacy code (deliberate unless marked)

1. **Local rain in the model-2 conditional.** The legacy code tests `r2(i, 2)`, the rain at column 2 of the row, instead of `r2(i, j)`. It is corrected here to the cell's own rate. A test checks that four columns in one row each get their own `K`.
2. **Depth level.** The legacy code uses `d(2, i, j)` in `c`. Here the pre-step ponded depth `h` is used, because MAPLE holds a single time level.
3. **Numerics.**
   - `def1 ≤ 100` / `exp` is replaced by `−expm1(−x)` with explicit limits. The difference is below `e^{−100}` relative for large `x`.
   - The legacy `0/0` (`K = 0` → `c = 0`) falls through `NaN ≤ 100` to `f = final_infilt = 0`. Here that limit is explicit.
   - The legacy λ literals are default REAL (single precision). They are used here as FP64 decimals, a relative difference of about 1e-8.
4. **Negative conductivity.**
   - Model 1 legacy maps `ksat < 0` to `1e-10 mm/s`, but only for `final_infilt`.
   - Drainage in both models uses the raw `ksat`, so a negative sampled `ksat` gives negative drainage, which creates water.
   - Model 2 legacy also uses the unguarded `ksat` as `K` on dry-rain steps, which gives a negative `K`.
   - Here any negative `Ksat` is rejected, never floored.
   - For Plot 1, the configured `normal` draw (std 0.001 mm/s > mean 0.00025 mm/s) is replaced by the deterministic XML mean. This follows the user's stated default after an optional question. It is not a reproduction of any legacy realization. Per-cell arrays can be supplied later from sampling outside the kernel.
5. **Calibration.** `ksat_mod` multiplies only the model-1 `final_infilt` in the legacy code, and `psi_mod` multiplies `psi`. The kernel has no calibration multipliers. The Plot 1 runner refuses a `calib.dat` in the legacy input folder, and none exists for Plot 1, so both multipliers are 1.
6. **Rain timing.** Rainfall is integrated exactly per record interval (Phase 3a), and steps are split at every knot. The legacy switch lags by up to one `dt`.
7. **Flow.** The legacy partial run-on branch also updates velocity and discharge (`v`, `q`) for routing. Routing is excluded here, and surface water is storage only.
8. **Diagnostics not ported.** `t_ponding` and `cum_drain` are not kept as state. The runner reports cumulative drainage, and `excess` is represented by `h'`.
9. **Legacy setup defect, not reached by Plot 1.** For `infiltration-parameter_type ≠ 2`, `storm_setting` lines 524–529 write the drainage-parameter mean into **`psi`** instead of `drain_par`. `drain_par` then stays 0 and suction becomes the drainage mean. Plot 1 uses type 2, whose path (lines 530–553) fills `drain_par` correctly. The runner refuses any other type.
10. **Retained choice with a scientific limitation.** Smith & Parlange (1978) write capacity in terms of cumulative infiltration since ponding or storm start. The MAHLERAN code substitutes `cum_inf`, the retained soil water, which includes antecedent water (75 mm for Plot 1). This MAHLERAN-inspired choice is kept deliberately, as the specification requires, and is recorded in the run summary's assumptions. Its consequence for Plot 1 is that `x ≈ 75 / (46.6 × 0.14) ≈ 11.5`, so the capillary enhancement is about `e^{−11.5} ≈ 1e-5`. Capacity is therefore essentially `K ≈ r − r²/(2λ)`, and rain excess is small, controlled by pavement through λ. Early-storm sorptivity is effectively absent for wet antecedent conditions; this is a documented limitation, not a correction.

The explicit ordering (intake, then drainage from pre-step moisture, then overflow) follows the legacy structure. The method is explicit in time, and the tests show first-order convergence under refinement. No exact Fortran equivalence is claimed.

## Plot 1 no-routing diagnostic

```
python -m maple_syrup.column_experiment --case-dir outputs/plot1 --max-dt-s 1 --output-dir outputs/plot1_columns
```

Before any physics, `case_import.verify_plot1_case` does the following:

- It checks the binding and the hashes of `case.yaml`, `provenance.yaml`, the report and the fields.
- It calls MAPLE's `load_compiled_case`, which re-hashes every processed artifact, and compares the identity and artifact hash.
- It re-hashes MAPLE's staged source files.
- It refuses a MAPLE source digest that differs from the one bound at import, unless `--allow-maple-source-change` is given; the drift is then recorded.
- It re-runs the Phase 2 `audit_plot1` in a temporary directory. The recipe, XML, source and staged-copy hashes, the resolved legacy settings and maps, and **every** sidecar array must agree exactly, and `check_compiled_plot1` must pass.
- It refuses `calib.dat`.

The rainfall is parsed from the hash-checked staged copy.

Parameters (only this configuration is accepted: model 2, parameter type 2, one surface type, rain type 2, no suction, drainage, initial-moisture or final-infiltration maps):

| Parameter | Value | Source |
|---|---|---|
| Ksat | 2.5e-7 m/s | XML mean 0.00025 mm/s; deterministic override |
| ψ | 0.0466 m | XML 46.6 mm, `psi_mod` = 1 |
| drain parameter | 0.05 | XML |
| θ0 | 0.25 | XML; S0 = 0.075 m |
| L | 0.3 m | XML `soil_thickness`. A soil-water depth, unrelated to the MAPLE sediment column. |
| θs | ≈ 0.39 | Sidecar `saturated_soil_moisture` from `thetasat39.asc` |
| pavement | 0–90 % | Sidecar `pavement_cover_fraction` |
| rain scale | 1 everywhere | Sidecar `rainfall_scaling` (legacy `rmask` interior) |

- **Run window.** The run goes from t = 0 to the end of the rainfall record (1620 s), not to the legacy `stormlength` of 5400 s.
- **Steps.** Every constant-rate segment is split into equal steps with `dt ≤ --max-dt-s`.
- **End state.** Ponded water is retained in a MAPLE `WaterState`. It is built with `dataclasses.replace(case.water, depth_m=final)`, so the loaded mobile array is kept, and it is checked with `validate_water_state`. The status string says explicitly that this is **not** an event completion.
- **Sediment state.** The voxel column, active layer, availability, water mobile mass, ledger and topography are never passed to the kernel. A SHA-256 digest of all their arrays is compared before and after the run.
- **Budget.** The summary reports the budget in m³:
  - Checked identities: `surface + soil + drainage = initial + rain`, the separate surface and soil identities, and the rainfall integral against the schedule's exact total.
  - Tolerance: `16 ε (n_steps + n_cells) × (initial + rain + intake + drainage + return)` depth sums, with `ε` the FP64 machine epsilon.
  - A residual beyond its tolerance aborts the run.
- **Output location.** After verification and before anything is created, `case_import._refuse_output` (the Phase 2 rule) refuses an existing path or any path inside:
  - the MAPLE source root and package directory actually imported (from the resolved dependency, not only `--expected-maple-root`);
  - the MAHLERAN root in use and the root recorded in the report;
  - the recipe directory.

  The case directory itself is refused earlier.
- **Code identity.** The summary's `provenance` block records the following:
  - `maple_syrup`: `capture_syrup_provenance` (package source digest) plus the SYRUP repository HEAD and a scoped `git status` of `src`, `tests`, `docs`, `cases` and `pyproject.toml`. An uncommitted implementation is identified by its digest and listed as dirty.
  - `maple`: the full `capture_maple_provenance` taken during verification, and whether it matches the import binding.
  - `source_stability`: the SYRUP and MAPLE package digests from before verification and after the step loop. Any change aborts the run before outputs are written, so a summary exists only for code that did not change while it ran.
  - The Python environment record.
- **Outputs.** The new directory holds `column_summary.json` and `final_columns.npz`. The npz contains only the final grids and cumulative fluxes, with no per-step history. The summary also records parameters, time plan, domain, source hashes, MAPLE identity and verification, backend fingerprint, loop transfer counters, and process CPU and wall timings for setup, the step loop and reporting. The timings are not benchmarks.

## Backend, synchronization, persistence

- All per-cell arrays live in the namespace chosen by MAPLE's `resolve_backend`, and the step loop allocates no host arrays.
- Each validated step costs **one batched scalar read** (`DeferredChecks` → `read_flags`) of about 20 flags. On NumPy this costs no transfer. On CuPy it is one device synchronization per step, with **no grid transfer** in the loop.
- Budget totals are accumulated on the device and read back in one stacked transfer at the end. The final grids are copied to the host once.
- `column_step(validate=False)` skips the flag read. It is intended for a future periodic-validation policy, where a failure would be detected only at the next check and the global budget. The runner currently validates every step.
- No CuPy run was performed, and no speed or memory claim is made. The CuPy parity test skips honestly without a device.
- The step is pure and its state is just (`h`, `S`, model time). Adding restart persistence later needs only those arrays, the parameters' provenance and `t`. No production restart exists yet.
