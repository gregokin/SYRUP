# CPU Newton: Plot 1 and RFID results

The selectable NumPy and Numba Newton solvers preserve the corrected MAHLERAN-style implicit hydraulic equation. Numba Newton reduces full-storm water-hydrology time by **14.3% on Plot 1** and **13.6% on RFID** relative to the corresponding Numba bisection. NumPy Newton reduces time by **38.1%** and **42.6%**. Bisection remains the default. There is no GPU Newton implementation in this task.

## Measured warm storm time

Seconds; median of three complete, validated storms. Parentheses show observed minimum–maximum, not confidence intervals.

| Method | Plot 1: 5400 model seconds | RFID: 2700 model seconds |
|---|---:|---:|
| NumPy bisection | 78.810 (78.061–79.167) | 159.243 (159.196–163.664) |
| NumPy Newton | 48.803 (48.621–49.599) | 91.413 (91.252–91.859) |
| Numba bisection | 2.050 (2.010–2.179) | 4.275 (4.139–5.026) |
| Numba Newton | **1.756 (1.746–1.760)** | **3.694 (3.683–3.772)** |
| Original Fortran Newton | Not run: existing adapter lacks model 2/pavement | 2.282 (2.255–2.401) |
| Original Fortran bisection | Not run: existing adapter lacks model 2/pavement | 11.360 (10.945–11.573) |

Numba Newton on RFID takes **1.619 times** the native Fortran Newton loop time. Native Fortran bisection is substantially slower than its Newton method. This is a water-only comparison; it does not measure sediment transport, voxel exchanges, wind, or total application startup.

The full storms include infiltration, drainage, ordered routing, adaptive acceptance/retries, and 60-second reporting. Maximum timestep is 1 s. Plot 1 has 1200 cells at 0.5 m and uses pavement/Hawkins infiltration; RFID has 5697 active cells in a 104×58 array at 0.1 m and uses fixed-Ksat infiltration. Terrain and routing remain frozen; splash and sediment are absent. Plot 1 used the existing 40 bisections and RFID 64; Newton allowed 50 passes with safeguarded bracket completion. These are the established case configurations, rather than an equal-iteration comparison.

Each contender had a separately recorded first call followed by a complete untimed warm storm. Timed rounds used forward, reverse, forward order. All **30 measured storms** succeeded. No task writer, test suite, or other task benchmark ran concurrently. Existing user MAPLE jobs were preserved on this shared WSL host; the RFID bisection range illustrates host variability. Repeated timings are more informative than a single ratio.

Numba first calls, including compilation, the first step and driver setup, took 2.778/2.356 s for Plot 1 bisection/Newton and 2.680/2.235 s for RFID. They are excluded from the table and are not isolated compiler timings. SYRUP table values time evolution; full case preparation and post-run guards/captures are outside the timer. Fortran values use the driver's internal loop timer; process launch, input and final output are excluded, with process times retained in the raw records. The new benchmark progress wrapper prints only before and after the original timers.

## Science and conservation checks

The change replaces only the root solve for `h + c k h^(3/2) = R`. The conservative storage/flux identity, D4 donor ordering, Darcy-Weisbach relation, column physics, timestep rejection checks, and tolerances are shared with bisection. NumPy evaluates roots in vectorized dependency levels; Numba compiles the ordered sweep and uses the existing prepared column/routing infrastructure. Safeguarded Newton retains a bracket and has bounded bisection completion for difficult numerical inputs. No measured case required a safeguard or fallback.

| Diagnostic, Numba Newton | Plot 1 | RFID |
|---|---:|---:|
| Mean Newton passes per iterated cell-step | 2.031 | 2.937 |
| Maximum passes | 3 | 4 |
| Iterated cell-steps | 2,892,496 | 14,322,963 |
| Safeguards / fallback cell-steps | 0 / 0 | 0 / 0 |
| Accepted / rejected timesteps | 5400 / 0 | 2772 / 72 |
| Full-storm water residual, m³ | 4.40e-14 | −2.96e-13 |

Iteration statistics come from separate untimed diagnostic runs. NumPy and Numba means differ slightly because their column kernels can differ by an ulp in libm calculations; root implementations agree on identical inputs. Both solvers retain the same timestep counts as bisection. The reason for each RFID rejection was not separately classified.

All SYRUP final depth, soil-water and discharge comparisons passed the **unchanged** `rtol=2e-12`, `atol=1e-14` backend bounds. Full-storm water budgets, frozen-bed checks and source-identity guards passed the existing MAPLE-derived conservation rules. Full default-bisection outputs (depth, soil water, discharge and hydrographs) remained **bitwise identical** to immutable pre-edit captures for both cases.

Plot 1 is dry at the end, so final zero surface fields alone would be insufficient. Independent wet snapshots at 600, 1200 and 1620 s compare depth and velocity as well. Newton versus bisection maximum wet depth difference was 1.81e-15 m and velocity difference 1.49e-14 m/s; all fields passed the existing bounds. Full-storm export changed by 2.33e-14 m³ (relative 1.42e-13), and the outlet-flow hydrograph relative L2 difference was 5.86e-13.

RFID wet snapshots at 600, 1200 and 2640 s likewise passed. Newton versus bisection maximum wet depth difference was 6.11e-15 m and velocity difference 4.16e-17 m/s. Full-storm export and peak outlet flow are identical across all four SYRUP methods. NumPy/Numba Newton wet fields and final fields passed the same unchanged bounds; complete storm outputs are not generally bitwise equal across these backends.

Wet snapshots are separate untimed driver runs with report-aligned segment boundaries, rather than captures inside the measured full-storm trajectories. The wet helper asserts NumPy/Numba Newton agreement and records Newton/bisection deviations; Codex's final numeric audit independently asserts every recorded wet comparison flag. These checks establish wet-field agreement for those runs.

### Native Fortran qualifications

The RFID driver calls the unchanged original MAHLERAN routines (`iroute=2` Newton, `iroute=5` bisection), compiled with GNU Fortran 13.3 and `-O2`. Its existing adapter implements infiltration model 1 with zero pavement. Actual Plot 1 uses model 2 and pavement; running it through this adapter would change hydrology. A pavement/model-2-aware driver is feasible but was not built here; Plot 1 Fortran contenders are explicitly refused.

The native RFID run takes 2700 fixed steps, while SYRUP takes 2772 accepted steps and 72 rejected trials. SYRUP also retains conservative corrections and checks rather than reproducing the native stale-inflow/numerical behavior. Fortran Newton retains a water gain of 0.008188 m³ (0.1416% of rain); its export is 0.2141% higher than SYRUP. Relative L2 final differences are 0.09455% for depth, 0.001235% for soil water and 2.7388% for discharge. These differences are disclosed, not tolerated away. Consequently the remaining 1.619 ratio is not a pure language/compiler comparison.

The harness does not report a full Fortran/SYRUP hydrograph norm: Fortran saves 45 regular reporting rows, whereas SYRUP saves 46 including the 2641 s forcing boundary. Export, peak, peak timing and final maps are compared. No claim of identical native/SYRUP storm histories is made.

## Verification, provenance and reproduction

Codex independent verification: **2261 passed, 4 intentional multiple-device skips**, with actual GPU 3 and the working Fortran toolchain. Ruff and whitespace checks passed. Additional full baseline, wet-field and actual-CuPy refusal checks passed. The GPU checks establish unchanged guards and rejection without mutation or counted transfers; they do not qualify GPU Newton. Restart and root edge cases are tested, but the timing qualification remains water-only and fixed-terrain.

Claude's final read-only review found no blocker in the results or Codex benchmark/verification helpers. Codex confirmed the disclosed qualifications, independently verified all archived source hashes and test counts, and recorded the review disposition in the task folder.

Environment: Intel i9-7900X, Python 3.12.3, NumPy 2.5.2, Numba 0.67.0, WSL Linux 6.18. The process-wide maximum host RSS was 358132 KiB for the Plot 1 invocation and 390308 KiB for RFID; these include preparation/other contenders/JIT and are **not per-solver memory peaks**.

Benchmark source is recorded by `agent_handoffs/tasks/newton_cpu/verification_manifest.json` and `verification_source.tar.gz` (369 files), and remained unchanged through timing and the numeric audit. SYRUP HEAD was `031bce75d10371fec0701bf8bc59685e5579f84a` plus archived pre-existing/task changes; HEAD alone is not the benchmark identity. Package digest: `5be19bf94e717dbdde84c13839196489c86b222c0f8355e41dbf7b61dfa43414`. Actual imported MAPLE source digest: `72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65`. MAHLERAN reference HEAD: `305bd95d32123f13708be2f9a88e42ddd45d6f28`, with original routine hashes and executable/compiler metadata in RFID's JSON. Documentation finalized after timing is recorded separately.

Raw results:

- `outputs/newton_cpu/{plot1_full,rfid_full}/comparison.json`: individual timings, guards, controls, startup, iterations, comparisons and provenance.
- Corresponding `final_fields.npz` and `hydrographs.npz`; RFID's `fortran/` holds native histories and final maps.
- `agent_handoffs/tasks/newton_cpu/`: independent test XML/logs, baseline captures, wet-field comparison/NPZ, launch/process logs, helper hashes and `final_numeric_audit.json`.

Reproduce with the pinned MAPLE dependency, sourcing environments **in this order**, and use new output directories:

```bash
source agent_handoffs/tasks/phase7_matched_benchmark/gpu_env.sh
source benchmarks/phase7d/candidate_env.sh
export PATH=/home/okin/MAPLE/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=
python benchmarks/newton_cpu/compare_cases.py --case plot1 \
  --case-dir outputs/plot1 --output-dir /tmp/newton_plot1_new \
  --contenders bisection_numpy,bisection_numba,newton_numpy,newton_numba \
  --rounds 3 --allow-maple-source-change
python benchmarks/newton_cpu/compare_cases.py --case rfid \
  --case-dir outputs/rfid/case --output-dir /tmp/newton_rfid_new \
  --contenders bisection_numpy,bisection_numba,newton_numpy,newton_numba,fortran_newton,fortran_bisection \
  --fortran-exe outputs/rfid/fortran_build_final/rfid_water_driver \
  --rounds 3 --allow-maple-source-change
```

The provenance override permits the documented adopted MAPLE snapshot versus the case's older recorded snapshot; it does not disable case content, bed or within-run source guards. Existing output directories should not be reused. New sources must be reverified, not treated as the archived benchmark.
