# Phase 7i — heterogeneous hydrology qualification (preliminary reproduction and contract)

**Status: executed and accepted heterogeneous Plot1 CPU hydrology qualification.** Codex executed the checked whole Fortran application, three matched SYRUP storms, three original-routine storms and regression tests. Results and limits: [qualification](../../docs/phase7i/qualification.md). Original author report remains audit history under the task handoff.

Purpose: qualify SYRUP's prepared Numba hydrology against the actual legacy MAHLERAN Fortran on the **heterogeneous**
Plot 1 storm (normal conductivity draw, positive-truncated), using that run's own **full-precision** conductivity
realization and **exact applied rainfall**, with frozen elevation/routing, 5400 s, no splash, no ET, no dry reset, and a hydrology-only SYRUP replay. The full Fortran application retains its nonsplash sediment calculations; this task qualifies water only. The MAPLE bed is never written.

## Scripts

| Script | Role |
|---|---|
| `prepare_capture.py` | Copies the heterogeneous no-splash Linux derivative (`outputs/phase7/mahleran_plot1_no_splash_linux`) and adds **read-only** diagnostic hooks to `MAHLERAN_storm_xml.f90` only. Preserves parent/original hashes, writes a unified patch and a derivative manifest; refuses unsafe paths, a non-heterogeneous XML, a tampered parent, missing/duplicate anchors. |
| `../phase7/build_mahleran.py`, `../phase7/run_mahleran.py` | Unchanged phase 7 build and run tools (checked build, isolated new run directory, input/executable/output SHA-256 in `execution.json`). |
| `capture_data.py` | Parse/validate the three capture files, row/unit conversion (Fortran ring+north-first → SYRUP south-first interior, mm → m), conductivity validation and injection, static-consistency gates, safe-output and run-record checks. |
| `run_syrup_hydrology.py` | SYRUP hydrology replay (`prepare_hydrology` / `prepared_coupled_step`) with the injected field and exact forcing; optional lock-step comparison of EVERY public field of the reference `storm.coupled_step` (rtol 2e-12, atol 1e-14); `--substeps 1/2/4` sensitivity. Refused steps are recorded reproducibly (`step_failure.json`, `failure_input.npz`), never skipped or relaxed. |
| `run_controlled_fortran.py` | Replays the same field and forcing through the **existing unchanged** controlled original-routine driver (`benchmarks/phase4/reference_storm_driver.f90`, hash-pinned executable) at dt = 1, 0.5, 0.25 s; reports the legacy budget residual, stale-inflow gain and Crank-Nicolson closure. |
| `compare_hydrology.py` | Compares saved outputs only; predeclared targets; JSON report + `series.npz`. |

## Reproduction (new output directories only; nothing is overwritten)

```bash
cd /home/okin/SYRUP
source agent_handoffs/tasks/phase6_complete_event/env.sh
source benchmarks/phase7d/candidate_env.sh          # verified optimized MAPLE package (72310c49...)
P="$SYRUP_PYTHON"

# 1. isolated derivative with read-only capture hooks (touches only MAHLERAN_storm_xml.f90)
$P benchmarks/phase7i/prepare_capture.py --output outputs/phase7i/mahleran_capture_source
# 2. checked build and run with the unchanged phase 7 tools (about 9 s run)
$P benchmarks/phase7/build_mahleran.py --source outputs/phase7i/mahleran_capture_source --output outputs/phase7i/build_capture_checked
$P benchmarks/phase7/run_mahleran.py --prepared outputs/phase7i/mahleran_capture_source \
     --build outputs/phase7i/build_capture_checked --output outputs/phase7i/mahleran_capture_run
# 3. SYRUP prepared hydrology + prepared-vs-reference parity (dt = 1 s), then sensitivity
$P benchmarks/phase7i/run_syrup_hydrology.py --capture-run outputs/phase7i/mahleran_capture_run \
     --output outputs/phase7i/syrup_hydrology_dt1 --allow-maple-source-change
$P benchmarks/phase7i/run_syrup_hydrology.py --capture-run outputs/phase7i/mahleran_capture_run --substeps 2 --parity none \
     --output outputs/phase7i/syrup_hydrology_dt0p5 --allow-maple-source-change
$P benchmarks/phase7i/run_syrup_hydrology.py --capture-run outputs/phase7i/mahleran_capture_run --substeps 4 --parity none \
     --output outputs/phase7i/syrup_hydrology_dt0p25 --allow-maple-source-change
# 4. existing unchanged controlled Fortran driver on the same exact field/forcing
for dt in 1 0.5 0.25; do
  $P benchmarks/phase7i/run_controlled_fortran.py --capture-run outputs/phase7i/mahleran_capture_run \
     --dt-s $dt --output outputs/phase7i/controlled_dt${dt/./p}
done
# 5. comparison (reads saved outputs only)
$P benchmarks/phase7i/compare_hydrology.py --capture-run outputs/phase7i/mahleran_capture_run \
     --syrup dt1=outputs/phase7i/syrup_hydrology_dt1 --syrup dt0.5=outputs/phase7i/syrup_hydrology_dt0p5 \
     --syrup dt0.25=outputs/phase7i/syrup_hydrology_dt0p25 \
     --controlled dt1=outputs/phase7i/controlled_dt1 --controlled dt0.5=outputs/phase7i/controlled_dt0p5 \
     --controlled dt0.25=outputs/phase7i/controlled_dt0p25 --output outputs/phase7i/comparison
# tests (synthetic data; the Plot 1 geometry test skips without outputs/plot1)
$P -m pytest tests/phase7i -q
```

The controlled driver executable is the one used by the earlier audits
(`outputs/phase4_reference_final/dt1/build/reference_storm`, SHA-256 `33e74258…`); the script refuses any other binary
(rebuild with `benchmarks/phase4/storm_reference.py` if it is missing). If a gfortran-built executable needs the
toolchain's runtime library path, export `LD_LIBRARY_PATH` as the earlier replay did.

## Capture contract

All capture files are plain text in `outputs/phase7i/mahleran_capture_run/Output/`, written with `status='new'`,
`ES25.16E3` (17 significant digits; decimal text round-trips every double and every widened single exactly), each ended by
a completion marker that exists only if the loop finished. Units are MAHLERAN's: depth mm, rate mm/s, unit discharge mm²/s,
velocity mm/s, conductivity mm/s, cell size mm. Fortran arrays are `(i, j)` with the exterior ring at `i = 1, nr2` and
`j = 1, nc2` and `i` running north → south; SYRUP uses the 60 × 20 interior, row 0 = south
(`capture_data.interior_south_first`).

* `syrup_hydro_static.txt` (after setup, before iteration 1): `ksat` (the realization), `psi`, `theta_sat`, `theta`,
  `cum_inf`, `stmax`, `drain_par`, `pave`, `slope`, `ff`, `rmask`, `aspect`, `order`, initial `d`/`q`, and the
  configuration scalars (`dt`, `dx`, `iroute`, `ff_type`, `inf_type`, `inf_model`, `ksat_mod`, ...).
* `syrup_hydro_steps.txt`: per iteration the rate **applied** in that step (read before `infilt`; includes the one-second
  switching lag), outlet discharge from the model's single-precision `q_plot` AND a double-precision sum over the same
  outlet cells, the Crank-Nicolson face export `½ dt dx (Σq_old + Σq_new)`, and interior surface/soil/drainage/excess sums.
* `syrup_hydro_final.txt`: synchronous depth/velocity at the model's own strict-greater single-precision peak (its
  `output_hydro_data_xml.f90` 489–505 logic replicated) and at the double-precision peak, the model's own `dmax`/`vmax`
  (cross-check, expected difference 0), and the final state.

The application declares `dt`, `dx`, `dy`, `rval`, `stormlength` and `q_plot` as default REAL by implicit typing; the
capture therefore holds single-precision values widened exactly, which is what the model really used.

## Refusals (before any simulation)

Unsafe or existing output paths; truncated/malformed/duplicated/non-finite capture; non-positive or non-finite or
wrongly shaped conductivity; execution record not completed or any recorded output hash changed; configuration other
than method 5, friction type 1, infiltration model 2, rain type 2, calibration 1, `dt = 1`; any of slope, friction,
saturation, suction, drainage, soil storage, initial moisture/soil water, pavement, rainfall scale, active mask,
outlets, aspect or routing order disagreeing with the SYRUP case after row conversion; non-dry initial state.


The captured maximum soil storage and initial soil are first checked against the XML soil thickness converted to the legacy REAL32 representation and explicitly bound for the comparison. The full imported mask is south-first and is reversed for exact comparison to the Fortran mask; positive non-export perimeter cells are permitted, with exact physical outlets required. Production initialization defaults remain unchanged.
