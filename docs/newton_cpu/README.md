# Safeguarded Newton root solver (CPU, NumPy and Numba)

Status: implemented by Claude and independently verified and benchmarked by Codex.
See [full-storm timings and qualifications](results.md): 2261 independent tests passed, with four intentional multiple-device skips.

## What it is

A **selectable CPU root solver** for the cell equation of the corrected method-5 step,

    h + c k h^{3/2} = R,    c = dt / (2 dx),   k = sqrt(8 g S / f),   R = h_start + c (Qin_old + Qin_new - q_old)

It is *not* the explicit or local-inertia physics. Everything else is shared with the bisection default and
unchanged: D4 donor order, Darcy-Weisbach `k`, coherent old flux, the storage identity `h_new = R - c q`, face
volumes, per-cell and global water balances, the constitutive check `|h_new - h_flow| <= root_tolerance_m`
(1e-11 m, unchanged), Courant and negative-RHS rejections, finite/negative output checks.

**Default stays bisection**, bit for bit: a full Plot 1 (5400 s) and a full RFID (2700 s) prepared-Numba storm with the
default control reproduce Codex's pre-change baseline captures (`agent_handoffs/tasks/newton_cpu/baseline_*_fields.npz`:
depth, soil water, discharge and the hydrograph) bitwise.

## Interface

* `routing.route_step(..., root_solver="bisection" | "newton", newton_max_iterations=50)`
* `StormControl(root_solver=..., newton_max_iterations=...)` (last two fields; `coupled_step`, `evolve`, the prepared
  Numba hydrology and sediment events all read them)
* `RouteStep.root_solver`, `.newton_max_iterations` (0 for bisection), `.root_stats` (None for bisection)
  and `.bisection_iterations` (**0** for Newton: no bisection count is configured; `bisection_iterations` is still
  validated). Newton failures name the Newton solver (`Newton root solver did not reach root_tolerance_m ...`); the
  bisection message is unchanged.
* `routing_newton` (new module): `newton_root_scalar` (pure-Python specification), `newton_root_level` (NumPy, fully
  vectorized per dependency level; the optional `small_level > 0` cell-by-cell helper is off by default and never used
  by `route_step` or the benchmarks), `compiled_root` / `compiled_sweep_newton` / `run_sweep` (Numba, no fastmath, no prange).
* `hydrology_numba.prepared_coupled_step` uses a second kernel set (`_kernels("newton")`) when
  `control.root_solver == "newton"`; the default kernel set only gained a trailing, unused `stats` argument.
* Refused explicitly, before anything is computed or mutated: Newton with `implementation="cuda"` (`route_step`,
  `StormControl.validated`, `storm.coupled_step`, and a guard in `hydrology_cuda.cuda_step_with_packet`), Newton on a
  non-NumPy graph, invalid solver names, `newton_max_iterations` outside `[1, 1000]`, bools, floats. There is no
  automatic switch to a different requested solver/backend. The Newton algorithm itself has the numerical bracketed
  bisection completion described below. `legacy_stale_inflow_step` (the non-conservative comparison tool) has no Newton option.

## Algorithm (per cell; `routing_newton` module docstring is normative)

1. `R > 0` false (0, -0, negative, NaN): `h_flow = 0`, exactly what bisection leaves.
2. `trial(R) = ((sqrt(R) R) k) c + R` not above `R` (k = 0 pit/zero conveyance, c = 0, flux term lost in rounding,
   subnormal R): `h_flow = R` analytically, zero iterations.
3. Otherwise `g(h) = trial(h) - R` is increasing and convex. The root obeys `h = R / (1 + a sqrt(h))` (a = c k), which
   gives the valid upper bound `x0 = R / (1 + a sqrt(R / (1 + a sqrt R)))`; Newton from the right converges
   monotonically. The ratio `x0 / root` is **not bounded** (it grows ~ `(a sqrt R)^(1/6)`); it is close in the observed
   physical cases, and the iteration cap with the bracketed fallback protects every range.
   `trial` is evaluated in the *bisection's own operation sequence*. A bracket `[lo, hi]` (`trial < R` on `lo`) is kept
   from `[0, R]`; a step outside it (or NaN) becomes the midpoint (counted as a safeguard step). Stop when the step is
   `<= 16 eps x`.
4. Finalization keeps the bisection invariant `trial(h_flow) < R` (hence `h_new >= h_flow >= 0` with no clipping):
   points `x (1 - m eps)`, m = 0, 1, 2, ..., 64 are tried. If Newton hit `newton_max_iterations` or no such point exists,
   the bracket is bisected (<= 1200 halvings, 4 eps wide; guaranteed to terminate) — the *fallback*.

Typical cost: 2-4 trial evaluations per cell (RFID/Plot 1 smoke runs: mean 2.0-2.9 passes, max 3-6, zero safeguard or
fallback steps) versus 40-64 for bisection. Roots are within a few ulp of an independent 60-digit reference
(`tests/newton_cpu/test_newton_scalar.py`) over wide ranges (R from 5e-324 to 1e8, k to 1e12, c = 0 to 1e3).

## Equivalence

* Pure-Python, NumPy-vectorized and Numba root functions are **bitwise identical** (value, pass count, safeguard
  steps, fallback flag): only IEEE `+ - * / sqrt` are used (no `cbrt`/`pow`/`exp`, no FMA contraction).
* `route_step` array vs numba Newton: all returned fields bitwise equal (tests). Prepared Numba Newton vs the reference
  `coupled_step` Newton: within the declared water bounds (rtol 2e-12, atol 1e-14; the column kernels call libm).
* Newton vs bisection: differences are bisection's own truncation `R 2^-40` (~1e-12 relative to the depth) and below;
  Newton's constitutive residual is smaller. Short smoke storms (not timing claims): RFID 900 s max |Δdepth| 4.4e-16 m,
  Plot 1 900 s 1.0e-15 m, export relative difference <= 1e-12.

## Restart / checkpoint

`StormControl` is stored in checkpoints. The two new fields are written **only when non-default**, and a
checkpoint without them loads as the bisection control, so existing checkpoints and default checkpoints are
unchanged (schema string unchanged). A Newton control round-trips and resumes equivalently (`test_newton_checkpoint.py`).

## Not covered

No GPU Newton. The existing original-routine driver is an RFID adapter (hardcoded fixed-Ksat model 1, zero pavement) while the
actual Plot 1 columns are `pavement_hawkins` (model 2), so it would compare different hydrology; a model-2/pavement-aware
adapter around the unchanged originals is feasible but not built in this task. Therefore
`benchmarks/newton_cpu/compare_cases.py --case plot1` refuses the Fortran contenders; `--case rfid` runs the
original `iroute = 2` (native Newton) and `iroute = 5` (bisection) through `benchmarks/rfid/fortran_timing.py`.
`storm_experiment` (the Plot 1 CLI) was not modified: use the benchmark scripts' `--root-solver`.

## Verification commands (author's)

    source agent_handoffs/tasks/phase7_matched_benchmark/gpu_env.sh; source benchmarks/phase7d/candidate_env.sh
    export CUDA_VISIBLE_DEVICES=            # focused tests need no GPU
    python -m pytest tests/newton_cpu -q
    ruff check src/maple_syrup tests/newton_cpu benchmarks/newton_cpu benchmarks/rfid benchmarks/hydraulic_candidates
    python benchmarks/newton_cpu/compare_cases.py --list-contenders

## Suggested benchmark commands (Codex runs these; frozen source)

    # RFID, everything incl. the originals (NumPy bisection is slow: run it apart with --contenders bisection_numpy)
    python benchmarks/newton_cpu/compare_cases.py --case rfid --case-dir outputs/rfid/case --output-dir <NEW> \
        --contenders bisection_numba,newton_numba,newton_numpy,fortran_newton,fortran_bisection --rounds 3 \
        --fortran-build-dir <NEW2> --allow-maple-source-change
    python benchmarks/newton_cpu/compare_cases.py --case rfid --case-dir outputs/rfid/case --output-dir <NEW> \
        --contenders bisection_numpy --rounds 3 --allow-maple-source-change
    # Plot 1
    python benchmarks/newton_cpu/compare_cases.py --case plot1 --case-dir outputs/plot1 --output-dir <NEW> \
        --contenders bisection_numba,newton_numba,newton_numpy --rounds 3 --allow-maple-source-change
    python benchmarks/newton_cpu/compare_cases.py --case plot1 --case-dir outputs/plot1 --output-dir <NEW> \
        --contenders bisection_numpy --rounds 3 --allow-maple-source-change
