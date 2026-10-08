# CUDA safeguarded Newton root solver (water only, fixed terrain)

Status: implemented by Claude (sole writer, task `gpu_newton`, baseline `d5861b4`), independently validated and timed by
Codex, **uncommitted**. See [results and qualifications](results.md): 3168 independent checks passed plus three CLI
follow-up checks; complete Plot1/RFID water storms were verified in both GPU modes. Claude's final evidence review found
no acceptance blockers; Codex accepted this bounded water-only task.

## What it is

The CPU safeguarded Newton of [`routing_newton.py`](../newton_cpu/README.md), ported operation for operation to FP64 device
code and selectable on the CUDA ordered sweep and on the prepared resident CUDA hydrology. It solves the SAME corrected
cell equation

    h + c k h^{3/2} = R,    c = dt / (2 dx),   R >= 0, k >= 0

(`R = h_start + c (Qin_old + Qin_new - q_old)`), then `q = k h^{3/2}` and the storage identity `h_new = R - c q`, with the
unchanged D4 donor order, coherent old flux, per-cell/global water balances, constitutive check `|h_new - h_flow| <=
root_tolerance_m` (1e-11 m), Courant and negative-RHS rejections and finite/negative checks. It is a root finder, not new
physics. Per cell (see `routing_newton.py` for the derivation): `R > 0` false -> `h_flow = 0`; `trial(R) <= R` (k = 0 pits,
lost flux term, tiny/subnormal R) -> `h_flow = R`; otherwise Newton from the proven upper bound `x0 = R/(1 + a sqrt(R/(1 +
a sqrt R)))` (`a = c k`) with a tightening bracket and a bisection safeguard step, cap `newton_max_iterations` (default 50,
1..1000), polishing to keep `trial(h_flow) < R` (so `h_new >= h_flow >= 0` without clipping), and a bounded (<= 1200
halvings) bisection completion of the bracket if the cap is hit. That completion is inside Newton by design; it is not a
bisection-solver substitution, and nothing is clipped or deleted.

## Use

    route_step(graph, h, h_old, dt, implementation="cuda", root_solver="newton", newton_max_iterations=50)
    StormControl(implementation="cuda", root_solver="newton", newton_max_iterations=50)   # coupled_step / evolve / prepared step
    python -m maple_syrup.storm_experiment ... --implementation cuda --backend cupy --root-solver newton [--newton-max-iterations N]

Bisection remains the default everywhere. Selecting Newton never changes the bisection kernels: their sources are
byte-identical to the baseline (SHA-256 pinned in `tests/gpu_newton/test_sweep.py`), and a default control launches only the
default module.

Modules: `maple_syrup.routing_newton_cuda` (the shared device text `newton_device_source()`, stand-alone sweep
`run_sweep(graph, base_lo, c, cap, mode, stats=False)`, per-root probe `solve_roots(rhs, k, c, cap)`, `ensure_loaded`,
`kernel_provenance`); `hydrology_cuda` (Newton variant of the five step kernels in a separate module,
`load_newton_kernels`, `kernel_source("newton")`). One device text is used by the stand-alone kernels and the coupled
kernels; the CPU constants are substituted at import so they cannot drift.

## Contracts

* Arithmetic: only `__dadd_rn/__dsub_rn/__dmul_rn/__ddiv_rn/__dsqrt_rn`, `--fmad=false --ftz=false`, no fast math. On
  identical inputs the root, pass count, bisection-step count and fallback flag were bitwise equal to the CPU pure-Python
  specification in every scalar and sweep test run by the author (log-uniform, subnormal, huge, NaN/Inf, k = 0, forced low
  caps). Whole-storm fields still carry device `expm1`/`pow` and reduction-order differences: the declared bounds
  rtol 2e-12 / atol 1e-14 (and MAPLE-derived water budget bounds) are unchanged and not loosened.
* Launch structures: routing `level` / `block` / `auto`; hydrology `fused` / `split` / `auto`; barriers, donor order and
  stream/context/device freshness contracts are those of the bisection path. No CPU fallback, no host root solve, no silent
  solver substitution. High-level validated entries (`route_step`, `StormControl.validated()`, `evolve`, the CLI) refuse an
  invalid solver name/cap/backend pairing before any work or mutation. The LOW-LEVEL prepared CUDA step
  (`prepared_coupled_step`, which accepts an unvalidated `StormControl`) keeps the existing column-first precedence: option
  errors are deferred, only the column-only stage runs, its one packet is read, and a column failure outranks an invalid
  Newton option, which is then raised as `RoutingError` with no result and no input mutation. Missing CuPy, device,
  compile or launch failure remain explicit (`CudaUnavailableError` / `RoutingError`).
* Transfers: the 144-byte packet, one counted download per attempted step, resident accumulation/report kernels and the one
  final synchronization are unchanged (a Newton storm and a bisection storm of the same steps have identical counted
  transfer totals in the author's test). Newton adds no per-step scalar read and no ADDITIONAL per-step diagnostic state, history or arrays compared with bisection; the
  existing fresh per-step output arrays are allocated exactly as before. A code-module footprint (second NVRTC module) exists.
* Diagnostics: iteration counters exist only in untimed probes (`run_sweep(..., stats=True)` returns a device `uint64[5]` =
  `routing_newton.STAT_NAMES`; `solve_roots` returns per-root passes/steps/fallback). Production steps (`route_step`,
  coupled hydrology) are uninstrumented: `RouteStep.root_stats` is `None` (NOT zeros). Use `newton_numba` statistics (same
  equation, same case) for pass counts of a case; the CUDA counters of a given state can be reproduced with the sweep probe.
* Metadata: `RouteStep.root_solver`, `.newton_max_iterations` (0 for bisection), `.bisection_iterations` (0 for Newton),
  Newton-specific failure text (`Newton root solver did not reach root_tolerance_m ... (newton_max_iterations = N)`); the CLI
  summary adds `routing.root_solver`/`newton_max_iterations` and `cuda_hydrology.preparation.newton_kernel_load_s` ONLY for
  Newton, and `kernel_provenance()` records the Newton source hashes, options and device.
* Startup: `evolve` preloads the Newton variant before its first step, so a caller that times an entire cold `evolve` call
  includes that preload. The benchmark harnesses preload explicitly outside the measured sample, and the CLI records
  preparation (`newton_kernel_load_s`) separately. A direct cold `prepared_coupled_step` (or `route_step` on a cold process)
  compiles on first use. Compilation therefore is not guaranteed to fall outside an external caller's timer.
* Memory/controls: no extra device state; the Newton module is code only (a second NVRTC module of the five step kernels).
  `newton_max_iterations` is the only added control; a low cap exercises the bracketed fallback and stays exact and safe
  (tests with caps 1..3).

## Limits (unchanged scope)

Water only, frozen terrain/routing. GPU sediment, evolving terrain, splash, vegetation and disk restart are NOT supported:
`SedimentEventControl` still refuses any CUDA storm (tested), and `checkpoint.py` is untouched, so no disk GPU storm
restart exists. Supported continuation is in memory (`evolve(..., cuda_context=ctx)` leg by leg); the author's test checks a
split Newton storm against the uninterrupted one. Control metadata round-trips as a plain `StormControl` dataclass only.
The existing Plot 1 native-Fortran adapter does not support its model-2/pavement case; the RFID native Fortran reference remains optional and
qualified as in [../newton_cpu](../newton_cpu/README.md). Old CPU timings in `../newton_cpu/results.md` are historical.

## Benchmark entry

    python benchmarks/gpu_newton/compare_cases.py --case rfid  --case-dir outputs/rfid/case  --output-dir <NEW> [--cuda-mode fused|split|auto]
    python benchmarks/gpu_newton/compare_cases.py --case plot1 --case-dir outputs/plot1       --output-dir <NEW>

A thin wrapper of `benchmarks/newton_cpu/compare_cases.py` (contenders `bisection_numba,newton_numba,bisection_cuda,newton_cuda`
by default; CPU defaults of the original harness unchanged). Actual verified MAPLE Plot 1 (1200 cells, model 2, pavement,
5400 s, 40 bisections) and RFID (5697 active / 6032 array cells, model 1, 2700 s, 64 bisections), frozen terrain/routing,
max dt 1 s, report 60 s, full warm-up, three balanced rounds, first call / preparation / Newton kernel load recorded apart,
final fields and the full outlet hydrograph captured outside every timer, water/bed/source guards on every sample, explicit
launch mode forcing. `benchmarks/rfid/run_rfid_timing.py` and `benchmarks/hydraulic_candidates/compare_plot1.py` forward
`--root-solver newton` to `legacy_cuda` as well. The author ran only a 900 s RFID smoke of the wrapper (not a timing).

## Author's checks

See `agent_handoffs/tasks/gpu_newton/implementation_report.md` for exact commands and outcomes.
