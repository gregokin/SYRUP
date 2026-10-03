# Experimental hydraulic alternatives — explicit kinematic wave and local inertia

Status: **experimental, water-only, implemented by Claude with file tools only; nothing here was run by its author.** Codex
records test, parity and benchmark results elsewhere. This page is usage and contract documentation, not evidence. The accepted
MAHLERAN-inspired method-5 hydrology (`routing`, `hydrology_numba`, `hydrology_cuda`, `storm`) and every CLI default are
unchanged; these are separate, selectable solvers. Neither is claimed equivalent to the legacy hydraulics, neither touches the
MAPLE bed, and there is no sediment, wind, splash, evapotranspiration, dry reset, plant growth or evolving terrain.

## The two candidates

Both split a step into (1) the **accepted column physics** (`infiltration.column_step` on the CPU,
`hydrology_cuda.prepared_column_step` on the GPU — no infiltration formula is repeated) giving the post-rain/infiltration
storage `h_c`, and (2) a lateral redistribution of that water. Units: m, m²/s, m³, s, FP64.

**Explicit kinematic wave** (`--solver explicit`; the fixed legacy D4 graph and conveyance `k = sqrt(8 g S / f)`):
`q = k h_c^(3/2)`, face volume `F = dt dx q` (used once: donor loss, receiver gain, donors gathered in the fixed `DONOR_SLOTS`
order), `h_new = h_c + (ΣF_in − F_out)/dx²`. First-order Euler, no ordered sweep, no bisection. The characteristic speed is
`dq/dh = 1.5 k sqrt(h)`; a step is admissible iff `1.5 k sqrt(h_c) dt/dx ≤ cfl_max ≤ 0.5`, which proves `h_new ≥ (2/3) h_c ≥ 0`.
A violating step is **rejected and the driver halves dt from the unchanged state** (no clipping).

**Uniform-grid local inertia** (`--solver local_inertial`; staggered signed face fluxes `qx (ny, nx+1)` east-positive,
`qy (ny+1, nx)` north-positive, both retained for continuation): advective acceleration dropped, local acceleration and
the water-surface gradient kept, Darcy–Weisbach friction (not Manning):
`q' = (q − g h_f dt Δη/dx) / (1 + dt (f/8)|q|/h_f²)`, `h_f = max(max(η_A, η_B) − max(z_A, z_B), 0)`; steady uniform flow gives
`v² = 8 g h S / f` (the legacy law). `h' = h_c − (dt/dx)[(qx_E − qx_W) + (qy_N − qy_S)]` from the SAME face arrays. Gravity-wave
CFL `dt sqrt(2 g h_f,max)/dx ≤ cfl_max`. **Boundary:** only the legacy outlet faces are open, with a declared Darcy normal-flow
outflow `q = k_b h^(3/2)`, `k_b = sqrt(8 g S_b/f)`, `S_b` = DEM bed drop to the ring cell; every other ring face and every face
touching an inactive cell is closed; an outlet draining into an interior inactive cell is refused. A negative depth after the
update **rejects the step** (default). `--limiter donor` enables the documented conservative donor limiter (each cell's outflow
limited to its available water by one factor applied to all its outgoing faces; the limited flux becomes the new momentum;
activity is counted and reported).
Documented departures: scalar directional friction using only a face's own normal flux; face friction = mean of the two cells.

Equation sources (read by the authors of the specification, not re-fetched here): Kirstetter et al., arXiv:1609.04711 (Darcy
friction slope `f q|q|/(8 g h³)`); the Deltares Wflow local-inertial documentation (staggered continuity, CFL pattern; Manning,
adapted). The semi-implicit update and the boundary are this project's experimental choices, not an exact implementation of either.

## Use

```
python -m maple_syrup.experimental_experiment --case-dir outputs/plot1 --output-dir outputs/exp_explicit_cpu \
    --solver explicit --backend numpy --max-dt-s 1 --end-s 5400 --allow-maple-source-change
python -m maple_syrup.experimental_experiment --case-dir outputs/plot1 --output-dir outputs/exp_li_gpu \
    --solver local_inertial --backend cupy --max-dt-s 1 --snapshot-times-s 600,1200,1620 --allow-maple-source-change
```

`--backend numpy` is the pure NumPy reference; `--backend cupy` the CUDA form (CuPy + device required, never Numba, no fallback;
without a device the run is refused with an actionable message). Outputs go to a NEW directory only, after the MAPLE-derived
water budget (`volume_roundoff_bound_m3`), the bed digest and the source-stability checks pass: `experiment_summary.json`,
`final_state.npz`, `hydrograph.npz/.csv`, optional `snapshots.npz`. The summary records solver, boundary, CFL/limiter options,
actual context/kernel metadata (GPU), preparation / loop / final-report transfer counters separately, timings, provenance.

```python
from maple_syrup.experimental_hydrology import CpuHydraulicSolver, HydraulicControl, build_local_inertial_geometry, open_faces_from_graph
from maple_syrup.experimental_cuda import CudaHydraulicSolver          # lazy CuPy
from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental

geometry = build_local_inertial_geometry(z_full, active, friction, dx, open_faces_from_graph(graph))   # local inertia only
solver = CudaHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=HydraulicControl(limiter="off"))
state0 = solver.initial_state(depth0, soil0)
result = evolve_experimental(solver, field, schedule, state0, 5400.0, ExperimentalControl(max_dt_s=1.0),
                             report_every_s=60.0, snapshot_times_s=(600.0, 1200.0))
```

One shared driver serves both methods and backends: the boundaries are `storm.plan_boundaries` plus requested snapshot times (extra
boundaries; the forcing is never averaged across an edge); a rejected step halves dt from the same state (bounded by
`max_retries`, the retry floor `min_dt_s` and `max_steps`); the dt cap doubles after every clean full-size step. Peaks are maxima over
accepted steps of the end-of-step values; there is no pickup/detachment field.

## Compiled CPU (Numba) forms — tested experimental tools

Executed checks, limitations and current CPU/GPU timings: [Numba results](numba_results.md).

`maple_syrup.experimental_numba.NumbaHydraulicSolver("explicit" | "local_inertial", graph, params, geometry=..., control=...)`
is the compiled CPU form of both candidates: the SAME equations, expression order, CFL / adaptive-retry / continuation contracts,
`HydraulicStep`, error classes and messages, donor limiter and diagnostics, as serial Numba kernels (no NumPy temporary chains in
the lateral hot loops). It subclasses the NumPy reference, so `step`, `initial_state`, `validate_state` and every public check are
inherited unchanged; only the column hook and the two lateral steps are replaced. The NumPy `CpuHydraulicSolver` stays the
independent oracle and the default; the CUDA form and the legacy default are untouched.

```
python -m maple_syrup.experimental_experiment --case-dir outputs/plot1 --output-dir NEW_OUT --solver local_inertial \
    --backend numpy --implementation numba --max-dt-s 1 --end-s 5400 --allow-maple-source-change
```

```python
from maple_syrup.experimental_numba import NumbaHydraulicSolver
solver = NumbaHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=HydraulicControl(limiter="off"))
result = evolve_experimental(solver, field, schedule, solver.initial_state(depth0, soil0), 5400.0, ExperimentalControl(), report_every_s=60.0)
```

- **Selection.** `--implementation numpy|numba|cuda` is optional; omitted, `--backend numpy` is the NumPy reference and
  `--backend cupy` the CUDA form (unchanged). Valid pairs: numpy+numpy, numpy+numba, cupy+cuda (`--implementation cuda` alone implies
  cupy; the Python API `run_plot1_experiment(backend=None, implementation=None)` resolves identically). Any other pair, a missing Numba (no fallback to the reference) or a missing CuPy/device is refused BEFORE the case is read.
  The summary records the requested and the resolved pair and, for the compiled form, the actual Numba/llvmlite versions, kernel
  options (no fastmath, no prange, `error_model="numpy"`), the module hash and the context/column-context metadata.
- **Column stage.** Not duplicated: the accepted `hydrology_numba.prepared_column_step` on a `HydrologyContext` prepared once in the
  constructor. Its wording for a wrong TYPE/dtype/shape of depth, soil or rain differs from the NumPy column's (both are
  `InfiltrationError`); the conditions and the precedence are the same.
- **Arithmetic.** Only `+ - * /`, `sqrt` and comparisons, no fastmath, no FMA contraction, no clipping. The bitwise statement is
  NARROW: the lateral stage is bitwise equal to the NumPy reference for IDENTICAL column outputs and state (root trial1: all 7440
  accepted local steps of the 5400 s storm fed the NumPy column and state match every public field bitwise; also in the tests where
  the column is the identity). It is not a whole-run claim: the compiled column (LLVM/libm `expm1`) may differ from NumPy's by an
  ulp (root: a complete Numba step on the same NumPy input state stays within the bounds on every step, column depth differences
  <= 4.3e-19 m, soil <= 1.4e-17 m), and independent evolving trajectories can amplify that. The declared bound (rtol 2e-12 /
  atol 1e-14) is unchanged and applies to everything. The scalar sums are `numpy.sum` on kernel-produced operand rows.
- **Measured exceptions (root, not by the author; no bound widened).** Explicit 5400 s storm: every public step field and every
  driver field passes. Local-inertial 5400 s storm: water closes and the accepted/rejected counts match the NumPy reference
  (7440 / 2040), but independent trajectories exceed the bounds in some fields (final soil water, cumulative intake, hydrograph,
  peak velocity ~6e-4 m/s, and the 600/1200/1620 s snapshot velocity maps). 90 s Plot 1 CLI, local inertia: only
  `final_state.npz:velocity_m_s` fails (3 cells, max abs 2.4673e-14 m/s, worst normalized error 2.086); every other saved field,
  count, metadata and budget passes. This is case-specific evidence consistent with rounding amplification (column rounding and
  the ill-conditioned near-dry local-inertial dynamics), NOT a universal cause and not an all-backend identity claim. The long
  storm limitations therefore apply to the compiled Numba form against the NumPy reference as well as to CUDA. A failing velocity
  is a declared experimental diagnostic, not an acceptable erosion velocity. Nothing here fixes the local-inertial conditioning,
  velocity or Froude limitations above.
- **Arrays.** Exact host float64 `numpy.ndarray` only (subclasses, masked arrays, CuPy, other dtypes refused, never converted). A
  strided/Fortran-ordered or read-only state array is accepted and COPIED once per step to a C-contiguous writable flat array (never
  written, never retained), so results do not depend on layout; outputs are fresh C-contiguous arrays.
- **Static data and guards.** The solver owns contiguous read-only flat copies of everything its kernels read (explicit: active,
  outlet, k, donors; local inertia: bed and the face tables) plus the column context, sealed with pointer/shape/dtype fingerprints.
  It is immutable after construction and re-checks the seal (method, shape, dx, CFL, limiter, both contexts, every array's
  fingerprint and read-only flag, the arrays the column kernel indexes) before EVERY compiled call: a forged context/attribute is
  refused with `ExperimentalHydrologyError` and no kernel runs. No static array is hashed per step. Contexts are immutable by
  contract (in-place content mutation of an owned array whose write flag was re-enabled is not detected beyond the flag); a new
  graph, parameters, geometry or control needs a NEW solver. Each step allocates its own bounded outputs and operand rows; nothing is
  retained, so a solver may be shared by threads.
- **Compilation.** Lazy: importing the module imports neither Numba nor CuPy; the first step JIT-compiles the lateral kernel (and the
  column kernels). Separate compilation from steady-state work when timing.
- **Harness.** `benchmarks/hydraulic_candidates/compare_plot1.py` accepts `explicit_numba` and `local_inertial_numba` by name (not in
  the default list); fresh contexts, per-sample guards and validation as for every other contender.
- **Status.** New and unqualified beyond `tests/hydraulic_candidates/test_numba_*.py` (differential against the NumPy reference on
  both column laws, off/donor limiter, inactive cells, open and closed boundaries, pure rejections, error class and message,
  forged-metadata refusal, strided/read-only arrays, continuation, a short real Plot 1 run, the CLI and the harness). Full-storm
  NumPy/Numba/GPU comparisons and warm timing against the best prepared legacy Numba are the root's pending measurements; no speed
  claim is made.

## Contracts (CUDA)

Constant kernel count per step (explicit 3; local inertia 4, or 7 with the limiter) plus the baseline column; per attempted step
**two counted packet reads** (column flags, lateral packet) and no grid transfer; fresh outputs, pure failures; the baseline context
seal/binding is reused and the experimental context adds its own sealed scalars, extents and array fingerprints, all checked
before any launch (a forged context is refused before any enqueue); `is_bound_to` ties an external context to the very graph,
parameter and geometry objects; the context is immutable by contract (content mutation of owned arrays is not detected). Lateral
kernels use only `+ − × ÷ sqrt` in strict FP64 (`--fmad=false`, RN intrinsics, no fast math), so they agree with the NumPy
reference operation for operation; only the sums (fixed tree vs NumPy order) and the baseline column's `expm1` can differ.
The field bound for step comparisons is rtol 2e-12 / atol 1e-14, with exact counts and flags. Full-storm exceptions are documented below.

## Transfer scope (what the counters do and do not prove)

MAPLE's `read_transfer_counters` counts only its instrumented helpers (`to_host`, `to_device`). Raw host conversions bypass these helpers, so "0 counted H2D" alone is not a complete transfer trace.
In the [CuPy14.2 source](https://github.com/cupy/cupy/blob/v14.2.0/cupy/_core/core.pyx), Python-scalar conversions allocate a device array
and fill it with a kernel; host arrays can incur memory copies. The earlier claim that the per-step scalar conversions were
proven memory uploads was incorrect: the demonstrated overhead was allocation and fill launches. The driver therefore creates no device array from a
host value at all: initial device scalars come from `xp.zeros` / `xp.full`; the time of a new peak is passed to `xp.where` as a
kernel argument; the report-row time, counters and smallest dt are written by value with a one-element `fill`. The CUDA loop's
host reads are two counted packets per attempted step; static uploads (counted) happen in preparation; final reporting downloads
are counted apart by the caller. Device-to-device copies, fills and allocations are not transfers. The same statement is returned
by `solver.describe()["transfer_scope"]` and saved in the CLI summary (`backend.transfer_scope`). The internals of the baseline
column kernels were not independently audited here; `test_driver_transfers.py` traps `cupy.asarray` / `cupy.array` of any
non-device argument during whole storms to check the candidate path.

## State-time contract and immutable solver (corrections)

`HydraulicState.t_s` must be a real number (not bool/str/None), finite and >= 0 in `initial_state`, `step` and continuation, on
CPU and GPU alike (`check_state_time`). For dt > 0, `t + dt` must be finite and must advance floating time (`advance_time`). All of
this is checked before any column work or launch, and nothing is mutated on refusal. The CUDA solver is immutable after
construction: method/shape/dx/control/context are read-only properties over a sealed record, re-checked before every launch.

## Velocity interpretation and a known local-inertial limitation

`HydraulicStep.velocity_m_s` for local inertia is the face-averaged flux divided by the END-of-step cell depth. Next to a cell the
step empties (or nearly empties), that depth is arbitrarily small while the fluxes are finite, so the value is unbounded: the
measured Plot 1 diagnostics show Fr_max of about 10 / 1.4 / 4e4 at 600 / 1200 / 1620 s (trialC1_froude_diagnostics.json). Nothing
is clipped. This reconstructed speed is NOT qualified as a detachment or grain-travel velocity. `stage_face_diagnostics(step)`
returns the stage-consistent alternative: each face flux divided by the face depth it was computed from (`face_flow_depth_m`,
`face_velocity_m_s`, `face_froude`; m, m/s, dimensionless; stage = after rain/infiltration, before the lateral update). It
separates the cell reconstruction from the face flows but is NOT a fix and is not claimed bounded or qualified: the root's
synchronous GPU observations (trialC2_face_stage_v2.json) give maximum normal-component face Fr of 1.65 (x) / 2.01 (y) at 600 s,
0.34 / 1.82 at 1200 s and 0.16 / 0.21 at 1620 s, with up to ~0.4% of positive-depth faces above Fr 0.5 (median ~0.07-0.13). So
the end-depth denominator is only part of the issue: a small subset of actual face flows also leaves the low-Froude range in which
local inertia is accurate (De Almeida and Bates 2013, already cited in the research note; no new source was fetched). Advection,
directional friction damping, momentum smoothing, wetting/drying and the outlet treatment are not qualified. Nothing is clipped
or Froude-limited, and no friction/momentum/limiter default was changed to hide it.

## Qualification status

Verified by tests (short steps and controlled storms, CPU reference vs CUDA): field agreement at rtol 2e-12 / atol 1e-14, exact
counts/flags, exact water identities. NOT met: the full 5400 s local-inertial Plot 1 storm (trialC2 CLI CPU vs GPU). Water closes
(~1.2e-13 m3 residual against a bound of ~8e-7) and the bed is preserved; final cumulative export differs by ~5.4e-14 m3
(0.14797315895128788 vs 0.14797315895134197); depth, face momentum and velocity final fields match exactly. But under the UNCHANGED
bounds some saved fields fail: final soil water (2.3e-13 m, normalized 1.31), cumulative intake (2.3e-13 m), hydrograph export /
surface-storage / interval rows (up to ~4e-12 m3) and max-depth rows (up to ~2.4e-12 m), peak velocity (4.8e-4 m/s,
normalized 1.8e6) and three snapshot velocity maps. No tolerance was relaxed or widened. Disposition: long-storm backend
equivalence of local inertia at the baseline bounds is NOT qualified; this is a numerical-conditioning limitation of an
experimental tool, not a blanket backend-identity claim and not evidence of mass loss. The explicit method's full-storm result
is not part of this finding. Open qualification: sensitivity of the long local-inertial storm to rounding (see the correction-3
report for the proposed perturbation experiment), advection, directional damping, wetting/drying, outlet treatment. Both tools are
testable experiments, neither is adopted, and no production equivalence is claimed. The same statement is returned by
`solver.describe()["qualification"]`.

## Continuation contract (adaptive step cap)

The driver's adaptive step cap shapes the step sequence and therefore the physical state, so it is continuation state, not
bookkeeping. Every accepted `HydraulicState` returned by `evolve_experimental` carries `next_dt_cap_s` (also
`result.next_dt_cap_s`); passing that state back resumes from the cap, clamped into `[min_dt_s, max_dt_s]` of the new control. With
the same forcing, report grid, grid and control, a run to a boundary plus a resumed run equals one continuous run: bitwise state
and face momentum, and the accepted/rejected counts add up. The cap is not reset at boundaries (that would hide, not fix, the
history). `next_dt_cap_s=None` (e.g. `solver.initial_state`) is a fresh event starting at `max_dt_s`: fresh-event dynamics are
unchanged. The cap is validated (None, or a real, finite, positive number; bool/str/arrays refused) before any step, in the
driver, `validate_state` and `step`. Solvers never read it. No disk restart exists; the CLI saves `state_t_s` and
`state_next_dt_cap_s` in `final_state.npz` and reports them under `final_state.continuation` in the summary, because the arrays
alone are not the whole numeric state.

## Tests and tools

`tests/hydraulic_candidates/`: `test_explicit_cpu.py`, `test_local_inertial_cpu.py`, `test_driver_cpu.py`, `test_contracts.py`,
`test_time_and_diagnostics_cpu.py`, `test_driver_scalars_cpu.py`, `test_benchmark_harness.py`, `test_continuation_cpu.py` (CPU
only, no Numba), `test_numba_contracts.py` (CPU, no Numba needed), `test_numba_candidates.py`, `test_numba_cli.py` (CPU, need Numba),
`test_cuda_candidates.py`, `test_cuda_corrections.py`, `test_driver_transfers.py`, `test_continuation_gpu.py`
(device), `test_experimental_cli.py` (Plot 1
short runs, device for the GPU case).
`benchmarks/hydraulic_candidates/compare_plot1.py` compares legacy GPU, the BEST prepared-Numba legacy and each candidate on the
same Plot 1 inputs (dt 1/0.5/0.25, full storm with recession, 60 s reports, synchronized maps downloaded after the timer). Every
timed sample is validated after its own timer and before selection (water budget with the MAPLE bound, bed digest, source
digests); the output path is refused if it exists or lies in a protected tree and files are created exclusively; legacy snapshot
maps need segmented legacy runs, so a snapshot-free single-event legacy run is also timed and validated. Smoke form:
`--case-dir outputs/plot1 --dts 1 --end-s 10 --snapshot-times-s 5 --warmup-s 1 --report-every-s 5 --allow-maple-source-change`.

## Limits

Experimental; not an event completion; first-order schemes with CFL-limited steps (the gravity-wave bound can force dt below 1 s at
depths above a few millimetres, so the local-inertial run may need more steps than the legacy hydrology — unmeasured); the donor
limiter changes the dynamics when it acts; scalar directional friction; the Darcy normal-flow outlet differs from the interior
hydraulics and from the legacy edge rule; fixed terrain; no restart to disk; asynchronous device faults surface at the next packet
read; no `compute-sanitizer` result; no performance claim.
