# Phase 7h — prepared, fused CPU hydrology (design)

Status: Claude-authored implementation independently reviewed and executed by Codex.
The scoped suite passed 351 tests (six device-dependent skips), and six comparisons
against original executed Fortran routing passed. See [performance and qualification](performance.md)
for full-storm comparisons, repeated CPU timings and remaining GPU work.

Baseline: `f40ee38550b3dddca88c949d26eab0735e25b68b` (clean), immutable snapshot
`outputs/phase7h_hydrology/baseline_snapshot`. Unchanged and still the oracle: `storm.py`, `infiltration.py`,
`routing.py`, `routing_numba.py`, the NumPy/CuPy reference paths and `routing_numba._sweep`.

## What changed

| File | Change |
|---|---|
| `src/maple_syrup/hydrology_numba.py` | NEW. Context, two Numba kernels, prepared coupled/column steps, provenance. |
| `src/maple_syrup/legacy_experiment.py` | `--hydrology-implementation {prepared,reference}`, `resolve_hydrology_implementation`, `require_compiled(..., hydrology_implementation)`, run-loop dispatch, summary/provenance fields. |
| `tests/phase7h/` | NEW: `test_hydrology_prepared.py` (differential/adversarial), `test_hydrology_selector.py` (selector, refusals, missing Numba, short CLI comparison), `conftest.py`. |
| `tests/phase7f/test_legacy_default.py` | Minimal: the `array` case also passes `--hydrology-implementation reference` (stays bitwise); the `default` case compares the two water arrays at the declared 2e-12/1e-14 instead of exactly. |

No shared MAPLE, sediment, wet-physics or bin code was touched.

## Interface

```python
ctx   = prepare_hydrology(graph, params)                    # once per fixed terrain/soil; timed, recorded
step  = prepared_coupled_step(ctx, rain, state, dt, control)  # == storm.coupled_step(graph, params, rain, state, dt, control)
col   = prepared_column_step(ctx, depth, soil, rain, dt)      # == infiltration.column_step(..., validate=True)
kernel_provenance()                                          # versions, options, module hashes, compiled?
```

`HydrologyContext` owns **contiguous read-only copies** (flat, cell-major) of: `active`, `outlet`, `conveyance`,
`level_order`, `conveyance_lo`, `donor_position`/`donor_mask` (4, n_active), `level_bounds`, and one packed
`column_static` `(n_cells, 6)` = ksat, suction, drainage, thickness, Smax, lambda. Nothing aliases the caller's
arrays; later mutation of the source graph/parameters cannot change a context, and a new graph (rerouting) or new
parameters (soil, model) needs a **new** context. Dynamic inputs (depth, soil water, previous discharge, rain
rate) are read from the caller's arrays on every step and are never written, retained or converted. A read-only or
non-C-contiguous caller array is copied once per call (normalising the Numba type keeps one compiled
specialisation); the kernels never write inputs.

### Preparation checks (so the unchecked kernels are memory- and dependency-safe)

Types/dtypes/shapes; both namespaces host NumPy (CuPy → refused, no transfer); graph/column active masks equal;
finite non-negative conveyance, Ksat, suction, drainage; 0 < theta_sat ≤ 1; thickness > 0; `Smax == theta_sat *
thickness` exactly and > 0; lambda > 0 for the Hawkins model; `level_order` a permutation of the active cells
consistent with `level` and `level_bounds`; every donor index in `[0, n_active)` and, where used, in an EARLIER
level, a genuine donor of that cell (`receiver` consistency), unique and complete; outlets exactly the exporting
active cells.

## Kernels and phases (CPU now; the mapping to a device later)

| Phase | Where | Parallel structure | Notes |
|---|---|---|---|
| A. column + branch | `column_kernel` | one thread per cell, no cross-cell dependency | input checks, intake/drainage/return, output checks, branch masks (complete > no run-on > partial), `hpre`, `q_old`, branch counts, previous-discharge checks |
| B1. route inputs | `route_kernel` | per cell | routing input checks, implied-depth consistency, old-flux Courant |
| B2. level sweep | `route_kernel` → existing `routing_numba._sweep` (called through its own dispatcher) | levels in series, cells of a level independent (donors are always at earlier levels) | coherent donor sum `0+a+b+c+d`, base RHS, the unchanged `[0, R]` bisection (fixed iteration count, same multiply order) |
| B3. outputs | `route_kernel` | per cell | scatter, velocity, face volume, balances, maxima, operand rows of the four sums |
| C. host | Python | 4 `numpy.sum` + scalar checks + error resolution | on a device: a small reduction and ONE flag read per step |

Layout qualification: `column_static` is an array-of-structures `(n_cells, 6)` table, chosen because it is CPU
friendly (one cell's six values share a cache line). It is NOT claimed to be GPU efficient: a later device
implementation should explicitly pack the static data once into structure-of-arrays `(6, n_cells)` so consecutive
threads read consecutive addresses (coalesced warp access). The CPU implementation establishes nothing about GPU
efficiency.

Device boundary for a later implementation: the other context arrays and the state/rain arrays are flat FP64/int64/
bool; a device step would take them as device pointers, return one 64-bit flag word
plus the four reduction operand rows' sums, and never transfer a grid to the host (no hidden current-value
transfer). No GPU kernel exists in this change and none is claimed.

## What is preserved

Equations, expression and evaluation order of `column_step`, `coupled_step` and `_route`; branch precedence;
bisection method, bracket, iteration count and multiply sequence; donor sum order; original timestep; every
tolerance (`LOCAL_BALANCE_RTOL`, `BALANCE_RTOL`, root tolerance, Courant); all diagnostic fields; failure atomicity
(the step is pure; every output is freshly allocated per call; the context holds no mutable scratch, so one context
may serve several threads). `fastmath=False`, no `prange`, no `parallel`, `error_model="numpy"` (division by zero →
inf/nan, then refused, never a Python exception).

Error categories and precedence follow the reference: column failures (`InfiltrationError`, the lowest recorded
check first) → scalar-option failures (`RoutingError`, the reference's own `_check_step_options`) → routing failures
in the reference's recorded order (`RoutingStepRejected` only for the Courant and negative-RHS rejections) →
(prepared-only) previous-discharge strictness. `dt = 0` ends in the reference's `RoutingError` after the column
inputs have been validated. Messages are the reference's own strings (the tests compare them).

Return values are the reference's own dataclasses with the same field types: Python float `dt_s`/`t_s`,
`numpy.float64` scalars, `numpy.int64` branch counts, `route.implementation == "numba"`.

### Scalar reductions

`storage_change`, `export`, `balance tolerance` and `outlet discharge` are `numpy.sum` over rows the kernel fills
(`h_new - h_start`; face where outlet; scale where active; q where outlet), so the pairwise summation order is
NumPy's, exactly as in the reference — no re-implemented reduction. The max-reductions are order independent.
The four sums and the scalar checks run under `numpy.errstate(all="ignore")` as the reference's did.

## Deliberate differences (all stricter, none for states the reference's own drivers produce)

1. The whole previous-discharge array must be finite, ≥ 0 and zero on inactive cells (the reference reads it only
   through the no-run-on branch, so a skipped branch could launder a NaN). `RoutingError`, reported last.
2. Dynamic arrays must be exact host `numpy.ndarray`; CuPy arrays, masked arrays and other subclasses are refused
   (a mask would hide a non-finite value from a flag check).
3. `Smax` must be positive, finite and exactly `theta_sat * thickness`; static graph arrays are validated (above)
   instead of trusted.
4. Only `StormControl.implementation == "numba"` is accepted (the prepared path *is* the numba sweep); other
   values are refused after the reference's option validation.
5. Structural errors on the previous-discharge array are `RoutingError` raised before the kernels run; the
   reference has no defined behaviour there.

Hand-built pathological statics the reference would accept (e.g. an inconsistent donor table) are refused at
preparation; the reference's behaviour on them is not claimed identical.

## Floating-point expectations (to be measured, not asserted)

`+ − × ÷`, `sqrt`, comparisons are IEEE-exact in both implementations, in the same order, so every field that does
not pass through `expm1`/`pow` should be **bitwise** equal. `expm1` (Smith–Parlange capacity and the Hawkins
final-infiltration rate) and `pow` (only the implied-depth threshold test) come from LLVM/libm here and NumPy's
loops (possibly SVML on AVX-512 hosts) in the reference and may differ by an ulp; the declared bound for water is
rtol 2e-12 / atol 1e-14, which Codex should report against measured differences — **not** relaxed if exceeded. The
saturated-column test is designed so the column arithmetic is libm-independent (`expm1(-x)` is exactly −1 for
x ≫ 40) and asserts bitwise equality of every field.

## Intended checks (not run by the author)

```
pytest tests/phase7h tests/phase7f tests/phase4 -q          # new + selector-affected + reference suites
pytest tests/phase7h/test_hydrology_prepared.py -q -k "bitwise or boundary or limiting"
```

then the old-default CLI comparison with the immutable `b350d36` script (existing 7f test, now also covering the
prepared default), a full 5400-step prepared vs reference vs baseline-snapshot run (water bound 2e-12/1e-14,
physics-dependent arrays 2e-11/1e-14, regime counts and peak timing exact), and warm/cold timing split
(`hydrology_preparation_s`, first step including JIT, steady state) from `legacy_summary.json`.

## Limitations and uncertain branches

* The nested dispatcher call (the compiled `_sweep` called from inside `route_kernel`) was reported by Codex to
  compile; execution evidence is Codex's, not the author's.
* **Known reference roundoff inconsistency (follow-up, unchanged here).** In the reference, `hpre = h - max(J - P, 0)`
  and the column depth `h* = (h + P - J) + O` can differ by one ulp when a wet cell infiltrates all its available
  water (J = A, O = 0): `hpre` may then exceed `h*` and `route_step` refuses with "old_flow_depth_m ... exceeds
  depth_start_m". Neither the reference nor the prepared path changes equations or guards for this optimisation;
  both refuse identically (`test_reference_roundoff_inconsistency_is_refused_identically_by_both_paths`). The
  domain is broader than J = A: with positive rain and partial intake (P < J < A), `h - (J - P)` and
  `(h + P) - J` are different FP64 expressions and can differ by an ulp at ordinary depths too
  (`test_partial_positive_rain_roundoff_refusal_is_identical_in_both_paths`, seeded search). Only P = 0 (identical
  expressions) and J < P (no run-on, `hpre = h`, `h*` larger by P - J) are safe. The ordinary differential tests
  therefore force positive rain >= 6e-5 m/s on ~80% of cells so capacity (Hawkins K <= lambda <= 1.2e-5, Ksat <= 3e-6,
  times <= ~2.6 for the Smith-Parlange factor with suction <= 1e-3 m and h <= 1e-2 m) stays below P, with exactly
  zero-rain pockets and a dry recession for the partial branch; explicit complete/no run-on/partial tie cases use
  binary-safe operands. The bound is a derivation from the parameter ranges, not an executed result. A fix belongs to
  a separate scoped change of the reference.
* Cold compilation now covers two kernels plus the sweep on the first step (no cache: `cache=False`, kernels are
  closures); the first-call time recorded by the driver includes it, as before for the wet-law kernel.
* `RoutingStepRejected` for a negative right-hand side cannot be triggered through the Courant-guarded coupled path;
  that branch is implemented but not exercised by a test.
* The prepared path is not used by `storm.evolve` (retry/halving driver), only by the legacy replay; adopting it
  there is a separate task. `legacy_stale_inflow_step` and the standalone `route_step` have no prepared version.
* A read-only or non-C-contiguous input costs one small copy per call.
* Concurrent use of one context is intended to be safe (no shared mutable scratch) but is not tested.
* GPU: nothing implemented; phases above are the intended decomposition only.
