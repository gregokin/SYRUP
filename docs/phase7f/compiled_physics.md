# Compiled wet physical laws for the frozen legacy replay (design and reproduction)

Subsequent hydrology optimization is documented in [Phase7h](../phase7h/performance.md).
Measurements below refer to the wet-law/default-workflow stage before that change.

Status: Claude-authored implementation independently reviewed and executed by Codex.
The scoped regression suite passed (138 tests, one device-dependent skip); three
direct original-Fortran equation checks also passed. Measurements and full-storm
comparisons are recorded in [compiled performance](compiled_performance.md).

## What changed

- `src/maple_syrup/legacy_physics_numba.py` (new, isolated): `prepare_legacy_physics(...)` builds a
  `LegacyPhysicsContext`; `legacy_physics_step(context, depth, velocity, rain, previous_velocity, dt)` runs one
  fused serial Numba kernel and returns the same `SedimentPhysicsStep` fields as the reference.
- `src/maple_syrup/legacy_experiment.py`: `--physics-implementation {numba,array}`. Default is the value of
  `--implementation` (so the default run is compiled water, compiled legacy transport and compiled wet laws).
  `array` runs the unchanged `sediment_physics_step` and may be combined with compiled hydrology/transport for a
  controlled comparison. Compiled physics without importable Numba is refused (exit 2, explicit message); there is
  no silent fallback. The summary JSON gains `physics_implementation`, `performance.physics_preparation_s` and
  `performance.physics_context`; ledger NPZ contents and the physical model are unchanged.
- Untouched: `sediment_physics.py` (the NumPy/CuPy reference and oracle), `legacy_transport.py`, `storm.py`,
  characteristic/bin/conservative code, MAPLE, MAHLERAN.

## Scope and validity

The context is valid **only for a frozen composition**: the legacy replay holds the initial active-layer holdings
fixed (`holdings0`), so class fractions, median diameter, cap rates and everything derived from them are constants.
An evolving MAPLE bed must not be prepared into a context; use the reference function or build a new context after
each change. Nothing in the context aliases a caller array, and no evolving holdings are cached. The CLI test compares the default
compiled run with the NumPy-physics run (bitwise, `--physics-implementation array`) and with the original script
(`rtol=2e-11`, `atol=1e-14` on physics-dependent float arrays; water, times and non-floats exact).

## Preparation (once, timed separately)

Validated once: parameters (re-checked, because the dataclass can be built by hand), grid slope/active/area,
vegetation cover (finite, within [0, 1]), holdings (finite, non-negative, per-cell total finite). The context stores
owned, read-only, flat C-contiguous arrays (cell index `i = iy*nx + ix`, class-minor) that are copies of the grid
slope/active mask or DERIVED from the inputs; the holdings and vegetation arrays themselves are not retained.
`context.nbytes()` sums every stored array (including the small config vector); it is static-array accounting,
not a measurement of process memory or a claim of reduced memory. Precomputed with the reference's own
NumPy expressions and operand order: class fractions, median diameter `d50` (the reference's `median_diameter_m`),
cap rates `f*hmax/ref`, vegetation factor, `(100 S)^c` per class, `4.554e-3 (excess d50)^1.5`, and grain-only constants
(`a/1200`, `sigma g D`, `D^-0.94`, `exp(-6.12683698 D 1000)`, particle mass and `mass^-0.425`, suspension criterion).
Non-finite static terms are refused at preparation (the reference would refuse every step). `preparation_s` is the
wall time of this step; Numba compilation happens on the first `legacy_physics_step` and is not included.

## Kernel

One loop over cells, inner loop over classes. Same FP64 equations, units, thresholds (`Re >= 2500`, `Re > 500`,
`D* <= 10`, suspension `ustar >= criterion`), clamps, caps (phi 2 raindrop cap, flow cap, 30 m, exponent 100),
legacy-literal versus selected conventions, and legacy-driver memory decay (`recession_velocity(1.0, dt, ...)` is
evaluated in Python so the scalar factor is the reference's). Multiplication order follows the reference term by
term; class-independent powers (`kf^2.35`, `S^0.981`, `xs^1.31`, ...) are hoisted per cell, which does not change
any value. `fastmath=False`, no `prange`, `error_model="numpy"` (division by zero gives inf/nan, never an exception;
non-finite values are then refused). `np.minimum`/`np.maximum` NaN propagation is reproduced explicitly so NaN is never
silently clamped. The kernel evaluates every intermediate the reference checks for every cell and class (including
inactive and masked ones) and sets a failure bit; the wrapper raises `SedimentPhysicsError` after the loop. Outputs are
freshly allocated every call, so a failure returns nothing and an earlier result is never changed by a later call.
The documented reference quirk that `diffuse velocity` and the concentrated distance/velocity are checked **after**
their `min` with the water velocity / 30 m cap is reproduced, to keep refusals identical.

## Floating-point differences and scope

- Identical by construction: regimes, masks, counts, `d50`, shear velocity, Reynolds number, stream power (only
  `+ - * / sqrt`).
- Float differences: `exp`, `log`, `log10`, `**` in Numba use libm; NumPy may use SIMD implementations. Expect a few
  ulp per transcendental, amplified where `xs = stream power - Bagnold` cancels. The test bound is `rtol = 1e-10`;
  Codex should record the observed maximum and tighten it if justified.
- A mask can differ only if a transcendental result lands on an exact threshold (`xs > 0`, `L > 0`) within an ulp.
- Preparation refuses non-finite grain constants and non-finite `(100 S)^c` up front; the reference refuses the same
  states at every step. Pathological separately overflowing static constants can be refused earlier or more strictly;
  the qualifying physical cases agree within the declared comparisons.
- Not implemented: GPU kernels. The prepared arrays are flat contiguous FP64 so a device version can reuse them.

## Reproduction (prepared Python/Numba environment with verified MAPLE dependency)

```
source agent_handoffs/tasks/phase6_complete_event/env.sh   # then the verified candidate_env.sh as in the notes
python -m pytest tests/phase7f/test_compiled_legacy_physics.py -q
python -m pytest tests/phase7f tests/phase5/test_sediment_physics.py -q
python -m maple_syrup.legacy_experiment --output <NEW dir>                              # compiled physics (default)
python -m maple_syrup.legacy_experiment --output <NEW dir> --physics-implementation array   # reference control
```

Compare the two outputs' `legacy_ledger.npz` arrays and read `performance.physics_preparation_s`,
`physics_s` and `first_step_including_warmup_s` from `legacy_summary.json`.
