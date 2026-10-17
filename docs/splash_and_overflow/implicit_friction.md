# Opt-in implicit quadratic Darcy-Weisbach friction (local-inertial solvers)

Status: implemented and independently exercised by Codex on CPU and GPU. See [qualification](qualification.md) and
[acceptance](acceptance.json) for exact source versions, results and limits; the component descriptions below are not whole-catchment qualification.

## Why

The original face update is `q' = q* / (1 + dt (f/8) |q_old| / h_f^2)` with `q* = q_old - g h_f dt grad(eta)`. On a first
wetting `q_old = 0`, the denominator is 1 and the first flux is the frictionless impulse `q* = -g h dt grad(eta)`. The
resulting velocity `g S dt` does not depend on the film depth. Root measured the consequence on a held native 32x32 crop
(f 21.45, dx 1, rain 120 mm/h, dt 0.25, dry start): at a wetting face with `h_f = 2.16014956755e-7 m` the first flux gave
`|u| = 0.2698 m/s`, `Fr = 185`. The faces with `Fr > 0.5` carried 0.0112 % of the gross flow, but the speed feeds
erosion, so it cannot be ignored or clipped. This is a time-lagging artifact, not a positivity, permission or GPU fault.

## Equation

`HydraulicControl.friction_treatment` (new last field):

| value | meaning |
|---|---|
| `"lagged"` (default everywhere except the pond `control=None` default) | the original formula, arithmetic and operation order untouched, bit-identical to before |
| `"implicit_quadratic"` | friction at the NEW flux: `q' + alpha |q'| q' = q*`, `alpha = dt f / (8 h_f^2)` |

Stable signed root: `q' = 2 q* / (1 + sqrt(1 + 4 alpha |q*|))`. Multiply numerator and denominator by `h` (check:
`4 alpha |q*| h^2 = dt f |q*| / 2`):

    q' = q* w,   w = 2 h / (h + D),   D = sqrt(h^2 + b^2),   b = sqrt(dt f / 2) * sqrt|q*|      (units: h, b, D in m)

Evaluation, identical on all backends and using only `+ - * /` and `sqrt` (no libm, no FMA):

    b = sqrt(c) * sqrt(|q*|),  c = (dt f) * 0.5        # never forms dt f |q*|: no overflow; never forms h^2: no underflow
    m = max(h, b),  n = min(h, b)
    D = m * sqrt(1 + (n/m)^2)                          # hypot-style, n/m <= 1
    t = h / D          (<= 1)
    q' = q* * (2 t / (1 + t))                   if t >= 1/2
    q' = ((q* / D) * h) * (2 / (1 + t))          if t <  1/2   (C1: t can underflow against a huge q*; q* t would be a false 0)

Properties (hold in floating point, not just in exact arithmetic):

* No subtraction of nearby numbers, hence no cancellation. `t <= 1` and `2t <= fl(1+t)`, so `0 < w <= 1`: `|q'| <= |q*|`,
  the sign of `q*` is kept, and `q' (q' - q*) = -alpha |q'| q'^2 <= 0` (friction is dissipative).
* `q* = 0` returns `q*` unchanged (including `-0`). `alpha = 0` (`f = 0` or `dt = 0`) gives `w = 1` exactly.
* Huge `alpha`: `w ~ 2h/b`, finite. Thin depth: correct and finite whenever the TRUE root is a normal FP64 number and
  `q*/D` does not overflow (`q*/D <= sqrt|q*| / sqrt(dt f / 2)`, so overflow needs `dt f` below ~1e-308 or a denormal `h`).
  The first (C1) version claimed this but flushed `q* = +/-1e300, h = 1e-200, f = 21.45, dt = 0.25` to +/-0 (`t = 1e-350`
  underflowed before the product); the 100-digit root is +/-1.2214096331049185e-50 and is now returned. The `t >= 1/2` branch is
  the previous arithmetic; ordinary cases changed only within the stated 32-eps native error. A dry face (`h_f = 0`) is masked
  to `q = 0` as before.
* Refusal, not silent failure: a non-finite `q*` (the existing overflow behaviour), an overflowing `D`, `q*/D` or result returns NaN. The
  existing non-finite flag then refuses the step on CPU, Numba and CUDA alike. No depth, mass, slope, Froude or velocity floor
  or clip is introduced.
* Fixed point: uniform flow `q = sign(S) sqrt(8 g S / f) h^(3/2)` is an exact fixed point for every `dt` (it is also the
  steady state of the lagged form). From rest the flux rises monotonically toward it and never overshoots, because
  `q' + alpha q'^2 = q + P` with `P = alpha q_eq^2` gives `q' <= q_eq` whenever `q <= q_eq`. A first wetting therefore has
  `Fr' < Fr_eq = sqrt(8 S / f)` (0.2026 for S = 0.11, f = 21.45) independent of depth. The root's scalar case gives `Fr ~ 0.202`.

Physics statement: this is a new time discretization of the same empirical shear `tau = rho f u^2 / 8` used by the
MAHLERAN-like steady Darcy-Weisbach law. It is not a Fortran routing identity and not full advective inertia. The outlet
normal-flow face (`q = k_b h^(3/2)`), the donor limiter, method 5, legacy routing, infiltration and the conservation policy are
unchanged.

## API

* `HydraulicControl(cfl_max=0.5, limiter="off", friction_treatment="lagged")`; validated early (before any kernel) in
  `HydraulicControl.validated()`, `build_pond_hydraulics` (as `PondHydraulicsError`) and every solver constructor. The explicit
  kinematic method refuses `"implicit_quadratic"`.
* `experimental_hydrology.implicit_quadratic_friction_flux(q_star, h_face, fric, dt)`: the NumPy helper (the arithmetic the other
  backends mirror). `local_inertial_face_flux(..., dt, dx, friction_treatment="lagged")`.
* `describe()` reports `control.friction_treatment` (all solvers) and `friction_treatment` (pond).
* Numba: the compiled `local_kernel` has a new argument `implicit_friction` right after `donor_limiter`; one branch flag per
  launch, no per-cell Python loop. The treatment is part of the sealed control and re-checked before every compiled call.
* CUDA: both face kernels (`maple_syrup_li_face_x/_y`) take a new trailing `const int implicit_friction`; device function
  `implicit_quadratic_flux`; no kernel added (7 local-inertial kernels, launch counts unchanged). The treatment is in the
  solver seal.
* Pond: `build_pond_hydraulics(control=None)` selects `"implicit_quadratic"` (`POND_DEFAULT_FRICTION_TREATMENT`). An explicit
  `HydraulicControl()` keeps `"lagged"`. `rebind_pond` keeps the pond's control.

## Qualification and limits

* NumPy/Numba/CUDA helper and solver checks were actually executed by Codex, including extreme thin-film roots and the small evolving native Chastre case. See [qualification](qualification.md) for recorded source identities and comparison results. Identical expressions alone are not used as proof of backend parity.
* The solver-level path cannot reach `q*` overflow with finite state (`q_old` is finite and the pressure term is small); the NaN
  refusal is tested at the helper and Numba-scalar level, and the CUDA code was only read for it.
* `PondEventControl` explicitly defaults to implicit quadratic friction. An explicit `HydraulicControl()` remains lagged. Anything that serialises `dataclasses.asdict(HydraulicControl)` into an identity now sees a third key.
* Existing `tests/pond` tests that call `build_pond_hydraulics(..., control=None)` now run the implicit treatment; any that assert
  lagged-specific values must pass an explicit `HydraulicControl()`.
* The Froude argument is a first-wetting/equilibrium statement for the Darcy-Weisbach friction; later transients with changing surface slope have no universal low-Froude guarantee. The coupled tests use measured face speeds with separate friction-based erosion slopes and report their Froude diagnostics. This is bounded integration evidence, not a universal validation of local inertia at high Froude number or against all MAHLERAN storm outputs.
