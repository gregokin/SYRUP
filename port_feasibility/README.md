# Isolated MAHLERAN-to-MAPLE feasibility tests

This directory is an executable numerical/integration experiment. It changes neither `/home/okin/MAPLE` nor `/home/okin/MAHLERAN`. It is **not a completed port**, a calibrated erosion model, or a demonstration of legacy output equivalence.

## Run

Use the existing MAPLE environment (NumPy, SciPy, pytest and MAPLE are available):

```sh
cd /home/okin/SYRUP
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORM_NAME=cpu /home/okin/MAPLE/.venv/bin/python -m pytest -q -ra -p no:cacheprovider port_feasibility/test_feasibility.py
PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python port_feasibility/benchmark.py
```

The MAPLE integration test uses the existing test-bed construction helper, with its root configurable through `MAPLE_ROOT`. The default is `/home/okin/MAPLE`. SciPy is used only for an independent scalar root reference. A standalone production package is not claimed.

## What these tests establish

- Ordered constant-friction water routing solves a scalar implicit balance, checked against SciPy's independent root solver.
- Branching inflows preserve water locally and globally, with outlet exports explicitly integrated over the timestep.
- A rainfall pulse can wet downstream cells using new upstream discharges in the same ordered sweep.
- Timestep refinement approaches an analytic single-cell drainage solution at the expected second-order rate.
- Source-free multiclass sediment routing agrees with an independently assembled dense linear-system solution.
- Invalid ordering/cycles and negative water/sediment right-hand sides fail before caller state is mutated. Negative mass is not clipped away.
- A checkpoint of this prototype's hydraulic depth/discharge reproduces uninterrupted execution exactly. This is a small NPZ checkpoint, **not MAPLE's production restart system**.
- Actual MAPLE routines cap excessive pickup against bed holdings. The prototype routes that actual mobile mass to a second cell; MAPLE deposits it into the same shared bed. Total bed mass is conserved and the final mobile pool is exactly empty. This uses an intentionally prescribed deposition demand to test plumbing, **not a physical settling law**. Boundary handling of the MAPLE helper's periodic grid is not exercised; the diagnostic has no external export.

Initial result: **13 tests passed**. Numerical tests are small and controlled. No full wind event, infiltration, rain splash, erosion-distance law, changing topography or production restart is implemented here.

## Relation to legacy equations

`kernels.py:water_step` uses the SI constant-friction form of `MAHLERAN/src/Subroutines_Water/route_water.for`, iroute=5:

`h_new + dt/(2 dx) k h_new^(3/2) = h_old + dt rain + dt/(2 dx)(qin_old - qout_old + qin_new)`

with `k = sqrt(8 g slope / friction)` and discharge `q = k h^(3/2)` in m²/s. Volumes use square-cell area; outlet flux uses cell width. The network is supplied explicitly and held fixed. Its single-receiver cells represent equal-width cardinal connections. No claim is made about automatic DEM routing, pits, masked cells or irregular catchments.

`sediment_step` uses the source-free SI mass form of method-2 continuity at `Subroutines_Sediment/route_sediment_xml.f90:279–292`. Pickup and deposition are outside the routing step. This explicitly chosen operator split makes a testable conservative experiment; it is **not claimed equivalent** to the complete legacy simultaneous source/sink update.

Deliberate differences from legacy: water bisection brackets the monotone root in `[0,rhs]`, uses a bounded floating-point stopping rule, and both kernels reject negative right-hand sides. The legacy bracket and absolute tolerance are not copied; there are no STOP calls or negative-mass floors. Inputs are checked, and new arrays are returned without mutating the caller. Matching these discrete equations does not prove matching Fortran execution.

## Hardest unresolved issue

The central risk is the consistency of the complete pickup–advection–deposition system. Legacy wet/splash rates have different timestep conventions, deposition can exceed its mobile supply, and clipping hides the imbalance. Making a model conserve mass is necessary but not sufficient to preserve the intended erosion and transport behavior.

Next experiments should bring in one selected infiltration option, actual wet detachment/deposition-distance tendencies and depletion, then quantify timestep/spatial sensitivity and outlet yield. Splash needs its own conversion and conservative transfer tests. A physical end-of-event settling/runoff policy must be selected. Only then should geometry changes and full MAPLE wind–water restart sequences be accepted.

## Performance and environment limits

`benchmark.py` times three fixed water-only steps on each of 16×16 and 32×32 networks and writes `benchmark_results.json`. These are plain-Python CPU measurements, not full model forecasts or evidence against any compiled implementation.

Numba is absent from the MAPLE environment and gfortran was not found in PATH. No packages were installed. Compiled/JIT timing and execution of the original Fortran remain untested. A follow-up could add an isolated environment for those comparisons without altering MAPLE's dependencies.
