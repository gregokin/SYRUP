# Subsurface column: first implementation guide

## Current Phase 1 status — 2026-11-12

The opt-in fixed-bed layered water and LEAF calendar is now bounded-qualified
on actual committed sources: SYRUP `e40a1b8`, LEAF `52f39fe`, MASY `a704002`,
and explicitly historical MAPLE `9f466d2`. See
[the release qualification](phase1_release_qualification.md) and the current
[parameter reference](parameter_reference.md). NumPy, Numba and CUDA complete
the small native example with equivalent state and exact same-backend restart.
The original component/design sections below retain their dated scope; their
statements that later integration is pending are historical, not a description
of the qualified release. No wind, evolving-soil, field or scaling acceptance
is implied. Phase 1.5 daily recharge remains the next implementation, after
Phase 1 publication, before Phases 2, 2.5 and 3.

Phase 1.5 daily recharge status, 2026-11-12 UTC: implemented as a **separate native source (N15)** in the candidate and independently accepted by Root through
S4 (bounded native daily lifecycle, CPU/CUDA, V3 physical restart); a bounded whole-phase acceptance is **recorded** (`phase15_acceptance.json`); publication is **complete** (source commit `06952dbc2f984af8fb75dbf09a9c9f7fdbba23fc`). See
[phase15_qualification.md](phase15_qualification.md) for the exact source (package `82e9c011…`, 114 files), controls and limits. The user rule is unchanged: P = 0 is a
no-op, 0 < P <= T attempts simplified recharge (equality included), P > T uses the native full storm, with whole-occurrence fallback from the unchanged parent. It is a
Python API (`maple_syrup.daily_water`), not a CLI option or weather/CSV reader; the nominal ordinal and native physical seconds are an explicit new versioned
contract that does not retrofit historical clocks. The C1 source and its dated clocks, recipes, locks and checkpoints are preserved and are never repointed to N15.
Phases 2, 2.5 and 3 remain queued with their numbering and requirements.

Status: first bounded implementation on `feature/subsurface-hydrology`, task
`agent_handoffs/tasks/subsurface_column_20261031/`. Opt-in only: no existing
event, benchmark, dry-reset or default changes. Implemented by Claude and independently reviewed and tested by Codex.
See `qualification.md` for exact executed checks, source identities, timings,
and limits; author reports alone are not acceptance evidence.

Modules: `src/maple_syrup/subsurface.py` (API, NumPy reference, streaming
initialisation, memory admission), `subsurface_numba.py` (compiled CPU),
`subsurface_cuda.py` (CuPy RawKernel). Tests: `tests/subsurface/`, listed
first in the default pytest `testpaths` (collection only; the conservative
suites stay ahead of `tests/phase2` as the `pyproject.toml` comment requires).
Codex's independent oracles live in the task directory and are not used by
these tests.

## What it is and is not

The column integrates antecedent soil moisture forward under **explicit,
time-resolved water actually directed to the soil** and a **supplied potential
evaporation rate**. The caller decides what enters layer 0 (matrix intake) and
what enters a deeper layer (separately admitted bypass). The column does not
compute rainfall, infiltration capacity, ponding, run-on, plant uptake, or a
profile from an end-of-storm scalar; it does not read or write the storm
bucket (`infiltration.column_step` / `StormState.soil_water_m`), checkpoints,
LEAF or MASY state. Those remain the later milestones of
`design.md` (stages 2 to 4).

## Usage

This milestone is a library API; there is no new storm CLI option yet. Save
the complete example below as `column_example.py` and, from the SYRUP root, run
`PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python column_example.py`. Use the
existing MAPLE environments; do not create a separate SYRUP environment.

```python
import numpy as np
from maple_syrup.subsurface import (
    ForcingRecord, initialize_profile, moisture, prepare_column,
    soil_layers, step, uniform_state,
)

n_cells = 4
layers = soil_layers(
    [0.0, 0.10, 0.30, 1.00],  # physical depth edges in metres
    theta_residual=0.05, theta_field_capacity=0.25, theta_saturation=0.45,
    diffusivity_m2_s=2e-7, theta_evaporation_stress=0.20,
    evaporation_factor=1.2, gravity_drainage=True,
)
column = prepare_column(
    layers, np.ones(n_cells, dtype=bool), cell_area_m2=1.0,
    backend="numba", reserve_bytes=0,  # isolated example; reserve other model memory when coupled
)
state = uniform_state(column, [0.10, 0.12, 0.15])  # antecedent volumetric moisture
intake = np.zeros((layers.n_layers, n_cells), dtype=np.float64)
intake[0] = 0.002  # 2 mm actually directed to the soil over the next hour, not gross rainfall
post_intake = step(
    column, state, 3600.0, added_water_m=intake,
    potential_evaporation_m_s=np.zeros(n_cells, dtype=np.float64),
)
quiet = (
    ForcingRecord(3600.0, np.zeros_like(intake), np.full(n_cells, 3e-8))
    for _ in range(24)
)
profile = initialize_profile(column, post_intake.state, quiet)
print("Final theta:", moisture(column, profile.state)[:, 0])
print("Quiet-period evaporation (m3):", profile.cumulative.evaporation_m3)
print("Water residual (m3):", profile.residual_m3)
```

For a CUDA version use MAPLE's `.venv-cupy`, choose an allocated idle GPU,
set `backend="cuda"`, and construct the mask and dynamic forcing arrays with
CuPy on that device. For example, obtain `xp` with
`from maple.core.backend import cupy_module; xp = cupy_module()` and use
`xp.ones`, `xp.zeros`, `xp.zeros_like`, and `xp.full` instead of NumPy for those
arrays. Layer configuration and the short antecedent theta list remain host
configuration. Keep arrays on the device between steps. Explicitly select the
allocated device with `CUDA_VISIBLE_DEVICES` when launching; do not use devices
belonging to other jobs.

`backend="numba"` compiles the kernel at `prepare_column`; `backend="cuda"`
needs CuPy and a visible device, binds the column to the **current** device,
and expects the mask and every dynamic array on it. An unavailable backend
raises `SubsurfaceBackendUnavailableError`; there is never a silent NumPy
fallback. `column_state(column, water_m, t_s)` wraps an explicit `(L, N)`
array (copied); `moisture` returns theta; `storage_m3` the active-cell total
(one host read); `summary(column)` provenance.

Dynamic arrays (`state.water_m`, `added_water_m` `(L, N)`,
`potential_evaporation_m_s` `(N,)`) must be exactly `numpy.ndarray` /
`cupy.ndarray` of the column namespace (no subclass, no masked array), float64,
C-contiguous and exactly shaped. Forcing must be zero on inactive cells.
Steps are pure: inputs are never written, outputs are fresh arrays, and a
refused step leaves everything untouched.

## Equations, units and order

State: `water_m[l, i]` water-equivalent depth (m) of layer `l` in cell `i`;
`theta = water_m / thickness`; volume = depth x `cell_area_m2`. Common
per-layer parameters: residual, field capacity and saturation theta with
`0 <= res <= fc <= sat <= 1`, `res < sat`; per-interface diffusivity
`D >= 0` (m2/s); layer-0 evaporation stress theta `> res[0]`; evaporation
factor `>= 0`.

One step of `dt > 0` on an active cell, in this order (`STAGE_ORDER`):

1. **Validation** (recorded as flag bits, resolved after the batched read).
2. **Addition.** `w_l += a_l`. If `w_l` exceeds the layer saturation depth
   `sat_l x dz_l`, the excess is **returned to the surface**
   (`surface_return_m`). Nothing is discarded or clipped.
3. **Gravity cascade** (`gravity_drainage=True`). Top to bottom within the
   step: `w_l += carry; excess = max(w_l - fc_l dz_l, 0); w_l -= excess;
   carry = excess`. `percolation_m[k]` is the carry across interface `k`;
   the bottom carry is `drainage_m`. With `gravity_drainage=False` nothing
   moves and water above field capacity stays.
4. **Evaporation** from layer 0 only (the declared upper evaporation layer of
   this scope). `m = 0` if `theta0 <= res0`; `(theta0 / theta_stress)^2` if
   `theta0 < theta_stress`; else `1`. `demand = factor x E x dt x m`;
   `actual = min(demand, max(w0 - res0 dz0, 0))`; `evaporation_unmet_m =
   demand - actual`. There is no "total ET <= potential" cap: factor 1.2 with
   a wet surface withdraws 1.2 E dt.
5. **Diffusive redistribution**, sequential pairs `(0,1), (1,2), ...`:
   `G_k = D_k / (c_{k+1} - c_k)` (m/s) with layer centres `c`;
   `s = 1/dz_k + 1/dz_{k+1}`; exact two-compartment amount
   `(theta_k - theta_{k+1}) / s x (-expm1(-G_k s dt))`, positive downward.
   Before transfer the amount is bounded by the donor's water above residual
   and by the receiver's room below saturation; the same number is subtracted
   from the donor and added to the receiver (`diffusion_m[k]`, signed).
6. **Output checks**: finite, `>= 0`, `<= saturation`, per-cell closure
   `(after - before) = added - return - drainage - evaporation`.

`dt == 0` is the identity (new state equals the old one bit for bit, zero
fluxes); any nonzero `added_water_m` at `dt == 0` is refused (bit 9) rather
than silently advancing time. Inactive cells are never changed and never
counted. An antecedent state above field capacity (but at or below
saturation) is valid and drains on the first positive gravity step.

## Tolerances

Every tolerance is the MAPLE-derived `conservation.volume_roundoff_bound_m3`
(the dimensionless FP64 summation coefficient of
`maple.surface.voxels._numerics.summation_error_bound_kg`). Per-cell closure:
coefficient for `4 L + 7` terms times the cell's operand sum. Global closure
(`StepBudget.bound_m3`): `4 L N + 7` terms on the largest volume operand.
Streaming: `4 L N n_steps + 7`. Capacity checks: `LAYER_OPERATION_TERMS = 8`
terms on the layer saturation depth (`SoilLayers.capacity_tolerance_m`), so a
value that lands an ulp above saturation after an exact-bound transfer is
accepted while a physical overshoot is refused. Nonnegativity is strict.

Failure precedence: the lowest recorded bit over all cells (`FLAG_MESSAGES`),
with the flagged cell count, from one packet read; then the global closure.

## Backends and host traffic

| backend | kernel | per-step host traffic |
|---|---|---|
| `numpy` | reference: loop over layers, vectorised cells, in place | none (host) |
| `numba` | `njit(fastmath=False)` serial cell/layer loops; per-interface `expm1` factor evaluated once per call into a local `(L-1,)` array before the cell loop; compiled and warmed at `prepare_column` | none (host) |
| `cuda` | two launches: a factor kernel (one thread per interface, writes the `(max(L-1,1),)` device scratch; skipped for one layer) then one RawKernel thread per column reading it, layer loops inside the thread, no fixed maximum layer count, options `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false` | 0 uploads, 1 counted download of the 9-value packet |

All three evaluate the same expressions in the same order; only `expm1`
(NumPy vs libm vs device) may differ by an ulp, and it enters only the
diffusive amount. Declared agreement: rtol 2e-12 / atol 1e-14; bitwise where
`D = 0`. The packet (`subsurface._read_packet`) stacks the reductions of the
kernel rows (storage before/after, added, lowest failure bit, flagged count)
and the flux sums; CuPy reductions run on the device and a single
`to_host` brings the 9 doubles back. No per-layer host synchronisation and
no full-field download occur.

Constant-factor reuse: the exchange factor `-expm1(-((G_k s_k) dt))` depends
only on the common layer properties and `dt`, so the compiled backends evaluate
it once per interface per step (same expression order, same math
implementation as before) and reuse it for every cell; there is no cache
across steps, so a changed `dt` never meets a stale factor (tested with
varying `dt` on one column against fresh columns, bitwise within each compiled
backend). On CUDA the factor scratch lives on the device, is held through the
synchronising packet read inside `step` and is never published. This reuse is
conditional on the CURRENT common per-layer properties; a per-cell
heterogeneous conductance map would require the per-cell evaluation again. No
speed figure is claimed here; Codex measures warm steps and compares outputs
bitwise against its pre-optimisation captures.

## Memory estimate and admission

`estimate_step_memory(L, N, backend, reserve_bytes)` is a backend-specific
upper bound, in bytes, of the arrays held for one step:

| field | content | numpy | numba | cuda |
|---|---|---|---|---|
| `input_bytes` | caller's state `(L,N)`, added water `(L,N)`, evaporation rate `(N,)` | host | host | device |
| `column_bytes` | owned bool + uint8 masks, thickness copy | host | host | device |
| `layer_bytes` | all thirteen `SoilLayers` vectors, `(L+1) + 9L + 3(L-1)` doubles | host | host | host |
| `parameter_bytes` | `KernelParameters` copies, `5L + 2 max(L-1,1)` doubles each | 0 | host | host + device |
| `output_bytes` | fresh state, two `(L-1,N)` interface fluxes, four `(N,)` vectors, four rows | host | host | device |
| `scratch_bytes` | transient temporaries of the step plus the packet | 16 doubles + 8 B per cell | 2 doubles + 8 B per cell | 2 doubles + 8 B per cell |

`required_bytes` is the admitted side (host total for numpy/numba, device
total for cuda) plus the caller's reserve; `host_bytes`, `device_bytes` and
`total_bytes` are reported separately; `step_allocation_bytes` is
`output_bytes + scratch_bytes`, what one step itself allocates at its peak.
The NumPy reference now evaluates every stage in place on the fresh output
arrays; its counted worst-case transient is 8 doubles per cell (the int64 flag
word plus at most seven float64 temporaries during the diffusion stage) and a
few boolean masks, declared at twice that count. `tests/subsurface/test_memory.py`
measures the tracemalloc peak of one 3 x 100000 reference step against
`step_allocation_bytes` (and the Numba suite does the same for its kernel);
the bound is a declared upper bound verified by measurement, not a formula
asserted against itself.

Admission happens **before every bulk allocation**: `prepare_column` admits
`required_bytes` before the mask copies, parameter copies, kernel compilation
and warm-up; `column_state` admits `required_bytes + L N 8` (its copy is one
extra state-sized array) before copying; `uniform_state` admits before
allocating and, on CUDA, uploads only the `L` per-layer depths and broadcasts
on the device; `step` admits before the kernel allocates. A refusal is a
`SubsurfaceMemoryError` with the inputs untouched. Default readings: host
`sysconf` available pages; on CUDA the driver free memory plus the default
pool's free bytes under its limit (the same arithmetic as
`storage_admission.effective_free_device_bytes`).

Scope (`MEMORY_SCOPE`): this bounds what the module and its caller hold for
one step. It is **not** the allocator pool or its retained blocks, the CUDA
context, Numba/NVRTC JIT and compile workspaces, Python object overhead,
other model state, or the whole MAPLE/LEAF process; a reading is not a
reservation, and the budget is never raised silently. No history is stored:
the streaming initialiser carries the current state and seven scalars, and it
releases the previous step's result (all flux arrays) before advancing the
next record, so across records only the current state array, the caller's own
inputs and the scalars are live (weakref-verified in the tests).

## Time and overflow

`state.t_s` must be a finite real number at step entry (also for states built
directly from the dataclass), and the candidate time `t + dt` must be finite;
otherwise the step is refused before any admission query or numerical work and
the state is untouched. The streaming initialiser validates every cumulative
scalar (elapsed time, added, returned, drained, evaporated and unmet volumes)
after each accepted step; an overflow raises `SubsurfaceStreamError` carrying
the last accepted state and the prior accepted budget, never a partially
published step. Finite layer inputs whose derived values are not representable
(inverse thickness, centre separation, conductance, exchange rate, storage
depths, capacity tolerance) are refused by `soil_layers`, never floored.

## Scientific references and departures

* [Tietjen, Zehe & Jeltsch (2009)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589): two-layer upper/lower hydrology with
  field-capacity drainage, moisture-gradient exchange and a quadratic
  moisture reduction of upper-layer ET. The configurable layer stack is an
  extension. The two-layer geometry can be represented by
  `depth_edges_m=[0, d1, d1+d2]`; this alone does not reproduce the full
  published model.
* MAHLERAN's daily adaptation (`/home/okin/MAHLERAN/src/Subroutines_Interstorm/`):
  `calc_inf.f90` lines 100-103 exchange
  `24 k_mean d (d1 + d2) (theta1 - theta2) / (0.5 (d1 + d2))` per day, i.e.
  flux = `D x gradient` with `D = K_mean x d x (dz0 + dz1)` and the centre
  separation as the gradient distance, which is exactly `G = D / separation`
  here (in SI units). This column keeps `D` as the physical coefficient at
  every interface and does **not** multiply it by each new pair's thickness
  sum. Selected homogeneous, no-flux tests approach independent diffusion
  references under joint layer and timestep refinement; this is not a blanket
  convergence guarantee for arbitrary stacks or fixed large timesteps. `calc_et.f90` lines
  85-92: `et_1 = et_pot x ((sm/depth)/wsc)^2 x (1.2 - 0.2 veg/100) x
  et_red_fac` between residual and the stress threshold, else the factor
  alone; the no-vegetation bare case is `factor = 1.2`. Here the factor is a
  user parameter and vegetation/crusting modifiers are not modelled.
* Timing departure: the MAHLERAN adaptation moves OLD storage one interface
  per daily update (`drain_1`, `sm_1to2` from `sm(rday-1)`) and then clips
  the new storage to `[theta_min, theta_sat * depth]` (`calc_sm.f90` lines 59-76),
  which can create or discard water. This column cascades top-to-bottom
  within the step (ECOTONE `hwdrain`-like) and never clips; above-saturation
  additions return to the surface explicitly and all transfers are bounded
  before they happen. **No numerical identity with the daily model is
  claimed.** The instantaneous field-capacity cascade is a declared
  assumption, not a finite-rate storm percolation law.
* The exact pair exchange is unconditionally stable and positivity-preserving
  for each pair; the sequential (Lie) splitting over pairs is a first-order
  multilayer extension, not simultaneous diffusion (tested against the
  discrete-ODE matrix exponential under timestep refinement).
* The moisture-gradient law assumes similar retention curves in adjacent
  layers (Tietjen's assumption); contrasting textures need a potential-based
  interface law that is out of scope. Not a Richards solver.
* Changing the thickness of layer 0 changes the physical evaporation zone and
  therefore the evaporation response; deeper evaporation, vegetation
  partitioning and transpiration are later work. Future LEAF uptake must be
  accounted in SYRUP through this state, never deducted twice.

## Not implemented here (next milestones)

Storm coupling (the column must replace or extend the Smith-Parlange bucket
in an explicit mode with intake limited by the receiving profile and
same-step surface return feedback), ponded-surface persistence, preferential
intake laws, checkpoint/restart identity (`DryHandoff` and existing
checkpoints are not compatible with this state), LEAF root/demand interface,
MASY scheduling, lateral flow, per-cell layer stacks, evolving-bed remapping
of soil water, and end-to-end storm/coupled-model performance qualification.
The qualified timings apply only to the isolated column.
