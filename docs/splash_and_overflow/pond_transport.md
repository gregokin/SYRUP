# Pond multi-face characteristic transport (`maple_syrup.pond_transport`)

Pure lateral sediment operator for the opt-in local-inertial pond backend. It does not touch the MAPLE bed,
creates no inventory, and changes no existing file or default (legacy, upwind and single-receiver
characteristic paths are untouched). No event, checkpoint, storm or hydrology claim is made.

The integrated [PondEvent](pond_event.md) uses this order:

    native pickup cap -> pond_transport_step on the ACTUAL pickup -> native deposit/export -> phase reconciliation

## API

| Name | Purpose |
|---|---|
| `face_network(active, dx_m, outlet_x=None, outlet_y=None, *, periodic=False, external_inflow=False, xp=None)` | Network from the host active mask and explicit exterior open-face booleans (`outlet_x (ny, nx+1)`, `outlet_y (ny+1, nx)`). |
| `face_network_from_geometry(active, fx_type, fy_type, dx_m, *, fx_sign=None, fy_sign=None, ...)` | Typed face codes (0 closed, 1 interior, 2 outlet), as in `LocalInertialGeometry`. |
| `face_network_from_pond_topology(topology, *, xp=None)` | Thin duck-typed adapter reading `.active`, `.dx_m`, `.geometry.{fx,fy}_{type,sign}`. |
| `outflow_weights(network, vol_x, vol_y)` | Validated `(ny, nx, 4)` weights, direction order east, north, west, south (`DIRECTIONS`). |
| `pond_transport_step(network, mobile_before_kg, phase, pickup_kg, v, r, settle_mask, vol_x, vol_y, dt_s, *, n_substeps, courant_max, implementation="array")` | The operator. Returns `PondTransportStep`. |
| `validate_phase(network, mobile_kg, phase)` | Phase invariants for a mobile field. |
| `empty_phase_state`, `PhaseState` | Re-exported from `characteristic_transport` unchanged. |

Units and conventions: row 0 = south; x faces `(ny, nx+1)` positive east; y faces `(ny+1, nx)` positive north.
Masses kg, volumes m3 (signed, one physical step), `v` m/s (virtual sediment velocity, not water velocity),
`r = 1/L` in 1/m, `dt_s` s. All FP64; `settle_mask` bool; arrays `(ny, nx, nc)`; phase `(ny, nx, nc, B)`.

Result `PondTransportStep`: `transport` (the unchanged `TransportStep`), `phase_after` (on the remaining pool),
`mobile_before_kg = M + P`, `mobile_remaining_kg`, `arrival_deposition_kg`, `crossing_kg`,
`face_crossing_by_direction_kg (ny, nx, nc, 4)`, `outflow_weight (ny, nx, 4)`, `blocked_cell (ny, nx)`.

## Topology rules

- Interior faces are open in both directions only between two active cells; a typed interior face of type 0 is a
  wall. Nothing crosses an inactive cell; interior holes stay closed.
- Export leaves only through designated exterior outlet faces of active cells. No outlet at all is a closed domain.
- `periodic=True`, `external_inflow=True` and `external_inflow_kg` at the step raise `TransportError`.

## Water versus sediment

Each accepted physical step passes signed actual water volumes through every face. A cell's outward volume
through face d is `max(sign_d * volume, 0)`; positive outward volumes, scaled by the largest first and
normalised, are the split weights. Refused before any output: non-finite volumes, nonzero volume on a closed
face, water entering the domain through an outlet, wrong dtype/shape.

## Law

Same exact-within-substep characteristic law as `characteristic_transport` (packet position `x` in `[0, dx)`,
pickup injected at `x = 0`, survival `exp(-r s)`, deposition `m (-expm1(-r s))`, one face per substep via
Courant <= 0.5). Eventual survival to a face is `exp(-dx / L)`; the well-mixed `L / (L + dx)` transmission is not
used. At a crossing the survivor is split over the current positive-outflow faces by weight; outlet shares become
export requests of the source cell, interior shares arrive in the neighbour, advance for the remaining time at the
receiver's `v` and `r`, may settle there if its `settle_mask` is set. Bin merging is the existing hazard-preserving
logarithmic mean (narrow `expm1`/`log1p`, wide log-sum-exp, mass-weighted for `r = 0`). With one weight-1 face per
cell the arithmetic equals the existing D4 operator (tested against it).

## Documented choices and approximations

1. **Flow blocking.** A cell with no positive outflow face this step makes no advection progress: effective `v = 0`
   (so no `v r` hazard), packets and phase are preserved, pickups and arrivals sit at `x = 0`. It deposits only via
   an explicit `settle_mask`; it never exports and never deletes. Transporting against or around the water would
   invent motion the actual water does not do. The velocity of such a cell is not Courant checked (it does not
   act); `blocked_cell` reports it. Cells with outflow are checked and a violation raises
   `TransportStepRejected`. No velocity is clipped.
2. **Turning flow / scalar progress.** One scalar distance `x` from "the" upstream face. Mass entering through one
   face and leaving through another keeps its scalar progress `v_r (dt - tau)`; the distance to the exit face is
   always `dx - x`. No geometric path inside the cell.
3. **Volume weighting.** Split by the whole step's water-volume ratio, frozen over the substeps; not by sediment
   concentration and not by per-substep flux. Intra-step reversal is represented by net sign only. A weight whose
   ratio underflows FP64 becomes 0 and its mass goes to the other faces.

## Bookkeeping

`T = remaining + Dep + E` (T keeps the export until native MAPLE removes it). Two accountings:

- pre-native pool: `T - (M + P) = In_internal - Out_internal` per cell/class (`internal_transfer_*`, which
  exclude export); `sum T = sum (M + P)` per class before any bed request;
- full face crossings (x/y arrays, include export): net divergence
  `D = (x_net[c+1] - x_net[c]) + (y_net[r+1] - y_net[r]) = Out_internal + E - In_internal`, so
  `D = (M + P) - (T - E)` (not `(M + P) - T`) and the post-native pool is `remaining = (M + P) - D - Dep`.
  Where there is no export, `D = (M + P) - T`.

Export requests only on cells that own an outlet face. FP64 residuals are measured against MAPLE's
`summation_error_bound_kg` with `operation_count(n_substeps, B) = 60 n_substeps (B + 1) + 8` (about 12 roundings
per slot for the own-cell path plus 12 for each of four directions); the weights `w_d` need not sum to exactly 1 in
FP64, so splitting adds rounding at the 1 ulp level inside that count. Nothing is clipped to hide a residual and
no mass tolerance is widened. Phase after: fractions positive and summing to 1 where the remaining pool > 0,
canonical (`e_0`, position 0) where it is 0.

## Backends, memory

`implementation="array"`: NumPy or CuPy via the network's namespace; loops over 4 directions and substeps only;
no host/device transfer in the step; memory `n nc 5 (B + 1)` candidate slots plus a few same-size temporaries,
no history kept. `"numba"`/other values raise `TransportError`; no compiled CUDA/Numba kernel is provided
(unbenchmarked, so none is claimed).
