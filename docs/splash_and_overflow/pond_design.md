# Pond hydraulic foundation — stage 1 design (hydraulics and topology only)

Status: implemented and independently exercised by Codex on CPU and GPU. See [qualification](qualification.md) and
[acceptance](acceptance.json) for exact source versions, results and limits; the component descriptions below are not whole-catchment qualification.

Modules: `src/maple_syrup/pond_network.py` (host topology), `src/maple_syrup/pond_hydraulics.py` (solvers, step, ledger,
diagnostics). Tests: `tests/pond/`. They are included in the default `testpaths` and can also be run explicitly (`pytest tests/pond`).

## 1. Conventions

Row 0 = south, north = +row, east = +col. Interior `(ny, nx)`; full bed `(ny+2, nx+2)` with the one-cell ring. x faces
`(ny, nx+1)` positive east, y faces `(ny+1, nx)` positive north. Units: elevations, depths, `dx` m; unit discharge m²/s;
volumes m³; speeds m/s; time s. FP64 everywhere.

## 2. Network (`pond_network`)

| Item | Definition |
| --- | --- |
| Interior faces | One signed face between every pair of D4-adjacent **active** cells (carries both directions). Built by the existing `build_local_inertial_geometry`. Faces touching an inactive cell are closed; this is asserted after the build. |
| Outlets | `OutletFace(row, col, d_row, d_col)`, explicit only: an active interior cell and a D4 direction onto the ring. Never inferred from the legacy D4 graph. |
| Boundary law | The existing Darcy normal-flow closure `q = k_b h^1.5`, `k_b = sqrt(8 g S_b / f)`, `S_b = (z_cell − z_ring)/dx`. |
| Refused | Outlets with a drop ≤ 0 (zero-drop, uphill, backwater, stage-controlled), outlets into the interior, off-grid or inactive cells, duplicates, ring cells holding the nodata value. |

### Genuine limit of reuse

The reused boundary needs a **real strictly positive bed drop** to the ring cell. This stage does not add a minimum slope, a
ghost slope or a bed perturbation. A flat interior with a really stepped-down ring cell is valid. A backwater or stage
boundary needs its own scoped boundary (not built here).

### Escape (spill) levels

`escape_level_m[c] = min over D4 paths c→outlet cell of max z along the path` (node-weighted minimax). It is computed by a
host priority queue (Dijkstra; ties broken by `(level, flat index)`), only at the initial build and at a triggered refresh.
An outlet cell's level is its own bed because the normal-flow law discharges at any positive depth. Values are exact bed
values (only `max` is applied).

- `fill_depth_m = escape − z`.
- **Closed components** (active cells with no path to an outlet): level `+inf`, `outlet_connected = False`,
  `closed_component_label ≥ 0` (connected components). They are refused unless `allow_closed_components=True` (explicit
  designation; a domain with no outlet needs it).
- `escape_path_next` is a tie-broken spanning-tree pointer (−1 outlet cell, −2 closed/inactive), **not** a unique spill face.
- No basin labels and no spill faces are claimed, because none are computed. A single-cell neighbour lip
  `max(z_c, z_n)` is a face lip, never a multicell spill level (tested with a 1×4 example whose naive lip is 1 and whose true
  level is 3).

The escape levels are topology/diagnostic data. Face activation in a step is decided by water head
(`h_face = max(eta) − zmax`), never by a topology rebuild.

### Immutability

Read-only private copies of every array; frozen dataclass; digests: `source_sha256` (held routing-source bed),
`identity_sha256` (mask, friction, dx, outlets, nodata; invariant across refreshes), `topology_sha256`.

### Net-elevation refresh

`update_pond_topology(topology, new_bed_full_m, threshold_m)`; the caller passes the **actual new physical bed** and the
**resolved MAPLE threshold** (m, finite, > 0). Gate:

```
max over ACTIVE cells |z_new − z_held_source| >= threshold_m
```

- Per-cell net versus the held source: same-cell deltas that cancel never trigger; deltas in different cells never add or
  cancel; slow drift accumulates until it triggers. No accounting, clock, water-level or step-count trigger.
- Below threshold the **same topology object** is returned (exact old geometry and source).
- Triggered: the new topology is built and fully validated first (including positive outlet drops); the module publishes
  nothing, the caller assigns it. The caller must have committed the native bed before adopting it; this module never reads
  or writes a native bed.
- The new bed is always validated (finite, float64, shape, active cells not nodata), even below threshold; the ring must equal
  the held ring (outlet identity is not an ordinary reroute); mask, friction, dx, outlets cannot change.
- Retained across a refresh: `qx, qy`, depth, soil water, `next_dt_cap_s` (the caller keeps the state; `rebind_pond`
  validates it against the new faces and returns the same object).

## 3. Hydraulics (`pond_hydraulics`)

### Reuse versus new code

| Reused modules/interfaces (imported; optional friction extension documented separately) | New in this stage |
| --- | --- |
| `CpuHydraulicSolver`, `NumbaHydraulicSolver`, `CudaHydraulicSolver` (local-inertial, Darcy–Weisbach friction, gravity-wave CFL, negative depth rejected with limiter off) | wrapper `PondHydraulics`, `build_pond_hydraulics` |
| `HydraulicState` / `HydraulicStep` / `HydraulicControl`, `stage_face_diagnostics`, `advance_time`, `check_dt_cap` | `step_once` (carries `next_dt_cap_s`), `evolve_pond` (bounded) |
| accepted column physics (`infiltration.column_step`, `hydrology_cuda.prepared_column_step`) via the solvers | `PondLedger` (per cell + global, directional face volumes) |
| `build_local_inertial_geometry` with `open_faces` | topology, escape levels, refresh gate |
| `ExperimentalControl`, adaptive-cap rule of `evolve_experimental` | outflow-volume-weighted cell speed, `rebind_pond` |
| `conservation.volume_roundoff_bound_m3` for every tolerance | metadata `RoutingGraph` builder |

### Exact adaptation from the local-inertial assumptions

1. The solvers' `RoutingGraph` argument is satisfied by a **metadata graph** (or a caller's native graph): the
   local-inertial path reads only shape, dx, active mask and friction; conveyance, outlet, receiver and level arrays are
   unused and the legacy graph is never the flow-direction authority. The default metadata graph raises every non-active cell
   to `max(active z)+1` so the legacy builder sees no exterior receiver or edge rule, and uses
   `allow_pit_storage`/`allow_flat_storage` so flats and pits are accepted. This elevated surround exists only inside that
   object.
2. Constructor constraint from reuse: the legacy graph builder needs friction ≥ 0.1 on active cells
   (`LEGACY_FRICTION_FLOOR`); the pond geometry itself needs only > 0. Friction below 0.1 is refused with an explanation.
3. Outlets are explicit; `graph.outlet` is ignored.
4. The bed used by a step is the **held routing source** (`geometry.z`), changed only by a refresh.
5. This is an explicit **alternative** to method 5 / option 6, not a numerical identity. The donor limiter is **refused**
   (it changes the dynamics). `cfl_max ≤ 0.5` validation, CFL rejection and negative-depth rejection are the solver's own and
   unchanged. No minimum slope, no depth clipping, no widened tolerance.

### Step and evolve

- `step_once(pond, rain_rate, state, dt)` is pure. A `HydraulicStepRejected` leaves the caller's state untouched (retry the
  same state with a smaller dt). The returned state keeps `qx, qy` and the caller's `next_dt_cap_s`.
- `evolve_pond(pond, rain_rate, state, duration, control)` advances **one** interval with a rain rate held constant over it
  (split at forcing knots yourself). Each accepted substep runs the accepted column physics with its own dt, so rain and
  infiltration are integrated exactly. A rejection halves dt from the unchanged state, bounded by `max_retries`, `min_dt_s`
  and `max_steps`. On a bound it **stops** and returns the last accepted state, `reached_target=False` and a `stop_reason`
  (`max_retries`, `min_dt`, `max_steps`, `time_resolution`); no hidden microsteps. `elapsed_s` is the actual elapsed time.
  The cap rule is that of `evolve_experimental` and lives in `state.next_dt_cap_s`; with dyadic dt a split run is bitwise the
  same step sequence as a continuous run (tested).

### Ledger (m³, area `dx²`)

```
surface (per cell): area (h_end − h_start) = area (Σ rain − Σ intake + Σ saturation_return) + net face inflow
soil    (per cell): area (S_end − S_start) = area (Σ intake − Σ saturation_return − Σ drainage)
global            : Σ(surface + soil change) + drainage + export = rain
```

Face volumes `V = dt·dx·q'` (signed) accumulate as `pos = Σ max(V,0)`, `neg = Σ max(−V,0)`, `net = Σ V`; gross = pos + neg.
Tolerances: `volume_roundoff_bound_m3(4·n_cells·n_steps + 7, largest operand)` globally (the existing event rule) and
`volume_roundoff_bound_m3(16·n_steps + 7, largest per-cell operand)` per cell. These checks passed the component and integrated tests; no absolute or mass floor is added. Inactive cells must be bitwise unchanged.

### Face-speed diagnostics

- Face velocity and Froude from the existing `stage_face_diagnostics`: `u = q/h_face` with the depth the flux was computed
  from (dry face → 0), `Fr = |u|/sqrt(g h_face)`. **Not clipped**; a counted statistic compares Fr with a caller reference.
  No drying-cell-centre velocity is used.
- Cell speed for future erosion coupling: `Σ_k V_out,k |u_k| / Σ_k V_out,k` over the cell's faces whose signed volume leaves
  the cell (outlet faces included); exactly 0 when nothing leaves (tested with a lake at rest).
- **Slope caveat for future wet laws:** the topographic bed slope that governs rain impact/splash is not the hydraulic
  (friction-driving) slope under unsteady or flat flow. Shear must use the flow's water-surface/friction slope; rain/splash use
  the bed slope. Stage 1 substitutes neither for the other and supplies only face speed and Froude.
- The inherited velocity qualification (`QUALIFICATION_STATUS`) still applies: face Froude can exceed the low-Froude range
  where local inertia is accurate.

## 4. Backends

`backend="numpy"` (reference), `"numba"` (optional, raises the existing unavailable error, no fallback), `"cuda"` (CuPy
params and graph, no fallback). `params` must be a `ColumnParameters` in the backend's namespace with the topology's active
mask. CPU/GPU tests use the same topology, parameters, forcing, dt and control and the existing agreement bounds
(rtol 2e-12, atol 1e-14); they skip only when the backend is unavailable.

## 5. Integration and follow-ups

Coupled sediment and restart are integrated in `PondEvent` and `pond_checkpoint`. The real small Chastre crop was exercised on CPU/GPU with a native positive-drop outlet; see [qualification](qualification.md). Explicit-stage/backwater boundaries, windowed integration and large-domain performance remain follow-ups. Face-speed diagnostics are used with the separate friction/topographic slope choices described in [the event](pond_event.md).
