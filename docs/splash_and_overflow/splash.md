# Direct rain-splash core (`maple_syrup.splash`)

This bounded redistribution component is integrated by [PondEvent](pond_event.md). Codex executed the CPU/GPU native-bed tests and original Fortran interior comparisons; see [qualification](qualification.md) for actual results and limits. Component tests alone are not a full storm qualification.

## Source law (MAHLERAN `splash_transport.for`, read-only reference)

Per source cell with actual pickup `P` (kg), class decay `spd` (1/m), `a = spd*dx`, `g = exp(-a)`:

| Part | Mass | Fortran |
|---|---|---|
| local deposit | `P (1-g)` (computed as `-expm1(-a)`) | 47-51 |
| ray `k`, cell `n = 1..m_k` | `P w_k g^n (1-g)` | 119-239 |
| ray `k` remaining tail beyond the last reachable cell | `P w_k g^(m_k+1)` | (cut by `nmax`/`10e-50 dt` in Fortran, **not here**) |

Ray weights `w` (each applied to the full `P`): cross-slope `0.25` twice, down `0.5(1 - 0.5 exp(-2.42 S))`, up
`0.5 - down`; flat (`S == 0`) `0.25` each. Down ray = aspect (1 N, 2 E, 3 S, 4 W, the MAPLE routing codes), up =
opposite. Ray slot order is the Fortran `splash_mv(1..4)` = east, **south**, west, north (legacy row `i+1` is the
MAPLE south because MAPLE row 0 is south). Slopes are dimensionless m/m.

Deliberately not reproduced: `nmax_splash`, the `vge = 10e-50*dt` tail stop, the extra edge-cell deposits at
247-267 (they add mass). Codex executed the original Fortran interior for five aspects; see [qualification](qualification.md).

## What ends a ray (tail disposition)

- **Open outer face** (per face, see "Exterior openness"; the default is ALL CLOSED): the tail is *exported*
  through that face. Reported per edge cell (`export_edge_cell_kg`, MAPLE's `boundary_export_by_cell_class_kg`
  convention: the cell that holds the mass when it crosses), per source cell (`export_source_cell_kg`, independent
  closed form `P w g^(m+1)`), and as the boundary face crossing.
- **Closed edge or inactive cell** (`active == False`: nodata mask, wall): the ray stops at the first inactive cell
  and the tail is *deposited at the last active cell before it* (the source itself if the very next cell is
  blocked, i.e. the tail returns to the source). Justification: droplets thrown against a wall or masked hole
  come to rest at its foot; this keeps all mass in the domain, needs no outside bookkeeping, treats walls, nodata
  and closed edges identically, never jumps a hole, never reflects, and is exact (no cut).

## Algorithm and cost

Per direction the ray mass obeys the line recurrence `F_out = g (F_in + P w)`, interior bin `F_in (1-g)`, which
equals the closed-form bins above exactly (FP64). Loops run over `nx` (east, west) and `ny` (north, south) with
each step a vectorized NumPy/CuPy operation over all lines and classes (`2(nx+ny)` steps total). No loop over
source cells, none over `1/spd`, no array sized by travel distance (`spd -> 0` costs the same as any other `spd`).
Static ray geometry (continuation mask, edge cells, reachable counts) is built once on the host in
`splash_grid`; the redistribution makes no host transfer (one batched validation flag read at the end, like
`sediment_transport`). Backend: `maple.core.backend` namespace of the grid (NumPy or CuPy). The array loop is
the reference; an optional CUDA sweep (below) removes the per-step launches on a CuPy grid.

## API (all units SI)

```python
params = splash_parameters(inverse_distance_per_m, xp=None)   # (nc,) 1/m, finite >= 0, REQUIRED, no default
grid   = splash_grid(slope, aspect, active, dx_m, open_x_faces=None, open_y_faces=None, open_sides=None, xp=None)
grid   = splash_grid_from_graph(graph)                        # RoutingGraph slope/aspect/active/dx + outlet faces
res    = splash_redistribute(params, grid, pickup_kg, implementation="array")  # (ny, nx, nc) float64 ACTUAL kg
ex     = apply_splash_exchange(state, context, params, grid, requested_pickup_kg, implementation="array")
```

### Exterior openness (C1)

Openness is per FACE in MAPLE's layout: `open_x_faces` `(ny, nx+1)` (index c = face west of column c) and
`open_y_faces` `(ny+1, nx)` (index r = face south of row r). Default: everything closed; an unspecified boundary
never exports. Validated: bool, exact shape, interior faces closed, and no open face next to an inactive
(nodata/closed) edge cell (error). The input arrays are copied; the grid keeps read-only copies. `open_sides=(west,
east, south, north)` is only a generic shorthand (opens the named side's faces of *active* edge cells, silently
skipping inactive ones; cannot be combined with explicit arrays). A pond network passes its own explicit face
arrays, independent of any D4 graph.

`splash_grid_from_graph` derives the faces from the graph's own export configuration: each `graph.outlet` cell
opens exactly the outer face in the direction of its `graph.aspect`; a graph with no outlets (for example the flat
pit-storage graph) is fully closed and a source there exports nothing. An outlet whose export receiver is an
in-grid inactive cell is not an outer face: it stays closed (an obstruction) and is counted in
`grid.ignored_interior_outlets`. It never reads the export ring directly. The grid metadata (shapes, dtypes,
namespace, open-face masks) is re-checked on every redistribution (`corrupt` error); values are not re-read from
the device.

### Implementation selection

`implementation="array"` (default, reference, NumPy or CuPy), `"cuda"` (CuPy grid only; no fallback; a missing
device or compile failure raises `SplashCudaUnavailableError`, a NumPy grid raises `SplashError`), `"auto"` =
`"cuda"` only for a CuPy grid *and* `splash_cuda.CUDA_QUALIFIED is True`, which this code never sets (so `"auto"`
means `"array"` until a qualified run flips it). Anything else is refused. Numba CPU is not provided.

### CUDA sweep (`splash_cuda.py`)

One thread per (line, class); each walks its ray sequentially with the flux in a register, so there is no per-source
history and no global scratch; four launches (east, south, west, north) on the current stream, accumulating into
the output arrays in the array path's slot order (no atomics needed: one cell belongs to one line per launch).
Cell-frame contiguous arrays and signed (line, start, step) strides replace the transposed/reversed views, so no
view is ever passed to the kernel. Compiled with `--fmad=false` (the `routing_cuda` options), round-to-nearest
intrinsics only, no fast math; same operations/order as `_array_rays`, so bitwise equality is intended and is tested
as a hypothesis. Not compiled or run here. The east/west launches are uncoalesced across lines and 1-line domains
are latency bound: no performance claim, root profiles.

`splash_redistribute` result (`SplashResult`): `deposition_kg` (total), `local_deposition_kg`,
`ray_bin_deposition_kg`, `obstruction_tail_deposition_kg`, `export_edge_cell_kg`, `export_source_cell_kg`
(all `(ny, nx, nc)`); `x_face_gross_kg`/`x_face_net_kg` `(ny, nx+1, nc)` (index c = face west of column c),
`y_face_gross_kg`/`y_face_net_kg` `(ny+1, nx, nc)` (index r = face south of row r), net `+` east/north, gross `>= 0`,
export crossings included (the `sediment_transport` convention); `.face_flux()` builds MAPLE's `FaceFluxResult`.

Closure (enforced before returning, `SplashError` otherwise; bounds from MAPLE's `summation_error_bound_kg`
with a declared 256 operations plus pairwise terms): per class `pickup = deposition + export_edge`; per cell and class
`deposition - pickup = net inflow through the four faces`. `source_export_residual_kg` (source-attributed vs
edge-attributed export, an independent formula) is returned with its tolerance but not enforced.

Validation precedes any output: types, namespace, shape, `float64`, finite, `>= 0`, zero on inactive cells,
`spd*dx` finite; inputs are never written.

## Dry demand (`sediment_physics.dry_raindrop_pickup_demand`)

Evaluates the *same* raindrop law as the wet step (energy model, vegetation form, Quansah, composition options,
phi-2 `hs` cap, 1 s reference rate times `dt`) via the shared `_raindrop_law`, on cells with
`active & depth == 0 & rain > 0` only. No depth is faked: the attenuation is `exp(0) = 1`. Wet cells are excluded
(the wet step already picks them up), so nothing is counted twice. Output is *demand*; MAPLE caps it.

## MAPLE helper (`apply_splash_exchange`, not event integration)

1. `apply_bed_demand(WaterProcessDemand(requested, 0))`: MAPLE caps by availability and holdings and moves the
   actual pickup into the mobile pool.
2. `splash_redistribute(actual_removal_by_cell_class_kg)`: the *actual* pickup, never the demand.
3. Mobile pool published as `preexisting mobile + deposition + export_edge` (the splash mass is the only addition),
   depth unchanged; `apply_bed_demand(WaterProcessDemand(0, deposition, face_flux, boundary_export))`.

Soil, depth, availability and ledger authority stay MAPLE's. Pickup requested on a wet or inactive cell, or a
class-count mismatch, is refused. `SplashExchangeAudit` carries per-class requested/actual pickup, shortfalls,
splash request vs native deposit/export, mobile change (must be ~0: preexisting water-mobile sediment is not
moved), MAPLE's measured net bed change and a conservation residual; `scalar_totals()` gives host scalar sums.
The deposition destination may be a wet cell (mass lands on the bed there); that is not a pickup.
The CuPy device/native-bed path is exercised by the integrated tests. The coupled `PondEvent` currently refuses windowed beds; this helper does not establish windowed storm qualification.

## Wet-physics preservation (`sediment_physics.py`)

The splash refactor: (1) lines of the wet step that computed the class fractions and the raindrop depth rate were
moved *verbatim and in the same order* into `_class_fractions` and `_raindrop_law`, which the step now calls with
the same arguments (`raindrop_cells` is the same mask, `d3` the same depth); every floating-point operation, its
operand order and its dtype are unchanged, so default results should be bitwise identical; (2) new public
`dry_raindrop_pickup_demand` / `DryRaindropDemand`. Coupled pond integration additionally adds optional `shear_slope` and `stream_power_slope` keywords, defaulting to the original terrain-slope arithmetic. Outputs and existing validation/checks are retained. Default wet-law bitwise equality was independently verified by Codex; a test
checks the wet step against the dry law times the legacy depth attenuation to 1e-12.
