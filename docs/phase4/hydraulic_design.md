# Phase 4a — hydraulic routing design (source-grounded; not implemented)

Status: design proposal for Codex review, 2026-09-29, task `phase4a_hydraulic_design`, baseline commit `ddf13c1`. No code was written or run. Line numbers are as read on this date from `/home/okin/MAHLERAN` (1.2.3) and `/home/okin/SYRUP`. The evidence is source reading only. There is no Fortran execution, no GPU claim and no new numerical result.

**Corrections after Codex review (Phase 4b, 2026-09-29).** The user-selected baseline keeps the legacy method-5 **bisection**. It is applied on the proven bracket `[0, R]` in depth, with a fixed iteration count. The Newton solver of §2.3 is only a documented alternative, not the baseline. Other corrections:

- There are **10 selected outlets** on the south edge. The 20 south-edge cells with a lower exterior neighbour are only candidates.
- The graph is built from the sidecar source DEM with no datum offset.
- The edge rule runs in the legacy in-place loop order.
- A carried `q(1)` is supported and checked, not only recomputed.
- `shared_data` holds `dt` and `dx` as default REAL(4).

The implemented contract is `docs/phase4/routing.md`. Where it differs from this document, it supersedes it.

## 1. What the selected legacy method actually does

Plot 1 root XML (`mahleran_input.xml` 66–77) selects:

- `flow_direction 4`;
- `flow-routing_solution_method 5`;
- `friction_factor_type 1`, mean 21.45 for surface type 1, std 0, deterministic, no friction map.

Storm setup 563–568 calls `calculate_surface_properties_from_types` with distribution 0. That routine (104–105) assigns the mean, so ff = 21.45 in every cell and stays **static**. The main loop (`MAHLERAN_storm_xml.f90` 95–183) runs `infilt → accumulate_flow → route_water → sediment → … → update_water_flow` with dt = 1 s.

### 1.1 Topography (`topog_attrib.for`; the `topog_attribute.for` variant is not called, per its header 1–4, and storm_setting 814 calls `topog_attrib`)

- **Grid and units.** Elevations are converted m → mm (storm_setting 183–189) and dx to mm (193–197). The computed interior is `i = 2..nr`, `k = 2..nc` with `nr = n_rows − 1` (204–205).
- **Aspect** (94–117). `sdir` = N(−1,0), E(0,+1), S(+1,0), W(0,−1). The receiver is the lowest D4 neighbour with strict `<`, skipping `nodata`. Aspect 0 is a sink or flat.
- **Slope** (117). `slope = (z − zmin)/dx` is the steepest D4 descent to that same receiver, in mm/mm = m/m, and 0 for sinks. Slopes above 1000 become 1 (218–224; unreachable for Plot 1). Masked cells get slope 0 (190–197).
- **Edge rule** (265–293). If a cell's receiver has `rmask < 0` and the cell's slope exceeds the slope of the opposite neighbour (cell − step) or is 0, the cell takes that neighbour's slope.
- **Order** (232–303). Active (`rmask ≥ 0`) cells are sorted ascending by `contrib`, the count of cells whose descent path passes through the cell (`Functions.for` 216ff, a Numerical Recipes quicksort). Every receiver has a strictly larger contrib than its donors, so this is a valid topological order.

### 1.2 Hydraulics, `iroute = 5` (`route_water.for` 527–925), mm and s

- Constants: `gfconst = 78480 mm/s² = 8g`, with g = 9.81 m/s² (21); `tol = 1e-8` mm (22).
- **Precision.** `shared_data.f90` has no `implicit` statement. `dt`, `dx` and `dtdx`, declared in its common blocks, are therefore default REAL(4). route_water also evaluates `1./dt` in single precision (857–861). Exact comparisons need `dt` and `dx` (mm) that are exact in binary32. dx = 500 mm and dt = 2^k s qualify.
- **Inflow** (535–568). `sdirin` (20) visits donors in the order (i+1, j), (i, j−1), (i−1, j), (i, j+1), which is (S, W, N, E) of the receiver. `qin(2,i,j) = Σ q(2, donor)` over donors whose aspect points at the cell and whose `rmask ≥ 0`. A negative donor q causes a STOP.
- **Right-hand side** (572–574):
  `C = d(1)/dt + qin(2)/(2dx) + excess − (q(1) − qin(1))/(2dx)`.
- **Root** (857–862, static-ff branch 842–855). Solve `d/dt + d·v(d)/(2dx) = C`, with `v = sqrt(8g d S/ff)` and `q = d·v = sqrt(8gS/ff)·d^{3/2}` in mm²/s.
- **Brackets** (579–624):
  - `C > 0`: `[0, 100(d(1)+excess)]`, or `[0, 0.5 mm]` if that is 0.
  - `C = 0`: `[0, 2(d(1)+excess)]`.
  - `C < 0`: STOP (602–613).
- **Bisection** (631–889). It stops when `dhigh − dlow ≤ 1e-8`. More than 10001 iterations causes a STOP.
- **Result** (890, 921–923). `d(2) = dmid`, then `v`, `q(2) = d(2)·v`.
- **Old-level interaction with infiltration** (`infilt.for`):
  - complete run-on (106–116): `d(1) = q(1) = v = 0`;
  - rain excess (125–131): `excess = r − f`, with d(1) and q(1) untouched;
  - partial run-on (141–157): `d(1) −= (f − r)dt`, and **`q(1)` is recomputed from the reduced `d(1)`** (154–156).
- **Level shift** (`update_water_flow.for` 30–41): `d(1) = d(2)`, `q(1) = q(2)`, `qin(1) = qin(2)`.
- **Not used by iroute 5.** `accumulate_flow`'s `add` feeds only iroute 1 (route_water 59–60) and chemistry. `downslope_vars` has no caller in `src/`.
- **Outlet hydrograph** (`output_hydro_data_xml.f90` 131–138). `q_plot = Σ q(2)` over active cells whose receiver has `rmask < 0`. It is reported as `q_plot·dx` (409), an instantaneous new-level value, not a step-integrated volume.

### 1.3 Legacy conservation defects and hazards (from reading the code; frequency and magnitude not measured)

1. **Sender/receiver old-flux mismatch.** After partial or complete run-on, the sender's CN old outflow uses the recomputed `q(1)` (smaller, or 0). The receiver's old inflow `qin(1)` still holds the unmodified previous `q(2)` of that donor. The receiver therefore gains `dt/(2dx)·(q_old − q(1)')` that the sender never lost, so water is **created** on every run-on infiltration step of a flowing cell.
2. **Bracket truncation.** The upper bracket mixes a depth and a rate (`d(1)+excess`) and is not proven to contain the root. A cell with complete run-on on the previous step has `d(1) = excess = 0`, so `dhigh = 0.5 mm`. If upstream inflow drives the root above that value, every `fmid < 0`, and bisection converges silently to `dhigh`. The step then loses `C·dt − dhigh − dt·q(dhigh)/(2dx)`. With dx = 500 mm and dt = 1 s this needs an inflow above about 5e-4 m²/s, which is of the order of `e·L` for a 30 m plot at tens of mm/h. It is plausible on concentrated paths below high-intake cells. Unmeasured.
3. **Root/storage closure.** `d(2) = dmid` satisfies the balance only to within the bracket tolerance, 1e-8 mm per cell per step. The error is small but not a conservation identity.
4. **Unmasked edge receivers.** If a cell's receiver is a ring cell with `rmask ≥ 0`, its outflow leaves the domain but is not counted in `q_plot`: a silent loss. Plot 1 has no such cell (Phase 2 audit: all 1200 cells reach the masked south ring).
5. **Negative RHS.** iroute 5 STOPs. The unselected iroute 2 and 7 replace `|C| < 1e-20` by `d = 0` and clamp `dnew < 0 → 0` (271–286, 484–486). None of this is ported.
6. **Sinks.** Aspect 0 means slope 0 and q = 0, so the cell stores water forever. This is conservative but never spills; overtopping exists only in iroute 6, which its own comment (930–931) calls unstable.

## 2. Recommended formulation

### 2.1 Equations (SI; per active cell i with a single D4 receiver r(i) or an outlet)

- `k_i = sqrt(8 g S_i / f_i)`, with g = 9.81 m/s², `f_i = 21.45` and `S_i` the legacy steepest-descent slope (§3).
- `q = k h^{3/2}` in m²/s (unit width); `v = q/h`.
- This is exactly the legacy law after scaling: `q_mm = 1e6 q_m`, `v_mm = 1e3 v_m`, `gfconst = 8·9810`.

Per step of length dt (`c = dt/(2 dx)`):

1. **Column step (Phase 3, unchanged).** `column_step(params, h^n, S^n, r, dt)` returns `h* = depth_m` (includes rain excess and saturation return) and `O = saturation_return_m`.
2. **Legacy post-infiltration depth.** `d1'_i = min(h^n_i, h*_i − O_i)`, where `h* − O = A − J`. Case by case this is exactly the legacy `d(1)` after `infilt`: complete run-on gives 0, rain excess gives `h^n`, partial run-on gives `h^n − (f−r)dt`. Also `h* = d1' + excess·dt`. Rain and infiltration are therefore **not subtracted twice**: routing adds no rain and no infiltration.
3. **Explicit (old) face flux.** `q^n_i = k_i (d1'_i)^{3/2}`. It is **used identically by the sender (outflow) and the receiver (old inflow)**.
4. **Ordered implicit solve**, upstream first:
   - `Qin^n_i = Σ_{donors u} q^n_u` (all cells at once);
   - `Qin^{n+1}_i = Σ_u q^{n+1}_u` (donors already solved);
   - `R_i = h*_i + c (Qin^n_i + Qin^{n+1}_i − q^n_i)`;
   - solve `h + c k_i h^{3/2} = R_i` (§2.3), giving `q^{n+1}_i`;
   - **storage by identity:** `h^{n+1}_i = R_i − c q^{n+1}_i`.
5. **Face volumes.** `F_i = (dt dx / 2)(q^n_i + q^{n+1}_i)` [m³], added to r(i), or to export if i is an outlet cell.

Discrete identities, exact up to FP64 rounding by construction:

- Cell: `dx² (h^{n+1}_i − h*_i) = Σ_{u→i} F_u − F_i`.
- Domain: `Σ dx² h^{n+1} = Σ dx² h* − Σ_{outlets} F_i`.
- Event: `surface_0 + soil_0 + rain = surface_end + soil_end + drainage + export`.

Internal transfers cancel pairwise because every face flux is a single number with a single owner.

This is the legacy iroute-5 equation term for term, except:

- (a) the receiver's old inflow uses the same `q^n` as the sender (fixes §1.3-1);
- (b) the root is closed through the storage identity (fixes §1.3-3);
- (c) the bracket and solver are provably valid (fixes §1.3-2);
- (d) rain excess enters through `h*` instead of a rate term. This is algebraically identical, because `h* = d1' + excess·dt`, plus the explicit saturation-return term that Phase 3 already accepted.

Where no upstream cell had run-on infiltration in a step, (a) changes nothing. There `R` equals the legacy `C·dt` up to root tolerance and roundoff, because the legacy `q(1) = k d(2)^{3/2}` of the previous step. The saturation-return case also matches: legacy adds overflow to `excess` and leaves `d(1)` unchanged, while here `d1'` excludes `O` and `h*` includes it.

### 2.2 Positivity, stability and time step

- Since `h* ≥ d1'` and inflows are ≥ 0, `R_i ≥ d1'_i (1 − Cr_i/2)`, with `Cr_i = v(d1'_i)·dt/dx`. **Hence `max Cr ≤ 2` guarantees `R ≥ 0`**, and negative right-hand sides cannot occur.
- Default `Cr_max = 1`, for accuracy margin.
- Step plan: `dt ≤ 1 s` (legacy dt), split at every rainfall knot (Phase 3 `_substeps`). A violated `Cr_max` rejects the pure step and retries with dt/2. Nothing is committed on rejection.
- There is no clipping, no STOP and no flooring, and water is never deleted. A rejection budget of at most k halvings, then a raised error, prevents an endless loop.
- Plot 1 expectation, not a measurement: with `h ≈ 3.5 mm`, `S ≈ 0.05` and `f = 21.45`, `v ≈ 0.025 m/s` and `Cr ≈ 0.05` at dt = 1 s. Rejections should not occur.
- The time discretization is Crank–Nicolson-type (second order in dt for smooth flow). The spatial discretization is first-order upwind finite volume on the D4 tree, the same as legacy. The kinematic wave (bed slope, no backwater, no adverse slope) is a retained legacy physical limitation.
- Wetting and drying: there is no depth threshold. Any positive `h` flows, `h = 0` gives `q = 0`, and a dry cell receiving inflow wets in the same sweep.

### 2.3 Root solver (per cell, vectorizable)

**Baseline as implemented (Phase 4b): bisection.**

- Bracket `[0, R]` in depth. `g(h) = h + c k h^{3/2} − R` is increasing, with `g(0) = −R ≤ 0` and `g(R) ≥ 0`.
- A fixed number of halvings is used, 40 by default, giving a width of `R·2^-40`. There is no per-level host synchronization.
- `lo` is kept only where the computed `lo + c q(lo) < R`, so `h_new = R − c q(lo) ≥ lo ≥ 0` exactly in floating point. There is no clipping.
- The constitutive residual `h_new − lo` is checked against 1e-11 m, the legacy 1e-8 mm. A failure raises; it is never hidden.

See `docs/phase4/routing.md` §3.

**Alternative, not the baseline:** Newton, as proposed originally:

- With `u = sqrt(h)` and `a = c k ≥ 0`, solve `φ(u) = a u³ + u² − R = 0`.
- φ is strictly increasing and convex on u > 0, so the root is unique and lies in `[0, sqrt(R)]`.
- Start value: `u₀ = min(sqrt(R), cbrt(R/a))`. Both terms are upper bounds, so `φ(u₀) ≥ 0`, and Newton on a convex increasing function then decreases monotonically to the root. At the worst case, where the two terms balance, `u₀ ≤ 2^{1/3} u*`, giving about 5–6 iterations to FP64. Use a **fixed `N = 8` iterations** (branch-free, and no per-level host synchronization on GPU).
- Special cases: `R == 0` gives `u = 0`, handled by `where`, because the Newton derivative is 0 at u = 0. `a == 0` is rejected at graph build (§3).
- Closure: `q^{n+1} = min(k u³, R/c)`, then `h^{n+1} = R − c q^{n+1} ≥ 0`. The cap only guards rounding and is conservative, because storage and export use the same `q`.
- Deferred validation flag: constitutive residual `|h^{n+1} − u²| ≤ 1e-12 · max(R, tiny)`. A failure rejects the step. Mass conservation does not depend on this tolerance.

## 3. Graph, outlets, masks

Build once per event on the host, from committed MAPLE bed elevation. This is a deterministic function of the terrain, mask, boundary spec and dx, so restart persists only its input hash.

- **Orientation.** MAPLE `(r, c)`, row 0 = south, flat index `r·nx + c`. Legacy aspect 1 = N = `r+1`, 3 = S = `r−1`, 2 = E = `c+1`, 4 = W = `c−1`. Legacy `(i, k) = (61 − r, c + 2)` (docs/phase2/plot1_import.md §4).
- **Receiver.** Use the legacy strict-`<` lowest D4 neighbour in N, E, S, W order over interior cells plus a **boundary elevation ring**. *As implemented:* the source DEM is taken from the sidecar 62×22 `legacy_full_elevation_m`, which is already south-first. There is no datum offset, and the comparison is in legacy mm. Aspect must equal Phase 2 `legacy_d4_aspect`.
- **Slope.** `(z_i − z_receiver)/dx`, then the legacy edge rule for outlet cells. *As implemented:* the rule runs in the legacy in-place loop order. An edge rule that would read a ring or inactive slope is rejected.
- **Outlet policy.** An active cell whose selected receiver is an export-flagged cell (legacy `rmask < 0`) is an **outlet**. It exports `F_i` and appears in the hydrograph. For Plot 1 there are **10** such cells, per Codex and verified by tests when they are run. The other south-edge cells with a lower ring neighbour drain to a lower interior neighbour instead.
- **Rejections.** Graph build **rejects** each of the following, listing the cells:
  - a receiver that is an unmasked ring cell (legacy silent loss);
  - an inactive interior receiver;
  - a sink or flat (aspect 0);
  - slope ≤ 0 on a non-sink cell, possible only via the edge rule;
  - any cycle or unreachable cell (Kahn check);
  - a masked cell adjacent to flow.

  No filling, no carving, no pit storage. Pit, flat and overtopping support is outside Phase 4. Plot 1 has none (Phase 2 acceptance).
- **Inactive cells** (`active_mask` false) neither send nor receive. Rain on them is already rejected by `column_step`.
- **Stored arrays**, on the MAPLE backend `xp` after one host→device copy:
  - `receiver` int32 (N; −1 = outlet);
  - `donor_slot` int32 (4, N): the neighbour flat index in legacy `sdirin` order (S, W, N, E) or a pad index N that points to a zero entry;
  - `is_donor` bool (4, N);
  - `k` float64 (N);
  - `outlet` bool (N);
  - `level_order` int32 (N_active): cells sorted by dependency level;
  - `level_bounds`: host Python ints;
  - `level(i) = 0` without donors, else `1 + max level(donor)`.

## 4. Ordering dependency and execution strategy

`q^{n+1}` of a cell depends on its donors' `q^{n+1}`. This is a genuine sequential dependency along flow paths. Dependency levels expose all the available parallelism: cells within one level are independent. The level sweep computes exactly the legacy sequential semantics, because any topological order gives the same values. The only difference is summation order, which is fixed by the gather.

- **Production baseline (Phase 4):** a level-synchronous vectorized sweep in `xp` (NumPy/CuPy).
  - `Qin^n` and `R`'s explicit part are computed for all cells in one pass.
  - Then, for each level: gather `Qin^{n+1}` over the 4 fixed donor slots (**gather, not scatter/atomics**, so the result is deterministic and backend-identical), run bisection with a fixed iteration count (as implemented; Newton was the original proposal), then write `q^{n+1}` and `h^{n+1}` into the level's contiguous slice of level-ordered arrays.
  - There is a Python loop over levels, not cells, and no host sync inside it.
- **Test oracle:** a scalar sequential CPU implementation in the legacy ascending-contrib order, with the same gather order. Its results must agree bitwise or to a few ulp.
- **Costs to measure, not claimed.** There are L levels per step and about 40 small array operations per level. On NumPy, per-call overhead dominates for 1200 cells. On CuPy, about L × 40 launches per step. The Plot 1 L (longest D4 path in cells) will be reported. A long 5400 s run at dt = 1 s may be tens of seconds on NumPy; this is a known limitation of the portable baseline.
- **Later compiled paths (not in Phase 4).** A compiled CPU sequential sweep (Numba or C; Numba and CuPy are absent from the MAPLE venv, per a file check of site-packages) or a single-launch CuPy RawKernel with dependency counters. Either must reproduce the oracle.
- **Synchronization.**
  - Each validated step costs one batched `DeferredChecks` flag read per kernel: column plus route. These can be merged later.
  - Flags covered: `R ≥ 0`, `Cr ≤ Cr_max`, finite and non-negative `h`/`q`, constitutive residual, per-cell balance.
  - Budgets and the hydrograph accumulate on device into a preallocated `(n_steps_chunk, n_terms)` buffer. It is read back in chunks or at the end.
  - There are no per-step grid transfers.
- **Memory.** Graph about 40 B/cell (int32 slots dominate). Workspace about 8 × 8 B/cell (`h*`, `d1'`, `q^n`, `q^{n+1}`, `R`, `u`, `Qin^n`, `Qin^{n+1}`), reused across steps. Total ≲ 110 B/cell beyond Phase 3.

## 5. State ownership, coupling, recession

- **Canonical state.** `WaterState.depth_m = h` (published through `dataclasses.replace` at a declared cadence, and before any MAPLE call). `mobile_mass_by_cell_class_kg` is untouched (zero). The bed is fixed: no commit, no terrain refresh, and a MAPLE bed digest check before and after, as in Phase 3.
- **Solver state.** In the original proposal, solver state is `(h, S, t)` only and `q^n` is recomputed from `d1'`. *Correction:* legacy keeps `q(1) = q(2)` unchanged when there is no run-on, sets it to 0 after complete run-on, and recomputes it after partial run-on. `route_step` accepts a carried `q_old` and checks it against `k h_old^{3/2}`. Whether the storm runner carries `q` (and so persists it) or recomputes it is a next-task choice. The two differ only at root tolerance. This is in-memory continuation, not production restart (Phase 6).
- **Diagnostics.** `v = q/h` (0 where h = 0) and `ff` may be reported through MAPLE `HydraulicDiagnostics` (non-canonical).
- **Run window.** Rain lasts to the end of the record (1620 s for Plot 1). Recession continues with r = 0: ponded-water intake, drainage and routing continue, with no ET, no dry reset and no sediment. The default end is the legacy `stormlength` of 5400 s (XML 16). The final surface storage is reported, and it is **not** an event completion claim (Phase 6).
- **Outputs.**
  - Per-step host series: t, dt, rain, intake, return, drainage, export volume, outlet `Q = Σ_outlets q^{n+1} dx` (m³/s; legacy-comparable `q_plot·dx`), surface and soil storage.
  - Final grids, peak depth and peak q.
  - No per-step grid archive.

## 6. Literal port versus recommended formulation

| Item | Literal iroute 5 | Recommended | Departure? |
|---|---|---|---|
| Law, friction, slope | `q = sqrt(8gS/ff) d^{3/2}`, ff = 21.45, steepest D4 slope, edge rule | same, in SI | none (scaling only) |
| Sender old outflow | `q(d1')` after infilt | same | none |
| Receiver old inflow | previous `q(2)` of the donor (pre-infiltration) | the same `q(d1')` as the sender | **yes**, fixes water creation |
| Rain excess | rate in C | inside `h*` | algebraically equal |
| Root bracket and solve | heuristic `[0, 100(d+e)]` or 0.5 mm; bisection to 1e-8 mm; STOP after 10001 iterations | as implemented: proven `[0, R]` in depth; bisection with fixed iterations (default 40); residual check (Newton only an alternative) | **yes** (bracket) |
| Storage closure | `d = dmid` (balance error = root error) | `h = R − c q` (exact identity) | **yes**, roundoff-level |
| Negative RHS | STOP | impossible for `Cr ≤ 2`; reject and halve dt | **yes** |
| dt | fixed 1 s | ≤ 1 s, rain knots, Courant halving | **yes** (Phase 3 already) |
| Outlet | masked-ring receivers; snapshot `q(2)·dx` | explicit outlet mask; CN-averaged export volume plus snapshot Q | accounting added |
| Unmasked ring receiver, sinks | silent loss; permanent storage | rejected | **yes** |
| Order | ascending contrib | dependency levels, legacy gather order | equivalent |

Recommendation: implement the right-hand column. A matched legacy comparison (when runnable) should expect differences only through run-on steps, bracket truncation, dt/rain timing and roundoff. Each of these can be isolated with the scalar legacy transcription (§7, test 3).

## 7. Smallest kernel API and tests

*Superseded by the implemented API in `docs/phase4/routing.md`. The sketch below is the original proposal.*

Module `src/maple_syrup/routing.py` (pure functions, `xp`-generic, no Python cell loops):

```python
@dataclass(frozen=True, eq=False)
class RoutingGraph:            # built on host, arrays moved once to xp
    shape: tuple[int, int]; dx_m: float
    receiver: Any; donor_slot: Any; is_donor: Any; outlet: Any; active: Any
    slope: Any; conveyance: Any          # k = sqrt(8 g S / f), m^0.5/s
    level_order: Any; level_bounds: tuple[int, ...]
    input_sha256: str; xp: ModuleType

def build_d4_graph(bed_elevation_m, boundary_elevation_m, boundary_outlet, active_mask,
                   friction_factor, dx_m, *, xp) -> RoutingGraph   # raises RoutingGraphError

@dataclass(frozen=True, eq=False)
class RouteStep:
    dt_s: float; depth_m: Any; discharge_m2_s: Any; explicit_discharge_m2_s: Any
    face_volume_m3: Any; export_m3: Any; outlet_discharge_m3_s: Any; max_courant: Any

def route_step(graph, depth_start_m, depth_old_flow_m, dt_s, *, courant_max=1.0,
               validate=True) -> RouteStep                          # pure; raises RoutingError
```

`hydrograph_experiment.py` (the Plot 1 CLI) composes `verify_plot1_case → column_step → route_step`. It reuses the Phase 3 verification, provenance, output refusal, bed digest and budget machinery.

Tests (`tests/phase4/`; all are Python equation-level evidence, not Fortran benchmarks):

1. **Root solver.** Compare with `scipy.optimize.brentq` for R and a spanning 0 to 1e±12. Check the monotone-from-above property, `R = 0`, and that the fixed 8 iterations reach the residual tolerance.
2. **Graph.**
   - Plot 1: aspect equals `legacy_d4_audit`; 1200 active cells, 0 sinks, all reaching outlets; outlet set equals the south `edge_outflow_side`; slopes equal a scalar transcription of `topog_attrib` 117 and 265–293.
   - Synthetic rejections: pit, flat, unmasked-ring receiver, inactive receiver, zero slope.
   - Nonmutation on rejection.
3. **Independent legacy transcription.** A scalar mm-unit iroute-5 bisection with legacy brackets and `infilt` `d(1)`/`q(1)` handling, on small networks.
   - Without run-on, it agrees with `route_step` to the bisection tolerance.
   - With run-on, the legacy global balance shows a nonzero creation term while ours closes. Report that term.
   - Construct and report one bracket-truncation case.
4. **Kinematic plane, analytic** (length L, uniform k, excess e, no infiltration; call `route_step` with `h* = h + e dt`):
   - rising limb `q_out = k (e t)^{3/2}` for `t < t_e = (L/(k e^{1/2}))^{2/3}`;
   - steady state `q_i = e x_i`. The discrete steady state is exact: `q_i = e·dx·i`;
   - recession from steady state: `L = q/e + 1.5 k^{2/3} q^{1/3} (t − t_r)` (solved with brentq);
   - short-pulse volume closure.
5. **Refinement.**
   - dt = 2, 1, 0.5, 0.25 s at a fixed grid: expect near second order in dt against an extrapolated reference.
   - dx halving 10 → 160 cells: expect about first order against the analytic solution (upwind).
6. **Convergence and branching.** Two planes into a channel, and a random DAG. Level sweep equals the scalar sequential oracle bitwise, or within a few ulp. Per-cell and global identities hold. Face-flux antisymmetry holds.
7. **Plot 1 coupled.**
   - 0–5400 s at `--max-dt-s 1`: budget residual within the Phase 3-style tolerance, hydrograph, and zero Courant rejections expected.
   - Sediment digest and MAPLE sources unchanged.
   - dt 2, 1, 0.5 s sensitivity of total runoff and peak Q.
   - A check that routing moves water (export > 0) against the Phase 3 no-routing ponded 0.2244 m³.
   - Runtime, levels, per-step cost, peak traced memory.
8. **Continuation and backends.** A split run from `(h, S, t)` is identical to an uninterrupted one. The CuPy parity test skips honestly without a device.

## 8. Decisions

Defaults taken, each defensible for Plot 1:

- receiver-side old-flux correction;
- bisection on `[0, R]` with storage closure (Codex/user correction; Newton only an alternative);
- `Cr_max = 1` with halving;
- legacy ring-based outlet slope and edge rule;
- reject all sinks, flats and unmasked edge receivers;
- static ff type 1 only (other ff types, which depend on Re, sediment or depth, are rejected);
- recession to 5400 s without completion claims.

No material scientific decision blocks Phase 4 for Plot 1. For later phases, record:

- pit/flat/overtopping support (legacy iroute 6 is flagged unstable in its own source);
- dynamic friction types if Phase 5 needs them;
- outlet elevation handling when Phase 5 commits change terrain (the boundary ring stays fixed; refresh is outside Phase 4).
