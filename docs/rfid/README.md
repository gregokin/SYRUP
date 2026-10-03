# RFID_2014 water-only timing benchmark (experimental)

Status: **RFID water-only timing completed.** Claude implemented the case and benchmark; Codex independently verified it and ran the original Fortran, CPU and GPU storms. See [measured results and qualifications](timing_results.md). Strict default routing and all MAPLE tolerances are preserved. Shared column arithmetic was reassociated to resolve the complete/partial infiltration rounding inconsistency, without changing the physical equations. No sediment transport, splash, erosion, evolving
terrain, wind, evapotranspiration, dry reset or restart is modelled or claimed.

## What is compared

One verified RFID_2014 case, one forcing, one routing graph, one set of column parameters, ten contenders:

| Contender | What runs |
|---|---|
| `fortran_iroute2` | ORIGINAL `infilt.for` / `route_water.for` (iroute 2, Newton-Crank-Nicolson, the native method) / `update_water_flow.for` |
| `fortran_iroute5` | the same originals with iroute 5 (bisection-Crank-Nicolson, the method SYRUP's legacy form reproduces) |
| `legacy_numba_prepared`, `legacy_cuda` | shared scheduler `storm.evolve` with the prepared Numba / CUDA hydrology |
| `explicit_{numpy,numba,cuda}` | explicit conservative D4 kinematic-wave candidate |
| `local_inertial_{numpy,numba,cuda}` | signed-face local-inertia candidate |

The Fortran driver (`benchmarks/rfid/rfid_water_driver.f90`) links the hash-pinned originals and contains no hydrology equation. It is
**not the MAHLERAN application** (no setup, sediment, output). Its timer covers the step loop (forcing assignment, `infilt`,
`route_water`, `update_water_flow` including the original's dummy six-class copy, finite checks, running totals, report rows);
parsing and outputs are outside. The SYRUP timer covers the shared scheduler loop with its validation, accumulation, 60 s reports
and the final device synchronisation. The workloads differ; report accepted/rejected steps per contender and do not read a
ratio as an algorithmic comparison.

## Geometry, mask, pits, boundary (evidence: `agent_handoffs/tasks/rfid_timing/{independent_notes.md,topology_audit.json}`)

* Files: 106 x 60 north-first ESRI ASCII, 0.1 m, one value per line, nodata -9999 (`read_legacy_grid`; `case_import.stage_legacy_ascii`
  cannot read this layout). The ring is the outer cell layer; the computed interior is 104 x 58 (legacy loops `i = 2..nr`,
  `topog_attrib.for` 44-45). MAPLE arrays are south-first.
* **5697 active** cells (rainfall-scaling map not nodata); **335 inactive** interior cells whose DEM and scaling are nodata (the
  audit requires inactive == DEM-nodata exactly). Inactive cells hold no hydrological water, receive no rain, are never a receiver
  (the legacy skip of nodata neighbours, `topog_attrib.for` 104-113) and all their faces are closed.
* **26 strict D4 sinks, no flats.** Kept as active **terminal storage** cells (`routing.PIT_STORAGE`, aspect 0, slope 0, conveyance 0,
  not an outlet). Legacy-style and explicit solvers store everything routed into them (no overtopping, no filling, carving or
  dummy drain). The native method does the same in effect (zero slope gives zero velocity). The **local-inertia candidate can
  move water out of a pit through its water-surface gradient: intentionally different physics**, not an identity.
* Depths in a conveyance-0 cell: the bisection root is the right-hand side itself, so 40 halvings leave `rhs * 2^-40`, above the
  unchanged `1e-11 m` root tolerance for depths over ~10 m (the native run ponded to 42 m). The legacy SYRUP contenders therefore
  use `bisection_iterations = 64` (an existing control; tolerance unchanged), against the Plot 1 default 40. This is an extra cost
  (more halvings on every wet cell, not just the pits) that Plot 1 timings do not carry, so RFID-vs-Plot 1 differences are not pure
  cell-count scaling. Tests show the default 40 failing loudly above ~10 m. Physical differences between contenders are the pit
  treatment (stored vs overtopped) and the outlet treatment (graph edge-rule slope vs the local-inertia bed-drop normal flow).
* Complete-infiltration old flow: `infilt.for` 112-115 sets `d(1) = 0` when infiltration takes everything. A tiny depth absorbed in
  `h + rain` could leave a roundoff residue as the old-flow depth and trip the unchanged `h_old <= h_start` guard (root reproducer
  `8.378794223761067e-76 m`); the reference, prepared Numba and CUDA column phases now set the old-flow depth to 0 in that branch.
  The column depth/soil arithmetic and every check are unchanged.
* **Fortran sample qualification.** A sample is a timing only if it exits 0 with the completion marker, reports the requested steps and
  method, has exactly `ceil(n / cadence)` history rows whose last time equals the requested end and whose rain total equals the
  common forcing integral (1e-10 relative), its final cells are exactly the active cells once with finite non-negative depth, soil
  water and discharge, and the executable, input, driver and reference hashes are unchanged across the sample. A reused build must
  match the current driver and the audited reference sources. Otherwise it is a recorded failure.
* **Boundary (a control change, identical for all methods).** The one D4 receiver on the ring with a valid elevation is north-first
  0-based (104, 23) -> (105, 23). The native rainfall scale there is **positive (0.9688)**, so the native outlet accounting
  (`output_hydro_data_xml.f90` 131-134) never flags it as an export (the native run reported zero export). The benchmark makes every
  ring cell an export receiver and sets the matched Fortran input ring `rmask` negative. The edge-rule slope of that outlet comes from
  one graph and is passed unchanged to every contender; the local-inertia candidate keeps its own documented normal-flow outlet
  (`k_b` from the bed drop to the ring cell). No internal open outlets exist; the existing refusal stays.

## Placeholder terrain (documented, invented, excluded from hydraulics)

MAPLE's bed needs a finite elevation in every cell. The 335 inactive bed cells receive the constant `min(active elevation)` in the MAPLE
bed ONLY. The original DEM with its nodata sentinel is stored in the sidecar and is what the graph uses; the placeholder never enters
a hydraulic computation. Bed: 0.1 m voxels, 0.002 m active layer, minimum fill 0.3 m, headroom 0.2 m, bulk density 1250 kg/m3,
grain particle density 2650 kg/m3, six classes at the ACTUAL supplied uniform map values [0, 0, 0, 0.092, 0.908, 0] (closed;
the XML per-type defaults are unused because `use_map_phi` is true). The compiled case has `nz` 14. The case is
compiled and reloaded by MAPLE; `case_import.check_compiled_plot1` (unchanged) validates elevation, mass, composition and MAPLE's own
validators. Recipe, XML, every staged raster, rainfall file, derived arrays, forcing record, report and MAPLE artifacts are bound by
SHA-256; `verify_rfid_case` re-audits from the sources and refuses any difference.

**MAPLE mask discrepancy (exact, retained and disclosed).** The placeholder is supplied as an already-filled prepared array, so the
compiled MAPLE masks label all 6032 interior cells `empirical_core` and none `gap_filled`, although 335 are invented. The authoritative
record of the 335 is `inactive_interior` in the sidecar. Using MAPLE's nodata plus `gap_fill` support was not adopted (it would need a
changed Plot 1 checker or a broader import rewrite). The compiled case is a water-only benchmark bed and is **not qualified for a
production wind handoff**. The compiled vertical size is `nz = 14`.

## Hydrology (sourced values; native setup bugs are not reproduced)

Native settings: infiltration model 1 / parameter type 1, routing method 2, friction type 1.

| Quantity | Value | Source / departure |
|---|---|---|
| model | `fixed_ksat` (Smith-Parlange capacity, linear drainage, saturation excess) | `infilt.for` 49-57, 80-105 |
| K | 0.028867846354842186 mm/s | exact native capture; `storm_setting` 420-426 `0.00585 + 0.000166667 rf_mean - pave`, NOT the XML 0.01 mm/s |
| suction | 23.6 mm | XML; **the native run overwrote it with 0.05 mm** (`storm_setting` 524-529 passes `psi` as the drainage target when `inf_type != 2`), not copied |
| drainage parameter | 0.05 | XML; the native run left 0, not copied |
| soil thickness | 0.21 m | XML; native FP32 value 0.209999993 not copied |
| theta_sat / theta0 | 0.36 / 0.004 | the maps (uniform) |
| friction f | 40 | XML, type 1 |

Forcing: the rate the executed native run **applied** each step (`syrup_hydro_steps.txt`): 0.03836299851536751 mm/s through 2641 s,
then 0 to 2700 s (the legacy one-second switching lag included; the parser's interval-ending reading ends rain at 2640 s). It is
compressed exactly to a two-piece schedule (`derive_applied_forcing`) and stored, hash-bound, in the case. `forcing.kind: legacy_file`
selects the parser schedule instead. The audit report states both depths. Nothing is extended or invented beyond 2700 s; running
5400 s simply adds zero-rain recession.

## Reproduction commands

```bash
# 1. case (NEW dir). The recipe already pins the actual native capture hash.
python -m maple_syrup.rfid_case --recipe cases/rfid/recipe.yaml --output-dir outputs/rfid/case \
    --applied-forcing-capture outputs/rfid/native_compat_diagnostic/Output/syrup_hydro_steps.txt --expected-maple-root <pinned MAPLE root>
# 2. tests
python -m pytest tests/rfid -q
python -m pytest tests/phase4 tests/hydraulic_candidates tests/phase4s tests/phase7h tests/phase7i -q   # regressions of the touched modules
# 3. short pilot of everything (checks first, no timing claims)
python benchmarks/rfid/run_rfid_timing.py --case-dir outputs/rfid/case --output-dir outputs/rfid/pilot --end-s 120 --rounds 1 \
    --fortran-build-dir outputs/rfid/fortran_build_pilot --allow-maple-source-change
# 4. full timing: complete untimed warm-up per contender, then 3 balanced rounds, 2700 s, max dt 1 s, 60 s reports
python benchmarks/rfid/run_rfid_timing.py --case-dir outputs/rfid/case --output-dir outputs/rfid/timing_2700 \
    --fortran-exe outputs/rfid/fortran_build_pilot/rfid_water_driver --allow-maple-source-change
```

Select the GPU with `CUDA_VISIBLE_DEVICES` before launch; serialize task tests/benchmarks on an idle GPU and record unrelated jobs on the shared machine. If an original routine
fails (a `STOP` returns status 0, so the history marker is checked) or a fixed step is unstable, the record says so; no failed run
is timed, and a different dt is chosen only by an explicit new invocation (common to all contenders, with the extra steps recorded).

## Touched shared code (small, opt-in, default behaviour preserved)

* `routing.build_routing_graph(allow_masked_nodata=False, allow_pit_storage=False)`, `PIT_STORAGE`, `RoutingGraph.pit_storage/policy`.
  With both flags off the behaviour, messages and digest are exactly the strict ones; with either on the digest includes the policy.
* `hydrology_numba.prepare_hydrology` and `routing_cuda._validate_host_graph`: accept the receiver code `PIT_STORAGE` only on active
  cells with zero conveyance that are not outlets; every other negative receiver is still refused.
* `compare_plot1.build_runner(inputs_factory=None, storm_overrides=None)`: defaults reproduce the previous behaviour.
* The `0/0` implied-depth consistency test for `q_old = 0, k = 0` evaluates to NaN and passes in all four implementations
  (NumPy, Numba `error_model="numpy"`, CUDA, `storm._validate_state`); no kernel was edited. A test documents it.

## Known limits

Case-specific; not a general landscape, not sediment, no GPU claim beyond what Codex measures. Placeholder terrain is invented
for the inactive MAPLE bed only. Native routing method 2 is only the separate Fortran contender; SYRUP implements method 5 and the
two candidates. The originals' stale-inflow and bracket behaviour is retained in the Fortran runs and is not corrected there, so their
water budget is reported but not claimed conservative. The explicit/local-inertia candidates retain their documented fidelity limits
(local-inertia velocity unqualified near drying cells, long-storm backend exceptions).
