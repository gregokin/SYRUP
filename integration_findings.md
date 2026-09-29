# MAHLERAN–MAPLE integration assessment

2026-09-28. Initial source assessment with independent Claude review incorporated. Neither model's source was modified.

**Recommendation:** use MAPLE as the host for a single sediment bed, composition, availability, accounting and event scheduler. Reuse MAHLERAN's water-process routines where a small wrapper can preserve their behavior, but first prove a water-only adapter. Alternating complete executables with only DEM exchange would lose essential sediment and hydrological history. The existing code provides substantial infrastructure, but a working combined model is not just a configuration change.

Sources: `/home/okin/MAHLERAN`, HEAD `305bd95d32123f13708be2f9a88e42ddd45d6f28` (clean when inspected); `/home/okin/MAPLE`, HEAD `74dd3e79d9a7df85244355439e3a4a762abb9273`, including existing uncommitted changes. This is not a review or acceptance of those changes. Detailed independent review: [Claude review](agent_handoffs/claude_review.md). The pre-review draft is retained in `agent_handoffs/findings_before_claude_review.md`.

## What can already be reused

| Component | Existing code | Integration implications |
|---|---|---|
| Mutually exclusive wind/water blocks | MAPLE `coupling/water_event.py`, `aeolian/scheduler/fluvial.py` | Reuse dispatch and shared time/accounting infrastructure. |
| One bed, active layer and available sediment | MAPLE `surface/active_layer/exchange.py`, `surface/availability/exchange.py`, voxel operations | Both processes must use actual permitted transfers and the resulting composition. |
| Water mass accounting and conversions | MAPLE `water/{interfaces,step,mahleran_compat,validation}.py` | Useful interfaces; they do not implement hydrology or lateral mobile-sediment movement. |
| Topographic commit and water-depth callback | MAPLE `surface/topographic_commit/commit.py`, `water/commit_callback.py` | Reuse transactional callbacks; add hydraulic geometry refresh and decide who owns the water budget. |
| Hydrology, entrainment, deposition-distance laws | MAHLERAN `Subroutines_Water/` and `Subroutines_Sediment/` | Candidate Fortran routines; isolate from driver, duplicate bed updates and legacy budget errors. |
| Output, snapshots, restart and conservation tests | MAPLE existing infrastructure | Extend for persistent hydrology and verify transitions; current public alternation is separate event runs joined through state/restart. |

## Main problems

### 1. Lateral transport is missing from MAPLE's current water interface — high priority

`water/step.py:271–284` adds erosion to the local mobile pool and caps deposition against that same pool. Supplied face flux is reported, not applied as flux divergence. A two-cell diagnostic confirms that 0.01 kg removed in A remains mobile in A when deposition is requested in B; B deposits nothing.

The important legacy distinction is that **`flow_distrib` supplies downstream deposition demands; it does not advect mobile sediment**. Wet sediment moves through `accumulate_flow.for` and the Euler/Crank–Nicolson continuity calculations in `route_sediment_xml.f90:215–305`, with `q_soil = d_soil*v_soil`. Splash is a separate direct redistribution mechanism.

Needed: a conservative mobile-pool transport step, separate handling of splash, and a declared order/integration scheme for pickup, advection and deposition. Reuse bed exchange and face-flux representations. Validate the per-cell, per-class identity:

`change in mobile mass = actual pickup − actual deposition + face inflow − face outflow + external exchange`.

Domain-total conservation alone cannot detect spatially wrong transfers. This missing capability is consistent with Phase 19's limited scope, not a newly introduced MAPLE defect.

### 2. Timestep units and detachment physics need a decision before coupling — high priority

`flow_detachment.for:27–36` divides pickup and its cap by `dt`. For fixed local conditions, multiplying the resulting rate by dt gives a fixed pickup per step; reducing dt can therefore increase erosion per unit simulated time. `rate_times_dt` conversion alone does not remove that dependence.

The wet sediment continuity calculation multiplies detachment/deposition by dt, but bed `z_change` at `route_sediment_xml.f90:348` does not. Conversely, `splash_transport.for:25` already multiplies detachment by dt. A combined legacy field can therefore mix wet rates and splash depths. One global unit-conversion switch cannot safely interpret every regime.

Declare whether the initial target is legacy-output reproduction or corrected timestep-consistent behavior. Test multiple dt values and wet/dry interfaces. Keep particle density (mass conversion) separate from MAPLE's bulk density (bed elevation); equal transported mass need not imply equal elevation change.

### 3. A shared bed must control both supply and changing composition — high priority

MAHLERAN's flow pickup uses `sed_propn*hs/dt`. The composition-renormalization code in `update_sediment_flow.for` is commented out. MAPLE changes active-layer composition, availability and exposed subsurface material after transport.

Feed the water physics from the current shared bed. Apply actual supply-limited pickup before trusting downstream transport or deposition. Passing independently calculated legacy bed changes into MAPLE would create conflicting inventories. Keep gross per-class removal and deposition separate; net `z_change` loses grain identity and turnover.

### 4. Switching events does not finish transport — high priority

MAPLE `aeolian/scheduler/state_validation.py:validate_event_boundary` refuses water entry while airborne material remains, and wind entry while water-mobile mass remains. These are exact emptiness requirements, including sub-resolution mass.

Legacy Euler sediment routing zeros mobile depth in dry cells; its continuous initialization resets water depth but does not equivalently resolve all sediment in transport. These behaviors are unsuitable as a conservative handoff. A settling/runoff tail, or an explicit terminal-deposition rule, is required before switching. Claude suggested depositing the remaining local pool through MAPLE's existing deposition machinery: that is a possible implementation mechanism, **not an already-validated physical settling law**.

Standing water and soil moisture also need a policy for wind entrainment. An empty sediment suspension does not mean the surface is dry.

### 5. Persistent hydrology and restart state are missing — high priority

MAPLE `WaterState` currently contains only depth and mobile sediment mass. MAHLERAN also carries water/sediment discharge and time levels, velocities, infiltration history, soil moisture and routing state; marker mode adds particles and RNG state. Some fields can be derived, others must persist.

Define the minimum authoritative hydraulic state and checkpoint it. Do not restart infiltration or reset moisture at every event change. The public MAPLE interface has a case-level event kind, requires a calm wind record for fluvial runs, and rejects coupled saltation configuration in a fluvial event. Its existing lower-level dispatch is reusable; a continuous mixed-event driver or carefully verified restart orchestration is still needed. Restart equivalence is a prerequisite for claiming working alternation.

### 6. Geometry refresh, legacy lifecycle and water budgets conflict — high priority

MAHLERAN `update_top_surface` changes elevation, subtracts bed change from water depth and refreshes drainage. MAPLE also owns topographic commits and a water-depth adjustment. Running both would duplicate these operations. Constant-free-surface adjustment alone is not a conservative water solver.

Two source-supported legacy hazards deserve isolated reproduction:

- At storm end `MAHLERAN_storm_xml.f90:190–195` overwrites `z_change` with total storm deposition minus detachment. The interstorm initialization branch does not reset it. A subsequent topographic update can reapply already committed change.
- `update_top_surface.for:43–48` chooses the lowest neighbor without the nodata exclusion used in `topog_attrib.for:104–109`; it can route toward nodata elevations after an update.

Use one bed-update authority and a tested geometry callback. `topog_attrib` is a candidate for reuse, but repeated allocation and its full side effects must be handled; it is not yet a proven drop-in callback. Refresh hydraulic routing and wind-derived geometry after the relevant commits, with rollback-safe publication.

### 7. Legacy floors and boundaries can break conservation — high priority

Both legacy sediment schemes floor negative mobile depth, while bed deposition still uses the unclipped deposition request. Under excessive deposition this can add bed mass without the corresponding mobile mass. Dry or masked cells also have sediment-zeroing paths.

For wet transport, an unassigned `flow_distrib` tail is **not by itself proof of lost mass**: unresolved deposition can leave sediment mobile for advection. Splash, interior sinks, masked cells, physical outlets and numerical truncation require separate budgets. Do not label every un-deposited tail as boundary export. The existing MAPLE adapter's comments overstate recovery of these losses; the actual gross-field converter only returns removal/deposition demands, and the separate face converter detects edge-directed discharge.

Limit transport/deposition by actual available reservoirs and record true external exports. Preserve the chosen scheme's time levels and integrated flux convention. Conservative changes can legitimately differ from legacy outputs, but must be documented.

### 8. Transport distance is not one interchangeable kernel — medium/high priority

MAHLERAN concentrated and suspended transport multiply a nominal mean distance by 0.693, then `flow_distrib` uses its reciprocal as an exponential rate. Diffuse flow and splash use different conventions. The intended calibration needs resolution before changing these formulas. Source inspection establishes the arithmetic mismatch, not what the original empirical literature intended.

MAPLE's wind code uses lognormal hop distributions and its own trajectory physics. Reuse interfaces and accounting first; sharing a transport kernel would require an explicit scientific equivalence argument.

### 9. Grid, class and version contracts need to be pinned — medium priority

The inspected XML driver sets `dy=dx`: legacy grids are square. Its outer DEM ring serves a boundary role and is part of the supplied raster, not an independently declared halo. Rows run north-first, sediment arrays are class-first, masks matter, and routing uses four directions. MAPLE uses south-first rows, class-last arrays, and explicit boundary kinds, including periodic boundaries unsupported by this legacy routing.

Start with square cells, carefully mapped physical cells/outlets, and the six exact legacy grain classes required by MAPLE's `mahleran_1_2_1` converter. Arbitrary class remapping is a separate scientific task. The supplied executable source is **1.2.3**, so the adapter's 1.2.1 name/comments do not establish compatibility. Select the actual XML routing and marker settings and trace their call path.

### 10. Reusing Fortran has packaging and performance costs — unresolved feasibility

The routines depend on module globals, cell/class cursors, initialization, I/O and STOP paths. A callable CPU wrapper or isolated worker is plausible, but build and lifecycle feasibility have not been demonstrated. No gfortran was found in PATH; the existing build files reference MinGW. Keep the source trees intact and build any proof in an isolated directory.

Claude proposed calling the legacy step unmodified and harvesting fields. Treat this as a candidate, not an accepted design: legacy-derived discharges may depend on uncapped pickup, clipped mobile stores or other discarded state. The bridge must reconcile these dependencies before claiming conservative reuse.

MAPLE's water-enabled configurations also disable some dry-only commit/coasting optimizations. CPU/GPU copying and changed commit cadence could materially reduce performance. Start on CPU and measure before considering a port or GPU rewrite.

## Proposed first implementation scope

1. Pin source snapshots, one water routing scheme, square cells and six grain classes. Resolve wet/splash units and timestep intent.
2. Build a small water-only wrapper and inventory every persistent/derived field; use existing physics where practical.
3. Prove two-cell advection and splash separately, supply-limited mixed-class pickup, wet/dry transition, outlet versus mask/sink, and nonunit dt budgets.
4. Add persistent hydrology/restart and conservative event finalization; demonstrate wind→water→wind state inheritance and interrupted/resumed equivalence.
5. Only then broaden classes, boundary types, marker modes or backends.

## Verification and limits

- `check_lateral_seam.py`: passed; actual removal `[0.01,0]` kg, deposition `[0,0]`, mobile remainder `[0.01,0]`.
- Targeted MAPLE integration checks: **12 passed, 58 deselected, 2.25 seconds**, covering state/composition/availability handoff, event guards, fluvial dispatch and existing seam conservation.
- Full water integration file: interrupted after 25 passing dots and no further progress. The repository documents sandbox storage hangs, but the cause here was not established. Full-file/restart verification remains incomplete.
- No legacy build or simulation, GPU comparison, full suite, or phase acceptance claimed. Claude performed a source-only review. Legacy defects above are pre-existing source-supported hazards, not a count of defects introduced by this task.

Commands (pytest from `/home/okin/MAPLE`):
```sh
PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python /home/okin/SYRUP/check_lateral_seam.py
# Interrupted:
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORM_NAME=cpu /home/okin/MAPLE/.venv/bin/python -m pytest -q -ra -p no:cacheprovider tests/integration/test_phase19_water_coupling.py
# Passed:
timeout 60s env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORM_NAME=cpu /home/okin/MAPLE/.venv/bin/python -m pytest -q -ra -p no:cacheprovider tests/integration/test_phase19_water_coupling.py -k 'hand_off_exact_state or mixed_bed_changes or availability_continues or refuses_the_aeolian_dormancy or entry_rejects_any or invokes_no_aeolian or fluvial_seam_is_usable or preflight_rejects_a_fluvial'
```

Directly consulted legacy symbols/files: `MAHLERAN_1_2_3`, `MAHLERAN_storm_xml`, `MAHLERAN_storm_setting_xml`, `shared_data`, `initialize_values_xml`, `flow_detachment`, `diffuse_flow_transport`, `conc_flow_transport`, `suspended_transport`, `flow_distrib`, `splash_transport`, `route_sediment_xml`, `update_sediment_flow`, `update_top_surface`, `topog_attrib`, `infilt`, `route_water`, `accumulate_flow`, `update_water_flow`, and build files. Claude's review includes additional source citations and explicitly separates its unresolved hypotheses. No literature validation was attempted.
