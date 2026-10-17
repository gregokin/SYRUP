# Direct splash and depression overflow

This opt-in water event adds direct rainfall splash and hydraulic filling/overflow of real flats and depressions over **MAPLE's actual evolving sediment bed**. Select `maple_syrup.pond_event.PondEvent` explicitly. Existing method-5 events, legacy replay, command-line defaults and their checkpoints retain their previous behavior.

The new event couples rainfall/infiltration, local-inertial face flows, MAHLERAN-derived wet detachment and splash, grain-specific sediment speed/distance, native supply-limited pickup, deposition, and boundary export. It supports NumPy, Numba hydraulics, and CUDA hydraulics/splash with CuPy-resident sediment exchange. No wind physics is copied; wind scheduling belongs to MASY.

## What changes physically

* Water moves according to free-surface gradients through D4 faces. Depressions retain water, adjacent wet surfaces can level, and flow can cross a real lip once the water surface overtops it. The terrain is neither filled nor tilted to manufacture drainage. Static escape levels are diagnostics and completion gates, **not** the hydraulic solver.
* This is an alternative to MAHLERAN's ordered method-5 sweep, not a bit-for-bit port of method 6. It uses local inertia with implicit quadratic Darcy–Weisbach friction; the surface-pressure gradient remains explicit. The old candidate's lagged-friction default remains unchanged; `PondEventControl` selects implicit friction for this new event. See [friction](implicit_friction.md).
* Direct dry-cell splash retains MAHLERAN's exponential distance bins, slope/aspect weights, class coefficients, and raindrop relationship. It uses actual available MAPLE material. Open faces export; closed/masked ray ends retain the remaining tail. Legacy tail truncation and nonconservative extra edge deposits are not reproduced. Explicit class-specific inverse distances are required.
* Wet detachment keeps the existing relationships, with separate topographic rain slope and configurable friction-based shear/stream-power slopes. Rain/splash slope and aspect come from the steepest strictly downward D4 neighbor on the held real bed, rather than a manufactured drainage graph. This separation matters in pond/backwater flow. It is a documented physical change from assigning terrain slope to every process.
* Multi-face sediment transport preserves separate virtual sediment speed and exponential travel distance. Survivors split among actual outward water-volume faces; changing direction uses scalar progress within a cell. This is the conservative characteristic/bin formulation, explicitly selected here, rather than the default unlimited-supply legacy replay.

## Using the event

Create a native `BedState` and `BedContext` using the existing MAPLE/SYRUP import and construction interfaces. Supply the usual rainfall schedule, infiltration columns and sediment parameters. Build `PondTopology` from the same physical surface, active mask, friction, cell size and **explicit** `OutletFace` entries. Then:

```python
from maple_syrup.pond_event import PondEvent, PondEventControl
from maple_syrup.splash import splash_parameters

# All supplied arrays must use the selected backend; there is no silent fallback.
event = PondEvent(
    context, column, sediment, schedule, topology,
    control=PondEventControl(backend="numba"),
    splash=splash_parameters(inverse_distance_per_m, xp=xp),
)
state = event.initial_state(bed, initial_soil_water_m)
result = event.evolve(state, end_s)
state = result.state
audit = event.audit(state)
```

Use `backend="cuda"` with device arrays and `splash_implementation="cuda"` in the control for compiled GPU splash. Numba compiles hydraulics; the multi-face sediment operator currently uses NumPy/CuPy array operations. Omitting `splash` disables direct splash. `evolve` reports incomplete bounded integrations; it never silently dries them. See [event API and completion](pond_event.md) and the [restart API](pond_checkpoint.md). Checkpoints bind the exact imported SYRUP/MAPLE content pair and forcing/configuration; keep that pair to resume.

MAPLE remains the only bed authority. Native accounting commits and hydraulic geometry refreshes are separate. Refresh occurs only when the maximum active-cell absolute **net elevation change since the last actual build** reaches MAPLE's trigger (normally `>= 0.00075 m`). Time, reporting, turnover, checkpoints and final accounting commits cannot trigger it by themselves.

## Current qualification and limits

See [qualification](qualification.md) for actual CPU/GPU tests, original Fortran equation comparisons, the native 32×32 Chastre test, exact fresh-process restarts, source identities and timing limits. That Chastre test includes rain-off recession but ends with conserved water and mobile sediment; it is not a completed catchment storm.

This first integrated version supports **dense beds with avalanching disabled and square cells**. Windowed beds are explicitly refused. Outlets require a real positive exterior bed drop and use the existing normal-flow closure; stage-controlled, zero-drop, uphill and backwater boundary conditions need further work. GPU correctness is qualified on the small tests; a fused multi-face transport kernel and windowed/sparse large-domain performance are follow-ups. Do not infer full Chastre memory fit or speedup from this subset.

Still water with no applicable wet transport law retains its mobile load. There is no separate time-based pond settling model. A caller may explicitly select conservative flow-end deposition after the hydraulic quiet hold; otherwise significant mobile load prevents completion. The subsequent dry-again assumption records both surface and soil water removed and conserves sediment. Splash/wet exchange is first-order split: wet-demand composition is sampled at step start, so splash-deposited changes enter that composition on the next step; actual pickup always uses current native holdings. The dry/wet switch uses post-hydraulic depth, whereas MAHLERAN's reference switch uses pre-hydrology depth. The original slope law gives zero rain detachment on an exact flat; no slope floor is inserted.

Vegetation growth, full ecohydrology/ET, nutrients, tracers, wind scheduling, and universal MAHLERAN output identity remain outside this change.

Component details: [splash](splash.md), [topology and hydraulics](pond_design.md), [sediment transport](pond_transport.md), [coupled event](pond_event.md), [implicit friction](implicit_friction.md), [checkpoint](pond_checkpoint.md).
