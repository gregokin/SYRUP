# Proposed scope: MAHLERAN-like water erosion within MAPLE

2026-09-28. This is a scope proposal following the user's clarification: physical consistency with MAHLERAN is required, identical source code, numerical schemes and outputs are not. Nothing here treats earlier suggestions to defer a process as a user-approved omission. The existing feasibility tests do not implement this full scope.

## Recommended scientific target

A storm erosion and landscape-change extension that alternates with MAPLE wind transport over one evolving, class-resolved bed. Retain the selected MAHLERAN process relationships, including distinct detachment, travel distance and sediment travel speed. Implement them conservatively in MAPLE. A flow-only proof is an implementation milestone, not the complete storm erosion model.

| Full MAHLERAN capability | Proposed extension | Consequence/qualification |
|---|---|---|
| Time-varying rainfall, rainfall maps/scaling, rainfall energy | Include storm forcing and rainfall-energy calculations needed by chosen erosion regimes | MAPLE input schemas need rainfall support; reproducing all stochastic generators and legacy rain-file formats is not necessary. |
| Infiltration, saturation/runoff generation, storage and drainage | Include a selected, validated formulation with spatial soil parameters and initial wetness | Do not automatically translate every legacy infiltration option. Account explicitly for rainfall, storage, infiltration/drainage and outlet discharge. |
| Surface-water depth, velocity, hydraulic roughness and flow routing | Include | Use one defensible numerical method initially; the old menu of routing algorithms is not itself required physics. |
| Rain-impact detachment and splash redistribution | Include in the completed storm-erosion target | A flow-only milestone can defer it, but then cannot represent erosion on rainfall-wetted surfaces without runoff. Splash remains distinct from wet sediment advection. |
| Diffuse-flow, concentrated/bedload and suspended transport regimes | Preserve distinctions for supported grain sizes and hydraulic conditions | A narrow first milestone can support fewer regimes only with explicit restrictions. Omitting suspension changes fine-sediment export; omitting diffuse/rain-assisted transport changes shallow-flow erosion. |
| Class-specific transport-distance relationships | Include as a defining feature | State whether parameters are mean or median, whether evaluated at pickup or along the path, and how changing hydraulics affect survival/deposition. Do not substitute an unrelated transport-capacity law. |
| Sediment travel/virtual velocity | Include with travel distance | Distance alone does not specify travel time, storm-end mobile load or hydrograph timing. Preserve the intended distinctions between slower bed movement and suspension travelling with water. |
| Grain-size composition and supply limitation | Replace legacy ownership with MAPLE active layer, voxels and availability | Water entrainment uses the current bed. Decide water-specific availability and entrainment-depth/rate rules, rather than assuming every wind availability restriction is physical for water. Six legacy classes are useful for comparison; arbitrary classes need their own parameterization. |
| Erosion/deposition-driven topographic change | Use MAPLE bed updates/commit framework; add hydraulic recomputation | Recompute slopes/drainage when needed after wind or water changes the surface. Select one water-depth authority and test volume balance. |
| Outlet sediment yield and hydrographs | Include through extended MAPLE accounting/output | Report per-class pickup, deposition, mobile mass and export, plus water budgets; preserve internal versus external transfer distinctions. |
| Spatial vegetation, pavement and surface-type effects | Include prescribed/current fields where they enter retained physics | Existing MAPLE vegetation effects on wind do not automatically supply rain interception/energy reduction, infiltration or hydraulic roughness effects. Map the meanings and units explicitly. |
| Inter-event state | Include hydrology persistence, explicit wetness evolution/forcing, and transport completion | Repeated events cannot silently reset moisture or erase mobile sediment. Long dry intervals require a defined drainage/evaporation treatment or prescribed moisture state with documented budget treatment. |
| Daily two-layer soil-water/evapotranspiration model and continuous calendar driver | Defer the full subsystem | Source calls calc_et, calc_sm/calc_inf and calendar/event orchestration. A simpler inter-event law or forcing is a model assumption, not equivalent ecological/hydrological prediction. |
| Dynamic vegetation: growth, mortality, dispersal and scenarios | Exclude initially | Prescribed vegetation cover remains possible. No automatic long-term vegetation–soil moisture feedback is claimed. |
| Dissolved nutrients and sediment-associated nutrient/carbon transport | Exclude initially | No nutrient concentration/export or chemical redistribution predictions. Sediment mass transport still works without these passive transport/accounting fields. |
| Marker-in-cell particle tracking and associated trajectories/RNG state | Exclude initially | Resolve sediment mass by class; no individual tracer histories. Numerical cohorts, if chosen, would not automatically reproduce the legacy marker model. |
| Automatic depression infilling/overtopping option | Defer generalized treatment; explicitly specify supported sink behavior | Start with draining surfaces/outlets, or retain ponded water in declared closed cells. Never erase water/sediment in pits; general wind-created depressions will require a policy before unrestricted landscape runs. |
| Alternative friction/infiltration/routing menus, calibration driver and stochastic parameter generation | Retain only selected relationships/options initially | Calibrated constants remain explicit; claiming one configuration does not imply coverage of every historical option. Numerical alternatives can be added when scientifically useful. |
| XML/GUSTAV-style inputs, standalone lifecycle and legacy output files | Replace with MAPLE configuration, event orchestration, diagnostics and restart | Changes to MAPLE state/schemas are required; no drop-in legacy input/output compatibility is promised. |

## Important boundaries on the ease claim

MAPLE already supplies much of the sediment inventory and landscape framework. It does not currently supply hydraulic routing, rainfall-driven detachment or water-specific transport-distance deposition. A conservative flow solver plus bed exchange alone is not the intended completed model.

Calling the work a **moderate extension** is defensible for a selected storm-process subset with restricted terrain and parameters. Covering all four storm transport regimes, unrestricted changing drainage, inter-event wetness and full restart is a larger validated model-development task. Full continuous ecohydrology, vegetation dynamics, chemistry and marker tracking would considerably broaden it again.

A numerical group/parcel implementation is one candidate, not a settled design. A pooled mobile-sediment formulation with a distance-dependent deposition law may be simpler. The scientific choice is whether transport distance is assigned at detachment or continuously adjusted by local hydraulics; those choices need not yield the same behavior.

## What must be decided before calling the extension physically consistent

1. Which MAHLERAN empirical relationships and regime criteria define the first supported scope.
2. Units, physical time basis and active-layer dependence of detachment.
3. Mean/median interpretation, transport velocity, distance-memory and deposition law.
4. Hydraulic/infiltration choices and spatial vegetation/pavement parameter mapping.
5. Actual supply limits, wet-to-dry transitions, terminal mobile load and inter-event wetness.
6. Boundaries, pits and rerouting after topographic change.
7. Validation targets: water/sediment budgets, trends with forcing/grain size, travel-distance and arrival-time distributions, timestep/grid sensitivity, storm hydrographs/yields and wind–water state/restart inheritance.

## Source grounding

- `MAHLERAN/src/Program_Control/MAHLERAN_1_2_3.f90:117–172`: event, continuous and marker variants.
- `MAHLERAN/src/Subroutines_Sediment/route_sediment_xml.f90:88–190`: flow-regime and suspension criteria, rain detachment, transitional conditions and dry splash.
- `Subroutines_Sediment/{diffuse_flow_transport,conc_flow_transport,suspended_transport}.for`: separate distance and virtual-speed calculations.
- `Subroutines_Sediment/raindrop_detachment.for:21–34`: vegetation modifies rain energy.
- `Program_Control/MAHLERAN_interstorm_xml.f90:65–107`, `Subroutines_Interstorm/calc_sm.f90`: daily ET/soil-water evolution and vegetation calls.
- `Subroutines_Vegetation/veg_dyn.f90`: growth, dispersal, mortality and cover updates.
- `Subroutines_Chemistry/route_chemistry.for` and `mahleran_input.xml:151–209`: dissolved and sediment-associated chemistry parameters/transport.
- `Subroutines_Water/route_water.for` and `Subroutines_In_out/dynamic_topog_attribute.for`: routing choices and overtopping option.
- MAPLE `water/`, `coupling/water_event.py`, `core/types/water.py`, and `surface/topographic_commit/`: reusable accounting/state hooks and currently absent water physics.

Sources establish the presence and implementation of these features, not their empirical validity or completeness under every option. This is source inspection, not an audit of the full MAHLERAN ecosystem model.

## Claude review incorporated

Claude reviewed the preceding feasibility answer (not this entire scope table) against both source trees. Report: [physics_native_review.md](agent_handoffs/physics_native_review.md). It agrees with the native-physics direction, rates a narrow fixed-geometry water subset moderate and the broader coupled model moderate–high, and emphasizes that water advection is new code and virtual sediment speed is as important as distance.

It proposes an Eulerian mobile-pool deposition law rather than water parcels: for exponential travel distances with constant mean L and constant speed v_s, the survival over a duration dt is exp(-v_s*dt/L). That is a useful candidate because deposition cannot exceed the available mobile pool. Qualification: this exact local decay does not make a spatially split, evolving-hydraulics solver timestep-independent. Recomputing L from local flow rather than retaining a source-assigned distance is also a modelling choice that requires validation. MAPLE's wind cohorts cannot simply be reused for converging drainage paths.

The temporal basis of pickup must be specified and calibrated; an explicit entrainment timescale is one option, not an already-approved new parameter. We should not claim legacy calibration transfers unchanged. Regime transitions and local detach/redeposit turnover also need explicit handling and tests.

No decision in this document authorizes removing a desired physical process. These categories distinguish a proposed complete storm-erosion target, smaller implementation milestones, and optional full-MAHLERAN subsystems for the user to assess.
