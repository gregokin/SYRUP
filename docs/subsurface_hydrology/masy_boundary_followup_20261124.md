# SYRUP follow-up: evolving crop-edge water boundaries

Status: **open; SYRUP developer follow-up requested by the user** (2026-11-23 Pacific / 2026-11-24 UTC).

The Moab Stream 3/Jornada coupled computational test exposed a fixed-outlet limitation. MAPLE wind/avalanching can lower an interior edge cell below its unchanged observed exterior ring. SYRUP's Darcy normal-flow outlet requires a strictly positive bed drop and correctly refuses that face. This is a boundary-law limitation, not a reason to lower the ring or invent a minimum slope.

The user approved a temporary test rule, explicitly selected as `close_uphill_at_water_boundary`. Before daily water processing or a new rain/recession occurrence, the native `floating_boundary_outlets.close_uphill_boundary_outlets` closes currently declared faces at or below the observed ring. Remaining observed downhill outlets retain the existing law. Previously closed faces are not reopened. If all outlets close, closed components are explicitly allowed and their water is retained. An outlet becoming unsupported during an active storm causes the existing native refusal when a displacement-triggered topology refresh next runs; the held geometry remains in use below that threshold.

The transition is owned by SYRUP. It preserves the actual bed, water inventories, W/J/I, property root, vegetation binding, cursor, forcing, ledger and clock. Closing a face zeros that instantaneous boundary discharge without transporting/removing water or sediment. Other face fluxes are preserved. This does not claim momentum conservation. A changed outlet set is an explicit geometry compatibility rebuild; ordinary rerouting keeps the configured MAPLE net-displacement gate. No-op calls retain the prior topology and source. Pending recession stays pending. The transition is limited to bounded Dense cases; it is not a combined restart implementation.

Required developer follow-up:

- Choose the scientific boundary condition for evolving edges: fixed external stage/backwater handling, a normal-flow alternative, or another explicit law.
- Define candidate outlets, reopening and changes during active storms, including the case of a level or uphill bed with water head high enough to export.
- Specify face-discharge/momentum treatment, closed components and conserved water/sediment exchanges.
- Bind changes to routing identity, provenance, atomic publication and exactly-once restart; qualify CPU/CUDA and any larger/windowed path separately.
- Replace or explicitly retain the test approximation after scientific validation. Do not silently broaden it to existing fixed-outlet profiles.

The current implementation does not resolve these scientific questions. See the MASY task `current_triplet_irregular_20261123` for source reviews and local computational evidence; saved terrain/forcing/array payloads remain local.
