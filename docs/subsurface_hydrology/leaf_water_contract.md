# LEAF–SYRUP water and growth contract

These rules record the user's decisions for the coupled subsurface development.
They are a specification, not acceptance of the complete scheduled simulation.
SYRUP owns physical water; LEAF owns plants, root geometry, potential demand and
biomass growth. The integrator owns the dated, nonoverlapping event schedule.

## Event timing

Wind, precipitation and interevent periods may occur in any order but never
overlap. Evaporation, transpiration and plant growth run only during enabled
interevent periods. Rainfall recession is part of the water event; there is no
biological catch-up for time spent in wind, rainfall or recession.

Independent process flags matter: disabling evaporation does not disable
transpiration, and disabling growth does not necessarily disable transpiration.
Dates, temperature and PET come from the shared weather/scheduler inputs.

## Potential demand and terrain PET

SYRUP calculates the terrain-adjusted PET field. LEAF uses the canopy-area-weighted
mean of that field for each plant's potential water demand. Canopy geometry is
separate from root geometry: this changes potential demand, not the distribution
of that demand among roots. The sparse canopy projection must declare its domain
and mask support, normalize by its supported overlap area, and refuse unsupported
zero-overlap plants rather than silently substitute a site mean.

For the initial coupled qualification, prescribed hydrology cover and
species-infiltration maps remain fixed. Plant growth does not silently update
them. LEAF-driven cover feedback belongs to Phase 2.5, alongside the vegetation
projection needed for splash.

## Equal access to available water

For plant p, let D[p] be its potential demand in a biological interval. Let
w[p,c,l] be the fixed fraction assigned by LEAF's roots to cell c and layer l.
The request on that connection is

    d[p,c,l] = D[p] * w[p,c,l].

SYRUP provides extractable water A[c,l] from its actual inventory and the declared
extraction floor, converted to the same volume units as plant demand. It never
creates water or withdraws below that floor. For each cell and layer,

    Q[c,l] = sum over plants of d[p,c,l]
    f[c,l] = min(1, A[c,l] / Q[c,l])  when Q[c,l] > 0
    u[p,c,l] = d[p,c,l] * f[c,l].

When Q is zero, deliveries are zero. Every plant in that cell and layer receives
the same fraction of its request. There is no ordering advantage, preferential
access, or transfer of unmet demand to another cell or depth. The sum delivered
never exceeds the resource. SYRUP debits that actual sum once and records it as
transpiration; LEAF uses the actual deliveries for its growth calculation.

A plant's supported fraction is its total delivery divided by its total demand,
with the existing LEAF zero-demand rule handled explicitly. Equivalently, where
root fractions partition demand, it is the root-weighted mean of the local
fractions met. It is not the minimum fraction met in any layer. This water factor
feeds the LEAF growth law; other temperature, seasonal and biomass constraints
still apply.

Example: one plant requests 100 volume units, split equally between two layers.
The layers contain 25 and 200 extractable units. Delivery is 25 + 50 = 75 units,
supporting 75% of the otherwise planned water-dependent growth. The layers finish
with 0 and 150 extractable units. The unused deeper water does not compensate
for the missing shallow water. The same calculation extends over multiple cells
and plants, using proportional sharing independently in every cell–layer pair.

## Transpiration while growth is disabled

Such an interval may withdraw real water and must contribute to SYRUP's water
ledger. It contributes no new demand, delivery, activity, temperature or elapsed
time credit toward later biomass growth. Previously earned growth credit is not
erased. The scheduler must finish any required pending growth at the last
growth-enabled endpoint before an upcoming disabled or transport interval;
biomass must not grow inside an explicitly growth-disabled interval.

## Conservation, transactions and memory

One accepted transition must join the calendar, authoritative water and plant
state. Failed candidates must leave their accepted parents intact. Restart must
restore the physical water only once and preserve biological accumulators,
geometry, source identities and the clock; it must not replay events or give
skipped time growth credit.

Use bounded sparse root and canopy connections and per-plant arrays. Dense
plant-by-cell or plant-by-layer-by-cell tensors are excluded. CPU and GPU paths
must preserve the same physical rules under the existing declared tolerances.
