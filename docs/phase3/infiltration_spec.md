# Phase 3 minimal infiltration/soil-water specification

Status: implemented and CPU-validated on 2026-09-29; see acceptance.md for evidence and limits.

## Scope and source

Port the selected infilt.for model-2 and inf_type<5 capacity/drainage/storage relationships into explicit SI, conservative local columns. Include model1 (fixed conductivity) only if needed for independent controlled benchmarks, not all legacy menus. Rainfall forcing comes from rainfall.py. No hydraulic routing, ET, dry-reset, plant dynamics or sediment exchange yet. Surface depth is actual MAPLE WaterState depth; bed/mobile/ledger are untouched.

Source: MAHLERAN src/Subroutines_Water/infilt.for; initialize_values_xml.f90 lines228–229 and360–362 initialize ciinit=theta0*soil_thick*1000, sminit=theta_sat*soil_thick*1000 and copy them to cum_inf/stmax. Earlier Phase2 wording that no soil-thickness storm consumer was found is superseded by this direct evidence.

## Explicit parameter decision

For the first Plot1 benchmark use XML's positive mean Ksat=0.00025 mm/s (2.5e-7 m/s), not a random draw. The configured normal standard deviation0.001mm/s exceeds its mean and allows negative conductivities. This deterministic override is documented, not a reproduction of the legacy realization. An optional question was offered to the user; the mean is the stated default absent contrary steering. Allow explicit nonnegative per-cell Ksat arrays so future parameter sampling is outside the kernel. Never silently floor/clip negative supplied conductivities.

Use selected suction46.6mm, drain parameter0.05, initial volumetric moisture0.25, soil thickness0.3m, and the case's theta_sat map (~0.39). Read/check source controls rather than silently apply values to a different configuration. Soil-water thickness is separate from finite sediment-column depth and need not equal it. Honor or explicitly reject calibration overrides; default unit multipliers only after confirming no calib.dat applies. No changes to immutable Phase2 case binding.

## State and equations

All depths in m, rates m/s, time s. Host clocks/forcing schedule; per-cell arrays in MAPLE's resolved NumPy/CuPy namespace. Let h be ponded depth, S retained soil water depth, L soil thickness, theta=S/L, Smax=theta_sat*L. Initialize S=theta0*L. MAHLERAN's cum_inf is retained storage including antecedent water and subtracting drainage, NOT monotonic cumulative infiltration; keep this distinction. Track cumulative infiltration and drainage separately if exposed.

For model2, native pavement p=cover_fraction*0.01 (legacy input percent times1e-4). Lambda in mm/s is -0.022891667*ln(p)-0.098575 for p>0, otherwise0.16. Convert once to m/s. For positive LOCAL rain r use K=lambda*(1-exp(-r/lambda)); for zero local rain use Ksat. Correct legacy r2(i,2) conditional to r(i,j) explicitly; test different columns. Validate lambda positivity for supported pavement range. Evaluate with expm1 for small arguments.

For c=(psi+h)*(theta_sat-theta)*K, the original capacity is K/(1-exp(-S*K/c)); algebraically S*K/c=S/((psi+h)*(theta_sat-theta)) for K>0. Use stable expm1 and explicit limiting cases to avoid divide-by-zero/overflow. K=0 gives zero capacity. When deficit or suction denominator vanishes at S>0, limit is K. At S=0 with positive suction/deficit, capacity is unbounded, but potential intake is capped by available water before infinities can contaminate state. Specify the doubly zero limit (no capillary term -> K). Never hide NaN or negative storage by cleanup.

For a substep wholly within a constant rainfall segment:
- P = integrated rainfall depth, available surface water A=h+P.
- Potential intake J=min(A,capacity*dt), calculated with safe limits.
- Drainage demand from pre-step moisture: Dreq=(theta/theta_sat)*Ksat*drain_parameter*dt; actual D=min(Dreq,S+J).
- Saturation return O=max(S+J-D-Smax,0). This is explicit returned water, not clipped-away mass.
- Snew=S+J-D-O; hnew=A-J+O.
- Net infiltration I=J-O. Report J, O, I and D with clear meanings; no double counting.

Require finite nonnegative state, Snew<=Smax, and hnew+Snew+D=h+S+P locally and globally within explicit scaled FP64 tolerances. Surface water is storage available for later routing, not outlet discharge. Reject invalid inputs before caller-owned arrays/state are modified. Dry-step dt0 is identity. Masked cells retain preexisting storage and have no forcing/exchange; masking never discards inventory.

This explicit ordering follows the legacy intake/drainage/overflow structure while separating physical stores and fluxes. Splitting/limiting differs where legacy mutations of d1/d2/q mix hydrology. Test numerical convergence; do not claim exact Fortran equivalence.

## Case integration and validation

Validate the Phase2 sidecar, recipe/report hashes, MAPLE compiled-case identity/artifact provenance before using water parameters or rainfall. Use actual MAPLE loader. Current MAPLE bed/active/availability and mobile sediment stay unchanged. A column runner is an explicit no-routing diagnostic, not a full water event. Supply exact interval-ending rainfall; subdivide at EVERY forcing knot and a configurable max dt, never average nonlinear capacity over a rain jump. Bound output cadence and avoid storing full per-step grid history by default. Keep arrays on selected backend within loop; aggregate diagnostics at reporting boundaries where possible, report any validation synchronization costs.

Tests: independent scalar legacy-equation reference in valid finite domain; all-infiltration, rainfall excess and ponded runon; saturation return/drainage; zero K and zero deficit/suction limits; spatial local-rain branch; input rejection/nonmutation and mask preservation; analytical water budget and timestep refinement; actual Plot1 no-routing diagnostic with end-to-end rainfall integral, water budget and unchanged bed; optional CuPy equivalence skipped honestly if absent. No randomized field silently substituted, no GPU speed claim without measurement. Fortran executable comparison deferred unless available.
