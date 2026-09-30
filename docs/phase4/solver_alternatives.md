# Phase 4 solver alternatives — literature review, 2026-09-29

The user selected MAHLERAN method 5 as the comparison baseline. The alternatives below are research candidates, not replacements implemented in this milestone. Scientific/numerical changes must be measured against that baseline on the same geometry, forcing and friction law.

**Updated user direction, 2026-09-29:** Numba compilation of the baseline belongs in Phase 4 and will support direct original-Fortran comparisons. Alternative numerical/hydraulic approaches are an important dedicated follow-up, **Phase 4R**, in [the implementation plan](../../syrup_implementation_phases.md). No measured speedup or accepted compiled implementation is asserted by this plan update. Compile the corrected, proven `[0, R]` root bracket, not the legacy bracket that can truncate the solution; retain explicit conservation corrections in comparison reports.

## Lowest-risk CPU acceleration: compile the same ordered solve

Compile the existing upstream-to-downstream loops and bisection with Numba nopython mode, leaving the discrete equations intact. Numba explicitly supports efficient numerical loops; its guidance recommends nopython compilation and reserves parallel loops for independent iterations. Do not apply `prange` across dependent downstream cells. Avoid `fastmath` initially because it permits changed floating-point behavior. [Numba performance guidance](https://numba.readthedocs.io/en/stable/user/performance-tips.html).

This is an engineering recommendation, not a measured SYRUP speedup. A serial compiled kernel removes Python dispatch without needing a new hydraulic approximation. Independent tributaries or cells at the same dependency level can be batched; a narrow, deep graph can offer little parallelism. Measure compile/startup separately from repeated storm work.

## Kinematic-wave alternatives

Wflow uses a nonlinear Newton solution for kinematic-wave surface routing and groups subbasins into a directed acyclic execution graph for threading. Its documentation also describes the limitations of assuming terrain controls flow, particularly where pressure gradients and inertia matter. [Wflow kinematic-wave documentation, version 0.8](https://deltares.github.io/Wflow.jl/v0.8/model_docs/lateral/kinwave/).

For SYRUP, replacing bisection with safeguarded Newton **on the same MAHLERAN scalar equation** is a candidate numerical optimization; adopting Wflow's whole discretization would be a larger change. A good initial bracket remains essential near drying. Compare roots, water residuals and rejected-step counts before measuring speed. Wflow is supporting design evidence, not a proposed dependency or a source of copied MAPLE infrastructure.

Explicit conservative kinematic-wave face updates are another candidate: each step can use old-state discharges in parallel, avoiding ordered implicit sweeps. This is our inference from the equation structure, not a speed result from the cited Wflow documentation. The tradeoff is a stability-limited timestep and changed numerical diffusion; assess total event runtime and hydrograph error, not time per update alone.

Landlab's `KinematicWaveRengers` provides a concrete Python example of explicit face-centered kinematic routing, with a depth-varying Manning relation. Its documented pit limitation matters here. This supports feasibility of the computational pattern, not hydraulic interchangeability with MAHLERAN: retain the Darcy–Weisbach law for a controlled numerical-method comparison. No Landlab dependency or code copy is proposed. [Landlab component documentation](https://landlab.csdms.io/generated/api/landlab.components.overland_flow.kinematic_wave_rengers.html).

## GPU candidate: local-inertial face fluxes

Sharifian et al. (2023) implement GPU versions of the LISFLOOD-FP local-inertial solver. It updates face discharges on a compact staggered stencil and conserves cell water through flux exchanges; it includes local acceleration but omits advective acceleration. The paper reports case-dependent GPU gains, not a universal hardware multiplier. [LISFLOOD-FP 8.1 paper](https://gmd.copernicus.org/articles/16/2391/2023/).

This is attractive for regular MAPLE grids because neighboring face operations can run concurrently. It changes the physics relative to MAHLERAN's terrain-directed kinematic approximation and requires additional discharge state and explicit boundary rules. The same paper notes friction-law limitations for shallow low-Reynolds-number rain-driven flow: a flood-model Manning law must not silently replace MAHLERAN's selected Darcy–Weisbach relation. Depth/velocity/shear errors matter for subsequent detachment even when total runoff agrees. Nonuniform-grid acceleration would complicate MAPLE's voxel alignment and is not the first candidate.

De Almeida et al. (2012) address stability of the simple local-inertial scheme; this is relevant to wetting fronts and low friction, rather than a guarantee of suitability for every plot-scale storm. The accessible primary-paper search excerpt was inspected; full PDF retrieval failed in this session, so detailed equations from that paper have not been audited. [Author-hosted paper](https://eprints.soton.ac.uk/356385/2/WRR_2012.pdf).

## Comparison order

1. MAHLERAN method-5 reference: identical network, fixed friction and CN equation; document necessary safety/conservation changes and separately quantify their effects.
2. Same equation compiled with Numba; then same equation with safeguarded Newton and/or dependency-level batching.
3. Explicit kinematic update and local-inertial solver as separate experiments only after baseline validation. They are not automatically accepted physics replacements.

For each: runoff volume, hydrograph peak/time, depth/velocity fields, recession, local/global water conservation, timestep/grid convergence, CPU/GPU parity, end-to-end wall time, peak memory and host/device transfers. Use both Plot 1 and larger/wider/deeper networks. GPU availability and Numba installation must be checked independently; no speedup or GPU equivalence has been demonstrated here. No sandpile formulation is proposed.

## Plot 1 regime screening (our calculation, not validation)

Under the selected constant Darcy–Weisbach law, Fr = v/sqrt(g h) = sqrt(8 S/f). The audited receiver-slope range 0.002–0.124 and f=21.45 imply Fr≈0.027–0.215 for positive depth. This is subcritical under that law and makes local-inertial comparison worth investigating. It does not establish accuracy of a different friction law, flow-direction rule, wetting-front treatment, or wave timing; compare those numerically against the baseline.
