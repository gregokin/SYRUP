# Alternate GPU hydrology approaches — research update, 2026-10-02

Codex researched this while Claude implemented the authorized exact-physics CUDA storm hydrology task. The user subsequently authorized implementing the explicit kinematic-wave and local-inertial candidates as testable experiments. Both now have separate CPU/CUDA implementations undergoing qualification; this research supplies their design rationale. See [experimental solver usage](../hydraulic_candidates/README.md). Same-equation Newton and the other families remain research candidates. This updates the [initial literature review](../phase4/solver_alternatives.md). The current implementation remains the corrected MAHLERAN-inspired method-5 baseline. No sandpile approach is proposed.

Our recommendation was to finish and measure that baseline, then evaluate safeguarded Newton on its existing scalar equation, and a conservative explicit kinematic-wave GPU stencil. Investigate a uniform-grid local-inertial solver next if its additional hydraulic behavior is useful. Keep rainfall, soil physics, MAPLE sediment authority, and the hydro-to-sediment interface unchanged during each controlled experiment.

## Where the present cost comes from

Plot1 has 1,200 active cells arranged into 66 dependency levels, with at most 86 cells per level. The accepted GPU sweep launches once per level; each wet cell performs 40 dependent bisections. Even a single whole-sweep block uses only a small part of the GPU. Codex's task-owned all-positive-RHS probe preserves four raw output arrays bitwise and reduces median Plot1 raw sweep time from 1.719 to 1.358 ms, but slows a 65,792-cell random network from 5.310 to 7.113 ms. These are 30 warm synchronized routing-only samples, not full-storm or default-CPU comparisons. Source, input construction and raw samples: agent_handoffs/tasks/phase4r_gpu_storm/single_block_probe.{py,json,log}. Full current task results will be reported separately.

Three different changes must be distinguished:

| Candidate | What it preserves | What it changes | Main GPU benefit | Main cost or risk |
|---|---|---|---|---|
| Fused kernels, launch replay, batched independent storms | Existing discrete equations and ordered dependency solve | Execution and buffer layout | Fewer launches/transfers; independent storms add parallelism | Narrow DAG remains; lifetime/retry safety |
| Safeguarded Newton for the scalar root | Same CN equation, D4 network and Darcy–Weisbach law | Root-finding arithmetic and termination | Potentially fewer square roots per wet cell | Different rounding, convergence near drying; DAG remains |
| Explicit conservative kinematic wave | Same runoff physics, fixed D4 directions, Darcy–Weisbach law | Time integration and possibly reconstruction | Whole-domain parallel face fluxes and cell updates | CFL-limited substeps, changed diffusion and peak timing |
| Uniform-grid local-inertial solver | Water conservation; existing column physics can remain | Momentum state, water-surface gradients, possible reversal | Compact staggered-grid stencil, no topographic ordering | Different hydraulic approximation, friction and wet/dry integration |
| Diffusive-wave solver | Friction-dominated flow and conservation | Water-surface-controlled slope and directions | Parallel neighbor stencil | Explicit diffusion stability constraint or a global implicit solve |
| Full shallow-water finite volume | Mass conservation; existing column source terms can remain | Adds both momentum components and their dynamics | Mature GPU face/cell decomposition | More state, gravity-wave CFL, well-balanced wetting/drying |
| Tree-contraction flow accumulation | Static flow-network accumulation | Usually removes transient storage/travel-time dynamics | Logarithmic graph passes | Not a replacement for this storm depth/discharge solver |

The engineering assessment in this table is our inference from the source equations and cited implementations. None of the papers measures performance of SYRUP or predicts a universal GPU speedup.

## Preserve the equation before changing the hydraulics

The present cell root is

    F(h) = h + c k h^(3/2) - R = 0,
    c = dt/(2 dx),  0 <= h <= R.

For nonnegative R and k, F is monotone. Its derivative at positive depth is 1 + (3/2)c k sqrt(h). A safeguarded Newton step could therefore reuse the proven bracket and fall back to bisection when necessary. This is a proposed numerical optimization derived from our source, not a tested algorithm. It must meet the existing field and conservation tolerances; stopping at the configured root tolerance alone does not establish close agreement with the 40-iteration lower-bracket baseline. The substitutions u=sqrt(h) and c k u^3 + u^2 - R=0 also identify a cubic, but a closed-form formula could suffer cancellation and introduce expensive transcendental functions. It is not automatically the faster or safer choice.

Wflow provides an existing nonlinear Newton kinematic-routing implementation and documentation useful for comparison. Its complete time discretization should not be silently substituted for ours. [Versioned Wflow documentation](https://deltares.github.io/Wflow.jl/v0.8/model_docs/lateral/kinwave/) was inspected again in this update; the current stable URL could not be fetched.

Exact launch replay is an engineering alternative: CuPy supports stream capture and graph replay. Capture disallows synchronous host transfers, so the validation-packet read must sit outside capture. Captured pointers must remain alive and match the buffers used on replay; variable timestep and retries require explicit handling. Replay reduces enqueue overhead but cannot remove the dependency chain or bisection work. [CuPy Stream documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html).

Batching separate storms or parameter ensembles is another physics-preserving option: each independent storm can expose additional parallel work. It increases device memory roughly with ensemble size and only helps when the scientific task actually needs several independent simulations. A persistent dependency-counter work queue is more complex: donor publication and counter updates need correct memory ordering, and inter-block spin waiting can deadlock without carefully bounded residency. The safe narrow-block alternative uses unconditional level barriers, which make prior writes visible within that block. [NVIDIA synchronization specification](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#synchronization-functions).

## First numerical alternative: explicit kinematic-wave fluxes

Kim, Park and Kim (2019) implement a GPU finite-volume kinematic-wave rainfall–runoff model in CUDA Fortran, with Green–Ampt infiltration. Their speedups increase with grid size. The paper is evidence for the computational family, not equivalent MAHLERAN physics or reusable Python GPU code; its Manning discharge law and infiltration would require deliberate adaptation. [Primary article and PDF](https://jkwra.or.kr/articles/article/rWmq/).

Landlab documents both the bed-slope kinematic approximation and the Darcy–Weisbach depth exponent 3/2, alongside Manning's 5/3. Its implicit implementation still walks upstream to downstream, so merely borrowing that implementation would retain our ordering bottleneck. Its components are useful equation, analytic-test and code-pattern references rather than a replacement for MAPLE infrastructure. [Landlab kinematic-wave theory](https://landlab.readthedocs.io/en/latest/tutorials/overland_flow/kinwave_implicit/kinwave_implicit_overland_flow.html).

Our proposed controlled experiment is narrower: retain D4 receivers, the existing k field, rainfall and MAHLERAN-derived column physics. Calculate each outgoing face volume from an old-state discharge, gather incoming volumes in the fixed donor-slot order, and update cells in parallel. Use each face volume exactly once for donor loss and receiver gain; do not perform unrelated scatter sums or clip negative storage. A two-stage SSP update is a candidate after a conservative first-order reference.

For q=k h^(3/2), the characteristic speed dq/dh is (3/2)k sqrt(h), rather than the sediment or fluid material speed q/h. This derivative is our calculation. A suitable explicit CFL/positivity bound must use the chosen discretization and maximum relevant stage state; the legacy Courant threshold cannot simply be inherited as its proof. More substeps could outweigh cheaper GPU updates. Rain/infiltration splitting and the returned velocity/face-volume contract must be explicit, because detachment and grain travel depend on them.

## Additional hydraulic candidate: local inertia

LISFLOOD-FP 8.1 includes GPU uniform-grid and nonuniform-grid local-inertial solvers. The model uses compact staggered face updates, retaining local acceleration while dropping advective acceleration. Its reported GPU gains are case-dependent. The authors also identify limitations of Manning friction for shallow rain-driven low-Reynolds-number flow. Start with a uniform grid; their adaptive-grid acceleration is a separate mesh change that complicates MAPLE voxel alignment. [Sharifian et al., 2023](https://gmd.copernicus.org/articles/16/2391/2023/).

De Almeida and Bates (2013) found the approximation particularly accurate in low subcritical flows, with differences increasing with Froude number and depth gradients; unsteady propagation could be slower than the full equations. That matters for runoff peaks and sediment timing even when integrated water export agrees. [Primary author-institution record](https://research-information.bris.ac.uk/en/publications/applicability-of-the-local-inertial-approximation-of-the-shallow--2/).

Our freshly verified Plot1 screening uses the current law v=k sqrt(h), so its implied Fr=k/sqrt(g)=sqrt(8 S/f) at positive depth is independent of h. Across all 1,200 active cells, min/median/max are 0.027312 / 0.130982 / 0.215051; none exceeds 0.5. This is a calculated proxy under the existing friction/slope assumptions, not a local-inertial solution or evidence of velocity equivalence. Exact task evidence: agent_handoffs/tasks/phase4r_gpu_storm/plot1_froude_proxy.json. A new solver would still need Darcy–Weisbach-consistent friction, face geometry and boundary handling; adopting a constant Manning coefficient would change the depth dependence.

For implementation, the official [Wflow local-inertial equations](https://deltares.github.io/Wflow.jl/quarto/model_docs/lateral/local-inertial.html) provide an additional reference for staggered continuity, gravity-wave timestep control and momentum smoothing. Wflow uses Manning friction and optional Froude limiting; neither should be inherited silently. Our Darcy friction adaptation follows the friction slope f q|q|/(8 g h^3), documented by [Kirstetter et al.](https://arxiv.org/html/1609.04711v1), and therefore contributes (f/8) q|q|/h^2 to the unit-width momentum equation. The semi-implicit update and normal-flow outlet choice are our experimental design, rather than an exact implementation of those papers. Directional treatment of friction and wetting/drying must be qualified separately.

## Other families worth retaining as comparisons

Park, Kim and Kim (2019) report a GPU diffusive-wave finite-volume scheme with rainfall/infiltration and depression tests. It uses water-surface slope rather than fixed terrain slope. This can represent hydraulic ponding and backwater, but the explicit stability cost must be measured. [Primary diffusive-wave paper](https://www.mdpi.com/2073-4441/11/7/1447). The primary article appeared in search; a subsequent direct fetch returned HTTP429, so detailed source code and its complete stability derivation were not audited.

TRITON is an existing open-source multi-GPU full shallow-water model using an explicit augmented-Roe scheme with a local implicit friction treatment. Its documented architecture is useful for face fluxes, domain decomposition and source-term integration, but its complete hydraulics are a larger change than an explicit kinematic experiment. [ORNL primary publication record](https://impact.ornl.gov/en/publications/triton-a-multi-gpu-open-source-2d-hydrodynamic-flood-model/), [official source project](https://code.ornl.gov/hydro/triton). The repository page exposed only metadata in this browsing session; no implementation/license audit or execution was performed.

LISFLOOD-FP's full-equation FV1/DG2 GPU comparisons demonstrate that solver complexity alone does not determine runtime. In that paper, CPUs were more efficient below 0.1 million elements and GPUs most efficient above a million; those are measurements of their models and hardware, not a SYRUP crossover threshold. [Shaw et al., 2021](https://gmd.copernicus.org/articles/14/3577/2021/).

SynxFlow supplies another actual GPU shallow-water implementation with Python-facing input tooling and rainfall-driven simulations. The tutorial supports spatial rainfall and GPU selection. This is a source to inspect for reuse and independent hydraulic benchmarks, not authorization to add a separate model or replace MAPLE's state/configuration authority. [Official repository](https://github.com/SynxFlow/SynxFlow), [rainfall tutorial](https://synxflow.readthedocs.io/en/latest/Tutorials/flood.html). Neither its solver source, infiltration details nor current hardware/toolchain compatibility were audited here.

FastFlow (Jain et al., 2024) uses rake/compress and pointer jumping to accelerate upstream accumulation and depression connectivity; released examples include CUDA kernels behind Python tensor frameworks. The accumulation is fundamentally different from our nonlinear transient depth solve. It could inform routing preprocessing, diagnostics or future rerouting, but replacing storm flow by upstream rainfall accumulation would discard transient storage and travel-time behavior. [Author-hosted paper](https://people.cs.uct.ac.za/~Jgain/wp-content/papercite-data/pdf/jain2024.pdf). Its linked GitLab source returned an anti-bot page, so code/license audit remains unperformed.

## Bounded follow-up experiments and acceptance

1. Keep the exact MAHLERAN-inspired GPU storm implementation as the comparison baseline; freeze source/input provenance and separate compilation, preparation and warm storm times.
2. Probe safeguarded Newton on the same scalar equation, then the same ordered timestep. Test dry and tiny RHS, strong conveyance, user iteration/root options, requested rejection behavior and every water balance. Do not claim legacy bitwise identity.
3. Prototype explicit D4 kinematic face transfers with existing column physics and Darcy–Weisbach k. Compare timestep-halving convergence against the baseline on Plot1 and steeper/wetter/wider cases.
4. Prototype uniform local inertia only as an explicitly different hydraulic method. Include backwater/ponding tests, thin overland flow, rainfall–infiltration interaction and recession. Retain full SWE as an independent reference where local inertia becomes questionable.
5. Investigate launch replay, batching, wet-cell/tile compaction and reporting fusion only where measurements identify those costs. Compaction must include cells receiving flow, retain dry-cell water/soil/sediment inventories and deterministic balances.
6. Leave adaptive meshes, mixed precision, learned surrogates and steady upstream-accumulation replacements outside the first comparison. Mixed precision would require new declared conservation/error contracts and is not a shortcut to preserving current FP64 invariants.

Measure whole-storm wall time including required extra substeps, peak device and host memory, transfers, runoff integral/peak/time, synchronous depth and velocity maps, local/global water residuals, and rejection counts. Once water-only fidelity is established, rerun the existing wet detachment and sediment transport using identical sediment inventory so small hydraulic differences are not accepted solely from runoff agreement. No absolute speedup or alternative method has been accepted by this research.

The algebraic Froude proxy above describes the existing kinematic constitutive law. It does not qualify the implemented local-inertial solver: provisional face and reconstructed-cell diagnostics expose high-Froude transients and drying sensitivity. See the experimental solver documentation and forthcoming measured comparison; no erosion coupling is qualified by that proxy.
