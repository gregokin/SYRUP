# Phase 5 — verification record

Status: **accepted as a bounded CPU Phase 5 milestone**, 2026-09-30.
Codex independently verified implementation and results; Claude reviewed
Codex-authored code and final evidence with no remaining blocker. GPU
execution and full original-MAHLERAN sediment-storm fidelity remain
unverified. This is not acceptance of the complete alternating-event model.

## Scope and ownership

MAHLERAN wet detachment laws supply class-specific pickup demands, travel
distances and sediment velocities. Actual MAPLE owns the voxel column,
active layer, available sediment, mobile-water state, exchange ledger and
committed terrain. SYRUP adds conservative lateral mobile transport; it
does not copy wind physics or create another sediment bed.

Rain-assisted wet detachment is included. Direct splash, vegetation growth,
nutrients, storm evapotranspiration, event-end drying, restart and alternating
wind events remain outside this phase. Phase 4R's alternative hydraulic
solvers and GPU investigation remains an important follow-up.

## Independent references

`benchmarks/phase5/reference_physics_driver.f90` executes the original
MAHLERAN raindrop/flow detachment and diffuse/concentrated/suspended
transport routines. Original source hashes are pinned in
`reference_sources.json`; the compiler helper records build inputs and
outputs. A recording stub replaces `flow_distrib` to observe the distance
supplied by each law. This checks original equations, **not** original
sediment routing or a full MAHLERAN erosion storm.

The comparisons use the literal legacy vegetation, distance, dimensionless
diameter and Bagnold-depth conventions. Original COMMON parameters and
literals largely use REAL32; comparison tolerance is 20 ppm, not bitwise
identity. The separate bed adapter received Claude review with no blocking
finding; Codex independently exercised actual MAPLE exchange, refill,
topographic commit, rerouting and failure atomicity.

## Numerical qualifications

The selected physical relationships and departures are recorded in
[physics.md](physics.md). In particular:

- Pickup uses an explicit one-second reference interval, independent of
  the numerical timestep. This avoids reproducing the legacy per-step
  pickup dependence under timestep refinement.
- Deposition uses the exponential hazard `v/L` and conservative upwind
  movement on the hydraulic graph. Exact reaction preserves the physical
  deposition timescale. Controlled constant-law spatial patterns converge with grid refinement;
  a full-storm grid study and exact legacy coarse-cell deposition bins are
  not claimed.
- The distance law is evaluated at local current hydraulics. It is not a
  parcel history or a mean used to approximate mixed pickup histories.
- MAPLE's current active-layer fractions and available supply control
  pickup. Particle density converts the legacy pickup law to solid mass;
  MAPLE bulk density converts net bed mass change to elevation.
- Hydraulic depth remains unchanged during MAPLE bed commits, preserving
  water volume. Routing is rebuilt from committed terrain. Newly formed
  pits are unsupported and must fail without publishing partial changes.

## CPU kernel scaling

`benchmarks/phase5/kernel_scaling.py` measures warmed CPU calls separately
from setup. Seven samples per case used fixed diffuse-flow inputs and six
classes on south-draining grids. Values below exclude rainfall/routing,
MAPLE bed exchanges, commits, output and compilation. Memory is incremental
traced allocation for one warmed call, **not total process RSS**.

| Cells | Wet laws, median ms | Transport, median ms | Wet-law peak MiB | Transport peak MiB |
|---:|---:|---:|---:|---:|
| 1,200 | 1.767 | 1.184 | 1.644 | 1.620 |
| 19,200 | 23.773 | 21.356 | 25.970 | 25.657 |
| 76,800 | 97.099 | 93.582 | 103.808 | 102.536 |

Time and temporary storage scale approximately linearly over this range.
Array fusion/compiled kernels may reduce temporary storage; this is a
performance follow-up, not evidence of GPU acceleration. No GPU execution
has been verified on this machine. The final kernel measurements and source/hardware metadata are preserved in
`kernel_scaling.json` beside this document.


## Verified full event and regression

All results below use the fixed actual-MAPLE dependency described in
[dependency.md](dependency.md), package digest `d3d007024…`, equal to the
original Plot 1 import binding. The live upstream began independent changes
mid-task; its new source was not silently adopted. SYRUP/MAHLERAN sources,
forcing, configuration overrides and output hashes are recorded in each
run's summary.

Full regression: **446 passed, 6 GPU/CuPy skips**, in 107.21 s. Original
Fortran compilation was enabled, so the equation comparisons executed.
Ruff and `git diff --check` passed. Compact verification metadata is in
[validation.json](validation.json). Independent Claude review found no
remaining blocking defect; Codex ran all verification. Raw logs and
compiler/test artifacts are under `agent_handoffs/tasks/phase5b_integration/`.
The additional Phase 5 tests cover actual MAPLE supply limits/refill,
sorting, dry-cell settling, retained coarse mobile load, branching,
conservation, timestep/grid convergence, atomic failures, terrain refresh,
post-commit diagnostics, morphology and wet array/Numba equivalence.

The 5400-second, 1-second-step CLI run (`outputs/phase5_storm/dt1`) passed
all output gates and wrote an actual MAPLE snapshot, final grids and
hydrographs. Rain ends at 1620 seconds; this is a configured simulation
window, not a new event-stop algorithm.

| Quantity | Measured value |
|---|---:|
| Rainfall | 2.895600 m³ |
| Water exported | 0.1626808193 m³ |
| Final surface water | 0 m³ |
| Final soil water, retained | 25.2186069114 m³ |
| Water residual | 4.62 × 10⁻¹⁴ m³ |
| Gross sediment pickup | 411.1164148 kg |
| Gross sediment deposition | 410.8992650 kg |
| Sediment exported | 0.2171497658 kg |
| Final mobile sediment | 0 kg |
| Largest absolute class residual | 5.82 × 10⁻¹¹ kg |
| Net morphological erosion across eroding cells | 4.4490911 kg |
| Net morphological deposition across depositing cells | 4.2319413 kg |

Gross exchange counts repeated pickup/deposition. Morphological erosion
and deposition instead use independent final-minus-initial MAPLE cell
inventories, summing grain classes before splitting positive and negative
changes. Their difference equals sediment export within the measured
residual. This case exercises diffuse-flow sediment; concentrated and
suspended laws have independent equation/controlled-test coverage, not
coverage from this storm.

The one-second run was measured separately from the refinement runs:
228.555 s step loop, 231.59 s whole process, maximum RSS 359760 KiB
(351.33 MiB), Intel Core i9-7900X. Startup/JIT is separately recorded in
the summary. This is one observed machine run, not an original-MAHLERAN
full-sediment speed comparison. The half- and quarter-second refinements
run concurrently for numerical comparison; their elapsed times must not
be compared as controlled throughput measurements.

A separate profiled 600-second event spent about 75% of loop time in the
two MAPLE bed-exchange calls per step, versus about 10% in the wet laws
and lateral sediment kernel together. Profiling perturbs timing. The
finding motivates a shared MAPLE transaction/validation performance
follow-up, with conservation and source/sink timing preserved; it does
not justify copied bed physics, unchecked updates or a different
accounting cadence without validation.

## Full-storm timestep sensitivity

All three runs share identical source, actual MAPLE case artifacts,
MAHLERAN XML and resolved settings except maximum timestep. The comparator
verifies their hashes and recorded output-file hashes before comparison.
The finest run is a numerical reference, **not ground truth or a full
MAHLERAN sediment-storm benchmark**.

| Timestep, s | Water export, m³ | Sediment export, kg | Export difference vs 0.25 s | Net erosion, kg | Max final elevation difference vs 0.25 s, µm |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1626808193 | 0.2171497658 | −0.2450% | 4.4490911 | 1.019 |
| 0.5 | 0.1626909220 | 0.2175057967 | −0.08145% | 4.4358718 | 0.842 |
| 0.25 | 0.1627003095 | 0.2176831036 | reference | 4.4288492 | 0 |

The successive total-export changes are 0.0003560 and 0.0001773 kg,
consistent with first-order timestep convergence on this case. The largest
class-relative export differences against the finest run are 0.2955% and
0.09964%, respectively. The coarsest class exports only about 3.68 × 10⁻¹⁰
kg, so relative differences must be read alongside the absolute class
masses in the JSON. Sediment export peaks at 1140 s in all runs; peak rates
are 0.00143807, 0.00144407 and 0.00144707 kg/s. Water-discharge peaks occur
at 1332, 1331 and 1331 s. Recorded plot curves use a 60-second cadence;
the reported peak metrics are sampled every accepted step and at commits.

No timestep was rejected and no internal sediment substepping was needed.
All three end with zero surface water and mobile sediment while retaining
soil water. Across all runs, absolute water residuals are below 2 × 10⁻¹³
m³ and the largest absolute class residual is approximately 1.16 × 10⁻¹⁰ kg. Independent
checks on saved per-cell bed-change grids plus export also close below
10⁻¹⁰ kg per class. Terrain commits number 829, 920 and 996; direction-change
counts are 61, 59 and 59 (counts over commits, not necessarily unique cells).
Local terrain errors do not show a clean factor-of-two reduction: thresholded
MAPLE commit/routing changes remain part of timestep sensitivity. The small
total-export difference does not establish full-storm spatial convergence
or validate untested hydraulic/sediment regimes.

[Machine-readable comparison](storm_comparison.json),
[SVG figure](storm_comparison.svg), and [PNG figure](storm_comparison.png)
retain results, source identities, raw sampled curves and output hashes.

## Reproduction

First prepare and activate the fixed dependency environment in
[dependency.md](dependency.md). The imported case must bind the same MAPLE
package digest. `outputs/plot1` is the original Phase 2 case; if absent,
import `cases/plot1/recipe.yaml` using the current fixed dependency.
All experiment output directories must be new.

```bash
/home/okin/MAPLE/.venv/bin/python -m maple_syrup.sediment_experiment \
  --case-dir outputs/plot1 --output-dir NEW_DT1 \
  --max-dt-s 1 --implementation numba --report-every-s 60
```

Repeat with `--max-dt-s 0.5` and `0.25` into separate new directories. The
compiled option accelerates the ordered hydraulic solve; wet-law and
sediment-array work are vectorized NumPy on this CPU run. JIT warmup is
outside the recorded step-loop timing. Plot and compare all three runs:

```bash
/home/okin/MAPLE/.venv/bin/python benchmarks/phase5/compare_events.py \
  --runs NEW_DT1 NEW_DT0P5 NEW_DT0P25 --output-prefix NEW_COMPARISON
/home/okin/MAPLE/.venv/bin/python benchmarks/phase5/kernel_scaling.py --help
```

Regression on this machine used the isolated original-Fortran toolchain:

```bash
export MAPLE_SYRUP_GFORTRAN=/tmp/syrup-fortran/root/usr/bin/gfortran-13
export MAPLE_SYRUP_GFORTRAN_FLAGS='-B/tmp/syrup-fortran/root/usr/libexec/gcc/x86_64-linux-gnu/13/ -B/usr/lib/gcc/x86_64-linux-gnu/13/'
export MAPLE_SYRUP_GFORTRAN_LDFLAGS='-L/tmp/syrup-fortran/root/usr/lib/gcc/x86_64-linux-gnu/13/'
/home/okin/MAPLE/.venv/bin/python -m pytest -q -rs -p no:cacheprovider
/home/okin/MAPLE/.venv/bin/python -m ruff check src/maple_syrup tests benchmarks/phase5
git diff --check
```

A normal available `gfortran` can replace that isolated toolchain on
another machine; preserve and inspect compiler/source metadata and do
not count skipped Fortran tests as reference validation. Matplotlib is
needed only for the comparison figure, not the model.

## Acceptance and carried-forward work

The CPU milestone meets per-class closure, current-bed supply limits,
depletion/sorting/local-exchange and downstream-transfer checks, controlled
travel-distance/timing checks, actual original-Fortran equation checks,
conservative imported-case erosion/deposition/discharge, terrain refresh,
source provenance and failure atomicity. CPU kernel memory scaling and
whole-event profiling are measured separately. No production source changed
between the final regression and the matched three storm runs.

Claude's independent final implementation and evidence reviews found no
remaining blocking defect. Raw reviews, commands, stdout/stderr, correction
history, source manifests and test/compiler artifacts are preserved under
`agent_handoffs/tasks/phase5a_physics`, `phase5_reference` and
`phase5b_integration`. All launched writers, reviewers and benchmark jobs
have finished. Implementation was accepted before committing; the user
subsequently authorized a local commit. No push was performed.

Carry forward GPU equivalence/acceleration and larger whole-event scaling,
full matched MAHLERAN sediment-storm and full-storm grid comparisons,
unrestricted terrain/pit handling, and the measured MAPLE transaction cost.
Phase 4R's alternative hydraulic solver investigation remains important.
Phase 6 covers physical event completion, explicit water reset accounting,
restart and dry handoff; zero final surface/mobile storage in this case does
not substitute for those contracts. Splash and ecology remain deferred.
