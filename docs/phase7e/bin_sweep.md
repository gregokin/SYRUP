# Phase 7e Stage 3 — characteristic position-bin sensitivity (1 .. 128 bins)

Status: eight sequential frozen-Plot1 storms on the ACCEPTED dependency (MAPLE `72310c49`, SYRUP
source as checked out at `8ba4279`), `benchmarks/phase7e/run_bin_sweep.sh`, outputs in
`outputs/phase7e/sweep/b{1,2,4,8,16,32,64,128}`, condensed by `benchmarks/phase7e/summarize_bins.py`
into `benchmarks/phase7e/bin_sweep_qualification.json`. The 32-bin run reproduces the accepted
Phase 7d storm bitwise (103 saved numeric fields), so the sweep's provenance is the accepted one.
The production default (32 bins) is NOT changed by this study.

Predeclared diagnostic thresholds: 1% and 5% relative to the 128-bin run (finest available), per
metric: storm export total and by class, integrated deposition and its spatial pattern (relative
L2 of the per-cell cumulative deposition and net bed change), cumulative export curve (relative
L2), cumulative timing (endpoint-weighted centroid, 10/50/90% times), raw one-second peak export
and 10 s / 60 s window peaks. All class budgets must close (MAPLE policy) for a run to count.

Bin convergence and the legacy comparison are different questions: the legacy's export includes an
artificial clipping source, so no bin count is selected to match it.

## Results

All eight storms close their class budgets. Relative to the finest available
128-bin run, requiring the chosen and every subsequently tested finer count to
meet the threshold:

| Diagnostic | Minimum at 1% | Minimum at 5% |
|---|---:|---:|
| Total export | 2 | 2 |
| Class export totals | 4 | 4 |
| Spatial net bed change, relative L2 | 8 | 4 |
| Maximum cumulative export gap / final export | 16 | 8 |
| Same cumulative measure by class | 64 | 8 |
| 60-second peak flux | 8 | 8 |
| 10-second peak flux | 64 | 16 |
| Raw one-second peak flux | None below 128 established | 64 |

Eight bins are a promising lower-cost option for this case if 5% cumulative/class
accuracy and 60-second peaks suffice. Four bins reproduce final class totals well
but have worse timing. One bin loses the second exported grain class entirely and
is unsuitable. Keep the default at 32 until additional storms and timestep studies
qualify any replacement; even 32 does not establish raw-peak convergence.

Observed loop times for 1/2/4/8/16/32/64/128 bins were
179.3/181.7/184.4/191.4/206.2/247.1/366.0/514.7 seconds. These are single observations
on the accepted dependency, with the overlap caveat below; do not multiply their
speed ratios by the separate selective-exchange speedup without a combined run.

No bin count is established as equivalent to legacy MAHLERAN. Converged bin export
is about 13.763 g versus legacy Fortran's 9.223 g. One bin exports 8.529 g but has
incorrect sorting; closeness of that scalar is not physical equivalence. The
benchmark-only legacy replay separates numerical-port agreement from this genuine
algorithmic difference. The 128-bin result is a comparison reference, not proof of
continuum convergence.
