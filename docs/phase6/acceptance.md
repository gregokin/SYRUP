# Phase 6 validation and review

Status: **accepted as a bounded CPU Phase6 milestone**, 2026-09-30.
Baseline: `a9e4d5fe54d005d932176c1a7f51f9f7fae3edb5` (accepted Phase4/5).
Claude independently reviewed the Codex-authored candidate after the usage reset
and found no reproducible blocker. Codex then ran the suggested actual-Plot1
terminal probe and a bounded transport stress experiment; both passed without
production changes. Review and disposition are in
`agent_handoffs/tasks/phase6_complete_event/claude_review.md` and
`agent_handoffs/tasks/phase6_review_resume/disposition.md`.
No Phase6 commit or push has been made. General GPU, larger-grid and full
MAHLERAN sediment-storm qualifications remain open.

See [design.md](design.md) for completion, actual MAPLE conservation policy,
conservative terminal deposition, external water removal, checkpoint and CLI
contracts. Wind invocation remains a later phase; no wind physics was copied.

## Verification executed

Using the unchanged pinned actual MAPLE package
`d3d007024ff65abc2a0ff179f0f03bcd1c3b27f091493136cce849c34f7a4264`,
NumPy 2.5.2/Python 3.12.3 and isolated optional Numba/Fortran tooling:

- Full phases 1–6: **485 passed, 6 skipped**, 119.50 s. The six skips are GPU
  checks; CuPy/CUDA is unavailable. Original Fortran equation comparisons ran.
- Subsequent completion edge-case coverage: **41 Phase 6 tests passed**,
  15.34 s, including two added tests for residual-only ledger flushing at the
  same committed time and post-commit reactivated flow refusing the dry reset
  without changing caller state. Production code is unchanged; the full-suite
  count above is the actual earlier run, not an inferred new full-suite result.
- Ruff passed for `src/maple_syrup`, `tests/phase6`, `benchmarks/phase6`.
  `git diff --check` passed.
- Controlled tests cover wet/dry completion, interrupted quiet hold, later
  rainfall after a gap, ponded storage despite zero outlet flow, significant
  mobile mass, duration/global-step refusal, actual nonzero terminal deposition,
  failed terminal commit preserving caller state, exact wet continuation,
  pending ledgers without checkpoint commits, NumPy/Numba equivalence, invalid
  source/forcing/config/hash/dtype/shape/counter/ledger data, completed reset
  replay refusal, and real Plot 1 pause/reload/protected-path behavior.

Reproduction uses the dependency environment in Phase 5 documentation; the
exact local exports and command logs are archived under
`agent_handoffs/tasks/phase6_complete_event/`.

```bash
python -m pytest -q -rs -p no:cacheprovider
python -m ruff check src/maple_syrup tests/phase6 benchmarks/phase6
python -m maple_syrup.event_experiment --case-dir outputs/plot1 \
  --output-dir outputs/phase6_validation/full --implementation numba
python -m maple_syrup.event_experiment --case-dir outputs/plot1 \
  --output-dir outputs/phase6_validation/resumed --implementation numba \
  --resume outputs/phase6_validation/full_checkpoints/step_000001200
python benchmarks/phase6/compare_restart.py \
  outputs/phase6_validation/full outputs/phase6_validation/resumed \
  --output benchmarks/phase6/validation.json
```

Use new output names when rerunning: existing outputs are deliberately refused.
Source identity is strict; changing source invalidates older continuation.

## Actual Plot 1 result

Curated machine-readable evidence: [validation.json](../../benchmarks/phase6/validation.json).
Default completion occurred at **4747 s**, after an uninterrupted quiet interval
starting at 4687 s. There were 4747 accepted physical steps and 819 actual MAPLE
commits. Surface water and mobile sediment were already exactly zero at
completion; terminal deposition was zero in this case. Nonzero terminal
settlement is exercised by the controlled actual MAPLE tests.

The wet checkpoint at 1200 s held **79.406663 kg mobile sediment**, maximum
surface depth **0.0034478012 m**, and **0.13638865 kg absolute pending bed change**.
A separate process loaded that checkpoint without a commit. Its final state,
completion time, cumulative budgets and quiet progress matched uninterrupted
execution. All **167 saved numerical arrays** were bitwise equal: 13 final-state,
55 hydrograph and 99 complete-continuation arrays. This includes availability,
ledgers, committed terrain, transport memory and cumulative diagnostics.

| Quantity | Result |
| --- | ---: |
| Rainfall | 2.895600 m³ |
| Runoff export | 0.1626808193 m³ |
| Storm drainage | 0.01255283272 m³ |
| Pre-reset soil water | 25.22036635 m³ |
| Surface water externally removed | 0 m³ |
| Soil water externally removed | 25.22036635 m³ |
| Storm and combined-reset water residual | 4.40e-14 m³ |
| MAPLE-coefficient water arithmetic bound | 5.10e-7 m³ |
| Maximum sediment closure residual by class | 5.82e-11 kg |
| MAPLE reservoir conservation bound per class | 1.70e-6 kg |

The soil removal implements the user's dry-again assumption. It is explicitly
separate from physical runoff/drainage; no evapotranspiration was simulated.
The tiny sediment residuals are measured errors, not new hard-coded tolerances.
No physical mass-resolution floor or numerical-residual allowance pads closure.

## Performance and remaining qualification

[checkpoint_profile.json](../../benchmarks/phase6/checkpoint_profile.json)
records five warm-filesystem CPU save/load trials for the actual 1200 s wet
checkpoint. Bundle size is **5,138,277 bytes**; median save/load wall times were
**0.0338/0.0344 s**. Profiling-process peak RSS was **213,556 KiB**, including
imports, the prepared case and simultaneous source/restored state. This is not
full-event peak memory or a scaling/GPU benchmark.

The uninterrupted invocation took about 208 s including setup, lazy compilation
and checkpoints. It overlapped regression and resumed execution, so this is an
observational duration, not a controlled speed comparison. Fixed-domain memory
and cadence remain bounded; larger-grid/backend qualification belongs to the
planned performance work.

Default stop thresholds completed this case with zero surface/mobile inventory.
A subsequent full run reduced all five water/mobile stopping thresholds by
100× and doubled the hold from 60 to 120 s; it retained MAPLE's numerical
conservation policy. Both runs first qualified at **4687 s**. The stricter run
finished at **4807 s**, exactly 60 s later. All eleven compared final physical
arrays (including bed, availability, terrain and dry water/transport state)
and class-by-class sediment export were bitwise identical. The longer physical
recession increased drainage before reset, reducing externally removed soil
water from 25.2203663480 to 25.2202046795 m³. Both budgets closed. Evidence:
[stopping_sensitivity.json](../../benchmarks/phase6/stopping_sensitivity.json);
reproduce the comparison with `benchmarks/phase6/compare_stopping.py`.

This supports the defaults for this Plot 1 event. General threshold/grid
sensitivity across other cases, GPU execution/equivalence, larger-event
memory scaling, full original-MAHLERAN sediment-storm comparison, alternative
hydraulic algorithms (important Phase 4R follow-up), and general pits remain
unverified. The checkpoint currently supports strict same-source/config CPU
continuation; it is not a general migration format. Independent Claude review
must be completed before Phase 6 acceptance.

## Review follow-up verification

Eight terminal-deposition probes on the actual completed Plot1 bed passed with
uniform/random remaining loads from1e-12 to1e-6kg per cell/class. The two largest
load scales explicitly raised only the physical stopping threshold; arithmetic
conservation bounds were unchanged. Maximum residual2.91e-11kg remained below
1.79e-10kg. Caller mobile arrays were unchanged. Separately,56seeded transport
stress cases across converging/random grids and1–64substeps passed all existing
checks (maximum class residual/bound0.014606). Evidence and reproducible probe
scripts are archived in the review-resume task. These are new focused checks,
not a new full-suite run or proof for arbitrary grids. No production edits.
