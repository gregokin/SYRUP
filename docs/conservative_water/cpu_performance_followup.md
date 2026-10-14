# CPU performance follow-up (2026-10-09)

## Latest observation — 2026-10-14 UTC

The rotated Chastre B104C2/M94 CPU reference completed 300 accepted seconds:
450 s setup and 6,800 s for the loop including two checkpoint writes (230 s
and 226 s). It published 14 deep versions, wrote 90.66 GB of new chunks and
reported 324.93 GB of logical deep reads. These are whole-invocation counters,
not a measured breakdown of steady physical steps. The GPU reference wrote
55.25 GB because its older retained store deduplicated more chunks; their wall
times do not isolate backend computation. Evidence:
`agent_handoffs/tasks/conservative_water/cw_r8n_root_cpu300_observation.json`.

The optional storage candidate changes publication and snapshot work, so profile
the final independently accepted source before choosing another CPU hot path.
Use a bounded wet continuation, separate compilation, ordinary steps,
publication/commits and checkpoint costs, and verify its physical continuation
against an uninstrumented same-source reference. Do not instrument an active
model or infer the current bottleneck from the historical profiles below.
Existing environments provide no external sampling profiler; no profiler was
installed or attached during this run.

## Historical profiles and proposals

The F8 genuine wet restart now supplies an executed B55/M44 profile; the optimization proposals below remain unimplemented. See `agent_handoffs/tasks/conservative_water/cw_f6_cpu_cost_review_report.md` for the read-only review of archived F3/M40 profiles against current M44/B55 code.

The F8 fresh-process wet 300→360 s continuation completed and matched all 163 numeric members exactly. Its first accounting attempt took 45.52 s, including 7.35 s of Numba compilation. Across the next 59 wet attempts, the accounting core averaged 29.62 s: actual MAPLE pickup/deposition calls consumed 18.26 s per attempt (61.7%), sparse characteristic transport 6.15 s (20.8%), sediment physics 3.55 s (12.0%), and coupled hydrology 1.14 s (3.8%). Inclusive subcalls overlap; these categories are not a full wall-time decomposition. The full invocation took 3152.97 s including setup, checkpoint/output work and concurrent host activity. This is an instrumented observation of immutable B55/M44, not a controlled Fortran comparison or a timing claim for the newer B76/M75 paired candidate. The wet profile confirms full-domain active-layer/availability/ledger work remains the main CPU target; repeated full 306-level voxel scans are not the steady-step bottleneck.

Evidence: `cw_f8_large_cpu_wet_candidate360_first_resumed_attempt.profile.txt`, `cw_f8_large_cpu_wet_candidate360_subsequent_attempts.profile.txt`, `cw_f8_large_cpu_wet_candidate360_receipt.json`, and the independently accepted exact wet-restart comparison in the task archive.

In the archived two-step profile, the second step took approximately 33.8 s. Pickup and deposition together consumed approximately 20.84 s (62%); sediment physics took about 4.35 s, empty characteristic transport 6.98 s, and coupled hydrology about 1.3 s. These are inclusive observations from an instrumented older source, not controlled current timings. Initial preparation, checkpointing, full partition validation and source/store hashing also cost hundreds of seconds; inclusive totals overlap and must not be added as independent costs.

Steady steps do not repeatedly scan all 306 voxel levels. They do copy and check the resident three-voxel window and its summaries, and make many full-domain/class passes for availability, proportional mixture, active-layer updates, ledger accumulation and validation. Deep-store streaming dominates initial/checkpoint validation; the archived steady step recorded only four deep reads (about 0.21 s).

The bounded 240-second large CPU run has finished with verified wet transport; profile representative wet and empty steps of the actual current source separately from preparation/checkpointing. Candidate improvements include exact-order Numba fusion of availability and active-layer checks, compiled ledger operations, and paired reductions. Any proposed copy removal needs explicit ownership and lifetime proof. Investigate zero-face construction only after proving transport is empty. Do not treat zero-demand bed exchange as an automatic no-op: existing tests expose a roughly 3.37e-8 kg normalization residual, and Kahan compensation can change even on a zero increment.

Maintain MAPLE conservation bounds, caller immutability, transactional refusal, backend correspondence and restart equivalence. Do not modify the source imported by a running simulation. Current M44 lifetime improvements and B46 empty-kernel work mean the archived profile cannot establish current per-step cost; a 29–31 s estimate for an empty step is only an inference. This follow-up is separate from the deferred broad bin convergence/performance phase and does not alter the default legacy benchmark workflow.

The read-only CW-F7 design selects availability closure as a small optional Numba fusion candidate, avoiding ordered class sums and Kahan state. It has not been implemented or benchmarked. See `agent_handoffs/tasks/conservative_water/cw_f7_cpu_exchange_design_report.md`; the tiny toolchain checks do not establish full-model speed or portability.
