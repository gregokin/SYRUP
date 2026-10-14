# Rotated Chastre CPU/GPU comparison — 2026-10-14 UTC

The actual B104C2/M94 CPU and GPU wet runs both reach 300 s with 300 accepted
steps, no rejected attempts, 24 accounting commits and no hydraulic reroutes.
The CPU/GPU comparison is **strict FAIL**, retained unchanged: four failed
records at 240 s and seven at 300 s. Every recorded bed inventory, deep-store,
hydraulic field, transport state, routing and decision comparison passes its
existing field bound. Each run independently closes its water and class
sediment budgets under the existing MAPLE conservation policy.

| Observed quantity | 240 s | 300 s |
|---|---:|---:|
| PASS / FAIL / diagnostic records | 187 / 4 / 4 | 261 / 7 / 6 |
| Maximum per-cell depth difference | 2.44e-19 m | 5.42e-19 m |
| Maximum mobile holding difference | 2.78e-16 kg | 5.55e-16 kg |
| Maximum resident voxel holding difference | 4.26e-14 kg | 7.11e-14 kg |
| Deep inventory manifest | identical, generation 7 | identical, generation 14 |

The four failed quantities at 240 s are the ledger's compensated-sum remainder,
near-zero requested-minus-actual deposition total, accumulated transport budget
residual and accumulated absolute numerical-residual counter. At 300 s, water
and pickup-reconciliation residuals also fail their cross-backend field checks;
the numerical-residual counter appears again in the summary, giving seven
records for six distinct quantities.

These checks compare near-zero differences of much larger operands with an
absolute floor of 1e-14. For example, the GPU water residual differs from the
CPU's zero by 1.819e-12 m³, exactly one ULP of the soil-water operand; both
independent water budgets close. The existing bounds are unchanged, and these
observations do not turn the strict FAIL into PASS.

Claude's separate read-only review found no concrete exchange defect or
decision divergence. It identified the ledger compensation as an accounting
remainder and the transport residual as a diagnostic within its own recorded
budget bound. The deposition difference has a source-supported reduction-order
explanation conditional on zero-request pickup removing exactly zero; that last
condition was not separately traced in the review. The absolute numerical
residual's physical attribution is unresolved at the scale of the full bed.
Some of these quantities feed later sums or acceptance/completion gates, so
matching decisions through 300 s does not establish longer-run equivalence.

This is the rotated/resampled scientific case on source pair 8ac/M94, not native
Chastre or a Fortran comparison. It does not qualify a full storm, an actual
large net-driven reroute, or the newer 5f/24c overlay source pair. The separate
GPU own-wet restart remains exact across 184 numeric members. See
[net-elevation routing](net_elevation_routing.md) for resource and timing limits.

Raw evidence: `agent_handoffs/tasks/conservative_water/cw_r8nc2_root_backend_240_comparison.json`,
`cw_r8nc2_root_backend_300_comparison.json` and `cw_r8nc3_report.md` in the same
directory. Initial checker refusals and all strict failures remain archived.
