# Phase 7 independent review and Codex disposition

Claude implemented the bounded frozen benchmark runner/comparator; Codex
independently inspected the source, actual MAPLE callers, validation/freeze
contract, executed tests and all measurements. Claude then reviewed Codex's
diagnostic/profiling harnesses, GPU parity-test correction and interpretation.
Final review completed 2026-09-30, exact session
44f8a7e6-84b9-4f39-9603-4f28ef8e441b, exec42362 exit0.
[Review text](../../benchmarks/phase7/qualification/claude_review.md) is retained
verbatim; prompts/raw logs remain in the local task archive.

No new blocking defect was identified. Codex accepts the reproducible
**diagnostic measurement baseline**, not whole-storm sediment fidelity.
The approximately24-fold export excess, composition/timing differences,
full-case spatial convergence and complete GPU runtime/memory/transfer costs
remain open. No source change, parameter tuning or tolerance relaxation was
made in response to the discrepancy. Phase7b records the transport follow-up.

Previously confirmed and corrected defects: comparator confused synchronous
MAHLERAN peak-outlet fields with per-cell storm maxima; CuPy diagnostic scatter
failed for an empty face orientation; one malformed-input test fixture hit a
path guard before its intended parser check. Codex reproduced the failures,
verified bounded corrections, and retained failed and successful logs. Final
production source digest01709bb52720923aa9b7211409d7ffd19155c4a9bfc87c36d9e1f0180a58d532
stayed unchanged through every qualified full run.

Verification: full CPU513passed8device-skipped (original Fortran tests ran);
all eight device-specific checks subsequently passed on actual CUDA hardware;
corrected comparator nine tests passed; lint and whitespace checks passed.
Full storms used5400/10800/21600acceptedsteps with zero rejections and closed
budgets. CPU array/Numba97savedarrays matched exactly. Four standalone synthetic
CPU windows and three-size synchronized GPU kernel study completed. No commit,
push or wind invocation was performed by this phase.

Codex qualifications to the final review (reviewer claims are not themselves
verification):

- The scaling probe has **36 mm/h prescribed rainfall**, not “no-rain.” Its
  source uses constant_rainfall(0, steps, 36); it starts with5mm surface water.
  This does not change the timing or closure results.
- Whole-process CPU percentages above100% include imports and first-use/JIT;
  they do not prove the warm event loop ran in parallel. Thread settings were
  unconstrained, so the report makes no certified single-thread claim.
- No rigorous “≤1-step” bound on the field comparison is demonstrated.
  Each model uses its own peak-outlet snapshot; exact unrounded MAHLERAN peak
  time is unsaved, with a rounded1328–1342s plateau. The report states that
  uncertainty without inventing a temporal error bound.
- Coarse-flow/suspension **whole-storm** agreement is untested by Plot1;
  earlier original-Fortran equation tests remain valid within their scopes.
- The review's corrections of its earlier scientific overstatements are
  accepted: independent global maxima do not bound every local distance;
  total demand agreement does not rule out compensating local differences;
  ring-deposition effects cannot be called minor relative to tiny export
  without measurement; the legacy erosion/export gap lacks a full ledger.
  The controlled upwind mechanism is proven, but attribution of the exact
  whole-storm ratio is not quantified.

Final tables and evidence were populated from completed artifacts. Author and
reviewer agree that conservation and timestep stability do not establish the
required sediment distance, sorting or timing fidelity.
