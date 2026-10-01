# Phase 7e qualification

This phase implements selective exchange in an isolated actual-MAPLE dependency,
adds a benchmark-only MAHLERAN transport replay, and evaluates 1–128 characteristic
bins. Wind physics, live upstream trees, production transport choice and the
32-bin default remain unchanged.

The final candidate package is
`9a4da48c397333eff8a7c8321894843bee7a3df60d7d337c1d16d2408966fb73`.
It is reconstructed from accepted `72310c49…` and the six-file shared MAPLE patch;
`benchmarks/phase7e/README.md` gives exact activation/reproduction instructions.
The final package was rebuilt independently from the patch with its expected hash.

## Evidence

- Targeted Phase7e CPU tests: 256 passed, 222 GPU parametrizations skipped.
- Actual GTX1080Ti selective-exchange tests: 448 passed, no skips.
- Actual original Fortran `flow_distrib` is compiled and exercised in the CPU tests.
- Transaction sequence compares physical arrays, transfer diagnostics and rolling
  state hashes exactly; capacity refusals must occur on the same steps and retain
  their error category. Error wording can describe only the gathered subset.
  Failure tests also assert the original input arrays remain unchanged.
- Cache tests cover writable mutation, CPU/GPU conversion, full validation,
  topographic commits and actual checkpoint continuation. CPU metadata is trusted
  only with enforced read-only ownership; writable device arrays require an
  explicitly trusted chain. Public untrusted input rebuilds its totals.
- The fastest opt-in bed-change diagnostic re-reads touched stored voxels after
  scatter. It remains independent of transfer requests/totals. Full validation at
  boundaries checks the cached totals against the actual stored bed.

Full SYRUP regression: **834 passed, 231 skipped** in 281.01 s. Historical Phase7d
optional tests were excluded because they require exactly their own dependency.
The 1543 transaction physical/state/hash fields are bitwise identical; 132 failure
records agree in category and step, with 43 differences only in subset wording.

Scoped actual-MAPLE water/voxel/active-layer/ledger tests: **728 passed** in 19.59 s.
An initial broad collection included unrelated upstream docs imports absent from
this scoped snapshot; the explicit file-family run avoids those unrelated tests.
An in-sandbox scoped run stalled in Zarr's event loop after 105 passing tests and
was interrupted; the identical scoped suite completed outside the sandbox.

Final reviewed Plot1 storm: **217.246 s** loop, **93.190 s** MAPLE exchanges,
**410540 KiB** peak RSS. Versus the recorded accepted baseline, time fell 30.36%
overall and 49.36% for exchanges. All 103 saved numeric benchmark fields match
exactly, cached totals validate, and all class budgets close. Original measured
and final runs retain separate source hashes and outputs. Historical measurements
in `stage1_qualification.json` are explicitly labeled as earlier-candidate results.

## Review and limitations

Claude implemented the initial work. Codex independently reproduced and corrected
an early writable-cache defect and rejected a version that still rescanned all
class mass. Claude Fable then exhausted its model credits with no reset time given.
Codex completed bounded corrections; provider-offered Claude Sonnet independently
reviewed them in read-only mode (session `9c021e39-a5d9-42a7-a8ea-9ae44626e983`).
It identified the need to re-read storage after scatter; that correction was made
and the follow-up review reported no remaining implementation blocker. The review
also corrected ring-deposition wording: this legacy walk diagnostic is not an
additional sink in the active-pool balance.

The source-based legacy deposition algorithm, clipping source, fixed composition
and unlimited supply are explicitly restricted to benchmarking. Close agreement
with Fortran is not conservation validation. Eight characteristic bins are a
promising Plot1 option for 5% cumulative/class accuracy, not a universal replacement
or equivalence to the legacy method. Keep 32 as the default pending broader cases.

Whole mass copies and metadata passes remain. GPU kernel correctness is verified;
a full GPU storm and end-to-end GPU speedup are not established. Performance figures
are single-run observations with exact provenance, not repeated distributions.
No upstream edits, commit or push are part of this phase.
