# CW-B65: accounting-only post-transport retention (private candidate)

**Parent:** the stopped B63 candidate. `sediment_event.py` was `96ecaff2…`; CUDA `d13f4bf5…` is unchanged and every
other package file equals B63.

**Trigger (measured B64):** the full transport and its validations passed. The step then refused inside the deposit
call: MAPLE `availability/exchange.py:141` (`shortfall = requested − capped`, 107,250,176 B with 9,055,316,480 B
retained, cap 9,092,045,209 B).

The design is the reviewed CW-B56 one, with root's corrections (`cw_b56_b58_root_design_qualifications.json`).

## Change (`src/maple_syrup/sediment_event.py` only)

- **Private record `_PhaseReconcileInputs`:** the SAME objects reconciliation reads (the post-transfer pool, the
  deposition and export requests, the kernel's remaining pool, and the phase), plus the storage kind.
- **Private bodies** `_reconcile_sparse_phase`, `_reconcile_dense_phase` and `_reconcile_phase_inputs`, plus
  `_reconciliation_bound_check`, run on that record with unchanged operations, bound, rules and diagnostics.
- **Public `reconcile_phase`:** signature, docstring, type guards and messages unchanged; it builds the record and
  calls the same bodies.
- **`_sediment_coupled_core`, full form:** byte-for-byte the original statements (`with_water`, the inline demand, the
  deposit call, `reconcile_phase(characteristic, …)`, the complete `SedimentCoupledStep`).
- **`_sediment_coupled_core`, accounting form,** in the same call order:
  1. `with_water`;
  2. `demand = water_demand_from_transport(transport, zeros)` (this demand carries all four faces);
  3. capture `TransportAccounting.of(transport)` and `_PhaseReconcileInputs.of(characteristic)` (the same objects the
     return-time records read);
  4. drop `transport` and `characteristic`;
  5. the deposit call (MAPLE validates the four faces);
  6. capture `DepositAccounting.of(deposit)`;
  7. drop `deposit` and `demand`;
  8. `_reconcile_phase_inputs(...)`;
  9. return the captured records.

**Unchanged:** arithmetic, order, validations, host reads, policy, defaults, tolerances and caps. No MAPLE change, no
ownership tokens, no B48.

## Memory — COMPUTED, not measured; no fit claim

- **At the deposit call:** the 8 full-domain diagnostics (decay, settled, internal in/out, divergence, mobile-before,
  arrival, crossing) are released. On the CUDA CSR path at Chastre that is 8 × 107,250,176 = **858,001,408 B**.
  The 2 per-class totals are released with them: 2 × 512 B pool-rounded (nc = 6), computed.
- **At reconciliation entry:** additionally the four faces (429,287,424 B total, from the corrected y-face sizes) and
  the deposit result's arrays that the accounting records and bed do not keep.
- B64 exceeded the cap by 70,521,447 B at its request. The device MemoryHook test derives the exact expectations from
  the actual allocations.

## Interception-seam consequence (stated)

The accounting path no longer calls the module-global `reconcile_phase`, as required: it calls the private inputs body
directly after dropping the complete result. Two inherited tests intercept that global and now fail:
- `phase7b/test_event_integration.py::test_reconciliation_failure_publishes_nothing`: the injected refusal is not
  reached;
- `cw_b20/test_accounting_step.py::test_owners_are_released_before_deposit_and_reconciliation[accounting-*]`: the
  reconcile snapshot is not taken.

Both pass unchanged on the B63 parent. The B53 profiler's `reconcile_phase` phase record is affected the same way.
`tests/cw_b65` covers the private seam: identical refusals, and liveness at `_reconcile_phase_inputs`.

**Correction 1 (tests only):** both inherited tests now intercept the entry actually invoked:
- the cw_b20 owner test uses `_reconcile_phase_inputs` for the accounting form and the public `reconcile_phase` for
  the full form, with all its liveness and caller assertions unchanged;
- the phase7b refusal test uses `_reconcile_phase_inputs`, because its `evolve_sediment_event` driver steps through
  the accounting form. It also asserts that the injected refusal ran exactly once, on a `_PhaseReconcileInputs`.

A B66 observer must wrap the private body. The immutable B53 observer is not edited.

## Tests

See `tests/cw_b65` and the task reports. After correction 1, the record is a complete recursive record, so every
nested scalar is compared exactly:
- dataclasses, dicts, tuples and MAPLE slot handles;
- raw array bytes and NumPy scalar bytes;
- native floats by raw bytes (signed zero distinguished);
- modules by name.

Negative controls perturb one nested counter and one array element by one ULP.

The device MemoryHook test also requires a ZERO live-byte difference at return, with its observers released. It
asserts that the deposit-call release is EXACTLY:
- the 8 diagnostic D grids, plus
- the 2 per-class `(nc,)` totals (`mobile_before_by_class_kg`, `mobile_after_by_class_kg`), which no accounting
  record reads.

The released allocations are alias-deduplicated. A CPU (NumPy buffer) test confirms the same composition in all
three cases.

**Correction 2 (tests only).**

*Counter negative controls:* the expected raw array-leaf relation now follows each counter's actual, source-derived
representation:
- regime counts and dense reconciliation counters (from `xp.sum`) are 0-D CuPy arrays on CuPy, so exactly that one
  leaf changes;
- sparse counters (`np.int64`) and every NumPy-backend counter are scalars, so the leaves stay identical.

The recursive record must refuse every perturbation.

*Reconciliation memory:* the expected reconciliation release no longer counts the core's `zeros` local
(`_sediment_coupled_core`, bound once and never deleted before return). `water_demand_from_transport` passes that same
object as the demand's `requested_removal_by_cell_class_kg`, so it stays alive at reconciliation in BOTH
implementations. One pool-shaped allocation, which is 512 B pool-rounded at the toy shape, explains root's measured
11776 B against the expected 12288 B.

A CPU weak-reference test proves the alias and its liveness, and measures B65's actual releases exactly; the previous
expectation is kept as a negative control. The device test checks the pointer of the caller's `zeros`.
