# Operand-qualified backend correspondence (CW-D1)

Historical task record; pending statements below apply to that task snapshot. See [the current implementation status](implementation.md) for executed results and remaining gates.

Module: `src/maple_syrup/backend_correspondence.py`. Tests: `tests/conservative_water/test_backend_correspondence.py`.
Status: comparator and tests written by Claude (file tools only, nothing executed); the root runs the tests and
reviews independently. **This slice is not a storm pass and changes no gate result.** The evidence the comparator
needs is not yet produced by the event runner (section 5).

## 1. Three questions that must not be conflated

| Question | Rule | Where it lives | Status |
|---|---|---|---|
| Is mass conserved on each backend? | per-class closure at MAPLE's summation bound with actual accounting calls and real reservoir operands | `maple_syrup.conservation`, `SedimentEventResult.closure` | unchanged, passes on both backends |
| Do the two backends' saved fields agree to a declared numerical identity? | `rtol 1e-10 / atol 1e-14` (sediment), `2e-12 / 1e-14` (water), exact integer controls | the matched-comparison scripts in the task folder | the declared strict gate; its recorded failure on the ordered-height full runs **stands** and is not reinterpreted here |
| Could two correct FP64 histories of the same physical exchanges differ by this much at this cell and class? | MAPLE's accumulated-roundoff rule, operand-qualified, capped at one mass resolution | this module | new, evidence not yet recorded |

The third question is the one the user's instruction points at ("MAPLE's tolerances, physically close enough,
not legacy identity"). It uses an existing MAPLE rule with no new constant, which is why no additional scientific
permission is needed for the rule itself; the *application* to backend correspondence is new and is documented
here as such, separately from conservation.

## 2. Derivation from the unchanged MAPLE policy

MAPLE distinguishes (`maple/surface/voxels/_numerics.py`, module docstring): local roundoff (`16 eps * scale`),
summation roundoff (`summation_error_bound_kg(n_terms, term_scale) = 4 n eps term_scale`, additive) and the
model's physical significance threshold `numerics.mass_resolution_kg`, which is never a roundoff allowance.

For a cell/class that has been through a sequence of physical exchanges, MAPLE's ledger reconciliation
(`maple/surface/topographic_commit/commit.py`, `_reconcile_ledger_with_physical_state`) bounds the roundoff that
the sequence may have accumulated:

```
fp64_tol  = per_call_eps_multiplier * max(n_physical_touches, 1) * eps * local_scale
local_scale = max(gross_physical_activity, |reference_inventory|)        per cell and class
tol       = min(fp64_tol, mass_resolution_kg)
```

`per_call_eps_multiplier` (8.0, measured and documented in that function) is read by the comparator from the
function's own signature rather than copied. `n_physical_touches` counts only exchanges where actual mass moved at
that cell/class; activity elsewhere cannot inflate a cell's allowance; a cell with no recorded touch gets exactly
the one-touch floor; the hard cap means a difference the model itself calls resolved can never be accepted.

Two backends that each satisfy this rule against the same physical truth can differ by at most the sum of their
allowances. The comparator therefore uses

```
per side  b_s = min(k * max(N_s, 1) * eps * max(G_s, |R_s|), m_res)
paired    B   = min(b_cpu + b_gpu, m_res)
accept    |x_cpu - x_gpu| <= B      elementwise, per cell and class (broadcast over the voxel axis)
```

**Stricter cap.** The sum of two capped allowances would be `2 m_res`; the paired bound is capped at **one**
`m_res`, so a resolved difference (`> mass_resolution_kg` in `classify_mass_resolution_difference`'s sense) always
fails regardless of touches. This is deliberately stricter than the arithmetic sum and is the only place the
comparator departs from a literal sum of the per-side rules.

What the criterion rejects, so it is not a blanket pass: any difference above `m_res`; any sub-resolution
difference at a cell/class whose recorded touches and operand scale do not account for it; any routing or
integer-control difference (bytewise); malformed or incomplete evidence (refused outright).

## 3. Derived net diagnostics

Quantities such as the per-cell/class bed change `(voxel.sum(axis=2) + active) - initial` are differences of
operands of hundreds of kilograms with nets near zero. Comparing two backends' *saved* values of such a quantity
against a bound scaled by the net is mis-scaled: a one-ulp difference of a 360 kg operand is 5.7e-14 kg whatever
the net. The comparator instead:

1. recomputes the quantity on the host from each backend's saved operands with ONE fixed expression;
2. bounds each recompute's own roundoff with `summation_error_bound_kg(nv + 2, largest operand)` where the largest
   operand is the column total, the active mass, the initial inventory or the result at that cell/class;
3. propagates the measured operand differences (`sum_v |dvoxel| + |dactive| + |dinitial|`), each of which is
   already judged by the reservoir comparison;
4. reports a backend's saved derived array against its own host recompute within that side's roundoff bound.

The derived comparison is a consistency report; the physical arrays remain the judged quantities.

## 3a. What the report claims, and what it does not

`correspondence_report` returns three separated sections and one verdict:

* `physical_reservoirs`: the operand-qualified comparison of the physical arrays actually supplied, with
  `examined`, `expected` (if the caller declares the full set), `missing`, `all_examined_passed` and per-field
  results.
* `exact_controls`: bytewise integer/routing identity, with `examined` and `all_examined_passed`.
* `derived_diagnostics`: host-recompute consistency reports (section 3) with `all_examined_consistent`; they
  never gate the verdict (`gates_physical_verdict: False`).
* `physical_correspondence_passed`: True only when at least one physical reservoir was examined, every examined
  reservoir passed, every supplied exact control passed, and no expected reservoir is missing.
  `physical_correspondence_reasons` names why it is False. A report built from controls only, derived operands
  only, or empty collections cannot claim physical correspondence; a field not listed under `examined` was not
  compared and is never counted as passed. Missing physical fields are not an implicit success of a storm.

**Evidence binding.** `BackendEvidence.binding` is a caller-recorded mapping (dependency and source pins, receipt
identity, per-array sha256 of the saved rows) echoed verbatim under `evidence.<side>.binding`. Constructing the
evidence object proves nothing about the history behind it; `history_complete` and `binding` are declarations
that a reviewer verifies against the run receipts and pins. The comparator only refuses what is malformed; it
cannot and does not establish that the touches are the genuine accumulated counters.

## 4. What the comparator refuses

Evidence is refused (exception, not failure) when: arrays are not host `numpy.ndarray` (no implicit device
transfer); masses are not float64, finite and nonnegative; touches are not integer and nonnegative; shapes do not
share one `(ny, nx, nc)` layout; mass resolution is not positive and finite or differs between sides;
`history_complete` is not `True`; a source string is empty; a reservoir is not rank 3 or 4; an exact control is a
floating array. Nothing is inferred: no touches from step counts, no missing history treated as zero.

Hydrologic quantities (depth, discharge, velocity, volumes) are not masses and are not compared here; they keep
their own comparator and bounds.

## 5. Required integration before any correspondence claim (not done in this slice)

The comparator needs, per backend and per cell/class, accumulated over the WHOLE event:

* `n_physical_touches`: MAPLE's ledger counter, captured immediately before every commit resets it and summed over
  commits, plus the interval after the last commit. MAPLE's current commit result does not expose the pre-reset
  counter or gross activity by cell; **B3b** must add exactly these to the commit result before the reset, in the
  shared tree, with its own tests.
* `gross_physical_activity_kg`: the same capture of the ledger's gross activity (summed over the process axis).
* `reference_inventory_kg`: the per-cell/class inventory at the event's reconciliation reference (initial bed).
* Coverage: every water call (pickup and deposit/export), every interval, pending-elevation and turnover commit,
  the forced final commit, avalanche transfers inside commits, and the dry-reset completion step; the integration
  sets `history_complete=True` only when all of these are covered.
* Persistence: the accumulators are part of `SedimentEventResult`, written to the saved outputs, carried through
  checkpoints and resumed continuations, and preserved across the final/dry reset; source and dependency pins and
  the NumPy/CuPy versions are recorded in the receipt alongside.

Until that exists, a caller has no genuine evidence; the module refuses rather than estimates.

## 6. Worked reading of the recorded ordered-height failure (no values hard-coded in code)

The fresh full GPU runs failed the strict gate on one voxel entry (1.9e-14 kg at a 64 µg holding) and on the
saved derived bed change (1 to 4 ulps of 64 to 512 kg operands). Under section 3 the derived failure is a
comparator-scaling artefact to be re-evaluated from host recomputes. Under section 2 the voxel entry can only be
judged once the actual touches and gross activity at that cell/class are recorded; the comparator will accept it
only if MAPLE's rule at that cell's own operand scale accounts for 1.9e-14 kg, and reject it otherwise. The strict
gate's recorded failure remains in the record either way.
