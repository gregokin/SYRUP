# CW-B51: explicit flow-end settlement (`deposit_mobile_at_flow_end`)

Integrated as an explicit option in the reviewed B55 runtime; the default remains false. The tests below retain their original candidate-stage context. Whole-component and large-GPU qualification remain open; see [current status](conservative_water/implementation.md).

## What it is

`CompletionPolicy.deposit_mobile_at_flow_end: bool = False` (appended last). It is an explicit **first-model
dry-again assumption** at the end of a water event:

> After rain has ended and every existing hydraulic quiet gate (maximum depth, surface volume, maximum unit
> discharge, outlet discharge; the terminal-storage policy unchanged) has held continuously for the unchanged
> `hold_s`, the remaining finite mobile load is deposited in place by the existing terminal transaction through the
> genuine MAPLE deposition.

It is **not**:
- a mass tolerance or a raised significance threshold;
- a change to wet laws, routing, bed or CSR defaults, or transport during the storm;
- a claim that the legacy physics deposits at that instant.

## Behaviour

- **False (default):** metrics, criteria, accounting, checkpoints and run identity are unchanged. The option is
  omitted from the bundle payload and from the run identity (`event_experiment.policy_identity`).
- **True:** only the mobile-mass part of the quiet criterion is not required.
  - The mobile pool is still validated as finite and nonnegative.
  - It is still reported in full (`max_mobile_cell_class_kg`, `mobile_by_class_kg`), plus `mobile_gate` and
    `mobile_below_significance`.
  - `finalize_event` is unchanged: one MAPLE deposition of the whole pool, refusal if anything stays mobile, the
    forced final commit with the exchange-history fold, quiet rechecked after the commit, the terminal sediment bound,
    and water and class budgets before the external water reset. Any failure leaves the wet input unmodified and
    publishes nothing.
  - `accounting['flow_end_settlement']` records class vectors and scalars only:
    - the policy and the assumption text;
    - the pre-settlement mobile load by class and its maximum;
    - the per-element significance limit, and whether the load was below it;
    - MAPLE's actual terminal deposition by class.

## Interfaces

- **API:** `CompletionPolicy(deposit_mobile_at_flow_end=True)`; must be a plain `bool`.
- **CLIs:** `maple-syrup` complete-event runner and windowed runner, `--deposit-mobile-at-flow-end`. The windowed
  runner also passes the existing `--terminal-storage-water`.
- **Checkpoint:** the field is optional (absent means False). A True bundle records it, and continuing with a
  different policy is refused. Unknown fields and non-bool values are refused.

## Why

The B49 probe, independently reproduced by the root, shows a valid state that the default can never complete:
- an actual MAPLE bed, saturated K = 1e-7 m/s, 5e-9 m depth;
- 1e-8 kg mobile in a `wet_no_law` cell (v = r = 0), no rain;
- all hydraulic gates pass and all budgets close.

That is a constructed fixture. How often real storms reach this state is not established.
