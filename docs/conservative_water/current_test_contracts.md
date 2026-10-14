# Current default-test contracts (CW-E47C1, 2026-10-14)

After the CW-E47 adoption of SYRUP 5f (`5f85a989…`) with the MAPLE 24c runtime (`24c6e147…`), 12 tests in the
default suites (`tests/phase7b` and the four conservative families) failed on both the root tree and the frozen
accepted 5f tree. **No production defect was established.** The tests encoded behaviour that the accepted runtime
deliberately changed:
- CW-B104 net-elevation-only rerouting;
- CW-B76 / B93 fused windowed CSR accounting;
- CW-B102 optional deep-publication control fields;
- the non-rerouting final forced commit.

CW-E47C1 corrected **only these six test files**. The production runtime, MAPLE's default
`pending_elevation_trigger_m` (0.00075 m, `>=`), every tolerance and every completion gate are unchanged. No routing
counter or graph is patched to fabricate a reroute.

Off-default historical static guards are **not** part of this correction and remain recorded failures, with no
waiver:
- `tests/cw_b54`: 4 AST / diff guards against B52;
- `tests/cw_b20`: B15 driver identity and an intermediate-lifetime case.

The strict CPU–GPU records also stay as recorded: 89 failures in current Plot1, and 4/7 in the earlier 8ac/M94
rotated Chastre comparison. All are qualified physical agreement only.

## Genuine reroutes: physical fixtures, default threshold

The below-threshold fixtures commit (time-interval accounting commits and the final forced commit) but never reach
the net threshold, so **0 reroutes is correct**. Each is kept as a **negative control**. A stronger *physical* storm
on the *same actual fixture*, at MAPLE's default threshold, makes the **genuine crossing**.

The positive tests then prove the crossing from the commit records:
- every rerouted commit has `net_displacement_m >= reroute_threshold_m == 0.00075`;
- every non-rerouted commit stays below it;
- the adopted graph is the rerouting commit's graph.

| Fixture | Negative control (unchanged) | Genuine crossing (exploration evidence `cw_e47c1_work/probe_*.json`) |
|---|---|---|
| phase5 valley 4×5, commit interval 10 s | 72 mm/h, 45 s: 5 commits, 0 reroutes, max net 5.19e-5 m | 600 mm/h, 90 s: 15 commits, 1 reroute at 0.000755 m (`n_net_commits` 1) |
| flat-basin pits | 72 mm/h for 40 s, to 120 s: 2 commits, 0 reroutes, max net 3.80e-5 m; both flats stay terminal | 300 mm/h for 120 s, to 180 s: `net_elevation` reroute at 77 s, 0.000751 m; terminal storage 2 → 1 |
| resident-window CSR chain (`window_inputs`), commit interval 2 s | original 2 s window: accounting commits only | 1000 mm/h, 90 s: `net_elevation` reroute at 69 s, 0.000751 m, after 34 `maple_trigger` accounting commits |

## Per test

**`tests/phase7b/test_event_integration.py`**
- **Old assertion:** `n_graph_changes >= 1` on the 45 s storm, which assumed a reroute at every commit.
- **Now:**
  - `test_accounting_commits_below_the_net_threshold_keep_the_graph_and_the_phase_progress` (negative): commits occur, 0 reroutes, the graph / network binding is kept, and phase positions are not reset.
  - `test_reroute_keeps_phase_progress_and_counts_steered_cells` (genuine crossing): budgets close, the refreshed graph is adopted, the phase is carried over, steered cells stay a subset of rerouted cells, and positions are kept.

**`tests/conservative_csr_event/test_csr_event.py`**
- **Reroute rebind:** dense vs CSR raw-byte equal for both storms. On the negative storm the CSR phase keeps its INITIAL network binding. On the genuine crossing it is explicitly rebound to the refreshed network.
- **Control identity / dense checkpoint schema:** the default control omits EXACTLY the opt-in keys holding their defaults: `phase_storage`, `phase_max_nnz`, `record_exchange_history`, `deep_publication='step'` and `deep_publication_high_water_records=None`. The expected set is asserted to equal `checkpoint._CSR_CONTROL_FIELDS ∪ _OPTIONAL_DEFAULT_FIELDS['SedimentEventControl']`; there is no catch-all.
  - A non-default `'boundary'` + 3,000,000 publication is part of the identity with exact types.
  - New strict refusals: unknown publication, boundary without high water, boundary on dense, bool / float high water, high water under `'step'`.
  - Checkpoint decoding refuses an unknown control field (`unknown checkpoint type or fields`) and a wrongly typed publication (`deep_publication must be one of`). An explicitly present default value decodes like any field.

**`tests/conservative_water/test_coupled_backends.py`** (lifecycle)
- **Contract:** the CUDA hydraulic context is re-prepared ONLY on a genuine reroute (`commit.rerouted`, `sediment_event` commit fold). Accounting commits and retries re-use it, and every commit still records its transfers. `n_graph_changes` is not used as a substitute.
- **Tests:**
  - negative storm: 1 preparation;
  - genuine crossing: `1 + n_reroutes` preparations, logged at the rerouting commits' times with their graph digests;
  - continuation and foreign-graph re-preparation, and the convenience form;
  - retries: accounting commits plus retries keep 1 preparation;
  - preparation failure: the failure is actually executed at the first genuine reroute (the second preparation request is for the refreshed graph). Nothing is published, the context is not adopted, and the transfers of exactly the commits up to the refused one are recorded.

**`tests/conservative_water/test_windowed_storm.py`**
- **Old problem:** the phase / routing late-failure injections were never reached, because no reroute happened. The genuine reroute builds its graph with `routing_graph_from_elevation` inside `commit_bed`; `refresh_routing` is the no-routing-source path.
- **Now:**
  - The run is paused one second before the genuine crossing (no forced final commit).
  - The continuation's first candidate reroutes. A hook on the actual reroute-only stage (`phase_for_network` of a rerouted commit, or the reroute graph rebuild in `commit_bed`) raises.
  - The test proves the hook was reached once, at the reference reroute time, with net ≥ threshold. The paused state and the deep store are unchanged.
  - The ordinary accounting-commit history failure is a separate test.
  - A positive-control test asserts the genuine crossing itself.
  - Limit: this small fixture's deep store is not written in that window even without a refusal, so the store-equality assertion is retained but weak here.

**`tests/conservative_water/test_intermediate_lifetime.py`**
- **Dense window storage:** the original two-call path (the `apply_bed_demand` pickup window / ledger arrays) is unchanged.
- **CSR window storage:** the fused path (CW-B76 paired MAPLE transaction, one `apply_bed_pickup_transport_deposit` per attempt, with no standalone pickup objects). Transparent wrappers on the real call and its callback check that the step-owned transients die before the next attempt: the MAPLE `PickupView`, the paired deposit result, the deposit demand, the physics demand and the host transport deposition request.
- **Negative controls:** deliberately retaining each owner is detected, together with exactly what it owns.

**`tests/conservative_pits/test_terminal_storage.py`**
- **Flat basin:** the original storm is the negative control (both flats terminal). The genuine crossing fills the fed flat, the first reroute removes one terminal cell, and budgets / terrain / phase are checked over the full run.
  - The strict terminal-cell claims are checked over the interval in which (1, 0) WAS terminal: the same storm ended at the genuine reroute time. Afterwards it is a moving cell that legitimately erodes.
- **Final commit:**
  - The forced final commit (`final_forced`) is a genuine MAPLE accounting commit. Below the threshold it keeps the caller's graph and storm (the `_commit_and_reroute` contract), and the quiet gate is re-evaluated on the committed bed (new positive test).
  - **Correction (CW-E47C2).** The CW-E47C1 refusal test forged `rerouted=True` and derived objects on a below-threshold commit. That is withdrawn: it did not exercise a genuine reroute. It is replaced by a GENUINE net crossing at the forced final commit:
    - **Fixture:** actual pit chain with no rain. A wet moving cell (2, 0) has initial depth 5e-9 m (≤ the quiet depth gate of 1e-8 m) and zero infiltration, and carries a finite 0.5 kg mobile pool set through MAPLE's water state.
    - **Why the pool persists:** after rain a Re ≤ 500 cell has no transport law (no capacity / settle rule, zero deposition rate). The explicit `deposit_mobile_at_flow_end` policy waives only the mobile quiet gate.
    - **Ready progress:** quiet at 11 s with water / class / request budgets closed, 0 commits and the pool intact. It is captured through a transparent `finalize_event` wrapper, i.e. the progress `complete_event` really finalizes; no hold is fabricated.
    - **Final commit:** `finalize_event` deposits the pool through MAPLE (+1.6 mm), and the forced final commit (`final_forced`, `force=True`) genuinely reroutes. Probe values: net 0.0016 m ≥ 0.00075 m, `graph_changed` true with a new graph digest; slopes change, receiver directions don't.
  - **Positive test** (`test_genuine_net_crossing_at_the_terminal_commit_reroutes_and_completes`): the refreshed graph / network / storm (discharge recomputed from the new conveyance at unchanged depth, 4.8e-13 m²/s in the probe) are adopted. The post-commit quiet gate is evaluated on that rerouted state and still holds, and the dry handoff completes with closed accounts.
  - **Refusal test** (`test_reactivated_moving_flow_after_a_genuine_terminal_reroute_refuses_the_handoff`):
    - The genuine reroute (`rerouted` true, net ≥ threshold) is asserted inside the wrapper BEFORE any change.
    - Only the moving pool cell's discharge in that genuinely rerouted result is then raised to twice the unit-discharge gate. This is a transparent failure injection; no flag, graph, network, grid, digest, source, gate or counter is altered.
    - The post-commit quiet check fails on `max_unit_discharge_m2_s` (moving depth still within its gate), and `finalize_event` refuses with `invalidated completion`. The genuine ready progress and the caller's initial state stay unchanged.
  - **Why not a natural refusal:** natural reactivation was not obtained. The pit settles its own load every step, so terminal deposition cannot fill the pond and make it a moving cell. The recomputed discharge of the rerouted moving cell stays far below its gate.
  - **Not a physics claim:** the refusal is a test of the post-commit quiet check on a genuinely rerouted terminal commit. It is not a claim that real events reactivate this way. An event-end reactivation refuses the handoff (input kept wet); it does not run an automatic recession.
  - The C1-era non-rerouting storm injection could never fire either. A non-rerouting commit's storm is never adopted, per the `_commit_and_reroute` contract.
  - Evidence: `agent_handoffs/tasks/conservative_water/cw_e47c2_report.md`, `cw_e47c2_work/probe_final*_x0*.json`.

## Runs

The task records are in `agent_handoffs/tasks/conservative_water/cw_e47c1_report.md` and `cw_e47c1_runs/`, with the final-commit correction in `cw_e47c2_report.md` and `cw_e47c2_runs/`. GPU parametrizations skip without a device ("no claim"). Root runs the GPU twins after this writer closes.

Independent ROOT verification (CW-E48, 2026-10-14): all four conservative CPU families and Phase 7b pass, with device skips explicit; net guards pass. Actual GPU storage/event, dense/fused lifetime and pit-completion checks pass. Fresh-process CPU/GPU continuations match 119/140 arrays exactly, including new deep publication after resume, and 19 corruption controls each pass. See [qualified acceptance](acceptance_20261014.md).
