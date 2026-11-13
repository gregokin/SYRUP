# Phase 2 D2 — floating-column matrix / preferential intake and requested root withdrawals

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_floating_intake_withdrawal_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-implemented, CPU/host verified only (NumPy, Numba, and the CUDA sources compiled and executed on the HOST through a plumbing double). **Not independently accepted. No GPU was initialised, probed or
run by the author; the CUDA kernels have no hardware evidence.** No native-storm, lifecycle, restart, observation or save-driver adoption is claimed (slice E / Gfinal).

## What it is

Typed entry points that apply the EXISTING intake and withdrawal laws to a D1 `FloatingSoilColumn`/`FloatingSoilState` (partial or zero porous soil, inactive placeholders). The physical text is shared: one NumPy body,
one Numba kernel body (a third compile-time variant) and one CUDA body per law. Nothing is copied; no floating owner masquerades as a spatial owner (the old public classes keep refusing the new ones).

| Law | Module | Shared body (unchanged) | Floating hook |
|---|---|---|---|
| Matrix Green–Ampt with Tietjen factors (persistent `J`, cumulative actual intake) | `floating_matrix_infiltration.py`, `..._cuda.py` | NumPy `matrix_infiltration._reference_step_body`; Numba `matrix_infiltration_numba._build(numba, False, True)`; CUDA `matrix_infiltration_cuda.render_source` | floating readers (`h`, `fc = theta_fc h0`, `sat = theta_sat h`, safe `dz0`) |
| Preferential (one aggregate rate, fixed receiving weights) | `floating_preferential_infiltration.py`, `..._cuda.py` | NumPy `preferential_infiltration._reference_step_body` + `_derive_pathway`; Numba `_build(numba, False, True)`; CUDA `preferential_infiltration_cuda.render_source` | floating readers (`sat = theta_sat h`) |
| Requested root withdrawals and layer availability | `floating_transpiration.py`, `..._cuda.py` | NumPy `subsurface_transpiration._fused_numpy`; Numba shared scalar helpers `_withdraw_cell`, `_withdraw_available`; CUDA `render_withdrawal_source` | floating readers (`res = theta_res h`, `sat bound`) |

`h = max(0, min(D, e_{l+1}) - e_l)` is the actual porous thickness (D1). Only the existing private Numba modules `matrix_infiltration_numba.py` and `preferential_infiltration_numba.py` gained a private `floating`
compile-time variant (dead-branch pruned: the default variants keep their signatures and arithmetic; a measured ABAB warm comparison shows ratios 0.98–1.01 of the previous kernels). No existing CUDA source, ABI or common
emitted text changed; the floating CUDA wrappers are the spatial head/tail re-targeted by fail-closed textual substitution (each fragment must occur exactly the expected number of times).

## Semantics (as assigned)

* **Matrix.** Capacity `theta_sat h0`; deficit `max(theta_fc h0 - W0, 0)`; `b = psi (deficit / h0)`. A wholly-rock column (`h0 = 0`) has top capacity 0: actual intake 0, offered pool unchanged, `J` and the cumulative
  intake unchanged; the denominator stand-in 1.0 is used ONLY there (the deficit is `0 - 0`); a positive `h0`, however small, is never replaced. `J` is a coordinate, not water.
* **History rebinding** (`rebind_floating_matrix_history(old_op, old_history, new_op)`): explicit and pure. Validates the old history against the OLD operator (identity, type, shape, device, finite and `>= 0`), requires
  the same dimensions, backend/device, active mask, nominal edges and cell area, then returns exact copies of `J` and the cumulative intake bound to the new operator identity (changed K/psi/g/properties/depth are
  legitimate). Never resets, rescales or recovers. A receipt (`FloatingMatrixRebinding`, `reset=False`) names both operators; it is a PROPOSED binding, native generation adoption is slice E. CUDA downloads nothing
  (two 8-byte validation diagnostics; receipt sums are `None`).
* **Preferential.** Receiving weights are fixed (never renormalised over porous layers). A receiver with no porous capacity admits nothing; its share stays on the surface (the existing rejected-share rule). Surface
  water already credited by matrix intake must be passed as the reduced pool (one aggregate supply).
* **Withdrawals.** `available = max(W - theta_res h, 0)` (0 where `h = 0`); `actual = min(request, available)`; demands stay at the fixed cell/depth fractions (no reallocation, no renormalisation). LEAF
  aggregates its plants' requests (its sparse root connections) and passes ONE aggregate `(L, N)` request; each plant's accepted amount is `request_p * scale` from the returned fulfilment fraction. There is NO
  dense per-plant `P * L * N` helper in production (removed in correction01): the native coupling must use the sparse connections and the aggregate grid withdrawal/scale.
* **Inactive placeholders.** Finite and `>= 0` is validated on every cell, but no capacity or rock law applies: their bound is `+inf` in the readers, they are bit-preserved and excluded from active budgets.
  Active `h0 = 0` water must be 0 (the D1 state policies: `floating_initial_state` strictly unsaturated, `floating_continued_state` allows a saturated continuation).

## Backends, memory and transfers

NumPy and Numba (host); CUDA through the typed owner. For a CUDA owner: the current device is checked FIRST (before the state/request metadata, readers, copies and accounting); metadata checks, then admission (device
and host quota/reserve/reading separately) BEFORE any value scratch; one fixed host packet per hot call (matrix 136 B raw + 136 B rebuilt, preferential 72 B + 72 B, withdrawal 64 B, availability 8 B), counted in the
admission; law/history validation uses one tiny kernel and one 8-byte diagnostic read; the cold identity hash streams the law arrays through a bounded counted staging buffer; the cold derivation of the
preferential receiver map uses a bounded 32-byte reducer (no library reduction). No 64-layer cap, no layer-sized per-thread array, no full bound tensor, no hidden CUB scratch, no per-step hashing, no download of `W` or
properties. All lanes (padded and inactive) reach every barrier. A CUDA owner has no host mask: the withdrawal closure bound uses the cell count in place of the active count (a larger, still valid bound).

## Tests (`tests/phase2_floating_intake_withdrawal/`)

Root's independent controls (matrix D=0.1 and D=0, preferential `[0,1,1]` rejected share, withdrawal `[0.015,0.010,0]`/0.625 and all-rock), common full-soil references (bitwise on NumPy), NumPy/Numba equivalence
for 1/2/3/65 layers and block edges, partial/sliver/zero depth, masks and placeholder bit preservation, invalid metadata/values and unchanged inputs, wrong owner/state, history binding/rebinding/refusals, zero-time
identity, saturated continuation vs unsaturated initialisation, conservation ledgers, demand-proportional split from the aggregate scale; HOST-compiled CUDA tests for each law (g++ C++20 `std::barrier` threads plus a numpy-backed CuPy
double: equivalence with NumPy, exact allocation/transfer accounting, device gate and admission order, source composition); and `test_gpu_definitions.py`, the GATED actual-GPU definitions (`SYRUP_RELEASE_GPU=1`,
written, not run: equivalence, N = 50003 allocation/transfer/plateau/release, kernel resource attributes, second-device gate, rebinding). The host double is not hardware evidence.

## Not included / limits

No site properties, texture inference, interface conductivity, root plasticity, E/T, gravity or time stepping in these stages; no native storm/restart/observation integration; the CUDA kernels are unrun.
