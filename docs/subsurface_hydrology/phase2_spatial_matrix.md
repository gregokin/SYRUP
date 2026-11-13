# Phase 2 S2-B2.a — fixed explicit spatial persistent Green–Ampt matrix infiltration (CPU trial)

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_spatial_matrix_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-written, CPU only (NumPy and compiled Numba), uncommitted, awaiting Root's independent review. No GPU, CUDA, preferential, native storm/lifecycle/checkpoint or LEAF claim.

## What it is
`maple_syrup.spatial_matrix_infiltration` runs the EXISTING Phase 1 matrix law (`matrix_infiltration`, see its module note) on a real `SpatialSoilColumn` / `SpatialSoilState`:
same equations, branch order, thresholds, Newton helpers (tolerance for 8 terms, 64 iterations, same start), flags, per-cell and global conservation, area applied once. It adds no physics.

* Top layer: `fc_m0 = theta_fc0 * dz0` (rounded first), `deficit = max(fc_m0 - W0, 0)`, `b = psi * (deficit / dz0)`; matrix room `max(sat_m0 - W0, 0)` (SATURATION; the field-capacity gravity threshold is a different quantity and is not executed here).
* History: unponded / pre-onset admitted intake advances `J` UNSCALED; ponded post-onset admitted intake advances `J` by `post / g` for `g > 0` (nothing for `g = 0`); the separate ledger adds the ACTUAL `pre + post`; rejected capacity never advances `J`.
* Daily simplified recharge is not matrix intake and does not touch `J`.

```
op   = prepare_spatial_matrix(column, conductivity_m_s=K, suction_m=psi, vegetation_factor=g, memory_budget_bytes=...)   # K/psi/g: host scalar or exact (N,) float64
hist = spatial_matrix_history(op, wetting_m=..., cumulative_intake_m=..., memory_budget_bytes=...)                       # explicit values; nothing defaults to zero
res  = spatial_matrix_step(op, pool, standing_before, incoming_rate, state, hist, dt_s, memory_budget_bytes=...)         # SpatialMatrixStep
```
Units: m, s, m/s; `dt_s >= 0`; the stage preserves `state.t_s`; `dt = 0` is the identity for otherwise valid inputs. Every cell is validated (inactive included) with the common reasons (`MatrixStepError.bit`); a refused trial writes nothing and publishes nothing; non-finite global volumes are refused (typed) before the strict MAPLE bound helper.

## One body per backend
* NumPy: `matrix_infiltration._reference_step_body` (the old `reference_step` body, textually unchanged apart from reading a private context) with a `_StepContext(n_layers, n_cells, mask, cell_tol, newton_tol)` and lazy per-layer readers of the owner's raw maps (`fc = theta_fc dz`, `sat = theta_sat dz`, `cap_tol = coef sat`). The common wrapper passes the cached `(L,)` arrays as before. The budget is factored into `_resolve_core(area, L, N, packet)`.
* Numba: one kernel body in `matrix_infiltration_numba._build(numba, common)`, compiled twice by dead-branch pruning of the closure constant `common` (S2-A.3 technique). The common variant is reached through a thin wrapper that keeps the historical 25-argument signature; the spatial variant has an explicit read-only signature, reads the raw maps by strides, and holds two `(L,)` local rows (`sat`, `cap_tol`). The scalar Newton helpers `log1pmx` / `ga_increment` are the unchanged originals; no helper receives an array per element beyond the IR-inlined layer binder.

## Identity, binding and what it does NOT mean
`sha256` = digest of (tag `maple_syrup.spatial_matrix.operator.v1`, column input SHA, backend, dimensions, each law input's name, layout and bytes), streamed in 1 MiB chunks at preparation; steps revalidate metadata only; `verify_operator_identity` re-hashes cold. Histories carry tag `maple_syrup.spatial_matrix.history.v1`. The binding is to INPUTS: twin operators from identical inputs share the identity and accept each other's histories; scalar and per-cell layouts of the same values differ by design. It is not a water generation, accepted-transition receipt, exactly-once record, LEAF adoption, lifecycle, restart or storm claim, and the public history constructor cannot verify a genuine run boundary (it validates and copies explicit arrays; no reset/recovery/rescaling exists). No old-run identity is rebound. Legacy and spatial operators/states/histories refuse each other (typed); a CUDA owner is refused before any copy.

## Memory model (host payload bytes; not a reservation)
step: `(4N + LN)8` outputs + `16 N 8` rows + scratch (numpy `48 N 8`, numba `2 N 8`; plus `16 N` mask bytes and 4 packets of 17 doubles) + NumPy lazy rows `6 N 8` | Numba local rows `2 L 8` + 64 KiB allowance + reserve. history: `2 N 8 + 2 N` + allowance. prepare: `3 N 8` (upper bound) + `2 N` + allowance. Excluded: caller state/forcing/history, the owner's retained maps/mask, JIT, interpreter, allocator pools. Admission (quota, reading, reserve) precedes every allocation, conversion, compile and NumPy lazy temporary.

## Limits
Fixed explicit geometry only; no moving frame, water fate, texture-to-property provider, K-averaging, root frame or history policy is chosen. Ordinary refinement is not forced to be exact: `b` is re-evaluated at every call (splitting changes `W0` and hence `b`); exact splitting is only claimed where `b` is constant. Numba vs NumPy differ only by libm (declared rtol 2e-12 / atol 1e-14). Qualification is CPU author evidence; CUDA is a later slice (S2-B2.c).


## Independent qualification

Root accepted the fixed explicit-property CPU trial at package `f05dd79fbd1fab1768c46b91f66895ef6428735530021b7e717bdc238054bcf4` (122 files). See [phase2_spatial_matrix_acceptance.json](phase2_spatial_matrix_acceptance.json). The final 325-test spatial suite and original 41-test matrix suite passed independently. Each CPU backend matched all five saved output arrays in the 224-column preimplementation comparison exactly. Existing common CUDA matrix/preferential kernels retained all 704 saved arrays; this does not provide spatial CUDA infiltration support.

Warm memory observations at three layers and 50003 cells were 19162661 B (NumPy) and 9254443 B (Numba), below estimates of 31667976 B and 10866776 B respectively. Tracing is incomplete for compiled-code allocations and does not establish universal invisibility of the Numba heap; the two local layer rows are counted from source and checked separately through NRT allocation statistics. These are component payload checks, not whole-process or large-storm memory bounds.
