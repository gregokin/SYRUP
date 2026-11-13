# Phase 2 current design: surface-following (floating) soil profile — 2026-11-13

Status: user-agreed DESIGN, not implemented, not qualified. Source of the choices: `floating_profile_decisions_20261113/user_decisions.json` (task directory). This page supersedes, for the new floating treatment only, earlier proposals that eroded soil water disappears; historical statements, receipts and tested components are preserved unchanged. Phase numbering (1, 1.5, 2, 2.5, 3) is unchanged. Phase 2 remains UNCOMMITTED and is NOT whole-phase accepted. Main verified to contain Phases 1 + 1.5 (merge `5c7c7238caac0e759a57444a498ab8f7147ab725`). Manuals are regenerated at the end of the whole phase, not now.

## 1. User-agreed choices

* **Geometry.** Layers have fixed, user-defined thicknesses referenced to the CURRENT (evolving) surface.
* **Moisture preserved = water mass / equivalent volume per layer**, not saturation fraction or suction. Geometry/property reassignment (surface rise or fall) preserves each layer's water inventory; implicit physical relocation of water with the floating profile and layer averaging as the initial hydraulic representation are accepted.
* **Roots translate with the soil.** Existing root geometry moves with it, keeping fixed depth/cell fractions and proportional demand competition; no root plasticity, no implicit re-normalisation.
* **No geometry-only water loss.** Floating-mode geometry alone creates no erosion-water sink. Real, separately accounted losses (runoff/export, bottom drainage, E, T) remain.
* **Hydraulic properties** come from the actual overlap/mass-weighted MAPLE texture plus the agreed explicit-input Saxton–Rawls (2006) method; they are not a copied bed and not a unique parameterisation.
* **SAT and FC are distinct.** Gravity movement starts above FIELD CAPACITY; storage is bounded by SATURATION.
* **Gravity drainage is always on in new production entry points.** Open bottom is the default; closed bottom is an explicitly selected special case. Historical isolated-test switches (`gravity_drainage=False`) and old profiles/defaults are NOT removed or rebound.
* **Initialisation.** The intended production policy is unsaturated initial layers, enforced at initialisation only, with no arbitrary moisture epsilon. Transient saturated states, continuations/restarts and test fixtures stay legal within the existing bounds.
* **Bottom drainage** is an explicit external loss counted exactly once, never omitted and never counted as internal percolation.

## 2. Correction of the current-code description

The native layered storm (`layered_storm.py`, `ss.step`) already runs zero-E column gravity/redistribution at EVERY hydraulic timestep, during rain and during recession. Rainfall end is not event completion. There is no new rain-off reset, drain-all operation, double drainage or daily interval. Water equation (surface + soil, internal percolation cancelling):

`inventory_before + rain + other explicit inputs = inventory_after + runoff/export + bottom drainage + E + T`

## 3. Capacity reconciliation after a property update (user-approved design 2026-11-13; not implemented)

A pure property update retains `W` per layer but can render a NEW saturation depth smaller than the retained water. The standard step validates BEFORE its gravity cascade, so a staged implementation is NECESSARY; the existing drainage is not already a working or qualified solution. Approved treatment:

* **Staging.** Retain per-layer water through the geometry change; stage the new properties and reuse the existing top-down gravity cascade on the NEW field capacity BEFORE the final new-capacity validation and adoption. No partially accepted invalid state.
* **Open bottom.** Forward water above NEW FIELD CAPACITY through all layers; bottom export is recorded exactly once. Because FC <= SAT, overfilled layers are resolved; the existing gravity law may transfer more than the saturation excess.
* **Closed bottom.** Capacity-limited downward redistribution; the unavoidable remaining excess returns to SURFACE water. It is never exported through the closed bottom and never erased, and it is not returned early if deeper storage can accommodate it.
* **Order/time.** Combine with ordinary drainage wherever possible: one cascade for the same staged update, no duplicated normal pass, no extra E/T/diffusion/rain/drain/day interval, no artificial elapsed time. The exact safe integration ordering is still engineering/design work.
* **Conservation.** Bottom export is external; surface return is internal to total surface + soil; internal percolation cancels. Residual surface water stays truthful and actual water-recession completion is required before an interevent; overflow is never erased to permit a handoff.
* **Efficiency.** Changed columns only, one bounded pass; one carry per cell through the L layers for open bottom; the existing bounded capacity treatment for closed bottom; compiled CPU and GPU-resident with no whole deep-bed copies, history copies or per-step host transfer.

## 4. Remaining choices versus engineering

**Science choices now RESOLVED by the user (2026-11-13, `bedrock_refresh_decisions_20261113/user_decisions.json`); all current Phase 2 physics questions are resolved for now:**

1. Property refresh: refresh affected columns at the transport-event handoff; properties are held during an event (no within-event texture feedback). Actual full rain/recession completion and truthful returned surface water come first. This supersedes the earlier non-approved "recommendation" wording.
2. Bedrock/support: the FULL MAPLE voxel-grid bottom is fixed, flat and impermeable (authoritative datum from MAPLE, never a resident-window edge); no bedrock map or parameters, no bedrock pore-water, no artificial soil, no refusal merely because floating layers reach rock; only actual soil portions store water (fully-rock portions have zero capacity and zero root-water supply). The open lower boundary applies only to a hydrological profile ending ABOVE rock and is explicit export below the modelled profile, never transmission through rock. This supersedes the earlier "fail closed" recommendation.

The detailed plan is in [phase2_completion_architecture.md](phase2_completion_architecture.md).

**Not a new question:** Green–Ampt persistent history and cumulative intake are retained without automatic reset (already approved). How the equation and history behave with refreshed parameters is a new qualification problem to be tested and reported precisely; no new rule is invented here, and a rule would require a precise physics question.

**Engineering work still pending:** the exact staged integration ordering with ordinary drainage; implementation of the staged transaction; NumPy/Numba/CUDA, conservation, transaction-atomicity and restart qualification.

## 5. Engineering requirements (later slices)

Changed-cells-only, bounded backends; update affected hydraulic properties whenever composition/texture changes, even if the net SURFACE ELEVATION change is zero, while reusing unchanged properties and not forcing updates where material/property change is zero; a soil-property refresh does not trigger routing (only the existing net-elevation threshold does); no reroute except the existing net-elevation threshold; staged atomic parent/generation/restart support; NumPy/Numba/CUDA tests later; no full-profile or bed copies every hot step. The S1 exact interval remap stays a tested ALTERNATE component, not the default water-motion policy.

## 6. Limits
No physics, backend, restart or large-domain qualification is claimed. The capacity, refresh and bedrock treatments are user-approved design only; implementation and independent qualification remain.
