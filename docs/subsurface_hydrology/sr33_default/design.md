# SR33 default field capacity — design, API and compatibility (2026-11-20)

Status: independently accepted native API change; see [qualification](qualification.md) and [acceptance record](acceptance.json). No acceptance of old source-locked capsules, coupled three-model runs or observed-site simulations is implied. Scope is the field-capacity (FC) parameter and its explicit
version/selection only. No hydrology law, refit, Ks/SAT/WP/residual/suction/OM/density/air-entry floor or PTF-domain change.

## Decision
Sources: [Saxton & Rawls 2006](https://doi.org/10.2136/sssaj2005.0117) (FC = 33 kPa, Table 2; Eq 2, 9, 15–17); [Tietjen et al. 2009](https://doi.org/10.1029/2007WR006589) (Tietjen 6 kPa field-capacity convention and the EcoHyD conductivity construction).

The 6 kPa field-capacity convention (`theta_6 = theta_33 + 27 (theta_S - theta_33)/(33 - psi_e)`, or `theta_S` when `psi_e >= 6`) is deprecated as a
rule and a default. The default derived FC is the final, density-adjusted `theta_33` of Saxton & Rawls 2006 (Table 1 Eq 2 and Eq 9; Table 2 defines
FC as 33 kPa), including a supplied final `theta_33_override`. `theta_6` is never relabelled `theta_33`.

## Final minimal API (`maple_syrup.soil_texture_properties`)
| item | new | legacy |
|---|---|---|
| keyword `field_capacity_method` of `estimate_soil_texture_properties` / `estimate_property_memory` | default `"saxton_rawls_33kpa"` | `"tietjen_6kpa_deprecated"` (explicit, named) |
| FC output name | `theta_field_capacity` (neutral; value = final theta_33) | `theta_field_capacity_6kpa` (true 6 kPa value only) |
| warning | none | one `FieldCapacityDeprecationWarning` (a `FutureWarning`) per call, never per cell |
| `fc_equals_saturation` provenance bit / `n_fc_equals_saturation` | never set / 0 (meaningless for 33 kPa) | unchanged |
| pressure metadata | `recipe_record()['field_capacity_pressure_kpa'] == 33.0`; no 6/27 marker anywhere in the default record | `legacy_6kpa` block (6, 27) only in the legacy record; module constants are `LEGACY_6KPA_PRESSURE_KPA`/`LEGACY_6KPA_SPAN_KPA` |
| `RECIPE_TAG`, `SCHEMA` | `..._v2_fc33_default`, `soil_texture_properties/2` | same v2 tag; method recorded in `recipe_record()["field_capacity_method"]` |

* Exactly ONE FC array is allocated in either method (no alias, no second full array); memory estimates are identical.
* The method is validated (exact string, before any allocation/backend probe); an unknown or non-string method is a typed `TexturePropertyError`.
* The method is part of `recipe_record()`, therefore of the result `sha256`; 33 vs 6 results can never hash equal. `SoilTextureProperties.field_capacity_method`
  and `.fc_output_name` expose it; `summary()` records it.
* The legacy name returns the same numbers the historical oracles describe; `array("theta_field_capacity_6kpa")` on a default (33) result is a typed refusal
  naming the method (no mislabeled alias).
* NumPy reference, Numba and CUDA RawKernel evaluate the same branch from the same flag (`fc_mode` 0 = 33, 1 = 6 kPa legacy); the kernel receives the mode as a
  scalar argument (CUDA macro `FC_MODE` is not baked, so the one compiled kernel serves both; no source fallback).

## Native floating soil
`native_floating_config` does NOT expose the method: native owners always consume the new default (a read-only CONSTANT property `NativeFloatingConfig.field_capacity_method` (not a dataclass field: `dataclasses.replace`/constructor cannot select another method),
bound into `config.sha256` as `|field_capacity_method:saxton_rawls_33kpa`). `_batch_properties` passes the method explicitly and reads the neutral output.
Initialization and event-boundary refresh share this one function. Explicit `latent_theta_field_capacity` (zero-mass slots) and `theta_33_override` stay
authoritative. Held-context / lifecycle checkpoint codecs reconstruct the config through `native_floating_config`, so:
* a NEW record round-trips (the reconstructed sha, now version-tagged, matches);
* an OLD record's config sha has no method tag, so reconstruction under the new default cannot reproduce it and the codec REFUSES (existing check
  "does not reproduce the recorded config sha256"). No automatic migration, no rebinding, source guards untouched. Legacy formulas inside the new source are
  not acceptance of historical checkpoint compatibility.

## No water rescale
Nothing in this change touches existing water states, rescales, resets or refreshes accepted states. A new owner/generation uses the actual water inventory and
the unchanged once-only zero-time cascade against the new, lower FC; the cascade moves the excess downward (an open bottom exports once). A closed bottom respects storage: it retains water above FC when the layers below are blocked, up to saturation, and returns to the surface ONLY the unavoidable excess above saturation. This is existing behaviour, not new physics.

## Qualification status
Focused CPU tests (author) plus separate Root hardware controls; see `qualification.md`. No observed-site field run/refit, no new three-model coupled or scaling acceptance.

## Impacts and audit of other choices: see `impacts.md`.
