# SR33 default field capacity — impacts, conductivity note and audit of other choices (2026-11-20, correction01)

Scope: no observed-site field run or refit was made; the native hydrology components and events WERE actually tested (see `qualification.md`).

Evidence base: Root's accepted JORN audit (`jorn_field_capacity_audit_20261120/root_audit.json`): density-adjusted theta_33 = 0.09746 / 0.09940 / 0.10914 versus the
old hybrid 6 kPa FC 0.39760 / 0.39867 / 0.39508 for the three JORN layers. No observed-site field run or refit was made for this task (actual native component and event tests WERE run, see `qualification.md`); statements about site behaviour are from code structure and the audit
numbers. The user resolved (2026-11-20): **keep the Tietjen/EcoHyD K as the defaults**; document the difference from Saxton-Rawls unsaturated K (Ku). No new K method, no refit.

Primary sources: [Saxton & Rawls 2006](https://doi.org/10.2136/sssaj2005.0117); [Tietjen et al. 2009](https://doi.org/10.1029/2007WR006589).

## 1. What changes and what does not
SYRUP laws are unchanged; only the field-capacity number they receive (and its explicit version) changes.
* **Saturation limits storage; field capacity triggers downward movement.** Lowering FC does NOT shrink the saturation capacity of a layer. It lowers the water content at which the
  existing gravity cascade starts to move water downward.
* **Gravity cascade.** The existing cascade transfers water above FC to the layer below automatically (instantaneous, not limited by any unsaturated conductivity). The SR33 default does
  NOT adopt SR Ku or K-limited gravity. A **closed** bottom can still hold water above FC when storage below is blocked (only unavoidable excess above saturation returns to the surface);
  it is NOT true that all water above FC returns to the surface. An **open** bottom exports what leaves the profile exactly once; with a lower FC the cascade can make more room below.
* **Below FC** redistribution keeps the Tietjen K and the existing interface exchange; `d` is the dimensionless calibration coefficient of that exchange (field case: donor-depth `D = d * K_interface * h_donor`), not a "constant interface diffusivity". The native explicit-diffusivity input mode is unchanged. No refit and no new Ku.
* **Zero-time reconciliation (event-boundary refresh, new owner/generation)** reuses the same cascade once on the NEW FC; open-bottom export and closed-bottom surface return are
  accounted exactly once. A column holding water above the lower FC redistributes/exports/returns in that single bounded pass.
* **Small-rain recharge** is a uniform recharge followed by the FC gravity cascade; saturation limits storage. It does not use the matrix law.
* **Matrix infiltration** uses the fillable deficit `max(FC*h - W, 0)` for wetting/onset capacity. The simplified small-rain recharge path DOES bypass the full matrix infiltration law (it does not use this deficit to admit storage); full-storm matrix infiltration uses the FC-dependent deficit.
* The numerical response of either to a lower FC has no guaranteed direction without a control run, and actual rain-limited intake is not necessarily changed. This task makes no direction claim.
* **Initial water inventory is not changed by the FC selection** (saturation and the observed initial condition are the same). Any drainage happens at the next actual update or event-boundary reconciliation; there is no source migration or reset.
* **Not directly changed:** Saxton-Rawls Eq 16 Ks (anchored on theta_33 in both methods; bitwise equal across methods in `test_sr33_default.py`), explicit matrix K, the diffusivity law,
  `theta_evaporation_stress`, residual, saturation, wilting point, suction, OM/density limits, air-entry computation and its strict guard.
* **Indirect:** soil-moisture trajectories, E/T supply, growth, runoff and every parameter fitted conditionally on the 6 kPa hybrid (d = 0.00102395, E = 1, stress = 0.2017, K-multiplier
  studies) are conditional on the OLD FC; they are not refit here and are unqualified with the SR33 default. Historical outputs/checkpoints keep their 6 kPa meaning.

## 2. Air entry is inert for the new FC / Ks / water flux
With FC = theta_33 and Eq 16 on theta_33, the air entry is not used in any new SR33 FC, Ks or water-flux calculation. The provider still computes it and its **strict guard is retained**
(negative or >= 33 kPa refuses; the JORN raw -0.608 kPa still needs the explicit study override 0), and it remains a diagnostic and the legacy-6 kPa input. The Green-Ampt wetting-front
suction is a separate explicit input. A future decoupling of the guard from FC/Ks is possible but is NOT done here (scope; domain widening is not authorized).

## 3. Eq 17 and the Tietjen K versus SR Ku (documentation only; UNADOPTED)
Saxton-Rawls 2006 Eq 15/17: `lambda = 1/B = (ln theta_33 - ln theta_1500)/(ln 1500 - ln 33)`, `K(theta) = Ks (theta/theta_S)^(3 + 2/lambda)` (equivalently `3 + 2B`). The earlier draft wrote `3 + 2/B`; that
was wrong. Root's 80-digit check for the JORN layer 1 anchors: lambda = 0.23658496583213, exponent 3 + 2/lambda = 11.45362254091457 (no unit/exponent error).

Quantities (mm/h; JORN layer 1 field-comparison recipe: Ks = 120.7865, theta_33 = 0.09746, theta_S = 0.46430). They are three DIFFERENT quantities:
* **SR Ku(theta)**: the SR moisture-dependent unsaturated conductivity: 2.78444e-6 (theta 0.10), 2.89481e-4 (0.15), 7.80947e-3 (0.20), 0.811904 (0.30), 21.9031 (0.40), 120.787 at saturation.
* **EcoHyD/Tietjen representative Ku = 12.0886**: the INTERMEDIATE value used to construct the effective WITHIN-LAYER K (it is not "the SYRUP default K input").
* **Effective within-layer K = 38.2117**: built from that representative Ku. The INTERFACE K is then averaged across the neighboring layers (38.0403 for the JORN top/middle interface); 38.2117 is not repeated at every interface.
The generic native API keeps its explicit matrix K or the opt-in `texture_ks` mode; this FC patch implemented no global automatic 0.316 substitution and no K code change. The numerical examples are the field-comparison recipe, not all native configs.
These are formula examples at stated water contents, not observed fluxes, and not an apples-to-apples comparison: the SYRUP default is a representative constant, the SR value is a steep function of theta.
Very small SR Ku near FC does not by itself contradict the retention FC, but SR K-limited drainage would differ materially from SYRUP's instantaneous above-FC cascade. Adopting Ku, K-limited gravity,
or re-fitting d is NOT part of this change (user: keep Tietjen K for now).

## 4. Audit of other choices that differ from a simpler Saxton-Rawls approach
Classification: **D** = definition/convention choice, **X** = extra model choice (documented, defensible, not part of S&R2006), **K** = kept by approved requirement. Nothing is changed here.
1. **FC at 6 kPa (Tietjen) vs S&R 33 kPa** — D. The old hybrid was an unqualified mix of pressure conventions (Tietjen's 6 kPa field-capacity convention evaluated on the S&R linear wet branch, Eq 12);
   Eq 12 itself was evaluated correctly: it is not an arithmetic error. Resolved by the default change.
2. **Strict air-entry refusal vs study-specific floors/overrides** — X. Retained.
3. **OM conversion (OM percent explicit; sites derive it from estimated carbon with a user factor)** — X; S&R do not define the conversion.
4. **Constant Tietjen K, interface-averaged K and the dimensionless exchange coefficient `d` vs S&R Ku(theta)** — X/K (user keeps Tietjen K). Previous fits were conditional on the 6 kPa FC.
5. **Independent stress/residual vs S&R theta_1500** — X. The fitted stress (0.2017) exceeds the SR33 theta_33 of JORN (~0.10). That is not automatically a model error; it marks the 6 kPa-conditional fit
   as unqualified under SR33 (conditional/unqualified, not a decision needed for this bounded task).
6. **Layer averaging (dry-mass mixing before density floors; mass-overlap composition)** — X/K (approved surface-following profile).
7. **Density handling (explicit density vs actual MAPLE packing, DF range [0.9, 1.3], gap >= 0.005)** — K.
8. **theta_S = 1 - rho/2.65** — S&R Eq 7-10 itself.
