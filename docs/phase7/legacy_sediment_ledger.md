# Plot1 legacy sediment accounting audit

This diagnostic-only derivative of the deterministic, no-splash MAHLERAN
reference reproduces all 14 numerical output files byte for byte. The parameter
report differs only in its execution clock. Original reference trees were not
edited. Preparation, build, execution and output hashes are retained under
`outputs/phase7b/mahleran_ledger_*_v3`; the reproducible preparation and audit
scripts and condensed results are in `benchmarks/phase7b`.

The post-routing hook reads the original source, deposition, mobile-depth and
face-flux arrays. A separate counter records the existing negative-depth clipping
before the original assignment to zero. No model state expression was changed.

| Storm-integrated quantity | kg |
|---|---:|
| Requested legacy detachment | 452.857327 |
| Active-domain deposition | 452.832214 |
| Net active-domain erosion | 0.025113 |
| Outlet export | 0.009223257 |
| Final mobile inventory | 0.307858251 |
| Effective source introduced by clipping | 0.291968559 |
| Deposition accumulated outside the active domain | 0.019239130 |

The independently assembled Crank–Nicolson identity is

`final mobile = initial mobile + detachment - active deposition - export + effective clipping source`.

The maximum absolute per-step, per-class residual is 1.63e-15 kg; internal
face cancellation leaves at most 4.05e-19 kg per step/class. The final pool is
approximately 0.145275 kg in class 1, 0.162461 kg in class 2 and 0.000122222 kg
in class 3. It is not simply the difference between erosion and export.

Clipping is an artificial source in this identity. If the untruncated implicit
solution is `d_trial < 0`, setting it to zero also changes the implicit outgoing
flux. The effective source is therefore
`(-d_trial) * (1 + dt*v/(2*dx)) * cell_mass_per_depth`, not just the change
in depth. The v2 diagnostic recorded the latter alone, leaving a 0.00207249 kg
residual; v3 includes the flux term and closes the identity. Both diagnostic
runs and the first build failure are retained as task evidence.

The outside-domain deposition tally is separate from the active-domain
Crank–Nicolson balance. Adding it as another sink would double-count a term
that is not in that balance. This audit does not support the earlier proposed
explanation that the routing ring imposes an extra full-cell survival factor.

Consequences for validation: retain the legacy export as an observed comparison,
but do not reproduce negative-concentration clipping to force agreement. SYRUP
must continue to conserve class mass through actual MAPLE holdings. The audit
does not establish how much of the exported 9.2 g is attributable to clipping;
that would require a separate controlled change to legacy numerics. It also
does not erase the independently demonstrated excess transmission in SYRUP's
well-mixed, upwind cell operator. The latter remains the correction target.
