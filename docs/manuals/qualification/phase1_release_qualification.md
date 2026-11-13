# Phase 1: qualified layered water and LEAF calendar

The opt-in layered model has completed a small, file-driven rain/interevent run
on NumPy, compiled Numba CPU and CUDA, using actual committed MAPLE, SYRUP,
LEAF and MASY sources. Soil moisture persists through the storm and interevent
periods. LEAF requests plant water; SYRUP alone debits the physical inventory.
Wind, rain/recession and interevent periods do not overlap. Evaporation,
transpiration and growth occur only during interevent periods.

This is a bounded qualification of fixed-bed, fixed-layer water and LEAF
coupling. It is not field validation, a wind–water–LEAF simulation, an
evolving-soil qualification or a large-domain performance result.

## Exact code selection

| Component | Code commit | Role |
|---|---|---|
| SYRUP | `e40a1b8fc56a96920db4ffbd182d4068fa5de3e8` | Physical surface and layered water, accounting and codecs |
| LEAF | `52f39fe06166a91e768740b19d3c26d4ffceaf6e` | Plant initialization, demand, root allocation and growth |
| MASY | `a70400230f26d7931624d49173196a1cfa05a90a` | Temporal coordinator, native adapters, CLI and combined restart |
| MAPLE | `9f466d2cfcd52991b24496cf67f4562d6268afa9` | Genuine native Dense bed and infrastructure |

MAPLE 9f is an explicitly selected historical qualification source, not the
latest live MAPLE checkout. Package-content identities and machine-specific
qualification recipe pins are in `phase1_release_qualification.json` and
`phase1_recipes/`. Later documentation commits do not silently rebind these
code identities, older profiles, runtime locks or checkpoints.

## Executed example

The input set is MASY's `examples/native_leaf_calendar_cli/`: 4 rows by
6 columns of 0.25 m square cells, a genuine Dense bed, two fixed soil layers,
explicit illustrative hydraulic properties, and one LEAF plant in a
two-species catalog. Its `SURFACE` establishment anchor is resolved from the
actual native bed. The supplied parameters are numerical qualification
values, not a field calibration.

The sequence is one day of interevent evaporation/transpiration/growth,
0.25 s of rain at 3600 mm/hour followed by the native storm recession to
genuine completion, then another day of interevent processes. The storm takes
597.25 s including recession; total physical elapsed time is 173397.25 s.
The large intensity and short duration are deliberate fixture choices.

The actual immutable runtime `prepare` and parent `verify` operations ran in
the existing MAPLE CPU and GPU environments. The worker independently
recomputed its source binding. NumPy, Numba and CUDA each completed all three
occurrences. On both CPU and GPU, separate processes paused after occurrence
1, resumed to a second pause after the completed storm, then resumed to the
final boundary. A time or resource limit was never treated as storm completion.

## Results and independent controls

| Check | Result |
|---|---|
| Portable equation/accounting lane | 154 passed, no skips; includes four CUDA controls |
| Production metadata and immutable LEAF capsule controls | 92 passed |
| Genuine old/new MAPLE delivery, water restart and reader controls | 19 passed |
| Actual committed-runtime full runs | NumPy, Numba, CUDA and both restarted continuations completed |
| Checkpoint state comparison | 29 floating arrays and 15 other state arrays exactly equal across backends |
| Provenance comparison | One additional NPZ entry checked separately with explicit metadata differences |
| Same-backend restart | All 44 state arrays exact; all four biology files byte-exact |
| Actual CLI negative controls | Six typed refusals, zero physical steps; accepted checkpoint unchanged |

The negative controls covered unknown configuration keys, a zero memory cap,
a wrong recipe hash, an unreviewed receipt, substitution of the other existing
virtual environment despite its identical base Python binary, and a wrong
restart seal. The manifest reader also rejects special files before opening,
admits memory before reading and holds only one hash chunk at a time.

For the example, rainfall input was 0.000375 m³. Requested and delivered
transpiration both totalled 0.00008461372331859172 m³; no request was unmet.
The plant's live dry aboveground biomass increased from 100 kg to
100.12256745482205 kg on each backend. These large initial biomass and other
fixture choices are not ecological recommendations. The largest reported
local water residual was approximately 1.32×10⁻¹⁷ m³, tested using MAPLE's
existing floating-point conservation bound without a new empirical tolerance.

The two independently sealed runtime stores have different source paths.
The scenario identity includes the weather library's actual import path, so
scenario hashes differ even though every scenario, weather and event input
byte and the weather library's source hash agree. Fresh-process controls
verified that path-only cause. Comparison permits that metadata difference;
the original hashes, model guards and restart checks remain unchanged.

Four older tests in one mixed historical collection fail identically on the
pre-change baseline: two assume an empty import state, one pins an earlier
private codec signature and one pins an earlier module list. They were not
silently rewritten or counted as passing. The current focused controls and
actual native runs above are the evidence for this release.

## Limits and next phase

The implementation retains `production_qualified: false`; verified immutable
runtime delivery and a bounded scientific fixture do not establish universal
production readiness. `production_runtime_verified: true` records the actual
runtime verification. Receipts attest the independent review they describe;
hashes alone prove consistency, not review or authenticity.

Memory admissions model named arrays and staging, not whole-process RSS.
No catchment-size or memory-fit inference follows from this tiny example.
The dated LEAF bridge requires event boundary instants representable exactly
in its microsecond calendar; it refuses unsupported instants rather than
rounding them. The new nominal-day convention belongs to Phase 1.5, not to an
automatic migration of these profiles.

Prescribed hydrology cover and preferential-pathway maps remain fixed within
these runs; growth does not yet update them. Automatic Green–Ampt wetting
history recovery remains unqualified. Only explicitly controlled resets
between quiescent test episodes are supported, preserving physical water.

After Phase 1 is committed and pushed, implement Phase 1.5 daily small-rain
recharge before Phase 2. The agreed rule is zero rain: no precipitation
operation; 0 < P ≤ T: attempt recharge; P > T: full storm. See
`phase15_daily_recharge.md`. Phases 2, 2.5 and 3 retain their numbering and scope.
