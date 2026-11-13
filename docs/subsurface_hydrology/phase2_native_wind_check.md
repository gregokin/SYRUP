# Phase 2 native wind -> refresh engineering control (DRAFT)

**Status: engineering control, not a simulation.** This page records one bounded check run by Root. It is not the integrated MAPLE-SYRUP model, not an integrated lineage, has no elapsed soil-water clock and no selected production wind adapter, and changes no texture class (only depth/support change). It does not show coupled wind production behaviour.

## What was run
- Probe: `root_native_wind_dirty_v5_probe.py` (launcher `root_native_wind_dirty_v5_launcher.py`), results `root_native_wind_dirty_v5_runs/{probe.out,probe.err,result.json}` and `root_native_wind_dirty_positive_echo_root_event_daily_bridge_cuda_c4_snapshot166.json`, under `agent_handoffs/tasks/phase2_evolving_soil_20261112/`.
- Source: SYRUP snapshot `root_event_daily_bridge_cuda_c4_snapshot166`; pinned native MAPLE CPU runtime (commit `9f466d2cfcd52991b24496cf67f4562d6268afa9`, package digest `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`; the probe reports its materialized store `5d850789…`). Run time 2.98 s on CPU; one process; no GPU.
- Native MAPLE aeolian calls only: `advance_wind_record` (two wind blocks, 0.07 s each, total 0.14 s, shear velocity 2 m/s, directions 90 and 180 degrees, baseline aeolian, avalanching disabled, no creep or growth), the native pure settling plan and `finalize_wind_event`. Spatial block diagnostics were enabled so per-cell detachment, splash removal and deposition are recorded.
- Boundaries: x prescribed with zero inflow, y periodic (the helper default). Dense bed, partial-bottom profile of nominal depth 0.26 m, soil texture fractions fixed at [0.05, 0.45, 0.5] (no texture change; only depth/support change).
- The finished wind bed was passed to the existing cold `stage_native_refresh` / `adopt_native_refresh` of a native floating owner.

## What the evidence shows (from `probe.out`)
| item | value |
|---|---|
| wind blocks / completed / end time | 2 / true / 0.14 s |
| cells touched by detachment, splash removal, step or terminal deposition | 16 |
| cells with a final net bed change (independent full-row comparison) | 12 (the touched set contains every changed cell) |
| detachment / step deposition / terminal deposition | 0.009824 / 0.002618 / 0.005806 kg |
| per-cell native finalizer deposit echo, maximum gap (long double) | 3.298e-16 kg, below the native bound 9.352e-13 kg |
| cells whose depth/support changed | 12 |
| per-layer water W and persistent J and I after the refresh | unchanged (array-equal) |

## Limits
- Gross touched cells come from the optional spatial diagnostics; the live ledger alone is not enough because the native ledger resets at each commit.
- The native settling-plan preview is a pure private helper of the pinned MAPLE source; it is used here only as an echo check, not as a public interface.
- The refresh performed a cold scan of the parent and child beds; no production selected-wind adapter exists.
- Nothing here validates erosion magnitudes, wind fetch or any field behaviour.
