# Independent qualification — 2026-10-17

Accepted and adopted as an **explicit windowed-bed option** for the coupled pond event. The previous feature commit was
`e519ea62`, pushed to `origin/main`; these integration changes remain uncommitted. Exact source identifiers and evidence
hashes are in [acceptance.json](acceptance.json).

## Executed coverage

* CPU: 519 unique passes, 91 skips, zero remaining failures across the full and corrected targeted runs. GPU skips in the CPU environment are not GPU evidence.
* GPU: 611 unique passes, 0 skips, zero remaining failures across the full and corrected targeted runs. Actual allocated GPU0, original 9,092,045,209-byte pool limit, fresh idle checks before each worker.

The full regression scope comprises the six existing splash/pond/transport/friction/event/checkpoint families and the two new
windowed families. The additional real-device terminal-deep-spill check is local evidence. The one intermediate full-suite
failure was the test helper omitting original NumPy reduction scalars while including decoded 0-d arrays; it now compares
both exact typed bytes. All new tests were rerun after correction. This is not an all-repository test pass claim: 1,615 tests
were collected, and the preceding baseline's separate phase-5 velocity-output assertion remains documented.

Checks include finite-supply wet and dry splash, genuine resident refill/spill, exact tiny changes and overlays, below/above
net-elevation gates, high-water/return/explicit publication, genuine native carried-cap failure with one flush/retry, independent
physical retry after a flush, clean oversized failure, late transport/geometry/counter failure atomicity, corrupt/oversized
checkpoint refusals, full/overlay device continuation, nonzero terminal deep deposition, explicit surface/soil water removal,
and completed handoff loading without repeated finalization. The genuine-cap test uses deliberately redundant native carried
records as a labelled storage fixture, followed by real wet deposition; the native exception is not injected.

Claude implemented bounded source-only assignments; Codex independently inspected and ran the checks. A separate read-only
Claude review found no blocking issue and approved Codex's terminal-journal peak-count correction. Actual terrain/forcing/
checkpoint arrays were never supplied to Claude. No live MAPLE edits, new environments, or other GPUs were used.

## Native Chastre subset

32×32 active cells, native 1 m spacing and original halo, 586×1 mm logical voxels, three resident levels; all cells have buried
storage. The original subset contains 118 strict pits and 118 distinct flat minima. One explicit genuine downhill east outlet
is open; other crop faces are closed. This is a new bounded crop boundary, not the whole catchment.

Before the event, eight real 0.5 mm native pickups at (8,8) are conservatively deposited at adjacent (8,9). This intentionally
exercises deep refill, spill and a genuine net threshold crossing; it is not rain-generated erosion. The GPU fixture publishes
the host-bound journal before moving native arrays and height packets. This is storage-only and preserves every physical
value and the original held geometry.

Forcing: synthetic 120 mm/h from 0–240 s, fixed conductivity 1e-6 m/s, friction 21.45, eight phase bins, maximum dt 0.25 s.
Runs reach **60 s, still raining**, with 240 accepted steps and one hydraulic refresh. All native water/class/mobile/export
budgets close. These are not a full storm, original RFID calibration, or a new Fortran comparison.

| Backend | Encoding | Setup (s) | Event loop (s) | Fresh 20→60 s restart |
|---|---|---:|---:|---|
| numba | full | 5.97 | 32.01 | exact |
| numba | overlay | 6.11 | 29.60 | exact |
| cuda | full | 6.29 | 63.69 | exact |
| cuda | overlay | 6.33 | 63.07 | exact |

Timings include validation, publication and a checkpoint at 20 s; compilation/startup is not rigorously separated and CPU
verification jobs overlapped. They are descriptive event timings, not a controlled speed benchmark. GPU remains slower on
this small domain. No large-domain speed or fit claim follows.

CPU/GPU comparisons pass the existing hydraulic 2e-12/1e-14 and sediment 1e-10/1e-14 policies over 66 common physical arrays;
independent native accounting diagnostics are checked through their operand-based budgets. Full/overlay logical bed values
and shared physical arrays are bit-exact within each backend. Fresh-process continuation is exact over 89 CPU or 94 GPU arrays,
metadata and all logical voxel values. CPU reductions represented as scalar values account for the array-count difference.
The preceding C2 dense baseline matches 66 physical arrays and the reconstructed logical bed; C3 changes only finalization
receipts/peak diagnostics, and all 89 wet-state arrays plus every logical voxel remain bit-exact to C2. Sources are recorded
separately in acceptance.json; no cross-source checkpoint was loaded.

The logical voxel array alone is 28,803,072 bytes. The resident window is 147,456 bytes and the maintained height packet is
106,496 bytes. These exclude other metadata, hydraulic/phase/scratch arrays and the bounded dense genesis/reference. Host RSS
includes that construction; reported GPU pool held-after values are not peak measurements. Full-Chastre coupled memory fit
and multi-face/phase GPU optimization remain follow-ups. External stores and all referenced versions must be retained.

See [README.md](README.md), [core.md](core.md) and [checkpoint.md](checkpoint.md) for controls, publication/restart contracts,
unavailable touched-splash diagnostics and resource limits. Raw local commands, reports, failures and result files are preserved
under `agent_handoffs/tasks/pond_windowed_20261017/`; they are local evidence rather than a portable data bundle.
