# Selected first comparison: fixed terrain, no direct splash

User authorized preparation on 2026-09-30, then separately authorized a full
MAHLERAN run. See [executed reference run](mahleran_reference_run.md) for results.
The initial setup described below did not itself launch a storm. Phase6 and this separate benchmark patch have since completed independent
review; see the linked reference-run report for disposition.

## Prepared MAHLERAN copy

Location: `/home/okin/SYRUP/outputs/phase7/mahleran_plot1_no_splash`.
Original `/home/okin/MAHLERAN` source and inputs are unchanged, verified by
before/after SHA-256 manifests. Reproduce into a NEW directory with:

```bash
python benchmarks/phase7/prepare_mahleran.py --output NEW_COPY
```

Only two copied files differ:

1. `src/Subroutines_Sediment/route_sediment_xml.f90`: the dry-cell, positive-rain
   branch explicitly sets `detach_soil(:,im,jm)` and `depos_soil(:,im,jm)` to
   zero instead of calling `raindrop_detachment` and `splash_transport`.
   The wet-cell `raindrop_detachment` call is unchanged. Existing mobile
   sediment arrays and velocity memory are not cleared by this patch.
2. `mahleran_input.xml`: input/output directory separators are normalized from
   Windows to POSIX relative paths. The selected existing settings remain
   `runtype=event`, `update_topography=n`, `flow-routing_solution_method=5`.
   Thus neither elevation updates nor dynamic water-surface routing mode 6
   are enabled. Topography-update interval 1 is inactive when updates are off.

The copied tree includes sources, Plot 1 inputs, original build files and an
empty Output directory, plus `benchmark_manifest.json` and `no_splash.patch`.
At initial preparation the application had not been built or run. The subsequent
Linux build and completed runs are documented in the linked reference-run report.
Original build files target Windows/NetBeans. The source patch is bound to the exact audited routine and root XML
hashes; changed reference files require re-audit rather than fuzzy application.

This is specifically the non-marker event configuration. Marker-in-cell splash
paths are unchanged and not covered. Original sediment bookkeeping, negative
mobile-load clipping, timestep conventions and hydraulic numerics are unchanged;
removing splash does not eliminate those existing comparison differences.

## Executed focused verification

`check_no_splash.py` compiled the actual original and patched routing routine,
with original shared modules and explicit test-double physics callees. Checked
compilation uses gfortran13, `-std=legacy -fcheck=all -O0`; raw build/run commands
and logs are in `outputs/phase7/no_splash_branch_check`.

- Original dry/raining cell calls rain detachment and splash; patched does not.
- Deliberately stale dry-cell detachment/deposition values are reset to zero.
- Existing dry-cell mobile load is retained in the controlled Crank–Nicolson
  test with zero transport velocity.
- Both versions call wet raindrop detachment once and diffuse transport for all
  six classes, with identical outputs.
- Reference and prepared source/input manifests remain unchanged by the test.

This verifies branch selection and immediate side effects, not complete storm
physics or general conservation of the legacy algorithm. The test doubles do
not provide evidence for erosion magnitudes. Curated results and source case
settings are alongside this document; the exact patch is in benchmarks/phase7.

## Required SYRUP benchmark behavior

When the actual comparison is requested, freeze **both elevation and routing**
throughout the compared window. Continue actual MAPLE conservative pickup,
deposition, availability and underlying sediment updates; do not freeze or
replace the sediment inventory. Retain wet rain-assisted detachment, omit
splash, and match rainfall, geometry, boundaries and hydraulic parameters.

Implemented and exercised by the Phase 7 frozen benchmark runner; see [qualification report](acceptance.md).
The normal Phase 6 complete-event runner commits and reroutes terrain and must
NOT be used unchanged and labelled a fixed-terrain comparison. Intermediate or
terminal commits, avalanching, routing rebuilds and restart must respect the
benchmark contract. Report bed mass changes separately from the deliberately
fixed hydraulic terrain. Later evolving-terrain comparisons are separate.

## Other supplied cases

All three supplied model XML files explicitly specify `update_topography=n`:
root Plot 1, `Input/RFID_2014/mahleran_input.xml` and its `(2)` variant. The latter
also has a preexisting XML syntax error at line79; its setting was read as text,
not treated as a runnable validated configuration.

Seven supplied `mahleran_input.dat` files identify legacy version1.01.4: Plot3,
Abbott, Wise, and input_p1–input_p4. They do not explicitly identify a terrain
update switch and are not the XML input format of the current executable. Their
actual legacy-version behavior needs its corresponding parser/model; we cannot
classify them as evolving terrain from these files alone. No supplied XML case
with terrain evolution enabled was found. Fixed terrain should not be equated
with identical sediment inventory/composition treatment in the two models.
