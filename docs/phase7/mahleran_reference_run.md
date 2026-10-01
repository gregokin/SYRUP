# Whole-program MAHLERAN reference run — 2026-09-30

User explicitly authorized the MAHLERAN run after the earlier setup-only task.
The 5,400-second Plot1 event completed twice with direct dry-cell splash disabled,
wet rain-assisted detachment retained, and elevation/routing fixed. Fourteen
numerical output files are byte-identical between runs. Independent Claude
review subsequently passed with limitations recorded below; no SYRUP event was run by this task.

## Artifacts and reproduction

- Prepared source: `outputs/phase7/mahleran_plot1_no_splash_linux`.
- Checked executable and all 95 compilation commands/logs:
  `outputs/phase7/build_checked_linux/build.json`.
- First run: `outputs/phase7/mahleran_fixed_no_splash`.
- Repeat: `outputs/phase7/mahleran_fixed_no_splash_repeat`.
- Normalized outlet time series, actual logged rainfall, south-first routing,
  and audit: `outputs/phase7/mahleran_reference_audit`.
- Curated summary: `docs/phase7/mahleran_run_summary.json`.
- Task commands/logs: `agent_handoffs/tasks/phase7_mahleran_run`.

Use the recorded Phase6 environment and new output directories:

```bash
source agent_handoffs/tasks/phase6_complete_event/env.sh
"$SYRUP_PYTHON" benchmarks/phase7/build_mahleran.py \
  --source outputs/phase7/mahleran_plot1_no_splash_linux --output NEW_BUILD
"$SYRUP_PYTHON" benchmarks/phase7/run_mahleran.py \
  --prepared outputs/phase7/mahleran_plot1_no_splash_linux \
  --build NEW_BUILD --output NEW_RUN
# Repeat the preceding command with a different NEW_REPEAT output.
"$SYRUP_PYTHON" benchmarks/phase7/audit_mahleran.py \
  --run NEW_RUN --repeat NEW_REPEAT --case outputs/plot1 --output NEW_AUDIT
```

The parent preparation manifest describes its historical setup state, not the
subsequent build status; execution/build manifests supply the latter. Recreating
the Linux derivative from the parent requires applying
`benchmarks/phase7/linux_input_diagnostic.patch` and recording its new prepared
source hash in a derivative manifest while retaining original reference hashes.
All reference files and production SYRUP source were preserved.

## Build adjustment and runtime

The first checked full run exposed a diagnostic `SIZE(data_array,...)` before
allocation in `read_spatial_data.f90`. The isolated Linux derivative guards that
print with `allocated(data_array)`. Its explicit patch does not change numerical
calculations. Failed build/run evidence is retained. The original reference tree
and initial no-splash preparation are unchanged.

GNU Fortran13, `-O2 -std=legacy -fcheck=all -fbacktrace`, compiled the original
95-source application with platform-appropriate linking. Successful wall times
were 9.0477 and 9.0893 seconds; peak child RSS was approximately21MiB. These are
whole-program timings including initialization, sediment and output, not a
water-kernel speed comparison with SYRUP.

## Comparison requirements and limitations

Routing matches all1,200 imported SYRUP cells after cropping MAHLERAN's exterior
ring and reversing north-first ASCII rows into MAPLE's south-first order.
SYRUP must freeze this geometry/routing while conservatively updating its actual
MAPLE sediment holdings. Its normal Phase6 evolving-terrain runner does not yet
provide that benchmark mode.

The original schedule integrates to9.652mm; iteration-start logged applied rain
integrates to9.656233333mm. The legacy `time > time_next` transition and update
after the physical step shift forcing boundaries. The saved per-step applied
rainfall allows that difference to be matched explicitly; do not silently use
exact-interval SYRUP forcing and label it identical. Rain is logged to0.01mm/h.
The legacy program also repeatedly reports read errors after reaching the end
of the rain file; applied rainfall thereafter is zero. Raw logs, including these
warnings, are retained. This task did not modify that legacy behavior.

Rounded stock outputs give approximately0.105795m³ water export and0.00783587kg
sediment export. `hydro001.dat` outlet discharge is mm³/s, converted by1e-9;
`sedtr001.dat` total outlet sediment and `seddisch001.dat` six class exports are
kg/s (see `output_hydro_data_xml.f90`, units57,58,104). Integrals use1-second
steps. Four-significant-digit output and legacy reductions prevent treating
these files as a full-precision conservation ledger. Total and summed class
exports can differ through rounding. No complete bed/mobile/water closure is
claimed from the stock files, and MAPLE conservation tolerances remain unchanged.

`depth001.asc` and `veloc001.asc` store synchronous fields at maximum outlet discharge, not per-cell storm maxima or final states (see `output_hydro_data_xml.f90:489–506`). `detac001.asc`
and `neter001.asc` are accumulated detachment/net erosion kg per cell, not a
MAPLE voxel inventory. `dschg001.asc` is cumulative cell throughflow in m³.
Compare physical interior60×20 cells separately from the exterior ring.

## Independent review and provenance clarifications

Claude reviewed the no-splash preparation, build/run tooling and audit after the
usage reset; no blocking defect found. Codex accepted this bounded reference
milestone after inspecting the findings. Report/disposition:
`agent_handoffs/tasks/phase7_reference_review/`. This does not accept a matched
SYRUP sediment benchmark. The later deterministic-conductivity diagnostic in
[initialization_audit.md](initialization_audit.md) identifies the appropriate
first matching configuration; the original sampled-conductivity run is preserved.

The complete compiler flags also include `-ffree-line-length-none`,
`-ffixed-line-length-none` and `-fallow-argument-mismatch`; the latter relaxes
legacy interface diagnostics. No separate experiment established whether it is
necessary for this build. Manifest inventory covers `src`, `Input/input_p1`,
`nbproject`, root XML and Makefile, not every unrelated file in MAHLERAN.
The prepared-manifest statuses describe preparation time; build/run manifests
supersede them for execution status. The Linux derivative has both the no-splash
patch and the allocated-array diagnostic guard recorded in its changed-files
and portability-fix metadata.

An additional post-review comparison confirms `current_parameters_from_xml.dat`
is byte-identical between repeats; `Output/param001.dat` differs only in the
output timestamp. The14-file identity list intentionally covers numerical output
products, not all files. Peak times are the first occurrence of rounded maxima.
Raw logs include NUL bytes; decoding with replacement does not remove NULs.
After-EOF rainfall is observed zero here, but the local `atime` variable is not
saved in `Set_rain_xml`; do not infer robust general EOF handling from these runs.
