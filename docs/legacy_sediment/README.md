# Original-routine Fortran sediment reference (task gpu_sediment, stage A2)

**Current status (commit of 2026-10-08): see [`../legacy_gpu/current_status.md`](../legacy_gpu/current_status.md).** The full 2700 s Chastre
run of this harness has COMPLETED from the pinned executable, input and 15 original source files (loop 13839.809219 s, kernel 13651.230445 s; not the whole
MAHLERAN application) and was compared with both completed CPU Numba references: no automatic 1% flag; pickup +0.0053155% (Numba relative to Fortran),
final mobile -0.0001070%, gram-scale class-5 pit/clip differences (+42.57% / +6.43% of tiny Fortran values), and a native surface-water surplus of
+24.952426 m3 after rain ends at 2641 s whose routine-level cause is NOT yet demonstrated (the original stale-inflow/bracket behaviour is suspected; SYRUP's
closed budget is kept and the suspected defect is not copied). These are observations, not an acceptance. Chastre has zero outlets, so its zero export is
vacuous. After the commit, the corrected golden helper was integrated into `compare_legacy_sediment.py` (exact pinned candidate bytes, uncommitted;
only the Plot 1 section changed, see "Plot 1" below); the shipped helper passed its regression suite and was checked against the saved real Plot1 ledger; final review passed
(`current_status.md` sections 7 and 9).

Executed status (documentation correction D1; historical): the harness was written by a file-only author; the **root (Codex) has since built and run it** with
ordinary gfortran 13.3 against the original MAHLERAN routines from 15 pinned source files (it is not the MAHLERAN application). Root-recorded checks
(`agent_handoffs/tasks/gpu_sediment/`, git-ignored): the A2 C1 check run `a2_c1_checks_actual.log` = 256 passed, 452 skipped (an earlier run, `a2_checks.log`, had 2 failures that
C1 fixed); the compiled kind probe `shared_kind_probe.json` (kind-4 defaults for `dt, dx, dx_m, density, sigma, dstar_const, radius, diameter, viscosity,
settling_vel, ustar, d50`; kind 8 for `re, ke, p_par, v_soil`); and the zero-slope probe `zero_slope_diffuse_probe.json`. A native Fortran 600 s run completed
before the full run; the native 600 s final maps were truncated by the 2026-10-06 environment restart and are never
qualified as an accepted result (`recovery_record.md`). A diagnostic comparison of the complete 600 s ledger blocks against the failed-publication CPU 600 s
run (`fortran600_vs_failed_cpu600_ledger_diagnostic.json`: onset 193 s in both, pickup relative difference 0.0316%, final mobile 8.08e-8) is an observation, not
acceptance. Full-storm Fortran differences are reported as observations with 1% investigation triggers, never fitted tolerances; smooth-equation probes use the
predeclared 2e-6 / 1e-14. Cumulative results: [`../legacy_gpu/results.md`](../legacy_gpu/results.md). Sections below saying "never run" / "not yet run" are historical
authoring notes.

## What it is

A controlled harness that links the ORIGINAL MAHLERAN 1.2.3 routines (water: `infilt`, `route_water`, `update_water_flow`,
`ff_type8`; sediment: `route_sediment_xml` with `flow_detachment`, `raindrop_detachment`, `diffuse_flow_transport`,
`conc_flow_transport`, `suspended_transport`, `flow_distrib`, `update_sediment_flow`) against the original `shared_data` and
`parameters_from_xml` modules, and drives them with state read from a binary file. The driver
(`benchmarks/legacy_sediment/legacy_sediment_driver.f90`) contains no erosion, transport or routing equation; it calls the originals
in the order of `MAHLERAN_storm_xml.f90` 100-173 and keeps read-only accounting. It is **not** the MAHLERAN application.

Derived inputs, all declared, pinned and re-checked at build time (`sources.py`):

* an isolated copy of `route_sediment_xml.f90`: the dry-cell splash call pair is replaced by zeroing that cell's rates (the no-splash
  benchmark change; the wet-cell raindrop detachment is untouched), and in the **hooked** build a read-only recorder just before the
  original clip stores the actual unclipped trial depth and the factor `1 + 0.5 dt v_soil / dx` in `syrup_hook` arrays no original
  routine reads. The **nohook** build proves by a bitwise comparison that the recorder has no effect;
* `syrup_derived_constants`, generated from verbatim line ranges (134-137, 157-173, 206-207, 209-219, 243-324) of
  `src/Subroutines_In_out/initialize_values_xml.f90`; each range is asserted by its first and last line and the whole file by SHA-256.

All 15 original source hashes are embedded in `sources.PINS` (from the audit `fortran_source_pins.json`); a different hash refuses the
build. Chemistry and marker-in-cell are absent (`route_markers_xml` is an error-stop stub, MiC = 0).

## Precision category (important)

`shared_data.f90` declares many globals by implicit typing, so they are default REAL (kind 4): `dt, dx, dx_m, density, sigma,
dstar_const, radius, diameter, viscosity, settling_vel, ustar, d50` and the widened `pi`, `fourth`; `re, ke, p_par, v_soil` are kind 8
(`agent_handoffs/tasks/gpu_sediment/shared_kind_probe.json`, compiled evidence). The harness uses the module unchanged, never
`-fdefault-real-8`, and never overrides the original radii with nominal double values. SYRUP's FP64 laws therefore differ from this
reference by REAL-precision effects: smooth-equation probes use the predeclared **rtol 2e-6 / atol 1e-14**; the walk probe (double
expressions on exactly representable `dx_m`, `dt`) uses 1e-12 / 1e-14; SYRUP backend bounds (sediment 2e-11/1e-14, water
2e-12/1e-14) are unchanged and not applied here. Differences are reported as precision categories, not fitted.

## Conventions

Crank-Nicolson sediment selector 2, primary water method iroute 5 (bisection-CN, the SYRUP-matched method); iroute 2 (native
Newton-CN) is supported by `prepare --iroute 2`. The RFID XML has no selector and the native diagnostic run used Euler (1): that
run is not this reference. The original reads the post-infiltration `d(1)` for the wet laws; A1's default is the previous step's depth.
Quantify (do not fit) that departure with `inject --depth previous` versus `post_infiltration`. The original water keeps its stale-inflow and
bracket behaviour; its budget is reported, never claimed conservative. Fixed composition, unlimited supply, fixed terrain.

## Commands (Codex)

```bash
export PATH=/home/okin/MAPLE/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=''
source benchmarks/chastre/env.sh
# tests (a configured compiler that fails is a FAILURE; only an unconfigured toolchain skips)
python -m pytest tests/legacy_sediment -q
# build both variants
python benchmarks/legacy_sediment/run_fortran_sediment.py build --build-dir outputs/legacy_sediment/build_hooked --hooked yes
python benchmarks/legacy_sediment/run_fortran_sediment.py build --build-dir outputs/legacy_sediment/build_nohook --hooked no
# pilots: RFID 600 s wet (A1 cases exist); capture/snapshots bound at 1, 300, 600
python benchmarks/legacy_sediment/run_fortran_sediment.py prepare --case-kind rfid --case outputs/rfid/case \
    --output-dir outputs/legacy_sediment/rfid_600_in --end-s 600 --capture-steps 300,600 --snapshot-steps 300,600 --allow-maple-source-change
python benchmarks/legacy_sediment/run_fortran_sediment.py run --build-dir outputs/legacy_sediment/build_hooked \
    --prepared outputs/legacy_sediment/rfid_600_in --output-dir outputs/legacy_sediment/rfid_600_fortran
python -m maple_syrup.legacy_driver --case-kind rfid --case outputs/rfid/case --output outputs/legacy_sediment/rfid_600_a1 \
    --end-s 600 --allow-maple-source-change --snapshot-times 300,600
python benchmarks/legacy_sediment/run_fortran_sediment.py compare --run outputs/legacy_sediment/rfid_600_fortran \
    --prepared outputs/legacy_sediment/rfid_600_in --a1 outputs/legacy_sediment/rfid_600_a1 --output outputs/legacy_sediment/rfid_600_compare.json
python benchmarks/legacy_sediment/run_fortran_sediment.py inject --run outputs/legacy_sediment/rfid_600_fortran \
    --prepared outputs/legacy_sediment/rfid_600_in --case-kind rfid --case outputs/rfid/case --step 600 --allow-maple-source-change
# Chastre: dry/adaptation 1 s and 60 s first, the wet 600 s pilot, then (only after review) the full 2700 s window (hours)
```

Chastre: ~4 GB of Fortran module arrays (estimate), stream binary I/O (no giant ASCII); the 2700 s run is expensive: label its cold
and warm costs, report `LOOP_SECONDS` (with diagnostics), `KERNEL_SECONDS` (diagnostics and capture removed) and peak RSS separately.

## Integrity rules (correction C1; authoring note, later executed by the root)

* Binary tags are exactly 8 characters including trailing spaces (`PRE_DS1 `); parsers use exact per-file SCHEMAS (tags, order, counts,
  dtypes, finite/sign roles; duplicates and unknown blocks refused) for the ledger, final maps and every requested capture and
  snapshot. Only the original's negative trial depth (`POST_TRL`) may be negative.
* A missing `result.txt` is a clean failed status (completion-marker reason) with the partial output preserved. The original source
  pins are checked on every run whether or not a build record is supplied; executable/input/source hashes must be unchanged across
  the run.
* All output roots, run directories and report files are validated BEFORE creation against the reference, project, case, MAPLE,
  build, prepared, input and executable trees; reports are exclusive-create. A requested output ROOT may already exist (reused) but
  `<root>/run` may not. Failed runs keep their evidence.
* `prepare` records and re-checks the SHA-256 of the bound XML/rainfall/vegetation/sidecar files (optionally the streamed Chastre tiles
  with `--hash-tiles`) and stores them in `meta.json`; `run` re-checks them after the run (`--rehash-tiles` for the tiles) and records
  exactly what was performed. Nothing is claimed unchanged without that check.
* The input is written as a stream (one block copy at a time) with a digest of the exact bytes; the driver reports progress on stderr
  every 60 steps (counted as diagnostics, not kernel time). Builds save the exact unified diff of the isolated `route_sediment_xml.f90`
  (`route_sediment_xml.patch.diff`) next to the hashes; only the splash pair and the read-only hook may appear in it.
* The injection check uses the widened kind-4 density the original used (`DENSITY_G_CM3` in `result.txt`), recorded with the nominal XML
  value. The comparison also flags the net-erosion map, compares the final depth/soil/discharge/velocity fields and the water integrals,
  and reports mobile inventories as final/peak values, never summed over time.

## Plot 1

Plot 1 uses infiltration model 2 (Hawkins/pavement), which this glue does not reproduce (`inputs_from_legacy_case` refuses it).
Its reference is the existing real-application ledger (`outputs/phase7b/mahleran_ledger_run_v3/Output/syrup_sediment_ledger.dat`):
`compare_legacy_sediment.plot1_golden(a1_ledger.npz, ledger.dat, *, partial_steps=None, time_atol_s=0.0)`.

The shipped helper (integrated 2026-10-08, uncommitted, regression verified) parses the ledger strictly: `fortran_number` accepts the
gfortran omitted-`E` three-digit exponent (`1.9762625833649862-323` is kept as a subnormal, not erased) and refuses non-finite or malformed
tokens; `parse_fortran_ledger` validates 14 tokens per line, iterations 1..steps with six consecutive rows, class IDs EXACTLY 1..6 in every
iteration, and a time that is identical within an iteration and strictly increasing. `plot1_golden` requires the NPZ `ledger (steps, 13, 6)`,
`columns` and `t_s`, all finite; refuses storms of unequal length unless `partial_steps` is declared (then the report is labelled partial with
both lengths); demands an exact time axis by default (a positive `time_atol_s` is an explicit, labelled caller choice; bool/NaN/Inf/negative
values are refused before any file is read). The report separates `columns` (transfer kg per step summed over the compared steps: pickup,
active deposition, clip source, CN export, per class and total) from `storage` (the old/new mobile INVENTORIES as final and peak values with
their times, per class and class-summed, never summed over time). `compare_runs`, `map_stats`, `injection_check`, `build_engine` and `_rel`
are unchanged and pinned by `tests/legacy_sediment/test_plot1_golden.py` (renamed during adoption).
The shipped helper was compared with all 5,400 saved real Plot1 steps: exact time alignment, with the same new-mobile peak time (1,201 s).

## Outputs of one run

`ledger.bin` (13 columns x 6 classes x steps, the A1 column set; `erased_deposition` is not measured by the original), per-step water
series, per-step wet-cell and detaching-cell counts, `final_maps.bin` (cumulative detachment/deposition/clip, final mobile, depth,
velocity, soil water, discharge, terminal mask; full grid with ring, north-first), optional `capture_NNNNNN.bin` (pre/post state for
injection) and `snapshot_NNNNNN.bin`, `result.txt` with timers, water totals and the completion marker. Clipping is the hook's
actual trial, never derived from the balance residual.
