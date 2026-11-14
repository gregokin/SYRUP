# Observed shallow-profile dry-down experiments (JORN and MOAB, NEON RELEASE-2026)

Status: actual CPU results exist and are analysed in `report.md` and `SYRUP_observed_drydown_results.docx`. The experiments are a generic user-approved reference-soil mechanism diagnostic with observed initial moisture and observed air temperature: not a site hydraulic validation, not a calibrated diffusivity, not a field evapotranspiration budget. **GPU / CUDA observed-profile qualification is pending and is not claimed**: the CUDA path of the driver is authored and tested on the host only, and no CUDA result exists.

## Contents
* `data/` — the executed CPU results and their explicit configs: `JORN4_A_training`, `JORN4_B_heldout`, `MOAB2_C_heldout` as `*_config.json` and `*_result.json`, plus `completion.json`, `runtime.json` and the frozen configs. The public CPU configs are `data/*_config.json`.
* `evidence/` — Root's receipts (data and source audits, selection and persistence audit, Decimal control, citations, seal).
* `report.md`, `SYRUP_observed_drydown_results.docx`, `figures/`, `report_tables.json` — the report, built by `build_report.py` from `data/` and `evidence/` only (no simulation, no raw NEON file).

## Verification status (CPU)
Root's CPU run: the unit tests pass (68 passed, 2 GPU-gated tests skipped); 62 actual scenarios (56 training including two closed-bottom sensitivities, 3 JORN B and 3 MOAB C held-out) ran on the native column with NumPy and Numba, each with a midpoint reconstruction, and the whole trajectories and ledgers agree; independent controls (original CSV joins, QA, forcing, exact initial integral, scores, a 70-digit decimal re-solution, and a selection and persistence audit) passed.

## Running the driver
```
python -m benchmarks.subsurface_validation.drydown_driver --config CONFIG.json [--out DIR] [--execute]
```
Run from the checkout that contains `benchmarks/`, with the sealed MAPLE 9f materialization and the SYRUP `src` on PYTHONPATH and the data locations in the `SYRUP_PHASE3_*` environment variables (see `benchmarks/subsurface_validation/paths.py`). The config schema is `syrup.observed_drydown.config/2`. Without `--execute` the driver prints the plan only; with `--execute` it runs the native column. No other flag or approval field is required or accepted: the explicit config (properties, pins, expected source identities) is the only authority. Every config pins the inventory file, the corrected-depth table, the air-temperature manifest and the expected SYRUP and MAPLE package identities (accepted source commit bb28af6c, package 37d89f23, 180 files; analysis baseline cd07f46); a different source is refused, never adopted automatically.

CPU configs list `controls.backends = ["numba", "numpy"]`. A GPU variant is made by changing ONLY `controls.backends` to `["cuda", "numpy", "numba"]`; it still needs a real CUDA device, and it never falls back to the CPU. Such a run has not been made.

## Building the report
```
MPLCONFIGDIR=/tmp/syrup-observed-mpl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/syrup-manuals-citation-tools-20261020 /home/okin/MAPLE/.venv/bin/python -B build_report.py
```
(existing python-docx helper bundle; nothing is installed). The builder states the hardware status as pending unless `data/` contains a genuine CUDA completion record.
