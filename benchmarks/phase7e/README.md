# Phase 7e: selective voxels, legacy replay, bin sensitivity

Shared bed changes remain a patch to actual MAPLE, not a copied SYRUP bed.
The existing Phase7d dependency and live MAPLE/MAHLERAN trees are preserved.
See [selective exchange](../../docs/phase7e/selective_exchange.md),
[legacy replay](../../docs/phase7e/legacy_benchmark.md), and
[bin study](../../docs/phase7e/bin_sweep.md).

## Rebuild and select

First reconstruct the accepted Phase7d dependency as documented in
`docs/phase7d/optimization.md`. Then, from the SYRUP root:

```bash
python3 benchmarks/phase7e/prepare_candidate.py \
  --output outputs/dependencies/maple_phase7e_reviewed \
  --expected-digest "$(cat benchmarks/phase7e/candidate_digest.txt)"
# Use an existing Python environment containing NumPy, Numba and actual MAPLE dependencies.
export PYTHONPATH=/tmp/syrup-numba
source benchmarks/phase7e/candidate_env.sh
/home/okin/MAPLE/.venv/bin/python benchmarks/phase7e/run_storm.py \
  --bins 32 --touched-inventory --allow-maple-source-change --output NEW_OUTPUT_DIRECTORY
```

The preparer refuses an existing output; the environment helper verifies the package,
pyproject and editable metadata. Override `MAPLE_SYRUP_PHASE7E_ROOT` for a different
new path. No live upstream edits or installations are performed. The extra test
patch changes only an instrumentation expectation about the previous padded ledger.
This candidate is an explicit dependency selection; ordinary existing environments
continue to use their selected dependency. The touched-storage diagnostic is opt-in.

## Reproduce separate studies

```bash
# Legacy unlimited-supply, fixed-composition diagnostic (NOT production conservative bed transport):
/home/okin/MAPLE/.venv/bin/python benchmarks/phase7e/run_legacy_benchmark.py \
  --implementation numba --allow-maple-source-change --output NEW_LEGACY_DIRECTORY
/home/okin/MAPLE/.venv/bin/python benchmarks/phase7e/compare_legacy.py \
  --legacy NEW_LEGACY_DIRECTORY --output NEW_COMPARISON_JSON
# Eight bin counts; runner explicitly selects accepted Phase7d for this comparison.
bash benchmarks/phase7e/run_bin_sweep.sh NEW_SWEEP_DIRECTORY
/home/okin/MAPLE/.venv/bin/python benchmarks/phase7e/summarize_bins.py \
  --root NEW_SWEEP_DIRECTORY --output NEW_BIN_REPORT_JSON
```

Inputs are the previously imported `outputs/plot1` and actual Fortran reference
artifacts named in the scripts. These are local generated fixtures, not downloaded
by these commands. Summarization requires all eight bins; failed or missing runs
cannot qualify a reduced count. Each completed run records source/input provenance.

## Checks

`pytest tests --ignore=tests/phase7d` runs the current SYRUP suite with this selected
dependency. Phase7d's optional differential tests deliberately require its exact
older package; do not loosen that identity guard. `tests/phase7e` includes actual
original Fortran walk comparison, legacy analytic checks, public/optimized MAPLE
parity, cache mutation/conversion, topographic commit, and checkpoint continuation.
`test_selective_exchange.py` exercises NumPy and CuPy when an actual device is
available; CPU skips are not GPU evidence. Final results are recorded in
`docs/phase7e/acceptance.md`.
