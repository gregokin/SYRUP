# Native Phase 2.5 projection example (Python API, not a command)

Files: `species_v2.json` (catalog `schema_version: 2`; declared **uncalibrated fixture**: grass receiving weights `[0, 0, 0]`, shrub `[0, 1, 1]`, three layers; derived from LEAF's `examples/phase0/species.json`), `phase25_projection_example.py`, `expected_result_cpu_run.json` (the actual output of the CPU run on the exact accepted source).

What it does: asserts the pinned MAPLE (`fbc504bb…`, 412 files), LEAF (`38009872…`, 28 files) and reviewed SYRUP (`37d89f23…`, 180 files) package digests and the external helper hash (and re-reads all four after the work), prints every imported path, validates the catalog, projects a declared 3 × 3 fixture (closed-form checks), builds explicit site controls, binds with strict external pins, captures / exports / cold-stages the compact cached recipe, and prepares the native vegetation law on a prescribed floating column (live and cached bindings give the same law identity; a column with another cell area is refused).

What it does not do: apply a vegetation generation to a lifecycle boundary (that needs a genuinely completed native event; see the repository tests `tests/phase25_vegetation_lifecycle`), write frames, or write any physical checkpoint. Every number is a fixture value, not a calibration.

Run (CPU, existing MAPLE environment; nothing is installed; paths are those of the qualification runtime, adjust to yours):

    export MAPLE=<...>/maple-materialized/5d850789f9e82dc456227cb1eb4dfcc255b3ee4dfb0ed74883813c93e98d0ab3
    export PYTHONPATH=$MAPLE/editable:$MAPLE/source/src:<SYRUP>/src:<LEAF_ROOT>
    CUDA_VISIBLE_DEVICES= NUMBA_CACHE_DIR=/tmp/phase25_example_numba PYTHONDONTWRITEBYTECODE=1 \
        <MAPLE>/.venv/bin/python -B phase25_projection_example.py --out result.json

`<LEAF_ROOT>` must supply both the `leaf` package and the `benchmarks` helper directory. The expected-result file's `runtime` block records the actual paths of the run: SYRUP source = the frozen reviewed source `37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0`, 180 files (the exact accepted source of SYRUP commit `bb28af6c58c2892cb3c065d73d33b88285c498f0`; run on a frozen byte-identical copy), LEAF = the frozen helper-qualified tree, MAPLE = the pinned materialization; the `after` block repeats the three digests and the helper hash and they are equal.
The script exits with code 2 (`PIN MISMATCH`) on any other SYRUP source (for example the earlier 175-file component source). The LEAF package hash alone is not a Git revision: reproduce with SYRUP commit `bb28af6c58c2892cb3c065d73d33b88285c498f0` (branch `feature/subsurface-vegetation`) and LEAF commit `ff127d29bd1d40f17a1800ffeef87bf080343f67` (branch `feature/syrup-hydrology-traits`), both published; their packages digest to the pinned values.
The shipped pins are the Root-recorded identities of the accepted release; a production run uses its own release record.
