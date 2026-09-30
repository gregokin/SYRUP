# Fixed MAPLE dependency for Phase 5 comparisons

During Phase 5 work on 2026-09-30, independent aeolian development changed
the live `/home/okin/MAPLE` package. The normal provenance check refused the
next experiment because it no longer matched the imported Plot 1 case.
SYRUP did not modify, reset or adopt that work.

For consistent comparisons, the accepted MAPLE package was reconstructed
in `outputs/dependencies/maple_d3d007024/source`. Its **complete 324-file
package digest** is exactly the case's original binding:

```
d3d007024ff65abc2a0ff179f0f03bcd1c3b27f091493136cce849c34f7a4264
```

The reconstruction uses upstream revision
`74dd3e79d9a7df85244355439e3a4a762abb9273` plus the two pre-existing local
changes in `experiment/orchestrator.py` and `surface/voxels/transfer.py`.
Those changes were already present when Plot 1 was imported. Their exact
patch and dependency identity are retained in
[`maple_baseline.patch`](../../benchmarks/phase5/maple_baseline.patch) and
[`maple_dependency.json`](../../benchmarks/phase5/maple_dependency.json).
An independent Git-archive-plus-patch reconstruction reproduced the same
package hash. This is a fixed installation of actual MAPLE, with no
SYRUP-authored wind implementation.

The snapshot and its local editable-install metadata are read-only. Pip
was run offline with `--no-deps --no-build-isolation --no-compile --target`;
neither the upstream tree nor MAPLE's Python environment was modified.
The editable metadata points at this fixed snapshot, not the live tree.
The full per-file manifest is beside the local snapshot.

## Run against the prepared dependency

From the SYRUP root:

```bash
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
export GIT_CEILING_DIRECTORIES="$PWD/outputs/dependencies/maple_d3d007024/source"
export PYTHONPATH=src:outputs/dependencies/maple_d3d007024/editable:outputs/dependencies/maple_d3d007024/source/src:/tmp/syrup-numba
/home/okin/MAPLE/.venv/bin/python -m maple_syrup.sediment_experiment --help
```

The metadata directory must precede the snapshot's `src` directory so
Python identifies the correct editable distribution. `/tmp/syrup-numba`
is this machine's isolated optional Numba installation; use an environment
with the declared Numba extra elsewhere. If supplying
`--expected-maple-root`, use the snapshot's absolute `source` path.
The Git discovery ceiling prevents MAPLE's own build-provenance helper
from mistaking the enclosing SYRUP repository for a MAPLE checkout. This
snapshot has no Git checkout; its upstream identity is the explicit
revision, patch and complete package hash above.

## Reconstruct elsewhere

These commands use a **new** destination, an upstream checkout containing
the recorded revision, and an environment with MAPLE's dependencies and
setuptools already available. They require no network access:

```bash
set -e
mkdir -p outputs/dependencies
mkdir outputs/dependencies/maple_d3d007024
mkdir outputs/dependencies/maple_d3d007024/source
git -C /home/okin/MAPLE archive 74dd3e79d9a7df85244355439e3a4a762abb9273 src/maple pyproject.toml | tar -x -C outputs/dependencies/maple_d3d007024/source
patch --batch --forward -p1 -d outputs/dependencies/maple_d3d007024/source -i "$PWD/benchmarks/phase5/maple_baseline.patch"
PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -m pip install --no-index --no-deps --no-build-isolation --no-compile --editable outputs/dependencies/maple_d3d007024/source --target outputs/dependencies/maple_d3d007024/editable
```

Before accepting the reconstruction, verify its package with
`maple_syrup.provenance.source_tree_digest` against the hash above and its
project file against `maple_dependency.json`. The experiment also checks
the package hash against the case binding before running, and checks
source stability afterward.

Future MAPLE updates remain upstream changes. Adopt them in an isolated
compatibility test, record the new complete source identity, and recheck
water/bed/ledger contracts and performance. Do not use
`--allow-maple-source-change` merely to suppress a mismatch during a
comparison.
