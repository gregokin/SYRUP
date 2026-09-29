# MAPLE-SYRUP

**Sediment Yield, Runoff, and Uptake by Plants**: a water erosion and transport extension for MAPLE (`/home/okin/MAPLE`). It is not a standalone model. MAPLE owns wind physics and the shared sediment bed (voxel column, active layer, grain classes, availability, ledger, topographic commits, snapshots, backend). MAPLE-SYRUP adds MAHLERAN-like water processes on top of that state and invokes MAPLE for everything else.

Project scope and rules: `AGENTS.md`, `claude.md`, `syrup_implementation_phases.md`.

## Current status: Phase 1 accepted, Phase 2 implemented (under review)

Implemented:

- `src/maple_syrup/dependency.py`: resolves the installed MAPLE and refuses a mismatched or unidentifiable one;
- `src/maple_syrup/provenance.py`: bounded source digest and read-only git state;
- `src/maple_syrup/probe.py`: builds a tiny real MAPLE state and checks it;
- `docs/phase1/interface_contract.md`: dependency, state ownership, gaps and selected integration approach;
- `src/maple_syrup/case_import.py` (Phase 2): audits MAHLERAN Plot 1 and compiles it into an actual MAPLE case with MAPLE's own `compile_case`/`load_compiled_case`; recipe `cases/plot1/recipe.yaml`; see `docs/phase2/plot1_import.md`.

There is no rainfall, infiltration, routing or sediment physics yet. The Plot 1 case is a documented **corrected** fixture (six distinct grain maps instead of the root XML's repeated map), not a reproduction of the root XML. `port_feasibility/` is an older, separate feasibility experiment.

## Dependency

MAPLE is used from its existing editable install in `/home/okin/MAPLE/.venv` (distribution `maple` 0.0.1). It is intentionally not listed in `pyproject.toml`, because the PyPI name `maple` belongs to an unrelated project. Its identity, source root and required API surface are checked at runtime; see interface contract section 1.

## Development commands

Nothing needs to be installed. Run from this directory with MAPLE's interpreter and `src` on `PYTHONPATH`:

```
cd /home/okin/SYRUP
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m maple_syrup.probe
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m pytest -p no:cacheprovider tests/phase1 -q
```

`PYTHONDONTWRITEBYTECODE=1` keeps the MAPLE import from writing `__pycache__` into the MAPLE tree; `-p no:cacheprovider` avoids a `.pytest_cache` directory.

Phase 2 (Plot 1 import; the output directory must not exist yet):

```
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python \
    -m maple_syrup.case_import --recipe cases/plot1/recipe.yaml --output-dir NEW_DIR \
    --expected-maple-root /home/okin/MAPLE
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m pytest -p no:cacheprovider tests/phase2 -q
```

Exit status: 0 success (`NEW_DIR/syrup/plot1_binding.json` written), 1 failed audit/check or MAPLE rejection (`NEW_DIR/syrup/FAILED.json` when the directory was created), 2 dependency error. `--no-compile` writes the audited case package without calling MAPLE's compiler.

Probe options:

- `--expected-maple-root /home/okin/MAPLE` (or `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT`): refuse any other MAPLE source.
- `--output FILE`: also write the JSON report; refuses an existing file or a path inside the MAPLE tree.
- `--include-file-manifest`: add per-file SHA-256 entries.
- `--backend cupy`: fails with `BackendUnavailableError` when CuPy or a CUDA device is unavailable (true for the current venv); never falls back to CPU.

Exit status: 0 success, 1 failed check or MAPLE source changed during the probe, 2 dependency/backend/usage error.
