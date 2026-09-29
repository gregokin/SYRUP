# Phase 2 acceptance: corrected Plot 1 import

Accepted 2026-09-29 after Claude implementation and independent Codex review/verification. This accepts initialization only, not water or wind simulation.

## Result

- Real MAPLE compile_case and load_compiled_case produced identical states for a 60 by 20, 0.5 m grid with six classes and 20 allocated voxel levels.
- Case identity: `e1afad46a27b3b428c40f2c7ab17a4af7f0b9d42dc974759d45b0c4c4d64efab`.
- Active layer: 0.002 m; voxel height: 0.1 m; assumed bulk density: 1250 kg/m3.
- Constant datum offset: 0.26 m; minimum sediment depth: 0.3 m. Source slopes retained. Homogeneous per-cell stratigraphy is an assumption.
- Maximum per-cell/class inventory discrepancy: 2.27374e-13 kg. Maximum reconstructed physical-surface discrepancy: 8.88178e-16 m.
- Water/mobile/ledger initially zero; no wind event launched.

## Scientific qualifications

The root XML repeats the first grain raster six times. This corrected fixture selects the six distinct supplied maps, normalizes only their small storage-roundoff mismatch, then applies MAHLERAN's pavement rescaling. It is not an exact root-XML reproduction. Raw inputs and changes are hash-bound in generated reports.

The DEM has a binary trailer; only its declared ASCII header/grid is staged for MAPLE, with omitted bytes recorded. Terrain audit shows all 1200 interior cells reach the south ring under the selected D4 convention, without filling or changing terrain. This is not hydraulic validation. Legacy ring loss/export accounting remains a later task.

Soil/vegetation/pavement/rainfall inputs and flags are recorded for later water laws. The selected stochastic infiltration initialization is unresolved for Phase 3; no exact random realization is claimed. GPU execution is not tested (CuPy absent in the existing environment). Case compilation is a host-side initialization operation.

## Verification

Commands from /home/okin/SYRUP:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m pytest -p no:cacheprovider tests/phase1 tests/phase2 -q
PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -m ruff check src/maple_syrup tests/phase1 tests/phase2
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m maple_syrup.case_import --recipe cases/plot1/recipe.yaml --output-dir outputs/plot1 --expected-maple-root /home/okin/MAPLE
```

Final result: 65 tests passed in 8.06 s; Ruff all checks passed; generation/reload status ok. Outputs are generated in a new directory and never overwrite an existing case. To reproduce, select another new output directory.

Codex checked source-control resolution, independent source-array fractions/orientation, terrain ring behavior, staging, inventory calculations, generated binding and actual compiler calls. Seven mechanical lint items were corrected. A suspected ring-percent error was disproven by an independent regression test: raw reader values were already percentages. The unnecessary conversion was reverted; the test remains. Earlier unaccepted generated reports were preserved separately and are not the accepted case.

No MAPLE/MAHLERAN source edits, dependency installations, solver implementation, GPU acceleration claim or push. Generated case data and raw orchestration evidence remain local and are excluded from Git. The initial repository commit includes the accepted Phase 1 foundation and historical assessment/prototype sources as well as Phase 2.
