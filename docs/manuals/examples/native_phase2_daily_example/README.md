# Native Phase 2 daily example (Python API, not a command)

`phase2_floating_daily_example.py` is the exact script published with the Native Phase 2 source (`docs/subsurface_hydrology/phase2_floating_daily_example.py`, SHA-256 `301e9b2d62950e046676a8c1a50dea6c25f947ed008ed05c26d7061896e8bc1b`). It is **not** a standalone CLI, a scenario or a calibrated simulation, and it is not part of the historical CLI release.

**Important wording note.** The script's own docstring and comment say "LEAF withdrawal"; in fact the withdrawal is a **prescribed demonstration request** written in the script (a constant 1e-5 m on the top layer of every cell) applied at loop index `d == 1`, i.e. on the **second completed day**. **No LEAF model, plant state or LEAF integration is involved.** The script bytes are kept exactly as published (SHA-256 above) because that hash is the receipt; this note, not the script text, states what it does.

What it does: builds a small genuine MAPLE bed and native floating owner from the **repository's own test fixtures**, starts a floating daily lineage, runs no-rain days (default 4; the qualification run used 7) with one prescribed demonstration withdrawal on the second day and an every-N-day moisture output, and prints the saved frames. `N` and the day count are ordinary arguments of this script.

Dependencies (not installed or provided by this archive): the Native Phase 2 source tree (`feature/subsurface-evolving-soil`, commit in `qualification/phase2_publication.json`), actual MAPLE `9f466d2cfcd52991b24496cf67f4562d6268afa9` importable as for the qualified tests, and the repository's `tests/` directory (the script adds `tests/phase2_native_floating_soil`, `tests/phase2_compact_soil_profile` and `tests/phase2_floating_daily` to `sys.path`; run it from the repository root):

    python docs/subsurface_hydrology/phase2_floating_daily_example.py OUTDIR 3 7

The qualification record ran this pattern for seven nominal days with `N = 3` (frames saved on days 3 and 6). This archive ships the script text only; it was not re-run when the documents were built.
