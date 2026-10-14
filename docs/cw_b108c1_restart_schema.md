# CW-B108C1: strict checkpoint deep-storage schema, source refusal order, separate-process restart proof

Corrections to the B108 candidate (review freeze `b260b27f…`); see `cw_b108_exact_overlay.md` for the option itself.

## Checkpoint `deep_storage` schema (`checkpoint.load_checkpoint`)

The KEY's presence decides, not its value. When `deep_storage` is present it must be an exact non-default record:
a dict with exactly the five keys, `bed_storage == 'windowed'`, a valid non-default option, `store_format_version` a
plain `int` in (1, 2) (bool, float such as `2.0`, strings and other values are refused) and both digests lowercase
64-hex strings. An explicit `null` (or any other value) on a default bundle is malformed, never "absent". All of this is
refused before any payload is hashed or read, with no writes and no change to the caller's context. The context
encoding / bound comparison and the later binding to the actually loaded MAPLE version (format, physical manifest,
logical manifest) are unchanged. A default bundle without the key loads exactly as before (M94 / B104C2 files).

## Source refusal order (`windowed_experiment`)

`run_complete_windowed` now refuses a non-default option on an unsupported MAPLE source (e.g. M94) before case
verification, output or store creation (`check_deep_storage_source`; the default needs no MAPLE read and is unchanged).
`_setup_windowed` checks the source before importing `OVERLAY_MAX_VOXELS`, which M94 does not define.

## Separate-process restart proof

`agent_handoffs/tasks/conservative_water/cw_b108c1_restart.py` runs reference / pause / resume as three separate child
processes (CPU NumPy/Numba and GPU0 CuPy/CUDA) on the declared 12×7×40×3 fixture with the same labelled settings as the
B108 tests. The resume process rebuilds context and parameters from the fixture and loads every state array from the
bundle and its referenced store. The in-process tests in `test_cw_b108_overlay_events.py` remain same-process reloads
and are not described as cross-process tests.

Limit: in this fixture all deep-store updates precede the 12 s pause (paused version v2, id 2). The continuation
(including terminal settlement and the final forced commit) creates no further deep version, so cross-process
continuation of new overlay publications is not exercised here.
