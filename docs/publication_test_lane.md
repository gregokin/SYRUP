# Portable subsurface science regression lane (`tests/subsurface_release`)

A small CPU lane of already-accepted equation/accounting tests of the layered SYRUP water kernels, run against whichever `maple_syrup` package you put on `PYTHONPATH`. It uses the selected SYRUP source with MAPLE’s backend infrastructure, NumPy, Numba and pytest. Select a qualified MAPLE runtime separately; these equation tests do not qualify an arbitrary dependency checkout.

## Run
    PYTHONPATH=<SYRUP>/src python -B -m pytest -q -p no:cacheprovider -c /dev/null --rootdir=$PWD --confcutdir=$PWD/tests/subsurface_release --import-mode=importlib tests/subsurface_release

* `-c /dev/null` ignores any repository pytest configuration and `--confcutdir=$PWD/tests/subsurface_release` (the lane directory itself, NOT the repository root) stops conftest discovery there, so a surrounding repository's root conftest is never loaded (verified with a failing canary conftest above a copy of the lane); `--import-mode=importlib` is needed because several directories reuse test file names.
* Python needs the repository's normal scientific environment (NumPy, Numba, pytest). No GPU is used or probed: the CUDA tests (the `gpu` fixture of each conftest) skip BEFORE the backend is imported unless `SYRUP_RELEASE_GPU=1` is set; running them additionally needs an operator-selected device (`CUDA_VISIBLE_DEVICES` set before the process starts) and hardware authorization. They are not run by this lane's authors; a skip makes no hardware claim.
* Expected (CPU environment above): 150 passed, 4 skipped (the CUDA ones).

## What is covered (existing accepted tests, byte-identical except one fixture path)
| directory | file | topic |
|---|---|---|
| `subsurface` | `test_step_physics.py` | field-capacity gravity vs saturation storage, open/closed bottom, exact water balance, 1/2/3 layers |
| `subsurface` | `test_numba.py` | NumPy reference vs genuine Numba equivalence (declared bound; bitwise without diffusion) |
| `matrix_infiltration` | `test_step_physics.py` | persistent Green-Ampt: supply, ponded moisture, vegetation factor, capacity-limited ponds, 90-digit Decimal oracle (`root_decimal_oracles.json`) |
| `matrix_infiltration` | `test_numba.py` | NumPy vs Numba |
| `preferential_infiltration` | `test_step_physics.py` | one-hour-equivalent response bounded by available water/capacity, all-zero weights, conservation |
| `preferential_infiltration` | `test_numba.py` | NumPy vs Numba |
| `subsurface_tietjen` | `test_postcap_evaporation.py` | soil-evaporation share applied AFTER the combined cap, actual E only debited, no duplicated T, closed/open bottom |

The helpers' `BIG = 1 << 40` is a declared UNIT-TEST budget for tiny arrays, not a free-memory reading, admission or scaling/RSS evidence (production and the native CLI use genuine host/device readers separately).

Helper modules (`column_cases.py, `matrix_cases.py`, `pref_cases.py`, `tietjen_cases.py`) are the originals. Edits: `matrix_cases.ROOT_ORACLES` points to the fixture shipped beside it; the `gpu` fixture of each conftest has the explicit opt-in guard; the `BIG` lines carry a clarifying comment. No test logic changed.

## What this is NOT
Unit/equation qualification of the low-level column, matrix, preferential and evaporation physics only. It is not native-bed or runtime-lock integration, not CPU/GPU equivalence, not field validation (the parameters are test values, not site properties) and not a whole coupled simulation. Field validation, scaling and full wind-water-LEAF coupling remain UNQUALIFIED and belong to later phases. The separate native CLI receipts cover only bounded native integration, backend and restart behaviour on explicit small cases; they are not field validation or whole-coupled acceptance. Not reused here (still in the qualified historical suites): streaming/memory/layers/diffusion/transpiration/closed-bottom/donor-depth tests, the implicit-equation reference-helper tests, the matrix history tests and all CUDA tests.
