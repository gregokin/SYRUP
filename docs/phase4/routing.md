# Phase 4b — D4 routing graph, method-5 step and original-Fortran harness

Status: implementation and independent CPU verification completed 2026-09-29 for the routing-kernel milestone. See [acceptance.md](acceptance.md) for executed checks (330 passed, 4 GPU skips), direct original-Fortran comparisons, measured costs and remaining Phase 4 work. Claude authored the implementation; Codex reviewed and verified it. There is no coupled storm runner or GPU acceptance yet.

Files:

- `src/maple_syrup/routing.py`
- `src/maple_syrup/routing_numba.py` (optional compiled sweep)
- `tests/phase4/test_routing.py`
- `tests/phase4/test_fortran_routing.py`
- `tests/phase4/fortran_reference.py`
- `benchmarks/phase4/reference_driver.f90`
- `pyproject.toml`: optional extra `numba`, and `tests/phase4` added to `testpaths`

## 1. Graph (`build_routing_graph`, `plot1_routing_graph`)

**Inputs.** The host constructor takes NumPy arrays only; runtime arrays are then placed once in the requested MAPLE namespace through `maple.core.backend.to_device`.

- `elevation_full_m` `(ny+2, nx+2)`: MAPLE orientation (row 0 = south), with a one-cell boundary ring. Every value must be finite.
- `nodata_value` (optional): declared metadata sentinel. Any elevation equal to it, ring included, is rejected. **Restriction:** the legacy aspect search skips `nodata` neighbours (topog_attrib 104–113); this builder does not reproduce that. Without the declaration a finite sentinel such as −9999 passes the finiteness check and would act as a very low receiver. Declare it (the Plot 1 builder passes the DEM header value) or make sure no sentinel is present. Terrains with real nodata holes are unsupported.
- `export_receiver_full` (bool, same shape): non-active cells that export what flows into them. This is the legacy `rmask < 0` semantics.
- `friction_factor` `(ny, nx)`: static Darcy–Weisbach f, finite and **≥ 0.1** on active cells. `route_water.for` 916–920 floors ff to 0.1 *after* the root is found, so for smaller f the legacy `v` and `q(2)` no longer belong to the solved depth and the two codes would disagree. Such friction is refused rather than emulated. Plot 1 (f = 21.45) is unaffected.
- `dx_m`. Rejected when `dx²` overflows or underflows FP64 (face volumes could not be finite).
- `active_mask`: optional. `dy_m ≠ dx_m` is rejected.

**Rules**, reproducing MAHLERAN `topog_attrib.for`:

- **Aspect** (94–117): the lowest D4 neighbour on elevation×1000 (mm), strict `<`, in N, E, S, W order. Codes are 1 = N (+row), 2 = E, 3 = S (−row), 4 = W, the same as the Phase 2 `legacy_d4_aspect`.
- **Slope** (117): `(z_mm − zmin_mm)/(dx_m·1000)`. Legacy arithmetic is used, so ties and slopes are bitwise legacy for representable dx.
- **Masking and cap:** inactive cells get slope 0 (190–197). Slopes above 1000 become 1 (218–224).
- **Edge rule** (265–293): runs **in place in the legacy loop order**. That order is north→south (MAPLE row descending), then column ascending.
- **Conveyance:** `k = sqrt(8·9.81·S/f)`.

**Rejected** (`RoutingGraphError`, with MAPLE and legacy indices of the offending cells):

- sinks and flats (aspect 0; no filling, carving or pit storage);
- a receiver that is neither active nor export-flagged (legacy silently loses that outflow and omits it from `q_plot`);
- an export flag on an active cell;
- an edge rule that would read a ring or inactive slope (legacy would copy 0, leaving an outlet that never drains);
- a zero slope after the edge rule;
- non-square cells;
- nodata or non-finite elevations;
- wrong dtype or shape, and non-host inputs.

**Network.**

- Receivers are flat indices, with `EXPORT = −1` and `INACTIVE = −2`.
- Outlets are the active cells whose receiver exports.
- Donor slots follow the legacy `sdirin` order: MAPLE south, west, north, then east neighbour (`route_water.for` 20).
- Dependency levels: 0 without donors, otherwise 1 + max(donor levels). They are computed on the host by a descending-elevation sweep, which is valid because elevation strictly decreases along every edge.
- Cells are ordered by (level, flat index).
- `donor_position`/`donor_mask` `(4, n_active)` index into level-ordered arrays, so every per-level read is a gather and every write is a contiguous slice.
- `input_sha256` binds elevation, export mask, friction, active mask and dx.
- `summary()` reports:
  - outlets;
  - level count and widths;
  - slope ranges before and after the edge rule;
  - how often the edge rule was applied;
  - aspect counts.

**Plot 1** (`plot1_routing_graph(fields, report)`):

- Uses the sidecar `legacy_full_elevation_m` and `legacy_full_rainfall_scaling`. Both are already south-first; there is no second flip.
- Exports go to cells with `rmask < 0`, which is the south ring. Active cells are those with interior `rmask ≥ 0`.
- f is the type-1 per-surface-type mean, 21.45. For type 1 the legacy setup always passes distribution 0 (storm_setting 563–568), whatever the distribution setting.
- It requires flow_direction 4, method 5, friction type 1 and no friction map.
- It checks that the full interior equals `elevation_source_m` and that the aspect equals the Phase 2 audit.

Codex reported, and the tests check against an independent transcription rather than hard-coding:

- 1200 active cells;
- **10 outlets**, all on the south edge. The 20 south-edge cells with lower exterior neighbours are only candidates.
- 66 levels, maximum width 86;
- raw slopes 0.002–0.124;
- the edge rule applied to 0 outlets.

## 2. Step (`route_step`)

```python
route_step(graph, depth_start_m, old_flow_depth_m, dt_s, *, old_discharge_m2_s=None,
           courant_max=1.0, bisection_iterations=40, root_tolerance_m=1e-11,
           implementation="array") -> RouteStep
```

**Implementation selector.** `implementation` chooses only the ordered sweep; validation, budgets, `RouteStep` and every departure below are shared.

- `"array"` (default): the MAPLE namespace path (NumPy or CuPy), one Python loop over dependency levels and bisection iterations, never over cells. This remains the reference and the GPU-compatible path; no GPU run is claimed.
- `"numba"`: `routing_numba.run_sweep`, one nopython call walking all active cells in level order. CPU NumPy only; a graph in another namespace is rejected before any transfer. A missing Numba raises `NumbaUnavailableError` (a `RoutingError`). **There is no silent fallback either way.** Any other value is rejected. `RouteStep.implementation` records which sweep produced the result.

**Inputs.** Units are SI: depth m, unit discharge m²/s, `c = dt/(2dx)`.

- `depth_start_m` = `h*`: storage after rain/infiltration. It equals the legacy `d(1) + excess·dt`.
- `old_flow_depth_m` = the legacy post-infiltration `d(1)`. It must be ≤ `h*` on active cells.
- `old_discharge_m2_s` = the legacy `q(1)`. The default is `k d(1)^{3/2}`. If supplied, it is checked in depth space: the depth it implies must match `old_flow_depth_m` to within `root_tolerance_m` (+64ε·h). That lets the caller carry the previous `q(2)` exactly as legacy does.

**Per active cell, in level order:**

```
Qin_old = donor sum of the SAME q_old         (coherent face flux)
R       = h* + c (Qin_old + Qin_new − q_old)
root    : h + c k h^{3/2} = R, bisection on [0, R]
q_new   = k lo^{3/2};  h_new = R − c q_new    (storage identity)
F       = dx² c (q_old + q_new)               (to the receiver, or export)
```

**Root: the user-selected bisection baseline.**

- The bracket is `[0, R]`, valid because `g(h) = h + c k h^{3/2} − R` is increasing with `g(0) ≤ 0 ≤ g(R)`.
- There is a fixed number of halvings, 40 by default, giving a width of `R·2^-40`. With no data-dependent loop, there is no per-level host synchronization.
- The flux term inside the comparison uses exactly the closure's floating-point operations. Hence `lo + c q(lo) < R` implies `h_new = R − c q(lo) ≥ lo ≥ 0` in floating point. **Nothing is clipped or capped.**
- `|h_new − lo|` is the constitutive residual. It must be ≤ `root_tolerance_m` (default 1e-11 m = the legacy 1e-8 mm), otherwise `RoutingError` ("increase bisection_iterations"). Mass conservation does not depend on it.
- The legacy bracket `[0, 100(d(1)+excess)]` (or 0.5 mm) is **not** copied, being dimensionally inconsistent and not proven to contain the root.

**Positivity and rejection.**

- `R ≥ d(1)(1 − Cr/2)` with `Cr = q_old·dt/(h_old·dx)`. `courant_max ∈ (0, 2]` (default 1) therefore guarantees `R ≥ 0`.
- A violation raises `RoutingError`, which is the legacy STOP condition; the caller retries with a smaller dt (next task). An `R < 0` flag is also checked.
- `dt ≤ 0` is rejected (not an identity).

**Transactional.**

- The step is pure. All input and output checks are `DeferredChecks` flags, resolved in **one** batched read at the end, so any failure raises before a result exists.
- Inputs are never modified, and outputs do not alias them.
- Inactive cells keep `depth_start_m` bit-for-bit (their stored inventory is preserved, never discarded), exchange nothing, and must have zero `q_old`. The legacy `update_water_flow` zeroes masked depths; that is not ported.

**RouteStep.** Grids are `(ny, nx)`; scalars are 0-d arrays in the graph namespace.

- `depth_m`, `flow_depth_m` (the bisection depth that q and v belong to), `discharge_m2_s` (legacy `q(2)`), `velocity_m_s = k sqrt(h_flow)` (legacy v).
- `inflow_m2_s` (legacy `qin(2)`), `old_discharge_m2_s`, `old_inflow_m2_s`, `face_volume_m3`.
- `export_m3`, and `outlet_discharge_m3_s` = Σ q_new·dx over outlets (legacy `q_plot·dx`, instantaneous).
- `storage_change_m3`, `budget_residual_m3` (flagged against `32ε·(n+2)·Σscale` in the conservative mode), `stale_inflow_gain_m3`.
- `max_courant_old`/`_new`, `max_constitutive_residual_m`, `max_cell_balance_residual_m`, `conservative`, `bisection_iterations`.

**Literal legacy comparison** (`legacy_stale_inflow_step`, **not conservative**, never for production). It is identical except that the receiver's old inflow is a supplied `stale_old_inflow_m2_s`: the legacy `qin(1)` rolled over by `update_water_flow`, which `accumulate_flow` never reassigns. The global balance is reported, not enforced:

- `budget_residual_m3` is the created (or destroyed) water;
- `stale_inflow_gain_m3 = dx² c Σ(stale − coherent)` is its expected value.

**Finite outputs.** Every grid and scalar output (depth, flow depth, discharge, velocity, inflows, face volume, balance scale, storage change, export, residual, tolerance, outlet discharge, stale gain) is flag-checked for finiteness *before* any tolerance comparison, and `q_old` and the right-hand side are checked for overflow. A NaN or infinity therefore raises instead of comparing `False` against a tolerance and passing. With the constitutive relation and `Courant ≤ 2`, physical states cannot reach these limits; the checks close the review-noted holes for pathological inputs and are exercised by tests.

**Compiled sweep (`routing_numba.py`).** By user direction the same corrected bisection is compiled, nothing else:

- `_sweep` mirrors `_route`/`_bisect` statement for statement: donor sum `((0 + d0) + d1) + d2) + d3` with non-donors adding `0.0`, `rhs = base + qin·c`, the bisection multiply sequence `((√mid·mid)·k)·c + mid < rhs`, and `q = (√lo·lo)·k`. `np.sqrt` of a negative right-hand side yields NaN, never an exception, so the shared `R < 0` check still reports it.
- `numba.njit(fastmath=False, nogil=True, boundscheck=False, cache=<off>)`; no `prange`, because downstream cells depend on upstream results. Compilation happens lazily on the first call (cold); later calls with the same argument types reuse it (warm). Cache writing is off unless `MAPLE_SYRUP_NUMBA_CACHE=1`, and then `NUMBA_CACHE_DIR` must point outside every reference tree.
- Parity expectation: **bitwise** equality with the array sweep, asserted by `test_numba_sweep_matches_array_sweep_bitwise`. Numba emits separate `fmul`/`fadd` without contraction flags, so no FMA fusion is expected. If Codex observes an ulp-level difference, that is a finding to report (contraction), not a tolerance to widen silently.
- Optional dependency: `pip install "maple-syrup[numba]"` (`numba>=0.61`). Codex is installing Numba 0.67.0 / llvmlite 0.49.0 for NumPy 2.5.2 / Python 3.12.3 in an isolated `/tmp/syrup-numba` tree, not in MAPLE's venv. `routing_numba.numba_versions()` returns the versions for provenance; `numba_available()` lets tests skip honestly.
- Numba CPU compilation does not establish GPU acceleration. CPU routing-step and cold-start costs are recorded in [acceptance.md](acceptance.md); whole-storm cost remains unmeasured.

**Backend, synchronization and memory.**

- Arrays stay in the graph's namespace. Python loops run over levels × bisection iterations; **there is no Python cell loop**.
- Per level there are:
  - one 4-slot gather;
  - about 9 in-place ufuncs per bisection iteration, into workspace buffers sized to the widest level;
  - slice writes.
- Per step: about 10 full-size float arrays plus 3 buffers of the widest-level size. There is no per-level full copy and one flag read.
- The array sweep scales with levels × iterations: 66 × 40 for Plot 1. Initial synthetic CPU measurements are recorded in [acceptance.md](acceptance.md); they do not establish whole-storm or GPU cost.

## 3. Departures from the original routines (documented, not hidden)

| Item | Original | Here |
|---|---|---|
| Receiver old inflow | stale `qin(1)` after infilt changed `q(1)` (water creation) | same `q_old` as the sender; literal mode available and reports the gain |
| Bracket | `[0, 100(d1+excess)]` or `[0, 0.5 mm]`; can truncate the root and lose water | `[0, R]`, proven |
| Stop test | width ≤ 1e-8 mm; STOP after 10001 iterations | fixed iterations; residual ≤ 1e-11 m checked, else error |
| Storage | `d(2) = dmid` | `h = R − c q(lo)` (exact balance); `|h − lo|` checked |
| Negative RHS | STOP | Courant (≤ 2) rejection, before publishing |
| Masked depth | zeroed by `update_water_flow` | preserved |
| Unmasked ring receiver, sinks, ring-slope edge rule | silent loss; permanent storage; zero slope | rejected |
| Order | ascending contrib quicksort | dependency levels; same donor summation order, so identical sums |
| Units, precision | mm, `dt`/`dx` REAL(4), `1./dt` single | m, FP64 throughout |
| Friction below 0.1 | ff floored to 0.1 after the root (916–920): v, q(2) inconsistent with d(2) | refused at graph build (outside the supported comparison domain) |
| Nodata elevations | skipped in the aspect search (topog_attrib 104–113) | unsupported; refused when a nodata value is declared, otherwise only finiteness is enforced |

## 4. Executed original `route_water` harness

Codex compiled the unchanged sources with GNU Fortran 13.3 (`tasks/phase4a_hydraulic_design/fortran_setup.md`).

- `tests/phase4/fortran_reference.py` builds `shared_data.f90`, `route_water.for` and `ff_type8.for` (linked, not selected) **from the reference tree unchanged**, with `-std=legacy -ffixed-line-length-none -fcheck=all`.
- It also builds `benchmarks/phase4/reference_driver.f90`, with `-std=legacy -fcheck=all`. Codex found that `-std=f2008` rejects the initialized COMMON variables of the original `shared_data` module on USE (viscosity, spq, sma, smb, radius), so the driver is compiled to the same legacy standard as the originals. gfortran still accepts the driver's F2008 statements (`newunit`, `error stop`, `storage_size`) under `-std=legacy`. The driver source is unchanged.
- `-J`/`-I` point at a scratch build directory. Source SHA-256 values are checked against the Codex record, both before the build and after all runs, together with directory listings of the reference tree.

**Driver.** It contains no routing equations. It:

1. reads the full legacy grid (north-first, 1-based, ring included, mm);
2. reads `nr2`, `nc2`, `ncell1`, iroute 5, ff_type 1, dt, dx, the routing order (our level order, a valid topological order), aspect, rmask, slope, ff, `d(1)`, `q(1)`, `qin(1)` and excess;
3. calls `route_water` once;
4. only then opens the output: `REAL_STORAGE_BITS_DT_DX`, then `d(2) q(2) qin(2) v` per cell in ES26.17, then the marker `SYRUP_ROUTE_WATER_DRIVER_COMPLETE`.

Error handling:

- A legacy STOP (possibly exit 0) leaves no marker, and the harness reports the run as incomplete with stdout and stderr.
- `dt` or `dx` not exact in the module's REAL kind is refused with `error stop`.
- Each run uses its own working directory, since route_water may write `fort.51`. Scratch files are reported.

**Legacy input mapping from a SYRUP step:**

- `d(1)` = old flow depth;
- `excess` = (h* − d(1))/dt;
- `q(1)` = old discharge;
- `qin(1)` = coherent donor sum (agreed cases) or a stale value;
- `rmask` = 1 active, −1 inactive interior, −9999 export ring, 1 other ring;
- ring aspect 0.

**Tests** (`test_fortran_routing.py`; all skip honestly without a compiler; the agreed cases and the two known-defect comparisons run for **both** `implementation="array"` and `"numba"`, the latter skipping without Numba):

1. Build provenance: flags (originals `-std=legacy`, driver `-std=legacy`, never `-std=f2008`), compiler version, source hashes.
2. Agreed one-step matches within the legacy tolerance of 1e-11 m, with q, qin and v tolerances derived from it:
   - a linear chain at dt 1 and 0.5 s;
   - a valley and a random branching network (ff 5–30);
   - dry and wetting cells;
   - Plot 1 (1200 cells, the real ring rmask).

   Each agreed case asserts that its roots lie inside the legacy bracket, so the agreement does not depend on the bracket heuristic.
3. **Stale-qin reproducer** (Codex `stale_inflow.f90`, rerun through the tracked driver): the original storage + export is 2.5e-5 m³, created from nothing. The literal SYRUP mode matches its depths; the coherent step stays at exactly 0.
4. **Bracket truncation**: a dry receiver after complete run-on, with 20 mm upstream. The original converges to 0.5 mm and loses about 1.6e-3 m³ in one step (a test expectation). The SYRUP root exceeds 0.5 mm and closes.
5. **Negative RHS**: dt 64 s, h 50 mm. The original prints `RHS < 0` and STOPs with no marker; `route_step` raises a Courant rejection.
6. The driver refuses dt = 0.1 s, which is not exact in REAL(4).
7. The reference tree is unchanged at the end.

This is an executed-**routine** benchmark of one routing step at a time. It is not a MAHLERAN model run and not a matched storm.

## 5. Proposed commands (NOT RUN)

```
cd /home/okin/SYRUP

# Array path only (Numba tests skip if Numba is not importable):
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /home/okin/MAPLE/.venv/bin/python -m pytest -q -rs -p no:cacheprovider tests/phase4/test_routing.py

# Array + compiled path. NUMBA_SITE is the site-packages directory of the isolated
# Numba install (for example /tmp/syrup-numba/lib/python3.12/site-packages);
# it is appended after src so it never shadows MAPLE or SYRUP:
NUMBA_SITE=/tmp/syrup-numba
NUMBA_CACHE_DIR=/tmp/syrup-numba-cache MAPLE_SYRUP_NUMBA_CACHE=0 \
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src:$NUMBA_SITE /home/okin/MAPLE/.venv/bin/python -c "import numba, llvmlite, numpy; print(numba.__version__, llvmlite.__version__, numpy.__version__)"
NUMBA_CACHE_DIR=/tmp/syrup-numba-cache MAPLE_SYRUP_NUMBA_CACHE=0 \
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src:$NUMBA_SITE /home/okin/MAPLE/.venv/bin/python -m pytest -q -rs -p no:cacheprovider tests/phase4/test_routing.py

# Original Fortran routine, both implementations:
MAPLE_SYRUP_GFORTRAN=/tmp/syrup-fortran/root/usr/bin/gfortran-13 \
MAPLE_SYRUP_GFORTRAN_FLAGS="-B/tmp/syrup-fortran/root/usr/libexec/gcc/x86_64-linux-gnu/13/" \
MAPLE_SYRUP_GFORTRAN_LDFLAGS="-B/usr/lib/gcc/x86_64-linux-gnu/13/ -L/tmp/syrup-fortran/root/usr/lib/gcc/x86_64-linux-gnu/13" \
NUMBA_CACHE_DIR=/tmp/syrup-numba-cache MAPLE_SYRUP_NUMBA_CACHE=0 \
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src:$NUMBA_SITE /home/okin/MAPLE/.venv/bin/python -m pytest -q -rs -s -p no:cacheprovider tests/phase4/test_fortran_routing.py

# Whole suite (tests/phase4 is now in testpaths), lint, whitespace:
NUMBA_CACHE_DIR=/tmp/syrup-numba-cache PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src:$NUMBA_SITE /home/okin/MAPLE/.venv/bin/python -m pytest -q -rs -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -m ruff check src/maple_syrup tests
git diff --check
```

Notes on the commands:

- The Numba site-packages path above is a guess at the isolated install's layout; use the actual directory. Appending it after `src` keeps MAPLE's own NumPy first on the path.
- `MAPLE_SYRUP_NUMBA_CACHE` defaults to off, so no cache files are written. `NUMBA_CACHE_DIR` is set anyway so nothing lands in a source tree if caching is ever enabled.
- If the driver fails to find `libgfortran.so.5` at run time, set `MAPLE_SYRUP_GFORTRAN_RUNPATH` to the directory that holds it, or add `-static-libgfortran` to the LDFLAGS.
- `-s` shows the printed graph summary, the build record and the legacy gain/loss numbers.
- Path constant: `REPO = Path(__file__).resolve().parents[2]` in `tests/phase4/*.py` is the repository root (`parents[0]` = `tests/phase4`, `parents[1]` = `tests`). A test asserts this.

## 6. Limits and next task

- Not yet present: storm runner, infiltration/rainfall coupling, adaptive dt retry, hydrograph outputs, restart sidecar, performance or memory figures (array or compiled), a CuPy run, and a matched Fortran water-only storm.
- Constant friction type 1 only, on both implementations. The Numba path is CPU/NumPy only; CuPy graphs must use `"array"`. Parity between the two sweeps is a test expectation until Codex runs it.
- Alternatives (safeguarded Newton, explicit kinematic, local-inertial, GPU dependency-counter kernels) are Phase 4R, after acceptance of this baseline. Nothing here changes the hydraulic method.
- Suggested narrow additions for the storm runner:
  - Per step, `hpre = max(h − max(J − P, 0), 0)`. This equals `min(h, h* − O)`, the legacy `d(1)`.
  - Use `route_step(h* = column depth, old_flow_depth = hpre, old_discharge = q_prev if J ≤ P, 0 if J ≥ h + P, else k hpre^{3/2})`, with the capacity evaluated at the pre-step depth, as in Phase 3.
  - Keep a separate literal comparison channel that feeds the legacy stale `qin(1) = Qin_new` of the previous step to `legacy_stale_inflow_step`, reporting its cumulative gain.
  - Halve dt on `RoutingError` Courant rejections, with a bounded count.
  - Record source, drainage, return and export separately.
- A matched Fortran storm needs a driver around the original `infilt`/`route_water`/`update_water_flow` loop. That is a separate, larger harness.
- Kinematic wave only: bed slope, no backwater, no adverse slopes. Static friction type 1 only. Pits, flats and overtopping are unsupported.
