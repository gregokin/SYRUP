# Phase 7e Stage 2 — MAHLERAN legacy transport replay (benchmark only)

Status: implemented as an explicit BENCHMARK-ONLY path. `src/maple_syrup/legacy_transport.py`
ports the legacy operator; `benchmarks/phase7e/run_legacy_benchmark.py` replays Plot1 on SYRUP's
hydrology; `benchmarks/phase7e/compare_legacy.py` compares with the actual Fortran reference. The
production event, MAPLE bed exchange and the characteristic scheme are untouched; nothing in the
legacy path writes to MAPLE state.

## What is ported, and from where

- `flow_distrib.for`: source-based deposition. The detached mass of a cell is deposited at once
  along its downslope D4 path with the exponential step-length bins `1 - exp(-dx/L)` (source),
  `exp(-l/L) - exp(-u/L)` with `u = (n + 2) dx`, `l = u - dx` (downstream cell n), while
  `n < nsteps` and the previous fraction exceeds `1e-19 dt`; a boundary-ring cell receives its bin
  and the walk exits on the ring's zero aspect. Walk limits `max(int(limit/dx + 0.5), 2)` with
  10 / 100 / 500 m for diffuse / concentrated / suspended (initialize_values_xml 261-263): 20 /
  200 / 1000 cells on Plot1's 0.5 m grid.
- `conc_flow_transport.for` 49-54: no excess stream power -> the detachment is deposited locally.
- `route_sediment_xml.f90` 236-299 (method 2): Crank-Nicolson pool routing in up- to downslope
  order with the legacy donor direction order (south, west, north, east), `d2 < 0 -> 0` clipping,
  `q2 = d2 v`; time level 1 is the previous step's level 2 (update_water_flow 33-38).
- Outlet export: `sum(q_soil(2) at outlets) * dx * density` (output_hydro_data_xml 130-142).
- Supply is unlimited and composition fixed (update_sediment_flow keeps `sed_propn`): detachment is
  the uncapped law demand and the initial active-layer composition is held constant.
- Virtual velocity: the law's value where a law applies, else the 0.9-per-step recession memory,
  never zeroed for 'settled' cells (update_sediment_flow 69).
- Time level of the laws: `d(1)` (previous step's depth) with the new velocity
  (`--depth-time-level previous`, default; `current` is an explicit variation).

In mass units with square cells (`M = d A`, `Q = M v / dx`):
`M2 = [M1/dt + 0.5 (Qin2 - Q1 + Qin1) + (Det - Dep)] / (1/dt + 0.5 v/dx)`, `Q2 = M2 v / dx`.

## Explicit non-conservation (never a production feature)

The receiving cell's pool is debited for deposition before the mass arrives, so it can go negative;
the legacy clips it to zero and thereby CREATES mass. The effective artificial source
`(-M2_trial)(1 + 0.5 dt v/dx)` is accumulated per cell and class as `clipping_source_kg` and
reported in every ledger row, exactly as in the Fortran audit
(docs/phase7/legacy_sediment_ledger.md). Ring deposition is a separately reported walk diagnostic. It is not debited from
the active mobile-pool equation and must NOT be added to CN export as another
conservation sink. Truncated walk tails likewise remain in the pool equation. The per-step identity
`new - old = (Det - Dep_active) dt - CN_export + clip` closes to FP64 roundoff in the replay
(`max_abs_identity_residual_kg` in the summary).

## Verification of the port

- `tests/phase7e/test_flow_distrib_fortran.py`: the ORIGINAL `flow_distrib.for` (compiled unmodified
  with `shared_data.f90` and a state-only driver) and the Python walk agree to 4e-15 relative on
  random meandering grids with a ring, walk limits 2..200 and travel distances 0.05..40 m, including
  ring deposits.
- `tests/phase7e/test_legacy_transport.py`: bin fractions and ring bin, walk limit and the
  no-capacity local deposit, topological order/donor table, the Crank-Nicolson identity with a
  positive clipping source, Python vs Numba kernel agreement.

## Plot1 replay versus the actual Fortran reference

The final replay uses Numba for both water routing and legacy sediment kernels:
`outputs/phase7e/legacy/plot1_numba`. It exports 0.009216339369260688 kg versus actual
Fortran's approximately 0.009223257 kg (0.075% lower); the peak is at 1261 s in both.
The effective clipping source is 0.291931567 kg, and the maximum per-class/step
identity residual is 1.65e-15 kg. This close replay supports the implementation of
the legacy algorithm; it does not validate that algorithm as conservative.

Loop time was 28.099 s, whole process 30.317 s, peak RSS 343332 KiB. Component times
were 9.099 s water, 14.091 s wet physical laws, and 3.296 s walk/CN. Excluding the
first step (which includes JIT), those components took 8.536, 14.088 and 2.602 s
across 5399 steps. These are one-run observations. This lightweight replay omits
MAPLE bed exchange, evolving composition, availability limits, characteristic
bins, and full production transaction validation; its speed is not a like-for-like
replacement for the complete conservative SYRUP storm.

An earlier array-water run is preserved at `plot1_previous_depth`; its loop took
106.315 s. Saved numerical ledger arrays are compared in
`benchmarks/phase7e/legacy_array_numba_parity.json`.
Detailed actual-Fortran comparisons are in
`benchmarks/phase7e/legacy_comparison_numba.json`; the earlier comparison is retained
as `legacy_comparison.json` with its own run path. Runtime source hashes are in each
summary; later docstring corrections do not change the executed kernels.
