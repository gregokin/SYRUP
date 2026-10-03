# Numba CPU hydraulic candidates: verification and timing

Claude implemented both compiled solvers and bounded corrections. Codex independently reviewed the accumulated changes, compiled and tested them, executed full Plot1 storms, and measured the results below. Both remain experimental water-only tools; existing legacy defaults are unchanged.

Actual MAPLE frozen Plot1 bed: 60×20 cells, 0.5 m spacing, mean conductivity, parsed rainfall, 5400 s storm/recession, maximum dt=1 s, reports every60 s and forcing edges. Same forcing and initialization for all contenders. No sediment, splash, evolving topography, ET or dry reset. MAPLE source digest72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65 is unchanged.

## Warm storm runtime

Intel Core i9-7900X; GTX1080Ti physicalGPU3. Three measured complete events per contender, forward/reverse/forward order, after an untimed complete5400 s warmup for each. Preparation/JIT and final downloads/budget/bed/source checks are outside the timer. Public state validation, per-step Numba context/column guards, driver accumulation/reporting, allocations during evolution and final device synchronization are inside. No other task test/benchmark ran concurrently; unrelated MAPLE jobs continued on the shared host. GPU3 was idle apart from19MiB display allocation at initial inspection. An interrupted GPU1 trial was excluded because another MAPLE job occupied it. Variation remains visible; these are measured medians, not isolated-hardware scaling claims.

| Approach | CPU NumPy reference | CPU Numba | GPU CUDA |
|---|---:|---:|---:|
| MAHLERAN-style legacy | — | 2.071 s | 8.046 s |
| Explicit kinematic wave | 2.826 s | 1.577 s | 6.618 s |
| Local-inertial | 6.731 s | 3.415 s | 12.477 s |

Explicit Numba takes 44.2% less time than its NumPy reference (1.79× speed), and 23.9% less than prepared legacy Numba. Local-inertial Numba takes 49.3% less than its NumPy reference (1.97× speed), but remains slower than legacy. GPUs remain slower than their compiled CPU counterparts on this small grid.

| Contender | Three samples (s) |
|---|---|
| legacy_numba_prepared | 2.071, 1.991, 2.099 |
| legacy_cuda | 8.046, 8.119, 8.012 |
| explicit_numpy | 2.908, 2.792, 2.826 |
| explicit_numba | 1.577, 1.499, 1.591 |
| explicit_cuda | 6.265, 7.092, 6.618 |
| local_inertial_numpy | 7.357, 6.720, 6.731 |
| local_inertial_numba | 3.415, 3.595, 3.396 |
| local_inertial_cuda | 12.400, 12.477, 14.798 |

Explicit and legacy each use5400 accepted steps/no rejection. Local inertia uses7440 accepted steps and2040 rejected attempts; extra column/lateral evaluations are included. Comparisons therefore describe each complete event, rather than equal numbers of kernel calls. GPU driver/kernel bookkeeping differs from the legacy fused path and remains inside timing.

Earlier GPU1 results used60 s warmups and a different measurement session; do not combine their samples with this table. Original MAHLERAN9.30–9.75 s historical timings include sediment and whole-process setup/output, so they are not a Fortran hydrology comparator. No new Fortran timing, cold-start comparison, large-grid event scaling or true peak-memory measurement is claimed.

## Numerical qualification

- Final actual CPU/GPU candidate suite:434 passed,12 intentional unsupported explicit+donor combinations skipped,60.84 s. No GPU-unavailable skip. Ruff and whitespace checks pass. Two runtime warnings come from deliberate nonfinite-input probes in the reference; two JUnit warnings concern diagnostic-property schema compatibility, not solver failures.
- Both final-source Numba CLIs complete5400 s, retain water state and preserve the actual MAPLE bed. Every timed sample closes the unchanged MAPLE water bounds and passes bed/source guards.
- Explicit NumPy/Numba independent trajectories pass every checked public step and full-driver field at rtol2e-12/atol1e-14 over5400 steps. Final Numba CLI saved arrays also pass against both prior verified NumPy and CUDA CLI outputs. No bounds widened.
- Local independent NumPy/Numba trajectories DO NOT pass every field bound. Counts agree7440/2040 and water closes; final soil/cumulative intake, hydrograph and velocity diagnostics exceed some strict bounds. At the cell of maximum peak-velocity difference, NumPy reports129.627367 m/s and Numba129.626756 m/s: absolute difference0.000611261 m/s, relative4.716e-06; maximum single-step velocity difference across independently evolved states is0.000688826 m/s. The already documented unbounded near-dry velocity and Froude limitations remain; neither CPU nor GPU local velocity is qualified for erosion.
- At90 s, the local CLI differs only in final velocity:3/1200 cells, max absolute2.4673e-14 m/s, normalized2.086. The test audits this known exception at unchanged bounds and asserts no other field failure; this is not a claim that velocity passes.
- Controlled local isolation on all7440 accepted NumPy trajectory steps: compiled lateral stages supplied IDENTICAL ColumnStep and state match every public field bitwise. Complete Numba steps supplied the same input state pass the bounds on every step; column differences reach4.337e-19 m surface depth and1.388e-17 m soil water. This case-specific evidence supports amplification of tiny column differences in independently evolved local trajectories. It does not establish universal correspondence or qualify erosion velocities.
- The isolated evidence used archived trial1; final numerical ASTs of both kernels, wrappers and guards are identical. Full-driver/CLI qualification and timing use the final source manifest. All earlier failures/raw evidence are preserved.

Explicit total exported water is0.164529616 m³ (legacy0.164433088 m³), about+0.0587%; local0.147973159 m³, about−10.01%. Those method differences are unchanged by compilation. See [earlier timestep and physical comparisons](results.md); this compilation task makes no hydraulic-physics change.

## Use and evidence

Select `--backend numpy --implementation numba` with either `--solver explicit` or `--solver local_inertial`. Python API: `experimental_numba.NumbaHydraulicSolver`, served by the existing shared storm driver. Missing Numba causes an explicit refusal; no silent fallback. NumPy reference and CUDA choices remain available. See [usage and contracts](README.md).

Raw evidence: agent_handoffs/tasks/candidate_cpu_numba/{final_tests.log,final_tests.xml,final_full_parity.json,final_cli_parity.json,final_balanced_storms.json,trial1_local_isolation.json,trial1_final_numerical_ast.json,final_source_manifest.json}. Exact commands, source snapshots, failures and Claude prompts/reports are archived in that task. Final CLI outputs: outputs/hydraulic_candidates/final_{explicit,local_inertial}_numba. No commit/push or upstream edits.
