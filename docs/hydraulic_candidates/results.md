# Experimental hydraulic candidates: measured Plot1 comparisons

Claude implemented both candidates and bounded corrections; Codex independently reviewed the code, ran actual CPU/GPU checks and storms, and measured these results. Both are usable **water-only experimental tools**. The existing legacy default remains unchanged. See [equations and usage](README.md).

Frozen actual MAPLE Plot1 bed and geometry, 60×20 cells, 0.5 m square cells, parsed rainfall, mean conductivity, existing MAHLERAN-inspired column physics, 5400-second storm and recession, reports every 60 seconds and forcing edges. No splash, sediment, vegetation growth, ET or dry reset. Comparison baseline is the corrected coherent method-5 implementation, rather than a new original-Fortran benchmark. Actual MAPLE package digest `72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65` is a source digest, from recorded upstream revision plus preserved changes.

## Warm complete-event runtime

Intel Core i9-7900X; GTX1080Ti physical GPU1. Three warm single-event runs each at maximum dt=1 s, in forward/reverse/forward contender order. Setup, JIT and final downloads/checks excluded; entry validation, accumulation, reporting and final synchronization included. All samples pass MAPLE water bounds and bed/source guards. No task tests ran concurrently; other user workloads were preserved. NumPy candidates are uncompiled reference implementations. The 60 s warm-up does not cover every deep-water or rejection regime; first-round variation remains visible, and later allocator growth is not excluded. Solver-specific accumulation and reporting costs remain inside the timings. These results do not establish larger-grid GPU scaling, peak memory or whole sediment-model performance.

| Solver | Median wall time | Samples (s) |
|---|---:|---|
| Legacy, prepared Numba CPU | 1.940 s | 1.929, 1.940, 1.974 |
| Legacy CUDA | 7.536 s | 7.452, 7.794, 7.536 |
| Explicit, NumPy reference | 2.456 s | 2.445, 2.930, 2.456 |
| Explicit CUDA | 5.804 s | 6.441, 5.804, 5.669 |
| Local inertia, NumPy reference | 6.180 s | 6.171, 6.180, 6.450 |
| Local inertia CUDA | 10.392 s | 10.673, 10.392, 10.358 |

Explicit CUDA takes 23.0% less time than legacy CUDA on this case, while taking 2.99× the prepared legacy CPU time. Local-inertial CUDA is slower here. No solver replacement is implied.

## Timestep and hydraulic differences

Each row is one full warm CUDA storm with synchronized snapshots at 600, 1200 and 1620 s. Differences use prepared legacy Numba at the SAME maximum timestep. Time here includes snapshot handling and, for legacy, four scheduler segments; use the single-event medians above for the fair runtime comparison. Timestep refinement has no compilation/startup inside the timer.

| CUDA solver | Max dt (s) | Accepted / rejected | Export (m³) | Export difference | Peak difference | Peak time (s) | Snapshot-run time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy CUDA | 1 | 5400 / 0 | 0.164433088 | -0.00000% | +0.0000% | 1333 | 7.521 |
| Explicit CUDA | 1 | 5400 / 0 | 0.164529616 | +0.05870% | +0.1414% | 1332 | 6.491 |
| Local inertia CUDA | 1 | 7440 / 2040 | 0.147973159 | -10.01011% | +7.3246% | 1339.5 | 10.811 |
| Legacy CUDA | 0.5 | 10800 / 0 | 0.164445850 | +0.00000% | +0.0000% | 1333.5 | 15.026 |
| Explicit CUDA | 0.5 | 10800 / 0 | 0.164494088 | +0.02933% | +0.0702% | 1333 | 14.613 |
| Local inertia CUDA | 0.5 | 10800 / 0 | 0.147868422 | -10.08078% | +7.3312% | 1339.5 | 13.310 |
| Legacy CUDA | 0.25 | 21600 / 0 | 0.164452237 | +0.00000% | -0.0000% | 1333.5 | 30.877 |
| Explicit CUDA | 0.25 | 21600 / 0 | 0.164476349 | +0.01466% | +0.0350% | 1333.25 | 22.977 |
| Local inertia CUDA | 0.25 | 21600 / 0 | 0.147850061 | -10.09544% | +7.3071% | 1339.75 | 33.147 |

Explicit export differences halve from +0.0587% to +0.0293% to +0.0147%; sampled hydrograph relative L2 differences likewise halve (0.1305%, 0.0650%, 0.0324%). At 600 s the depth-map relative L2 difference falls from 0.3482% to 0.1736% to 0.0867%. This supports timestep convergence toward the baseline on this case, without proving equivalence for other terrain/forcing.

Local-inertial runoff remains about 10% lower and its sampled hydrograph L2 difference about 13%. At 600 s its depth-map L2 difference is 28.77%, 28.15%, 28.08% as dt decreases. Those persistent differences reflect a different hydraulic approximation and boundary treatment; their separate causes are not isolated. The dt=1 run has 2040 rejected attempts; dt=0.5 and 0.25 have none here. The default local-inertial cell velocity is not qualified for detachment.

## Conservation, numerical correspondence and limits

All 18 timestep/backend runs and all18 balanced timing samples close water using the unchanged MAPLE-derived volume rule, retain water state, preserve the actual bed and pass source-stability checks. Candidate GPU steps reuse the existing compiled column stage; lateral kernels use strict FP64, shared face exchanges and constant launch counts. Static arrays are prepared once.

333 final candidate tests pass, with 5 intentional unsupported explicit+donor combinations skipped; actual GPU cases execute. Coverage includes analytic conservation/steady Darcy/lake-at-rest/backwater tests, both column laws, inactive storage, boundaries, pure rejection, metadata/device guards, short CPU/GPU differentials, CLI and adaptive continuation across nonbinary forcing edges. After text-only corrections to the transfer description,18 affected contracts/device checks pass again. Ruff and whitespace checks pass.

Full dt=1 local-inertial CPU/GPU saved-field comparison DOES NOT meet every existing 2e-12/1e-14 bound. Budgets close and final export differs by 5.4e-14 m³, but small cumulative fields and drying-sensitive velocities fail those bounds. Peak-cell speed reaches about 129.6m/s at a nearly empty cell; its CPU/GPU difference is 0.000476m/s (3.67e-6 relative). Short tests do not qualify the full storm. Final-source CPU/GPU CLIs were rerun for both methods. Explicit passes all saved-field bounds; local reproduces the trialC2 common fields bitwise within each backend and retains the documented cross-backend failures. No bounds were widened. The saved-field comparison at all three timesteps passes for explicit routing; some local-inertial velocity snapshots still exceed the bound at dt=0.5 and 0.25 (worst normalized errors 16.85 and 1.14), while their saved final storage and face fields pass. Refinement reduces sensitivity without establishing universal correspondence.

Cell speed divides updated face flux by END cell depth. Stage-consistent face diagnostics separate that conditioning issue, but do not fix the dynamics: sampled maximum normal-face Froude numbers reach 2.01 and 1.82 at 600/1200 s, with up to 0.42% of positive-depth faces above 0.5. At 1620 s maximum face Froude is0.21 while reconstructed cell Froude reaches~4e4. Neither velocity representation is qualified for erosion; momentum smoothing, wetting/drying, directional friction, outlet sensitivity, omitted advection and long-storm backend conditioning are explicit follow-ups in Phase 4T.

Adaptive continuation retains both face-momentum arrays and the numerical next-step cap. Disk restart is not implemented. Static context contents remain immutable by caller contract; metadata is guarded, content is not hashed each step. No compute-sanitizer or true peak-device-memory result.

## Transfer accounting

For a normal explicit storm the loop counts 10800 packet reads / 1,468,800 bytes; local dt=1 counts 18,960 / 2,578,560 bytes, including recoverable attempts. One initial scalar read and final explicit synchronization are separate. No MAPLE helper upload is counted in either loop, and device tests reject raw host-object array conversions. These are helper-level counters, not a full CUDA transfer trace.

In [CuPy14.2](https://github.com/cupy/cupy/blob/v14.2.0/cupy/_core/core.pyx), Python-scalar conversion uses a device-array allocation and fill kernel; our earlier inference that this proved hidden scalar memory uploads was incorrect. The revised driver removes those per-step allocations, passes times by value, and fills report scalars directly. General host-array conversions can still bypass helper counters.

## Reproduction and evidence

Use the configured MAPLE/CuPy/Numba environment and verified `outputs/plot1`:

```bash
python -m maple_syrup.experimental_experiment --case-dir outputs/plot1 --output-dir NEW_EXPLICIT_OUTPUT \
  --solver explicit --backend cupy --max-dt-s 1 --end-s 5400 --allow-maple-source-change
python -m maple_syrup.experimental_experiment --case-dir outputs/plot1 --output-dir NEW_LOCAL_OUTPUT \
  --solver local_inertial --backend cupy --max-dt-s 0.5 --end-s 5400 --allow-maple-source-change
python benchmarks/hydraulic_candidates/compare_plot1.py --case-dir outputs/plot1 --output-dir NEW_COMPARISON \
  --dts 1,0.5,0.25 --end-s 5400 --snapshot-times-s 600,1200,1620 --allow-maple-source-change
python -m pytest -q tests/hydraulic_candidates
```

Timestep matrix: `outputs/hydraulic_candidates/final_plot1_dts/{comparison.json,maps.npz,final_arrays.npz}` binds source/control/input/kernel metadata and per-sample budgets/guards. It used source manifest v4. The v5 changes only clarify transfer metadata/comments and test wording; numerical functions/kernels are unchanged. Balanced single-event timings use manifest v5, archived in `agent_handoffs/tasks/gpu_hydrology_candidates/final_balanced_storms_v5.json`, with script, logs and all samples. Other manifests, failed reproducers, corrective prompts, exact Claude session/process logs and qualification records remain in that task folder. All work is uncommitted; no upstream tree was changed.
