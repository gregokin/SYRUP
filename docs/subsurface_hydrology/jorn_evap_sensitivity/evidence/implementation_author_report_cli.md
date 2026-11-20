Stage 1 is done: the eight-case evaporation/stress setup, the generator, 55 focused tests and the setup report. The tests pass on CPU with CUDA hidden and `ruff` is clean. I ran no 160-day runs and made no production, old-helper, Git, GPU or network changes. I'm stopping for Root's review before any full runs.

**Cases and runs**
- **Base:** the accepted SR33 1xK case, with its SHA asserted.
- **E family:** baseline, then E = 0.8, 0.6 and 0.4 at stress 0.20170581117963332.
- **Stress family:** stress multipliers 1.5, 2, 2.5 and 3 at E = 1, relative to that same baseline. The baseline is shared, so there are 8 unique cases.
- **Runs:** each case has numba 1800 s, numpy 1800 s and numba 900 s configs, 24 in all. Every runner config and probe uses absolute paths and points at this candidate, with no path back to the SR33 candidate. Each case folder has a byte-identical copy of the guarded bootstrap beside its runner configs and its own frozen-helper manifest.

**What differs from the base:** only `physics.evaporation_factor` or `layers.theta_evaporation_stress`, the case name, a new provenance record, and the variant's backend or storm step. The provenance record has the changed controls, the E/r² coefficient and the full stress calibration chain, and says it is not refit. Ks, K, d, SR33 field capacity and saturation, initial water, forcing and clocks are bit-identical. Source pins stay at SYRUP 180 files (4e38a472) and MAPLE 412 files (fbc504bb), and the old 37d89f23 pin is refused.

**Stress above saturation:** stress 0.504 and 0.605 exceed saturation (0.464). The config and native layers accept them without clamping, and saturation, wilting point and field capacity are unchanged.

**Native component control:** for r = 1.5, 2, 2.5 and 3, I ran the real native soil step on the real case column in three states: below field capacity, near residual, and above field capacity so that gravity acts first. A column with stress r times the baseline matches a column with stress at the baseline and E/r² on actual E (rel 1e-14), unmet E and layer water. Both match an independent hand value, and the mass budget is exact. The evaporation stage sees exactly field capacity after gravity, so the wet branch is unreachable. A supply-limited case gives the same E and unmet in both, and the E family scales demand linearly. This shows each stress case is equivalent to E/r², not an independent calibration, so no extra 160-day run was needed.

**Setup checks:** real input preparation gives identical initial water (22.9424 mm), 25.088 mm rain, PET and observation targets across all eight cases. A relative output path resolves to absolute paths and writes nothing outside the output folder.

Not covered: full runs, restart, GPU, scaling and scores. Root runs the 24 configs, then resumes me for the DOCX.

Files are in `/home/okin/SYRUP/agent_handoffs/tasks/jorn_evap_sensitivity_20261120/`:
- `setup_report.md`
- `config_paths.json`
- `setup_tests.log`
- `study/` (eight case folders with the 24 configs)
- `analysis_candidate/benchmarks/jorn_evap_sensitivity/generate.py`
- `analysis_candidate/tests/jorn_evap_sensitivity/test_setup.py`