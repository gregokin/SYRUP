The setup stage is done: the generator, the 14 × 3 configs, 80 focused tests (all pass, `ruff` clean) and the setup report. I ran no 160-day runs and made no production, old-helper, Git, GPU or network changes. I'm stopping for Root's independent review.

**Cases and runs**
- **Base:** the accepted SR33 1xK case. The SHA and the frozen helper hashes are asserted before and after generation.
- **Grid:** field capacity 9 to 15 vol. % in all three layers, crossed with stress 0.2017 (exact) and 0.30255, giving 14 unique cases. The keys run from `fc09_s1` to `fc15_s1p5`. Each case has numba 1800 s, numpy 1800 s and numba 900 s configs, 42 runs in all.
- **Run files:** every path is absolute and points at this candidate. Each case folder has a byte-identical copy of the guarded bootstrap beside its runner configs. `config_paths.json` follows the earlier layout, with `stress1`, `stress1p5` and per-FC pair families.

**What differs from the base:** only `layers.theta_field_capacity`, `layers.theta_evaporation_stress`, the name, a role note and a new provenance record, plus the backend or storm step of each variant. A whitelist test asserts this for every case.
- **Fixed values:** Ks, K, saturation, wilting point, residual, d, E = 1, forcing, initial water and clocks are bit-identical.
- **Provenance:** the record says field capacity is an explicit override and that nothing else was recomputed. It also says 0.2017 is a rounded value, not the earlier fitted 0.2017058, and is never newly fitted.
- **Historical values:** the original depth-specific field capacity values in the provenance layer properties are left untouched as context.
- **Sources:** these stay at SYRUP 180 files (4e38a472) and MAPLE 412 files (fbc504bb). The old 37d89f23 pin is refused.

**Native controls** (real soil step on the real case column, native conservation bound):
- From [0.12, 0.05, 0.05], FC 9 cuts layer 0 back to 9 vol. % and moves 3 mm to layer 1 even though layer 0 is far below saturation. FC 15 leaves the same state unchanged. So gravity triggers at field capacity, not saturation. Saturation is identical in all cases.
- From [0.14, 0.14, 0.14], bottom drainage matches an independent hand value for every FC. It is strictly decreasing from FC 9 to FC 13 and about zero at FC 14 and 15. FC 9 also drains more than FC 15 with the real d and interface conductivities. The water budget closes within the native bound.
- The stress value cannot change the gravity stage.

The diffusion coefficient is set to zero in the isolating gravity controls only.

**Other setup checks:**
- All 14 × 3 configs pass the real closed validation and source guard.
- Every layer starts below FC 9 vol. %, and every FC sits between the wilting point and saturation.
- Every stress is above every FC.
- Real input preparation is identical to the base.
- No path back to the SR33 or evaporation candidates remains.
- A relative output path resolves to absolute paths and writes nothing outside the output folder.

Not covered: full runs, scores, restart, GPU and scaling. The output directories don't exist yet, because the launcher refuses overwrite.

Files are in `/home/okin/SYRUP/agent_handoffs/tasks/jorn_fc_sensitivity_20261120/`:
- `setup_report.md`
- `config_paths.json`
- `setup_tests.log`
- `study/` (the 14 case folders)
- `analysis_candidate/benchmarks/jorn_fc_sensitivity/generate.py`
- `analysis_candidate/tests/jorn_fc_sensitivity/test_setup.py`