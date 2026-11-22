Stage 1 is done: the JORN defaults preset, the six dry-window cases, the generator, 55 focused tests (all pass, `ruff` clean) and the setup report. I made no dry-window runs and no GPU, Git, network or install changes. I'm stopping for Root's review.

**Defaults preset**
- `cases/subsurface_validation/jorn_defaults.json` holds the user's choice: field capacity 0.10 in all layers (a volumetric fraction, no percent conversion), stress 0.30255, E = 1, d = 0.0010239461376024856, 1x SR33 conductivities with harmonic interfaces, and the 3 × 10 cm geometry.
- It records the source as commit 99465040 and the tracked fc10 stress-1.5 case file (SHA 027dfdd3…). It states that it is a JORN case-level choice, not a refit, a global default or a MOAB default.
- The third-layer wetting discrepancy is recorded as unresolved and deferred, and T = 0 is not mandated for future LEAF coupling.
- `benchmarks/jorn_dry_stress/preset.py` loads and validates it, including finite and range checks, the source SHA, and agreement with the tracked case. The generator builds the JORN configs from it, and `run.py` refuses a JORN config that disagrees.
- `AGENTS.md` has an appended paragraph, with the first 45,188 bytes identical to HEAD (checked with `cmp`). I also added `docs/subsurface_hydrology/jorn_defaults.md`.

**Six cases**
- **Keys:** JORN4_A, JORN4_B and MOAB2_C, each at stress 0.2017 and at 0.30255 (the preset value).
- **Templates:** the frozen historical harmonic configs, SHA-pinned and read-only.
- **Differences:** per case only the field capacity, stress, d, the scenario name, the SYRUP pin (37d89f23 → 4e38a472), the case name and a provenance record of the exact deltas. A whitelist test enforces this.
- **MOAB:** it keeps its own SR33 field capacity from the native default provider on the original MOAB texture and density: 0.1393, 0.1562, 0.1562. This matches the frozen receipt, and the provider reproduces the template's saturation, Ks and wilting point. It is not the old 6 kPa template value and not JORN's 10%.
- **Index:** each case has one config with both backends and the midpoint reconstruction inside it, plus a runner config, a probe, a copy of the bootstrap and a helper manifest. `config_paths.json` is the single absolute run index.

**What the setup checks showed**
- The new source pair passes the source guard and the unmodified template's 37d89f23 pin is refused.
- Real input preparation is identical to the templates: 53 QA observations (28, 11 and 14), the same initial inventory, with initial moisture 3–7 vol. %, below every field capacity.
- One real native daily step per case shows gravity is inactive: gravity on versus off gives identical water, with no drainage and a mass budget within the native bound. So field capacity is not identifiable from these dry data, and the dry test cannot validate the FC choice. I only checked this for the first day. Confirming it over the full windows is left to Root's runs.

Not covered: any window run, backend or restart results, scores, GPU and scaling. The runner sets the data locations absolutely because the worktree has no raw data.

Files are in `/home/okin/SYRUP/agent_handoffs/tasks/jorn_dry_stress_20261122/`:
- `setup_report.md`
- `config_paths.json`
- `setup_tests.log`
- `study/` (the six case folders)
- `analysis_candidate/cases/subsurface_validation/jorn_defaults.json`
- `analysis_candidate/benchmarks/jorn_dry_stress/`
- `analysis_candidate/tests/jorn_dry_stress/test_setup.py`