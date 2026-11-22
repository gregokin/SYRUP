# JORN case-level defaults (user selection, 2026-11-22)

`cases/subsurface_validation/jorn_defaults.json` (schema `syrup.jorn_defaults/1`, version 1) records the user's choice for FUTURE JORN soil-column runs that load it explicitly:

| Parameter | Value | Note |
|---|---|---|
| Field capacity, all three layers | 0.10 (10 vol. %) | scalar override; unit is a volumetric FRACTION |
| Evaporation stress | 0.30255 | = 1.5 x 0.2017; a normalisation scale, user-selected, not newly fitted |
| Evaporation factor E | 1 | |
| Diffusion coefficient d | 0.0010239461376024856 | earlier joint fit at E = 1, T = 0, old field capacity (not refitted) |
| Conductivities | 1x SR33-derived Ks, matrix K, harmonic interface K | unchanged; not recomputed when the field capacity is overridden |
| Layers | 3 x 10 cm | context |

Source: accepted study commit `99465040d947ec6ff377003a88f3b2636d923f15`, tracked case file `docs/subsurface_hydrology/jorn_fc_sensitivity/data/case_numba_fc10_s1p5.json` (SHA-256 `027dfdd313c43646f46763da4fee8096454c511a356ebf57a911c22d9d546c9d`).

Scope. This is a JORN case-level choice made by the user from visual preference. It is NOT a new calibration, a global physics default or a MOAB default (MOAB keeps its own SR33-derived field capacity and properties). Old explicit configs, restart identities, source locks and published results are frozen and are not migrated. The preset does not require T = 0 for future LEAF coupling; T = 0 holds only in the current controlled experiments.

Deferred. The model 20-30 cm wetting at the January 2024 storm against the 25.8 cm sensor is unresolved and deferred by user decision. Outputs and measurements are retained; no sensor, wetting-front or support-cause investigation, censoring or refit is based on it.

Use. `benchmarks/jorn_dry_stress/preset.py` loads and validates the file (closed schema, finite values, source SHA, agreement with the tracked source case) and `benchmarks/jorn_dry_stress/generate.py` builds the JORN dry-window configs from it.
