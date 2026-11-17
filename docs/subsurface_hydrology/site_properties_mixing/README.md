# JORN/MOAB harmonic vs geometric interface-mixing report (2026-11-17)

* `SYRUP_JORN_MOAB_interface_comparison_20261117.docx` — the Word report (781337 bytes).
* `report.md` — the editable Markdown version (same text, tables and figures).
* `build_report.py` — regenerates everything below from `data/` (and `../observed_drydown/data` for the earlier generic trajectories).
* `verification.md` — what was verified and what was not.
* `figures/` — PNG figures used in the report; `tables/` — every table as CSV plus `report_tables.json`.
* `data/`, `config_templates/`, `protocol.md` — sealed inputs, read only.

Rebuild (no install, no network, no models):

```
PYTHONPATH=<task>/report_tools MPLCONFIGDIR=/tmp/syrup-site-mixing-mpl PYTHONDONTWRITEBYTECODE=1 \
  /home/okin/MAPLE/.venv/bin/python -B build_report.py
```

Scope: diagnostic dry-down comparison, not field validation or a native checkpoint/scaling claim. Publication status is tracked separately.
