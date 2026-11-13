# Portable independent controls (copied, not re-derived)

These four files are byte copies of Root's independent Decimal80 mathematical controls from the task directory
(`root_sr2006_normal_density_oracles.json`, `root_sr2006_density_oracles.json` and the two generating scripts
`root_sr2006_decimal_oracles.py`, `root_sr2006_density_oracles.py`). They were written by Root, without any provider code, from the
primary paper Saxton & Rawls 2006 (SSSAJ 70:1569, doi 10.2136/sssaj2005.0117; source PDF SHA-256
24038366b95d508de140786b5b71fb634164dc9f89d1de158944c445c77186e5). They check a mathematical TRANSCRIPTION of the Table 1 coefficients
plus agreement with the printed Table 3 rounding (1 vol %, 0.1 mm/h); they are not field validation and not an independent check of
every coefficient. The tests read only these copies, so the suite has no dependency on the (ignored) task directory.
