# Report correction03 author report — format only

Changed only: `build_report.py` (csv.writer now uses lineterminator="\n", line 485) and the 11 `tables/*.csv` (CRLF → LF). Nothing else touched: DOCX, report.md, figures, README, verification.md were NOT regenerated; no table content or numbers changed.

Checks: for all 11 CSVs the parsed rows (csv.reader) are identical before and after, and no CR byte remains (grep -c CR = 0). `ruff check build_report.py`: all checks passed. The builder was not re-run (a rerun would rewrite the DOCX); the change is a one-argument edit that makes future output LF.
