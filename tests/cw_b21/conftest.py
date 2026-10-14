"""CW-B21 INITIAL-only reimport: shared paths and the genuine compiled synthetic Chastre fixtures (CW-B15 copies)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/phase4", "tests/phase4s", "tests/hydraulic_candidates", "tests", "tests/rfid", "tests/phase5",
            "tests/conservative_water", "tests/cw_b15"):
    sys.path.insert(0, str(ROOT / sub))

from cw_b15_case_fixtures import (  # noqa: F401
    chastre_case_dir,
    dtm_path,
    rfid_source,
    synthetic_recipe_file,
)
