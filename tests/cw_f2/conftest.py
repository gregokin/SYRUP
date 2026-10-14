"""CW-F2 large resident-water restart byte bound: shared paths and fixtures (CPU only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/phase4", "tests/phase4s", "tests/hydraulic_candidates", "tests", "tests/rfid", "tests/phase5",
            "tests/conservative_water", "tests/cw_b15", "tests/cw_b30", "tests/cw_f2"):
    sys.path.insert(0, str(ROOT / sub))

# Genuine compiled synthetic Chastre / RFID cases (tiny 10 x 8 interior in a nodata ring; actual MAPLE compile).
from cw_b15_case_fixtures import (  # noqa: F401
    chastre_case_dir,
    dtm_path,
    rfid_source,
    synthetic_recipe_file,
)
from test_sediment_experiment import (
    plot1_case,  # noqa: F401  (verified generated Plot 1 case)
)
