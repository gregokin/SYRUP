import sys
from pathlib import Path

# Reuse the verified actual-MAPLE physical bed constructors from Phase5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'phase5'))

from test_sediment_experiment import plot1_case  # noqa: F401
