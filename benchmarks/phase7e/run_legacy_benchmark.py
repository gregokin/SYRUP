"""Compatibility wrapper: the legacy replay now lives in `maple_syrup.legacy_experiment`.

Same options as before (--output, --case, --applied-rainfall, --implementation, ...). The numerical code,
ordering and outputs are unchanged; see src/maple_syrup/legacy_experiment.py.
"""
import sys

from maple_syrup.legacy_experiment import main

if __name__ == "__main__":
    sys.exit(main())
