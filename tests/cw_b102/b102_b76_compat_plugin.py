"""CW-B102 compatibility shim for the HISTORICAL cw_b76 oracle harness (the historical test file is not edited).

`tests/cw_b76/test_cw_b76_paired.py::as_types` rebuilds a candidate object as the frozen B73 oracle type by passing
EVERY init field. CW-B102 adds two opt-in `SedimentEventControl` fields (`deep_publication`,
`deep_publication_high_water_records`) and one `SedimentEventResult` field (`deep_publication`) that the B73 oracle
classes do not have, so the unchanged harness raises `TypeError` before any differential runs. This plugin wraps
`as_types` to omit ONLY those B102 fields, and ONLY when they hold their defaults ('step' / None / None) -- the values
under which the candidate's code path is the B100 one. A non-default value is never dropped; the wrapper then calls the
original, which refuses. Load with `-p b102_b76_compat_plugin` (this directory on sys.path)."""
from __future__ import annotations

import dataclasses
import sys

B102_DEFAULTS = {'SedimentEventControl': {'deep_publication': 'step', 'deep_publication_high_water_records': None},
                 'SedimentEventResult': {'deep_publication': None}}


def pytest_collection_modifyitems(session, config, items):
    module = sys.modules.get('test_cw_b76_paired')
    if module is None or getattr(module, '_b102_shimmed', False):
        return
    original = module.as_types

    def as_types(oracle, obj):
        drop = B102_DEFAULTS.get(type(obj).__name__, {})
        if drop and all(getattr(obj, k) == v for k, v in drop.items()):
            cls = getattr(oracle, type(obj).__name__)
            return cls(**{f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init and f.name not in drop})
        return original(oracle, obj)

    module.as_types = as_types
    module._b102_shimmed = True
