"""Case builders shared by the CW-B20 tests and the subprocess driver dump (synthetic actual-MAPLE beds)."""
from __future__ import annotations

import numpy as np

CASES = ("dense", "window_csr", "window_dense")


def resolve(backend):
    from maple.core.backend import resolve_backend

    return resolve_backend(backend).xp


def build(case, xp, tmp_path):
    """`(s, ctx, control, schedule)`: `dense` = the wet 3-cell chain with dense characteristic phase, interval commits
    and exchange history; `window_csr` / `window_dense` = the deep resident-window MAPLE bed with CSR / dense phase."""
    from test_exchange_history import inputs
    from test_windowed_storm import window_inputs

    implementation = "array" if xp is np else "cuda"
    if case == "dense":
        return inputs(xp, commit_interval_s=2.0, implementation=implementation)
    if case == "window_csr":
        return window_inputs(xp, tmp_path, "csr")
    if case == "window_dense":
        return window_inputs(xp, tmp_path, "dense")
    raise ValueError(case)


def host(value):
    from maple.core.backend import to_host

    return np.asarray(to_host(value))
