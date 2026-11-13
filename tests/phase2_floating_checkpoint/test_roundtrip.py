"""R2a: in-process dense/windowed save -> load -> continue bit-exact against the uninterrupted run (the fresh-process variant is test_fresh_process.py)."""
from __future__ import annotations

import numpy as np
import pytest
from case import AVAIL, IDENT, RAIN, build, run, snapshot

from maple_syrup import floating_checkpoint as fc


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def same(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), k


@pytest.mark.parametrize("windowed", [False, True])
def test_save_load_continue_is_bit_exact(tmp_path, backend, windowed):
    ev, st, vert = build(tmp_path / "a", backend=backend, windowed=windowed)
    (tmp_path / "a").mkdir(exist_ok=True)
    st = run(ev, st, 8)
    assert vert.stats.depth_changed_columns > 0
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    ref = run(ev, st, 6)
    ev2, _st0, vert2 = build(tmp_path / "b", backend=backend, windowed=windowed)
    loaded = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    same(snapshot(ev2, loaded, vert2), snapshot(ev, st, vert))  # the restored state equals the saved one
    got = run(ev2, loaded, 6)
    same(snapshot(ev2, got, vert2), snapshot(ev, ref, vert))  # and the continuation equals the uninterrupted run
    assert np.asarray(RAIN) > 0
