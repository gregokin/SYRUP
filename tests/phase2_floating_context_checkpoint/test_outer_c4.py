# ruff: noqa: F811, RUF059
"""H2 correction04: SAVE composed admission includes the cold canonical-depth stage; the Windowed canonical stage uses the REAL deep-store mapping bound (from the verified deep record), not the resident archive size."""
from __future__ import annotations

import tracemalloc

import fpond_builders as fb
import numpy as np
import pytest
import static_case as sc
from case import AVAIL, build
from test_outer_c2 import case_state  # noqa: F401
from test_outer_c3 import IDENT, deep_original

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import native_floating_soil as nfs


def first_save_need(ev, st, tmp, reserve=0):
    seen = []
    real = fc._read
    fc._read = lambda fn, need, what: (seen.append((what, need)), real(fn, need, what))[1]
    try:
        fc.save_floating_pond_checkpoint(tmp / "probe", st, ev, IDENT, available_bytes_fn=AVAIL, reserve_bytes=reserve)
    finally:
        fc._read = real
    return next(n for w, n in seen if "save" in w)


def save_controls(ev, st, tmp):
    (tmp / "w").mkdir()
    first_save_need(ev, st, tmp / "w")  # warm-up: the first-use cold source-pair hash allowance is counted only while that cache is cold (existing R1 behaviour)
    (tmp / "n").mkdir()
    need = first_save_need(ev, st, tmp / "n")
    # reserve counted once
    (tmp / "r2").mkdir()
    assert first_save_need(ev, st, tmp / "r2", reserve=13) == need + 13
    for bad in (0, need - 1):
        dest = tmp / f"bad{bad}"
        with pytest.raises(nfs.NativeFloatingMemoryError):
            fc.save_floating_pond_checkpoint(dest, st, ev, IDENT, available_bytes_fn=lambda b=bad: b)
        assert not dest.exists() and not list(tmp.glob(f".bad{bad}.*"))  # refused before any owned work
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    fc.save_floating_pond_checkpoint(tmp / "exact", st, ev, IDENT, available_bytes_fn=lambda: need)  # EXACT published save need
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak <= need, (peak, need)
    return need


def test_dense_deep_10000_voxel_save_exact_need_is_the_canonical_stage_composed(tmp_path, case_state):
    ev, st = deep_original(10000)
    need = save_controls(ev, st, tmp_path)
    assert need >= 18_066_744  # the real canonical reader requirement (Root's probe refused the old 12 927 876 B model)


def test_windowed_resident_v300_container_and_child_exact_need(tmp_path, case_state):
    (tmp_path / "a").mkdir()
    ev, st, vert = build(tmp_path / "a", windowed=True)
    col = st.bed.voxel_column
    deep = fc.pc._deep_record(col, ev)
    bound = fc.deep_cache_bound_bytes(deep)
    assert bound >= int(col.deep_store.mapped_bytes_bound)  # the model never undercuts the REAL handle
    assert bound == int(col.deep_store.mapped_bytes_bound)  # default policy, full encoding: exact (7 372 800 B in Root's fixture)
    (tmp_path / "s").mkdir()
    save_controls(ev, st, tmp_path / "s")
    sc.CASE.update(splash=True)
    ck = tmp_path / "ck"
    cc.save_floating_context_checkpoint(ck, st, ev, IDENT, pref_spec=_pref(ev), available_bytes_fn=AVAIL)
    need = cc.context_load_need_bytes(ck, IDENT, available_bytes_fn=AVAIL)
    loads, real = [], np.load
    np.load = lambda *a, **k: (loads.append(1), real(*a, **k))[1]
    try:
        for bad in (0, need - 1):
            with pytest.raises(nfs.NativeFloatingMemoryError):
                cc.load_floating_context_checkpoint(ck, sc.factory("numpy", True), IDENT, available_bytes_fn=lambda b=bad: b)
        assert loads == []
    finally:
        np.load = real
    assert cc.context_load_need_bytes(ck, IDENT, available_bytes_fn=AVAIL, reserve_bytes=9) == need + 9
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    e2, s2 = cc.load_floating_context_checkpoint(ck, sc.factory("numpy", True), IDENT, available_bytes_fn=lambda: need)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak <= need and s2.counters.n_accepted_steps == st.counters.n_accepted_steps
    # the standalone child at its own published model
    (tmp_path / "c").mkdir()
    child_need = first_save_need(ev, st, tmp_path / "c")
    fc.load_floating_pond_checkpoint(ck / "child", e2, IDENT, available_bytes_fn=lambda: max(need, child_need))


def _pref(ev):
    from maple_syrup import floating_held_context as fh

    return fh.PreferentialSpec(np.full(ev.vertical.n_cells, 2.0e-3), 0.5, 1e-5, np.array([0.0, 1.0, 1.0]), (), None)


def test_larger_v_and_chunks_windowed_bound_matches_the_real_handle(tmp_path, case_state, monkeypatch):
    orig = fb._windowed_helpers

    def larger():  # each call loads a FRESH helper module: patch its voxel count/size so the REAL native bed is larger (NV x DZ stays 0.3 m)
        m = orig()
        m.NV, m.DZ = 1200, 0.00025
        return m

    monkeypatch.setattr(fb, "_windowed_helpers", larger)
    (tmp_path / "a").mkdir()
    ev, st, _v = build(tmp_path / "a", windowed=True)
    col = st.bed.voxel_column
    assert int(col.n_voxels) == 1200
    assert fc.deep_cache_bound_bytes(fc.pc._deep_record(col, ev)) == int(col.deep_store.mapped_bytes_bound)
    assert int(col.deep_store.mapped_bytes_bound) > 7_372_800  # larger V: a different, larger real bound
