# ruff: noqa: F811
"""H2 correction03: the cold canonical-depth reader peak is part of the PUBLISHED composed need (Root's 10000-voxel deep fixture, wide, large-N/small-V), explicit batch policy, true traced peak."""
from __future__ import annotations

import tracemalloc

import fpond_builders as fb
import numpy as np
import pytest
import static_case as sc
from case import AVAIL
from test_outer_c2 import case_state, wide_original  # noqa: F401

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import floating_pond as fp
from maple_syrup import native_floating_soil as nfs
from maple_syrup import probe
from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.pond_event import PondEvent, PondEventControl
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_physics import plot1_sediment_parameters
from maple_syrup.splash import splash_parameters

IDENT = {"case": "h2c3", "v": 1}


def deep_original(n_voxels=10000):
    """Root's valid deep fixture: actual native 3x4 cells with `n_voxels` voxels (producer-side only)."""
    real = probe.build_minimal_state

    def producer(*a, **kw):
        kw.update(n_voxels=n_voxels, voxel_dz_m=0.001)
        return real(*a, **kw)

    probe.build_minimal_state = producer
    try:
        bed, context = fb.ph.make_bed(sc.terrain(), depth=0.004)
    finally:
        probe.build_minimal_state = real
    own = nfs.initialize_native_floating_soil(fb.native_bed(bed, context), fb.config(bottom="closed"), initial_relative_saturation=0.97, available_bytes_fn=AVAIL)
    vert = fp.FloatingPondVertical(own, available_bytes_fn=AVAIL)
    ctl = PondEventControl(backend="numpy", time=ExperimentalControl(max_dt_s=0.2, min_dt_s=1 / 1024, max_retries=12, max_steps=50000))
    ev = PondEvent(context, None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, 1000.0, 300.0), sc.topology(), control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, 6), xp=np),
                   floating_vertical=vert)
    return ev, ev.initial_state(bed, vert.initial_soil_water())


def exercise(tmp_path, ev, st, *, traced=True):
    ck = tmp_path / "ck"
    cc.save_floating_context_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    need = cc.context_load_need_bytes(ck, IDENT, available_bytes_fn=AVAIL)
    loads, real = [], np.load
    np.load = lambda *a, **k: (loads.append(1), real(*a, **k))[1]
    try:
        for bad in (0, need - 1):
            with pytest.raises(nfs.NativeFloatingMemoryError):
                cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), IDENT, available_bytes_fn=lambda b=bad: b)
        assert loads == []  # refused before any array copy
    finally:
        np.load = real
    assert cc.context_load_need_bytes(ck, IDENT, available_bytes_fn=AVAIL, reserve_bytes=11) == need + 11
    with pytest.raises(nfs.NativeFloatingMemoryError):
        cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), IDENT, available_bytes_fn=lambda: need + 10, reserve_bytes=11)
    if traced:
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
    e2, s2 = cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), IDENT, available_bytes_fn=lambda: need)  # EXACT published need
    if traced:
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        assert peak <= need, (peak, need)
    assert s2.counters.n_accepted_steps == 0 and e2.vertical._owner is None
    return need


def test_root_deep_10000_voxel_fixture_exact_published_need_loads(tmp_path, case_state):
    ev, st = deep_original(10000)
    need = exercise(tmp_path, ev, st)
    assert need > 17_312_722  # the C2 model (root_h2_valid_deep) omitted the canonical stage that needs 18 066 744 B


def test_wide_and_large_n_small_v_exact_need_loads(tmp_path, case_state):
    ev, st = wide_original(32)
    (tmp_path / "a").mkdir()
    exercise(tmp_path / "a", ev, st)
    sc.TERRAIN.update(ny=3, nx=4)
    ev, st = wide_original(48)
    (tmp_path / "b").mkdir()
    exercise(tmp_path / "b", ev, st)


def test_canonical_stage_is_what_the_real_reader_admits_and_standalone_child_reports_it(tmp_path, case_state):
    ev, st = deep_original(10000)
    v = ev.vertical
    ny, nx = ev.shape
    stage = fc.canonical_depth_bytes(v.n_layers, ny * nx, ny * nx, ev.nc, 10000, windowed=False, batch_cells=v.batch_cells)
    seen = []
    real = nfs._admit
    nfs._admit = lambda model, budget, reader, what: (seen.append((what, model["required_bytes"])), real(model, budget, reader, what))[1]
    try:
        ck = tmp_path / "child"
        fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
        fc.load_floating_pond_checkpoint(ck, ev, IDENT, available_bytes_fn=AVAIL)
    finally:
        nfs._admit = real
    canon = [r for w, r in seen if "canonical" in w]
    assert canon and max(canon) == stage  # the pure model equals the real accepted-reader requirement (reserve 0)
    # the standalone child's published model now covers the stage: exact-model reader loads, below it refuses
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fc.load_floating_pond_checkpoint(ck, ev, IDENT, available_bytes_fn=lambda: stage - 1)


def test_factory_batch_above_the_declared_policy_is_refused(tmp_path, case_state):
    ev, st = deep_original(300)
    ck = tmp_path / "ck"
    cc.save_floating_context_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    with pytest.raises(fc.FloatingCheckpointError, match="batch_cells"):
        cc.load_floating_context_checkpoint(ck, sc.factory("numpy", False), IDENT, available_bytes_fn=AVAIL, batch_cells=fp.nfs.DEFAULT_BATCH_CELLS - 1)
    with pytest.raises(fc.FloatingCheckpointError):
        cc.context_load_need_bytes(ck, IDENT, available_bytes_fn=AVAIL, batch_cells=True)
