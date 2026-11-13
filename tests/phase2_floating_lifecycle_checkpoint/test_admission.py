"""S1 correction01: ACTUAL-dimension admission controls (native headers, resident cache bound) on genuine lifecycle Boundaries: deep Dense (V=5461, K6), Wide (32x32, V=60, NB32), native Windowed deep V=5000. Published need reaches and
finishes the load; need-1 / 0 refuse before any hash or array; the reserve is counted once; a huge/malformed header is refused before any array; the Windowed need holds the REAL mapping bound (never a compressed-file proxy)."""
from __future__ import annotations

import json
import shutil
import tracemalloc

import fpond_builders as fb
import numpy as np
import pytest
from bridge_builders import AVAIL, complete
from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as fcc
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_lifecycle_checkpoint as lcp
from maple_syrup import native_floating_soil as nfs
from maple_syrup import pond_checkpoint as pc
from maple_syrup import probe
from maple_syrup.pond_event import PondCompletionPolicy, PondEvent
from pond_helpers import plane

from lcp_builders import IDENT

REFUSE = (lcp.LifecycleCheckpointError, fc.FloatingCheckpointError, pc.PondCheckpointError, nfs.NativeFloatingMemoryError)
POL = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)


def _factory(ev0):
    def factory(vert, topology=None, schedule=None):
        return PondEvent(ev0.context, None, ev0.sediment, ev0.schedule if schedule is None else schedule, ev0._initial_topology if topology is None else topology, control=ev0.control, policy=ev0.policy,
                         splash=ev0.splash, floating_vertical=vert)
    return factory


def _boundary(owner, bed, ev0):
    factory = _factory(ev0)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    original = occ.event._initial_topology
    h = complete(occ.event, st)  # a genuine zero-rain quiet completion
    lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    return lin, factory, original


def deep_dense(n_voxels=5461):
    real = probe.build_minimal_state
    probe.build_minimal_state = lambda *a, **kw: real(*a, **{**kw, "n_voxels": n_voxels, "voxel_dz_m": 0.001})
    try:
        ev0, _s, bed, _c, _v, owner = fb.make(surface=plane(base=0.12, drop=0.004), rain_mm_h=0.0, end_s=1.0, max_dt_s=0.2, policy=POL, depth=0.0)  # genuine ZERO surface depth, Rain0: a storage fixture, no hydrology
    finally:
        probe.build_minimal_state = real
    assert int(bed.voxel_column.mass_kg.shape[2]) == n_voxels
    return _boundary(owner, bed, ev0)


def wide(n=32, n_voxels=60):
    real = probe.build_minimal_state
    probe.build_minimal_state = lambda *a, **kw: real(*a, **{**kw, "n_voxels": n_voxels, "voxel_dz_m": 0.01})
    try:
        ev0, _s, bed, _c, _v, owner = fb.make(surface=plane(ny=n, nx=n, base=0.12, drop=0.001), rain_mm_h=0.0, end_s=1.0, max_dt_s=0.2, policy=POL, depth=0.0)
    finally:
        probe.build_minimal_state = real
    assert bed.voxel_column.mass_kg.shape[:3] == (n, n, n_voxels) and ev0.nc == 6
    return _boundary(owner, bed, ev0)


def windowed_deep(tmp, n_voxels=5000):
    orig = fb._windowed_helpers
    made = []

    def larger():
        m = orig()
        made.append((m, m.NV, m.DZ))  # remember the ACTUAL old values of every module object this helper hands out
        m.NV, m.DZ = n_voxels, 0.3 / n_voxels
        return m

    fb._windowed_helpers = larger
    try:
        ev0, _s, _v, owner, bed, _c = fb.make_pair(tmp, surface=plane(base=0.12, drop=0.004), depth=0.0, backend="numpy", rain_mm_h=0.0, end_s=1.0, splash=False, max_dt_s=0.2, policy=POL)[1]
    finally:
        fb._windowed_helpers = orig
        for m, nv, dz in made:
            m.NV, m.DZ = nv, dz  # restore: no module-state pollution of later fixtures / static recipes
    assert int(bed.voxel_column.n_voxels) == n_voxels
    return _boundary(owner, bed, ev0)


def exercise(tmp, lin, factory, original, *, windowed=False):
    saved = lcp.save_floating_lifecycle_checkpoint(tmp / "ck", lin, factory, IDENT, original_topology=original, available_bytes_fn=AVAIL)
    kw = {"original_topology": original, "available_bytes_fn": AVAIL}
    need = lcp.lifecycle_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=AVAIL)
    assert lcp.lifecycle_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=AVAIL, reserve_bytes=7) == need + 7  # the reserve is counted once
    tripped = []
    real_hash, real_load = pc._hash_file, np.load
    pc._hash_file = lambda *a, **k: (tripped.append("hash"), real_hash(*a, **k))[1]
    np.load = lambda *a, **k: (tripped.append("load"), real_load(*a, **k))[1]
    try:
        for bad in (0, need - 1):
            with pytest.raises(REFUSE):
                lcp.load_floating_lifecycle_checkpoint(saved.path, factory, IDENT, saved.manifest_sha256, original_topology=original, available_bytes_fn=lambda b=bad: b)
        with pytest.raises(REFUSE):
            lcp.load_floating_lifecycle_checkpoint(saved.path, factory, IDENT, saved.manifest_sha256, original_topology=original, available_bytes_fn=lambda: need + 6, reserve_bytes=7)
        assert tripped == []  # refused before any payload hash or array
        # a huge child header, GENUINELY resealed (child file record + outer manifest recomputed, so the size/hash/seal checks pass): the composed need from the declared dimensions is refused as a MEMORY
        # error before any payload hash or array. This is a declared-dimension model check, not an analytic proof for every conceivable header (the child re-verifies the real arrays after admission).
        d = tmp / "huge"
        shutil.copytree(saved.path, d)
        cm = json.loads((d / "child" / fc.MANIFEST_NAME).read_text())
        cm["binding"]["n_classes"] = 10**12
        (d / "child" / fc.MANIFEST_NAME).write_text(json.dumps(cm))
        om = json.loads((d / lcp.MANIFEST_NAME).read_text())
        rec = om["child"]["files"][fc.MANIFEST_NAME]
        rec["size"] = (d / "child" / fc.MANIFEST_NAME).stat().st_size
        rec["sha256"] = real_hash(d / "child" / fc.MANIFEST_NAME, 1 << 40)[0]
        om["child"]["manifest_sha256"] = rec["sha256"]
        sealed = fcc._seal(om)
        (d / lcp.MANIFEST_NAME).write_text(pc._json_dumps(sealed))
        assert lcp.lifecycle_load_need_bytes(d, IDENT, sealed["manifest_sha256"], available_bytes_fn=lambda: 1 << 40) > 1 << 40  # the declared dimensions publish an impossible need
        with pytest.raises(nfs.NativeFloatingMemoryError):
            lcp.load_floating_lifecycle_checkpoint(d, factory, IDENT, sealed["manifest_sha256"], original_topology=original, available_bytes_fn=lambda: 1 << 40)
        assert tripped == []
    finally:
        pc._hash_file, np.load = real_hash, real_load
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    r = lcp.load_floating_lifecycle_checkpoint(saved.path, factory, IDENT, saved.manifest_sha256, available_bytes_fn=lambda: need, **{k: v for k, v in kw.items() if k != "available_bytes_fn"})  # EXACT published need
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert r.kind == "boundary" and peak <= need, (peak, need)
    with pytest.raises(lcp.LifecycleCheckpointError, match="already exists"):
        lcp.save_floating_lifecycle_checkpoint(tmp / "ck", lin, factory, IDENT, original_topology=original, available_bytes_fn=AVAIL)  # never overwrites
    return need, saved


def test_deep_dense_v5461_exact_need_loads_and_need_minus_one_refuses(tmp_path):
    lin, factory, original = deep_dense(5461)
    need, saved = exercise(tmp_path, lin, factory, original)
    cman = json.loads((saved.path / "child" / fc.MANIFEST_NAME).read_text())
    assert cman["native_dense"]["voxel_shape"][2] == 5461 and need > sum(f["size"] for f in json.loads((saved.path / lcp.MANIFEST_NAME).read_text())["child"]["files"].values()) // 8


def test_wide_32x32_v60_nb32_exact_need_loads(tmp_path):
    lin, factory, original = wide(32, 60)
    exercise(tmp_path, lin, factory, original)


def test_native_windowed_deep_v5000_need_holds_the_real_mapping_bound(tmp_path):
    lin, factory, original = windowed_deep(tmp_path / "w", 5000)
    bound = int(lin.head.bed.voxel_column.deep_store.mapped_bytes_bound)
    assert bound > 7_372_800  # larger V: a different, larger REAL native bound than V=300
    need, _saved = exercise(tmp_path, lin, factory, original, windowed=True)
    assert need >= bound  # the actual resident-cache bound, not a compressed-file size


def test_the_windowed_fixture_leaves_no_helper_module_state_behind(tmp_path):
    import static_case as sc

    before = (fb._windowed_helpers().NV, fb._windowed_helpers().DZ, sc.NV, sc.DZ)
    windowed_deep(tmp_path / "w", 1200)
    assert (fb._windowed_helpers().NV, fb._windowed_helpers().DZ, sc.NV, sc.DZ) == before == (300, 0.001, 300, 0.001)
