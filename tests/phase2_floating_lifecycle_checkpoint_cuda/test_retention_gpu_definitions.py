"""Actual-GPU WEAK-REFERENCE RETENTION definitions of the CUDA lifecycle physical file (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi was touched; no device pass is claimed). Opt-in (SYRUP_RELEASE_GPU=1).

The original GPU checkpoint task required an actual weakref proof that the transient restore carrier is RELEASED while the default `RestoredLifecycle` (lineage, head, closed) and the current physical water stay alive. The accepted
GPU definitions assert only the minimal result fields and the CPU proof (`phase2_floating_lifecycle_checkpoint/test_controls.py::_spy_load`) does not establish CUDA retention. These four definitions (Dense and native Windowed x
Boundary and CLOSED-day file) reuse the ACTUAL GPU helpers of `test_gpu_definitions.py` (`sides`/`save`/`load`/`pack`/`day_from`, genuine matched pit fixture, genuine save/load) and spy the actual `lc.verified_restore`:
the wrapper records ONLY weak references (the returned token, the restored event, state and vertical, the state's two device phase arrays and its native water and sediment ledger objects); no strong spy array is kept.
With the result, head and closed state held alive and a `gc.collect()`, every transient reference must be dead, the original physical W/J/I/D/H/q/bed must be unchanged and valid, and the NEXT actual native day (closed
file) or native event (boundary file) must run from the restored head alone.

CuPy ndarray weakref support is UNVERIFIED here (only Root's hardware can tell): `weak_array` tries the array itself, then the array's backing allocation object (`data.mem`); if neither supports weak references the array is
tracked through its owner (the state, which is weak-referenced) and the proxy used is recorded in the printed evidence; no fake phase object is substituted. Nothing here claims large-N/peak-memory or performance."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import weakref
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gd = _load("gpu_lifecycle_defs_for_retention", "test_gpu_definitions.py")  # the ACTUAL accepted GPU helpers (and, through them, the matched pit fixtures)
bd, ld, lc, br = gd.bd, gd.ld, gd.lc, gd.br
BIG = gd.BIG
host = gd.host


@pytest.fixture(scope="module")
def cp():
    return gd._gpu()


def weak_array(a):
    """A weak handle on a (device) array: the array itself, else its backing allocation, else (recorded) its owner only."""
    for obj, tag in ((a, "array"), (getattr(getattr(a, "data", None), "mem", None), "allocation")):
        if obj is None:
            continue
        try:
            return weakref.ref(obj), tag
        except TypeError:
            continue
    return None, "owner-only"


def spy_load(monkeypatch, cp, g, path, sv, partition):
    """Load while recording WEAK references to the transient carrier. The wrapper keeps no strong reference (the token is returned to the loader and dropped by it)."""
    seen: dict = {}
    proxies: dict = {}
    real = lc.verified_restore

    def spy(event, state, seal):
        tok = real(event, state, seal)
        seen.update(token=weakref.ref(tok), event=weakref.ref(event), state=weakref.ref(state), vert=weakref.ref(event.vertical), water=weakref.ref(state.water), sediment=weakref.ref(state.sediment))
        for name in ("phase_fraction", "phase_position_m"):
            ref, tag = weak_array(getattr(state, name))
            proxies[name] = tag
            if ref is not None:
                seen[name] = ref
        return tok

    monkeypatch.setattr(lc, "verified_restore", spy)
    try:
        r = gd.load(cp, g, path, sv, partition=partition)
    finally:
        monkeypatch.undo()
    return r, seen, proxies


def original_physical(g, kind, closed):
    """Host bytes of the ORIGINAL current physical water/history/depth/surface/faces (before the save/load)."""
    if kind == "boundary":
        return gd.pack(g.b)
    surf = closed.surface
    head = g.lin.head
    depth, qx, qy = (head.surface_depth_m, head.qx_m2_s, head.qy_m2_s) if surf is None else surf
    return {"W": host(closed.water.water_m), "J": host(closed.history.wetting_m), "I": host(closed.history.cumulative_intake_m), "D": host(head.depth_m), "H": host(depth), "qx": host(qx), "qy": host(qy)}


def restored_physical(r, kind):
    if kind == "boundary":
        return gd.pack(r.head)
    c, head = r.closed, r.head
    surf = c.surface
    depth, qx, qy = (head.surface_depth_m, head.qx_m2_s, head.qy_m2_s) if surf is None else surf
    return {"W": host(c.water.water_m), "J": host(c.history.wetting_m), "I": host(c.history.cumulative_intake_m), "D": host(head.depth_m), "H": host(depth), "qx": host(qx), "qy": host(qy)}


@pytest.mark.parametrize("kind", ["boundary", "closed_day"])
@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_the_cuda_restore_releases_every_transient_carrier_object_while_the_result_head_closed_day_and_physical_water_stay_alive_and_work(cp, tmp_path, monkeypatch, windowed, kind):
    g, _c = gd.sides(tmp_path, windowed)
    closed = None
    if kind == "closed_day":
        _st, res = g.day(g.b, 0)
        _t, closed = g.finish(res)  # the genuine closed day of the head hold (T = 0 request: the closed-day identity is what is tested)
    before = original_physical(g, kind, closed)
    sv = gd.save(g, tmp_path / "ck", closed=closed)
    r, seen, proxies = spy_load(monkeypatch, cp, g, tmp_path / "ck", sv, bd.PART if kind == "closed_day" else None)
    assert {"token", "event", "state", "vert", "water", "sediment"} <= set(seen)  # the minimum weak handles always exist (a missing array handle is recorded through `proxies`)
    # the result (lineage, head, closed) is held HERE; the transient carrier must be garbage: token (consumed, one-shot), event, state, the transient vertical, both phase arrays, water and sediment ledgers
    gc.collect()
    cp.cuda.Device().synchronize()
    dead = {k: v() is None for k, v in seen.items()}
    assert all(dead.values()), (dead, proxies)
    assert set(gd.cpul.RestoredLifecycle.__dataclass_fields__) == {"lineage", "head", "closed", "kind", "manifest_sha256"} and not hasattr(r, "event") and not hasattr(r, "state") and not hasattr(r, "token")
    assert r.lineage.head is r.head and (r.closed is None) == (kind == "boundary")
    # the ORIGINAL physical W/J/I/D/H/q/bed bytes are unchanged and valid after the release: the restored result carries the identical values
    gd.same(before, restored_physical(r, kind))
    if kind == "boundary":
        gd.same(before, gd.pack(g.b))  # the saving lineage's own boundary was never touched
    # the NEXT actual native step runs from the restored head ALONE (the saving lineage and the loader's carrier play no part)
    if kind == "closed_day":
        assert (r.closed.execution_digest, r.closed.digest) == (closed.execution_digest, closed.digest)
        b2 = br.hand_back(r.lineage, r.closed, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
        _t2, c2 = gd.day_from(g, r.lineage, b2, 1)
        assert float(c2.t_s) > float(b2.t_s) and abs(b2.ledger["residual_m3"]) <= b2.ledger["residual_bound_m3"]
    else:
        occ, s0 = lc.begin_occurrence(r.lineage, r.head, g.factory, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
        h = ld.complete(occ, s0)  # the genuine bounded completion loop (no fake quiet)
        b2 = lc.close_occurrence(r.lineage, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
        assert b2.ledger["n_occurrences"] >= 2 and abs(b2.ledger["residual_m3"]) <= b2.ledger["residual_bound_m3"]
    gc.collect()
    print("ROOT_RETENTION_EVIDENCE", {"windowed": windowed, "kind": kind, "weak_handles": sorted(seen), "array_proxy": proxies})


def test_the_169_production_source_bytes_are_exactly_the_public_phase2_source_manifest():
    """HOST (no GPU, no skip): this tree's `src/maple_syrup` is EXACTLY the public actual-source manifest `docs/subsurface_hydrology/phase2_source_manifest.json` -- the SAME existing algorithm (`provenance.source_tree_digest`:
    sha256 over sorted relpath NUL file-sha256 LF, `__pycache__`/bytecode excluded, symlinks/special files refused), over EVERY non-bytecode file, so a stray `.orig`/cache file in a checkout fails here instead of being
    hidden by a `*.py` glob. Digest, file count, total bytes and every (path, sha256) pair must match."""
    from maple_syrup import provenance

    root = _HERE.parents[1]
    manifest = json.loads((root / "docs" / "subsurface_hydrology" / "phase2_source_manifest.json").read_text())
    assert manifest["schema"] == "syrup.phase2.source_manifest.v1" and manifest["scope"] == "src/maple_syrup"
    got = provenance.source_tree_digest(root / "src" / "maple_syrup")
    assert got.digest_sha256 == manifest["digest_sha256"] and got.file_count == manifest["file_count"] == len(manifest["files"]) == 169 and got.total_bytes == manifest["total_bytes"]
    assert [list(pair) for pair in got.files] == manifest["files"]
    src = root / "src" / "maple_syrup"
    want = {rel: h for rel, h in manifest["files"] if rel.endswith(".py") and "/" not in rel}
    have = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob("*.py")}
    assert have == want and len(want) == 169  # every manifest file is a top-level production module (no data/native extra)
