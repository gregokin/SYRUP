"""Correction02: D6 (the nested metadata walk must not materialise hostile containers before the quota) and the audit of the public capture / cached-binding stages for remaining unmodelled
identifier / metadata copies. REAL genuine capture / stage / prepare; LEAF = the final frozen H3 package; nothing faked."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import pytest
from cached_support import (
    PSI,
    K,
    controls,
    expected,
    fingerprint,
    live,
    loadable,
    make_column,
    prepare,
    stage,
    wh,
    wvb,
)
from test_cached_inputs_correction01 import _big_species_case, traced
from veg_binding_support import BIG

ERR = wvb.VegetationBindingError
MEM = wvb.VegetationBindingMemoryError


def _args(rec, p, b, mask):
    return {**expected(p, b, rec), "active_mask": mask}   # built OUTSIDE any traced region (leaf_pin hashes the LEAF files)


def _hostile_docs():
    deep = []
    node = deep
    for _ in range(60):
        nxt = []
        node.append(nxt)
        node = nxt
    cyc_d: dict = {}
    cyc_d["self"] = cyc_d
    cyc_l: list = []
    cyc_l.append(cyc_l)
    return {
        "list_1M": ["x"] * 1_000_000,                                                    # the Root probe: 8,006,571 B owned before the quota
        "list_100k": ["x"] * 100_000,                                                    # under the 1 MiB document limit: still must not be copied before the quota
        "list_of_empty_dicts": [{}] * 300_000,
        "wide_unknown_map": {f"k{i}": 1 for i in range(200_000)},
        "wide_map_of_lists": {f"k{i}": [] for i in range(100_000)},
        "nested_depth_60": deep,
        "cyclic_dict": cyc_d,
        "cyclic_list": cyc_l,
        "huge_string": "y" * (5 << 20),
        "huge_int": 10 ** 300_000,
        "tuple_value": (1, 2),
        "numpy_scalar": np.int64(5),
        "bytes_value": b"abc",
        "non_str_key": {1: 2},
        "nan_float": float("nan"),
        "set_value": {1, 2},
    }


@pytest.mark.parametrize("name", list(_hostile_docs()))
def test_d6_hostile_nested_metadata_is_refused_with_o1_owned_memory_before_the_quota(leaf, name):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    meta["identity"]["unexpected"] = _hostile_docs()[name]                                # created BEFORE the tracer
    args = _args(rec, p, b, m)
    snap = fingerprint({k: v for k, v in meta.items() if k != "identity"}, arrays)
    r, peak = traced(lambda: wh.stage_cached_inputs(meta, arrays, max_bytes=1, **args))
    assert r[0] == "err" and isinstance(r[1], ERR), (name, r)
    assert peak < 64 * 1024, (name, peak)                                                 # the Root probe measured 8,006,571 B for the 1M list
    assert fingerprint({k: v for k, v in meta.items() if k != "identity"}, arrays) == snap  # the original inputs are unchanged
    if name in ("list_100k", "list_of_empty_dicts", "wide_unknown_map", "wide_map_of_lists"):
        assert isinstance(r[1], MEM) or "exceeds" in str(r[1])                            # either under the document limit (then the quota refuses) or over it (bounded rejection)


def test_d6_the_root_probe_exactly(leaf):
    """root_cached_inputs_c1_nested_metadata_probe.py: identity.unexpected_large_list = ['x'] * 1_000_000, max_bytes=1."""
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    meta["identity"]["unexpected_large_list"] = ["x"] * 1_000_000
    args = _args(rec, p, b, m)
    r, peak = traced(lambda: wh.stage_cached_inputs(meta, arrays, max_bytes=1, **args))
    assert r[0] == "err" and isinstance(r[1], ERR) and "1048576" in str(r[1]) and peak < 16 * 1024, (r, peak)


def test_d6_json_bound_is_a_stream_o_depth_and_linear_time(leaf):
    big = {"a": ["x"] * 1_000_000}
    r, peak = traced(lambda: wh._json_bound(big, 1 << 40))
    assert r[0] == "ok" and peak < 4096, peak                                             # a million elements walked with O(depth) memory
    t = []
    for n in (200_000, 800_000):
        doc = [1] * n
        t0 = time.perf_counter()
        wh._json_bound(doc, 1 << 40)
        t.append(time.perf_counter() - t0)
    assert t[1] / max(t[0], 1e-9) < 8.0, t                                                # 4x the elements, ~4x the time
    r, _ = traced(lambda: wh._json_bound(big, 1 << 20))
    assert r[0] == "err" and "exceeds" in str(r[1])
    start = time.perf_counter()
    with pytest.raises(ERR):
        wh._json_bound([["x"] * 1_000_000], 1 << 20)
    assert time.perf_counter() - start < 1.0                                              # stops at the limit, not after the whole list


def test_d6_the_bound_is_a_truthful_upper_bound_with_no_int_underestimate(leaf):
    import sys
    old_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)                                                          # (only so the REFERENCE json.dumps can render the huge ints)
    rng = np.random.default_rng(1)
    docs = [{"a": 10 ** 5000}, [-(10 ** 700), 0, 7, 2 ** 63], {"é": "ü☃\n\"\\", "k": [1.5, 1e300, True, None, False]}, {"s": "plain ascii"}, ["x" * 300, {"n": {"m": [[1, 2], [3]]}}],
            {"i": [int(v) for v in rng.integers(-10 ** 12, 10 ** 12, 50)]}]
    for d in docs:
        assert wh._json_bound(d, 1 << 30) >= len(json.dumps(d, sort_keys=True, separators=(",", ":")))
    assert wh._json_bound(10 ** 5000, 1 << 30) >= 5001                                   # the old flat 24 B per int under-estimated this by ~5 KB
    sys.set_int_max_str_digits(old_limit)


def test_d6_depth_bound_and_exact_types_and_frozen_mode(leaf):
    ok = {"a": {"b": {"c": [1, {"d": 2}]}}}
    assert wh._json_bound(ok, 1 << 20) > 0
    deep = ok
    for _ in range(wh.MAX_META_DEPTH):
        deep = {"x": deep}
    with pytest.raises(ERR, match="nested deeper"):
        wh._json_bound(deep, 1 << 30)
    import types
    frozen = types.MappingProxyType({"a": (1, np.int64(2), np.float64(3.0)), "b": types.MappingProxyType({"c": [True]})})
    assert wh._json_bound(frozen, 1 << 20, frozen=True) > 0
    for bad in (frozen, (1, 2), {"a": np.int64(1)}):
        with pytest.raises(ERR):
            wh._json_bound(bad, 1 << 20)
    with pytest.raises(ERR):
        wh._json_bound({"a": float("inf")}, 1 << 20, frozen=True)


def test_d6_valid_genuine_capture_stage_prepare_unchanged_and_ids_equal_the_c1_snapshot(leaf, tmp_path):
    """The genuine flow is unchanged, and the identities equal those computed by Root's frozen C1 snapshot in a fresh process on the pinned prefix (the walk does not touch any hash)."""
    import subprocess
    import sys
    from pathlib import Path

    from cached_support import LEAF_PACKAGE, MAPLE_PACKAGE, NATIVE, native_env
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    rec2 = stage(rec, p, b, meta, arrays, m)
    law = prepare(make_column(), wh.cached_binding(rec2, expected_binding_sha256=b.sha256, max_bytes=BIG))
    assert rec2.record_sha256 == rec.record_sha256 and rec2.binding_sha256 == b.sha256 and law.law_sha256 == prepare(make_column(), b).law_sha256
    c1 = Path(__file__).resolve().parents[3] / "root_cached_inputs_c1_snapshot"
    script = f"""
import json
from pathlib import Path
import maple
assert Path(maple.__file__).resolve().is_relative_to(Path({str(NATIVE)!r}) / "source" / "src"), maple.__file__
from maple_syrup.provenance import source_tree_digest
assert source_tree_digest(Path(maple.__file__).resolve().parent).digest_sha256 == {MAPLE_PACKAGE!r}
import maple_syrup
assert Path(maple_syrup.__file__).resolve().is_relative_to(Path({str(c1 / "src")!r})), maple_syrup.__file__
import veg_binding_support
import leaf
assert source_tree_digest(Path(leaf.__file__).resolve().parent).digest_sha256 == {LEAF_PACKAGE!r}
from leaf import state, traits, water_vegetation
import cached_support as cs
lf = {{"state": state, "traits": traits, "wv": water_vegetation}}
p, b, surf, mask = cs.live(lf)
r = cs.wh.capture_cached_inputs(b, max_bytes=cs.BIG)
law = cs.prepare(cs.make_column(), cs.wh.cached_binding(r, expected_binding_sha256=b.sha256, max_bytes=cs.BIG))
print(json.dumps({{"binding": b.sha256, "record": r.record_sha256, "content": r.content_sha256, "law": law.law_sha256, "matrix": law.matrix_sha256, "pref": law.preferential_sha256}}))
"""
    env = native_env()
    env["PYTHONPATH"] = env["PYTHONPATH"].replace(str(Path(__file__).resolve().parents[2] / "src"), str(c1 / "src"))
    out = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, env=env, check=False)
    assert out.returncode == 0, out.stderr[-2500:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    cb = prepare(make_column(), wh.cached_binding(rec2, expected_binding_sha256=b.sha256, max_bytes=BIG))
    assert got == {"binding": b.sha256, "record": rec.record_sha256, "content": rec.content_sha256, "law": cb.law_sha256, "matrix": cb.matrix_sha256, "pref": cb.preferential_sha256}


# ---------------------------------------------------------------- audit of the public owning stages
def test_audit_capture_does_not_copy_the_species_ids_and_models_the_metadata_terms(leaf):
    p, b, _surf, _m = _big_species_case(leaf, S=400)
    est = wh.estimate_capture_bytes_from_binding(b)
    assert est["meta_json_bytes"] > 0 and est["peak_bytes"] == 3 * est["meta_json_bytes"] + wvb.PY_RECORD_ALLOWANCE_BYTES and est["borrowed_bytes"] > 0
    wh.capture_cached_inputs(b, max_bytes=est["peak_bytes"])                                # warm
    with pytest.raises(MEM):
        wh.capture_cached_inputs(b, max_bytes=est["peak_bytes"] - 1)
    r, peak = traced(lambda: wh.capture_cached_inputs(b, max_bytes=est["peak_bytes"]))      # the EXACT admitted capture
    assert r[0] == "ok" and peak <= est["peak_bytes"], (peak, est)
    rec = r[1]
    assert rec.species_ids is p.species_ids or tuple(rec.species_ids) == tuple(p.species_ids)
    assert "species_ids" not in json.loads(rec.identity_json)
    r1, peak1 = traced(lambda: wh.capture_cached_inputs(b, max_bytes=1))
    assert r1[0] == "err" and isinstance(r1[1], MEM) and peak1 < 32 * 1024


def test_audit_capture_peak_is_independent_of_the_species_count_beyond_the_modelled_terms(leaf):
    peaks = {}
    for S in (8, 400):
        _p, b, _surf, _m = _big_species_case(leaf, S=S)
        est = wh.estimate_capture_bytes_from_binding(b)
        wh.capture_cached_inputs(b, max_bytes=BIG)
        _r, peak = traced(lambda b=b: wh.capture_cached_inputs(b, max_bytes=BIG))
        peaks[S] = (peak, est["peak_bytes"])
        assert peak <= est["peak_bytes"]
    assert peaks[400][0] < peaks[8][0] + 4096                                               # 50x the species: no S-proportional copy remains in capture


def test_audit_cached_binding_admission_is_the_literal_parse_bytes(leaf):
    _p, b, _surf, _m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    need = wvb.PY_RECORD_ALLOWANCE_BYTES + 2 * (len(rec.source_json) + len(rec.package_json))
    with pytest.raises(MEM):
        wh.cached_binding(rec, expected_binding_sha256=b.sha256, max_bytes=need - 1)
    with pytest.raises(MEM):
        wh.cached_binding(rec, expected_binding_sha256=b.sha256, max_bytes=need + 2, reserve_bytes=3)
    cb = wh.cached_binding(rec, expected_binding_sha256=b.sha256, max_bytes=need)
    assert cb.sha256 == b.sha256
    r, peak = traced(lambda: wh.cached_binding(rec, expected_binding_sha256=b.sha256, max_bytes=need))
    assert r[0] == "ok" and peak <= need


def test_audit_remaining_public_stages_hold_no_s_proportional_copy(leaf):
    """verify / compact / fingerprint: O(1) in the species count (the fingerprint of the id tuple was an O(S) allocation before this correction); the accessor copies are documented."""
    res = {}
    for S in (8, 400):
        _p, b, _surf, _m = _big_species_case(leaf, S=S)
        rec = wh.capture_cached_inputs(b, max_bytes=BIG)
        wh.verify_cached_inputs(rec)
        wvb.verify_binding(b)
        res[S] = (traced(lambda rec=rec: wh.verify_cached_inputs(rec))[1], traced(lambda b=b: wvb.verify_binding(b))[1], traced(lambda rec=rec: wh.cached_compact(rec))[1])
    for k in range(3):
        assert abs(res[400][k] - res[8][k]) < 1024, res                                       # 50x the species, the same owned bytes
    r, peak = traced(lambda: wh.cached_compact(rec))
    assert r[1]["species_ids"] is rec.species_ids
    r, peak = traced(lambda: wh.cached_compact(rec, with_identity=True))
    assert peak >= 8 * 400                                                                  # the documented accessor copy (caller-owned) is the only S-proportional one
    assert rec.identity["species_ids"] == list(rec.species_ids)
    _ = (K, PSI, controls, np, copy)
