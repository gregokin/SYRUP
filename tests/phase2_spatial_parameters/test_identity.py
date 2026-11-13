"""Deterministic, chunk-size invariant identity of the explicit INPUTS; layout/mask/option sensitivity; distinct schema from SoilLayers. Author-written."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import make_spec, params, prepare

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss


def P(seed=1, L=3, N=13, mode="constant"):
    return params(np.random.default_rng(seed), L, N, mode=mode)


def test_identity_is_deterministic_and_independent_of_chunking(cpu_backend):
    p = P()
    ref = prepare(make_spec(p), backend=cpu_backend)
    for kw in ({}, {"chunk_columns": 1}, {"chunk_columns": 5}, {"hash_chunk_bytes": 8}, {"hash_chunk_bytes": 24}, {"chunk_columns": 3, "hash_chunk_bytes": 40}):
        other = prepare(make_spec(p), backend=cpu_backend, **kw)
        assert (other.parameters_sha256, other.mask_sha256, other.sha256) == (ref.parameters_sha256, ref.mask_sha256, ref.sha256), kw


def test_numpy_and_numba_backends_share_the_identity():
    pytest.importorskip("numba")
    p = P()
    assert prepare(make_spec(p), backend="numpy").sha256 == prepare(make_spec(p), backend="numba").sha256


@pytest.mark.parametrize("key", ["edges", "res", "fc", "sat", "iface"])
def test_any_input_value_changes_the_identity(cpu_backend, key):
    p = P()
    a = prepare(make_spec(p), backend=cpu_backend)
    q = {k: v.copy() for k, v in p.items()}
    flat = q[key].reshape(-1)
    flat[len(flat) // 2] = np.nextafter(flat[len(flat) // 2], np.inf) if key != "edges" else flat[len(flat) // 2] + 1e-9
    b = prepare(make_spec(q), backend=cpu_backend)
    assert a.parameters_sha256 != b.parameters_sha256 and a.mask_sha256 == b.mask_sha256


def test_layout_changes_the_identity_even_with_equal_values(cpu_backend):
    L, N = 3, 8
    shared = np.array([0.02, 0.03, 0.04])
    cell = np.repeat(shared[:, None], N, axis=1)
    base = P(L=L, N=N)
    a = prepare(make_spec(base, res=shared), backend=cpu_backend)
    b = prepare(make_spec(base, res=cell), backend=cpu_backend)
    c = prepare(make_spec(base, res=0.02), backend=cpu_backend)
    assert len({a.parameters_sha256, b.parameters_sha256, c.parameters_sha256}) == 3
    sc = prepare(make_spec(base, res=np.full(L, 0.02)), backend=cpu_backend)
    assert sc.parameters_sha256 != c.parameters_sha256  # scalar vs shared array of the same value


def test_mask_options_and_modes_change_the_right_parts(cpu_backend):
    p = P()
    base = prepare(make_spec(p), backend=cpu_backend)
    m = np.ones(13, dtype=bool)
    m[2] = False
    masked = prepare(make_spec(p), backend=cpu_backend, mask=m)
    assert masked.mask_sha256 != base.mask_sha256 and masked.parameters_sha256 == base.parameters_sha256 and masked.sha256 != base.sha256
    for kw in (dict(stress=0.51), dict(evaporation_factor=1.1), dict(bottom_drainage="closed"), dict(gravity_drainage=False)):
        spec_kw = {"stress": kw.get("stress", 0.5)}
        extra = {k: v for k, v in kw.items() if k != "stress"}
        other = prepare(make_spec(p, **spec_kw, **extra), backend=cpu_backend)
        assert other.parameters_sha256 != base.parameters_sha256, kw
    assert prepare(make_spec(p), backend=cpu_backend, area=9.0).sha256 != base.sha256 and prepare(make_spec(p), backend=cpu_backend, area=9.0).parameters_sha256 == base.parameters_sha256
    d = P(mode="tietjen_donor_depth")
    donor = prepare(make_spec(d, mode="tietjen_donor_depth"), backend=cpu_backend)
    assert donor.parameters_sha256 != base.parameters_sha256
    assert prepare(make_spec(d, mode="tietjen_donor_depth", d=0.31), backend=cpu_backend).parameters_sha256 != donor.parameters_sha256


def test_schema_is_distinct_from_the_common_layer_identity_and_module_hash_is_recorded(cpu_backend):
    lay = ss.soil_layers(np.array([0.0, 0.1, 0.3]), theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.4, diffusivity_m2_s=1e-7, theta_evaporation_stress=0.5)
    col = prepare(make_spec(P(L=2, N=3)), backend=cpu_backend)
    assert col.parameters_sha256 != lay.identity()["sha256"] and sp.SCHEMA.startswith("maple_syrup.spatial_soil_layers/")
    s = col.summary()
    assert s["schema"] == sp.SCHEMA and len(s["module_sha256"]) == 64 and s["sha256"] == col.sha256


def test_equal_inputs_in_different_array_objects_give_equal_identity(cpu_backend):
    # the identity is a function of the explicit input VALUES, layouts and options only (never of object identity, derived maps or memory addresses)
    p = P()
    q = {k: v.copy() for k, v in p.items()}
    q["sat"] = q["sat"] + 0.0
    assert prepare(make_spec(p), backend=cpu_backend).parameters_sha256 == prepare(make_spec(q), backend=cpu_backend).parameters_sha256
