"""S2-B2.b correction01: the raw-weight part of the operator identity is streamed in small fixed chunks (O(1) scratch independent of L) with EXACTLY the former canonical bytes. Author-written, CPU only."""
from __future__ import annotations

import gc
import hashlib
import tracemalloc

import numpy as np
import pytest
from conftest import BIG, full, hand_column, hand_op, hand_state, sp, spf, step


def original_identity(op):
    """The pre-correction `_identity` (whole `raw.tobytes().hex()` f-string) transcribed verbatim, as the continuity reference."""
    from maple_syrup import spatial_matrix_infiltration as sm

    h = hashlib.sha256()
    for part in (spf.OPERATOR_TAG, op.column.sha256, op.column.backend, str((op.column.n_layers, op.column.n_cells)), repr(float(op.max_fraction_per_hour)), repr(sorted(dict(op.layouts).items()))):
        h.update(part.encode())
        h.update(b"\0")
    for label, rec in (("background", op.background), *((f"species{s}", w) for s, w in enumerate(op.species))):
        h.update(f"{label}|{'none' if rec is None else rec.raw.tobytes().hex()}".encode())
        h.update(b"\0")
    for name, a in zip(("rate_s", "max_rate_m_s", "receiver"), op._law, strict=True):
        sm._hash_array(h, name, "map", a)
    return h.hexdigest()


def ops(backend):
    col3, col5 = hand_column(3, 4, backend), hand_column(5, 4, backend)
    c = [np.array([0.0, 0.5, 1.0, 2.0]), np.array([1.0, 0.0, 3.0, 0.1]), np.array([0.2, 0.2, 0.2, 0.2])]
    yield hand_op(col3, a3=0.4)  # background only
    yield hand_op(col3, a3=0.4, background=None, species=[[0, 1, 1]], contributions=c[:1])  # background absent ('none')
    yield hand_op(col3, a3=0.4, background=[0, 0, 0], species=[[0, 0, 1], [0, 0, 0]], contributions=c[:2])  # zeros / disabled species
    yield hand_op(col3, a3=0.6, background=[0.0, 7.5, 7.5], species=[[0.3, 0.2, 0.9], [1, 0, 0], [0, 0.4, 0]], contributions=c)  # unnormalised, multiple species
    yield hand_op(col5, a3=0.35, background=[1, 2, 3, 4, 5], species=[[5, 4, 3, 2, 1]], contributions=c[:1])


@pytest.mark.parametrize("chunk", [1, 3, 7, 8, 9, 64, 4096, 10**6])
def test_streamed_digest_equals_the_original_format_for_every_chunk_boundary(backend, monkeypatch, chunk):
    monkeypatch.setattr(spf, "WEIGHT_HASH_CHUNK_BYTES", chunk)
    for op in ops(backend):
        assert op.sha256 == original_identity(op)  # the identity minted at preparation (chunked) equals the former whole-string digest
        assert spf.verify_operator_identity(op) is True  # cold verification uses the same bounded path
        assert spf._identity(op.column, op._law, dict(op.layouts), op.max_fraction_per_hour, op.background, op.species) == original_identity(op)


def test_identity_is_unchanged_from_the_recorded_pre_correction_digests(backend):
    """Digests recorded BEFORE the correction for Root's public column (N = 1, depths linspace 0..1, backgrounds of ones)."""
    recorded = {1024: "e741d6c216a7b0da29c37c68fbc8f1732945a856773b7d5190990ed7cb92ebe1", 4096: "5467ad79d467ab00e114f9133ce1ab15702fd896f9a569e727c4cc390f06fefc",
                8192: "e22914bbbc57f2059a65ba286e9e1a7df5557554b3ea2a07a37981439f4e1669"}
    for L, sha in recorded.items():
        col = _root_column(L, backend)
        op = spf.prepare_spatial_preferential(col, a3=0.35, max_fraction_per_hour=1.0, max_intake_rate_m_s=2e-6, background_receiver_weights=np.ones(L), memory_budget_bytes=BIG)
        if backend == "numpy":
            assert op.sha256 == sha  # the numba backend tag differs by design
        assert op.sha256 == original_identity(op)


def test_the_hash_update_size_is_bounded_and_independent_of_L():
    class Rec:
        def __init__(self, raw):
            self.raw = raw

    class Spy:
        def __init__(self):
            self.sizes, self.data = [], bytearray()

        def update(self, b):
            self.sizes.append(len(b))
            self.data += bytes(b)

    for L in (4, 1000, 100_000):
        raw = np.linspace(0.0, 3.0, L)
        spy = Spy()
        spf._hash_weights(spy, "species0", Rec(raw))
        assert bytes(spy.data) == b"species0|" + raw.tobytes().hex().encode() + b"\0"  # the canonical bytes
        assert max(spy.sizes) <= 2 * spf.WEIGHT_HASH_CHUNK_BYTES  # one small hex buffer at a time
    none = Spy()
    spf._hash_weights(none, "background", None)
    assert bytes(none.data) == b"background|none\0"


def _root_column(L, backend="numpy"):
    spec = sp.spatial_layer_spec(np.linspace(0.0, 1.0, L + 1), theta_residual=0.03125, theta_field_capacity=0.25, theta_saturation=0.5, theta_evaporation_stress=0.375, diffusivity_m2_s=0.0, n_cells=1)
    return sp.prepare_spatial_column(spec, np.array([True]), 2.5, backend=backend, memory_budget_bytes=1 << 34)


@pytest.mark.parametrize("L", [1024, 4096, 8192])
def test_public_preparation_at_the_exact_quota_stays_within_the_model(L):
    """Root's reproducer: N = 1, large L, NumPy path (no JIT: tracemalloc sees every NumPy/Python buffer of the build), at EXACTLY the estimated prepare requirement."""
    col = _root_column(L)
    bg = np.ones(L)
    before = bg.copy()
    quota = spf.estimate_spatial_preferential_memory(L, 1, backend="numpy", operation="prepare", n_species=0).required_bytes
    gc.collect()
    tracemalloc.start()
    try:
        op = spf.prepare_spatial_preferential(col, a3=0.35, max_fraction_per_hour=1.0, max_intake_rate_m_s=2e-6, background_receiver_weights=bg, memory_budget_bytes=quota)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak <= quota, (L, peak, quota)  # traced payload within the unchanged model and the unchanged quota
    np.testing.assert_array_equal(bg, before)  # no input mutation
    assert spf.verify_operator_identity(op) is True


def test_the_cold_verification_and_not_the_step_run_the_weight_hash(monkeypatch, backend):
    col = hand_column(3, 4, backend)
    op = hand_op(col, a3=0.5, background=[1, 2, 3])
    calls = []
    real = spf._hash_weights
    monkeypatch.setattr(spf, "_hash_weights", lambda *a, **k: calls.append(1) or real(*a, **k))
    step(op, full(col, 0.01), hand_state(col), 60.0)
    assert calls == []  # steps never hash
    assert spf.verify_operator_identity(op) is True and calls == [1]  # one background record, hashed once on the explicit cold path
