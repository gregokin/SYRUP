"""H0 correction02: the admitted `PreferentialSpec` constructor model (raw payload + the declared per-array metadata allowance + allowance), against REAL tracemalloc peaks at S = 0/2/65/1000 (small N) and moderate N = 4096."""
# ruff: noqa: C408
from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest

from maple_syrup import floating_held_context as fh


def inputs(S, N, L=3):
    a3 = np.full(N, 2e-3)
    bg = np.array([0.0, 1.0, 1.0][:L])
    sw = tuple(bg.copy() for _ in range(S))
    sc = tuple(np.full(N, 0.5 / max(S, 1)) for _ in range(S)) if S else None
    return dict(a3=a3, max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=bg, species_receiver_weights=sw, species_contributions=sc)


def need_of(kw):
    return fh.PreferentialSpec.construction_need(kw["a3"], kw["max_intake_rate_m_s"], kw["background_receiver_weights"], kw["species_receiver_weights"], kw["species_contributions"])


@pytest.mark.parametrize("S,N", [(0, 3), (2, 3), (65, 3), (1000, 3), (65, 4096), (1000, 4096)])
def test_the_modelled_need_bounds_the_real_traced_peak_of_the_construction(S, N):
    kw = inputs(S, N)
    need = need_of(kw)
    raw = sum(int(x.nbytes) for x in (kw["a3"], kw["background_receiver_weights"], *kw["species_receiver_weights"], *(kw["species_contributions"] or ())))
    n_arrays = 2 + 2 * S
    assert need == raw + n_arrays * fh.SPEC_ARRAY_OVERHEAD_BYTES + fh.ALLOWANCE_BYTES  # raw payload + per-array metadata + allowance (the formula of the docstring)
    gc.collect()
    tracemalloc.start()
    spec = fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need)  # the reader returns EXACTLY the declared need
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak <= need, (S, N, peak, need)  # the reproduced run: need 113 584 B vs peak 307 621 B (S1000 L3 N3)
    assert peak >= raw  # non-vacuous: the sealed copies are really made
    assert len(spec.species_receiver_weights) == S


@pytest.mark.parametrize("S", [0, 2, 65, 1000])
def test_memory_zero_and_need_minus_one_refuse_before_any_copy(S, monkeypatch):
    kw = inputs(S, 3)
    need = need_of(kw)
    calls = {"seal": 0, "array": 0, "stack": 0}
    for obj, name, key in ((fh, "_seal", "seal"), (np, "array", "array"), (np, "stack", "stack")):
        real = getattr(obj, name)
        monkeypatch.setattr(obj, name, lambda *a, _r=real, _k=key, **k: calls.__setitem__(_k, calls[_k] + 1) or _r(*a, **k))
    reads = []
    for avail in (0, need - 1):
        with pytest.raises(fh.HeldContextError, match="reported available"):
            fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda a=avail: reads.append(1) or a)
    assert calls == {"seal": 0, "array": 0, "stack": 0} and len(reads) == 2  # the reader spoke, nothing was copied
    fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need)  # exact need admits
    assert calls["seal"] > 0


def test_the_reserve_is_counted_once_and_is_strict():
    kw = inputs(5, 3)
    need = need_of(kw)
    with pytest.raises(fh.HeldContextError):
        fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need, reserve_bytes=1)
    fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need + 7, reserve_bytes=7)
    with pytest.raises(fh.HeldContextError):
        fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need + 6, reserve_bytes=7)
    for bad in (-1, True, 1.5, "0", None, np.float64(2.0)):
        with pytest.raises(fh.HeldContextError):
            fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need + 10**6, reserve_bytes=bad)


def test_unbounded_iterators_are_never_consumed_and_the_need_is_metadata_only(monkeypatch):
    kw = inputs(3, 3)
    consumed = []

    def endless():
        while True:
            consumed.append(1)
            yield np.ones(3)

    for field in ("species_receiver_weights", "species_contributions"):
        bad = dict(kw)
        bad[field] = endless()
        with pytest.raises(fh.HeldContextError, match="bounded sequence"):
            fh.PreferentialSpec.admitted(**bad, available_bytes_fn=lambda: 10**12)
    assert consumed == []
    copies = []
    real = np.array
    monkeypatch.setattr(np, "array", lambda *a, **k: copies.append(1) or real(*a, **k))
    need_of(kw)
    assert copies == []  # `construction_need` reads array metadata only: no copy, no list or tuple of parts


def test_the_spec_owns_independent_sealed_copies_of_unchanged_inputs():
    kw = inputs(4, 6)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else None) for k, v in kw.items()}
    spec = fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need_of(kw))
    assert np.array_equal(kw["a3"], before["a3"])  # the raw inputs are unchanged
    kw["a3"][0] = 9.0
    kw["species_contributions"][0][0] = 9.0
    assert spec.a3[0] == 2e-3 and spec.species_contributions[0][0] != 9.0  # independent copies
    assert spec.a3.flags.writeable is False and all(not c.flags.writeable for c in spec.species_contributions)
