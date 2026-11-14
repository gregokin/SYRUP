"""SiteControls: explicit values only, native validation, sealed bytes, caller arrays preserved, admission need-1/exact, sealed type. No LEAF needed."""
from __future__ import annotations

import copy
import dataclasses
import math
import pickle

import numpy as np
import pytest
from veg_binding_support import A1, A2, INF

from maple_syrup import water_vegetation_binding as wvb

BIG = 1 << 30
ERR = wvb.VegetationBindingError


def ctl(**kw):
    base = {"a1": A1, "a2": A2, "a3": 0.4, "max_fraction_per_hour": 0.7, "max_intake_rate_m_s": 1e-5, "background_receiver_weights": [0.0, 1.0, 1.0], "n_layers": 3, "n_cells": 4, "max_bytes": BIG}
    base.update(kw)
    return wvb.site_controls(**base)


def test_scalars_and_raw_background_are_kept_raw_and_sealed():
    c = ctl(background_receiver_weights=[0.0, 3.0, 1.0])
    assert (c.a1, c.a2, c.a3, c.max_fraction_per_hour, c.max_intake_rate_m_s) == (A1, A2, 0.4, 0.7, 1e-5)
    assert c.a3_layout == c.max_rate_layout == "scalar"
    assert c.background_receiver_weights.tolist() == [0.0, 3.0, 1.0]  # raw: not normalised here (SYRUP's native record normalises)
    assert not c.background_receiver_weights.flags.writeable and c.background_receiver_weights.base is not None


def test_none_background_and_infinite_max_rate_are_legal():
    c = ctl(background_receiver_weights=None, max_intake_rate_m_s=INF)
    assert c.background_receiver_weights is None and math.isinf(c.max_intake_rate_m_s)


def test_maps_copied_not_aliased_and_caller_arrays_untouched():
    a3 = np.array([0.1, 0.2, 0.3, 0.4])
    mr = np.array([1e-6, 2e-6, INF, 4e-6])
    bg = np.array([0.0, 1.0, 2.0])
    keep = (a3.copy(), mr.copy(), bg.copy())
    c = ctl(a3=a3, max_intake_rate_m_s=mr, background_receiver_weights=bg)
    assert c.a3_layout == c.max_rate_layout == "cell" and c.n_cells == 4
    a3[0], mr[0], bg[0] = 0.9, 9.0, 9.0  # the caller edits afterwards: the sealed record must not follow
    assert c.a3[0] == 0.1 and c.max_intake_rate_m_s[0] == 1e-6 and c.background_receiver_weights[0] == 0.0
    assert np.array_equal(keep[0][1:], a3[1:]) and not c.a3.flags.writeable and not np.shares_memory(c.a3, a3)
    c2 = ctl(a3=keep[0], max_intake_rate_m_s=keep[1], background_receiver_weights=keep[2])
    assert c2.sha256 == c.sha256 != ctl().sha256  # content identity follows the bytes, not the object


@pytest.mark.parametrize("kw,why", [
    ({"a1": -0.01}, "a1 negative"), ({"a2": 0.0}, "a2 zero"), ({"a1": 0.5, "a2": 0.1}, "a1 > a2: g could exceed 1"), ({"a1": float("nan")}, "a1 nan"), ({"a1": True}, "a1 bool"),
    ({"a3": 1.0001}, "a3 > 1"), ({"a3": -0.1}, "a3 < 0"), ({"a3": float("nan")}, "a3 nan"), ({"a3": INF}, "a3 inf"), ({"a3": np.array([0.1, 0.2, 0.3])}, "a3 map wrong length"),
    ({"a3": np.array([0.1, 0.2, 0.3, 1.5])}, "a3 map out of range"), ({"a3": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)}, "a3 map dtype"),
    ({"a3": np.zeros((2, 2))}, "a3 2-D"), ({"max_fraction_per_hour": 1.1}, "max_fraction > 1"), ({"max_fraction_per_hour": np.array([0.5])}, "max_fraction array"),
    ({"max_intake_rate_m_s": -1.0}, "max rate negative"), ({"max_intake_rate_m_s": float("nan")}, "max rate nan"), ({"max_intake_rate_m_s": -INF}, "max rate -inf"),
    ({"max_intake_rate_m_s": np.array([1.0, float("nan"), 1.0, 1.0])}, "max rate map nan"), ({"background_receiver_weights": [0.0, 1.0]}, "weights wrong length"),
    ({"background_receiver_weights": [0.0, -1.0, 1.0]}, "negative weight"), ({"background_receiver_weights": [0.0, float("inf"), 1.0]}, "infinite weight"),
    ({"background_receiver_weights": 1.0}, "scalar weights"), ({"n_layers": 0}, "no layers"), ({"n_cells": 0}, "no cells"),
])
def test_invalid_controls_refused(kw, why):
    with pytest.raises(ERR):
        ctl(**kw)


def test_map_without_n_cells_refused():
    with pytest.raises(ERR):
        ctl(a3=np.array([0.1, 0.2]), n_cells=None)


def test_no_defaults_every_control_is_required():
    full = {"a1": A1, "a2": A2, "a3": 0.4, "max_fraction_per_hour": 0.7, "max_intake_rate_m_s": 1e-5, "background_receiver_weights": None, "n_layers": 3, "max_bytes": BIG}
    for k in ("a1", "a2", "a3", "max_fraction_per_hour", "max_intake_rate_m_s", "background_receiver_weights", "n_layers", "max_bytes"):
        kw = {x: v for x, v in full.items() if x != k}
        with pytest.raises(TypeError):
            wvb.site_controls(**kw)


def test_sealed_type_no_direct_construction_copy_pickle_or_forgery():
    c = ctl()
    with pytest.raises(ERR):
        wvb.SiteControls(1.0, 1.0, 0.1, "scalar", 0.5, 1.0, "scalar", None, 3, None, "x", 0)
    for f in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(ERR):
            f(c)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.a1 = 0.0  # frozen
    object.__setattr__(c, "a1", 0.02)  # a forged in-place alteration must be detected by every consumer
    with pytest.raises(ERR):
        wvb._check_controls(c)


def test_admission_need_minus_one_refuses_exact_admits_and_reader_and_reserve():
    kw = {"a3": np.array([0.1] * 4), "max_intake_rate_m_s": np.array([1e-5] * 4)}
    need = wvb.estimate_controls_bytes(3, a3_cells=4, max_rate_cells=4)["peak_bytes"]
    with pytest.raises(wvb.VegetationBindingMemoryError):
        ctl(max_bytes=need - 1, **kw)
    assert ctl(max_bytes=need, **kw).a3_layout == "cell"
    with pytest.raises(wvb.VegetationBindingMemoryError):
        ctl(max_bytes=need + 10, reserve_bytes=11, **kw)  # the caller reserve is counted
    with pytest.raises(wvb.VegetationBindingMemoryError):
        ctl(max_bytes=BIG, available_bytes_fn=lambda: need - 1, **kw)
    assert ctl(max_bytes=BIG, available_bytes_fn=lambda: need, **kw)
    with pytest.raises(ERR):
        ctl(max_bytes=BIG, available_bytes_fn=lambda: True, **kw)  # a Boolean reading is not a size


def test_refusal_allocates_nothing_before_admission(monkeypatch):
    calls = []
    monkeypatch.setattr(wvb, "_seal", lambda a: calls.append(1) or a)
    with pytest.raises(wvb.VegetationBindingMemoryError):
        ctl(a3=np.array([0.1] * 4), max_bytes=1)
    assert calls == []


def test_native_vegetation_function_is_the_validator_for_a1_a2():
    from maple_syrup import matrix_infiltration as mi
    with pytest.raises(mi.MatrixError):
        mi.reference_vegetation_function(0.0, -1.0, 1.0)  # the native refusal the controls rely on
    with pytest.raises(ERR):
        ctl(a1=-1.0)
