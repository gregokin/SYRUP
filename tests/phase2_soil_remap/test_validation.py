"""Metadata and value refusals; nothing is published; inputs unchanged. Author-written."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import BIG, arrays, digest, random_columns

from maple_syrup import soil_water_remap as sr


def base(L=2, C=3, seed=1):
    return list(arrays(*random_columns(np.random.default_rng(seed), L, C)))


def call(args, backend, **over):
    eo, en, W, up, lo = args
    kw = dict(upper_theta=up, lower_theta=lo, cell_area_m2=1.0, backend=backend, memory_budget_bytes=BIG)
    kw.update(over)
    return sr.remap_intervals(eo, en, W, **kw)


@pytest.mark.parametrize("mutate", [
    lambda a: a.__setitem__(0, a[0].astype(np.float32)),
    lambda a: a.__setitem__(2, a[2].astype(np.int64)),
    lambda a: a.__setitem__(0, np.asfortranarray(a[0])),
    lambda a: a.__setitem__(1, a[1][:-1]),  # wrong number of edges
    lambda a: a.__setitem__(2, a[2][:, :-1]),  # wrong C
    lambda a: a.__setitem__(3, a[3][:-1]),
    lambda a: a.__setitem__(4, a[4].reshape(-1, 1)),
    lambda a: a.__setitem__(2, np.ma.masked_array(a[2])),
    lambda a: a.__setitem__(0, a[0].tolist()),
    lambda a: a.__setitem__(2, np.zeros((0, 3))),  # L = 0
], ids=["f32", "int", "fortran", "short_edges", "short_water_cols", "short_theta", "theta2d", "masked", "list", "zero_layers"])
def test_metadata_refusals(cpu_backend, mutate):
    a = base()
    mutate(a)
    with pytest.raises(sr.SoilRemapError):
        call(a, cpu_backend)


def test_non_contiguous_slice_refused(cpu_backend):
    a = base(C=4)
    big = np.zeros((a[2].shape[0], 8))
    big[:, ::2] = a[2]
    a[2] = big[:, ::2]
    assert not a[2].flags.c_contiguous
    with pytest.raises(sr.SoilRemapError):
        call(a, cpu_backend)


@pytest.mark.parametrize("which,bit", [(0, sr.FLAG_OLD_EDGES), (1, sr.FLAG_NEW_EDGES)])
@pytest.mark.parametrize("kind", ["nan", "inf", "collapsed", "decreasing", "overflow_width"])
def test_edge_value_refusals(cpu_backend, which, bit, kind):
    a = base(L=3, C=5)
    arr = a[which]
    if kind == "nan":
        arr[1, 2] = np.nan
    elif kind == "inf":
        arr[3, 2] = np.inf
    elif kind == "collapsed":
        arr[2, 2] = arr[1, 2]
    elif kind == "decreasing":
        arr[2, 2] = arr[1, 2] - 0.1
    else:
        arr[0, 2], arr[1, 2] = -1.7e308, 1.7e308  # the width is not representable
    before = digest(*a)
    with pytest.raises(sr.SoilRemapValidationError) as info:
        call(a, cpu_backend)
    assert info.value.bit == bit and info.value.first_column == 2 and info.value.n_flagged == 1
    assert digest(*a) == before


@pytest.mark.parametrize("kind", ["negative", "above_width", "nan", "inf"])
def test_water_value_refusals(cpu_backend, kind):
    a = base(L=2, C=4)
    w = a[0][1, 1] - a[0][0, 1]
    a[2][0, 1] = {"negative": -1e-9, "above_width": w * (1 + 1e-9), "nan": np.nan, "inf": np.inf}[kind]
    with pytest.raises(sr.SoilRemapValidationError) as info:
        call(a, cpu_backend)
    assert info.value.bit == sr.FLAG_WATER and info.value.first_column == 1


@pytest.mark.parametrize("which", [3, 4])
@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf])
def test_theta_value_refusals(cpu_backend, which, value):
    a = base()
    a[which][2] = value
    with pytest.raises(sr.SoilRemapValidationError) as info:
        call(a, cpu_backend)
    assert info.value.bit == sr.FLAG_THETA and info.value.first_column == 2


@pytest.mark.parametrize("area", [0.0, -1.0, np.nan, np.inf, True, "1", None, [1.0]])
def test_area_refusals(cpu_backend, area):
    with pytest.raises(sr.SoilRemapError):
        call(base(), cpu_backend, cell_area_m2=area)


@pytest.mark.parametrize("kw", [{"backend": "cupy"}, {"backend": 1}, {"memory_budget_bytes": -1}, {"memory_budget_bytes": 1.5}, {"memory_budget_bytes": True}, {"reserve_bytes": -1},
                                {"reserve_bytes": 2.0}, {"available_bytes_fn": 3}, {"available_bytes_fn": lambda: -1}, {"available_bytes_fn": lambda: 1.5}, {"available_bytes_fn": lambda: True}])
def test_option_refusals(cpu_backend, kw):
    kw = dict(kw)
    kw.setdefault("backend", cpu_backend)
    with pytest.raises(sr.SoilRemapError):
        call(base(), kw.pop("backend"), **kw)


def test_cuda_backend_without_cupy_arrays_is_refused_or_unavailable():
    # on a CPU lane the backend is either unavailable (typed, no fallback) or, if CuPy exists, refuses NumPy arrays (no implicit transfer)
    with pytest.raises((sr.SoilRemapBackendUnavailableError, sr.SoilRemapError)):
        call(base(), "cuda")
