"""Phase 4S Task B correction: the context's launch-dependent SCALAR metadata is sealed at preparation.

`dataclasses.replace(ctx, n_cells=..., n_active=..., dx_m=..., model_code=..., mode=..., level_bounds=...)` keeps every
owned device array (so the pointer/shape/dtype fingerprints still match) while changing what reaches the unchecked
kernels. Every forgery must be refused BEFORE any kernel enqueue by every public entry that launches: the coupled step,
the column step and the storm accumulator. The tests intercept `hydrology_cuda._function`, so a refusal that came too
late would show as a recorded launch; nothing here can launch a kernel with forged metadata. Nothing here was run by its
author (file-only tools); Codex records results.
"""
from __future__ import annotations

import dataclasses

import hydro_cases as hcs
import pytest

pytest.importorskip("maple")

from maple_syrup import hydrology_cuda as hc
from maple_syrup.routing import RoutingError
from maple_syrup.storm import StormControl

pytestmark = pytest.mark.usefixtures("gpu")
CUDA = StormControl(implementation="cuda")


def cupy():
    import cupy as cp

    return cp


@pytest.fixture
def launches(monkeypatch):
    log: list = []

    # Trap the ENQUEUE (`_launch`), not the kernel lookup: preparation legitimately loads every kernel through
    # `_function` before any validation, so a `_function` trap would fail valid preparation. `_launch` is the only
    # path to `kernel(grid, block, args)`, so a refusal that came too late is recorded and fails the test, and no
    # raw launch with forged metadata can happen.
    def refuse(name, *args, **kwargs):
        log.append(name)
        raise AssertionError(f"kernel {name} was enqueued with a context that must have been refused")

    monkeypatch.setattr(hc, "_launch", refuse)
    return log


def _other_mode(mode):
    return "split" if mode == "fused" else "fused"


FORGERIES = {
    "n_cells_larger": lambda c: {"n_cells": c.n_cells * 4},
    "n_cells_smaller": lambda c: {"n_cells": c.n_cells - 1},
    "n_active_larger": lambda c: {"n_active": c.n_active + 1000},
    "n_active_smaller": lambda c: {"n_active": c.n_active - 1},
    "dx_scaled": lambda c: {"dx_m": c.dx_m * 2.0},
    "dx_nan": lambda c: {"dx_m": float("nan")},
    "model_code_flipped": lambda c: {"model_code": 1 - c.model_code},
    "model_name_changed": lambda c: {"model": "pavement_hawkins" if c.model == "fixed_ksat" else "fixed_ksat"},
    "mode_changed": lambda c: {"mode": _other_mode(c.mode)},
    "mode_invalid": lambda c: {"mode": "speculative"},
    "requested_mode_changed": lambda c: {"requested_mode": _other_mode(c.mode)},
    "level_bounds_end_extended": lambda c: {"level_bounds": (*c.level_bounds[:-1], c.level_bounds[-1] + 7)},
    "level_bounds_dropped_level": lambda c: {"level_bounds": c.level_bounds[:1] + c.level_bounds[2:]},
    "level_bounds_as_list": lambda c: {"level_bounds": list(c.level_bounds)},
    "level_bounds_none": lambda c: {"level_bounds": None},
    "max_level_width_huge": lambda c: {"max_level_width": 10**9},
    "shape_changed": lambda c: {"shape": (c.shape[0] + 1, c.shape[1])},
    "shape_transposed": lambda c: {"shape": (c.shape[1], c.shape[0])},
    "shape_none": lambda c: {"shape": None},
    "shape_as_list": lambda c: {"shape": list(c.shape)},  # equals the tuple but is mutable: refused
}


def _entries(ctx, dev, cp):
    """Every public entry that would launch a kernel with this context."""
    h, s = dev.state.depth_m, dev.state.soil_water_m
    grids = [cp.zeros(dev.graph.shape) for _ in range(6)]
    return {
        "coupled_step": lambda forged: hc.prepared_coupled_step(forged, dev.rate_on, dev.state, 1.0, CUDA),
        "step_with_packet": lambda forged: hc.cuda_step_with_packet(forged, dev.rate_on, dev.state, 1.0, CUDA),
        "column_step": lambda forged: hc.prepared_column_step(forged, h, s, dev.rate_on, 1.0),
        "accumulator": lambda forged: hc.CudaStormAccumulator(
            forged, cum_rain=grids[0], cum_intake=grids[1], cum_return=grids[2], cum_drain=grids[3],
            peak_depth=grids[4], peak_velocity=grids[5], peak_q=cp.zeros(()), peak_t=cp.zeros(())),
    }


@pytest.mark.parametrize("name", sorted(FORGERIES))
@pytest.mark.parametrize("mode", ["fused", "split"])
def test_forged_scalar_metadata_is_refused_before_any_launch(launches, name, mode):
    cp = cupy()
    case = hcs.build_case(31, kind="valley", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)
    forged = dataclasses.replace(ctx, **FORGERIES[name](ctx))
    for label, entry in _entries(ctx, dev, cp).items():
        with pytest.raises(RoutingError, match="metadata|extents"):
            entry(forged)
        assert launches == [], f"{name}/{label}: a kernel was requested"


def test_forged_device_id_is_refused_as_a_device_mismatch(launches):
    cp = cupy()
    case = hcs.build_case(32, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    forged = dataclasses.replace(ctx, device_id=ctx.device_id + 1)
    for label, entry in _entries(ctx, dev, cp).items():
        with pytest.raises(RoutingError, match="device"):
            entry(forged)
        assert launches == [], label


@pytest.mark.parametrize("name", ["n_cells_larger", "n_active_larger", "level_bounds_end_extended",
                                  "level_bounds_dropped_level", "shape_changed"])
def test_a_consistently_resealed_forgery_is_still_refused_by_the_extent_check(launches, name):
    """Replacing the signature too (an attacker who knows the format) cannot pass: the sealed counts must also match
    the extents of the owned device arrays, which the fingerprint checks keep fixed."""
    cp = cupy()
    case = hcs.build_case(33, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    changed = dataclasses.replace(ctx, **FORGERIES[name](ctx))
    resealed = dataclasses.replace(changed, scalar_signature=hc._scalar_signature(changed))
    for label, entry in _entries(ctx, dev, cp).items():
        with pytest.raises(RoutingError, match="extents|metadata"):
            entry(resealed)
        assert launches == [], f"{name}/{label}"


def test_non_tuple_metadata_equal_to_the_sealed_values_is_still_refused(launches):
    cp = cupy()
    case = hcs.build_case(35, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    assert isinstance(ctx.shape, tuple) and isinstance(ctx.level_bounds, tuple)  # valid values stay tuples
    for changes in ({"shape": list(ctx.shape)}, {"level_bounds": list(ctx.level_bounds)},
                    {"level_bounds": [*ctx.level_bounds]}):
        forged = dataclasses.replace(ctx, **changes)
        for label, entry in _entries(ctx, dev, cp).items():
            with pytest.raises(RoutingError, match="metadata"):
                entry(forged)
            assert launches == [], label


def test_the_seal_costs_no_device_access_and_an_unforged_context_still_works():
    cp = cupy()
    case = hcs.build_case(34, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    assert ctx.scalar_signature == hc._scalar_signature(ctx)
    from maple.core import backend as mb

    before = mb.read_transfer_counters()
    cp_ctx = dataclasses.replace(ctx)  # an unchanged copy keeps the same seal and the same owned arrays
    hc._check_context(cp, cp_ctx)
    delta = mb.read_transfer_counters().delta(before)
    assert delta.device_to_host == delta.host_to_device == delta.scalar_reads == 0
    step = hc.prepared_coupled_step(cp_ctx, dev.rate_on, dev.state, 1.0, CUDA)
    assert step.route.implementation == "cuda"
