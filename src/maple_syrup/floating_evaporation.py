"""G3 S0: genuine E-only evaporation staging for a FLOATING porous column (`floating_soil_layers.FloatingSoilColumn`; NumPy, Numba and CUDA owners).

The fixed-column helper `soil_evaporation.stage_evaporation` is bound to `subsurface.SoilColumn`/`SpatialSoilColumn` and refuses a floating column (classes keep refusing each other). The arithmetic is NOT duplicated:
`soil_evaporation._stage_laws` is the ONE law body (extracted verbatim from `stage_evaporation`; the fixed wrapper is bit for bit unchanged) and this module is its second, independently validated caller. It reuses the
existing `EvaporationPartition`, `SurfaceEvaporationLedger`, `StagedEvaporation` (`native_step_kwargs()` pairs the combined potential rate and the post-cap share for `floating_soil_layers.floating_step`) and
`EvaporationMemory` types. The surface branch, the unpartitioned combined soil potential and the post-cap soil share are exactly those documented in `soil_evaporation`; no transpiration, no growth, no clock.

    op     = prepare_floating_evaporation(column, partition, available_bytes_fn=...)
    staged = stage_floating_evaporation(op, pet_terrain_m_s=..., cover=..., surface_water_m=..., dt_s=..., available_bytes_fn=...)
    step   = floating_soil_layers.floating_step(column, state, dt, added_water_m=..., **staged.native_step_kwargs())

Provenance and validation (BEFORE any allocation, in this order): exact operator type minted here and bound to a genuine `FloatingSoilColumn` (never a fixed column, never a copy), CUDA: the CURRENT device is the
column's device FIRST; the column's `evaporation_factor` must be EXACTLY 1.0 (the EcoHyD factor is `partition.soil_factor`, applied here only, never twice); the three inputs are exactly C-contiguous float64 `(N,)`
arrays of the column's namespace (and device); strict quota arguments; then admission; then the shared law body (which re-checks its own explicit arguments). The mask is the column's own active map.

Memory (declared, modelled bytes, not a whole-process bound): the fixed model of `soil_evaporation.estimate_evaporation_memory` for `N` cells (3 inputs, 5 outputs, declared scratch, packet) plus an explicit interpreter
allowance (`INTERPRETER_ALLOWANCE_BYTES`) on the HOST side. CPU/Numba: everything is host; CUDA: the compute set is DEVICE bytes (admitted against the CURRENT device's reading, device reserve once) and the host side is
ONE packet (`11 + 1` doubles: the device active count is the twelfth entry of the same single read, because a device column does not carry a host `n_active`) plus the allowance (admitted separately against the host
reading and the host reserve once). Nothing of the water, the maps or the inputs is downloaded or hashed. Not run on a GPU by its author; Root runs the CUDA definitions.
"""

from __future__ import annotations

import contextlib
import threading
import weakref
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup import floating_soil_layers as fsl
from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface as ss

__all__ = ["INTERPRETER_ALLOWANCE_BYTES", "FloatingEvaporationOperator", "estimate_floating_evaporation_memory", "prepare_floating_evaporation", "stage_floating_evaporation"]

INTERPRETER_ALLOWANCE_BYTES = 1 << 16  # declared interpreter/NumPy-call overhead allowance of one staging on the host side (modelled, not measured per process)
_PACKET_EXTRA_BYTES = 8  # the CUDA device active count: a twelfth packet entry of the SAME single host read

_MINT = threading.local()
_MINTED: weakref.WeakSet = weakref.WeakSet()


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def estimate_floating_evaporation_memory(n_cells: int, *, backend: str, reserve_bytes: int = 0, host_reserve_bytes: int = 0) -> se.EvaporationMemory:
    """Pure arithmetic: the fixed evaporation model for `N` cells with the floating side-split. CPU/Numba: `host_bytes = compute + allowance`, `required_bytes = host + reserve`. CUDA: `device_bytes = compute`,
    `required_bytes = device + reserve` (the DEVICE side), `host_bytes = packet + 8 + allowance` and `host_reserve_bytes` is carried in `extra_reserve_bytes` for the host admission."""
    if backend not in fsl.BACKENDS:
        raise fsl.FloatingBackendUnavailableError(f"backend must be one of {fsl.BACKENDS}, got {backend!r}")
    base = se.estimate_evaporation_memory(fsl._count(n_cells, "n_cells", minimum=1), backend=backend, reserve_bytes=fsl._count(reserve_bytes, "reserve_bytes"))
    extra = fsl._count(host_reserve_bytes, "host_reserve_bytes")
    if backend == "cuda":
        host = base.packet_bytes + _PACKET_EXTRA_BYTES + INTERPRETER_ALLOWANCE_BYTES
        return replace(base, packet_bytes=base.packet_bytes + _PACKET_EXTRA_BYTES, host_bytes=host, extra_reserve_bytes=extra, total_bytes=host + base.device_bytes + base.reserve_bytes + extra)
    host = base.host_bytes + INTERPRETER_ALLOWANCE_BYTES
    return replace(base, host_bytes=host, required_bytes=host + base.reserve_bytes, total_bytes=host + base.reserve_bytes + base.extra_reserve_bytes)


@dataclass(frozen=True, eq=False)
class FloatingEvaporationOperator:
    """An `EvaporationPartition` bound to one genuine floating column (identity, namespace, mask, area). Holds no water. Minted ONLY by `prepare_floating_evaporation`."""

    column: Any
    partition: se.EvaporationPartition
    backend: str
    n_cells: int

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise fsl.FloatingSoilError("a FloatingEvaporationOperator is minted only by prepare_floating_evaporation (no direct construction or replace())")
        _MINTED.add(self)

    def __copy__(self):
        raise fsl.FloatingSoilError("a FloatingEvaporationOperator is not copyable; prepare another")

    def __deepcopy__(self, memo):
        raise fsl.FloatingSoilError("a FloatingEvaporationOperator is not copyable; prepare another")

    def __reduce_ex__(self, protocol):
        raise fsl.FloatingSoilError("a FloatingEvaporationOperator is not serialisable")

    def summary(self) -> dict[str, Any]:
        return {"partition": self.partition.summary(), "backend": self.backend, "n_cells": self.n_cells, "column_sha256": self.column.sha256, "cell_area_m2": float(self.column.cell_area_m2),
                "column_evaporation_factor": float(self.column.evaporation_factor), "partition_level": se.PARTITION_LEVEL, "shared_law_body": "soil_evaporation._stage_laws",
                "scope": "pure staged forcing; the soil branch is the UNPARTITIONED combined potential rate plus the post-cap share for floating_step (native_step_kwargs); no transpiration/growth/clock"}


def _check_column(column: Any) -> None:
    """Provenance FIRST: exactly a genuine floating column (CUDA: the current device before anything else), evaporation factor exactly 1.0."""
    if type(column) is not fsl.FloatingSoilColumn:
        raise fsl.FloatingSoilError(f"column must be a genuine FloatingSoilColumn, got {type(column).__name__} (a fixed or spatial column is refused here, as it refuses a floating one)")
    fsl._gate(column)
    fsl._check_column(column)
    if float(column.evaporation_factor) != 1.0:
        raise fsl.FloatingSoilError(f"the column's evaporation_factor is {column.evaporation_factor!r}; this E-only helper applies the EcoHyD factor itself (partition.soil_factor), so the column factor must "
                                    "be exactly 1.0 to avoid applying it twice")


def _admit(column: Any, memory: se.EvaporationMemory, quota: dict, what: str) -> None:
    """CPU/Numba: host bytes against the declared budget/reading. CUDA: DEVICE side first (device reading of the CURRENT device by default), then the host packet side separately, each reserve once."""
    reserve = fsl._count(quota["reserve_bytes"], "reserve_bytes")
    budget = quota["memory_budget_bytes"]
    budget = None if budget is None else fsl._count(budget, "memory_budget_bytes")
    if column.backend != "cuda":
        fsl._admit(memory.host_bytes + reserve, budget, quota["available_bytes_fn"], what)
        return
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    device_fn = quota["available_bytes_fn"] or fcu.device_reader(cp)
    fsl._admit(memory.device_bytes + reserve, budget, device_fn, what + " (device)")
    host_reserve = fsl._count(quota["host_reserve_bytes"], "host_reserve_bytes")
    host_budget = quota["host_memory_budget_bytes"]
    fsl._admit(memory.host_bytes + host_reserve, None if host_budget is None else fsl._count(host_budget, "host_memory_budget_bytes"), quota["host_available_bytes_fn"], what + " (host packet)")


def prepare_floating_evaporation(column: Any, partition: se.EvaporationPartition, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0, available_bytes_fn: Any = None,
                                 host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Any = None) -> FloatingEvaporationOperator:
    """Bind `partition` to a genuine floating column. Order: provenance (CUDA device gate first, factor exactly 1) -> partition type -> strict quota -> admission of the per-staging requirement (nothing allocated)."""
    _check_column(column)
    if not isinstance(partition, se.EvaporationPartition):
        raise fsl.FloatingSoilError(f"partition must be an EvaporationPartition, got {type(partition).__name__}")
    quota = fsl._quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, column)
    memory = estimate_floating_evaporation_memory(column.n_cells, backend=column.backend, reserve_bytes=fsl._count(reserve_bytes, "reserve_bytes"),
                                                  host_reserve_bytes=fsl._count(host_reserve_bytes, "host_reserve_bytes"))
    _admit(column, memory, quota, "floating evaporation staging preparation")
    with _minting():
        op = FloatingEvaporationOperator(column, partition, column.backend, int(column.n_cells))
    return op


def stage_floating_evaporation(operator: Any, *, pet_terrain_m_s: Any, cover: Any, surface_water_m: Any, dt_s: Any, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                               available_bytes_fn: Any = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Any = None) -> se.StagedEvaporation:
    """One pure staging on the floating column (module docstring). Inputs are exactly C-contiguous float64 `(N,)` arrays of the column's namespace (and device) and are never written; `dt_s >= 0`."""
    if type(operator) is not FloatingEvaporationOperator or operator not in _MINTED:
        raise fsl.FloatingSoilError("operator must come from prepare_floating_evaporation (a copy or reconstruction does not carry the binding)")
    column = operator.column
    _check_column(column)  # device gate first, factor 1, genuine column
    if operator.backend != column.backend or operator.n_cells != int(column.n_cells) or not isinstance(operator.partition, se.EvaporationPartition):
        raise fsl.FloatingSoilError("operator record does not match its column; rebuild it with prepare_floating_evaporation")
    n = int(column.n_cells)
    dt = ss._check_dt(dt_s)
    for name, array in (("pet_terrain_m_s", pet_terrain_m_s), ("cover", cover), ("surface_water_m", surface_water_m)):
        fsl._check_vec(column, name, array, (n,))  # exact namespace type (and device), float64, C-contiguous, shape: metadata only
    quota = fsl._quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, column)
    memory = estimate_floating_evaporation_memory(n, backend=column.backend, reserve_bytes=fsl._count(reserve_bytes, "reserve_bytes"), host_reserve_bytes=fsl._count(host_reserve_bytes, "host_reserve_bytes"))
    _admit(column, memory, quota, "floating evaporation staging")
    if column.backend == "cuda":
        from maple_syrup import floating_soil_layers_cuda as fcu

        xp, n_active = fcu._rc().require_cuda(), None
    else:
        xp, n_active = np, int(column.n_active)
    after, evap, unmet, soil, share, ledger = se._stage_laws(xp, operator.partition, pet_terrain_m_s, cover, surface_water_m, dt, column._maps["active"], float(column.cell_area_m2), n_active)
    return se.StagedEvaporation(after, evap, unmet, soil, share, dt, ledger, memory, operator.partition)
