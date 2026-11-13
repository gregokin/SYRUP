"""G3 S3: the CUDA plumbing of the floating daily lineage (`floating_daily`): quota normalisation, the composed DEVICE/HOST admission plans and the resident final-close audit, built ONLY from existing qualified pieces
(`floating_soil_layers_cuda.operation_terms/admit/gate_device/state_check/raise_flags`, `floating_transpiration_cuda.operation_terms`, the S0 evaporation estimate). No numerical body lives here and nothing here was run on a
GPU by its author; CuPy is reached only lazily through `floating_soil_layers_cuda._rc().require_cuda()` inside functions.

Pools. A CUDA lineage has TWO separate pools, as in the FSL/FT CUDA owners: the DEVICE pool (`available_bytes_fn`, `reserve_bytes`, optional `memory_budget_bytes`; the default reader is the CURRENT device's) and the
HOST pool (`host_available_bytes_fn`, `host_reserve_bytes`, optional `host_memory_budget_bytes`; the default is the real host reader). The cold source witness scan and the first-use source-pair hash are HOST work and are
admitted against the HOST pool only; device zero blocks and per-operation payloads against the device pool only; each reserve is counted once in its own pool. A CPU lineage rejects every host-pool argument (meaningless
there) and keeps its historical single-reader contract. Order of every CUDA transaction: current device gate FIRST (before any reader, hash or array operation), metadata, composed two-pool admission, then the work.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import floating_transpiration_cuda as ftc

__all__ = ["admit", "audit_plan", "day_plan", "gate", "quota", "start_plan", "withdrawal_plan"]

STORAGE_PACKET_BYTES = 8  # the storage query's single 8-byte host read (inside `operation_terms("storage")["host"]`)


def gate(column: Any) -> int:
    """The CURRENT device must be the column's device BEFORE any reader, hash or array operation (typed refusal; never selects or changes a device)."""
    cp = fcu._rc().require_cuda()
    return fcu.gate_device(cp, column.device_id)


def quota(available_bytes_fn: Any, reserve_bytes: Any, host_available_bytes_fn: Any, host_reserve_bytes: Any, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None) -> dict:
    """Strict two-pool quota dict for `fcu.admit` and the component calls (non-Boolean integers >= 0; callables or None)."""
    for name, fn in (("available_bytes_fn", available_bytes_fn), ("host_available_bytes_fn", host_available_bytes_fn)):
        if fn is not None and not callable(fn):
            raise fsl.FloatingSoilError(f"{name} must be callable or None")
    return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": fsl._count(reserve_bytes, "reserve_bytes"), "available_bytes_fn": available_bytes_fn,
            "host_memory_budget_bytes": host_memory_budget_bytes, "host_reserve_bytes": fsl._count(host_reserve_bytes, "host_reserve_bytes"), "host_available_bytes_fn": host_available_bytes_fn}


def component_quota(q: dict) -> dict:
    """The SAME readers/budgets with ZERO reserves: the composed admission already counted each pool's reserve once."""
    return {**q, "reserve_bytes": 0, "host_reserve_bytes": 0}


def admit(column: Any, device_bytes: int, host_bytes: int, q: dict, what: str) -> None:
    """Composed two-pool admission (device then host, each with its own reader/budget/reserve) BEFORE any owned array, readback or hash."""
    cp = fcu._rc().require_cuda()
    fcu.admit(cp, column.device_id, int(device_bytes), int(host_bytes), q, what)


def start_plan(L: int, N: int) -> tuple[int, int]:
    """DEVICE plan of the lineage construction: the `(L, N)` and `(N,)` zero blocks plus the initial-storage query (state check scratch + storage scratch). The HOST side (cold scan, first-use source hash, host
    scalars) is planned by `floating_daily._construction_need` against the host pool."""
    chk, sto = fcu.operation_terms("state_check", L, N), fcu.operation_terms("storage", L, N)
    device = 8 * L * N + 8 * N + chk["scratch"] + sto["device"]
    host = chk["host"] + sto["host"]
    return device, host


def day_plan(L: int, N: int, *, recharge: bool) -> tuple[int, int]:
    """One daily trial: the live E-only staging (3 caller inputs + 5 outputs + declared scratch, device) and the one FSL step (outputs + scratch; host = raw + rebuilt packet), plus the E packet on the host."""
    e = fe.estimate_floating_evaporation_memory(N, backend="cuda")
    s = fcu.operation_terms("step_with_recharge" if recharge else "step", L, N)
    device = int(e.device_bytes) + s["device"]
    host = int(e.packet_bytes) + s["host"]
    return device, host


def withdrawal_plan(L: int, N: int) -> tuple[int, int]:
    """One aggregate withdrawal (FT CUDA payload terms; the request is already device resident and caller-owned)."""
    t = ftc.operation_terms("withdraw", L, N)
    return t["outputs"] + t["scratch"], t["host"]


def audit_plan(L: int, N: int) -> tuple[int, int]:
    """The resident final-close audit: the state check (diag scratch + 32-byte packet) and the storage block tree (8-byte read)."""
    chk, sto = fcu.operation_terms("state_check", L, N), fcu.operation_terms("storage", L, N)
    return chk["scratch"] + sto["device"], chk["host"] + sto["host"]


def check_state_resident(column: Any, water: Any) -> None:
    """Resident validity of the CURRENT water through the existing state-check kernel: ONE 32-byte packet, nothing downloaded or hashed (the CPU private value checks never see a CuPy array)."""
    cp = fcu._rc().require_cuda()
    fcu.gate_device(cp, column.device_id)
    fcu.raise_flags(fcu.state_check(cp, column, water, initial=False), fcu.STATE_FLAG_TEXT, "final close")


def zeros(column: Any, shape: tuple) -> Any:
    """A device zero block on the column's device (device gate first); the caller admitted its bytes."""
    cp = fcu._rc().require_cuda()
    fcu.gate_device(cp, column.device_id)
    out = cp.zeros(shape, dtype=np.float64)
    return out
