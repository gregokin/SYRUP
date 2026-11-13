"""Host tests (CPU/.venv: NO GPU, NO CuPy) of the CUDA held-context factory seam: interface strictness, ordering, and that the shared water-free CUDA setup keeps the owner constructor's models unchanged."""
# ruff: noqa: RUF059
from __future__ import annotations

import sys
from types import SimpleNamespace

import hbuild as hb
import pytest
from hbuild import AVAIL, make

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_pond as fp
from maple_syrup import floating_pond_cuda as fpc_
from maple_syrup import floating_soil_layers_cuda as fcu


def cpu_context():
    vert, owner, bed = make()
    rec = fh.export_held_context(vert, None, available_bytes_fn=AVAIL)
    return vert, fh.reconstruct_held_context(rec, hb.expected_of(vert), available_bytes_fn=AVAIL)


def test_cpu_factory_rejects_every_cuda_only_argument_before_any_read():
    vert, ctx = cpu_context()
    reads = []
    reader = lambda: reads.append(1) or hb.BIG
    for kw in ({"memory_budget_bytes": 5}, {"host_memory_budget_bytes": 5}, {"host_reserve_bytes": 1}, {"host_available_bytes_fn": reader}):
        with pytest.raises(fp.FloatingPondError, match="only to a CUDA context"):
            fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=reader, **kw)
    assert reads == []
    v = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)  # the CPU factory is unchanged
    assert v._owner is None and v._initial is None and v._registry == {} and v._provisional is None and v._held is ctx
    for call in (v.initial_soil_water, lambda: v.check_initial(None, None, 0.0)):
        with pytest.raises(fp.FloatingPondError, match="no initial payload"):
            call()
    t = v.held_template()
    assert t.column is ctx.column and t.generation == ctx.generation and t.source_witness == ctx.source_witness


@pytest.mark.parametrize("bad", [None, {"column": 1}, object(), "ctx"])
def test_a_foreign_or_forged_context_is_refused_before_any_read(bad):
    reads = []
    with pytest.raises(fh.HeldContextError):
        fp.FloatingPondVertical.from_held_context(bad, available_bytes_fn=lambda: reads.append(1) or hb.BIG)
    assert reads == []


def test_the_shared_cuda_setup_keeps_the_owner_models_and_the_held_variant_differs_only_by_the_depth_copy(monkeypatch):
    """CONTROL OF THE MODEL ARITHMETIC ONLY: the CuPy/device entry points are replaced by stand-ins that return a placeholder namespace and the current device id, and a REAL CPU floating column supplies L, N, layouts;
    nothing of the stand-ins reaches a numerical body. The owner path must give the models the pre-refactor formulas gave; the held path equals it except the retained term (no sealed initial depth copy)."""
    from maple_syrup import floating_matrix_infiltration_cuda as fmc
    from maple_syrup import floating_preferential_infiltration_cuda as fpc

    monkeypatch.setattr(fcu, "_rc", lambda: SimpleNamespace(require_cuda=lambda: SimpleNamespace(name="cp")))
    monkeypatch.setattr(fcu, "gate_device", lambda cp, dev_id: 0)
    vert, owner, bed = make(L=3, spec=hb.pref_spec(3, 3, species=2))
    col, pref = owner.column, vert.held_template().preferential
    quota = {"memory_budget_bytes": None, "reserve_bytes": 0, "available_bytes_fn": None, "host_memory_budget_bytes": None, "host_reserve_bytes": 0, "host_available_bytes_fn": None}
    a, b = SimpleNamespace(), SimpleNamespace()
    fpc_._set_models(a, col, pref, quota, 64, initial_depth_copy=True)
    fpc_._set_models(b, col, pref, quota, 64, initial_depth_copy=False)
    L, N = col.n_layers, col.n_cells
    mm, pm, cl = fmc.operation_terms("step", L, N), fpc.operation_terms("step", L, N), fcu.operation_terms("step", L, N)
    chk, rec = fcu.operation_terms("state_check", L, N), fcu.operation_terms("reconcile", L, N)
    payload = (L + 2) * N * 8
    stage = 8 * N * 8 + mm["outputs"] + pm["outputs"] + cl["outputs"] + max(mm["scratch"], pm["scratch"], cl["scratch"]) + 2 * payload  # the pre-refactor formulas, restated independently
    exch = 14 * N * 8 + (L * N + 2 * N) * 8 + rec["outputs"] + rec["scratch"] + 2 * N + 2 * chk["scratch"] + L * N * 8 + payload
    assert a.retained_bytes == (L * N + 2 * N) * 8 and a.stage_step_bytes == stage and a.exchange_bytes == exch
    assert a.construction_bytes == a.retained_bytes + 3 * payload + max(stage, exch) and a.step_bytes == stage
    assert b.retained_bytes == a.retained_bytes - N * 8 and (b.stage_step_bytes, b.exchange_bytes, b.payload_bytes) == (a.stage_step_bytes, a.exchange_bytes, a.payload_bytes)
    assert b.construction_bytes == a.construction_bytes - N * 8 and b.reserve_bytes == 0 and b.host_op_bytes == fpc_.HOST_OP_BYTES


def test_the_gate_is_the_first_operation_of_the_cuda_setup(monkeypatch):
    vert, ctx = cpu_context()
    order = []

    def refuse(*a, **k):
        order.append("gate")
        raise RuntimeError("wrong device")

    monkeypatch.setattr(fcu, "_rc", lambda: SimpleNamespace(require_cuda=lambda: SimpleNamespace(name="cp")))
    monkeypatch.setattr(fcu, "gate_device", refuse)
    v = SimpleNamespace()
    reads = []
    q = {"memory_budget_bytes": None, "reserve_bytes": 0, "available_bytes_fn": lambda: reads.append(1) or hb.BIG, "host_memory_budget_bytes": None, "host_reserve_bytes": 0, "host_available_bytes_fn": None}
    with pytest.raises(RuntimeError, match="wrong device"):
        fpc_._set_models(v, ctx.column, ctx.preferential, q, 64, initial_depth_copy=False)
    assert order == ["gate"] and reads == [] and "cupy" not in sys.modules and not hasattr(v, "retained_bytes")
