"""CUDA event lifecycle HOST tests (CPU/host only: NO GPU, NO CuPy). They prove what the host can: the CPU lifecycle is unchanged in behaviour, meaningless CUDA arguments are refused on CPU, the CUDA branches sit where the
design says (dispatch before any CPU-only refresh/np body), the new module never mirrors water to the host, and importing it does not import CuPy. Every device behaviour is defined in `test_gpu_definitions.py` (unrun)."""
# ruff: noqa: RUF059
from __future__ import annotations

import ast
import dataclasses
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from lifecycle_builders import AVAIL, build, run_to_complete

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_event_lifecycle_cuda as flc
from maple_syrup import floating_pond as fp
from maple_syrup import floating_pond_cuda as fpc
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase25_source_profile"))
import p25_source_profile as p25


def test_importing_the_cuda_lifecycle_module_does_not_import_cupy():
    tree = ast.parse(inspect.getsource(flc))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("cupy" in ast.dump(n) for n in top)


def test_boundary_root_new_fields_are_trailing_optional_and_default_none():
    """Branch chosen ONLY by the literal-pinned package (the imported package under test): the historical base169 package must end exactly with the three CUDA host fields; the Root-locked final package must keep those three
    in the SAME order and default None and then carry EXACTLY the one Phase 2.5 `vegetation` tail field, also default None."""
    branch = p25.require_branch(Path(lc.__file__).resolve().parent, historical=("base169_02388ad",))
    fields = dataclasses.fields(lc.BoundaryRoot)
    names = [f.name for f in fields]
    old_tail = ["host_active", "edges_host", "host_depth"]
    if branch.kind == "historical":
        assert names[-3:] == old_tail
        tail = fields[-3:]
    else:
        assert names[-4:] == [*old_tail, "vegetation"] and names.count("vegetation") == 1             # old order preserved, exactly one new trailing field
        assert names.index("host_active") == len(names) - 4
        tail = fields[-4:]
    for f in tail:
        assert f.default is None



def test_cpu_open_close_begin_refuse_meaningless_cuda_quota_arguments(tmp_path):
    owner, bed, factory, pref, ev0 = build(tmp_path / "a", "numpy", False)
    for kw in ({"memory_budget_bytes": 10}, {"host_memory_budget_bytes": 10}, {"host_reserve_bytes": 5}, {"host_available_bytes_fn": lambda: 1}):
        with pytest.raises(lc.LifecycleError, match="only to the CUDA"):
            lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref, **kw)
        assert owner.is_current  # refused before the vertical, the factory or any mutation


def test_cpu_lifecycle_round_trip_is_unchanged_and_boundary_has_no_host_metadata(tmp_path):
    owner, bed, factory, pref, ev0 = build(tmp_path / "a", "numpy", False)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)
    h = run_to_complete(occ.event, st)
    cap = occ.event.completion_capability(h, h.state)
    b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=AVAIL)
    assert lin.head is b and b.root.host_active is None and b.root.edges_host is None and b.root.host_depth is None
    with pytest.raises(fp.FloatingPondError, match="only to a CUDA boundary"):
        fp.FloatingPondVertical.from_boundary(b.root, available_bytes_fn=AVAIL, memory_budget_bytes=1)
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)
    assert lin.head is occ2


def _calls_before(fn_src: str, marker: str, later: tuple) -> bool:
    i = fn_src.index(marker)
    return all(fn_src.find(x) == -1 or fn_src.find(x) > i for x in later)


def test_cuda_dispatch_precedes_every_cpu_only_body_in_open_and_close():
    o = inspect.getsource(lc.open_floating_event)
    c = inspect.getsource(lc.close_occurrence)
    assert _calls_before(o, "flc.open_cuda", ("fp.FloatingPondVertical(", "_storage_m3(", "np.asarray"))
    assert _calls_before(c, "flc.close_cuda", ("fer.stage_completed_event_refresh", "np.asarray", "np.array"))
    assert 'backend == "cuda"' in c and 'backend == "cuda"' in o


def test_the_cuda_module_never_mirrors_water_to_the_host():
    src = inspect.getsource(flc)
    code = "\n".join(line for line in src.splitlines() if not line.strip().startswith(("#", '"""')))
    for banned in (".get()", "np.asarray(state", "np.asarray(payload", "water_m)", "asnumpy"):
        assert banned not in code.replace("state.hydraulic.soil_water_m", ""), banned
    assert code.count("to_host(") == 1  # the ONE counted scalar read helper


def test_open_gate_is_the_first_statement_of_the_cuda_opener_and_close_gate_precedes_the_reader():
    o = inspect.getsource(flc.open_cuda)
    c = inspect.getsource(flc.close_cuda)
    assert o.index("_gate(owner.column)") < o.index("quota_of(") < o.index("FloatingPondVertical(")
    assert c.index("_gate(") < c.index("quota_of(") < c.index("stage_completed_event_refresh_cuda")


def test_storage_model_is_pure_arithmetic_and_monotone():
    d1, h1 = flc.storage_model_cuda(3, 100)
    d2, h2 = flc.storage_model_cuda(3, 1000)
    d3, _ = flc.storage_model_cuda(3, 100, extra_device=800)
    assert d2 > d1 and d3 == d1 + 800 and h1 == h2 == flc.HOST_OP_BYTES


def test_init_cuda_boundary_is_additive_and_documented_water_free():
    assert callable(fpc.init_cuda_boundary) and "init_cuda_boundary" in fpc.__all__
    src = inspect.getsource(fpc.init_cuda_boundary)
    assert "_set_models(" in src and "to_host" not in src and "to_device" not in src and "SimpleNamespace" not in src


REQUIRED_VERTICAL_ATTRS = ("config", "backend", "_provisional", "available_bytes_fn", "_owner", "_held", "_initial", "_registry", "stats", "shape", "n_layers", "n_cells", "area", "identity", "cp", "device_id", "xp", "quota",
                           "reserve_bytes", "batch_cells", "host_op_bytes", "retained_bytes", "payload_bytes", "stage_step_bytes", "exchange_bytes", "surface_step_bytes", "last_exchange_required", "step_bytes",
                           "construction_bytes", "_zero_added", "_zero_pet", "_depth_scratch", "_host_active", "_edges_host", "_initial_depth", "_opened", "_boundary_bed", "_root_facts")


def _closed_boundary(tmp_path, **kw):
    owner, bed, factory, pref, ev0 = build(tmp_path, "numpy", False, **kw)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)
    h = run_to_complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    return lin, b, factory, occ


def test_cuda_boundary_constructor_sets_every_attribute_the_event_binding_needs(tmp_path, monkeypatch):
    """Root's device failure (AttributeError n_cells at `bind_shape`): the REAL `from_boundary` CUDA branch and `init_cuda_boundary` run here with NumPy standing in for CuPy and the device gate/admission recorded (a labelled
    ARITHMETIC/ATTRIBUTE control only, no CUDA claim): every attribute the constructors set for the other CUDA verticals must exist, and the real `bind_shape` (the failing call) must work."""
    lin, b, factory, occ = _closed_boundary(tmp_path / "a")
    root, col = b.root, b.root.column
    adm = []
    monkeypatch.setattr(fcu, "_rc", lambda: SimpleNamespace(require_cuda=lambda: np))
    monkeypatch.setattr(fcu, "gate_device", lambda cp, dev: 0)
    monkeypatch.setattr(fcu, "admit", lambda cp, dev, d, h, q, what: adm.append((d, h, what)))
    object.__setattr__(root, "host_active", np.asarray(col._maps["active"]))
    object.__setattr__(root, "edges_host", np.asarray(col._maps["edges"]))
    object.__setattr__(root, "host_depth", np.asarray(root.depth_m))
    old_backend = col.backend
    object.__setattr__(col, "backend", "cuda")
    object.__setattr__(col, "device_id", 0)
    try:
        v = fp.FloatingPondVertical.from_boundary(root, available_bytes_fn=AVAIL, reserve_bytes=3, host_reserve_bytes=5)
        missing = [a for a in REQUIRED_VERTICAL_ATTRS if not hasattr(v, a)]
        assert not missing, missing
        assert (v.n_layers, v.n_cells) == (col.n_layers, col.n_cells) and v.area == float(col.cell_area_m2) and v._opened is False and v._boundary_bed is root.bed
        assert len(adm) == 1 and adm[0][0] == v.construction_bytes and adm[0][1] == v.host_op_bytes  # ONE two-pool admission before any owned array
        v.bind_shape(tuple(occ.event.shape))  # the call that raised on the device
        assert v.shape == tuple(occ.event.shape)
    finally:
        object.__setattr__(col, "backend", old_backend)


def test_cuda_boundary_without_the_sealed_host_metadata_is_refused_typed_before_any_admission(tmp_path, monkeypatch):
    lin, b, factory, occ = _closed_boundary(tmp_path / "a")
    col = b.root.column
    adm = []
    monkeypatch.setattr(fcu, "_rc", lambda: SimpleNamespace(require_cuda=lambda: np))
    monkeypatch.setattr(fcu, "gate_device", lambda cp, dev: 0)
    monkeypatch.setattr(fcu, "admit", lambda *a: adm.append(a))
    old = col.backend
    object.__setattr__(col, "backend", "cuda")
    object.__setattr__(col, "device_id", 0)
    try:
        with pytest.raises(fp.FloatingPondError, match="sealed host mask and edges"):
            fp.FloatingPondVertical.from_boundary(b.root, available_bytes_fn=AVAIL)
    finally:
        object.__setattr__(col, "backend", old)
    assert adm == []


def test_a_failed_begin_leaves_the_accepted_boundary_the_head_and_a_retry_works(tmp_path):
    lin, b, factory, occ = _closed_boundary(tmp_path / "a")
    calls = []

    def bad(vert, topology=None):
        calls.append(1)
        raise RuntimeError("injected factory failure")

    with pytest.raises(RuntimeError, match="injected"):
        lc.begin_occurrence(lin, b, bad, available_bytes_fn=AVAIL)
    assert lin.head is b and calls == [1]
    with pytest.raises(nfs.NativeFloatingMemoryError):
        lc.begin_occurrence(lin, b, factory, available_bytes_fn=lambda: 0)  # a refusing reader: nothing adopted
    assert lin.head is b
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)
    assert lin.head is occ2


def test_real_next_step_after_reopen_and_second_close_accumulate_one_ledger_and_native_history(tmp_path):
    """CPU twin of the strengthened device definition: reopen, advance REAL native steps with rain, complete, close again: ONE cumulative closed ledger, J/I never decrease, a fresh refresh was counted."""
    from maple_syrup.rainfall import constant_rainfall

    lin, b, factory, occ = _closed_boundary(tmp_path / "a", loose=True)
    j0 = np.array(b.root.history.wetting_m, copy=True)
    i0 = np.array(b.root.history.cumulative_intake_m, copy=True)
    sched = constant_rainfall(float(b.t_s), float(b.t_s) + 0.4, 30.0)
    occ2, st2 = lc.begin_occurrence(lin, b, lambda v, topology=None: factory(v, topology, sched), available_bytes_fn=AVAIL)
    st3 = st2
    for _ in range(4):
        st3 = occ2.event.step(st3, 30.0 / 3.6e6, 0.1).state
    p = occ2.vertical.current_payload(st3.hydraulic.soil_water_m)
    assert np.all(np.asarray(p.history.cumulative_intake_m) >= i0) and np.all(np.asarray(p.history.wetting_m) >= j0)
    assert float(np.sum(np.asarray(p.history.cumulative_intake_m) - i0)) > 0.0  # real native intake happened
    h = run_to_complete(occ2.event, st3)
    b2 = lc.close_occurrence(lin, occ2, occ2.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert lin.head is b2 and b2.ledger["n_refreshes"] == 2 and b2.ledger["n_occurrences"] == 2 and b2.ledger["rain_m3"] > b.ledger["rain_m3"]
    assert abs(b2.ledger["residual_m3"]) <= b2.ledger["residual_bound_m3"]





def _perturb_after_completion(monkeypatch, fc_scale=0.5, sat_scale=0.5):
    """Provider perturbation installed AFTER the genuine open and the genuine completion and BEFORE the close: the held FC/SAT are then the ORIGINAL ones, the CURRENT W is positive and the refresh's new provider values are
    smaller, so the real changed-column cascade must return the unavoidable excess to the surface (a controlled provider perturbation, not a forged changed set, receipt or generation)."""
    real = nfs._batch_properties

    def wrapped(batch, config, K, provider_bytes):
        f, sat, k = real(batch, config, K, provider_bytes)
        return f * fc_scale, sat * sat_scale, k

    monkeypatch.setattr(nfs, "_batch_properties", wrapped)


def test_pending_return_is_actually_exercised_hprime_once_ledger_closes_and_the_gate_holds(tmp_path, monkeypatch):
    """CPU twin of the device definition (same lifecycle source path): the capacity return is POSITIVE (asserted, never a silent skip), H' = H + the returned water exactly once, the cumulative ledger closes and the boundary is a
    `pending_recession`: not eligible for a day/wind; positive native rain is refused BEFORE any head change; a genuine RAIN-0 continuation is allowed and one real native zero-rain step keeps the audit closed. A full quiet
    completion of the added-capacity recession is NOT required here (limit stated)."""
    from maple_syrup.rainfall import constant_rainfall

    owner, bed, factory, pref, ev0 = build(tmp_path / "a", "numpy", False)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)
    h = run_to_complete(occ.event, st)
    cap = occ.event.completion_capability(h, h.state)
    held_fc = np.array(occ.vertical.held_template().column._maps["theta_field_capacity"], copy=True)
    _perturb_after_completion(monkeypatch)
    b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=AVAIL)
    assert np.any(np.asarray(b.root.column._maps["theta_field_capacity"]) < held_fc)  # the refresh really adopted lower provider values than the held ones (real dirty column from the real terminal deposit)
    assert b.ledger["surface_return_m3"] > 0.0 and b.role == "pending_recession" and not b.eligible_for_interevent
    area = float(h.state.topology.dx_m) ** 2
    returned = float(np.asarray(b.surface_depth_m).sum() - np.asarray(h.state.hydraulic.depth_m).sum()) * area
    assert abs(returned - b.ledger["surface_return_m3"]) <= b.ledger["residual_bound_m3"]  # H' = H + the returned water, once
    assert abs(b.ledger["residual_m3"]) <= b.ledger["residual_bound_m3"]  # the cumulative soil + surface ledger closes
    sched = constant_rainfall(float(b.t_s), float(b.t_s) + 0.4, 30.0)
    with pytest.raises(lc.LifecycleError, match="ZERO native rainfall"):
        lc.begin_occurrence(lin, b, lambda v, topology=None: factory(v, topology, sched), available_bytes_fn=AVAIL)
    assert lin.head is b
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)  # the schedule is rain-0 after the boundary clock
    assert lin.head is occ2
    st3 = occ2.event.step(st2, 0.0, 0.05).state
    assert occ2.event.audit(st3)["water"]["closes"]
