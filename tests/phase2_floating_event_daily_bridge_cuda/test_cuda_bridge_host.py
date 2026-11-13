"""CUDA event<->daily bridge HOST tests (CPU/host only: NO GPU, NO CuPy). They prove what the host can: the reconciliation of the accepted GPU lifecycle with the reviewed CPU bridge hooks (DailyHold carries the sealed GPU
host metadata, the pending-storm proof is ONE shared backend-neutral body that the CUDA close runs, the unit-rain-scale proof is a resident device check), the CUDA branches sit where the design says (dispatch/gate/quota/admission
order, scalar-only wet fragments), CPU behaviour is unchanged (meaningless CUDA arguments refused) and the new module never mirrors water to the host. Every device behaviour is defined in `test_gpu_definitions.py` (unrun)."""
# ruff: noqa: RUF059
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_daily_bridge_cuda as bc
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_event_lifecycle_cuda as flc
from maple_syrup import floating_pond as fp


def test_daily_hold_carries_the_sealed_gpu_host_metadata_as_trailing_optional_fields():
    names = [f.name for f in dataclasses.fields(lc.DailyHold)]
    assert names[-3:] == ["host_active", "edges_host", "host_depth"]
    assert all(f.default is None for f in dataclasses.fields(lc.DailyHold)[-3:])
    src = inspect.getsource(lc)
    assert src.count("hold.host_active, hold.edges_host, hold.host_depth") == 2  # rebase_boundary_from_day and day_runtime rebuild the root WITH the host metadata
    assert "host_active=r.host_active, edges_host=r.edges_host, host_depth=r.host_depth" in inspect.getsource(lc.stage_daily_hold)


def test_the_pending_storm_proof_is_one_shared_backend_neutral_body_run_by_both_closes():
    assert "_pending_storm_proof(occurrence, event, state, audit)" in inspect.getsource(lc.close_occurrence)
    assert "lc._pending_storm_proof(occurrence, event, state, audit)" in inspect.getsource(flc.close_cuda)
    body = inspect.getsource(lc._pending_storm_proof)
    assert "forcing_conformity" in body and "_check_storm_binding" in body  # producer proof for ANY pending cursor + bound forcing/end/amount
    chk = inspect.getsource(lc._check_storm_binding)
    assert "unit_rain_scale_cuda" in chk and "np.all(scale[act] == 1.0)" in chk  # CPU body unchanged; the device branch is a resident scalar check
    src = inspect.getsource(flc.close_cuda)
    assert src.index("_pending_storm_proof") < src.index("consume_completion")  # proven BEFORE the consume and the head
    assert "occurrence.cursor" in src and "storm_match" in src  # cursor / pending forcing / match carried into the GPU boundary


def test_unit_rain_scale_cuda_is_one_resident_reduction_with_the_gate_first():
    src = inspect.getsource(flc.unit_rain_scale_cuda)
    assert src.index("_gate(") < src.index("cp.all(") and src.count("_scalar(") == 1 and "np.asarray" not in src


def test_cpu_open_day_refuses_meaningless_cuda_arguments_and_cuda_is_dispatched_before_any_cpu_array():
    src = inspect.getsource(br.open_day)
    assert src.index("open_day_cuda") < src.index("_build_context(")  # the CPU arrays are built only inside the shared `_build_context` (CPU), reached after the CUDA dispatch
    assert "only to a CUDA boundary" in src


def test_open_day_cuda_order_gate_quota_admission_then_arrays_then_head_last():
    src = inspect.getsource(bc.open_day_cuda)
    assert src.index("fdc.gate(") < src.index("fdc.quota(") < src.index("fdc.admit(") < src.index("fdc.zeros(") < src.index("install_daily_hold")
    assert src.index("stage_daily_hold") > src.index("fdc.admit(")  # nothing is staged before the admission
    assert "root.depth_m" in src and "root.edges_host" in src and "root.host_active" in src  # the CURRENT device D and the sealed host metadata by reference (no D2H, no bed/NVK scan)
    assert "bed_inventory" not in src and "NativeFloatingSoil" not in src.replace("no `NativeFloatingSoil`", "")


def test_open_plan_is_pure_arithmetic_and_monotone():
    d1, h1 = bc.open_plan_cuda(3, 100)
    d2, h2 = bc.open_plan_cuda(3, 1000)
    assert d2 > d1 and h2 >= h1


def test_the_wet_fragment_on_cuda_leaves_scalars_only_and_the_cuda_module_never_mirrors_water():
    step = inspect.getsource(fp._FloatingSolver._step)
    cuda_branch = step[step.index('if v.backend == "cuda":  # SCALAR terms only'):step.index("else:\n                self.last_day")]
    assert "np.asarray" not in cuda_branch and "e.ledger.evaporation_m3" in cuda_branch
    mod = inspect.getsource(bc)
    code = "\n".join(line for line in mod.splitlines() if not line.strip().startswith(("#", '"""')))
    assert code.count("to_host(") == 1 and ".get()" not in code and "asnumpy" not in code
    bridge = inspect.getsource(br.wet_day_action)
    for banned in ("np.asarray(hyd.", "np.array(hyd."):
        assert banned not in bridge  # the wet loop copies through the vertical's namespace (`xp`), never `np` on a device array


def test_dry_day_on_cuda_is_the_existing_dispatch_with_the_host_pool_arguments_only_for_cuda():
    src = inspect.getsource(br.day_action)
    assert "_cuda_args(" in src and src.count("**cq") == 2  # BOTH the dry and the wet branch receive the host-pool arguments
    helper = inspect.getsource(br._cuda_args)
    assert 'backend == "cuda"' in helper and "only to a CUDA" in helper  # a CPU lineage refuses them instead of stripping


def test_the_cuda_lifecycle_and_cpu_bridge_modules_import_without_cupy():
    for mod in (bc, flc):
        tree = ast.parse(inspect.getsource(mod))
        assert not any("cupy" in ast.dump(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))


def test_cpu_open_day_rejects_cuda_only_arguments(tmp_path):
    from bridge_builders import RUN, dry_boundary
    from case import AVAIL

    from maple_syrup import soil_evaporation as se

    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    part = se.EvaporationPartition(0.45, 0.4, 0.5, 0.0, 0.3)
    with pytest.raises(br.BridgeError, match="only to a CUDA boundary"):
        br.open_day(lin, b, partition=part, run_digest=RUN, available_bytes_fn=AVAIL, host_reserve_bytes=5)
    assert lin.head is b  # refused before any mutation


def test_production_diff_against_the_bridge_cuda_baseline():
    """Per-interface proof against `event_daily_bridge_cuda_baseline.json` (accepted main164 + the exact CPU_C3 overlay): ONLY the four declared owning/bridge modules differ and ONLY the new CUDA bridge helper is added; the CPU_C3
    producer proof (pond_event), the daily state/output, every kernel/provider/native/child-codec module and the accepted GPU refresh are byte-identical."""
    here = Path(__file__).resolve()
    manifest = here.parents[3] / "completion_program_20261113" / "event_daily_bridge_cuda_baseline.json"
    if not manifest.exists():
        pytest.skip("bridge cuda baseline manifest not present")
    src = here.parents[2] / "src" / "maple_syrup"
    rel = {k.split("/")[-1]: v for k, v in json.loads(manifest.read_text())["files"].items() if k.startswith("src/maple_syrup/") and k.endswith(".py")}
    differing = {n for n, h in rel.items() if hashlib.sha256((src / n).read_bytes()).hexdigest() != h}
    present = {p.name for p in src.glob("*.py")}
    # CC/S1 composition (physical_lifecycle_cuda candidate): the C4 bridge baseline plus the accepted continued-CUDA codec, its CPU twin and the CPU closed-file hooks (daily producer record, lifecycle restore hooks) differ; each is
    # separately pinned by `tests/phase2_floating_lifecycle_checkpoint_cuda` against the pre-edit manifest of THIS task
    assert differing == {"floating_event_lifecycle.py", "floating_event_lifecycle_cuda.py", "floating_pond.py", "floating_event_daily_bridge.py", "floating_daily.py", "floating_checkpoint_cuda.py", "floating_continued_checkpoint.py"}
    assert present - set(rel) == {"floating_event_daily_bridge_cuda.py", "floating_continued_checkpoint_cuda.py", "floating_lifecycle_checkpoint.py", "floating_lifecycle_checkpoint_cuda.py"}
    for name in ("pond_event.py", "floating_daily_output.py", "floating_pond_cuda.py", "floating_event_refresh_cuda.py", "native_floating_soil_cuda.py", "floating_soil_layers_cuda.py",
                 "floating_daily_cuda.py", "floating_completion.py"):
        assert hashlib.sha256((src / name).read_bytes()).hexdigest() == rel[name], name


# ---- correction01: the caller's two-pool quota must reach the WET day (Root's device probe: the host reader was never called) -----------------------------------------------------------------------------------
class _Stop(RuntimeError):
    pass


def _wet_parent(tmp):
    from bridge_builders import RUN, wet_boundary
    from case import AVAIL

    from maple_syrup import soil_evaporation as se

    lin, b, factory = wet_boundary(tmp, "numpy", False)
    day = br.open_day(lin, b, partition=se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75), run_digest=RUN, available_bytes_fn=AVAIL)
    return lin, b, factory, day, RUN


def test_wet_day_forwards_the_callers_two_pool_readers_and_reserves_before_any_runtime_or_array(tmp_path, monkeypatch):
    """Labelled ATTRIBUTE/ORDER control: a genuine CPU wet bridge state whose column is flagged CUDA and whose device-only pieces (gate/quota/admission/input packet/validation) are recorded stand-ins; the REAL `wet_day_action`
    and `day_runtime` argument flow is what is observed: the caller's device reader/reserve and HOST reader/reserve reach the strict quota, the composed admission (before the runtime) and the vertical constructor."""
    import numpy as np
    from bridge_builders import RUN

    from maple_syrup import floating_daily as fd
    from maple_syrup import recharge as rc

    lin, b, factory, day, _run = _wet_parent(tmp_path / "a")
    col = day.ctx.column
    calls = []
    dev_r, host_r = (lambda: 1 << 40), (lambda: 1 << 41)
    monkeypatch.setattr(bc, "wet_quota", lambda column, a, r, ha, hr: calls.append(("quota", a, r, ha, hr)) or {"memory_budget_bytes": None, "reserve_bytes": r, "available_bytes_fn": a, "host_memory_budget_bytes": None,
                                                                                                                  "host_reserve_bytes": hr, "host_available_bytes_fn": ha})
    monkeypatch.setattr(bc, "admit_wet", lambda column, q: calls.append(("admit", q["reserve_bytes"], q["host_reserve_bytes"])))
    monkeypatch.setattr(bc, "inputs_ok_cuda", lambda *a: calls.append(("inputs",)) or True)
    monkeypatch.setattr(fd, "_validate_inputs", lambda *a: None)

    def runtime(hold, state, factory_, **kw):
        calls.append(("runtime", kw))
        raise _Stop

    monkeypatch.setattr(lc, "day_runtime", runtime)
    object.__setattr__(col, "backend", "cuda")
    try:
        n = col.n_cells
        with pytest.raises(_Stop):
            br.day_action(day, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=0), pet_terrain_m_s=np.zeros(n), cover=np.zeros(n), available_bytes_fn=dev_r, reserve_bytes=7, event_factory=factory,
                          host_available_bytes_fn=host_r, host_reserve_bytes=11)
    finally:
        object.__setattr__(col, "backend", "numpy")
    names = [c[0] for c in calls]
    assert names == ["quota", "admit", "inputs", "runtime"]  # strict quota -> composed admission -> validation packet -> runtime (nothing allocates before the admission)
    assert calls[0][1:] == (dev_r, 7, host_r, 11) and calls[1][1:] == (7, 11)
    kw = calls[3][1]
    assert kw["available_bytes_fn"] is dev_r and kw["reserve_bytes"] == 7 and kw["host_available_bytes_fn"] is host_r and kw["host_reserve_bytes"] == 11  # the vertical (and so every stage) gets BOTH pools


def test_day_runtime_forwards_every_pool_argument_to_the_boundary_vertical_and_the_wet_loop_uses_the_quota_for_storage():
    rt = inspect.getsource(lc.day_runtime)
    assert "host_available_bytes_fn=host_available_bytes_fn" in rt and "host_reserve_bytes=host_reserve_bytes" in rt and "memory_budget_bytes=memory_budget_bytes" in rt
    wet = inspect.getsource(br.wet_day_action)
    assert "**vkw" in wet and wet.count("**comp") == 2  # both native storage queries use the caller's readers
    assert wet.index("wet_quota(") < wet.index("admit_wet(") < wet.index("inputs_ok_cuda(") < wet.index("day_runtime(")
    hb = inspect.getsource(br.hand_back)
    assert "wet_quota(" in hb and "cuda_quota=q" in hb  # the hand-back storage also runs on the caller's pools


def test_cpu_lineage_refuses_every_cuda_only_pool_argument_instead_of_stripping_it(tmp_path):
    from case import AVAIL

    from maple_syrup import recharge as rc

    lin, b, factory, day, run = _wet_parent(tmp_path / "a")
    n = day.ctx.n_cells
    import numpy as np

    pl = rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=run, ordinal=0)
    kw = {"plan": pl, "pet_terrain_m_s": np.full(n, 3e-8), "cover": np.full(n, 0.3), "available_bytes_fn": AVAIL, "event_factory": factory}
    for extra in ({"host_available_bytes_fn": lambda: 1}, {"host_reserve_bytes": 5}):
        with pytest.raises(br.BridgeError, match="only to a CUDA"):
            br.day_action(day, **kw, **extra)
        with pytest.raises(br.BridgeError, match="only to a CUDA"):
            br.wet_day_action(day, **kw, **extra)
    assert isinstance(lin.head, lc.DailyHold) and day.ledger.days_applied == 0  # nothing ran


@pytest.mark.parametrize(("L", "N"), [(1, 1), (3, 12), (3, 15), (3, 30), (3, 4096), (5, 16641), (65, 50003), (3, 1_000_000)])
def test_open_plan_covers_the_existing_evaporation_preparation_requirement_across_dimensions(L, N):
    """Root's device reproducer: the advertised exact outer need was 1008 B (+ reserve) while the EXISTING `fe.prepare_floating_evaporation` guard requires 2152 B at that N; the guard correctly refused the nested stage at the
    advertised exact need. The outer model is now `max(own live payload, preparation requirement)` per pool (pure arithmetic, no guard change, no pad)."""
    from maple_syrup import floating_daily_cuda as fdc
    from maple_syrup import floating_evaporation as fe

    d, h = bc.open_plan_cuda(L, N)
    prep = fe.estimate_floating_evaporation_memory(N, backend="cuda")
    sd, sh = fdc.start_plan(L, N)
    assert d >= int(prep.device_bytes) and h >= int(prep.host_bytes)  # the nested preparation fits at exactly the advertised need
    own_d = sd + 3 * 8 * N + 8 * N
    assert d == max(own_d, int(prep.device_bytes))  # never summed, never padded
    assert d >= own_d  # and the operation's own payload stays covered
    assert bc.open_plan_cuda(L, N + 1)[0] >= d  # monotone in N


def test_open_plan_at_the_failing_root_fixture_dimensions_is_at_least_the_2152_byte_requirement():
    """Root's fixture (the 3 x 5 lake, N = 15, L = 3): the preparation required 2152 B on the device pool while the old outer model advertised less."""
    d, _h = bc.open_plan_cuda(3, 15)
    assert d >= 2152


def test_cuda_and_cpu_closes_use_the_same_roundoff_bound_operands_including_e_and_t():
    """Source-level unit metadata: the lifecycle cumulative-closure bound uses the SAME `volume_roundoff_bound_m3` count (12) and the SAME operand set on both backends, E and T included (the CUDA close previously omitted them from the
    operand scale while subtracting them from the expected storage)."""
    import re

    def call(fn):
        src = inspect.getsource(fn)
        m = re.search(r"volume_roundoff_bound_m3\((\d+), max\((.*?)\)\)", src, re.DOTALL)
        assert m, fn
        ops = [o.strip() for o in re.split(r",\s*(?![^()]*\))", m.group(2))]
        return int(m.group(1)), ops

    n_cpu, ops_cpu = call(lc.close_occurrence)
    n_gpu, ops_gpu = call(flc.close_cuda)
    assert n_cpu == n_gpu == 12 and len(ops_cpu) == len(ops_gpu) == 9
    assert any("evaporation_m3" in o for o in ops_cpu) and any("transpiration_m3" in o for o in ops_cpu)
    assert "ev" in ops_gpu and "tr" in ops_gpu and ops_gpu[-1] == "1e-30"
    assert "- ev - tr" in inspect.getsource(flc.close_cuda)  # the expected storage subtracts the same terms


def test_the_wet_fragment_reuses_the_verticals_readonly_zero_row_as_the_rain_row():
    src = inspect.getsource(fp._FloatingSolver.step_day)
    assert "self.vertical._zero_pet" in src and "self.xp.zeros(" not in src  # no per-fragment (N,) allocation
    init = inspect.getsource(fp.FloatingPondVertical.from_boundary)
    assert "_zero_pet.flags.writeable = False" in init  # the CPU row is read-only; the CUDA row is logically immutable


@pytest.mark.parametrize("n", [1, 12, 30, 4096, 16641, 50003])
def test_wet_runtime_plan_cuda_is_a_two_pool_model_with_the_evaporation_terms_in_their_own_pools(n):
    """Root's source-confirmed defect: the old composed line routed `step_bytes + fe.host_bytes + 6N*8 + q*8` to the DEVICE pool (the evaporation host packet+allowance charged to the device) and left the host at the default hot-op
    side. Pure arithmetic of the corrected model: DEVICE = step + fe.device + 6 N rows + carried q copies; HOST = host_op + the fe host PACKET only (its interpreter allowance is counted once by the admission)."""
    from maple_syrup import floating_evaporation as fe

    step, host_op, q = 123_456, 4096, 2 * n + 3
    d, h = flc.wet_runtime_plan_cuda(step, host_op, n, q)
    est = fe.estimate_floating_evaporation_memory(n, backend="cuda")
    assert d == step + int(est.device_bytes) + 6 * n * 8 + q * 8
    assert h == host_op + int(est.host_bytes) - fe.INTERPRETER_ALLOWANCE_BYTES
    assert d < step + int(est.host_bytes) + 6 * n * 8 + q * 8 or int(est.host_bytes) <= int(est.device_bytes)  # the host packet+allowance is no longer charged to the device pool
    assert h < host_op + int(est.host_bytes)  # and the allowance is not double counted (admit adds it once)
    assert flc.wet_runtime_plan_cuda(step, host_op, n + 1, q)[0] >= d  # monotone in N


def test_day_runtime_uses_the_two_pool_admission_on_cuda_and_the_historical_cpu_arithmetic_is_byte_unchanged():
    src = inspect.getsource(lc.day_runtime)
    assert 'vert.backend == "cuda"' in src and "wet_runtime_plan_cuda(" in src and "fpc_.admit(vert, d_need, what, h_need)" in src
    cpu = 'vert._read_and_admit(int(vert.step_bytes) + int(fe.estimate_floating_evaporation_memory(N, backend=vert.backend).host_bytes) + 6 * N * 8 + (int(qx.size) + int(qy.size)) * 8, what)'
    assert cpu in src  # the CPU line is the unchanged historical arithmetic
    assert src.index("fpc_.admit(vert") < src.index("vert.check_initial(")  # BEFORE any carried copy / initial state
