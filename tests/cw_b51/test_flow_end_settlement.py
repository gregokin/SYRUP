"""CW-B51: the explicit opt-in `CompletionPolicy.deposit_mobile_at_flow_end` (default False).

The B49 fixture (root-reproduced): a 3-cell actual MAPLE chain, saturated fixed-Ksat column (K = 1e-7 m/s,
theta0 = theta_sat), 5e-9 m depth, 1e-8 kg of mobile class-2 sediment in the middle cell (canonical CSR phase), zero
previous sediment velocity, no rain. Every hydraulic quiet gate passes from 1 s while the wet laws give `wet_no_law`
(v = r = 0): the default policy cannot complete; the option completes at 61 s through the existing terminal MAPLE
deposition. CPU tests run here; `cupy` parametrizations skip without a device and are for the root to run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

from conftest import TREE  # noqa: E402
from pit_helpers import column_for_grid, pit_inputs  # noqa: E402  (tests/conservative_pits)
from test_complete_event import fixture as phase6_fixture  # noqa: E402  (tests/phase6)
from test_coupled_backends import H, device_bed, graph_for_xp  # noqa: E402  (tests/conservative_water)
from test_sediment_event import chain_elevation, make_bed  # noqa: E402  (tests/phase5)

from maple_syrup import complete_event as ce  # noqa: E402
from maple_syrup.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from maple_syrup.complete_event import (  # noqa: E402
    FLOW_END_SETTLEMENT_ASSUMPTION,
    CompletionPolicy,
    DryHandoff,
    EventProgress,
    IncompleteEventError,
    complete_event,
    quiet_metrics,
)
from maple_syrup.rainfall import constant_rainfall, rainfall_field  # noqa: E402
from maple_syrup.sediment_bed import bed_inventory, with_water  # noqa: E402
from maple_syrup.sediment_event import (  # noqa: E402
    SedimentEventControl,
    SedimentEventError,
    exchange_history_totals,
    initial_event_state,
)
from maple_syrup.sediment_physics import plot1_sediment_parameters  # noqa: E402
from maple_syrup.storm import StormControl  # noqa: E402

MOBILE_KG = 1e-8
MOBILE_CELL = (1, 0, 1)
IDENTITY = {"case": "cw_b51 B49 fixture", "source": "test"}
FLOW_END = CompletionPolicy(deposit_mobile_at_flow_end=True)
TIMING_FIELDS = {"first_step_wall_s", "first_step_cpu_s", "remaining_wall_s", "remaining_cpu_s", "commit_wall_s"}


def backend_xp(name):
    if name == "numpy":
        return np
    from maple.core.backend import BackendUnavailableError, gpu_execution_available, resolve_backend

    if not gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; device variant not exercised (no claim)")
    try:
        return resolve_backend(name).xp
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))


# --- the B49 fixture ------------------------------------------------------------------------------------------------
def b49(xp=np, *, depth=5e-9, mobile=MOBILE_KG, record=False, storage="csr", schedule=None, windowed_dir=None):
    """`(args, kw_base)` for `complete_event` on the B49 inputs in namespace `xp` (dense or resident-window bed)."""
    bed, ctx = make_bed(chain_elevation(3), depth_m=depth)
    if record:
        ctx = replace(ctx, record_exchange_history=True)
    pool = np.zeros_like(np.asarray(bed.active_layer.mass_kg))
    pool[MOBILE_CELL] = mobile
    bed = with_water(bed, ctx, np.asarray(bed.water.depth_m), pool)
    implementation = "numba" if xp is np else "cuda"
    control = SedimentEventControl(storm=StormControl(max_dt_s=1.0, implementation=implementation, max_steps=400),
                                   phase_storage=storage, record_exchange_history=record)
    if windowed_dir is not None:
        bed = to_windowed(bed, ctx, windowed_dir)
    bed, ctx = device_bed(bed, ctx, xp)
    if windowed_dir is not None and xp is not np:
        from maple.surface.voxels.windowed_prefix import move_prefix_packets

        bed = replace(bed, voxel_column=move_prefix_packets(bed.voxel_column, to_windowed.last_host, xp))
    graph, terrain = graph_for_xp(bed, xp)
    column, soil0 = column_for_grid(graph, np.full(graph.shape, 1e-7), xp, theta0=0.4, theta_sat=0.4)
    state0 = initial_event_state(graph, terrain, bed, ctx, soil0, control=control)
    schedule = constant_rainfall(0.0, 1.0, 0.0) if schedule is None else schedule
    args = (state0, ctx, column, rainfall_field(*graph.shape, xp=xp), schedule, xp.zeros(graph.shape),
            plot1_sediment_parameters(xp=xp))
    return args, {"max_end_s": 120.0, "control": control, "report_every_s": 10.0}


def to_windowed(bed, ctx, store):
    """A genuine resident-window MAPLE bed of the same dense column (the existing window-test construction)."""
    from maple.core.types.sediment_ledger import process_index, zeros_sediment_ledger_state
    from maple.surface.topographic_commit.windowed_commit import windowed_committed_state_from_dense_baseline
    from maple.surface.voxels.windowed_inventory import windowed_column_from_dense

    column = windowed_column_from_dense(bed.voxel_column, ctx.geometry, ctx.mass_resolution_kg, store_dir=store,
                                        max_store_bytes=10**8,
                                        class_ids=tuple(g.class_id for g in ctx.grain_classes.classes),
                                        window_voxels=min(4, bed.voxel_column.mass_kg.shape[2]), chunk_columns=64,
                                        with_prefix_packets=True)
    committed, _ = windowed_committed_state_from_dense_baseline(bed.committed_topography, column, bed.active_layer,
                                                                ctx.geometry)
    ledger = zeros_sediment_ledger_state(ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes),
                                         present_processes=(process_index("water_erosion_deposition"),))
    to_windowed.last_host = column
    return replace(bed, voxel_column=column, committed_topography=committed, ledger=ledger)


def run(args, kw, policy, **extra):
    return complete_event(*args, **kw, policy=policy, **extra)


def snapshot(obj):
    from maple.core.backend import array_leaves

    return [H(a).tobytes() for a in array_leaves(obj)]


def store_pins(store):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(store).rglob("*") if p.is_file()}


def assert_identical(a, b, path="value"):
    """Deep exact comparison (arrays by dtype/shape/raw bytes, floats by IEEE bytes); timing fields excluded."""
    if hasattr(a, "__array__") and not isinstance(a, (float, int, bool, str)) and not dataclasses.is_dataclass(a):
        u, v = H(a), H(b)
        assert u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes(), path
        return
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert type(a).__name__ == type(b).__name__, path
        for f in dataclasses.fields(a):
            if f.name not in TIMING_FIELDS:
                assert_identical(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            assert_identical(a[k], b[k], f"{path}[{k!r}]")
        return
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_identical(x, y, f"{path}[{i}]")
        return
    if isinstance(a, float):
        assert isinstance(b, float) and np.float64(a).tobytes() == np.float64(b).tobytes(), (path, a, b)
        return
    if isinstance(a, types.ModuleType) or callable(a):
        assert a is b, path
        return
    slots = getattr(type(a), "__slots__", None)
    if slots and type(a).__eq__ is object.__eq__:
        for name in slots:
            assert_identical(getattr(a, name), getattr(b, name), f"{path}.{name}")
        return
    assert a == b, (path, a, b)


# --- validation -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [1, 0, 1.0, "true", None, np.bool_(True)])
def test_option_must_be_a_plain_bool_before_any_model_work(bad, monkeypatch):
    with pytest.raises(SedimentEventError, match="deposit_mobile_at_flow_end must be bool"):
        CompletionPolicy(deposit_mobile_at_flow_end=bad).validated()
    monkeypatch.setattr(ce, "evolve_sediment_event", lambda *a, **k: pytest.fail("model work before validation"))
    args, kw = b49()
    with pytest.raises(SedimentEventError, match="deposit_mobile_at_flow_end must be bool"):
        run(args, kw, CompletionPolicy(deposit_mobile_at_flow_end=bad))
    from maple_syrup.event_experiment import run_complete_case

    with pytest.raises(SedimentEventError, match="deposit_mobile_at_flow_end must be bool"):
        run_complete_case(TREE / "no-such-case", TREE / "no-such-output",
                          policy=CompletionPolicy(deposit_mobile_at_flow_end=bad))


def test_field_is_last_and_default_false():
    names = [f.name for f in dataclasses.fields(CompletionPolicy)]
    assert names[-1] == "deposit_mobile_at_flow_end" and names[-2] == "terminal_storage_water"
    assert CompletionPolicy().deposit_mobile_at_flow_end is False


# --- the B49 trigger --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("bed_kind", ["dense", "windowed"])
def test_b49_default_is_incomplete_and_the_option_settles_through_maple(backend, bed_kind, tmp_path):
    xp = backend_xp(backend)
    store = tmp_path / "store_default"
    args, kw = b49(xp, windowed_dir=store if bed_kind == "windowed" else None)
    with pytest.raises(IncompleteEventError, match="maximum event duration"):
        run(args, kw, CompletionPolicy())
    store2 = tmp_path / "store_flow_end"
    args, kw = b49(xp, windowed_dir=store2 if bed_kind == "windowed" else None)
    saved = snapshot(args[0])
    initial = H(bed_inventory(args[0].bed)).copy()
    out = run(args, kw, FLOW_END)
    assert isinstance(out, DryHandoff) and out.dry_state.t_s == 61.0
    assert snapshot(args[0]) == saved  # the caller's input state is untouched
    acc = out.accounting
    fes = acc["flow_end_settlement"]
    assert fes["policy"] == "deposit_mobile_at_flow_end" and fes["assumption"] == FLOW_END_SETTLEMENT_ASSUMPTION
    assert fes["mobile_before_settlement_by_class_kg"][1] == MOBILE_KG and sum(fes["mobile_before_settlement_by_class_kg"]) == MOBILE_KG
    assert fes["max_mobile_cell_class_kg_before_settlement"] == MOBILE_KG
    assert fes["mobile_below_significance_before_settlement"] is False
    assert fes["mobile_per_element_limit_kg"] == args[1].mass_resolution_kg
    assert fes["terminal_deposition_by_class_kg"] == acc["terminal_deposition_by_class_kg"]
    assert acc["terminal_deposition_by_class_kg"][1] == MOBILE_KG
    before = acc["completion_before_terminal"]
    assert before["max_mobile_cell_class_kg"] == MOBILE_KG and before["mobile_gate"].startswith("waived")
    assert before["max_depth_m"] <= FLOW_END.max_depth_m and before["rainfall_finished"]
    assert acc["sediment_closure"]["closed"] and abs(acc["combined_water_residual_m3"]) <= acc["storm_water_budget"]["tolerance_m3"]
    assert np.all(np.abs(np.asarray(acc["terminal_sediment_residual_kg"])) <= acc["terminal_sediment_tolerance_kg"])
    assert not np.any(H(out.dry_state.bed.water.mobile_mass_by_cell_class_kg))
    final = H(bed_inventory(out.dry_state.bed))
    assert abs(final[1] - (initial[1] + MOBILE_KG)) <= 1e-14 and np.all(np.abs(np.delete(final - initial, 1)) <= 1e-14)
    if bed_kind == "windowed":
        assert out.dry_state.bed.voxel_column.pending_deep is None


# --- gates are unchanged with the option ---------------------------------------------------------------------------------
def test_rain_ongoing_positive_water_and_unmet_hold_still_refuse_with_the_option():
    args, kw = b49(schedule=constant_rainfall(0.0, 500.0, 0.5))  # gentle rain never ends inside the event
    with pytest.raises(IncompleteEventError):
        run(args, kw, FLOW_END)
    args, kw = b49(depth=1e-5)  # surface water above the depth gate on a saturated soil
    with pytest.raises(IncompleteEventError):
        run(args, kw, FLOW_END)
    args, kw = b49()
    with pytest.raises(IncompleteEventError):  # quiet from 1 s, hold 60 s not yet met at 50 s
        run(args, dict(kw, max_end_s=50.0), FLOW_END)


@pytest.mark.parametrize("gate", ["max_depth_m", "surface_volume_m3", "max_unit_discharge_m2_s",
                                  "outlet_discharge_m3_s"])
def test_each_hydraulic_gate_still_applies_with_the_option(gate):
    """The B49 state passes every gate; tightening ONE gate below its value makes the option refuse too."""
    args, _ = b49()
    state, ctx = args[0], args[1]
    ok, metrics = quiet_metrics(state, ctx, FLOW_END, 0.0)
    assert ok and not metrics["mobile_below_significance"]
    limit = metrics[gate] / 2.0  # just below the state's own value
    tight = replace(FLOW_END, **{gate: limit})
    assert quiet_metrics(state, ctx, tight, 0.0)[0] is False
    assert quiet_metrics(state, ctx, FLOW_END, 1e9)[0] is False  # rain not finished
    strict_ok, strict = quiet_metrics(state, ctx, CompletionPolicy(), 0.0)
    assert not strict_ok and "mobile_gate" not in strict and "mobile_below_significance" not in strict


def test_false_metrics_are_exactly_the_original_dictionary():
    args, _ = b49()
    _, metrics = quiet_metrics(args[0], args[1], CompletionPolicy(), 0.0)
    assert set(metrics) == {"max_depth_m", "surface_volume_m3", "max_unit_discharge_m2_s", "outlet_discharge_m3_s",
                            "max_mobile_cell_class_kg", "mobile_by_class_kg", "mobile_per_element_limit_kg",
                            "rainfall_finished"}


# --- interplay ------------------------------------------------------------------------------------------------------
def test_dry_fixture_is_identical_with_and_without_the_option():
    args, kw = phase6_fixture()
    off = complete_event(*args, **kw)
    on = complete_event(*args, **dict(kw, policy=replace(kw["policy"], deposit_mobile_at_flow_end=True)))
    assert_identical(off.dry_state, on.dry_state, "dry_state")
    assert_identical(off.progress.result, on.progress.result, "result")
    acc_on = dict(on.accounting)
    fes = acc_on.pop("flow_end_settlement")
    acc_on["completion_before_terminal"] = {k: v for k, v in acc_on["completion_before_terminal"].items()
                                            if k not in ("mobile_gate", "mobile_below_significance")}
    acc_on["completion_after_terminal"] = {k: v for k, v in acc_on["completion_after_terminal"].items()
                                           if k not in ("mobile_gate", "mobile_below_significance")}
    assert acc_on == off.accounting
    assert fes["mobile_below_significance_before_settlement"] is True and sum(fes["terminal_deposition_by_class_kg"]) == 0.0


def test_wet_event_completes_with_closed_budgets_under_the_option():
    args, kw = phase6_fixture(wet=True, implementation="numba", phase_storage="csr")
    out = complete_event(*args, **dict(kw, policy=replace(kw["policy"], deposit_mobile_at_flow_end=True)))
    assert isinstance(out, DryHandoff)
    acc = out.accounting
    assert acc["sediment_closure"]["closed"] and abs(acc["combined_water_residual_m3"]) <= acc["storm_water_budget"]["tolerance_m3"]
    assert acc["flow_end_settlement"]["terminal_deposition_by_class_kg"] == acc["terminal_deposition_by_class_kg"]
    assert out.progress.quiet.since_s >= args[4].end_s


def test_terminal_pit_storage_and_flow_end_options_combine():
    schedule = constant_rainfall(0.0, 10.0, 100.0)
    inputs = pit_inputs(np, ksat=1e-5, pit_ksat=0.0)
    policy = CompletionPolicy(hold_s=10.0, terminal_storage_water=True, deposit_mobile_at_flow_end=True)
    control = SedimentEventControl(storm=StormControl(max_dt_s=1.0, max_steps=20000))
    out = complete_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"], schedule,
                         inputs["vegetation"], inputs["sediment"], max_end_s=900.0, control=control, policy=policy,
                         report_every_s=10.0)
    assert isinstance(out, DryHandoff)
    acc = out.accounting
    assert acc["terminal_storage_cells"] == 1 and "flow_end_settlement" in acc
    assert acc["completion_after_terminal"]["terminal_storage_water_excluded"]
    assert acc["sediment_closure"]["closed"] and abs(acc["combined_water_residual_m3"]) <= acc["storm_water_budget"]["tolerance_m3"]


# --- atomic refusal -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bed_kind", ["dense", "windowed"])
@pytest.mark.parametrize("failure", ["raise", "leave_mobile"])
def test_terminal_maple_failure_refuses_atomically(bed_kind, failure, tmp_path, monkeypatch):
    store = tmp_path / "store"
    args, kw = b49(windowed_dir=store if bed_kind == "windowed" else None)
    saved = snapshot(args[0])
    pins = store_pins(store) if bed_kind == "windowed" else None
    original = ce.apply_bed_demand

    def failing(bed, context, demand):
        if failure == "raise":
            raise ValueError("injected terminal MAPLE refusal")
        new_bed, result = original(bed, context, demand)
        return replace(new_bed, water=bed.water), result  # MAPLE "left" the pool mobile

    monkeypatch.setattr(ce, "apply_bed_demand", failing)
    with pytest.raises((ValueError, SedimentEventError)) as info:
        run(args, kw, FLOW_END)
    if failure == "leave_mobile":
        assert "no inventory erased" in str(info.value)
    assert snapshot(args[0]) == saved
    if pins is not None:
        assert store_pins(store) == pins  # no deep version was published


# --- exchange history --------------------------------------------------------------------------------------------------
def test_exchange_history_counts_the_terminal_deposition_and_physics_is_unchanged_by_recording():
    args, kw = b49(record=True)
    rec = run(args, kw, FLOW_END)
    args2, kw2 = b49(record=False)
    plain = run(args2, kw2, FLOW_END)
    totals = exchange_history_totals(rec.progress.result)
    assert totals["history_complete"] and totals["pending_ledger_empty"]
    touches = H(totals["arrays"]["n_physical_touches"])
    gross = H(totals["arrays"]["gross_physical_activity_kg"])
    # no detachment happens in this event (wet_no_law, no rain): the only exchange at the mobile cell is the terminal
    # deposition, folded through the forced final commit
    assert gross[MOBILE_CELL] == pytest.approx(MOBILE_KG, rel=1e-12, abs=0) and touches[MOBILE_CELL] >= 1
    for name in ("voxel_column", "active_layer"):
        assert H(getattr(rec.dry_state.bed, name).mass_kg).tobytes() == H(getattr(plain.dry_state.bed, name).mass_kg).tobytes()
    assert rec.accounting["flow_end_settlement"] == plain.accounting["flow_end_settlement"]


# --- checkpoint / restart ------------------------------------------------------------------------------------------------
def test_mid_quiet_fresh_restart_is_raw_exact_and_policy_identity_is_enforced(tmp_path):
    args, kw = b49()
    paused = run(args, kw, FLOW_END, checkpoint_callback=lambda p: p.result.state.t_s >= 30.0)
    assert isinstance(paused, EventProgress) and paused.quiet.since_s == 1.0 and not paused.quiet.ready
    path = tmp_path / "wet"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    policy_node = json.loads((path / "checkpoint.json").read_text())["payload"]["fields"]["policy"]["fields"]
    assert policy_node["deposit_mobile_at_flow_end"] is True
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    resumed = run((restored.result.state, *args[1:]), kw, FLOW_END, continuation=restored)
    full = run(*b49(), FLOW_END)
    assert isinstance(resumed, DryHandoff) and isinstance(full, DryHandoff)
    assert_identical(resumed.dry_state, full.dry_state, "dry_state")
    assert_identical(resumed.progress.result, full.progress.result, "result")
    assert resumed.accounting == full.accounting
    # a different stopping policy cannot continue this event
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    with pytest.raises(SedimentEventError, match="restart stopping or numerical controls changed"):
        run((restored.result.state, *args[1:]), kw, CompletionPolicy(), continuation=restored)
    # identity mismatch at load (the run identity carries the policy)
    with pytest.raises(SedimentEventError, match="identity"):
        load_checkpoint(path, args[1], args[2], args[6], dict(IDENTITY, completion_policy={"flow_end": False}))


def test_default_bundle_omits_the_field_and_malformed_values_are_refused(tmp_path):
    args, kw = b49(depth=1e-5)  # stays wet: pause at 20 s under the default policy
    paused = run(args, kw, CompletionPolicy(), checkpoint_callback=lambda p: p.result.state.t_s >= 20.0)
    path = tmp_path / "default"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    manifest = json.loads((path / "checkpoint.json").read_text())
    node = manifest["payload"]["fields"]["policy"]["fields"]
    assert "deposit_mobile_at_flow_end" not in node and "terminal_storage_water" not in node
    assert load_checkpoint(path, args[1], args[2], args[6], IDENTITY).policy == CompletionPolicy()
    for label, mutate in (("int", lambda n: n.update(deposit_mobile_at_flow_end=1)),
                          ("float", lambda n: n.update(deposit_mobile_at_flow_end=1.0)),
                          ("string", lambda n: n.update(deposit_mobile_at_flow_end="true")),
                          ("unknown", lambda n: n.update(settle_everything=True))):
        bad = json.loads(json.dumps(manifest))
        mutate(bad["payload"]["fields"]["policy"]["fields"])
        target = tmp_path / f"bad_{label}"
        target.mkdir()
        for f in path.iterdir():
            (target / f.name).write_bytes(f.read_bytes())
        (target / "checkpoint.json").write_text(json.dumps(bad))
        with pytest.raises(SedimentEventError):
            load_checkpoint(target, args[1], args[2], args[6], IDENTITY)
    # an explicit False is decoded as the default (both encodings name the same policy)
    explicit = json.loads(json.dumps(manifest))
    explicit["payload"]["fields"]["policy"]["fields"]["deposit_mobile_at_flow_end"] = False
    target = tmp_path / "explicit_false"
    target.mkdir()
    for f in path.iterdir():
        (target / f.name).write_bytes(f.read_bytes())
    (target / "checkpoint.json").write_text(json.dumps(explicit))
    assert load_checkpoint(target, args[1], args[2], args[6], IDENTITY).policy == CompletionPolicy()


def test_completed_flow_end_handoff_round_trips(tmp_path):
    args, kw = b49()
    out = run(args, kw, FLOW_END)
    path = tmp_path / "dry"
    save_checkpoint(path, out, args[1], args[2], args[6], IDENTITY)
    loaded = load_checkpoint(path, args[1], args[2], args[6], IDENTITY, allow_complete=True)
    assert isinstance(loaded, DryHandoff) and loaded.accounting == out.accounting
    assert loaded.progress.policy == FLOW_END


# --- run identity and CLIs ------------------------------------------------------------------------------------------
def test_policy_identity_omits_the_default_and_records_the_option():
    from maple_syrup.event_experiment import policy_identity

    default = policy_identity(CompletionPolicy())
    assert "deposit_mobile_at_flow_end" not in default
    assert default == {k: v for k, v in dataclasses.asdict(CompletionPolicy()).items() if k != "deposit_mobile_at_flow_end"}
    assert policy_identity(FLOW_END)["deposit_mobile_at_flow_end"] is True


def test_both_clis_expose_the_opt_in(monkeypatch):
    from maple_syrup import event_experiment, windowed_experiment

    seen = []

    class Done:
        summary = {"ok": True}

    def capture(*a, **k):
        seen.append(k["policy"])
        return Done()

    monkeypatch.setattr(event_experiment, "run_complete_plot1", capture)
    assert event_experiment.main(["--case-dir", "c", "--output-dir", "o"]) == 0
    assert event_experiment.main(["--case-dir", "c", "--output-dir", "o", "--deposit-mobile-at-flow-end"]) == 0
    monkeypatch.setattr(windowed_experiment, "run_complete_windowed", capture)
    base = ["--case-dir", "c", "--output-dir", "o", "--store-dir", "s", "--case-kind", "rfid"]
    assert windowed_experiment.main(base) == 0
    assert windowed_experiment.main(base + ["--terminal-storage-water", "--deposit-mobile-at-flow-end"]) == 0
    assert [p.deposit_mobile_at_flow_end for p in seen] == [False, True, False, True]
    assert seen[0] == CompletionPolicy() and seen[2] == CompletionPolicy()
    assert seen[3] == CompletionPolicy(terminal_storage_water=True, deposit_mobile_at_flow_end=True)


# --- False path against the baseline function ------------------------------------------------------------------------------
BASELINE_COMPLETE_EVENT_SHA256 = "d18624c1d2c1b46db62d94be59c56656e065f8cfcf986626420a0755db4a4be8"


B46_REFERENCE_ENV = "MAPLE_SYRUP_B46_REFERENCE"  # root of a B46 runtime snapshot (the directory holding src/)


def baseline_candidates() -> list[Path]:
    """Where the pinned B46 `complete_event.py` may be, in order: the explicit `MAPLE_SYRUP_B46_REFERENCE` snapshot root;
    this repository's `outputs/conservative_water/cw_b46_runtime_frozen` (the SYRUP root tree); a sibling
    `cw_b46_runtime_frozen` (a private candidate tree living under `outputs/conservative_water`)."""
    rel = Path("src") / "maple_syrup" / "complete_event.py"
    explicit = os.environ.get(B46_REFERENCE_ENV)
    if explicit:
        return [Path(explicit) / rel]
    return [TREE / "outputs" / "conservative_water" / "cw_b46_runtime_frozen" / rel,
            TREE.parent / "cw_b46_runtime_frozen" / rel]


def baseline_complete_event():
    """The pinned baseline `complete_event.py` (B46 runtime). A file with the wrong SHA-256 FAILS (never a silent
    substitute); an explicitly configured but missing snapshot FAILS; no snapshot anywhere is an explicit skip."""
    candidates = baseline_candidates()
    for path in candidates:
        if path.is_file():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != BASELINE_COMPLETE_EVENT_SHA256:
                pytest.fail(f"baseline {path} hash differs from the pinned baseline manifest")
            module = types.ModuleType("cw_b51_baseline_complete_event")
            module.__file__ = str(path)
            sys.modules[module.__name__] = module
            exec(compile(data, str(path), "exec"), module.__dict__)  # noqa: S102 -- pinned frozen baseline
            return module
    if os.environ.get(B46_REFERENCE_ENV):
        pytest.fail(f"{B46_REFERENCE_ENV}={os.environ[B46_REFERENCE_ENV]!r} holds no src/maple_syrup/complete_event.py")
    pytest.skip(f"pinned B46 baseline complete_event.py not found (looked in {[str(c) for c in candidates]}; set "
                f"{B46_REFERENCE_ENV}); this differential is NOT run and makes no claim")


@pytest.mark.parametrize("wet", [False, True])
def test_false_path_is_byte_identical_to_the_baseline_function(wet):
    old = baseline_complete_event()
    args, kw = phase6_fixture(wet=wet, implementation="numba", phase_storage="csr")
    new_out = complete_event(*args, **kw)
    old_kw = dict(kw, policy=old.CompletionPolicy(**{f.name: getattr(kw["policy"], f.name)
                                                     for f in dataclasses.fields(old.CompletionPolicy)}))
    old_out = old.complete_event(*args, **old_kw)
    assert_identical(new_out.dry_state, old_out.dry_state, "dry_state")
    assert_identical(new_out.progress.result, old_out.progress.result, "result")
    assert new_out.accounting == old_out.accounting
    assert new_out.progress.quiet == old_out.progress.quiet or dataclasses.asdict(new_out.progress.quiet) == dataclasses.asdict(old_out.progress.quiet)
    # the B49 trigger: both refuse identically under the default policy
    args, kw = b49()
    msgs = []
    for module, policy in ((ce, CompletionPolicy()), (old, old.CompletionPolicy())):
        with pytest.raises(SedimentEventError) as info:
            module.complete_event(*args, **kw, policy=policy)
        msgs.append((type(info.value).__name__, str(info.value)))
    assert msgs[0] == msgs[1]
