"""CW-B20: the private event-loop accounting step.

* the public `sediment_coupled_step` still returns the COMPLETE `SedimentCoupledStep`;
* every field of the private `SedimentAccountingStep` is byte-identical to the corresponding full-result value on
  the same inputs (same objects for the MAPLE / transport fields; the loop's own `.sum(axis=(0, 1))` for the two
  physics class sums), for the dense chain and the deep resident-window bed with CSR and dense phase;
* owners are released at the documented points: the complete physics result and the MAPLE pickup result before
  the deposit call, the post-pickup bed / ledger before reconciliation, the attempt before the next attempt;
* a transport rejection leaves the caller's physical bytes and the deep store unchanged;
* the COMPLETE driver (quiet completion + dry reset, wet pause, checkpoint, resumed completion) of this tree is
  byte-identical to the ORIGINAL B15 public-step driver, run from the B15 tree in a subprocess.

GPU parametrizations are written for the root's GPU0 run; without a device they skip (no claim)."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from b20_cases import CASES, build, host

import maple_syrup.sediment_event as event_module
from maple_syrup.sediment_event import (
    DepositAccounting,
    PhysicsAccounting,
    PickupAccounting,
    SedimentAccountingStep,
    SedimentCoupledStep,
    SedimentEventError,
    TransportAccounting,
    _sediment_accounting_step,
    sediment_coupled_step,
)
from maple_syrup.sediment_physics import SedimentPhysicsStep
from maple_syrup.sediment_transport import TransportStepRejected

BACKENDS = ["numpy", "cupy"]  # cupy skips without an actual device (no GPU claim)


def backend_xp(backend):
    from maple.core.backend import (
        BackendUnavailableError,
        gpu_execution_available,
        resolve_backend,
    )

    if backend == "cupy" and not gpu_execution_available():
        pytest.skip("actual CuPy device unavailable; no GPU claim")
    try:
        return resolve_backend(backend).xp
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))


def step_args(s, schedule):
    rate = s["field"].apply(schedule.rate_after_m_per_s(0.0))
    return (s["state0"], s["ctx"], s["column"], rate, s["vegetation"], s["sediment"], 1.0)


def same(a, b):
    a, b = host(a), host(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def leaves(obj):
    from maple.core.backend import array_leaves

    return [host(v) for v in array_leaves(obj)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
def test_public_step_is_complete_and_private_records_are_byte_identical(backend, case, tmp_path):
    xp = backend_xp(backend)
    s, _ctx, ctl, schedule = build(case, xp, tmp_path)
    args = step_args(s, schedule)
    full = sediment_coupled_step(*args, ctl)
    acc = _sediment_accounting_step(*args, ctl)
    # the public contract: the complete result, every field present and of its original type
    assert type(full) is SedimentCoupledStep and type(acc) is SedimentAccountingStep
    assert isinstance(full.physics, SedimentPhysicsStep)
    assert hasattr(full.pickup, "new_ledger") and hasattr(full.deposit, "new_ledger")
    assert full.characteristic is not None and full.phase is not None and full.phase_reconciliation is not None
    assert hasattr(full.transport, "x_face_gross_kg") and hasattr(full.characteristic, "phase_after")
    # the private records: the same values
    assert isinstance(acc.physics, PhysicsAccounting) and isinstance(acc.pickup, PickupAccounting)
    assert isinstance(acc.transport, TransportAccounting) and isinstance(acc.deposit, DepositAccounting)
    assert same(acc.physics.sediment_velocity_m_s, full.physics.sediment_velocity_m_s)
    assert same(acc.physics.raindrop_pickup_by_class_kg, full.physics.raindrop_pickup_kg.sum(axis=(0, 1)))
    assert same(acc.physics.flow_pickup_by_class_kg, full.physics.flow_pickup_kg.sum(axis=(0, 1)))
    assert set(acc.physics.regime_counts) == set(full.physics.regime_counts)
    assert all(same(acc.physics.regime_counts[k], full.physics.regime_counts[k]) for k in full.physics.regime_counts)
    for record, source, names in ((acc.pickup, full.pickup, event_module._PICKUP_ACCOUNTING_FIELDS),
                                  (acc.deposit, full.deposit, event_module._DEPOSIT_ACCOUNTING_FIELDS),
                                  (acc.transport, full.transport, event_module._TRANSPORT_ACCOUNTING_FIELDS)):
        for name in names:
            assert same(getattr(record, name), getattr(source, name)), name
    for name in ("bed", "storm", "phase"):
        a, b = leaves(getattr(acc, name)), leaves(getattr(full, name))
        assert len(a) == len(b) and all(u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes()
                                        for u, v in zip(a, b, strict=True)), name
    assert set(acc.phase_reconciliation) == set(full.phase_reconciliation)
    assert all(same(acc.phase_reconciliation[k], full.phase_reconciliation[k]) for k in full.phase_reconciliation)
    assert (acc.dt_s, acc.n_substeps, acc.n_transport_rejections) == \
        (full.dt_s, full.n_substeps, full.n_transport_rejections)


def test_retain_is_validated():
    with pytest.raises(SedimentEventError, match="retain"):
        event_module._sediment_coupled_core(None, None, None, None, None, None, 1.0, None, zero_demand=None,
                                            hydraulics=None, retain="partial")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("form", ["accounting", "full"])
def test_owners_are_released_before_deposit_and_reconciliation(backend, case, form, tmp_path, monkeypatch):
    """Weak references to the ACTUAL owners: the complete physics result, the MAPLE pickup result and the post-pickup
    bed's ledger / resident column objects. The accounting form releases physics and pickup before the deposit call
    (the post-pickup bed is still alive there: it is the deposit's input) and the post-pickup bed before
    reconciliation. The full form keeps them all (its public contract), which shows the probes can see them."""
    xp = backend_xp(backend)
    s, _ctx, ctl, schedule = build(case, xp, tmp_path)
    refs, seen = {}, {}
    # CW-B65: the accounting form reconciles through the private `_reconcile_phase_inputs` body (the complete kernel
    # result is already released); the full form still enters through the public `reconcile_phase`.
    reconcile_name = "_reconcile_phase_inputs" if form == "accounting" else "reconcile_phase"
    real_physics, real_apply, real_reconcile = (event_module.sediment_physics_step, event_module.apply_bed_demand,
                                                getattr(event_module, reconcile_name))

    def physics(*a, **k):
        out = real_physics(*a, **k)
        refs["physics"] = weakref.ref(out)
        return out

    def apply(bed, *a, **k):
        if "pickup" in refs:  # second call: the deposit
            gc.collect()
            seen["at_deposit"] = {n: r() is not None for n, r in refs.items()}
        new_bed, result = real_apply(bed, *a, **k)
        if "pickup" not in refs:
            refs["pickup"] = weakref.ref(result)
            refs["post_pickup_ledger"] = weakref.ref(new_bed.ledger)
            refs["post_pickup_column"] = weakref.ref(new_bed.voxel_column)
        return new_bed, result

    def reconcile(*a, **k):
        gc.collect()
        seen["at_reconcile"] = {n: r() is not None for n, r in refs.items()}
        return real_reconcile(*a, **k)

    monkeypatch.setattr(event_module, "sediment_physics_step", physics)
    monkeypatch.setattr(event_module, "apply_bed_demand", apply)
    monkeypatch.setattr(event_module, reconcile_name, reconcile)
    step = _sediment_accounting_step if form == "accounting" else sediment_coupled_step
    attempt = step(*step_args(s, schedule), ctl)
    if form == "accounting":
        assert seen["at_deposit"] == {"physics": False, "pickup": False, "post_pickup_ledger": True,
                                      "post_pickup_column": True}
        assert seen["at_reconcile"] == {"physics": False, "pickup": False, "post_pickup_ledger": False,
                                        "post_pickup_column": False}
    else:
        assert all(seen["at_deposit"].values()) and all(seen["at_reconcile"].values())
    del attempt


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
def test_attempt_records_are_released_before_the_next_attempt(backend, case, tmp_path, monkeypatch):
    from test_exchange_history import run

    xp = backend_xp(backend)
    s, ctx, ctl, schedule = build(case, xp, tmp_path)
    real = event_module._sediment_accounting_step
    previous, calls = {}, [0]

    def observed(*a, **k):
        if previous:
            gc.collect()
            live = [n for n, r in previous.items() if r() is not None]
            assert not live, f"previous attempt retained into the next attempt: {live}"
        attempt = real(*a, **k)
        previous.clear()
        previous.update({"attempt": weakref.ref(attempt), "pickup_record": weakref.ref(attempt.pickup),
                         "transport_record": weakref.ref(attempt.transport),
                         "deposit_record": weakref.ref(attempt.deposit)})
        calls[0] += 1
        return attempt

    monkeypatch.setattr(event_module, "_sediment_accounting_step", observed)
    result = run(s, ctx, ctl, schedule, 3.0, cadence=1.0)
    assert calls[0] >= 3 and result.closure()["closed"]


def store_pins(tmp_path):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
def test_injected_transport_rejection_leaves_the_caller_unchanged(backend, case, tmp_path, monkeypatch):
    xp = backend_xp(backend)
    s, ctx, ctl, schedule = build(case, xp, tmp_path)
    before = [v.tobytes() for v in leaves(s["state0"])]
    pins = store_pins(tmp_path)

    def reject(*a, **k):
        raise TransportStepRejected("injected transport rejection")

    for name in ("sparse_characteristic_step", "characteristic_step", "transport_step"):
        monkeypatch.setattr(event_module, name, reject)
    with pytest.raises(TransportStepRejected):
        _sediment_accounting_step(*step_args(s, schedule), ctl)
    from test_exchange_history import complete

    with pytest.raises(SedimentEventError):  # the loop halves dt until its retry floor, then refuses
        complete(s, ctx, ctl, schedule)
    assert [v.tobytes() for v in leaves(s["state0"])] == before
    assert store_pins(tmp_path) == pins  # no deep version, chunk or manifest published or changed


# --- the complete driver against the ORIGINAL B15 public-step driver -----------------------------------------------
B20_ROOT = Path(__file__).resolve().parents[2]
BASELINE_MANIFEST = Path("/home/okin/SYRUP/agent_handoffs/tasks/conservative_water/cw_b20_baseline_manifest.json")


def original_b15_tree():
    """The B15 tree this one was cloned from, verified unchanged against the B20 baseline manifest for every file
    the driver can import (src) and every test helper it uses; otherwise skip (no comparison claim)."""
    if not BASELINE_MANIFEST.is_file():
        pytest.skip("B20 baseline manifest unavailable")
    manifest = json.loads(BASELINE_MANIFEST.read_text())
    tree = Path(manifest["source"])
    for rel, digest in manifest["files"].items():
        if rel.startswith(("src/", "tests/phase5/", "tests/conservative_water/")):
            path = tree / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                pytest.skip(f"original B15 tree changed since the B20 baseline ({rel}); no comparison claim")
    return tree


def run_driver(src_tree, case, backend, tmp_path, label):
    maple = os.environ.get("MAPLE_SYRUP_EXPECTED_MAPLE_ROOT")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and not p.rstrip("/").endswith("/src")
             or (maple and p.startswith(maple))]
    env["PYTHONPATH"] = os.pathsep.join([*parts, str(src_tree / "src")])
    out = str(tmp_path / label)
    work = tmp_path / f"{label}_work"
    work.mkdir()
    proc = subprocess.run([sys.executable, str(B20_ROOT / "tests" / "cw_b20" / "driver_dump.py"),
                           str(src_tree / "tests"), case, backend, out, str(work)],
                          env=env, capture_output=True, text=True, timeout=1800, check=False)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    with np.load(out + ".npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    return arrays, json.loads(Path(out + ".json").read_text())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
def test_complete_driver_is_byte_identical_to_the_original_b15_driver(backend, case, tmp_path):
    backend_xp(backend)
    original = original_b15_tree()
    a_arrays, a_counts = run_driver(original, case, backend, tmp_path, "b15")
    b_arrays, b_counts = run_driver(B20_ROOT, case, backend, tmp_path, "b20")
    assert Path(a_counts.pop("maple_syrup")) == original / "src" / "maple_syrup"
    assert Path(b_counts.pop("maple_syrup")) == B20_ROOT / "src" / "maple_syrup"
    assert set(a_arrays) == set(b_arrays) and len(a_arrays) > 100
    bad = [k for k in a_arrays if not (a_arrays[k].dtype == b_arrays[k].dtype and a_arrays[k].shape == b_arrays[k].shape
                                       and a_arrays[k].tobytes() == b_arrays[k].tobytes())]
    assert not bad, f"{len(bad)} arrays differ, first {bad[:5]}"
    assert a_counts == b_counts
    assert a_counts["reference"]["type"] == "DryHandoff" and a_counts["resumed"]["type"] == "DryHandoff"
    assert a_counts["paused"]["type"] == "EventProgress"
