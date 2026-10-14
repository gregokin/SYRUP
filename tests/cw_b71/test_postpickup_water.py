"""CW-B71: the accounting step releases the post-pickup MAPLE water before the characteristic transport.

Oracle: the B65 `sediment_event.py` (hard SHA `67d149a7...`), executed as an ISOLATED module in the current context
with its OWN `SedimentEventState` / `SedimentEventControl` types (rebuilt from the same fields), so every differential
compares two actual implementations. Every other package file must equal the B65 parent tree.

CPU (author, NumPy):
* pins and scope; only `_sediment_coupled_core` changed, two private definitions added; full / upwind text kept;
* accounting and full steps raw-identical to the oracle (complete recursive record incl. every nested scalar, signed
  zeros as raw bytes) on dense / window-CSR / window-dense / upwind, from the initial state AND from an advanced state
  with a NON-ZERO prior mobile pool and a non-zero actual MAPLE pickup;
* identical refusals (deposit call, reconciliation bound, transport rejection, new-water validation), an in-core
  transport retry, retry exhaustion through the outer loop's dt halving; caller bytes and deep-store files unchanged;
* the COMPLETE driver (quiet completion + dry reset, wet pause, checkpoint save/load, resumed completion) of this
  tree byte-identical to the B65 tree's, in subprocesses;
* weak-reference liveness with an old-source negative control: the post-pickup pool (and post-pickup BedState) is
  DEAD at transport entry in B71 and ALIVE in B65 on the characteristic paths, alive in both on upwind; the deposit
  call receives the same non-water field objects and T, its input bed recursively identical to B65's; nothing extra
  at return;
* the private rebuild equals `with_water` (same field objects, same water) and refuses the same invalid water with
  the same error type and message; the record carries every non-water `BedState` field, holds no water array, and
  refuses a changed field set.
DEVICE (root, GPU0; skipped here, a skip claims nothing): the same differentials on CuPy and an exact MemoryHook
live-byte difference B65 - B71 at transport entry equal to the ONE identified post-pickup mobile allocation, and zero
at return.
"""
from __future__ import annotations

import ast
import dataclasses
import gc
import hashlib
import struct
import sys
import types
import weakref
from collections import OrderedDict
from typing import ClassVar
from pathlib import Path, PurePath

import numpy as np
import pytest
import test_accounting_step as b20  # the existing CW-B20 accounting-step harness of THIS tree
from b20_cases import CASES, build, host

import maple_syrup.sediment_event as new

PRIVATE_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_PKG = PRIVATE_ROOT / "src" / "maple_syrup"
B65_ROOT = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b65_posttransport_retention")
B65_PKG = B65_ROOT / "src" / "maple_syrup"
ORACLE_SHA256 = "67d149a75076481f8bfbecd9b093c0358eef2fb8f4124c2fedfabee22ccbd176"
CUDA_SHA256 = "d13f4bf5d2642f65f823514b2ea09ea0ddb1c0275aedafb57596adfbb372d5b7"
BACKENDS = ["numpy", "cupy"]  # cupy skips without an actual device (no GPU claim)
ORACLE_MODULE = "_cw_b71_b65_sediment_event"
ALL_CASES = (*CASES, "upwind")
STARTS = ("initial", "advanced")
KERNELS = ("sparse_characteristic_step", "characteristic_step", "transport_step")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def old():
    path = B65_PKG / "sediment_event.py"
    if not path.is_file():
        pytest.skip("B65 oracle unavailable; no before/after claim")
    if _sha(path) != ORACLE_SHA256:
        pytest.fail("B65 sediment_event.py no longer has its pinned hash; the oracle comparison would be invalid")
    module = types.ModuleType(ORACLE_MODULE)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module  # dataclasses resolve string annotations through the module registry
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102 -- hash-pinned oracle text
    return module


def as_types(module, obj):
    """`obj` (a private-module SedimentEventState / SedimentEventControl) rebuilt as `module`'s OWN type."""
    cls = getattr(module, type(obj).__name__)
    return cls(**{f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init})


def leaves(obj):
    from maple.core.backend import array_leaves

    return [(str(np.asarray(host(a)).dtype), np.asarray(host(a)).shape, np.asarray(host(a)).tobytes())
            for a in array_leaves(obj)]


# --- the corrected B65 exact recursive record (copied verbatim, module names adapted) --------------------------------
_EVENT_MODULES = {new.__name__, ORACLE_MODULE}
_CACHE_SLOTS = {"maple.surface.voxels.deep_store.DeepStoreVersion": {"_memmaps", "_memmap_lock"}}


def _type_name(t):
    module = "maple_syrup.sediment_event" if t.__module__ in _EVENT_MODULES else t.__module__
    return f"{module}.{t.__qualname__}"


def _slot_names(t):
    names = []
    for k in reversed(t.__mro__):
        slots = k.__dict__.get("__slots__", ())
        names += [slots] if isinstance(slots, str) else list(slots)
    return names


def _is_array(value):
    return isinstance(value, np.ndarray) or (type(value).__module__.split(".")[0] == "cupy"
                                             and hasattr(value, "__cuda_array_interface__"))


def tree(value, path="step", active=None):
    """Exact recursive record (see tests/cw_b65): dataclasses, dicts, sequences, slot handles, arrays (backend, dtype,
    shape, raw bytes), NumPy scalars (dtype, raw bytes), native floats (raw IEEE bytes: -0.0 != 0.0), native scalars,
    paths and backend modules; any other type refuses."""
    active = set() if active is None else active
    if isinstance(value, types.ModuleType):
        return ("module", value.__name__)
    if value is None:
        return ("None",)
    if type(value) in (bool, int, str):
        return (type(value).__name__, value)
    if type(value) is float:
        return ("float", struct.pack("<d", value).hex())
    if isinstance(value, np.generic):
        return ("numpy-scalar", value.dtype.str, value.tobytes())
    if _is_array(value):
        a = np.ascontiguousarray(host(value))
        return ("array", type(value).__module__.split(".")[0], a.dtype.str, a.shape, a.tobytes())
    if isinstance(value, PurePath):
        return ("path", str(value))
    if id(value) in active:
        raise AssertionError(f"reference cycle at {path}")
    active.add(id(value))
    try:
        t = type(value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return ("dataclass", _type_name(t), tuple((f.name, tree(getattr(value, f.name), f"{path}.{f.name}", active))
                                                      for f in dataclasses.fields(value)))
        if t in (dict, OrderedDict):
            return (t.__name__, tuple((tree(k, f"{path}<key>", active), tree(v, f"{path}[{k!r}]", active))
                                      for k, v in value.items()))
        if t in (tuple, list):
            return (t.__name__, tuple(tree(v, f"{path}[{i}]", active) for i, v in enumerate(value)))
        slots = _slot_names(t)
        if slots and not hasattr(value, "__dict__"):
            skip = _CACHE_SLOTS.get(_type_name(t), set())
            return ("slots", _type_name(t), tuple((n, tree(getattr(value, n), f"{path}.{n}", active))
                                                  for n in slots if n not in skip and hasattr(value, n)))
    finally:
        active.discard(id(value))
    raise AssertionError(f"unrecorded type {_type_name(type(value))} at {path}")


def record(step):
    return {"type": _type_name(type(step)), "arrays": leaves(step), "tree": tree(step)}


# --- cases -----------------------------------------------------------------------------------------------------------
def build_case(case, xp, tmp_path):
    """The CW-B20 cases plus `upwind` (the accounting path without a characteristic kernel result)."""
    if case != "upwind":
        return build(case, xp, tmp_path)
    import test_exchange_history as teh

    bed, ctx = teh.make_bed(teh.chain_elevation(3), depth_m=.001, commit_spec=None)
    implementation = "array" if xp is np else "cuda"
    control = teh.SedimentEventControl(storm=teh.StormControl(max_dt_s=1, implementation=implementation,
                                                              max_steps=10000), transport_scheme="upwind")
    s = (teh.setup(bed, ctx, ksat=1e-5, control=control) if xp is np
         else teh.setup_xp(bed, ctx, xp, ksat=1e-5, control=control))
    return s, s["ctx"], control, teh.constant_rainfall(0, 10, 100)


ADVANCE_S = 3.0


def case_args(case, start, xp, tmp_path):
    """`(s, control, args)` with `args` the accounting-step positional arguments at the chosen start. `advanced`:
    the state after 3 s of the actual event (wet, NON-ZERO prior mobile pool; window cases have committed and
    published a deep version) and the rate at that time."""
    s, ctx, ctl, schedule = build_case(case, xp, tmp_path)
    if start == "initial":
        return s, ctl, b20.step_args(s, schedule)
    from test_exchange_history import run

    state = run(s, ctx, ctl, schedule, ADVANCE_S, cadence=1.0).state
    assert host(state.bed.water.mobile_mass_by_cell_class_kg).any(), "advanced start has no prior mobile pool"
    rate = s["field"].apply(schedule.rate_after_m_per_s(ADVANCE_S))
    s = dict(s, state0=state)
    return s, ctl, (state, ctx, s["column"], rate, s["vegetation"], s["sediment"], 1.0)


def call(module, form, args, ctl, old):
    fn = module._sediment_accounting_step if form == "accounting" else module.sediment_coupled_step
    if module is new:
        return fn(*args, ctl)
    return fn(as_types(old, args[0]), *args[1:], as_types(old, ctl))


def actual_pickup(step):
    return step.pickup.actual_removal_by_cell_class_kg


# =============================================================================================================
# CPU: pins, scope, untouched public text
# =============================================================================================================
def test_pins_and_all_other_package_files_equal_the_b65_parent():
    assert Path(new.__file__).resolve() == PRIVATE_PKG / "sediment_event.py"
    private = {str(p.relative_to(PRIVATE_PKG)): _sha(p) for p in PRIVATE_PKG.rglob("*.py") if "__pycache__" not in p.parts}
    parent = {str(p.relative_to(B65_PKG)): _sha(p) for p in B65_PKG.rglob("*.py") if "__pycache__" not in p.parts}
    assert set(private) == set(parent) and len(parent) == 55
    assert sorted(k for k in private if private[k] != parent[k]) == ["sediment_event.py"]
    assert parent["sediment_event.py"] == ORACLE_SHA256 and private["characteristic_cuda_csr.py"] == CUDA_SHA256


def _top(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_only_the_core_changed_and_full_upwind_text_is_kept():
    old_defs = _top((B65_PKG / "sediment_event.py").read_text())
    new_defs = _top((PRIVATE_PKG / "sediment_event.py").read_text())
    added = {"_PostPickupBed", "_bed_with_transport_water", "_postpickup_release_eligible"}  # correction 2
    assert set(new_defs) == set(old_defs) | added
    for name in set(old_defs) - {"_sediment_coupled_core"}:
        assert ast.dump(old_defs[name]) == ast.dump(new_defs[name]), name
    core_old, core_new = ast.unparse(old_defs["_sediment_coupled_core"]), ast.unparse(new_defs["_sediment_coupled_core"])
    for line in ("bed, pickup = apply_bed_demand(bed, context, WaterProcessDemand(requested_pickup, zeros))",
                 "transport = transport_step(state.network, bed.water.mobile_mass_by_cell_class_kg, velocity, "
                 "deposition_rate, settle_mask, dt, courant_max=control.sediment_courant_max, n_substeps=n_sub)",
                 "bed = with_water(bed, context, route.depth_m, transport.mobile_after_transfer_kg)",
                 "bed, deposit = apply_bed_demand(bed, context, water_demand_from_transport(transport, zeros))",
                 "phase, reconciliation = reconcile_phase(characteristic, bed.water.mobile_mass_by_cell_class_kg, xp)",
                 "bed, deposit = apply_bed_demand(bed, context, demand)"):
        assert line in core_old and line in core_new, line
    # the change is exactly: the record / release after the pickup and the rebuild branch before the deposit demand
    old_lines = [ln.strip() for ln in core_old.splitlines()]
    new_lines = [ln.strip() for ln in core_new.splitlines()]
    import difflib

    delta = [d for d in difflib.ndiff(old_lines, new_lines) if d[:1] in "+-"]
    assert [d for d in delta if d.startswith("-")] == []
    assert [d[2:] for d in delta] == [
        "post_pickup = None",
        "if not full and (control.csr or control.characteristic) and _postpickup_release_eligible(bed):",
        "post_pickup = _PostPickupBed.of(bed)",
        "del bed",
        "if post_pickup is None:",
        "else:",
        "bed = _bed_with_transport_water(post_pickup, context, route.depth_m, transport.mobile_after_transfer_kg)",
        "del post_pickup",
    ], delta


# =============================================================================================================
# Differentials against the isolated B65 module
# =============================================================================================================
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ALL_CASES)
@pytest.mark.parametrize("form", ["accounting", "full"])
@pytest.mark.parametrize("start", STARTS)
def test_step_raw_identical_to_the_b65_oracle(old, backend, case, form, start, tmp_path):
    xp = b20.backend_xp(backend)
    s, ctl, args = case_args(case, start, xp, tmp_path)
    before = [v.tobytes() for v in b20.leaves(args[0])]
    got = call(new, form, args, ctl, old)
    ref = call(old, form, args, ctl, old)
    assert [v.tobytes() for v in b20.leaves(args[0])] == before  # caller unchanged
    assert type(ref).__module__ == old.__name__ and type(got).__module__ == new.__name__  # two implementations
    assert record(got)["arrays"] == record(ref)["arrays"]
    assert record(got) == record(ref)
    if start == "advanced":  # a genuine NON-ZERO actual MAPLE pickup on a non-zero prior pool
        assert host(actual_pickup(got)).any() and host(args[0].bed.water.mobile_mass_by_cell_class_kg).any()


# =============================================================================================================
# Refusals, retries and atomicity
# =============================================================================================================
def _deposit_call(module, monkeypatch, action):
    real, calls = module.apply_bed_demand, [0]

    def apply(bed, context, demand):
        calls[0] += 1
        new_bed, result = real(bed, context, demand)
        if calls[0] == 2:
            return action(new_bed, result)
        return new_bed, result

    monkeypatch.setattr(module, "apply_bed_demand", apply)


def _patch_kernels(module, monkeypatch, wrap):
    for name in KERNELS:
        monkeypatch.setattr(module, name, wrap(getattr(module, name)))


def _poison_t(real):
    """The kernel's actual result with ONE element of T set to NaN (the new water must be refused)."""
    def kernel(*a, **k):
        out = real(*a, **k)
        transport = out.transport if hasattr(out, "transport") else out
        bad = transport.mobile_after_transfer_kg.copy()
        bad.reshape(-1)[0] = float("nan")
        poisoned = dataclasses.replace(transport, mobile_after_transfer_kg=bad)
        return dataclasses.replace(out, transport=poisoned) if hasattr(out, "transport") else poisoned
    return kernel


def _reject_once(real, calls):
    from maple_syrup.sediment_transport import TransportStepRejected

    def kernel(*a, **k):
        calls[0] += 1
        if calls[0] == 1:
            raise TransportStepRejected("injected single transport rejection")
        return real(*a, **k)
    return kernel


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ALL_CASES)
@pytest.mark.parametrize("kind", ["deposit_refusal", "reconciliation_refusal", "transport_rejection",
                                  "new_water_validation", "retry_then_accept"])
def test_refusals_retries_identical_and_caller_unchanged(old, backend, case, kind, tmp_path, monkeypatch):
    from maple.core.types.water import WaterState

    from maple_syrup.sediment_transport import TransportStepRejected

    xp = b20.backend_xp(backend)
    outcomes = {}
    # ONE actual case for both implementations (the step is pure; a store directory is part of the exact record)
    s, ctl, args = case_args(case, "advanced", xp, tmp_path)
    for label, module in (("new", new), ("old", old)):
        before = [v.tobytes() for v in b20.leaves(args[0])]
        pins = b20.store_pins(tmp_path)
        with monkeypatch.context() as m:
            calls = [0]
            if kind == "deposit_refusal":
                def refuse(bed, result, module=module):
                    raise module.SedimentEventError("injected deposit-call refusal")

                _deposit_call(module, m, refuse)
            elif kind == "reconciliation_refusal":
                def tamper(bed, result):
                    pool = bed.water.mobile_mass_by_cell_class_kg + 1e-3
                    return dataclasses.replace(bed, water=WaterState(bed.water.depth_m, pool)), result

                _deposit_call(module, m, tamper)
            elif kind == "transport_rejection":
                def reject_all(real):
                    def kernel(*a, **k):
                        raise TransportStepRejected("injected transport rejection")
                    return kernel

                _patch_kernels(module, m, reject_all)
            elif kind == "new_water_validation":
                _patch_kernels(module, m, _poison_t)
            else:
                _patch_kernels(module, m, lambda real, calls=calls: _reject_once(real, calls))
            accepts = kind == "retry_then_accept" or (case == "upwind" and kind == "reconciliation_refusal")
            if accepts:
                step = call(module, "accounting", args, ctl, old)
                outcomes[label] = ("ok", record(step), step.n_transport_rejections)
                del step
            else:
                with pytest.raises(Exception) as info:
                    call(module, "accounting", args, ctl, old)
                outcomes[label] = (type(info.value).__name__, str(info.value))
        assert [v.tobytes() for v in b20.leaves(args[0])] == before
        assert b20.store_pins(tmp_path) == pins
    if outcomes["new"][0] == "ok":
        assert outcomes["old"][0] == "ok" and outcomes["new"][1] == outcomes["old"][1]
        assert outcomes["new"][2] == outcomes["old"][2]
        if kind == "retry_then_accept":
            assert outcomes["new"][2] == 1  # the in-core retry actually happened
    else:
        assert outcomes["new"] == outcomes["old"], outcomes
    if kind == "new_water_validation":
        assert outcomes["new"][0] != "ok"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ALL_CASES)
def test_retry_exhaustion_through_outer_dt_halving_identical(old, backend, case, tmp_path, monkeypatch):
    """Every transport attempt rejected: the in-core loop exhausts its substeps / the outer loop halves dt until its
    retry floor, then refuses -- identically in both implementations; caller and deep store unchanged."""
    from maple_syrup.sediment_transport import TransportStepRejected

    xp = b20.backend_xp(backend)
    outcomes = {}
    for label, module in (("new", new), ("old", old)):
        s, ctx, ctl, schedule = build_case(case, xp, tmp_path / label)
        before = [v.tobytes() for v in b20.leaves(s["state0"])]
        pins = b20.store_pins(tmp_path / label)
        with monkeypatch.context() as m:
            def reject_all(real):
                def kernel(*a, **k):
                    raise TransportStepRejected("injected transport rejection")
                return kernel

            _patch_kernels(module, m, reject_all)
            state = s["state0"] if module is new else as_types(old, s["state0"])
            control = ctl if module is new else as_types(old, ctl)
            with pytest.raises(Exception) as info:
                module.evolve_sediment_event(state, ctx, s["column"], s["field"], schedule, s["vegetation"],
                                             s["sediment"], ADVANCE_S, control, report_every_s=1.0)
            outcomes[label] = (type(info.value).__name__, str(info.value))
        assert [v.tobytes() for v in b20.leaves(s["state0"])] == before
        assert b20.store_pins(tmp_path / label) == pins
    assert outcomes["new"] == outcomes["old"] and outcomes["new"][0] == "SedimentEventError", outcomes
    assert "rejected" in outcomes["new"][1] or "retry floor" in outcomes["new"][1]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
def test_complete_driver_byte_identical_to_the_b65_tree(backend, case, tmp_path):
    """Quiet completion + dry reset, wet pause, checkpoint save/load and fresh-process resumed completion, each tree in
    its own subprocess (the CW-B20 driver)."""
    b20.backend_xp(backend)
    if not B65_ROOT.is_dir():
        pytest.skip("B65 tree unavailable")
    a_arrays, a_counts = b20.run_driver(B65_ROOT, case, backend, tmp_path, "b65")
    b_arrays, b_counts = b20.run_driver(PRIVATE_ROOT, case, backend, tmp_path, "b71")
    assert a_counts.pop("maple_syrup") == str(B65_PKG) and b_counts.pop("maple_syrup") == str(PRIVATE_PKG)
    assert a_counts == b_counts
    assert sorted(a_arrays) == sorted(b_arrays) and len(a_arrays) > 100
    for key in a_arrays:
        assert a_arrays[key].dtype == b_arrays[key].dtype and a_arrays[key].shape == b_arrays[key].shape, key
        assert a_arrays[key].tobytes() == b_arrays[key].tobytes(), key


# =============================================================================================================
# Liveness (CPU NumPy weak references) with an old-source negative control
# =============================================================================================================
NONWATER = tuple(f.name for f in dataclasses.fields(new.BedState) if f.name != "water")


def _probe(module, monkeypatch, refs, seen, record_check=True):
    real_apply, calls = module.apply_bed_demand, [0]

    def apply(bed, context, demand):
        calls[0] += 1
        if calls[0] == 2:  # the deposit call: its input bed
            seen["deposit_fields_same"] = all(getattr(bed, n) is refs[n]() for n in NONWATER)
            seen["deposit_water_is_t"] = bed.water.mobile_mass_by_cell_class_kg is refs["T"]()
            seen["deposit_depth_is_route"] = bed.water.depth_m is refs["post_pickup_depth"]()
            seen["deposit_bed_tree"] = tree(bed, "deposit_bed")
        new_bed, result = real_apply(bed, context, demand)
        if calls[0] == 1:  # the pickup call: the actual post-pickup bed MAPLE returned
            refs["post_pickup_bed"] = weakref.ref(new_bed)
            refs["post_pickup_pool"] = weakref.ref(new_bed.water.mobile_mass_by_cell_class_kg)
            refs["post_pickup_depth"] = weakref.ref(new_bed.water.depth_m)
            for n in NONWATER:
                refs[n] = weakref.ref(getattr(new_bed, n))
            if module is new and record_check:  # the record carries every non-water field and NO water array
                from maple.core.backend import array_leaves

                rec = new._PostPickupBed.of(new_bed)
                seen["record_names"] = tuple(n for n, _v in rec.fields)
                seen["record_same_objects"] = all(v is getattr(new_bed, n) for n, v in rec.fields)
                pool, depth = new_bed.water.mobile_mass_by_cell_class_kg, new_bed.water.depth_m
                seen["record_holds_water"] = any(a is pool or a is depth for a in array_leaves(rec.fields))
                del rec, pool, depth
        return new_bed, result

    monkeypatch.setattr(module, "apply_bed_demand", apply)
    for name in KERNELS:
        real = getattr(module, name)

        def kernel(*a, real=real, **k):
            gc.collect()
            seen["at_transport"] = {n: r() is not None for n, r in refs.items()}
            out = real(*a, **k)
            transport = out.transport if hasattr(out, "transport") else out
            refs["T"] = weakref.ref(transport.mobile_after_transfer_kg)
            return out

        monkeypatch.setattr(module, name, kernel)


@pytest.mark.parametrize("case", ALL_CASES)
@pytest.mark.parametrize("start", STARTS)
def test_postpickup_pool_dead_at_transport_with_old_source_negative_control(old, case, start, tmp_path, monkeypatch):
    s, ctl, args = case_args(case, start, np, tmp_path)
    results = {}
    for label, module in (("new", new), ("old", old)):
        refs, seen = {}, {}
        with monkeypatch.context() as m:
            _probe(module, m, refs, seen)
            attempt = call(module, "accounting", args, ctl, old)
        gc.collect()
        seen["at_return"] = {n: r() is not None for n, r in refs.items()}
        seen["record"] = record(attempt)
        results[label] = seen
        del attempt
    n, o = results["new"], results["old"]
    characteristic = case != "upwind"
    if characteristic:
        # B71: the post-pickup pool and the post-pickup BedState are DEAD at transport entry; the non-water fields
        # (the record's same objects) are alive
        assert not n["at_transport"]["post_pickup_pool"] and not n["at_transport"]["post_pickup_bed"]
        assert all(n["at_transport"][f] for f in NONWATER)
        assert n["record_names"] == NONWATER and n["record_same_objects"] and not n["record_holds_water"]
    else:  # upwind: the post-pickup pool IS the transport input -> unchanged, alive
        assert n["at_transport"]["post_pickup_pool"] and n["at_transport"]["post_pickup_bed"]
    # old-source negative control: B65 keeps the post-pickup bed and its pool through the transport
    assert o["at_transport"]["post_pickup_pool"] and o["at_transport"]["post_pickup_bed"]
    # the deposit call receives the post-pickup non-water objects, T and the route depth -- in both
    for seen in (n, o):
        assert seen["deposit_fields_same"] and seen["deposit_water_is_t"] and seen["deposit_depth_is_route"]
    assert n["deposit_bed_tree"] == o["deposit_bed_tree"]
    # at return: the post-pickup pool / bed are dead in both; results identical
    for seen in (n, o):
        assert not seen["at_return"]["post_pickup_pool"] and not seen["at_return"]["post_pickup_bed"]
    assert n["record"] == o["record"]


# =============================================================================================================
# The private record / rebuild against `with_water`
# =============================================================================================================
def _post_pickup(case, tmp_path):
    """An ACTUAL post-pickup bed: the advanced state's bed through `with_water` + the MAPLE pickup call."""
    from maple.water import WaterProcessDemand

    s, ctl, args = case_args(case, "advanced", np, tmp_path)
    state, ctx = args[0], args[1]
    pool = state.bed.water.mobile_mass_by_cell_class_kg
    demand = WaterProcessDemand(pool * 0.5 + 1e-9, np.zeros_like(pool))
    bed, _result = new.apply_bed_demand(new.with_water(state.bed, ctx, state.bed.water.depth_m), ctx, demand)
    return bed, ctx, state.bed.water.depth_m, state.bed.water.mobile_mass_by_cell_class_kg.copy()


@pytest.mark.parametrize("case", CASES)
def test_rebuild_equals_with_water_and_refuses_identically(case, tmp_path):
    bed, ctx, depth, t = _post_pickup(case, tmp_path)
    ref = new.with_water(bed, ctx, depth, t)
    got = new._bed_with_transport_water(new._PostPickupBed.of(bed), ctx, depth, t)
    assert type(got) is new.BedState
    assert all(getattr(got, n) is getattr(bed, n) for n in NONWATER)
    assert got.water.depth_m is depth and got.water.mobile_mass_by_cell_class_kg is t
    assert tree(got) == tree(ref)
    bad_nan = t.copy()
    bad_nan.reshape(-1)[0] = np.nan
    bad_neg = t.copy()
    bad_neg.reshape(-1)[0] = -1e-30
    for bad in (bad_nan, bad_neg, t.astype(np.float32), t[..., :-1].copy(), depth.astype(np.float32)):
        is_depth = bad.ndim == depth.ndim
        args_ref = (depth if not is_depth else bad, t if is_depth else bad)
        outcomes = []
        for fn, first in ((new.with_water, bed), (new._bed_with_transport_water, new._PostPickupBed.of(bed))):
            with pytest.raises(Exception) as info:
                fn(first, ctx, *args_ref)
            outcomes.append((type(info.value).__name__, str(info.value)))
        assert outcomes[0] == outcomes[1], outcomes


def test_private_record_guards_and_carries_every_nonwater_field(tmp_path, monkeypatch):
    """The PRIVATE record (original B71): exact `BedState` only -- the core never invokes it for any other bed (see
    `_postpickup_release_eligible`); a `BedState` that gains a field is carried by name; a field-set drift refuses."""
    bed, ctx, depth, t = _post_pickup("dense", tmp_path)
    sub = type("PrivateSubBed", (new.BedState,), {})
    with pytest.raises(new.SedimentEventError, match="actual BedState"):
        new._PostPickupBed.of(sub(**{f.name: getattr(bed, f.name) for f in dataclasses.fields(bed)}))
    rec = new._PostPickupBed.of(bed)
    extended = dataclasses.make_dataclass("BedState", [(f.name, object) for f in dataclasses.fields(new.BedState)]
                                          + [("extra_field", object)], frozen=True)
    marker = object()
    monkeypatch.setattr(new, "BedState", extended)
    ext = extended(**{f.name: getattr(bed, f.name) for f in dataclasses.fields(bed)}, extra_field=marker)
    ext_rec = new._PostPickupBed.of(ext)
    assert dict(ext_rec.fields)["extra_field"] is marker and "water" not in dict(ext_rec.fields)
    rebuilt = new._bed_with_transport_water(ext_rec, ctx, depth, t)
    assert type(rebuilt) is extended and rebuilt.extra_field is marker
    with pytest.raises(new.SedimentEventError, match="fields changed"):
        new._bed_with_transport_water(rec, ctx, depth, t)


# --- CW-B71 correction 2: eligibility rule; valid subclasses take the ORIGINAL path through the actual core ----------
class SubBed(new.BedState):
    """A plain (non-dataclass-decorated) subclass of the actual BedState: same fields, no additions."""


@dataclasses.dataclass(frozen=True, eq=False)
class _Marker:
    label: str


@dataclasses.dataclass(frozen=True)
class ExtBed(new.BedState):
    """A frozen dataclass subtype: extra scalar, extra object field, an `init=False` derived scalar and an `init=False`
    derived ARRAY (a namespace operand of the original `with_water` walk), recomputed by `__post_init__`, which also
    validates the scalar and counts its calls."""

    tag: str = "ext"
    marker: object = None
    derived: int = dataclasses.field(init=False, default=-1)
    derived_arr: object = dataclasses.field(init=False, default=None)
    post_init_calls: ClassVar[list] = []

    def __post_init__(self):
        if not self.tag:
            raise ValueError("ExtBed tag must be non-empty")
        object.__setattr__(self, "derived", len(self.tag))
        object.__setattr__(self, "derived_arr", np.full(2, float(len(self.tag))))
        ExtBed.post_init_calls.append(self.tag)


def _as(bed_type, bed, **extra):
    return bed_type(**{f.name: getattr(bed, f.name) for f in dataclasses.fields(new.BedState)}, **extra)


SUBTYPES = {"plain_subclass": lambda bed: _as(SubBed, bed),
            "extended": lambda bed: _as(ExtBed, bed, tag="abcd", marker=_Marker("extra object field"))}


def test_eligibility_rule():
    bed_fields = dataclasses.fields(new.BedState)
    assert len(bed_fields) == len(new.BedState.__dataclass_fields__) and all(f.init for f in bed_fields)
    plain = new.BedState(**{f.name: None for f in bed_fields})
    assert new._postpickup_release_eligible(plain)
    assert not new._postpickup_release_eligible(SubBed(**{f.name: None for f in bed_fields}))
    assert not new._postpickup_release_eligible(object())
    assert not new._postpickup_release_eligible(new.BedState)


def _spy_namespace(monkeypatch, walks):
    """Record, per `resolve_operand_namespace` call, the (dtype, shape) sequence of the array leaves it visits (the
    ACTUAL M67 walk), so the fallback's namespace operands are compared with the original path's."""
    import maple.core.backend as backend
    from maple.core.backend import array_leaves

    real = backend.resolve_operand_namespace

    def spy(*operands):
        walks.append(tuple((str(np.asarray(host(a)).dtype), np.asarray(host(a)).shape) for a in array_leaves(*operands)))
        return real(*operands)

    monkeypatch.setattr(backend, "resolve_operand_namespace", spy)


@pytest.mark.parametrize("subtype", sorted(SUBTYPES))
@pytest.mark.parametrize("case", ALL_CASES)
@pytest.mark.parametrize("form", ["accounting", "full"])
def test_valid_subtypes_take_the_original_path_through_the_core(old, subtype, case, form, tmp_path, monkeypatch):
    """Coupled-core old/new: a valid plain subclass and an extended subtype (extra scalar / object, init=False derived
    scalar and array) are accepted by BOTH implementations through the ORIGINAL `with_water` rebuild: results raw- and
    recursively identical, `result.bed` keeps its type, `__post_init__` runs the same number of times, the private
    record / rebuild is never invoked, and the actual namespace walks (incl. the init=False array) are identical."""
    s, ctl, args = case_args(case, "advanced", np, tmp_path)
    state = dataclasses.replace(args[0], bed=SUBTYPES[subtype](args[0].bed))
    sub_args = (state, *args[1:])
    before = [v.tobytes() for v in b20.leaves(state)]
    outcomes = {}
    for label, module in (("new", new), ("old", old)):
        walks, private = [], []
        with monkeypatch.context() as m:
            _spy_namespace(m, walks)
            if module is new:
                for name in ("_PostPickupBed", "_bed_with_transport_water"):
                    real = getattr(new, name)
                    m.setattr(new, name, types.SimpleNamespace(of=lambda *a, n=name: private.append(n))
                              if name == "_PostPickupBed" else (lambda *a, n=name, **k: private.append(n)))
            ExtBed.post_init_calls.clear()
            step = call(module, form, sub_args, ctl, old)
            outcomes[label] = (record(step), type(step.bed), list(ExtBed.post_init_calls), walks, private)
            del step
    assert [v.tobytes() for v in b20.leaves(state)] == before
    n, o = outcomes["new"], outcomes["old"]
    assert n[0] == o[0] and n[1] is o[1] is type(state.bed)
    assert n[2] == o[2]
    if subtype == "extended":
        assert n[2] and all(tag == "abcd" for tag in n[2])  # constructor + __post_init__ actually ran
    assert n[3] == o[3] and n[3]  # identical actual namespace walks
    if subtype == "extended":  # the init=False derived array is a visited namespace operand
        assert any(("float64", (2,)) in walk for walk in n[3])
    assert n[4] == []  # the private record / rebuild was never invoked for a non-base bed


@pytest.mark.parametrize("subtype", sorted(SUBTYPES))
@pytest.mark.parametrize("case", CASES)
def test_subtype_fallback_retains_the_original_bed_at_transport(old, subtype, case, tmp_path, monkeypatch):
    """Liveness: on the fallback the post-pickup bed and its pool stay ALIVE at transport entry (original behaviour,
    like B65); only the exact base BedState releases them (covered by the base liveness test)."""
    s, ctl, args = case_args(case, "advanced", np, tmp_path)
    state = dataclasses.replace(args[0], bed=SUBTYPES[subtype](args[0].bed))
    sub_args = (state, *args[1:])
    for label, module in (("new", new), ("old", old)):
        refs, seen = {}, {}
        with monkeypatch.context() as m:
            _probe(module, m, refs, seen, record_check=False)
            attempt = call(module, "accounting", sub_args, ctl, old)
        assert seen["at_transport"]["post_pickup_pool"] and seen["at_transport"]["post_pickup_bed"], label
        assert type(attempt.bed) is type(state.bed)
        del attempt


# =============================================================================================================
# Device: exact MemoryHook live-byte difference (root)
# =============================================================================================================
class _Hook:
    def __init__(self):
        self.live, self.current = {}, 0

    def hook(self):
        import cupy as cp

        tracker = self

        class Hook(cp.cuda.MemoryHook):
            name = "CWB71LiveBytes"

            def malloc_postprocess(self, **kw):
                if kw.get("mem_ptr"):
                    key = (int(kw["device_id"]), int(kw["pmem_id"]))
                    tracker.current += int(kw["mem_size"]) - tracker.live.get(key, 0)
                    tracker.live[key] = int(kw["mem_size"])

            def free_postprocess(self, **kw):
                key = (int(kw["device_id"]), int(kw["pmem_id"]))
                tracker.current -= tracker.live.pop(key, 0)

        return Hook()


def _allocations(*objs):
    from maple.core.backend import array_leaves

    out = {}
    for obj in objs:
        for a in array_leaves(obj):
            mem = getattr(getattr(a, "data", None), "mem", None)
            if mem is not None and a.size:
                out[int(mem.ptr)] = int(mem.size)
    return out


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("start", STARTS)
def test_device_live_bytes_released_at_transport_and_return(old, case, start, tmp_path, monkeypatch):
    """B65 - B71 live bytes (MemoryHook): at transport entry EXACTLY the pool size of the ONE identified post-pickup
    mobile allocation (pointer from B65's actual post-pickup bed, verified not aliased by any array the B71 frame still
    holds: the caller state, the non-water bed fields), and ZERO at return (both returned results kept alive; every
    observational reference released before the measurement). Each implementation is warmed up first (unhooked)."""
    xp = b20.backend_xp("cupy")
    s, ctl, args = case_args(case, start, xp, tmp_path)
    live, ident, kept = {}, {}, {}
    for label, module in (("new", new), ("old", old)):
        state = args[0] if module is new else as_types(old, args[0])
        control = ctl if module is new else as_types(old, ctl)
        module._sediment_accounting_step(state, *args[1:], control)  # warm-up, outside every measurement
        gc.collect()
        tracker, marks = _Hook(), {}
        with monkeypatch.context() as m:
            real_apply, calls = module.apply_bed_demand, [0]

            def apply(bed, context, demand, real_apply=real_apply, calls=calls, marks=marks):
                calls[0] += 1
                new_bed, result = real_apply(bed, context, demand)
                if calls[0] == 1:  # ints only: no owner retained
                    pool = new_bed.water.mobile_mass_by_cell_class_kg
                    marks["pool"] = (int(pool.data.mem.ptr), int(pool.data.mem.size))
                    marks["nonwater"] = set(_allocations(*(getattr(new_bed, n) for n in NONWATER)))
                    del pool
                return new_bed, result

            m.setattr(module, "apply_bed_demand", apply)
            for name in KERNELS:
                real = getattr(module, name)

                def kernel(*a, real=real, marks=marks, tracker=tracker, **k):
                    xp.cuda.Device().synchronize()
                    gc.collect()
                    marks.setdefault("transport", tracker.current)
                    return real(*a, **k)

                m.setattr(module, name, kernel)
            with tracker.hook():
                kept[label] = module._sediment_accounting_step(state, *args[1:], control)
                gc.collect()
                xp.cuda.Device().synchronize()
                marks["return"] = tracker.current
        live[label] = marks
    ptr, size = live["old"]["pool"]
    assert ptr not in set(_allocations(args[0])) and ptr not in live["old"]["nonwater"]  # alias-aware: not double counted
    got = {k: live["old"][k] - live["new"][k] for k in ("transport", "return")}
    want = {"transport": size, "return": 0}
    print(f"CW-B71 {case}/{start}: live-byte differences {got}, expected {want} (post-pickup pool {ptr:#x})")
    assert got == want, (got, want)
    assert record(kept["new"]) == record(kept["old"])
