"""CW-B65: accounting-only post-transport retention in `sediment_event._sediment_coupled_core`.

Oracle: the unchanged B63 `sediment_event.py` (hard SHA `96ecaff2...`), executed as an ISOLATED module in the current
context with its OWN `SedimentEventState` / `SedimentEventControl` types (rebuilt from the same fields), so every
differential compares two actual implementations. Every other package file must equal the B63 parent tree.

CPU (author, NumPy): pins and scope; untouched public text; accounting and full steps raw-identical to the oracle on
the dense / resident-window CSR / resident-window dense cases, caller and deep-store immutability; the COMPLETE driver
(quiet completion + dry reset, wet pause, checkpoint, resumed completion) of this tree byte-identical to the B63 tree's
in subprocesses; identical refusals (transport rejection, deposit-call refusal, reconciliation bound refusal); weak-
reference liveness with an old-source negative control: the complete kernel result, its transport step and the 8
unused full-domain diagnostics are dead at the deposit call while the four faces are alive and reach MAPLE's validated
deposit result; the faces and the complete deposit result are dead at reconciliation; nothing extra at return.
DEVICE (root, GPU0; skipped here, a skip claims nothing): the same differentials on CuPy and an exact MemoryHook
live-byte difference at the deposit and reconciliation entries, derived from the actual allocations.
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
from pathlib import Path, PurePath

import numpy as np
import pytest
import test_accounting_step as b20  # the existing CW-B20 accounting-step harness of THIS tree
from b20_cases import CASES, build, host

import maple_syrup.sediment_event as new

PRIVATE_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_PKG = PRIVATE_ROOT / "src" / "maple_syrup"
B63_ROOT = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b63_cuda_outlet_flag_lifetime")
B63_PKG = B63_ROOT / "src" / "maple_syrup"
ORACLE_SHA256 = "96ecaff22231c041a0d2e350f315a55745fc7a46dc094aef829cc647cb6a5118"
CUDA_SHA256 = "d13f4bf5d2642f65f823514b2ea09ea0ddb1c0275aedafb57596adfbb372d5b7"
BACKENDS = ["numpy", "cupy"]  # cupy skips without an actual device (no GPU claim)
DIAGNOSTICS = (("transport", "decay_deposition_kg"), ("transport", "settled_kg"),
               ("transport", "internal_transfer_in_kg"), ("transport", "internal_transfer_out_kg"),
               ("transport", "divergence_kg"), ("step", "mobile_before_kg"), ("step", "arrival_deposition_kg"),
               ("step", "crossing_kg"))
FACES = ("x_face_gross_kg", "y_face_gross_kg", "x_face_net_kg", "y_face_net_kg")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def old():
    path = B63_PKG / "sediment_event.py"
    if not path.is_file():
        pytest.skip("B63 oracle unavailable; no before/after claim")
    if _sha(path) != ORACLE_SHA256:
        pytest.fail("B63 sediment_event.py no longer has its pinned hash; the oracle comparison would be invalid")
    import sys

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


ORACLE_MODULE = "_cw_b65_b63_sediment_event"
_EVENT_MODULES = {new.__name__, ORACLE_MODULE}  # the two isolated, equivalent implementations of one module
# Declared non-state slots: `DeepStoreVersion`'s per-handle LRU of memmap references and its lock ("two handles of one
# version may cache differently and agree on every byte"); its content identity is the manifest digest, recorded.
_CACHE_SLOTS = {"maple.surface.voxels.deep_store.DeepStoreVersion": {"_memmaps", "_memmap_lock"}}


def _type_name(t):
    """Module-qualified type name; the logical module name ONLY for the old/new `sediment_event` dataclasses."""
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
    """Exact recursive record: dataclass fields (logical type name), dicts (keys and insertion order), tuples/lists,
    MAPLE's opaque slot handles (declared slots), arrays (backend, dtype, shape, raw bytes), NumPy scalars (dtype, raw
    bytes), native floats (raw IEEE-754 bytes, so -0.0 != 0.0), native bool/int/str/None, paths, and backend MODULES
    by name (classified before arrays). Any other type refuses, so nothing is silently skipped; no repr is used."""
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
    """Raw record of a step result (accounting or full): every array leaf (the original raw-array comparison) and the
    complete recursive `tree`, which compares every nested scalar (reconciliation counters, regime counts, route
    diagnostics, ...) exactly."""
    return {"type": _type_name(type(step)), "arrays": leaves(step), "tree": tree(step)}


ALL_CASES = (*CASES, "upwind")


def build_case(case, xp, tmp_path):
    """The CW-B20 cases plus `upwind`: the same wet 3-cell chain with the upwind (`transport_step`) scheme, i.e. the
    accounting path WITHOUT a characteristic kernel result (no reconciliation)."""
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


def run_both(old, backend, case, tmp_path, form):
    xp = b20.backend_xp(backend)
    s, _ctx, ctl, schedule = build_case(case, xp, tmp_path)
    args = b20.step_args(s, schedule)
    before = b20.leaves(s["state0"])
    fn_new = new._sediment_accounting_step if form == "accounting" else new.sediment_coupled_step
    fn_old = old._sediment_accounting_step if form == "accounting" else old.sediment_coupled_step
    got = fn_new(*args, ctl)
    ref = fn_old(as_types(old, args[0]), *args[1:], as_types(old, ctl))
    after = b20.leaves(s["state0"])
    assert [v.tobytes() for v in after] == [v.tobytes() for v in before]  # caller unchanged
    return got, ref


# =============================================================================================================
# CPU: pins, scope, untouched public text
# =============================================================================================================
def test_pins_and_all_other_package_files_equal_the_b63_parent():
    assert Path(new.__file__).resolve() == PRIVATE_PKG / "sediment_event.py"
    private = {str(p.relative_to(PRIVATE_PKG)): _sha(p) for p in PRIVATE_PKG.rglob("*.py") if "__pycache__" not in p.parts}
    parent = {str(p.relative_to(B63_PKG)): _sha(p) for p in B63_PKG.rglob("*.py") if "__pycache__" not in p.parts}
    assert set(private) == set(parent) and len(parent) == 55
    assert sorted(k for k in private if private[k] != parent[k]) == ["sediment_event.py"]
    assert parent["sediment_event.py"] == ORACLE_SHA256 and private["characteristic_cuda_csr.py"] == CUDA_SHA256


def _top(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_only_the_reconciliation_and_core_definitions_changed():
    old_defs = _top((B63_PKG / "sediment_event.py").read_text())
    new_defs = _top((PRIVATE_PKG / "sediment_event.py").read_text())
    changed = {"_reconciliation_bound_check", "_reconcile_sparse_phase", "reconcile_phase", "_sediment_coupled_core"}
    added = {"_PhaseReconcileInputs", "_reconcile_phase_inputs", "_reconcile_dense_phase"}
    assert set(new_defs) == set(old_defs) | added
    for name in set(old_defs) - changed:
        assert ast.dump(old_defs[name]) == ast.dump(new_defs[name]), name
    # the public reconcile_phase keeps its signature, docstring, type guards and messages
    o, n = old_defs["reconcile_phase"], new_defs["reconcile_phase"]
    assert ast.dump(o.args) == ast.dump(n.args) and ast.dump(o.returns) == ast.dump(n.returns)
    assert ast.get_docstring(o) == ast.get_docstring(n)
    guards = [ast.unparse(s.test) for s in n.body if isinstance(s, ast.If)]
    assert guards == ["isinstance(step, SparseCharacteristicStep)", "not isinstance(step, CharacteristicStep)"]
    assert ast.unparse(n.body[2].body[0]) == ast.unparse(o.body[3].body[0])  # the identical raise
    # the full branch of the core is the original statement sequence
    core_old, core_new = ast.unparse(old_defs["_sediment_coupled_core"]), ast.unparse(new_defs["_sediment_coupled_core"])
    for line in ("bed = with_water(bed, context, route.depth_m, transport.mobile_after_transfer_kg)",
                 "bed, deposit = apply_bed_demand(bed, context, water_demand_from_transport(transport, zeros))",
                 "phase, reconciliation = reconcile_phase(characteristic, bed.water.mobile_mass_by_cell_class_kg, xp)"):
        assert line in core_old and line in core_new, line


# =============================================================================================================
# Differentials against the isolated B63 module (CPU NumPy; CuPy for root)
# =============================================================================================================
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ALL_CASES)
@pytest.mark.parametrize("form", ["accounting", "full"])
def test_step_raw_identical_to_the_b63_oracle(old, backend, case, form, tmp_path):
    got, ref = run_both(old, backend, case, tmp_path, form)
    assert type(ref).__module__ == old.__name__ and type(got).__module__ == new.__name__  # two implementations
    assert record(got)["arrays"] == record(ref)["arrays"]
    assert {k: v for k, v in record(got).items() if k not in ("arrays",)} == \
           {k: v for k, v in record(ref).items() if k not in ("arrays",)}
    if form == "accounting":
        for name in new._TRANSPORT_ACCOUNTING_FIELDS:
            assert b20.same(getattr(got.transport, name), getattr(ref.transport, name)), name
        for name in new._DEPOSIT_ACCOUNTING_FIELDS:
            assert b20.same(getattr(got.deposit, name), getattr(ref.deposit, name)), name


def _bump_counter(mapping, key):
    out = dict(mapping)
    out[key] = out[key] + 1
    return out


def _one_ulp(array):
    """A copy of `array` with ONE element moved by one ULP (the largest element, so it is non-zero)."""
    import importlib

    out = array.copy()
    idx = tuple(int(i) for i in np.unravel_index(int(np.argmax(np.asarray(host(array)))), array.shape))
    module = np if isinstance(array, np.ndarray) else importlib.import_module("cupy")
    out[idx] = module.nextafter(out[idx], module.inf)
    return out


def _counter_kind(value):
    """The ACTUAL representation of a diagnostic counter."""
    if isinstance(value, np.generic):
        return "numpy-scalar"
    if _is_array(value) and not isinstance(value, np.ndarray) and value.ndim == 0:
        return "cupy-0d"
    raise AssertionError(f"unexpected counter representation {_type_name(type(value))}")


def _expected_counters(backend, case):
    """{counter: expected representation}, from the source, per backend and case (no blanket backend waiver):
      physics.regime_counts["diffuse"]           `xp.sum(...)` (sediment_physics.py) -> a NumPy scalar on NumPy, a 0-D
                                                 CuPy array on CuPy;
      phase_reconciliation["n_canonicalized"]    dense reconciliation `xp.sum(...)` -> as above; SPARSE reconciliation
                                                 (window_csr) `np.int64(...)` after its one counted host read -> a NumPy
                                                 scalar on both backends; upwind reconciles no phase (no counter)."""
    device = "cupy-0d" if backend == "cupy" else "numpy-scalar"
    out = {("physics", "regime_counts", "diffuse"): device}
    if case != "upwind":
        out[(None, "phase_reconciliation", "n_canonicalized")] = "numpy-scalar" if case == "window_csr" else device
    return out


def _with_counter(step, owner, field, key):
    if owner is None:
        return dataclasses.replace(step, **{field: _bump_counter(getattr(step, field), key)})
    inner = getattr(step, owner)
    return dataclasses.replace(step, **{owner: dataclasses.replace(inner, **{field: _bump_counter(getattr(inner, field),
                                                                                                    key)})})


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ALL_CASES)
@pytest.mark.parametrize("form", ["accounting", "full"])
def test_record_refuses_one_nested_scalar_and_one_array_element(old, backend, case, form, tmp_path):
    """Negative controls on the comparison itself: the oracle-identical step with (a) ONE nested diagnostic counter
    incremented (the physics regime count; the phase-reconciliation counter where a phase is reconciled), each
    separately, and (b) ONE element of a nested array (the post-deposit bed's mobile pool) moved by one ULP. Each
    perturbed step must be refused by the complete recursive record, while the unperturbed step still matches.
    The raw ARRAY-LEAF relation follows the counter's actual representation (asserted against `_expected_counters`):
    a NumPy scalar is no array leaf, so the leaves stay identical; a 0-D CuPy counter IS an array leaf, so exactly
    that one leaf changes (int64 bytes n -> n + 1) and every other leaf stays identical."""
    from maple.core.types.water import WaterState

    got, ref = run_both(old, backend, case, tmp_path, form)
    assert record(got) == record(ref)
    assert bool(got.phase_reconciliation) == (case != "upwind")
    ref_leaves = record(ref)["arrays"]
    for (owner, field, key), kind in _expected_counters(backend, case).items():
        holder = got if owner is None else getattr(got, owner)
        value = getattr(holder, field)[key]
        assert _counter_kind(value) == kind, (owner, field, key)
        before = np.asarray(host(value)).tobytes()
        after = np.asarray(host(value + 1)).tobytes()
        step = _with_counter(got, owner, field, key)
        assert _counter_kind(getattr(step if owner is None else getattr(step, owner), field)[key]) == kind
        assert record(step)["tree"] != record(ref)["tree"] and record(step) != record(ref)
        step_leaves = record(step)["arrays"]
        if kind == "numpy-scalar":
            assert step_leaves == ref_leaves, (field, key)
        else:  # exactly ONE leaf differs: the counter itself
            assert len(step_leaves) == len(ref_leaves)
            changed = [i for i, (a, b) in enumerate(zip(step_leaves, ref_leaves, strict=True)) if a != b]
            assert len(changed) == 1, (field, key, changed)
            assert ref_leaves[changed[0]] == ("int64", (), before)
            assert step_leaves[changed[0]] == ("int64", (), after)
    water = got.bed.water
    pool = water.mobile_mass_by_cell_class_kg
    step = dataclasses.replace(got, bed=dataclasses.replace(got.bed, water=WaterState(water.depth_m, _one_ulp(pool))))
    assert record(step)["arrays"] != record(ref)["arrays"] and record(step)["tree"] != record(ref)["tree"]
    assert record(got) == record(ref)  # the controls copied; the actual step is untouched


def test_tree_raw_scalar_representation():
    assert tree(0.0) != tree(-0.0) and tree(np.float64(0.0)) != tree(np.float64(-0.0))
    assert tree(np.int64(1)) != tree(np.int32(1)) != tree(1) and tree(True) != tree(1)
    assert tree(np) == ("module", "numpy")  # a backend module is a module, never an array
    with pytest.raises(AssertionError, match="unrecorded type"):
        tree(object())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
def test_complete_driver_byte_identical_to_the_b63_tree(backend, case, tmp_path):
    """Quiet completion + dry reset, wet pause, checkpoint save/load and resumed completion, each tree in its own
    subprocess (the CW-B20 driver)."""
    b20.backend_xp(backend)
    if not B63_ROOT.is_dir():
        pytest.skip("B63 tree unavailable")
    a_arrays, a_counts = b20.run_driver(B63_ROOT, case, backend, tmp_path, "b63")
    b_arrays, b_counts = b20.run_driver(PRIVATE_ROOT, case, backend, tmp_path, "b65")
    assert a_counts.pop("maple_syrup") == str(B63_PKG) and b_counts.pop("maple_syrup") == str(PRIVATE_PKG)
    assert a_counts == b_counts
    assert sorted(a_arrays) == sorted(b_arrays)
    for key in a_arrays:
        assert a_arrays[key].dtype == b_arrays[key].dtype and a_arrays[key].shape == b_arrays[key].shape, key
        assert a_arrays[key].tobytes() == b_arrays[key].tobytes(), key


def _deposit_call(module, monkeypatch, action):
    """Intercept `module.apply_bed_demand`; `action(bed, result)` runs on the SECOND (deposit) call."""
    real, calls = module.apply_bed_demand, [0]

    def apply(bed, context, demand):
        calls[0] += 1
        new_bed, result = real(bed, context, demand)
        if calls[0] == 2:
            return action(new_bed, result)
        return new_bed, result

    monkeypatch.setattr(module, "apply_bed_demand", apply)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ALL_CASES)
@pytest.mark.parametrize("kind", ["deposit_refusal", "reconciliation_refusal", "transport_rejection"])
def test_refusals_identical_and_caller_unchanged(old, backend, case, kind, tmp_path, monkeypatch):
    from maple.core.types.water import WaterState

    from maple_syrup.sediment_transport import TransportStepRejected

    xp = b20.backend_xp(backend)
    outcomes = {}
    for label, module in (("new", new), ("old", old)):
        s, _ctx, ctl, schedule = build_case(case, xp, tmp_path / label)
        args = b20.step_args(s, schedule)
        before = [v.tobytes() for v in b20.leaves(s["state0"])]
        pins = b20.store_pins(tmp_path / label)
        with monkeypatch.context() as m:
            if kind == "deposit_refusal":
                def refuse(bed, result, module=module):
                    raise module.SedimentEventError("injected deposit-call refusal")

                _deposit_call(module, m, refuse)
            elif kind == "reconciliation_refusal":
                def tamper(bed, result):  # the published pool differs beyond the FP bound
                    pool = bed.water.mobile_mass_by_cell_class_kg + 1e-3
                    return dataclasses.replace(bed, water=WaterState(bed.water.depth_m, pool)), result

                _deposit_call(module, m, tamper)
            else:
                def reject(*a, **k):
                    raise TransportStepRejected("injected transport rejection")

                for name in ("sparse_characteristic_step", "characteristic_step", "transport_step"):
                    m.setattr(module, name, reject)
            state = args[0] if module is new else as_types(old, args[0])
            control = ctl if module is new else as_types(old, ctl)
            if case == "upwind" and kind == "reconciliation_refusal":
                # no characteristic phase: nothing is reconciled, so both implementations accept the step
                outcomes[label] = ("ok", record(module._sediment_accounting_step(state, *args[1:], control)))
            else:
                with pytest.raises(Exception) as info:
                    module._sediment_accounting_step(state, *args[1:], control)
                outcomes[label] = (type(info.value).__name__, str(info.value))
        assert [v.tobytes() for v in b20.leaves(s["state0"])] == before
        assert b20.store_pins(tmp_path / label) == pins
    if outcomes["new"][0] == "ok":
        assert outcomes["new"][1]["arrays"] == outcomes["old"][1]["arrays"]
        assert outcomes["new"][1] == outcomes["old"][1]  # incl. the complete recursive scalar record
        assert outcomes["old"][0] == "ok"
    else:
        assert outcomes["new"] == outcomes["old"], outcomes
    if kind == "reconciliation_refusal" and case != "upwind":  # upwind has no phase to reconcile
        assert "beyond the declared FP64 bound" in outcomes["new"][1]


# =============================================================================================================
# Liveness (CPU NumPy weak references) with an old-source negative control
# =============================================================================================================
def _probe(module, monkeypatch, refs, seen, reconcile_name):
    """Weak references to the actual kernel result, its transport step, the 8 diagnostics and the 4 faces; snapshots
    at the deposit call and at reconciliation entry."""
    for name in ("sparse_characteristic_step", "characteristic_step"):
        real = getattr(module, name)

        def kernel(*a, real=real, **k):
            step = real(*a, **k)
            refs["step"] = weakref.ref(step)
            refs["transport"] = weakref.ref(step.transport)
            for owner, field in DIAGNOSTICS:
                obj = step.transport if owner == "transport" else step
                refs[field] = weakref.ref(getattr(obj, field))
            for field in FACES:
                refs[field] = weakref.ref(getattr(step.transport, field))
            return step

        monkeypatch.setattr(module, name, kernel)
    real_apply, calls = module.apply_bed_demand, [0]

    def apply(bed, context, demand):
        calls[0] += 1
        if calls[0] == 2:
            gc.collect()
            seen["at_deposit"] = {n: r() is not None for n, r in refs.items()}
            seen["faces_are_demand"] = all(getattr(demand.face_flux, f) is refs[n]()
                                           for f, n in zip(("x_face_crossing_mass_kg", "y_face_crossing_mass_kg",
                                                            "x_face_net_crossing_mass_kg",
                                                            "y_face_net_crossing_mass_kg"), FACES, strict=True))
        new_bed, result = real_apply(bed, context, demand)
        if calls[0] == 2:  # MAPLE validated the result, which carries the same four face arrays
            seen["faces_validated"] = all(getattr(result.face_flux, f) is refs[n]()
                                          for f, n in zip(("x_face_crossing_mass_kg", "y_face_crossing_mass_kg",
                                                           "x_face_net_crossing_mass_kg",
                                                           "y_face_net_crossing_mass_kg"), FACES, strict=True))
            refs["deposit_result"] = weakref.ref(result)
        return new_bed, result

    monkeypatch.setattr(module, "apply_bed_demand", apply)
    real_reconcile = getattr(module, reconcile_name)

    def reconcile(*a, **k):
        gc.collect()
        seen["at_reconcile"] = {n: r() is not None for n, r in refs.items()}
        return real_reconcile(*a, **k)

    monkeypatch.setattr(module, reconcile_name, reconcile)


@pytest.mark.parametrize("case", CASES)
def test_liveness_at_deposit_and_reconciliation_with_old_source_negative_control(old, case, tmp_path, monkeypatch):
    s, _ctx, ctl, schedule = build(case, np, tmp_path)
    args = b20.step_args(s, schedule)
    results = {}
    for label, module, reconcile_name in (("new", new, "_reconcile_phase_inputs"), ("old", old, "reconcile_phase")):
        refs, seen = {}, {}
        with monkeypatch.context() as m:
            _probe(module, m, refs, seen, reconcile_name)
            state = args[0] if module is new else as_types(old, args[0])
            control = ctl if module is new else as_types(old, ctl)
            attempt = module._sediment_accounting_step(state, *args[1:], control)
        gc.collect()
        seen["at_return"] = {n: r() is not None for n, r in refs.items()}
        results[label] = (seen, attempt)
    diag = [f for _o, f in DIAGNOSTICS]
    new_seen, old_seen = results["new"][0], results["old"][0]
    # new: kernel result, transport step and the 8 diagnostics released BEFORE the deposit call; faces alive, passed
    # to MAPLE as the demand's face flux and carried by its validated result
    assert not new_seen["at_deposit"]["step"] and not new_seen["at_deposit"]["transport"]
    assert not any(new_seen["at_deposit"][f] for f in diag), new_seen["at_deposit"]
    assert all(new_seen["at_deposit"][f] for f in FACES)
    assert new_seen["faces_are_demand"] and new_seen["faces_validated"]
    # new: faces and the complete deposit result released BEFORE reconciliation
    assert not any(new_seen["at_reconcile"][f] for f in FACES) and not new_seen["at_reconcile"]["deposit_result"]
    # old-source negative control: everything still alive at both points
    assert all(old_seen["at_deposit"].values()) and all(old_seen["at_reconcile"].values())
    assert old_seen["faces_are_demand"] and old_seen["faces_validated"]
    # at return: the accounting step keeps no extra aliases in either implementation
    assert not any(new_seen["at_return"].values()) and not any(old_seen["at_return"].values())
    del results


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
            name = "CWB65LiveBytes"

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
    """{allocation pointer: allocation size} of every CuPy array reachable from `objs` (pool-rounded sizes)."""
    from maple.core.backend import array_leaves

    out = {}
    for obj in objs:
        for a in array_leaves(obj):
            mem = getattr(getattr(a, "data", None), "mem", None)
            if mem is not None and a.size:
                out[int(mem.ptr)] = int(mem.size)
    return out


def _released(everything, kept):
    return {p: z for p, z in everything.items() if p not in kept}


def _named(step, fields):
    return [getattr(step.transport if owner == "transport" else step, name) for owner, name in fields]


def _phase_sets(step, demand, deposit, bed, allocations):
    """The B63 allocations B65 no longer holds, derived from the actual B63 objects at the phase marks:
      at the deposit call      the kernel result's arrays NOT reachable from the kept records (TransportAccounting,
                               reconciliation inputs) or the deposit demand (incl. the four faces);
      at reconciliation entry  the kernel result's, deposit result's and demand's arrays not reachable from the kept
                               records, DepositAccounting, the post-deposit bed or the core's `zeros` local (the faces
                               among them).
    `zeros` (`_sediment_coupled_core`: `zeros = xp.zeros(...) if zero_demand is None else zero_demand`) is a frame local
    of BOTH implementations that lives until the core returns; `water_demand_from_transport(transport, zeros)` passes
    that SAME object as the demand's `requested_removal_by_cell_class_kg`, so releasing the demand does not release it
    (`test_core_zeros_alias_stays_live_at_reconciliation_cpu` proves the alias and its liveness).
    `reconcile_without_zeros` is the previous (over-counting) expectation, kept as the negative control.
    Returns only {pointer: size} integers and named pointer sets: no array owner is retained by the caller."""
    records = (new.TransportAccounting.of(step.transport), new._PhaseReconcileInputs.of(step))
    zeros = demand.requested_removal_by_cell_class_kg
    everything = allocations(step, deposit, demand)
    kept_after = allocations(*records, new.DepositAccounting.of(deposit), bed)
    out = {"deposit": _released(allocations(step), allocations(*records, demand)),
           "reconcile": _released(everything, {**kept_after, **allocations(zeros)}),
           "reconcile_without_zeros": _released(everything, kept_after),
           "core_zeros": allocations(zeros),
           "core_zeros_shape": tuple(zeros.shape),
           "diagnostics": set(allocations(*_named(step, DIAGNOSTICS))),
           "class_totals": set(allocations(*_named(step, CLASS_TOTALS))),
           "faces": set(allocations(*(getattr(step.transport, f) for f in FACES))),
           "pool_shape": tuple(bed.water.mobile_mass_by_cell_class_kg.shape),
           "diagnostic_shapes": [tuple(a.shape) for a in _named(step, DIAGNOSTICS)],
           "class_total_shapes": [tuple(a.shape) for a in _named(step, CLASS_TOTALS)]}
    return out


# The kernel's per-class totals `sum(M + P)` / `sum(T)` (`TransportStep.mobile_before_by_class_kg` /
# `mobile_after_by_class_kg`; on CUDA CSR the separate `pairwise_sum_over_leading_axes` results of
# characteristic_cuda_csr.py) are read by no accounting record, so B65 releases them at the deposit call together with
# the 8 full-domain diagnostics. They are (nc,) vectors, not D grids.
CLASS_TOTALS = (("transport", "mobile_before_by_class_kg"), ("transport", "mobile_after_by_class_kg"))


def _assert_released_composition(sets):
    """The deposit-call release is EXACTLY the 8 diagnostic D grids plus the 2 per-class totals (alias-deduplicated
    allocations), and the reconciliation release contains all four faces."""
    assert len(sets["diagnostics"]) == 8 and len(sets["class_totals"]) == 2
    assert not sets["diagnostics"] & sets["class_totals"]
    assert set(sets["deposit"]) == sets["diagnostics"] | sets["class_totals"], sets
    assert all(shape == sets["pool_shape"] for shape in sets["diagnostic_shapes"])
    assert all(shape == sets["pool_shape"][-1:] for shape in sets["class_total_shapes"])
    assert sets["faces"] <= set(sets["reconcile"]) and set(sets["deposit"]) <= set(sets["reconcile"])
    # the core's live `zeros` (one allocation of the pool's shape) is the ONLY difference to the previous expectation
    assert len(sets["core_zeros"]) == 1 and sets["core_zeros_shape"] == sets["pool_shape"]
    assert set(sets["reconcile_without_zeros"]) - set(sets["reconcile"]) == set(sets["core_zeros"])
    assert set(sets["reconcile"]) <= set(sets["reconcile_without_zeros"])


def _buffers(*objs):
    """CPU analogue of `_allocations`: {id of the owning base buffer: nbytes} of every non-empty NumPy array."""
    from maple.core.backend import array_leaves

    out = {}
    for obj in objs:
        for a in array_leaves(obj):
            if a.size:
                base = a
                while isinstance(base.base, np.ndarray):
                    base = base.base
                out[id(base)] = int(base.nbytes)
    return out


def _capture_phase_sets(module, monkeypatch, allocations, on_deposit=None):
    """Observer: derive `_phase_sets` from the actual objects of one accounting step without retaining any owner
    beyond the call (the kernel result is held only between the kernel and the deposit call, which the step itself
    holds in B63)."""
    held, sets, calls = {}, {}, [0]
    for name in ("sparse_characteristic_step", "characteristic_step"):
        real_k = getattr(module, name)

        def kernel(*a, real_k=real_k, **k):
            step = real_k(*a, **k)
            held["step"] = step
            return step

        monkeypatch.setattr(module, name, kernel)
    real_apply = module.apply_bed_demand

    def apply(bed, context, demand):
        calls[0] += 1
        if calls[0] == 2 and on_deposit is not None:
            on_deposit()
        new_bed, result = real_apply(bed, context, demand)
        if calls[0] == 2:
            sets.update(_phase_sets(held.pop("step"), demand, result, new_bed, allocations))
        return new_bed, result

    monkeypatch.setattr(module, "apply_bed_demand", apply)
    return sets


@pytest.mark.parametrize("case", CASES)
def test_released_composition_cpu(old, case, tmp_path, monkeypatch):
    """CPU (NumPy base buffers) evidence for the composition the device test asserts on actual allocations."""
    s, _ctx, ctl, schedule = build(case, np, tmp_path)
    args = b20.step_args(s, schedule)
    with monkeypatch.context() as m:
        sets = _capture_phase_sets(old, m, _buffers)
        old._sediment_accounting_step(as_types(old, args[0]), *args[1:], as_types(old, ctl))
    _assert_released_composition(sets)


class _Registry:
    """Weak-reference registry of NumPy owning base buffers with unique tokens: a buffer is identified by object
    identity while ALIVE (an `id` reused by a later buffer after the first died gets a NEW token), never retained."""

    def __init__(self):
        self.by_id, self.refs, self.nbytes = {}, [], []

    def tokens(self, *objs):
        from maple.core.backend import array_leaves

        out = set()
        for obj in objs:
            for a in array_leaves(obj):
                if not a.size:
                    continue
                base = a
                while isinstance(base.base, np.ndarray):
                    base = base.base
                token = next((t for t in self.by_id.get(id(base), ()) if self.refs[t]() is base), None)
                if token is None:
                    token = len(self.refs)
                    self.refs.append(weakref.ref(base))
                    self.nbytes.append(int(base.nbytes))
                    self.by_id.setdefault(id(base), []).append(token)
                out.add(token)
        return out

    def dead(self, tokens):
        gc.collect()
        return {t for t in tokens if self.refs[t]() is None}


@pytest.mark.parametrize("case", CASES)
def test_core_zeros_alias_stays_live_at_reconciliation_cpu(old, case, tmp_path, monkeypatch):
    """CPU proof of the measured 512 B device discrepancy (CW-B65 correction 2). In BOTH implementations the deposit
    demand's `requested_removal_by_cell_class_kg` IS the core's frame local `zeros` (same object, pool shape) and that
    buffer is still alive at reconciliation entry. B65's ACTUAL releases (weak references, no strong reference held by
    the observer past each hook) are measured:
      deposit call          exactly kernel buffers - (kept records + demand): the 8 diagnostics + 2 class totals;
      reconciliation entry  exactly (kernel + demand + deposit buffers) - (kept records + DepositAccounting + bed +
                            core `zeros`);
    negative control: the previous expectation without `zeros` differs from the measured release by exactly the
    `zeros` buffer. B63 (old-source control) releases none of them; nothing extra survives the return in B65."""
    s, _ctx, ctl, schedule = build(case, np, tmp_path)
    args = b20.step_args(s, schedule)
    seen = {}
    for label, module, reconcile_name in (("new", new, "_reconcile_phase_inputs"), ("old", old, "reconcile_phase")):
        reg, obs = _Registry(), {}
        with monkeypatch.context() as m:
            for name in ("sparse_characteristic_step", "characteristic_step"):
                real_k = getattr(module, name)

                def kernel(*a, real_k=real_k, reg=reg, obs=obs, **k):
                    step = real_k(*a, **k)
                    obs["kernel"] = reg.tokens(step)
                    obs["kept"] = reg.tokens(new.TransportAccounting.of(step.transport),
                                             new._PhaseReconcileInputs.of(step))
                    obs["diagnostics"] = reg.tokens(*_named(step, DIAGNOSTICS))
                    obs["class_totals"] = reg.tokens(*_named(step, CLASS_TOTALS))
                    return step

                m.setattr(module, name, kernel)
            real_apply, calls = module.apply_bed_demand, [0]

            def apply(bed, context, demand, real_apply=real_apply, calls=calls, reg=reg, obs=obs):
                calls[0] += 1
                if calls[0] != 2:
                    return real_apply(bed, context, demand)
                obs["dead_at_deposit"] = reg.dead(obs["kernel"])
                obs["demand"] = reg.tokens(demand)
                zeros = demand.requested_removal_by_cell_class_kg
                obs["zeros"] = reg.tokens(zeros)
                obs["zeros_shape"] = tuple(zeros.shape)
                del zeros
                new_bed, result = real_apply(bed, context, demand)
                obs["deposit"] = reg.tokens(result)
                obs["kept_after"] = reg.tokens(new.DepositAccounting.of(result), new_bed)
                obs["pool_shape"] = tuple(new_bed.water.mobile_mass_by_cell_class_kg.shape)
                return new_bed, result

            m.setattr(module, "apply_bed_demand", apply)
            real_r = getattr(module, reconcile_name)

            def reconcile(*a, real_r=real_r, reg=reg, obs=obs, **k):
                everything = obs["kernel"] | obs["demand"] | obs["deposit"]
                obs["dead_at_reconcile"] = reg.dead(everything)
                frame = sys._getframe(1)
                (token,) = obs["zeros"]
                obs["caller"] = frame.f_code.co_name
                obs["zeros_is_core_local"] = frame.f_locals["zeros"] is reg.refs[token]()
                del frame
                return real_r(*a, **k)

            m.setattr(module, reconcile_name, reconcile)
            state = args[0] if module is new else as_types(old, args[0])
            control = ctl if module is new else as_types(old, ctl)
            attempt = module._sediment_accounting_step(state, *args[1:], control)
        everything = obs["kernel"] | obs["demand"] | obs["deposit"]
        obs["alive_at_return"] = everything - reg.dead(everything)
        obs["kept_by_result"] = reg.tokens(attempt)
        obs["zeros_bytes"] = sum(reg.nbytes[t] for t in obs["zeros"])
        del attempt
        seen[label] = obs
    for label in ("new", "old"):  # the alias and its liveness, in BOTH implementations
        obs = seen[label]
        assert obs["caller"] == "_sediment_coupled_core" and obs["zeros_is_core_local"], label
        assert len(obs["zeros"]) == 1 and obs["zeros"] <= obs["demand"] and obs["zeros_shape"] == obs["pool_shape"]
        assert not obs["zeros"] & obs["dead_at_reconcile"], label
        assert not obs["zeros"] & (obs["kept"] | obs["kept_after"]), label  # NOT reachable from any kept record
        assert obs["zeros_bytes"] == int(np.prod(obs["pool_shape"])) * 8
    n, o = seen["new"], seen["old"]
    # B65 measured releases equal the derived expectations exactly
    assert n["dead_at_deposit"] == n["kernel"] - (n["kept"] | n["demand"])
    assert n["dead_at_deposit"] == n["diagnostics"] | n["class_totals"] and len(n["dead_at_deposit"]) == 10
    everything = n["kernel"] | n["demand"] | n["deposit"]
    expected = everything - (n["kept"] | n["kept_after"] | n["zeros"])
    without_zeros = everything - (n["kept"] | n["kept_after"])
    assert n["dead_at_reconcile"] == expected
    assert without_zeros - n["dead_at_reconcile"] == n["zeros"]  # negative control: the prior over-count, exactly
    # old-source negative control: B63 holds every one of them at both marks
    assert not o["dead_at_deposit"] and not o["dead_at_reconcile"]
    # at return both keep only what their returned results reach (the core's `zeros` dies with its frame)
    for obs in (n, o):
        assert obs["alive_at_return"] <= obs["kept_by_result"] and not obs["zeros"] & obs["alive_at_return"]


@pytest.mark.parametrize("case", CASES)
def test_device_live_bytes_released_at_deposit_reconciliation_and_return(old, case, tmp_path, monkeypatch):
    """B63 - B65 live bytes, measured by a MemoryHook:
      deposit call / reconciliation entry  equal the pool sizes of the allocations B65 no longer holds (`_phase_sets`,
                                           derived from the actual B63 objects, never from typed sizes); at the deposit
                                           call EXACTLY the 8 diagnostic D grids plus the 2 per-class totals;
      return                               ZERO: both equivalent returned results stay alive and every observational
                                           reference has been released before the measurement.
    The four faces reach MAPLE's validated deposit result in both implementations. Each implementation is warmed up
    first (unhooked), so first-call caches are not attributed to either measured step."""
    xp = b20.backend_xp("cupy")
    s, _ctx, ctl, schedule = build(case, xp, tmp_path)
    args = b20.step_args(s, schedule)
    live, sets, kept_results = {}, {}, {}
    for label, module, reconcile_name in (("new", new, "_reconcile_phase_inputs"), ("old", old, "reconcile_phase")):
        state = args[0] if module is new else as_types(old, args[0])
        control = ctl if module is new else as_types(old, ctl)
        module._sediment_accounting_step(state, *args[1:], control)  # warm-up, outside every measurement
        gc.collect()
        tracker, marks, faces = _Hook(), {}, {}
        with monkeypatch.context() as m:
            real_apply, calls = module.apply_bed_demand, [0]

            def apply(bed, context, demand, real_apply=real_apply, calls=calls, marks=marks, tracker=tracker,
                      faces=faces):
                calls[0] += 1
                if calls[0] == 2:
                    xp.cuda.Device().synchronize()
                    marks["deposit"] = tracker.current
                    faces["zeros_ptr"] = int(demand.requested_removal_by_cell_class_kg.data.mem.ptr)  # an int only
                new_bed, result = real_apply(bed, context, demand)
                if calls[0] == 2:  # the four faces passed in the demand are MAPLE's validated result's faces
                    faces["validated"] = all(getattr(result.face_flux, f) is getattr(demand.face_flux, f)
                                             for f in ("x_face_crossing_mass_kg", "y_face_crossing_mass_kg",
                                                       "x_face_net_crossing_mass_kg", "y_face_net_crossing_mass_kg"))
                return new_bed, result

            m.setattr(module, "apply_bed_demand", apply)
            if label == "old":  # derive the expected sets from B63's actual objects (B63 holds them all anyway)
                sets = _capture_phase_sets(old, m, _allocations)
            real_r = getattr(module, reconcile_name)

            def reconcile(*a, real_r=real_r, marks=marks, tracker=tracker, **k):
                xp.cuda.Device().synchronize()
                marks["reconcile"] = tracker.current
                # AFTER the mark: private pointer observation of the caller's live `zeros` local (ints only; the frame
                # handle is dropped at once and the core's frame ends before the return measurement)
                frame = sys._getframe(1)
                marks["caller"] = frame.f_code.co_name
                marks["zeros_ptr"] = int(frame.f_locals["zeros"].data.mem.ptr)
                del frame
                return real_r(*a, **k)

            m.setattr(module, reconcile_name, reconcile)
            with tracker.hook():
                kept_results[label] = module._sediment_accounting_step(state, *args[1:], control)
                gc.collect()
                xp.cuda.Device().synchronize()
                marks["return"] = tracker.current
        live[label] = marks
        assert faces["validated"], label
        # the demand's removal request IS the core's `zeros` local, still alive at reconciliation entry
        assert marks["caller"] == "_sediment_coupled_core" and marks["zeros_ptr"] == faces["zeros_ptr"], label
    assert {live["old"]["zeros_ptr"]} == set(sets["core_zeros"])
    got = {k: live["old"][k] - live["new"][k] for k in ("deposit", "reconcile", "return")}
    want = {"deposit": sum(sets["deposit"].values()), "reconcile": sum(sets["reconcile"].values()), "return": 0}
    print(f"CW-B65 {case}: live-byte differences {got}, expected {want}, "
          f"diagnostics {len(sets['diagnostics'])}, class totals {len(sets['class_totals'])}, "
          f"faces {len(sets['faces'])} allocations, core zeros {sets['core_zeros']}")
    assert got == want, (got, want)
    _assert_released_composition(sets)
    # negative control: the previous expectation (omitting the live `zeros`) over-counts by exactly that allocation
    without = sum(sets["reconcile_without_zeros"].values())
    assert without - want["reconcile"] == sum(sets["core_zeros"].values()) > 0 and got["reconcile"] != without
    assert record(kept_results["new"]) == record(kept_results["old"])
