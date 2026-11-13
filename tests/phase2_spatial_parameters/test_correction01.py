"""S2-A.1 correction01 regressions: live-metadata revalidation, one-layer scalar contract, read-only public metadata, no CuPy probe on the NumPy path, ownership (weakrefs), the owner's public
mapping invariants, unchanged numerics behind the three new typed guards. Author-written; no GPU kernel is launched and no malformed device pointer is ever constructed."""
from __future__ import annotations

import dataclasses
import gc
import importlib.util
import sys
import tracemalloc
import warnings
import weakref
from pathlib import Path

import numpy as np
import pytest
from conftest import BIG, host, make_spec, params, prepare

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss


def P(L=2, N=4, seed=1, mode="constant"):
    return params(np.random.default_rng(seed), L, N, mode=mode)


def spec_of(p, **kw):
    return make_spec(p, **kw)


# ---- 1 stale/edited metadata is revalidated before admission/allocation -------------------------------------------------------------
def _set_dtype(arr, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            arr.dtype = dtype
        except (AttributeError, ValueError, TypeError):
            pytest.skip("this NumPy refuses in-place dtype assignment")


def _no_allocation_refusal(spec, backend, exc=sp.SpatialLayerError):
    tracemalloc.start()
    with pytest.raises(exc):
        prepare(spec, backend=backend)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 250_000  # nothing of the preparation (maps, chunks) was allocated before the refusal


def test_dtype_changed_after_spec_creation_is_a_typed_refusal(cpu_backend):
    p = P()
    p["res"] = np.zeros(2)  # a shared (L,) theta
    spec = spec_of(p)
    _set_dtype(p["res"], np.float32)  # Root's repro: the array now has shape (4,) and 16 bytes
    _no_allocation_refusal(spec, cpu_backend)


@pytest.mark.parametrize("key,new_shape", [("edges", (1, 12)), ("edges", (12,)), ("fc", (4, 2)), ("sat", (8,)), ("iface", (4,))])
def test_shape_changed_after_spec_creation_is_a_typed_refusal(cpu_backend, key, new_shape):
    p = P(L=3, N=4)  # edges (4,4)=16 elems, fc/sat (3,4)=12, iface (2,4)=8
    spec = spec_of(p)
    arr = p[key]
    new_shape = {"edges": (1, 16), "fc": (4, 3), "sat": (12,), "iface": (2, 2, 2)}[key] if key != "edges" or new_shape == (1, 12) else (16,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # NumPy 2.5 deprecates setting .shape: that in-place reinterpretation IS the caller edit under test
        arr.shape = new_shape  # in-place reinterpretation of the caller's array
    _no_allocation_refusal(spec, cpu_backend)


def test_layout_change_after_spec_creation_is_refused_not_trusted(cpu_backend):
    L, N = 3, 4
    p = P(L=L, N=N)
    p["sat"] = np.full(L * N, 0.5)[:L].copy()  # shared layout recorded in the spec
    spec = spec_of(p)
    assert spec.layouts()["theta_saturation"] == "shared"
    p["sat"].resize(L * N, refcheck=False)  # now the live array has L*N elements: no longer a recorded layout
    _no_allocation_refusal(spec, cpu_backend)


def test_values_changed_after_spec_creation_are_revalidated_as_values(cpu_backend):
    p = P()
    spec = spec_of(p)
    p["res"][0, 1] = -1.0  # value edits are legal until prepare, and are then caught by the VALUE checks
    with pytest.raises(sp.SpatialLayerError) as info:
        prepare(spec, backend=cpu_backend)
    assert info.value.bit == sp.FLAG_REL and info.value.first_cell == 1


def _edited(spec, **fields):
    return dataclasses.replace(spec, **fields)


@pytest.mark.parametrize("mutate", [
    lambda s: _edited(s, n_layers=9), lambda s: _edited(s, n_layers=True), lambda s: _edited(s, n_cells=99), lambda s: _edited(s, n_cells=0), lambda s: _edited(s, namespace="cupy"),
    lambda s: _edited(s, namespace="x"), lambda s: _edited(s, diffusion_mode="tietjen_donor_depth"), lambda s: _edited(s, diffusion_mode="nope"), lambda s: _edited(s, bottom_drainage="x"),
    lambda s: _edited(s, gravity_drainage=1), lambda s: _edited(s, diffusion_coefficient=0.3), lambda s: _edited(s, theta_evaporation_stress=float("nan")),
    lambda s: _edited(s, evaporation_factor=-1.0), lambda s: _edited(s, inputs={}), lambda s: _edited(s, inputs={**dict(s.inputs), "extra": s.inputs["theta_residual"]}),
    lambda s: _edited(s, inputs={k: v for k, v in s.inputs.items() if k != "theta_residual"}), lambda s: _edited(s, inputs=None), lambda s: _edited(s, inputs=[1, 2]),
    lambda s: _edited(s, inputs={**dict(s.inputs), "theta_residual": "record"}),
    lambda s: _edited(s, inputs={**dict(s.inputs), "theta_residual": dataclasses.replace(s.inputs["theta_residual"], kind="shared")}),
    lambda s: _edited(s, inputs={**dict(s.inputs), "theta_residual": dataclasses.replace(s.inputs["theta_residual"], nbytes=1)}),
    lambda s: _edited(s, inputs={**dict(s.inputs), "theta_residual": dataclasses.replace(s.inputs["theta_residual"], rows=99)}),
    lambda s: _edited(s, inputs={**dict(s.inputs), "theta_residual": dataclasses.replace(s.inputs["theta_residual"], array=None, scalar=None)}),
    lambda s: _edited(s, inputs={**dict(s.inputs), "theta_residual": dataclasses.replace(s.inputs["theta_residual"], name="other")}),
], ids=["n_layers9", "n_layers_bool", "n_cells99", "n_cells0", "ns_cupy", "ns_x", "mode_donor", "mode_nope", "bottom_x", "gravity_int", "d_in_constant", "stress_nan", "factor_neg", "inputs_empty", "inputs_extra", "inputs_missing", "inputs_none", "inputs_list", "record_str", "record_kind", "record_nbytes", "record_rows", "record_noarray", "record_name"])
def test_hand_edited_or_replaced_specs_are_typed_refusals(cpu_backend, mutate):
    spec = spec_of(P())
    try:
        bad = mutate(spec)
    except TypeError:
        return  # replace() itself refused the edit
    _no_allocation_refusal(bad, cpu_backend)


def test_directly_constructed_spec_is_revalidated_too(cpu_backend):
    p = P()
    good = spec_of(p)
    forged = sp.SpatialLayerSpec(2, 4, "numpy", dict(good.inputs), "constant", "open", True, None, 0.5, 1.0)  # same values by hand: accepted only because the live arrays agree
    assert prepare(forged, backend=cpu_backend).n_cells == 4
    wrong = sp.SpatialLayerSpec(3, 4, "numpy", dict(good.inputs), "constant", "open", True, None, 0.5, 1.0)
    _no_allocation_refusal(wrong, cpu_backend)


def test_spec_inputs_mapping_is_read_only():
    spec = spec_of(P())
    with pytest.raises(TypeError):
        spec.inputs["theta_residual"] = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.n_layers = 3


def test_a_subclassed_spec_is_refused(cpu_backend):
    @dataclasses.dataclass(frozen=True, eq=False)
    class Sub(sp.SpatialLayerSpec):
        pass

    spec = spec_of(P())
    sub = Sub(spec.n_layers, spec.n_cells, spec.namespace, spec.inputs, spec.diffusion_mode, spec.bottom_drainage, spec.gravity_drainage, spec.diffusion_coefficient,
              spec.theta_evaporation_stress, spec.evaporation_factor)
    with pytest.raises(sp.SpatialLayerError, match="exactly"):
        prepare(sub, backend=cpu_backend)


# ---- 2 one-layer scalar contract --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("bad", [-1.0, -1e-30, float("nan"), float("inf"), -float("inf"), True, "1"])
def test_one_layer_scalar_interface_input_must_be_finite_and_nonnegative(mode, bad):
    p = P(L=1, mode=mode)
    with pytest.raises(sp.SpatialLayerError):
        make_spec(p, mode=mode, iface=bad)


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("good", [0.0, 1e-7, 3, np.float64(2.5), np.zeros(0), np.zeros((0, 4))])
def test_one_layer_valid_interface_input_still_accepted_without_interface_maps(cpu_backend, mode, good):
    p = P(L=1, mode=mode)
    col = prepare(make_spec(p, mode=mode, iface=good), backend=cpu_backend)
    assert set(col.retained_names()) == {"theta_residual", "theta_field_capacity", "theta_saturation", "dz"}


def test_one_layer_forged_scalar_record_is_refused_at_preparation(cpu_backend):
    spec = make_spec(P(L=1), iface=1e-7)
    key = "diffusivity_m2_s"
    for bad in (-1.0, float("nan"), float("inf")):
        forged = dataclasses.replace(spec, inputs={**dict(spec.inputs), key: dataclasses.replace(spec.inputs[key], scalar=bad)})
        _no_allocation_refusal(forged, cpu_backend)


def test_multi_layer_scalar_interface_values_are_validated_by_the_value_checks(cpu_backend):
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(sp.SpatialLayerError) as info:
            prepare(make_spec(P(L=3), iface=bad), backend=cpu_backend)
        assert info.value.bit == sp.FLAG_IFACE


# ---- 3 public metadata is read-only and canonical ---------------------------------------------------------------------------------------
def test_public_layout_and_plan_metadata_is_immutable(cpu_backend):
    col = prepare(spec_of(P(L=3, N=6, mode="tietjen_donor_depth"), mode="tietjen_donor_depth"), backend=cpu_backend)
    sha, summary = col.sha256, col.summary()
    for target, key in ((col.layouts, "theta_residual"), (col.estimate.layouts, "theta_residual"), (col.estimate.retained, "dz"), (col.estimate.input_bytes, "theta_residual"),
                        (col.estimate.terms, "retained_maps"), (col.estimate.retained["dz"], "kind"), (col._maps, "dz")):
        with pytest.raises(TypeError):
            target[key] = "cell"
    for name in ("n_cells", "layouts", "estimate", "parameters_sha256"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(col, name, 1)
    assert col.sha256 == sha and col.summary() == summary and col.layouts["theta_residual"] == "cell"
    out = col.summary()
    out["input_layouts"]["theta_residual"] = "scalar"  # the summary is a COPY
    out["retained"]["dz"]["kind"] = "shared"
    assert col.summary() == summary


def test_estimator_copies_the_callers_layout_dict(cpu_backend):
    layouts = {"depth_edges_m": "cell", "theta_residual": "cell", "theta_field_capacity": "cell", "theta_saturation": "cell", "diffusivity_m2_s": "cell"}
    est = sp.estimate_spatial_memory(3, 10, layouts=layouts, backend=cpu_backend)
    before = est.retained_bytes, dict(est.layouts)
    layouts["depth_edges_m"] = "shared"
    assert (est.retained_bytes, dict(est.layouts)) == before


def test_owner_is_not_copyable_or_serialisable(cpu_backend):
    import copy
    import pickle

    col = prepare(spec_of(P()), backend=cpu_backend)
    for fn in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(sp.SpatialLayerError):
            fn(col)


# ---- 4 the NumPy path never probes CuPy ------------------------------------------------------------------------------------------------
@pytest.fixture
def cupy_spy(monkeypatch):
    import maple.core.backend as be

    calls = []

    def spy():
        calls.append("cupy_module")
        raise AssertionError("the NumPy path must not probe CuPy")

    monkeypatch.setattr(be, "cupy_module", spy)
    return calls


def test_numpy_spec_and_preparation_never_probe_cupy(cpu_backend, cupy_spy):
    spec = spec_of(P(L=3, N=7))
    col = prepare(spec, backend=cpu_backend)
    assert col.n_cells == 7 and cupy_spy == []
    sp.estimate_spatial_memory(3, 7, layouts=spec.layouts(), backend=cpu_backend)
    assert cupy_spy == []


@pytest.mark.parametrize("bad_edges", [[0.0, 0.1, 0.2], np.ma.masked_array(np.zeros(3)), "x", None, (0.0, 1.0)])
def test_unsupported_edge_objects_are_typed_refusals_without_probing_cupy(cupy_spy, bad_edges):
    p = P(L=2, N=3)
    with pytest.raises(sp.SpatialLayerError):
        sp.spatial_layer_spec(bad_edges, theta_residual=p["res"], theta_field_capacity=p["fc"], theta_saturation=p["sat"], theta_evaporation_stress=0.5, diffusivity_m2_s=p["iface"], n_cells=3)
    assert cupy_spy == []


def test_a_cupy_typed_object_triggers_one_probe_and_a_typed_setup_failure_without_fallback(monkeypatch):
    import maple.core.backend as be

    monkeypatch.setattr(be, "cupy_module", lambda: None)
    monkeypatch.setattr(be, "cupy_import_error", lambda: "ImportError: no cupy here")
    fake_type = type("ndarray", (), {"__module__": "cupy._core.core"})
    p = P(L=2, N=3)
    with pytest.raises(sp.SpatialLayerBackendUnavailableError, match="no cupy here"):
        sp.spatial_layer_spec(fake_type(), theta_residual=p["res"], theta_field_capacity=p["fc"], theta_saturation=p["sat"], theta_evaporation_stress=0.5,
                              diffusivity_m2_s=p["iface"], n_cells=3)
    monkeypatch.setattr(be, "cupy_module", lambda: (_ for _ in ()).throw(RuntimeError("probe exploded")))
    with pytest.raises(sp.SpatialLayerBackendUnavailableError, match="probe exploded"):
        sp.spatial_layer_spec(fake_type(), theta_residual=p["res"], theta_field_capacity=p["fc"], theta_saturation=p["sat"], theta_evaporation_stress=0.5,
                              diffusivity_m2_s=p["iface"], n_cells=3)


def test_an_explicit_cuda_request_without_a_usable_cupy_is_a_typed_setup_failure_and_never_falls_back(monkeypatch):
    import maple.core.backend as be

    monkeypatch.setattr(be, "cupy_module", lambda: None)
    monkeypatch.setattr(be, "cupy_import_error", lambda: "ImportError: no cupy")
    spec = spec_of(P())
    with pytest.raises(sp.SpatialLayerBackendUnavailableError):
        sp.prepare_spatial_column(spec, np.ones(4, dtype=bool), 1.0, backend="cuda", memory_budget_bytes=BIG)


# ---- 5 ownership: nothing of the caller's spec/arrays is retained --------------------------------------------------------------------------
def test_prepared_owner_retains_neither_the_spec_nor_the_callers_arrays(cpu_backend):
    p = P(L=3, N=9)
    spec = spec_of(p)
    refs = [weakref.ref(spec)] + [weakref.ref(v) for v in p.values()]
    col = prepare(spec, backend=cpu_backend)
    sha = col.sha256
    del spec, p
    gc.collect()
    assert all(r() is None for r in refs), [i for i, r in enumerate(refs) if r() is not None]
    assert col.sha256 == sha and col.n_layers == 3


# ---- 6 active-owner public mapping invariants ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("case", ["all_cell", "mixed"])
def test_owner_public_metadata_matches_its_private_maps_and_the_estimate(cpu_backend, mode, case):
    L, N = 4, 13
    p = P(L=L, N=N, mode=mode)
    if case == "mixed":
        spec = make_spec({"edges": p["edges"][:, 0].copy(), "res": 0.03, "fc": p["fc"], "sat": p["sat"], "iface": 2e-7}, mode=mode, N=N, iface=2e-7)
    else:
        spec = make_spec(p, mode=mode)
    mask = np.arange(N) % 3 != 0
    col = prepare(spec, backend=cpu_backend, mask=mask)
    assert dict(col.layouts) == spec.layouts() == {k: v for k, v in col.summary()["input_layouts"].items()}
    assert set(col.estimate.retained) == set(col.retained_names()) == set(col.summary()["retained"])
    total = 0
    for name in col.retained_names():
        arr, info = col._maps[name], col.estimate.retained[name]
        expected_shape = (1,) if info["kind"] == "scalar" else ((info["rows"],) if info["kind"] == "shared" else (info["rows"], N))
        assert arr.shape == expected_shape and arr.dtype == np.float64 and arr.nbytes == info["bytes"] and not arr.flags.writeable, name
        total += arr.nbytes
    assert total == col.estimate.retained_bytes == col.retained_bytes and col._mask.nbytes == col.estimate.mask_bytes == N
    assert col.n_active == int(mask.sum()) and col._mask.dtype == np.bool_
    assert col.estimate.input_bytes == spec.input_bytes()


# ---- 7 the three new typed guards leave proper SoilColumn numerics unchanged ------------------------------------------------------------------
OLD = Path(__file__).resolve().parents[3] / "s2_a1_pre_correction01_sources"


def _load_old(name):
    path = OLD / f"{name}.py"
    if not path.is_file():
        pytest.skip("the pre-correction source copy is not available in this tree (task evidence only)")
    spec = importlib.util.spec_from_file_location(f"old_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _column(mod):
    lay = mod.soil_layers(np.array([0.0, 0.1, 0.3, 0.7]), theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.4, diffusivity_m2_s=[1e-7, 2e-7],
                          theta_evaporation_stress=0.5)
    return lay, mod.prepare_column(lay, np.ones(6, dtype=bool), 4.0, backend="numpy", reserve_bytes=0)


def _same(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    else:
        assert x == y


def test_reference_step_runner_and_memory_estimate_are_bit_identical_to_the_pre_guard_sources():
    old = _load_old("subsurface")
    _, col_old = _column(old)
    _, col_new = _column(ss)
    rng = np.random.default_rng(3)
    water = rng.uniform(0.0, 1.0, size=(3, 6)) * np.array([[0.1], [0.2], [0.4]]) * 0.9
    added = rng.uniform(0.0, 1e-3, size=(3, 6))
    pet = rng.uniform(0.0, 1e-8, size=6)
    for fn in ("reference_step", "reference_runner"):
        a, b = getattr(old, fn)(col_old, water, added, pet, 60.0), getattr(ss, fn)(col_new, water, added, pet, 60.0)
        assert len(a) == len(b) and a._fields == b._fields
        for x, y in zip(a, b):
            _same(x, y)
    a, b = old.reference_runner(col_old, water, None, pet, 60.0, None, 0.003), ss.reference_runner(col_new, water, None, pet, 60.0, None, 0.003)  # the pulse-aware call
    for x, y in zip(a, b):
        _same(x, y)
    old_t = _load_old("subsurface_transpiration")
    from maple_syrup import subsurface_transpiration as st

    assert dataclasses.asdict(st.estimate_combined_memory(col_new, query_available=False)) == dataclasses.asdict(old_t.estimate_combined_memory(col_old, query_available=False))  # the two modules define distinct (equal-shaped) classes
