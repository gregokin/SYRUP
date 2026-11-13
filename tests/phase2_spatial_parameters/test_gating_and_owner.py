"""Foreign-object refusals, class separation and ownership of the prepared column. Author-written."""
from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
from conftest import BIG, host, make_spec, params, prepare

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss

# every PUBLIC function of the existing sources whose first parameter is `column` (found by introspection of the existing modules; listed explicitly so a new one is noticed)
TYPED_REFUSAL = [
    ("subsurface", n) for n in ("admit_step", "column_state", "initialize_profile", "moisture", "step", "step_memory", "step_with_recharge", "storage_m3", "summary", "uniform_state")] + [
    ("matrix_infiltration", "prepare_matrix"), ("preferential_infiltration", "prepare_preferential"), ("soil_evaporation", "prepare_evaporation"),
    ("subsurface_transpiration", "prepare_transpiration"), ("layered_storm", "prepare_layered_storm"), ("water_lifecycle", "prepare_lifecycle")]
# (S2-A.1 correction01: `reference_step`, `reference_runner` and `estimate_combined_memory` received the existing `_check_column` guard, so ALL 19 refuse typed.)
TYPED_REFUSAL = TYPED_REFUSAL + [("subsurface", "reference_runner"), ("subsurface", "reference_step"), ("subsurface_transpiration", "estimate_combined_memory")]
COLUMN_ENTRY_POINTS = TYPED_REFUSAL


@pytest.fixture(scope="module")
def column():
    p = params(np.random.default_rng(1), 2, 4)
    return prepare(make_spec(p))


def _call(column, module, name):
    fn = getattr(importlib.import_module("maple_syrup." + module), name)
    args = [column] + [None] * (len([p for p in inspect.signature(fn).parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) - 1)
    kwargs = {p.name: None for p in inspect.signature(fn).parameters.values() if p.default is p.empty and p.kind is p.KEYWORD_ONLY}
    return fn(*args, **kwargs)


@pytest.mark.parametrize("module,name", TYPED_REFUSAL)
def test_existing_column_entry_points_refuse_the_spatial_column_typed_and_early(column, module, name):
    with pytest.raises(ss.SubsurfaceError):  # a typed refusal, never AttributeError/TypeError/silent half-work
        _call(column, module, name)


def test_class_separation():
    assert not issubclass(sp.SpatialSoilColumn, ss.SoilColumn) and not issubclass(sp.SpatialSoilColumn, ss.SoilLayers)
    assert not issubclass(sp.SpatialLayerSpec, ss.SoilLayers) and not issubclass(sp.SpatialLayerSpec, ss.SoilColumn)


def test_spatial_preparation_refuses_foreign_objects():
    lay = ss.soil_layers(np.array([0.0, 0.1, 0.3]), theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.4, diffusivity_m2_s=1e-7, theta_evaporation_stress=0.5)
    with pytest.raises(sp.SpatialLayerError, match="SoilLayers"):
        sp.prepare_spatial_column(lay, np.ones(3, dtype=bool), 1.0, memory_budget_bytes=BIG)
    for bad in (None, "spec", {"n": 1}, object()):
        with pytest.raises(sp.SpatialLayerError):
            sp.prepare_spatial_column(bad, np.ones(3, dtype=bool), 1.0, memory_budget_bytes=BIG)
    col = ss.prepare_column(lay, np.ones(3, dtype=bool), 1.0)
    with pytest.raises(sp.SpatialLayerError):
        sp.prepare_spatial_column(col, np.ones(3, dtype=bool), 1.0, memory_budget_bytes=BIG)


def test_the_column_is_tagged_parameter_only_with_no_hydrology_surface(column):
    assert column.component_tag == "parameter_component_only" and column.hydrology_step_supported is False
    for attr in ("step", "layers", "runner", "water_m", "state", "dz", "arrays", "maps", "thickness_m", "kernel_parameters"):
        assert not hasattr(column, attr), attr  # no public mutable view or hydrology entry
    assert column.summary()["hydrology_step_supported"] is False


def test_all_19_column_entry_points_are_covered_and_typed():
    assert len(TYPED_REFUSAL) == 19 and len(set(TYPED_REFUSAL)) == 19


def test_direct_construction_and_replace_are_refused(column):
    import dataclasses

    args = (column.backend, column.n_layers, column.n_cells, column.n_active, column.cell_area_m2, column.diffusion_mode, column.bottom_drainage, column.gravity_drainage,
            column.diffusion_coefficient, column.theta_evaporation_stress, column.evaporation_factor, dict(column.layouts), "a", "b", column.estimate, None, {}, np.ones(1, bool))
    with pytest.raises(sp.SpatialLayerError):
        sp.SpatialSoilColumn(*args)
    with pytest.raises(sp.SpatialLayerError):
        dataclasses.replace(column, n_cells=999)  # a replace() would otherwise forge an owner with another meaning
    with pytest.raises(sp.SpatialLayerError):
        dataclasses.replace(column, parameters_sha256="0" * 64)


def test_numpy_retained_maps_are_read_only_and_inspection_returns_detached_copies(column):
    for name in column.retained_names():
        assert not column._maps[name].flags.writeable
        with pytest.raises(ValueError):
            column._maps[name][...] = 0.0
    a = host(column, "dz")
    a[...] = -1.0
    assert (host(column, "dz") > 0).all()  # the copy is detached
    assert a.flags.writeable


def test_inspection_is_admitted_before_it_copies(column):
    need = column._maps["dz"].nbytes
    with pytest.raises(sp.SpatialLayerMemoryError):
        column.inspect("dz", host_memory_budget_bytes=need - 1)
    with pytest.raises(sp.SpatialLayerMemoryError):
        column.inspect("dz", host_memory_budget_bytes=need + 10, host_reserve_bytes=11)
    with pytest.raises(sp.SpatialLayerMemoryError):
        column.inspect("dz", host_memory_budget_bytes=BIG, host_available_bytes_fn=lambda: need - 1)
    with pytest.raises(sp.SpatialLayerError):
        column.inspect("nope", host_memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        column.inspect("dz", host_memory_budget_bytes=BIG, host_available_bytes_fn=lambda: 1.5)
    column.inspect("dz", host_memory_budget_bytes=need, host_available_bytes_fn=lambda: need)  # exactly enough


def test_caller_mutation_after_preparation_does_not_change_the_column(cpu_backend):
    p = params(np.random.default_rng(2), 3, 8)
    spec = make_spec(p)
    col = prepare(spec, backend=cpu_backend)
    before = {n: host(col, n) for n in col.retained_names()}
    sha = col.parameters_sha256
    for key in ("edges", "res", "fc", "sat", "iface"):
        p[key][...] = p[key] * 0.5 + 1e-3  # the caller reuses/mutates its arrays
    for n, want in before.items():
        np.testing.assert_array_equal(host(col, n), want)
    assert col.parameters_sha256 == sha and not any(np.shares_memory(col._maps[n], p[k]) for n in col.retained_names() for k in p)


def test_preparation_never_modifies_inputs_and_mask_copy_is_owned(cpu_backend):
    from conftest import params as pr

    p = pr(np.random.default_rng(3), 3, 6)
    mask = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
    snap = {k: v.copy() for k, v in p.items()}
    mask0 = mask.copy()
    col = prepare(make_spec(p), backend=cpu_backend, mask=mask)
    for k in p:
        np.testing.assert_array_equal(p[k], snap[k])
    np.testing.assert_array_equal(mask, mask0)
    assert col.n_active == 4 and not np.shares_memory(col._mask, mask)
    mask[:] = False
    assert int(np.count_nonzero(col._mask)) == 4
