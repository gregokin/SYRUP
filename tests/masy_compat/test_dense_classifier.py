"""`sediment_bed.is_windowed_bed`: exact class-object identity on the public `maple.core.types.voxel` module.

Synthetic module bindings are installed test-only (monkeypatch) on the bound voxel module or as a `sys.modules`
stand-in carrying the real public bindings. They exercise classification identity only; they are not evidence of
physical windowed storage."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from masy_compat_support import (
    MISSING,
    VOXEL_MODULE,
    ForeignDenseColumn,
    ForeignWindowedColumn,
    SyntheticVoxelImportFailure,
    install_broken_voxel_import,
    synthetic_voxel_module,
)

from maple_syrup.dependency import (
    REQUIRED_MAPLE_API,
    ApiRequirement,
    MapleApiIncompatibleError,
    check_required_api,
)
from maple_syrup.sediment_bed import BedIntegrationError, is_windowed_bed


def _voxel():
    return importlib.import_module(VOXEL_MODULE)


# --- genuine Dense, optional Windowed absent ----------------------------------------------------------------------------
def test_real_dense_bed_is_not_windowed(dense_bed):
    state = dense_bed[0]
    assert type(state.voxel_column) is _voxel().VoxelColumnState
    assert is_windowed_bed(state) is False


def test_truly_absent_windowed_class_is_tolerated(dense_bed, monkeypatch):
    monkeypatch.delattr(_voxel(), 'WindowedVoxelColumnState', raising=False)
    assert 'WindowedVoxelColumnState' not in vars(_voxel())
    assert is_windowed_bed(dense_bed[0]) is False


def test_module_getattr_synthesis_is_not_presence(dense_bed, monkeypatch):
    """Only the module namespace counts: a module `__getattr__` cannot synthesize the windowed class."""
    module = synthetic_voxel_module(monkeypatch, WindowedVoxelColumnState=MISSING)
    module.__getattr__ = lambda name: ForeignWindowedColumn if name == 'WindowedVoxelColumnState' else None
    assert is_windowed_bed(dense_bed[0]) is False
    with pytest.raises(BedIntegrationError, match='unsupported bed representation'):
        is_windowed_bed(replace(dense_bed[0], voxel_column=ForeignWindowedColumn()))


# --- present but invalid symbols ---------------------------------------------------------------------------------------
def _a_function():
    return None


@pytest.mark.parametrize('value', [None, 3, _a_function, 'WindowedVoxelColumnState'], ids=['None', 'int', 'function', 'str'])
def test_present_invalid_windowed_refuses_even_a_genuine_dense_bed(dense_bed, monkeypatch, value):
    monkeypatch.setattr(_voxel(), 'WindowedVoxelColumnState', value, raising=False)
    with pytest.raises(BedIntegrationError, match='WindowedVoxelColumnState is present but not a class'):
        is_windowed_bed(dense_bed[0])


@pytest.mark.parametrize('value', [None, 3, _a_function], ids=['None', 'int', 'function'])
def test_present_invalid_dense_refuses(dense_bed, monkeypatch, value):
    monkeypatch.delattr(_voxel(), 'WindowedVoxelColumnState', raising=False)
    monkeypatch.setattr(_voxel(), 'VoxelColumnState', value)
    with pytest.raises(BedIntegrationError, match='VoxelColumnState is present but not a class'):
        is_windowed_bed(dense_bed[0])


def _legacy_marker():
    """The old helper's private absence marker when the implementation still has one (so these tests fail on it), else an
    arbitrary fresh object. Either way it is a PRESENT non-class binding and must be refused, never read as absence."""
    from maple_syrup import sediment_bed

    return getattr(sediment_bed, '_MISSING', object())


def test_present_legacy_marker_windowed_binding_is_refused_not_absent(dense_bed, monkeypatch):
    monkeypatch.setattr(_voxel(), 'WindowedVoxelColumnState', _legacy_marker(), raising=False)
    assert 'WindowedVoxelColumnState' in vars(_voxel())
    with pytest.raises(BedIntegrationError, match='WindowedVoxelColumnState is present but not a class'):
        is_windowed_bed(dense_bed[0])


def test_present_legacy_marker_dense_binding_is_refused_not_absent(dense_bed, monkeypatch):
    monkeypatch.delattr(_voxel(), 'WindowedVoxelColumnState', raising=False)
    monkeypatch.setattr(_voxel(), 'VoxelColumnState', _legacy_marker())
    with pytest.raises(BedIntegrationError, match='VoxelColumnState is present but not a class'):
        is_windowed_bed(dense_bed[0])


def test_present_arbitrary_object_bindings_are_refused(dense_bed, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(_voxel(), 'WindowedVoxelColumnState', sentinel, raising=False)
    with pytest.raises(BedIntegrationError, match='WindowedVoxelColumnState is present but not a class'):
        is_windowed_bed(dense_bed[0])
    monkeypatch.delattr(_voxel(), 'WindowedVoxelColumnState')
    monkeypatch.setattr(_voxel(), 'VoxelColumnState', sentinel)
    with pytest.raises(BedIntegrationError, match='VoxelColumnState is present but not a class'):
        is_windowed_bed(dense_bed[0])


# --- required Dense absence and module import failures ------------------------------------------------------------------
def test_absent_required_dense_class_raises_attribute_error(dense_bed, monkeypatch):
    monkeypatch.delattr(_voxel(), 'WindowedVoxelColumnState', raising=False)
    monkeypatch.delattr(_voxel(), 'VoxelColumnState')
    with pytest.raises(AttributeError, match='lacks required VoxelColumnState'):
        is_windowed_bed(dense_bed[0])


def test_blocked_voxel_module_import_propagates(dense_bed, monkeypatch):
    state = dense_bed[0]
    monkeypatch.setitem(sys.modules, VOXEL_MODULE, None)
    with pytest.raises(ImportError):
        is_windowed_bed(state)


def test_broken_present_voxel_module_import_propagates_unchanged(dense_bed, monkeypatch):
    state = dense_bed[0]
    install_broken_voxel_import(monkeypatch)
    with pytest.raises(SyntheticVoxelImportFailure):
        is_windowed_bed(state)


# --- legitimate re-exports / aliases: the public binding is authoritative -------------------------------------------------
def test_reexported_windowed_class_is_recognized_by_identity(dense_bed, monkeypatch):
    synthetic_voxel_module(monkeypatch, WindowedVoxelColumnState=ForeignWindowedColumn)
    assert ForeignWindowedColumn.__module__ != VOXEL_MODULE
    assert ForeignWindowedColumn.__qualname__ != 'WindowedVoxelColumnState'
    assert is_windowed_bed(replace(dense_bed[0], voxel_column=ForeignWindowedColumn())) is True
    assert is_windowed_bed(dense_bed[0]) is False

    class WindowedSubclass(ForeignWindowedColumn):
        pass

    with pytest.raises(BedIntegrationError, match='unsupported bed representation'):
        is_windowed_bed(replace(dense_bed[0], voxel_column=WindowedSubclass()))


def test_reexported_dense_class_is_recognized_by_identity(dense_bed, monkeypatch):
    real_dense = _voxel().VoxelColumnState
    synthetic_voxel_module(monkeypatch, WindowedVoxelColumnState=MISSING, VoxelColumnState=ForeignDenseColumn)
    assert is_windowed_bed(replace(dense_bed[0], voxel_column=ForeignDenseColumn())) is False
    # The private definition is NOT authority: an instance of the original class is no longer the bound Dense class.
    assert type(dense_bed[0].voxel_column) is real_dense
    with pytest.raises(BedIntegrationError, match='unsupported bed representation'):
        is_windowed_bed(dense_bed[0])

    class DenseSubclass(ForeignDenseColumn):
        pass

    with pytest.raises(BedIntegrationError, match='unsupported bed representation'):
        is_windowed_bed(replace(dense_bed[0], voxel_column=DenseSubclass()))


def test_real_windowed_class_identity_when_available(dense_bed):
    windowed = vars(_voxel()).get('WindowedVoxelColumnState')
    if windowed is None:
        pytest.skip('the bound MAPLE has no WindowedVoxelColumnState (current committed MAPLE 5b1a052e lacks it); '
                    'real windowed classification needs a separately identified windowed-capable lineage')
    # Class identity only (no constructor, no physics): an uninitialised exact instance.
    instance = object.__new__(windowed)
    assert is_windowed_bed(replace(dense_bed[0], voxel_column=instance)) is True
    subclass = type('WindowedSubclass', (windowed,), {})
    with pytest.raises(BedIntegrationError, match='unsupported bed representation'):
        is_windowed_bed(replace(dense_bed[0], voxel_column=object.__new__(subclass)))


# --- unknown representations ---------------------------------------------------------------------------------------------
def test_unknown_and_dense_subclass_representations_refuse(dense_bed):
    state = dense_bed[0]
    dense = type(state.voxel_column)
    subclass = type('DenseSubclass', (dense,), {})
    for column in (object(), SimpleNamespace(mass_kg=state.voxel_column.mass_kg), subclass(mass_kg=state.voxel_column.mass_kg)):
        with pytest.raises(BedIntegrationError, match='unsupported bed representation'):
            is_windowed_bed(replace(state, voxel_column=column))


# --- dependency.py declarations ------------------------------------------------------------------------------------------
DENSE_REQUIREMENTS = (
    ApiRequirement("maple.core.types.voxel", "VoxelColumnState"),
    ApiRequirement("maple.aeolian.flux.step", "bed_inventory_by_class_kg", parameters=("voxel_column", "active_layer")),
    ApiRequirement(
        "maple.surface.topographic_commit.commit", "is_already_committed",
        parameters=("ledger", "committed_state", "voxel_column", "active_layer", "geometry",
                    "mass_resolution_kg", "current_time_s"),
    ),
)


def test_dense_requirements_are_declared_and_met():
    for requirement in DENSE_REQUIREMENTS:
        assert requirement in REQUIRED_MAPLE_API
    check_required_api(DENSE_REQUIREMENTS)


def test_check_required_api_reports_missing_dense_class(monkeypatch):
    monkeypatch.delattr(_voxel(), 'VoxelColumnState')
    with pytest.raises(MapleApiIncompatibleError, match=r'maple\.core\.types\.voxel\.VoxelColumnState: attribute missing'):
        check_required_api(DENSE_REQUIREMENTS)


def test_check_required_api_reports_changed_dense_signatures(monkeypatch):
    from maple.aeolian.flux import step
    from maple.surface.topographic_commit import commit

    monkeypatch.setattr(step, 'bed_inventory_by_class_kg', lambda column, layer: None)
    monkeypatch.setattr(commit, 'is_already_committed', lambda ledger, committed: None)
    with pytest.raises(MapleApiIncompatibleError) as info:
        check_required_api(DENSE_REQUIREMENTS)
    text = str(info.value)
    assert "bed_inventory_by_class_kg: missing parameter(s) ['voxel_column', 'active_layer']" in text
    assert "is_already_committed: missing parameter(s) ['committed_state'" in text
