"""Test-only helpers of tests/masy_compat: array comparison, a synthetic stand-in for the PUBLIC
`maple.core.types.voxel` module (installed in `sys.modules` by monkeypatch only), and a broken-import finder.
Nothing here is a production fallback or a fake MAPLE class used outside a test."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
from dataclasses import dataclass, fields, is_dataclass

import numpy as np

VOXEL_MODULE = 'maple.core.types.voxel'


def arrays(value):
    """Every NumPy array of a (nested) dataclass, in field order."""
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value):
        for f in fields(value):
            yield from arrays(getattr(value, f.name))


def assert_same_arrays(a, b):
    left, right = list(arrays(a)), list(arrays(b))
    assert len(left) == len(right) > 0
    for x, y in zip(left, right, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


@dataclass(frozen=True)
class ForeignWindowedColumn:
    """A class defined OUTSIDE MAPLE (its `__module__`/`__qualname__` differ from any MAPLE name) that a synthetic
    public voxel module re-exports as `WindowedVoxelColumnState`. Classification identity only: it has no physics."""

    n_voxels: int = 0
    pending_deep: object = None
    height_packet: object = None


@dataclass(frozen=True)
class ForeignDenseColumn:
    """Same, re-exported as `VoxelColumnState` by a synthetic module."""

    mass_kg: object = None


def synthetic_voxel_module(monkeypatch, **bindings):
    """Install (test-only) a module object as `sys.modules['maple.core.types.voxel']` carrying every public binding
    of the real module, then `bindings` (a value of `MISSING` deletes the name). Returns the synthetic module."""
    real = importlib.import_module(VOXEL_MODULE)
    module = types.ModuleType(VOXEL_MODULE)
    module.__dict__.update({k: v for k, v in vars(real).items() if not k.startswith('__')})
    for name, value in bindings.items():
        if value is MISSING:
            module.__dict__.pop(name, None)
        else:
            module.__dict__[name] = value
    monkeypatch.setitem(sys.modules, VOXEL_MODULE, module)
    return module


MISSING = object()


#: Native MAPLE physical apply / inventory / commit entries the SYRUP bed callers dispatch to (module, attribute).
#: Windowed entries are spied only when the bound MAPLE provides them.
NATIVE_PHYSICAL_ENTRIES = (
    ('maple.water', 'apply_water_process_demand'),
    ('maple.aeolian.flux.step', 'bed_inventory_by_class_kg'),
    ('maple.surface.topographic_commit.commit', 'is_already_committed'),
    ('maple.surface.topographic_commit.commit', 'commit_topography'),
    ('maple.surface.topographic_commit.triggers', 'evaluate_commit_triggers'),
    ('maple.surface.active_layer.diagnostics', 'diagnose_combined_surface_state'),
    ('maple.water.windowed_step', 'apply_windowed_water_process_demand'),
    ('maple.surface.voxels.windowed_inventory', 'windowed_bed_inventory_by_class_kg'),
    ('maple.surface.voxels.windowed_inventory', 'column_mass_by_class_kg'),
    ('maple.surface.topographic_commit.windowed_commit', 'is_windowed_already_committed'),
    ('maple.surface.topographic_commit.windowed_commit', 'stage_windowed_topography'),
    ('maple.surface.topographic_commit.windowed_commit', 'windowed_committed_surface_elevation_m'),
    ('maple.surface.voxels.windowed_pending', 'publish_pending_column'),
)


class NativeEntrySpy:
    """Callable stand-in for a native MAPLE entry that fails loudly. The marker lets teardown prove no alias survives."""

    _masy_native_spy = True

    def __init__(self, label, calls):
        self.label, self.calls = label, calls

    def __call__(self, *args, **kwargs):
        self.calls.append(self.label)
        raise AssertionError(f'native MAPLE entry {self.label} reached by an invalid bed representation')


def resolve_native_entries(entries):
    """Import EVERY available entry module and capture the real function bindings BEFORE any spy exists, so a module that
    is first imported here executes its `from module import function` against real functions only. Modules absent from the
    bound MAPLE (`ModuleNotFoundError` naming the module itself) are skipped; any other failure propagates.
    Returns `[(module, attribute, original)]` for attributes present in the module namespace."""
    import importlib

    modules = {}
    for module_name, _ in entries:
        if module_name in modules:
            continue
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            modules[module_name] = None
    resolved = []
    for module_name, attribute in entries:
        module = modules[module_name]
        if module is not None and attribute in vars(module):
            resolved.append((module, attribute, vars(module)[attribute]))
    return resolved


def leaked_native_spies():
    """`module.attribute` names of every loaded `maple*` / `maple_syrup*` module attribute that is still a `NativeEntrySpy`."""
    leaks = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.split('.')[0] in ('maple', 'maple_syrup'):
            continue
        for attribute, value in list(vars(module).items()):
            if isinstance(value, NativeEntrySpy):
                leaks.append(f'{name}.{attribute}')
    return leaks


class SyntheticVoxelImportFailure(RuntimeError):
    pass


class _BrokenLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise SyntheticVoxelImportFailure('synthetic failure while executing maple.core.types.voxel')


class _BrokenVoxelFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == VOXEL_MODULE:
            return importlib.util.spec_from_loader(fullname, _BrokenLoader())
        return None


def install_broken_voxel_import(monkeypatch):
    """A PRESENT voxel module whose import raises (test-only): the cached module is removed and a finder whose loader
    fails is placed first. Both are restored by monkeypatch."""
    importlib.import_module(VOXEL_MODULE)  # make sure the real module is cached so monkeypatch restores it
    monkeypatch.delitem(sys.modules, VOXEL_MODULE)
    monkeypatch.setattr(sys, 'meta_path', [_BrokenVoxelFinder(), *sys.meta_path])
