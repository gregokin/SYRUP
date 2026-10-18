"""MASY-03 prerequisite A: the two native checkpoint registries (`checkpoint._registry`, `pond_checkpoint._native_registry`)
built from `checkpoint._native_ledger_types` (not run by its author; file-only tools).

The first test reads the ACTUAL bound MAPLE and asserts whichever branch it has (compact present or truly absent); it never
skips. Every other test is a PURE registry / refusal path on a SEPARATE synthetic module object installed by the
`ledger_binding` fixture (the real module is never edited; its exact restoration is proven at teardown). The save-refusal
tests build their genuine native state BEFORE any binding is installed and only check that the save is refused with the
propagated native error and writes nothing; no physical step runs under a synthetic binding."""

from __future__ import annotations

import dataclasses
import importlib
import types

import numpy as np
import pytest
from checkpoint_compat_support import LEDGER_MODULE, MISSING, ArrayStore, SyntheticLedgerImportFailure

from maple_syrup import checkpoint
from maple_syrup import pond_checkpoint as pc
from maple_syrup.pond_checkpoint import PondCheckpointError
from maple_syrup.sediment_bed import BedIntegrationError

#: Every non-ledger type of the legacy whitelist (unchanged by prerequisite A).
LEGACY_OTHER = ('CompletionPolicy', 'DryHandoff', 'EventProgress', 'QuietProgress', 'SedimentEventControl',
                'SedimentEventResult', 'StormControl', 'StormState', 'TerrainReference', 'CommittedTopographyState',
                'WaterState')


def _real():
    return importlib.import_module(LEDGER_MODULE)


def _registries():
    return checkpoint._registry(), pc._native_registry()


# --- the actual bound MAPLE (both branches asserted, never skipped) ----------------------------------------------------
def test_registries_bind_exactly_the_public_ledger_classes_of_the_bound_maple():
    from maple.core.types.topographic_commit import CommittedTopographyState

    namespace = vars(_real())
    dense = namespace['SedimentLedgerState']
    assert isinstance(dense, type)
    legacy, pond = _registries()
    assert legacy[dense.__name__] is dense and pond[dense.__name__] is dense
    assert pond[CommittedTopographyState.__name__] is CommittedTopographyState
    assert set(LEGACY_OTHER) <= set(legacy)
    if 'CompactSedimentLedgerState' in namespace:
        compact = namespace['CompactSedimentLedgerState']
        assert checkpoint._native_ledger_types() == (dense, compact)
        assert legacy[compact.__name__] is compact and pond[compact.__name__] is compact
        assert set(legacy) == set(LEGACY_OTHER) | {dense.__name__, compact.__name__}
        assert set(pond) == {dense.__name__, compact.__name__, CommittedTopographyState.__name__}
    else:
        assert checkpoint._native_ledger_types() == (dense,)
        assert 'CompactSedimentLedgerState' not in legacy and 'CompactSedimentLedgerState' not in pond
        assert set(legacy) == set(LEGACY_OTHER) | {dense.__name__}
        assert set(pond) == {dense.__name__, CommittedTopographyState.__name__}


# --- optional compact class: absence / presence ------------------------------------------------------------------------
def test_truly_absent_compact_class_is_omitted_from_both_registries(ledger_binding):
    dense = vars(_real())['SedimentLedgerState']
    ledger_binding.module(CompactSedimentLedgerState=MISSING)
    assert checkpoint._native_ledger_types() == (dense,)
    for registry in _registries():
        assert registry[dense.__name__] is dense
        assert 'CompactSedimentLedgerState' not in registry


def test_module_getattr_synthesis_is_not_presence(ledger_binding):
    """Presence is literal key membership in `vars(module)`; a module `__getattr__` that would answer the name is ignored."""
    dense = vars(_real())['SedimentLedgerState']
    module = ledger_binding.module(CompactSedimentLedgerState=MISSING)
    module.__dict__['__getattr__'] = lambda name: dense
    assert getattr(module, 'CompactSedimentLedgerState') is dense  # what attribute access would have produced
    assert checkpoint._native_ledger_types() == (dense,)
    for registry in _registries():
        assert 'CompactSedimentLedgerState' not in registry


def test_present_class_binding_is_included_by_identity_under_its_own_name(ledger_binding):
    """Registry-only: a present class binding is included as the identical class object, keyed by its own `__name__` (the
    existing whitelist keying). The class has no fields or physics and is never encoded, decoded or stepped."""

    class RegistryProbeOnly:
        pass

    dense = vars(_real())['SedimentLedgerState']
    ledger_binding.module(CompactSedimentLedgerState=RegistryProbeOnly)
    assert checkpoint._native_ledger_types() == (dense, RegistryProbeOnly)
    for registry in _registries():
        assert registry['RegistryProbeOnly'] is RegistryProbeOnly
        assert 'CompactSedimentLedgerState' not in registry


def test_reexported_dense_class_is_accepted(ledger_binding):
    """The synthetic module is a different module object re-exporting the real class: accepted by class identity."""
    dense = vars(_real())['SedimentLedgerState']
    module = ledger_binding.module()
    # Once installed, the import resolves to the synthetic module; distinctness is against the retained real module.
    assert _real() is module
    assert module is not ledger_binding.real and vars(module)['SedimentLedgerState'] is dense
    for registry in _registries():
        assert registry[dense.__name__] is dense


# --- present non-class bindings: refused, never treated as absence -----------------------------------------------------
def _instance():
    return _real().zeros_sediment_ledger_state(1, 1, 1)  # a genuine ledger INSTANCE (not a class)


NONCLASS = {
    'none': lambda: None,
    'int': lambda: 3,
    'function': lambda: (lambda *a, **k: None),
    'object': lambda: object(),
    'module': lambda: types.ModuleType('not_a_class'),
    'ledger_instance': _instance,
}


@pytest.mark.parametrize('name', ['CompactSedimentLedgerState', 'SedimentLedgerState'])
@pytest.mark.parametrize('kind', sorted(NONCLASS))
def test_present_nonclass_binding_is_refused_by_both_registries(ledger_binding, name, kind):
    value = NONCLASS[kind]()
    ledger_binding.module(**{name: value})
    for build in (checkpoint._native_ledger_types, checkpoint._registry, pc._native_registry):
        with pytest.raises(BedIntegrationError, match=f'{name} is present but not a class'):
            build()


# --- required Dense absence and broken modules: native errors propagate unchanged --------------------------------------
def test_required_dense_class_absent_raises_attribute_error(ledger_binding):
    ledger_binding.module(SedimentLedgerState=MISSING)
    for build in (checkpoint._native_ledger_types, checkpoint._registry, pc._native_registry):
        with pytest.raises(AttributeError, match='lacks required SedimentLedgerState'):
            build()


def test_none_module_entry_propagates_module_not_found(ledger_binding):
    ledger_binding.none_entry()
    for build in (checkpoint._native_ledger_types, checkpoint._registry, pc._native_registry):
        with pytest.raises(ModuleNotFoundError):
            build()


def test_module_execution_error_propagates_unchanged(ledger_binding):
    ledger_binding.broken_import()
    for build in (checkpoint._native_ledger_types, checkpoint._registry, pc._native_registry):
        with pytest.raises(SyntheticLedgerImportFailure) as info:
            build()
        assert type(info.value) is SyntheticLedgerImportFailure  # not wrapped, not converted to ImportError


# --- the codecs refuse a foreign object named like the compact class ----------------------------------------------------
def test_foreign_dataclass_named_compact_is_refused_by_the_pond_encoder():
    """Refusal only: a test-local dataclass that merely carries the NAME is never the registered class (whether the bound
    MAPLE has the genuine compact class or not)."""

    @dataclasses.dataclass(frozen=True)
    class CompactSedimentLedgerState:
        pending_bed_mass_change_kg: object = None

    with pytest.raises(PondCheckpointError, match='unregistered native type CompactSedimentLedgerState'):
        pc._native_encode(CompactSedimentLedgerState(np.zeros((1, 1, 1))), ArrayStore().add)


# --- saves are refused before any filesystem mutation ------------------------------------------------------------------
REFUSALS = {
    'compact_none': (lambda b: b.module(CompactSedimentLedgerState=None), BedIntegrationError),
    'compact_function': (lambda b: b.module(CompactSedimentLedgerState=lambda: None), BedIntegrationError),
    'dense_absent': (lambda b: b.module(SedimentLedgerState=MISSING), AttributeError),
    'module_none': (lambda b: b.none_entry(), ModuleNotFoundError),
    'module_broken': (lambda b: b.broken_import(), SyntheticLedgerImportFailure),
}


@pytest.fixture(scope='module')
def legacy_paused():
    """Genuine wet legacy continuation from the phase6 authored fixture (built under the REAL bindings)."""
    from phase6.test_complete_event import fixture

    from maple_syrup.complete_event import complete_event

    args, kw = fixture(wet=True)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    return args, paused


@pytest.fixture(scope='module')
def pond_paused():
    """Genuine wet pond state from the pond_checkpoint authored fixture (built under the REAL bindings)."""
    from pond_fixture import advance, make_case

    case = make_case(pickup_kg_per_class=5e-4)
    event = case.event()
    return event, advance(event, event.initial_state(case.bed, case.soil), 3)


@pytest.mark.parametrize('kind', sorted(REFUSALS))
def test_legacy_save_is_refused_with_the_native_error_and_writes_nothing(ledger_binding, legacy_paused, tmp_path, kind):
    args, paused = legacy_paused
    install, error = REFUSALS[kind]
    target = tmp_path / 'out' / 'checkpoint'
    install(ledger_binding)
    with pytest.raises(error):
        checkpoint.save_checkpoint(target, paused, args[1], args[2], args[6], {'case': 'refusal'})
    assert list(tmp_path.iterdir()) == []  # not even the parent directory was created


@pytest.mark.parametrize('kind', sorted(REFUSALS))
def test_pond_save_is_refused_with_the_native_error_and_writes_nothing(ledger_binding, pond_paused, tmp_path, kind):
    event, paused = pond_paused
    install, error = REFUSALS[kind]
    install(ledger_binding)
    with pytest.raises(error):
        pc.save_pond_checkpoint(tmp_path / 'ck', paused, event, {'case': 'refusal'})
    assert list(tmp_path.iterdir()) == []
