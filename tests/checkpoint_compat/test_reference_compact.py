"""MASY-03 prerequisite A: the genuine compact ledger class keeps its codec behaviour where the bound MAPLE has it (the
reference root). Skips explicitly, and ONLY these tests, when the bound MAPLE truly lacks `CompactSedimentLedgerState`
(current `5b1a052e`); the class is never deleted, rebound or imitated (not run by its author; file-only tools).

Codec-level only: a genuine compact ledger from MAPLE's own factory goes through the pond native whitelist codec. The
compact / windowed checkpoint behaviour of whole events is covered by the existing `tests/pond_windowed_checkpoint`,
`tests/pond_checkpoint` and `tests/phase6` suites, which Codex runs on the reference root."""

from __future__ import annotations

import dataclasses
import importlib

import numpy as np
import pytest
from checkpoint_compat_support import LEDGER_MODULE, ArrayStore

from maple_syrup import checkpoint
from maple_syrup import pond_checkpoint as pc
from maple_syrup.pond_checkpoint import PondCheckpointError


@pytest.fixture
def ledger_module():
    module = importlib.import_module(LEDGER_MODULE)
    if 'CompactSedimentLedgerState' not in vars(module):
        pytest.skip('the bound MAPLE has no CompactSedimentLedgerState (genuine compact codec checks are not applicable)')
    return module


def _genuine_compact(module):
    """A genuine compact ledger (one present process) from MAPLE's factory, with distinct deterministic values in its
    pending and directional arrays so a round trip cannot pass vacuously (codec-only: never stepped or committed)."""
    ledger = module.zeros_sediment_ledger_state(2, 3, 2, present_processes=(3,))
    assert type(ledger) is vars(module)['CompactSedimentLedgerState']
    pending = np.arange(12, dtype=np.float64).reshape(2, 3, 2) * 1e-6
    positive = np.arange(12, dtype=np.float64).reshape(1, 2, 3, 2) * 1e-7
    return dataclasses.replace(ledger, pending_bed_mass_change_kg=pending, actual_mass_positive_kg=positive,
                               largest_residual_kg=1e-12)


def test_reference_registries_bind_the_genuine_compact_class(ledger_module):
    compact = vars(ledger_module)['CompactSedimentLedgerState']
    dense = vars(ledger_module)['SedimentLedgerState']
    assert checkpoint._native_ledger_types() == (dense, compact)
    assert checkpoint._registry()[compact.__name__] is compact
    assert pc._native_registry()[compact.__name__] is compact


def test_genuine_compact_ledger_round_trips_through_the_pond_native_codec(ledger_module):
    ledger = _genuine_compact(ledger_module)
    store = ArrayStore()
    node = pc._native_encode(ledger, store.add)
    assert node['type'] == 'CompactSedimentLedgerState'
    assert set(node['fields']) == {f.name for f in dataclasses.fields(type(ledger))}
    decoded = pc._native_decode(node, store.get)
    assert type(decoded) is type(ledger)
    assert decoded.present_processes == ledger.present_processes == (3,)
    assert decoded.largest_residual_kg == ledger.largest_residual_kg
    for f in dataclasses.fields(type(ledger)):
        a, b = getattr(ledger, f.name), getattr(decoded, f.name)
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape, f.name
            np.testing.assert_array_equal(a, b, err_msg=f.name)


def test_genuine_compact_node_without_its_compact_field_is_refused(ledger_module):
    """The old dense field set under the compact name is never accepted (no default is filled in)."""
    store = ArrayStore()
    node = pc._native_encode(_genuine_compact(ledger_module), store.add)
    node['fields'].pop('present_processes')
    with pytest.raises(PondCheckpointError, match='native fields of CompactSedimentLedgerState do not match'):
        pc._native_decode(node, store.get)
