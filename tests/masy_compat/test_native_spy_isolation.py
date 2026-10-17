"""Regression for the `native_spy` fixture: installing and undoing the native-entry spies must leave NO spy alias behind.

The defect it guards: spies were installed while the entry modules were still being imported lazily, so a module first
imported mid-way (e.g. `windowed_commit` doing `from ...windowed_inventory import column_mass_by_class_kg`) captured an
earlier SPY permanently, and later genuine windowed tests failed with "native MAPLE entry ... reached by an invalid bed
representation". Test bounds only; no MAPLE code is patched or relaxed. The tests are order dependent on purpose
(`test_a_*` before `test_b_*`); the root additionally runs this directory and then the windowed suites in one process."""

from __future__ import annotations

import sys
import types

from masy_compat_support import (
    NATIVE_PHYSICAL_ENTRIES,
    NativeEntrySpy,
    leaked_native_spies,
    resolve_native_entries,
)


def test_a_spies_are_installed_then_undone_without_alias(dense_bed, native_spy):
    resolved = resolve_native_entries(NATIVE_PHYSICAL_ENTRIES)
    assert resolved, 'no native entry resolved'
    # While active, the module attributes are spies (this is what the refusal tests rely on).
    assert all(isinstance(vars(module)[attribute], NativeEntrySpy) for module, attribute, _ in resolved)


def test_b_after_the_fixture_every_native_entry_is_genuine_again():
    assert leaked_native_spies() == []
    for module, attribute, original in resolve_native_entries(NATIVE_PHYSICAL_ENTRIES):
        assert not isinstance(original, NativeEntrySpy), f'{module.__name__}.{attribute} is still a spy'


def test_leak_detector_sees_a_captured_alias(monkeypatch):
    """Positive control for the detector: a module that captured a spy by `from module import function`."""
    probe = types.ModuleType('maple_syrup._masy_spy_probe')
    probe.captured = NativeEntrySpy('probe', [])
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    assert leaked_native_spies() == ['maple_syrup._masy_spy_probe.captured']


def test_resolution_precedes_every_spy(monkeypatch):
    """`resolve_native_entries` returns real functions even when called first; a second call under an active spy shows
    spies (so the fixture must, and does, resolve before patching)."""
    first = resolve_native_entries(NATIVE_PHYSICAL_ENTRIES)
    assert first and not any(isinstance(original, NativeEntrySpy) for _, _, original in first)
    module, attribute, _ = first[0]
    monkeypatch.setattr(module, attribute, NativeEntrySpy('x', []))
    second = {(m.__name__, a): o for m, a, o in resolve_native_entries(NATIVE_PHYSICAL_ENTRIES)}
    assert isinstance(second[(module.__name__, attribute)], NativeEntrySpy)
