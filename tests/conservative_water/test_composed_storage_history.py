"""Provisional integration gates: real history, CSR/dense storage and wet restart together."""
from dataclasses import replace

import numpy as np
import pytest
from test_coupled_backends import H, compare_bitwise
from test_exchange_history import IDENTITY, complete, inputs

from maple_syrup.checkpoint import load_checkpoint, save_checkpoint
from maple_syrup.complete_event import DryHandoff
from maple_syrup.sediment_event import exchange_history_totals, initial_event_state


def _qualification(xp, storage, tmp_path):
    implementation = "numba" if xp is np else "cuda"
    s, ctx, control, schedule = inputs(xp, commit_interval_s=2.0, implementation=implementation)
    control = replace(control, phase_storage=storage)
    old = s["state0"]
    s["state0"] = initial_event_state(old.graph, old.terrain, old.bed, ctx,
                                      old.storm.soil_water_m, control=control)
    paused = complete(s, ctx, control, schedule, checkpoint_callback=lambda p: True)
    assert H(paused.result.state.bed.water.mobile_mass_by_cell_class_kg).sum() > 0
    before = exchange_history_totals(paused.result)
    path = tmp_path / "wet"
    save_checkpoint(path, paused, ctx, s["column"], s["sediment"], IDENTITY)
    restored = load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY)
    assert restored.control.phase_storage == storage and restored.control.record_exchange_history
    after = exchange_history_totals(restored.result)
    for key, value in before["arrays"].items():
        assert H(value).dtype == H(after["arrays"][key]).dtype
        assert H(value).tobytes() == H(after["arrays"][key]).tobytes()
    reference = complete(s, ctx, control, schedule)
    resumed = complete(dict(s, state0=restored.result.state), ctx, control, schedule, continuation=restored)
    assert isinstance(reference, DryHandoff) and isinstance(resumed, DryHandoff)
    compare_bitwise(reference.progress.result, resumed.progress.result)
    a = exchange_history_totals(reference.progress.result)
    b = exchange_history_totals(resumed.progress.result)
    assert a["history_complete"] and b["history_complete"]
    assert a["pending_ledger_empty"] and b["pending_ledger_empty"]
    assert a["n_commits_recorded"] == b["n_commits_recorded"]
    for key, value in a["arrays"].items():
        assert H(value).tobytes() == H(b["arrays"][key]).tobytes()
    from maple.core.backend import array_leaves
    aa, bb = list(array_leaves(reference.dry_state)), list(array_leaves(resumed.dry_state))
    assert len(aa) == len(bb)
    for a, b in zip(aa, bb, strict=True):
        assert H(a).dtype == H(b).dtype and H(a).shape == H(b).shape
        assert H(a).tobytes() == H(b).tobytes()
    for key in ("surface_removed_m3", "soil_removed_m3", "water_added_m3", "combined_water_residual_m3"):
        assert reference.accounting[key] == resumed.accounting[key]
    assert reference.accounting["sediment_closure"]["closed"] and resumed.accounting["sediment_closure"]["closed"]


@pytest.mark.parametrize("storage", ("dense", "csr"))
def test_recorded_storage_wet_checkpoint_and_completion_on_cpu(storage, tmp_path):
    _qualification(np, storage, tmp_path)


@pytest.mark.parametrize("storage", ("dense", "csr"))
def test_recorded_storage_wet_checkpoint_and_completion_on_actual_gpu(gpu, storage, tmp_path):
    _qualification(gpu, storage, tmp_path)
