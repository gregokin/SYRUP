"""CW-C2: the host checkpoint bundle carries the opt-in graph policy, the terminal-water completion option and
the terminal inventories, reconstructs the pit graph / network / grid on load, resumes bitwise in the same
backend, and keeps the strict bundle encoding byte-for-byte as before (no optional fields written for the
strict default; a bundle without them decodes to the strict default).

Nothing here was run by its author (file-only tools); the root records the executed results.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("maple")

from pit_helpers import pit_inputs
from test_checkpoint_backend import (
    IDENTITY,
    arrays_any,
    assert_bitwise,
    assert_science_equal_any,
)
from test_complete_backend import fixture_xp
from test_coupled_backends import H

from maple_syrup.checkpoint import load_checkpoint, save_checkpoint
from maple_syrup.complete_event import CompletionPolicy, DryHandoff, complete_event
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_event import SedimentEventControl, SedimentEventError
from maple_syrup.storm import StormControl


def complete(inputs, schedule, *, policy, **kw):
    control = SedimentEventControl(storm=StormControl(max_dt_s=1.0))
    return complete_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"], schedule,
                          inputs["vegetation"], inputs["sediment"], max_end_s=900.0, control=control,
                          policy=policy, report_every_s=2.0, **kw)


def manifest(path):
    return json.loads((path / "checkpoint.json").read_text())


def test_strict_bundle_encoding_is_unchanged_and_decodes_to_the_strict_default(tmp_path):
    args, kw = fixture_xp(np, wet=True)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    path = tmp_path / "strict"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    m = manifest(path)
    policy_fields = m["payload"]["fields"]["policy"]["fields"]
    terrain_fields = m["payload"]["fields"]["result"]["fields"]["state"]["state"]["terrain"]["fields"]
    assert "terminal_storage_water" not in policy_fields
    assert not ({"allow_masked_nodata", "nodata_value", "allow_pit_storage", "allow_flat_storage"} & set(terrain_fields))
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    assert restored.policy.terminal_storage_water is False and restored.result.state.terrain.strict
    assert restored.result.state.network.n_terminal == 0 and restored.result.state.grid.terminal is None


def test_pit_event_wet_pause_round_trips_its_policy_terminal_inventories_and_resumes_bitwise(tmp_path):
    schedule = constant_rainfall(0.0, 10.0, 100.0)
    policy = CompletionPolicy(hold_s=10.0, terminal_storage_water=True)
    inputs = pit_inputs(np, ksat=1e-5, pit_ksat=0.0)
    ctx, column, sediment = inputs["ctx"], inputs["column"], inputs["sediment"]
    paused = complete(inputs, schedule, policy=policy, checkpoint_callback=lambda p: p.result.state.t_s >= 6.0)
    state = paused.result.state
    assert paused.status == "wet" and state.t_s == 6.0
    assert np.any(H(state.bed.water.mobile_mass_by_cell_class_kg) > 0) and np.any(H(state.bed.ledger.pending_bed_mass_change_kg) != 0)
    assert H(state.storm.depth_m)[0, 0] > 0.0 and not np.any(H(state.bed.water.mobile_mass_by_cell_class_kg)[0, 0])
    path = tmp_path / "pit"
    save_checkpoint(path, paused, ctx, column, sediment, IDENTITY)
    m = manifest(path)
    assert m["payload"]["fields"]["policy"]["fields"]["terminal_storage_water"] is True
    terrain_fields = m["payload"]["fields"]["result"]["fields"]["state"]["state"]["terrain"]["fields"]
    assert terrain_fields["allow_pit_storage"] is True and terrain_fields["allow_flat_storage"] is False
    assert terrain_fields["allow_masked_nodata"] is False and terrain_fields["nodata_value"] is None
    restored = load_checkpoint(path, ctx, column, sediment, IDENTITY)
    rs = restored.result.state
    assert restored.policy == policy and rs.terrain.build_options() == state.terrain.build_options()
    assert np.array_equal(rs.terrain.initial_full_elevation_m, state.terrain.initial_full_elevation_m)
    assert np.array_equal(rs.graph.pit_storage, state.graph.pit_storage) and rs.graph.policy == state.graph.policy
    assert rs.graph.input_sha256 == state.graph.input_sha256
    assert rs.network.n_terminal == 1 and rs.grid.terminal is not None and bool(H(rs.grid.terminal)[0, 0])
    assert_bitwise(rs.network.terminal_flat, state.network.terminal_flat, "terminal_flat")
    paused_arrays, restored_arrays = dict(arrays_any(state)), dict(arrays_any(rs))
    assert paused_arrays.keys() == restored_arrays.keys()
    for name, value in paused_arrays.items():
        assert_bitwise(value, restored_arrays[name], name)
    full = complete(inputs, schedule, policy=policy)
    resumed = complete({**inputs, "state0": rs}, schedule, policy=policy, continuation=restored)
    assert isinstance(full, DryHandoff) and isinstance(resumed, DryHandoff)
    assert_science_equal_any(full, resumed)
    assert resumed.accounting["terminal_storage_removed_m3"] == full.accounting["terminal_storage_removed_m3"] > 0.0
    # the completed handoff round-trips with its terminal accounting and cannot replay
    handoff = tmp_path / "handoff"
    save_checkpoint(handoff, full, ctx, column, sediment, IDENTITY)
    loaded = load_checkpoint(handoff, ctx, column, sediment, IDENTITY, allow_complete=True)
    assert_science_equal_any(full, loaded)
    assert loaded.accounting["terminal_storage_cells"] == 1
    with pytest.raises(SedimentEventError, match="wet checkpoint"):
        complete({**inputs, "state0": loaded.dry_state}, schedule, policy=policy, continuation=loaded)
    # a changed policy option refuses the resume
    with pytest.raises(SedimentEventError, match="controls changed"):
        complete({**inputs, "state0": rs}, schedule, policy=CompletionPolicy(hold_s=10.0), continuation=restored)
