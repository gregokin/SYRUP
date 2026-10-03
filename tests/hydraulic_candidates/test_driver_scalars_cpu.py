"""The experimental driver's host numbers (time of a new peak, report-row time/counters/smallest dt) are passed BY VALUE and must
still land in the right places. CPU only, no Numba. The device-side transfer trap lives in `test_driver_transfers.py` (GPU).
Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from cand_cases import build, field, schedule

pytest.importorskip("maple")

from maple_syrup.experimental_storm import (
    EXPERIMENT_HYDROGRAPH_COLUMNS,
    ExperimentalControl,
    evolve_experimental,
)

COLUMN = {name: i for i, name in enumerate(EXPERIMENT_HYDROGRAPH_COLUMNS)}


class ScriptedOutlet:
    """Wraps a solver; the k-th accepted step reports the scripted outlet discharge (everything else is untouched)."""

    def __init__(self, solver, values):
        self._solver, self._values, self._k = solver, list(values), 0

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def step(self, rain, state, dt):
        step = self._solver.step(rain, state, dt)
        value = self._values[self._k]
        self._k += 1
        return dataclasses.replace(step, outlet_discharge_m3_s=np.float64(value))


@pytest.mark.parametrize("method", ("explicit", "local_inertial"))
def test_the_time_of_peak_is_the_end_time_of_the_step_with_the_largest_outlet_discharge(method):
    # Peak bookkeeping only: a dry state with ZERO rain has nothing to route, so no step can be rejected (no CFL or positivity
    # retry, no extra accepted sub-steps) and exactly six 1 s steps run; the discharge each step reports is scripted.
    cs = build("valley:6x5", method, depth="dry", seed=1)
    scripted = ScriptedOutlet(cs.solver, [1.0, 3.0, 2.0, 7.0, 4.0, 7.0])  # a tie keeps the FIRST maximum (strict >)
    result = evolve_experimental(scripted, field(cs), schedule([0.0, 6.0], [0.0]), cs.state, 6.0,
                                 ExperimentalControl(max_dt_s=1.0), report_every_s=3.0)
    assert result.n_accepted_steps == 6 and result.n_rejected_attempts == 0
    assert float(result.peak_outlet_discharge_m3_s) == 7.0 and float(result.time_of_peak_outlet_s) == 4.0


@pytest.mark.parametrize("method", ("explicit", "local_inertial"))
def test_the_row_time_and_counters_written_by_value_are_exact(method):
    cs = build("valley:6x5", method, depth="dry", seed=2)
    result = evolve_experimental(cs.solver, field(cs), schedule([0.0, 8.0], [60.0]), cs.state, 8.0,
                                 ExperimentalControl(max_dt_s=0.5), report_every_s=2.0)
    rows = result.hydrograph
    np.testing.assert_array_equal(rows[:, COLUMN["t_s"]], result.boundaries)
    steps = rows[:, COLUMN["accepted_steps"]]
    assert np.all(np.diff(steps) > 0) and steps[-1] == result.n_accepted_steps == 16  # 8 s at 0.5 s
    assert rows[-1, COLUMN["rejected_attempts"]] == result.n_rejected_attempts == 0
    assert rows[-1, COLUMN["min_accepted_dt_s"]] == result.min_accepted_dt_s == 0.5
    assert 0.0 <= float(result.time_of_peak_outlet_s) <= 8.0


def test_the_first_report_row_is_written_even_before_any_peak_exists():
    cs = build("valley:6x5", "explicit", depth="dry", seed=3)
    result = evolve_experimental(cs.solver, field(cs), schedule([0.0, 4.0], [0.0]), cs.state, 4.0,
                                 ExperimentalControl(max_dt_s=1.0), report_every_s=2.0)
    assert float(result.peak_outlet_discharge_m3_s) == 0.0
    assert float(result.time_of_peak_outlet_s) == 0.0  # no step ever exceeded the initial zero: the start time is kept
    assert np.all(result.hydrograph[:, COLUMN["accepted_steps"]] > 0)
