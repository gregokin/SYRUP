import json

import numpy as np
import pytest

from maple_syrup.case_import import Plot1ImportError
from maple_syrup.event_experiment import run_complete_plot1
from maple_syrup.sediment_event import SedimentEventError


def test_real_plot1_pause_and_fresh_load_continue_without_reset(plot1_case, tmp_path):
    common = {'implementation': 'array', 'report_every_s': 2., 'checkpoint_every_s': 2.}
    paused = run_complete_plot1(plot1_case, tmp_path / 'paused', pause_at_s=2., **common)
    checkpoint = paused.summary['last_checkpoint']
    assert paused.summary['status'] == 'paused'
    assert not (tmp_path / 'paused' / 'handoff').exists()
    resumed = run_complete_plot1(plot1_case, tmp_path / 'resumed', resume=checkpoint, pause_at_s=4., **common)
    full = run_complete_plot1(plot1_case, tmp_path / 'full', pause_at_s=4., **common)
    for name in ('hydrograph', 'sediment_hydrograph', 'cumulative_pickup_kg', 'cumulative_deposition_kg'):
        np.testing.assert_array_equal(getattr(resumed.outcome.result, name), getattr(full.outcome.result, name))
    assert resumed.summary['sediment_closure'] == full.summary['sediment_closure']
    assert resumed.summary['water_budget_before_reset'] == full.summary['water_budget_before_reset']
    assert json.loads((tmp_path / 'resumed' / 'completion_summary.json').read_text())['status'] == 'paused'
    with pytest.raises(SedimentEventError, match='identity'):
        run_complete_plot1(plot1_case, tmp_path / 'changed', resume=checkpoint, max_dt_s=.5, **common)
    with pytest.raises(SedimentEventError, match='new'):
        run_complete_plot1(plot1_case, tmp_path / 'paused', **common)
    with pytest.raises(Plot1ImportError, match='inside the case tree'):
        run_complete_plot1(plot1_case, plot1_case / 'forbidden', **common)
    assert not (plot1_case / 'forbidden').exists()


@pytest.mark.parametrize('value', [True, -1., float('nan'), float('inf')])
def test_invalid_maximum_time_refused_before_io(tmp_path, value):
    with pytest.raises(SedimentEventError, match='maximum end time'):
        run_complete_plot1(tmp_path / 'missing', tmp_path / 'out', max_end_s=value, implementation='array')
