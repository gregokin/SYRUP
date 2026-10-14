"""CW-F2 correction1: negative controls of the CW-B30 summary comparison's single tolerated additive key.

Only a validated top-level `restart_resource_bound` absent from the baseline summary is removed; any other difference,
an invalid record, a record in another place or a record lost by the candidate is refused. Genuine runner records are
validated by `test_restart_byte_bound.py` (dense default/explicit, window default/tightened)."""
from __future__ import annotations

import copy

import pytest
from test_bytes_vs_s27 import (
    RESTART_BOUND_APPLIES_TO,
    RESTART_BOUND_IDENTITY,
    RESTART_BOUND_KEY,
    RestartBoundRecordError,
    comparable_candidate_summaries,
    validate_restart_bound,
)

GIB = 1024**3


def record(**changes):
    value = {'max_checkpoint_uncompressed_bytes': GIB, 'resolution': 'default of the dense runner (1 GiB)',
             'applies_to': RESTART_BOUND_APPLIES_TO, 'identity': RESTART_BOUND_IDENTITY}
    value.update(changes)
    return value


def window_record(bound=10**8, host=10**8, resolution='resident-window default: equal to the explicit max_host_bytes cap'):
    return record(max_checkpoint_uncompressed_bytes=bound, resolution=resolution, max_host_bytes=host)


BASELINE = {'complete': {'status': 'complete', 'identity': {'schema': 's'}, 'time_s': 4.0, 'nested': {'a': 1}},
            'load': {'type': 'DryHandoff', 'accounting': {'x': 1}}}


def candidate(rec=None, **top):
    value = copy.deepcopy(BASELINE)
    value['complete'][RESTART_BOUND_KEY] = record() if rec is None else rec
    value['complete'].update(top)
    return value


VALID = [record(), record(max_checkpoint_uncompressed_bytes=12345, resolution='caller-explicit'), window_record(),
         window_record(bound=7, resolution='caller-explicit, tightened within max_host_bytes'),
         window_record(bound=10**8, resolution='caller-explicit, tightened within max_host_bytes')]


@pytest.mark.parametrize('rec', VALID)
def test_documented_records_are_tolerated_and_everything_else_stays_equal(rec):
    old = copy.deepcopy(BASELINE)
    new = candidate(rec)
    assert comparable_candidate_summaries(old, new) == old
    assert RESTART_BOUND_KEY in new['complete']  # the candidate input is not mutated


def test_baseline_without_the_field_is_still_valid_against_itself():
    assert comparable_candidate_summaries(BASELINE, copy.deepcopy(BASELINE)) == BASELINE


def test_record_in_both_trees_is_compared_strictly():
    old = candidate()
    assert comparable_candidate_summaries(old, candidate()) == old
    assert comparable_candidate_summaries(old, candidate(record(max_checkpoint_uncompressed_bytes=5,
                                                                resolution='caller-explicit'))) != old


INVALID = {
    'missing key': {k: v for k, v in record().items() if k != 'applies_to'},
    'extra key': record(note='x'),
    'window key on dense': record(max_host_bytes=GIB),
    'dense record missing host': {k: v for k, v in window_record().items() if k != 'max_host_bytes'},
    'bool bound': record(max_checkpoint_uncompressed_bytes=True, resolution='caller-explicit'),
    'zero bound': record(max_checkpoint_uncompressed_bytes=0, resolution='caller-explicit'),
    'negative bound': record(max_checkpoint_uncompressed_bytes=-1, resolution='caller-explicit'),
    'float bound': record(max_checkpoint_uncompressed_bytes=float(GIB)),
    'string bound': record(max_checkpoint_uncompressed_bytes=str(GIB)),
    'undocumented resolution': record(resolution='unbounded'),
    'default not 1 GiB': record(max_checkpoint_uncompressed_bytes=GIB + 1),
    'explicit equal to default': record(resolution='caller-explicit'),
    'applies_to text': record(applies_to='anything'),
    'identity text': record(identity='part of the identity'),
    'window default differs from host': window_record(bound=10**7),
    'window tightened above host': window_record(bound=10**8 + 1,
                                                 resolution='caller-explicit, tightened within max_host_bytes'),
    'invalid host': window_record(host=0, bound=10**8),
    'not a mapping': [GIB],
}


@pytest.mark.parametrize('name', sorted(INVALID))
def test_invalid_resource_record_is_refused(name):
    with pytest.raises(RestartBoundRecordError):
        validate_restart_bound(INVALID[name])
    with pytest.raises(RestartBoundRecordError):
        comparable_candidate_summaries(BASELINE, candidate(INVALID[name]))


@pytest.mark.parametrize('change', [
    {'unrelated_new_field': 1},  # an unrelated additive summary key
    {'time_s': 4.5},  # a changed existing value
    {'nested': {'a': 1, RESTART_BOUND_KEY: record()}},  # the record anywhere but the top level
])
def test_any_other_summary_difference_still_fails(change):
    assert comparable_candidate_summaries(BASELINE, candidate(**change)) != BASELINE


def test_record_outside_a_runner_summary_or_lost_by_the_candidate_fails():
    new = copy.deepcopy(BASELINE)
    new['load'][RESTART_BOUND_KEY] = record()  # a valid record on the load handoff record, not a runner summary
    assert comparable_candidate_summaries(BASELINE, new) != BASELINE
    lost = copy.deepcopy(candidate())
    assert comparable_candidate_summaries(lost, copy.deepcopy(BASELINE)) != lost  # baseline has it, candidate lost it
    with pytest.raises(RestartBoundRecordError, match='labels differ'):
        comparable_candidate_summaries(BASELINE, {**copy.deepcopy(BASELINE), 'extra': {}})
