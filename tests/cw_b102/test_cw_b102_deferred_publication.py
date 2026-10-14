"""CW-B102: opt-in deferred deep publication (`SedimentEventControl.deep_publication = "boundary"`).

In-process unit checks (control validation, identity / bundle omission at defaults, the narrow M94 append-cap
recognizer, static change scope against the accepted B100 freeze) plus one CPU subprocess pair of the focused driver
(step vs boundary on the seeded resident-window fixture). The full CPU / GPU0 matrix (callbacks, flush / redo, real M94
overflow, injected failures and I/O orphans, completion and own-checkpoint restart) is run by the task orchestrator
`agent_handoffs/tasks/conservative_water/cw_b102_run_checks.py`.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
B100 = Path('/home/okin/SYRUP/outputs/conservative_water/cw_b100_frozen')
M94 = Path('/home/okin/SYRUP/outputs/dependencies/maple_cw_b97_verified_tests_frozen')
sys.path.insert(0, str(TREE / 'src'))

from maple_syrup.event_experiment import control_identity  # noqa: E402
from maple_syrup.routing import RoutingStepRejected  # noqa: E402
from maple_syrup.sediment_bed import (  # noqa: E402
    BedContext,
    is_carried_append_cap_refusal,
    resolved_journal_cap,
)
from maple_syrup.sediment_event import SedimentEventControl, SedimentEventError  # noqa: E402


def csr_control(**kw):
    return SedimentEventControl(phase_storage='csr', **kw)


def test_default_control_is_step_and_identity_omits_the_new_keys():
    c = SedimentEventControl().validated()
    assert c.deep_publication == 'step' and c.deep_publication_high_water_records is None
    rec = control_identity(c)
    assert 'deep_publication' not in rec and 'deep_publication_high_water_records' not in rec
    b = csr_control(deep_publication='boundary', deep_publication_high_water_records=10).validated()
    rec = control_identity(b)
    assert rec['deep_publication'] == 'boundary' and rec['deep_publication_high_water_records'] == 10


@pytest.mark.parametrize('kw,match', [
    ({'deep_publication': 'later'}, 'must be one of'),
    ({'deep_publication': 'boundary'}, 'needs deep_publication_high_water_records'),
    ({'deep_publication': 'boundary', 'deep_publication_high_water_records': 0}, 'high_water'),
    ({'deep_publication': 'boundary', 'deep_publication_high_water_records': True}, 'high_water'),
    ({'deep_publication_high_water_records': 5}, "is a 'boundary' option"),
])
def test_control_validation_refusals(kw, match):
    with pytest.raises(SedimentEventError, match=match):
        csr_control(**kw).validated()
    with pytest.raises(SedimentEventError, match='characteristic CSR'):
        SedimentEventControl(deep_publication='boundary', deep_publication_high_water_records=5).validated()


def _ctx(cap):
    return BedContext(None, None, 1e-9, None, None, None, window_max_journal_records=cap)


def test_resolved_cap_is_explicit_context_or_the_actual_m94_default():
    from maple.surface.voxels.deep_store import DEFAULT_MAX_JOURNAL_RECORDS
    assert resolved_journal_cap(_ctx(4_000_000)) == 4_000_000
    assert resolved_journal_cap(_ctx(None)) == DEFAULT_MAX_JOURNAL_RECORDS == 1_000_000


def test_recognizer_accepts_only_the_exact_append_cap_text_with_a_carried_journal():
    ctx = _ctx(4_000_000)
    good = ValueError('pending journal exceeds max_journal_records=4000000')
    assert is_carried_append_cap_refusal(good, ctx, 1)
    assert not is_carried_append_cap_refusal(good, ctx, 0)  # nothing carried: a genuine single-step overflow
    for text in ('pending journal exceeds max_journal_records=1000000',  # another cap
                 'prior pending journal exceeds max_journal_records',  # R1 oversized prior journal
                 'journal has 5 records, above max_records=4000000; this is a hard limit of the attempt, never '
                 'silently truncated',  # R3
                 'journal has 5 records, above max_records=4000000',  # R4
                 'pending journal exceeds max_journal_records=4000000 ',  # reworded / padded
                 'injected late candidate refusal'):
        assert not is_carried_append_cap_refusal(ValueError(text), ctx, 1), text
    assert not is_carried_append_cap_refusal(RoutingStepRejected(str(good)), ctx, 1)  # subclass with the same text
    assert not is_carried_append_cap_refusal(RuntimeError(str(good)), ctx, 1)


def _defs(path):
    tree = ast.parse(Path(path).read_text())
    return {n.name: ast.dump(n) for n in tree.body if hasattr(n, 'name')}


def test_change_scope_against_the_accepted_b100_freeze():
    changed = {}
    for path in sorted((B100 / 'src/maple_syrup').glob('*.py')):
        new = TREE / 'src/maple_syrup' / path.name
        if new.read_bytes() != path.read_bytes():
            a, b = _defs(path), _defs(new)
            changed[path.name] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert changed == {
        'checkpoint.py': [],  # module-level opt-in default table only
        'complete_event.py': ['complete_event'],
        'event_experiment.py': ['control_identity', 'run_complete_case'],
        'sediment_bed.py': ['bound_maple_package_digest', 'is_carried_append_cap_refusal', 'pending_journal_records',
                            'resolved_journal_cap'],  # plus the module constants below
        'sediment_event.py': ['SedimentEventControl', 'SedimentEventResult', 'evolve_sediment_event'],
    }, changed
    import maple_syrup.sediment_bed as sb
    import maple_syrup.sediment_event as se
    assert se.DEEP_PUBLICATIONS == ('step', 'boundary')
    assert sb.JOURNAL_RECOGNIZER_MAPLE_DIGEST == 'eb14da66e0ffee1e11cd37cc98d3379b01b9e8f54e562ad0d12fc22afb11abd6'
    # the public / private step bodies are untouched
    a, b = _defs(B100 / 'src/maple_syrup/sediment_event.py'), _defs(TREE / 'src/maple_syrup/sediment_event.py')
    for name in ('sediment_coupled_step', '_sediment_accounting_step', '_sediment_coupled_core', '_commit_and_reroute',
                 '_transport_with_substep_retry'):
        assert a[name] == b[name], name


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('b102')
    out = {}
    for mode in ('step', 'boundary'):
        env = {'HOME': os.environ.get('HOME', '/tmp'), 'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
               'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': f'{M94}/editable:{M94}/source/src',
               'MAPLE_SYRUP_EXPECTED_MAPLE_ROOT': str(M94 / 'source'), 'NUMBA_CACHE_DIR': str(tmp / 'numba')}
        (tmp / mode).mkdir()
        proc = subprocess.run([sys.executable, '-B', str(HERE / 'b102_driver.py'), '--tree', str(TREE), '--backend',
                               'numpy', '--scenario', 'run', '--mode', mode, '--tmp', str(tmp / mode), '--out',
                               str(tmp / f'{mode}.json')], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-4000:]
        out[mode] = json.loads((tmp / f'{mode}.json').read_text())['result']
    return out


def test_step_and_boundary_are_physically_identical_with_carried_staged_commits(pair):
    st, bd = pair['step'], pair['boundary']
    assert [s['digest'] for s in st['per_step']] == [s['digest'] for s in bd['per_step']]
    assert [(b['t'], b['dense_sha256']) for b in st['boundaries']] == [(b['t'], b['dense_sha256']) for b in bd['boundaries']]
    assert [s['staged'] for s in bd['per_step']] == [True] * 6 + [False] * 6
    assert [s['commit_count'] for s in bd['per_step']][:6] == [0, 1, 1, 2, 2, 3]  # staged commits with a carried journal
    assert bd['deep_publication']['n_publications'] == 1 and st['deep_publication'] is None
    assert bd['final_published'] and all(b['published'] for b in bd['boundaries'])
    assert all(not any(p['alive'].values()) for p in st['probes'] + bd['probes'])  # B100 lifetime kept
    assert st['inputs_unchanged'] and bd['inputs_unchanged']


# --- CW-B102C1 regressions (CPU subprocesses of the focused driver / storage fixture) --------------------------------
def _child(tmp, script, *args):
    env = {'HOME': os.environ.get('HOME', '/tmp'), 'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
           'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': f'{M94}/editable:{M94}/source/src',
           'MAPLE_SYRUP_EXPECTED_MAPLE_ROOT': str(M94 / 'source'), 'NUMBA_CACHE_DIR': str(tmp / 'numba')}
    work = tmp / 'work'
    work.mkdir()
    proc = subprocess.run([sys.executable, '-B', str(HERE / script), *args, '--tmp', str(work), '--out',
                           str(tmp / 'out.json')], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    data = json.loads((tmp / 'out.json').read_text())
    return data.get('result', data)


def test_post_flush_exceptions_propagate_unchanged_and_the_failed_payload_is_dead(tmp_path):
    r = _child(tmp_path, 'b102_driver.py', '--tree', str(TREE), '--backend', 'numpy', '--scenario', 'retry',
               '--mode', 'boundary')
    dead = {'failed_payload_alive': False, 'exception_active': False}
    ok = r['flush_then_ok']
    assert ok['status'] == 'ok' and ok['deep_publication']['n_journal_flushes'] == 1 and ok['n_rejected'] == 0
    assert ok['digests'] == r['reference']['digests'] and ok['publication_probes'][0] == dead
    assert ok['calls'][2]['failed_payload_alive_at_entry'] is False and not ok['calls'][2]['exception_active_at_entry']
    for third, typ in (('routing', 'RoutingStepRejected'), ('transport', 'TransportStepRejected'),
                       ('valueerror', 'ValueError')):
        x = r[f'post_flush_{third}']
        assert x['status'] == 'error' and x['type'] == typ and x['message'].startswith('B102C1 post-flush')
        assert len(x['calls']) == 3 and x['calls'][2]['dt_s'] == x['calls'][1]['dt_s']  # no 4th attempt, no halving
        assert x['publication_probes'][0] == dead and x['inputs_unchanged'] and x['preexisting_files_preserved']
    x = r['first_ordinary_rejection']
    assert x['status'] == 'ok' and x['n_rejected'] == 1 and x['calls'][2]['dt_s'] == 0.5 * x['calls'][1]['dt_s']


def test_invalid_combinations_are_refused_before_input_validation(tmp_path):
    e = _child(tmp_path, 'b102_driver.py', '--tree', str(TREE), '--backend', 'numpy', '--scenario', 'entry')
    for k in ('boundary_public_hook', 'both_hooks_step', 'private_not_callable_step', 'boundary_on_boundary_not_callable'):
        assert e[k]['refused'] and e[k]['check_event_inputs_calls_before_refusal'] == 0, (k, e[k])
    assert e['default_no_hooks']['check_event_inputs_calls'] == 1


def test_storage_contract_staggered_read_and_overwrite_of_pending_keys(tmp_path):
    out = {}
    for mode in ('step', 'deferred'):
        sub = tmp_path / mode
        sub.mkdir()
        out[mode] = _child(sub, 'b102c1_storage.py', '--backend', 'numpy', '--mode', mode)
    s, f = out['step'], out['deferred']
    assert [x['logical_sha256'] for x in s['rows']] == [x['logical_sha256'] for x in f['rows']]
    assert [x['resident'] for x in s['rows']] == [x['resident'] for x in f['rows']]
    assert s['final_dense_sha256'] == f['final_dense_sha256'] and s['final_class_inventory_hex'] == f['final_class_inventory_hex']
    assert [x['overlay_records_applied'] for x in f['rows']] == [0, 0, 124, 124]  # pending keys READ via the overlay
    assert f['rows'][3]['appended_records_this_transaction'] == 24  # and OVERWRITTEN (canonical count unchanged)
    assert f['rows'][3]['canonical_journal_records'] == f['rows'][2]['canonical_journal_records'] == 372
    assert s['deep_store_counters']['versions_published'] == 3 and f['deep_store_counters']['versions_published'] == 1
    assert s['preexisting_files_preserved'] and f['preexisting_files_preserved']
