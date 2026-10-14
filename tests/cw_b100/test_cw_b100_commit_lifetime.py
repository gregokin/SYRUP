"""CW-B100: the private event loop releases the PREVIOUS accepted state (bed / ledger / velocity / phase / storm) before
MAPLE's topographic commit allocates its fresh ledger (the R6 rotated-Chastre OOM site,
windowed_commit.py:588 -> sediment_ledger.py:896), with unchanged results.

Each test runs the B100 subprocess driver (`b100_driver.py`: an actual MAPLE resident-window bed with interval commits
on a 64 x 32 valley, deep store in tmp) once with the accepted R1 tree and once with THIS tree, against the same frozen
M94 MAPLE, CPU only here (CUDA hidden). The GPU0 variants are run by the task orchestrator
(agent_handoffs/tasks/conservative_water/cw_b100_run_checks.py), which probes GPU0 before every GPU process.
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
R1 = Path('/home/okin/SYRUP/outputs/conservative_water/cw_r1_rotated_chastre_frozen')
M94 = Path('/home/okin/SYRUP/outputs/dependencies/maple_cw_b97_verified_tests_frozen')
TIMING = {'result.commit_wall_s', 'result.first_step_cpu_s', 'result.first_step_wall_s', 'result.remaining_cpu_s',
          'result.remaining_wall_s'}


def drive(tree, tmp, name, *extra):
    work = tmp / name
    work.mkdir()
    env = {'HOME': os.environ.get('HOME', '/tmp'), 'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
           'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': f'{M94}/editable:{M94}/source/src',
           'MAPLE_SYRUP_EXPECTED_MAPLE_ROOT': str(M94 / 'source'), 'NUMBA_CACHE_DIR': str(tmp / 'numba')}
    out = tmp / f'{name}.json'
    proc = subprocess.run([sys.executable, '-B', str(HERE / 'b100_driver.py'), '--tree', str(tree), '--backend', 'numpy',
                           '--tmp', str(work), '--out', str(out), *extra], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(out.read_text())


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('b100')
    return drive(R1, tmp, 'r1'), drive(TREE, tmp, 'b100')


def test_only_the_event_loop_changed_from_r1():
    def defs(path):
        tree = ast.parse(Path(path).read_text())
        named = {n.name: ast.dump(n) for n in tree.body if hasattr(n, 'name')}
        return named, [ast.dump(n) for n in tree.body if not hasattr(n, 'name')]
    (a, ra), (b, rb) = defs(R1 / 'src/maple_syrup/sediment_event.py'), defs(TREE / 'src/maple_syrup/sediment_event.py')
    assert set(a) == set(b) and ra == rb
    assert sorted(k for k in a if a[k] != b[k]) == ['evolve_sediment_event']
    for path in sorted((R1 / 'src/maple_syrup').glob('*.py')):
        if path.name != 'sediment_event.py':
            assert (TREE / 'src/maple_syrup' / path.name).read_bytes() == path.read_bytes(), path.name


def test_previous_state_is_released_before_the_commit_fresh_ledger(pair):
    r1, new = pair
    assert r1['status'] == new['status'] == 'ok' and r1['n_commits'] == new['n_commits'] >= 3
    # R1 keeps every tracked previous-state array (ledger C-grids, window, active layer, water, availability,
    # velocity C-grid, phase, storm) alive while MAPLE allocates the commit's fresh ledger; B100 keeps none.
    assert all(p['n_prev_alive'] == len(p['prev_alive']) >= 9 for p in r1['probes'])
    assert all(p['n_prev_alive'] == 0 for p in new['probes'])
    saved = [p['used_bytes_before_fresh_ledger'] - q['used_bytes_before_fresh_ledger']
             for p, q in zip(r1['probes'], new['probes'], strict=True)]
    assert min(saved) >= new['c_grid_bytes'], saved
    assert new['tracemalloc_peak'] < r1['tracemalloc_peak']


def test_results_and_deep_store_are_raw_exact_to_r1(pair):
    r1, new = pair
    diff = sorted(k for k in set(r1['result_raw']) | set(new['result_raw'])
                  if r1['result_raw'].get(k) != new['result_raw'].get(k))
    assert set(diff) <= TIMING, diff
    assert len(r1['result_raw']) > 500
    assert r1['store_files'] == new['store_files']
    assert r1['inputs_unchanged'] and new['inputs_unchanged']
    assert r1['store_preexisting_preserved'] and new['store_preexisting_preserved']


@pytest.mark.parametrize('fail', ['commit:3', 'reroute:2'])
def test_injected_commit_or_reroute_failure_publishes_nothing_and_preserves_inputs(tmp_path, fail):
    r1, new = drive(R1, tmp_path, 'r1', '--fail', fail), drive(TREE, tmp_path, 'b100', '--fail', fail)
    for d in (r1, new):
        assert d['status'] == 'error' and 'B100 injected' in d['exception']['message']
        assert d['inputs_unchanged'] and d['store_preexisting_preserved']
    assert new['store_files'] == r1['store_files']  # the failed step published no deep version in either tree
