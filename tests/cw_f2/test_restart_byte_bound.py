"""CW-F2: explicit bounded resume byte limit through the real runners into the real loader (CPU only).

Genuine small cases only: the generated verified Plot 1 case (dense runner) and the generated tiled synthetic Chastre
case (resident-window runner, actual MAPLE store). The existing large Chastre checkpoint is READ through its zip
central directories only (no array, no step); the retained root measurement is the fallback when it is absent.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from test_bytes_vs_s27 import (
    validate_restart_bound,  # the CW-B30 comparison's exact record check
)

import maple_syrup
from maple_syrup import checkpoint as ck
from maple_syrup import event_experiment as ee
from maple_syrup import windowed_experiment as we
from maple_syrup.complete_event import CompletionPolicy
from maple_syrup.sediment_event import SedimentEventError

TREE_SRC = Path(__file__).resolve().parents[2] / 'src' / 'maple_syrup'  # the tree under test (private or adopted)
FROZEN_M29 = Path('/home/okin/SYRUP/outputs/dependencies/maple_cw_b29_ledger_frozen/source/src/maple')
SYRUP = Path('/home/okin/SYRUP')
LARGE_CHECKPOINT = SYRUP / 'outputs/conservative_water/chastre_M24_cpu_pause60_b32_checkpoints/step_000000060'
RETAINED_MEASUREMENT = SYRUP / 'agent_handoffs/tasks/conservative_water/cw_f1_D1_independent_checkpoint_sizes.json'
GIB = 1024**3
BAD_BOUNDS = [True, False, 0, -1, 1.5, 2.0, float('nan'), float('inf'), '1073741824', np.float64(5.0)]


def test_resolved_module_paths():
    import maple

    assert Path(maple_syrup.__file__).resolve().parent == TREE_SRC
    assert Path(maple.__file__).resolve().parent == FROZEN_M29


def payload_bytes(checkpoint):
    """Summed uncompressed payload bytes from the zip central directories (no array is read)."""
    total = 0
    for name in ck.PAYLOADS:
        with zipfile.ZipFile(Path(checkpoint) / name) as archive:
            total += sum(info.file_size for info in archive.infolist())
    return total


def tree_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


class LoaderSpy:
    """Wraps the REAL `load_checkpoint` the runner calls: records the forwarded bound, counts `np.load` calls made
    inside the loader (array materialization) and hashes the caller's context / column / sediment array leaves before
    and after the call."""

    def __init__(self, monkeypatch):
        self.calls = []
        real = ck.load_checkpoint

        def spy(path, context, column, sediment, identity, **kw):
            from maple.core.backend import array_leaves

            def leaves():
                return [np.asarray(a).tobytes() for obj in (context, column, sediment) for a in array_leaves(obj)]

            record = {'kwargs': dict(kw), 'np_load_calls': 0, 'error': None}
            self.calls.append(record)
            before = leaves()
            original_load = np.load

            def counting_load(*a, **k):
                record['np_load_calls'] += 1
                return original_load(*a, **k)

            monkeypatch.setattr(np, 'load', counting_load)
            try:
                return real(path, context, column, sediment, identity, **kw)
            except BaseException as exc:
                record['error'] = exc
                raise
            finally:
                monkeypatch.setattr(np, 'load', original_load)
                record['caller_arrays_unchanged'] = leaves() == before

        monkeypatch.setattr(ee, 'load_checkpoint', spy)


def forbid_event(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError('the event loop must not start after a refused resume')
    monkeypatch.setattr(ee, 'complete_event', refuse)


# --- validation ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('bad', [*BAD_BOUNDS, None])
def test_dense_runner_refuses_invalid_bound_before_setup(bad, tmp_path):
    with pytest.raises(SedimentEventError, match='max_checkpoint_bytes must be a positive integer'):
        ee.run_complete_case(tmp_path / 'missing_case', tmp_path / 'out', max_checkpoint_bytes=bad,
                             implementation='array')
    assert not (tmp_path / 'out').exists() and not (tmp_path / 'out_checkpoints').exists()


@pytest.mark.parametrize('bad', BAD_BOUNDS)
def test_public_loader_refuses_invalid_bound_before_any_read(bad, tmp_path):
    # A nonexistent bundle and no parameters: the bound is refused first (no file is opened).
    with pytest.raises(SedimentEventError, match='max_bytes must be a positive integer'):
        ck.load_checkpoint(tmp_path / 'absent', None, None, None, {}, max_bytes=bad)
    with pytest.raises(SedimentEventError, match='max_bytes must be a positive integer'):
        ck.load_checkpoint(tmp_path / 'absent', None, None, None, {}, max_bytes=None)


@pytest.mark.parametrize('bad', [*BAD_BOUNDS, 10**8 + 1])
def test_window_runner_refuses_invalid_or_looser_bound_before_setup(bad, tmp_path, monkeypatch):
    def no_setup(**kw):
        raise AssertionError('setup must not run')
    monkeypatch.setattr(we, '_setup_windowed', no_setup)
    monkeypatch.setattr(we, '_new_pool_limit', no_setup)
    with pytest.raises(SedimentEventError, match='max_checkpoint_bytes'):
        we.run_complete_windowed(tmp_path / 'case', tmp_path / 'out', case_kind='chastre', store_dir=tmp_path / 's',
                                 max_host_bytes=10**8, max_checkpoint_bytes=bad, nodata_value=-9999.)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize('bad', [True, 0, -5, 1.5, float('nan'), None])
def test_window_runner_refuses_invalid_host_cap_before_setup(bad, tmp_path, monkeypatch):
    monkeypatch.setattr(we, '_setup_windowed', lambda **kw: pytest.fail('setup must not run'))
    with pytest.raises(SedimentEventError, match='max_host_bytes'):
        we.run_complete_windowed(tmp_path / 'case', tmp_path / 'out', case_kind='chastre', store_dir=tmp_path / 's',
                                 max_host_bytes=bad, nodata_value=-9999.)


def test_defaults_and_cli_options():
    import inspect

    assert ck.DEFAULT_MAX_CHECKPOINT_BYTES == GIB
    assert inspect.signature(ck.load_checkpoint).parameters['max_bytes'].default == GIB
    assert inspect.signature(ee.run_complete_case).parameters['max_checkpoint_bytes'].default == GIB
    assert inspect.signature(we.run_complete_windowed).parameters['max_checkpoint_bytes'].default is None
    assert we.resolve_window_checkpoint_bytes(40 * GIB) == (40 * GIB, {
        'max_checkpoint_uncompressed_bytes': 40 * GIB,
        'resolution': 'resident-window default: equal to the explicit max_host_bytes cap', 'max_host_bytes': 40 * GIB})
    bound, record = we.resolve_window_checkpoint_bytes(np.int64(100), np.int64(7))
    assert bound == 7 and type(bound) is int and record['resolution'].startswith('caller-explicit')
    assert we.resolve_window_checkpoint_bytes(100, 100)[0] == 100  # equal to the cap is allowed


def test_dense_cli_forwards_the_bound(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(ee, 'run_complete_plot1', lambda **kw: seen.update(kw) or pytest.skip('stop'))
    with pytest.raises(pytest.skip.Exception):
        ee.main(['--case-dir', 'c', '--output-dir', str(tmp_path / 'o'), '--max-checkpoint-bytes', '12345'])
    assert seen['max_checkpoint_bytes'] == 12345
    seen.clear()
    with pytest.raises(pytest.skip.Exception):
        ee.main(['--case-dir', 'c', '--output-dir', str(tmp_path / 'o')])
    assert seen['max_checkpoint_bytes'] == GIB


def test_window_cli_forwards_the_bound(monkeypatch, tmp_path):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        raise ValueError('stop')
    monkeypatch.setattr(we, 'run_complete_windowed', fake)
    base = ['--case-dir', 'c', '--output-dir', 'o', '--store-dir', 's', '--case-kind', 'chastre']
    assert we.main([*base, '--max-checkpoint-bytes', '999']) == 1 and seen['max_checkpoint_bytes'] == 999
    assert we.main(base) == 1 and seen['max_checkpoint_bytes'] is None


# --- the recorded large manifest: default high enough, no array allocated -------------------------------------------
def large_measurement():
    """Live central-directory read of the existing Chastre paused checkpoint (metadata only) when present, and the
    retained root measurement. Both must agree when both exist."""
    retained = json.loads(RETAINED_MEASUREMENT.read_text()) if RETAINED_MEASUREMENT.is_file() else None
    live = None
    if all((LARGE_CHECKPOINT / name).is_file() for name in ck.PAYLOADS):
        live = payload_bytes(LARGE_CHECKPOINT)
    if retained is None and live is None:
        pytest.skip('neither the large checkpoint nor the retained measurement is available')
    if retained is not None and live is not None:
        assert live == retained['uncompressed_total']
    return live if live is not None else retained['uncompressed_total'], live is not None


def test_window_default_admits_the_recorded_large_checkpoint_and_dense_default_does_not():
    total, _live = large_measurement()
    assert total == 3_975_620_838
    window_default, _ = we.resolve_window_checkpoint_bytes(40 * 2**30)
    assert total <= window_default  # the supported resident-window default admits the actual large bundle
    assert total > GIB  # the unchanged dense default (= the S30 windowed path) refuses it: the F1 D1 defect
    # A caller may still tighten below the large bundle; it is then refused by the bound, as intended.
    tightened, _ = we.resolve_window_checkpoint_bytes(40 * 2**30, total - 1)
    assert tightened < total


# --- dense runner: real forwarding, real disk continuation, tightened refusal ---------------------------------------
def test_dense_default_unchanged_and_explicit_bound_forwarded_to_real_wet_continuation(plot1_case, tmp_path,
                                                                                      monkeypatch):
    common = {'implementation': 'array', 'report_every_s': 2., 'checkpoint_every_s': 2.}
    paused = ee.run_complete_plot1(plot1_case, tmp_path / 'paused', pause_at_s=2., **common)
    assert paused.summary['status'] == 'paused'
    assert paused.summary['restart_resource_bound']['max_checkpoint_uncompressed_bytes'] == GIB
    assert 'max_checkpoint' not in json.dumps(paused.summary['identity'])
    validate_restart_bound(paused.summary['restart_resource_bound'])
    checkpoint = paused.summary['last_checkpoint']
    size = payload_bytes(checkpoint)
    assert size < GIB
    full = ee.run_complete_plot1(plot1_case, tmp_path / 'full', pause_at_s=4., **common)
    bundle_before = tree_hashes(checkpoint)

    spy = LoaderSpy(monkeypatch)
    # default: the loader receives exactly the unchanged 1 GiB bound
    default = ee.run_complete_plot1(plot1_case, tmp_path / 'default', resume=checkpoint, pause_at_s=4., **common)
    assert spy.calls[-1]['kwargs'] == {'max_bytes': GIB} and spy.calls[-1]['error'] is None
    # explicit inclusive bound equal to the actual uncompressed size: forwarded and accepted
    exact = ee.run_complete_plot1(plot1_case, tmp_path / 'exact', resume=checkpoint, pause_at_s=4.,
                                  max_checkpoint_bytes=size, **common)
    assert spy.calls[-1]['kwargs'] == {'max_bytes': size} and spy.calls[-1]['np_load_calls'] > 0
    assert exact.summary['restart_resource_bound'] == {
        'max_checkpoint_uncompressed_bytes': size, 'resolution': 'caller-explicit',
        'applies_to': 'summed uncompressed payload bytes of a resumed bundle (zip central directories, checked '
                      'before any array is materialized)',
        'identity': 'resource control only; excluded from the run and checkpoint identity'}
    for label, run in (('default', default), ('exact', exact)):
        validate_restart_bound(run.summary['restart_resource_bound'])
        assert run.summary['identity'] == full.summary['identity']  # the bound is not identity
        for name in ('hydrograph', 'sediment_hydrograph', 'cumulative_pickup_kg', 'cumulative_deposition_kg'):
            np.testing.assert_array_equal(getattr(run.outcome.result, name), getattr(full.outcome.result, name))
        assert run.summary['sediment_closure'] == full.summary['sediment_closure']
        assert run.summary['water_budget_before_reset'] == full.summary['water_budget_before_reset']
        with np.load(tmp_path / label / 'final_state.npz') as a, np.load(tmp_path / 'full' / 'final_state.npz') as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                assert a[key].tobytes() == b[key].tobytes(), key

    # tightened by one byte: refused before any array materialization, step or output; caller arrays unchanged
    forbid_event(monkeypatch)
    with pytest.raises(SedimentEventError, match='checkpoint exceeds bounded uncompressed size'):
        ee.run_complete_plot1(plot1_case, tmp_path / 'tight', resume=checkpoint, pause_at_s=4.,
                              max_checkpoint_bytes=size - 1, **common)
    record = spy.calls[-1]
    assert record['kwargs'] == {'max_bytes': size - 1} and record['np_load_calls'] == 0
    assert record['caller_arrays_unchanged'] is True
    assert not (tmp_path / 'tight').exists() and not (tmp_path / 'tight_checkpoints').exists()
    assert not list(tmp_path.glob('.tight.partial-*'))
    assert tree_hashes(checkpoint) == bundle_before


# --- resident window: genuine tiled case, complete / paused / fresh resume ------------------------------------------
WINDOW_COMMON = {'case_kind': 'chastre', 'backend': 'numpy', 'implementation': 'numba', 'max_store_bytes': 10**8,
                 'max_host_bytes': 10**8, 'phase_max_nnz': 10_000, 'max_journal_records': 10_000,
                 'allow_maple_source_change': True, 'nodata_value': -9999., 'max_end_s': 20000., 'report_every_s': 60.,
                 'checkpoint_every_s': 600., 'max_dt_s': 10.}


def window(case, out, store, **kw):
    return we.run_complete_windowed(case, out, store_dir=store, **{**WINDOW_COMMON, **kw},
                                    policy=CompletionPolicy(terminal_storage_water=True))


def test_window_complete_pause_fresh_resume_and_bounds(chastre_case_dir, tmp_path, monkeypatch):
    from maple.core.backend import array_leaves
    from maple.surface.voxels.windowed_inventory import windowed_to_dense

    full = window(chastre_case_dir, tmp_path / 'full', tmp_path / 'store_full')
    assert full.summary['status'] == 'complete'
    assert full.summary['restart_resource_bound']['max_checkpoint_uncompressed_bytes'] == 10**8
    assert full.summary['restart_resource_bound']['resolution'].startswith('resident-window default')
    assert 'max_checkpoint' not in json.dumps(full.summary['identity'])
    assert 'max_checkpoint' not in json.dumps(full.summary['bed_storage'])
    validate_restart_bound(full.summary['restart_resource_bound'])

    paused = window(chastre_case_dir, tmp_path / 'paused', tmp_path / 'store_b', pause_at_s=600.)
    assert paused.summary['status'] == 'paused'

    def physics_identity(run):  # bed_storage binds the run's own store (initial assembly paths), not physics
        return {k: v for k, v in run.summary['identity'].items() if k != 'bed_storage'}
    assert physics_identity(paused) == physics_identity(full)
    checkpoint = paused.summary['last_checkpoint']
    size = payload_bytes(checkpoint)
    assert 0 < size < 10**8
    store_before, bundle_before = tree_hashes(tmp_path / 'store_b'), tree_hashes(checkpoint)

    spy = LoaderSpy(monkeypatch)
    # source/config identity behaviour preserved: a changed control is refused, not reinterpreted
    with pytest.raises(SedimentEventError, match='identity'):
        window(chastre_case_dir, tmp_path / 'changed', tmp_path / 'store_b', resume=checkpoint, max_dt_s=5.)
    assert spy.calls[-1]['kwargs'] == {'max_bytes': 10**8} and spy.calls[-1]['np_load_calls'] == 0

    # tightened below the actual bundle: refused before materialization, step, output or any store change
    with monkeypatch.context() as m:
        m.setattr(ee, 'complete_event', lambda *a, **k: pytest.fail('event must not start'))
        with pytest.raises(SedimentEventError, match='checkpoint exceeds bounded uncompressed size'):
            window(chastre_case_dir, tmp_path / 'tight', tmp_path / 'store_b', resume=checkpoint,
                   max_checkpoint_bytes=size - 1)
    record = spy.calls[-1]
    assert record['kwargs'] == {'max_bytes': size - 1} and record['np_load_calls'] == 0
    assert record['caller_arrays_unchanged'] is True
    assert not (tmp_path / 'tight').exists() and not (tmp_path / 'tight_checkpoints').exists()
    assert tree_hashes(tmp_path / 'store_b') == store_before and tree_hashes(checkpoint) == bundle_before

    # fresh resume with the exact inclusive bound: forwarded to the real loader; same physics as the uninterrupted run
    resumed = window(chastre_case_dir, tmp_path / 'resumed', tmp_path / 'store_b', resume=checkpoint,
                     max_checkpoint_bytes=size)
    assert spy.calls[-1]['kwargs'] == {'max_bytes': size} and spy.calls[-1]['error'] is None
    assert spy.calls[-1]['np_load_calls'] > 0
    assert resumed.summary['status'] == 'complete' and resumed.summary['identity'] == paused.summary['identity']
    assert resumed.summary['restart_resource_bound'] == {
        'max_checkpoint_uncompressed_bytes': size, 'resolution': 'caller-explicit, tightened within max_host_bytes',
        'max_host_bytes': 10**8,
        'applies_to': 'summed uncompressed payload bytes of a resumed bundle (zip central directories, checked '
                      'before any array is materialized)',
        'identity': 'resource control only; excluded from the run and checkpoint identity'}
    validate_restart_bound(resumed.summary['restart_resource_bound'])
    for key in ('reset_accounting', 'sediment_closure', 'water_budget_before_reset', 'n_accepted_steps', 'n_commits',
                'time_s'):
        assert resumed.summary[key] == full.summary[key], key
    a = windowed_to_dense(full.state.bed.voxel_column, max_bytes=10**8)
    b = windowed_to_dense(resumed.state.bed.voxel_column, max_bytes=10**8)
    assert a.mass_kg.tobytes() == b.mass_kg.tobytes()
    leaves_a, leaves_b = array_leaves(full.state), array_leaves(resumed.state)
    assert len(leaves_a) == len(leaves_b)
    for u, v in zip(leaves_a, leaves_b, strict=True):
        assert u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes()
    for name in ('final_state.npz', 'hydrograph.npz'):
        with np.load(tmp_path / 'full' / name) as x, np.load(tmp_path / 'resumed' / name) as y:
            assert set(x.files) == set(y.files)
            for key in x.files:
                assert x[key].tobytes() == y[key].tobytes(), (name, key)
    assert full.outcome.accounting == resumed.outcome.accounting


# --- confirmed nodata mismatch: refused before assembly ------------------------------------------------------------
@pytest.mark.parametrize('bad', [-1.0, 12.0, -9998.0, None])
def test_window_nodata_mismatch_refused_before_assembly(chastre_case_dir, tmp_path, monkeypatch, bad):
    import maple_syrup.windowed_case as wc

    for name in ('assemble_windowed_case', 'restore_initial_assembly', 'assembly_to_backend'):
        monkeypatch.setattr(we, name, lambda *a, **k: pytest.fail('assembly/restore/upload must not run'))
    assert wc.assemble_windowed_case is not None
    match = 'differs from the verified case grid' if bad is not None else 'masked case requires explicit nodata_value'
    with pytest.raises(SedimentEventError, match=match):
        window(chastre_case_dir, tmp_path / 'out', tmp_path / 'store', nodata_value=bad)
    assert not (tmp_path / 'store').exists() and not (tmp_path / 'out').exists()


def test_window_matching_integer_nodata_reaches_assembly(chastre_case_dir, tmp_path, monkeypatch):
    """The declared value given as an int equal to the verified float passes the guard (only the value matters)."""
    class Reached(Exception):
        pass

    def stop(*a, **k):
        raise Reached
    monkeypatch.setattr(we, 'assemble_windowed_case', stop)
    with pytest.raises(Reached):
        window(chastre_case_dir, tmp_path / 'out', tmp_path / 'store', nodata_value=-9999)
