"""CW-F2 NEGATIVE CONTROL against the frozen S30 runner plumbing (read-only import of the frozen tree).

Run explicitly with the frozen S30 `src` on PYTHONPATH instead of the private tree (see `env_s30.sh`); it is not
collected by the normal CW-F2 run. Reproduces, on a small genuine tiled Chastre case:

* the missing bound: S30 runners have no resume byte-bound parameter, refuse one, and call the loader with NO
  `max_bytes` (so the 1 GiB loader default applies). A spy scales that default by the retained real large
  central-directory ratio (1 GiB / 3,975,620,838 B) onto the small bundle and the real loader refuses it, which is
  the D1 refusal at small scale; no huge array is fabricated.
* the nodata trigger path: a caller nodata_value different from the verified grid value passes S30 setup, the
  initial store IS assembled, and the run fails only later in the masked graph rebuild.
Results are written as JSON to $CW_F2_NEGATIVE_LOG.
"""
from __future__ import annotations

import inspect
import json
import os
import zipfile
from pathlib import Path

import pytest

import maple_syrup
from maple_syrup import checkpoint as ck
from maple_syrup import event_experiment as ee
from maple_syrup import windowed_experiment as we
from maple_syrup.complete_event import CompletionPolicy
from maple_syrup.sediment_event import SedimentEventError

FROZEN_S30 = Path('/home/okin/SYRUP/outputs/conservative_water/cw_b30_integrated_frozen/src/maple_syrup')
LARGE_TOTAL = 3_975_620_838
GIB = 1024**3
COMMON = {'case_kind': 'chastre', 'backend': 'numpy', 'implementation': 'numba', 'max_store_bytes': 10**8,
          'max_host_bytes': 10**8, 'phase_max_nnz': 10_000, 'max_journal_records': 10_000, 'allow_maple_source_change': True,
          'nodata_value': -9999., 'max_end_s': 20000., 'report_every_s': 60., 'checkpoint_every_s': 600., 'max_dt_s': 10.}
RESULTS = {}


@pytest.fixture(scope='module', autouse=True)
def write_log():
    yield
    path = os.environ.get('CW_F2_NEGATIVE_LOG')
    if path:
        Path(path).write_text(json.dumps(RESULTS, indent=2, default=str) + '\n')


def window(case, out, store, **kw):
    return we.run_complete_windowed(case, out, store_dir=store, **{**COMMON, **kw},
                                    policy=CompletionPolicy(terminal_storage_water=True))


def test_running_the_frozen_s30_source():
    assert Path(maple_syrup.__file__).resolve().parent == FROZEN_S30
    RESULTS['maple_syrup'] = str(Path(maple_syrup.__file__).resolve())


def test_s30_has_no_bound_parameter_and_refuses_one(tmp_path):
    assert 'max_checkpoint_bytes' not in inspect.signature(ee.run_complete_case).parameters
    assert 'max_checkpoint_bytes' not in inspect.signature(we.run_complete_windowed).parameters
    assert inspect.signature(ck.load_checkpoint).parameters['max_bytes'].default == GIB
    with pytest.raises(TypeError, match='max_checkpoint_bytes'):
        ee.run_complete_case(tmp_path / 'c', tmp_path / 'o', max_checkpoint_bytes=8 * GIB)
    with pytest.raises(TypeError, match='max_checkpoint_bytes'):
        we.run_complete_windowed(tmp_path / 'c', tmp_path / 'o', case_kind='chastre', store_dir=tmp_path / 's',
                                 nodata_value=-9999., max_checkpoint_bytes=8 * GIB)
    RESULTS['signature'] = 'S30 run_complete_case / run_complete_windowed: no bound parameter; TypeError when passed'


def test_s30_resume_uses_loader_default_and_scaled_default_refuses(chastre_case_dir, tmp_path, monkeypatch):
    paused = window(chastre_case_dir, tmp_path / 'paused', tmp_path / 'store', pause_at_s=600.)
    checkpoint = paused.summary['last_checkpoint']
    size = 0
    for name in ck.PAYLOADS:
        with zipfile.ZipFile(Path(checkpoint) / name) as archive:
            size += sum(i.file_size for i in archive.infolist())
    scaled = size * GIB // LARGE_TOTAL
    calls = []
    real = ck.load_checkpoint

    def spy(*a, **kw):
        calls.append(dict(kw))
        return real(*a, **{**kw, 'max_bytes': scaled})  # the S30 default, scaled to this small bundle

    monkeypatch.setattr(ee, 'load_checkpoint', spy)
    with pytest.raises(SedimentEventError, match='checkpoint exceeds bounded uncompressed size') as info:
        window(chastre_case_dir, tmp_path / 'resumed', tmp_path / 'store', resume=checkpoint)
    assert calls == [{}]  # S30 forwards NO max_bytes: the loader's 1 GiB default is the only bound
    assert not (tmp_path / 'resumed').exists()
    RESULTS['missing_bound'] = {'small_bundle_uncompressed_bytes': size, 'scaled_default_bytes': scaled,
                                'scale': f'{GIB} / {LARGE_TOTAL}', 'loader_kwargs_from_s30_runner': calls[0],
                                'refusal': f'{type(info.value).__name__}: {info.value}',
                                'large_real_total_vs_default': [LARGE_TOTAL, GIB, LARGE_TOTAL > GIB]}


def test_s30_nodata_mismatch_assembles_then_fails_late(chastre_case_dir, tmp_path, monkeypatch):
    from maple_syrup import routing

    seen = []
    real_build = routing.build_routing_graph

    def build(*a, **k):
        seen.append(k.get('nodata_value'))
        return real_build(*a, **k)

    monkeypatch.setattr(routing, 'build_routing_graph', build)
    with pytest.raises(Exception) as info:
        window(chastre_case_dir, tmp_path / 'out', tmp_path / 'store', nodata_value=-1.0, max_end_s=60.)
    store_files = sorted(str(p.relative_to(tmp_path / 'store')) for p in (tmp_path / 'store').rglob('*') if p.is_file())
    assert store_files, 'S30 assembled and persisted the initial store before the mismatch surfaced'
    assert seen[-1] == -1.0 and -9999.0 in seen[:-1]
    RESULTS['nodata_trigger'] = {'exception': f'{type(info.value).__name__}: {info.value}',
                                 'graph_build_nodata_values_in_order': seen,
                                 'n_store_files_persisted_before_failure': len(store_files),
                                 'output_created': (tmp_path / 'out').exists()}
