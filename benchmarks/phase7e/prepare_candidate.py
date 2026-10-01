"""Build the isolated Phase 7e MAPLE candidate: accepted snapshot 72310c49 plus the selective-exchange patch.

Never edits the accepted dependency or live MAPLE. Reuses the Phase 7d preparer's manifest logic with the
accepted Phase 7d snapshot as the baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'phase7d'))
from prepare_candidate import package_manifest

EXPECTED_BASELINE = '72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65'


def prepare(baseline: Path, output: Path, patch: Path, expected_digest: str | None = None):
    if output.exists():
        raise FileExistsError(f'candidate directory already exists: {output}')
    baseline = baseline.resolve()
    output = output.resolve()
    baseline_package = package_manifest(baseline / 'source' / 'src' / 'maple')
    if baseline_package['digest_sha256'] != EXPECTED_BASELINE:
        raise ValueError('baseline package does not match the accepted Phase 7d MAPLE snapshot')
    baseline_info = json.loads((baseline / 'candidate_manifest.json').read_text())
    if baseline_info['candidate_package_digest'] != EXPECTED_BASELINE:
        raise ValueError('baseline manifest does not record the accepted digest')
    source = output / 'source'
    shutil.copytree(baseline / 'source', source)
    shutil.copytree(baseline / 'editable', output / 'editable')
    for item in output.rglob('*'):
        item.chmod(item.stat().st_mode | stat.S_IWUSR)
    metadata = output / 'editable' / 'maple-0.0.1.dist-info' / 'direct_url.json'
    metadata.write_text(json.dumps({'dir_info': {'editable': True}, 'url': source.as_uri()}) + '\n')
    (output / 'editable' / '__editable__.maple-0.0.1.pth').write_text(str(source / 'src') + '\n')
    apply_env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(output))
    subprocess.run(['git', 'apply', '--check', str(patch.resolve())], cwd=source, env=apply_env, check=True)
    subprocess.run(['git', 'apply', str(patch.resolve())], cwd=source, env=apply_env, check=True)
    test_patch = Path(__file__).with_name('upstream_test_expectation.patch')
    subprocess.run(['git', 'apply', str(test_patch.resolve())], cwd=source, env=apply_env, check=True)
    candidate_package = package_manifest(source / 'src' / 'maple')
    before, after = dict(baseline_package['files']), dict(candidate_package['files'])
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if not changed:
        raise ValueError('patch did not change the candidate package')
    if expected_digest is not None and candidate_package['digest_sha256'] != expected_digest:
        raise ValueError(f"candidate package digest {candidate_package['digest_sha256']} != expected {expected_digest}")
    result = {'baseline_package_digest': baseline_package['digest_sha256'],
              'candidate_package_digest': candidate_package['digest_sha256'],
              'changed_package_files': changed, 'baseline': str(baseline), 'candidate': str(output),
              'baseline_manifest_sha256': hashlib.sha256((baseline / 'candidate_manifest.json').read_bytes()).hexdigest(),
              'upstream_test_patch_sha256': hashlib.sha256(test_patch.read_bytes()).hexdigest(),
              'patch': str(patch.resolve()), 'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
              'note': 'Isolated Phase 7e candidate (selective bed exchange); not adopted; verify digest before use.'}
    (output / 'candidate_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('outputs/dependencies/maple_72310c49'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--patch', type=Path, default=Path('benchmarks/phase7e/maple_selective_exchange.patch'))
    p.add_argument('--expected-digest')
    a = p.parse_args()
    prepare(a.baseline, a.output, a.patch, a.expected_digest)
