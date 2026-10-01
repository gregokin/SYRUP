"""Build an isolated MAPLE candidate from the accepted dependency plus a patch.

Never edits the accepted dependency or live MAPLE checkout. Distribution
metadata is copied and rebound to the isolated source. No installation needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

EXPECTED_BASELINE = 'd3d007024ff65abc2a0ff179f0f03bcd1c3b27f091493136cce849c34f7a4264'


def package_manifest(root):
    entries = []
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        if path.is_symlink():
            raise ValueError(f'symlink in dependency package: {path}')
        if path.is_file():
            entries.append((path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    payload = ''.join(name + '\0' + digest + '\n' for name, digest in entries)
    return {'digest_sha256': hashlib.sha256(payload.encode()).hexdigest(), 'files': entries}


def prepare(baseline: Path, output: Path, patch: Path, expected_digest: str | None = None):
    if output.exists():
        raise FileExistsError(f'candidate directory already exists: {output}')
    baseline = baseline.resolve()
    output = output.resolve()
    baseline_package = package_manifest(baseline / 'source' / 'src' / 'maple')
    if baseline_package['digest_sha256'] != EXPECTED_BASELINE:
        raise ValueError('baseline package does not match the accepted MAPLE snapshot')
    baseline_info = json.loads((baseline / 'manifest.json').read_text())
    if hashlib.sha256((baseline / 'source' / 'pyproject.toml').read_bytes()).hexdigest() != baseline_info['pyproject_sha256']:
        raise ValueError('baseline pyproject.toml differs from the accepted manifest')
    source = output / 'source'
    shutil.copytree(baseline / 'source', source)
    shutil.copytree(baseline / 'editable', output / 'editable')
    # The accepted snapshot is read-only. Grant owner-write only on this new copy.
    for item in output.rglob('*'):
        item.chmod(item.stat().st_mode | stat.S_IWUSR)
    metadata = output / 'editable' / 'maple-0.0.1.dist-info' / 'direct_url.json'
    metadata.write_text(json.dumps({'dir_info': {'editable': True}, 'url': source.as_uri()}) + '\n')
    pth = output / 'editable' / '__editable__.maple-0.0.1.pth'
    pth.write_text(str(source / 'src') + '\n')
    apply_env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(output))
    subprocess.run(['git', 'apply', '--check', str(patch.resolve())], cwd=source, env=apply_env, check=True)
    subprocess.run(['git', 'apply', str(patch.resolve())], cwd=source, env=apply_env, check=True)
    candidate_package = package_manifest(source / 'src' / 'maple')
    before, after = dict(baseline_package['files']), dict(candidate_package['files'])
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if not changed:
        raise ValueError('patch did not change the candidate package')
    if expected_digest is not None and candidate_package['digest_sha256'] != expected_digest:
        raise ValueError('candidate package does not match the expected patched digest')
    result = {'baseline_package_digest': baseline_package['digest_sha256'],
              'candidate_package_digest': candidate_package['digest_sha256'],
              'changed_package_files': changed,
              'baseline': str(baseline), 'candidate': str(output),
              'baseline_manifest_sha256': hashlib.sha256((baseline / 'manifest.json').read_bytes()).hexdigest(),
              'patch': str(patch.resolve()), 'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
              'note': 'Isolated candidate, not adopted upstream; exact source digest must be verified before use.'}
    (output / 'candidate_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/dependencies/maple_d3d007024'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patch', type=Path, required=True)
    parser.add_argument('--expected-digest', help='require the recorded package digest before accepting this copy')
    args = parser.parse_args()
    prepare(args.baseline, args.output, args.patch, args.expected_digest)
