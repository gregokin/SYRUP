"""Build the isolated MAPLE candidate for the Chastre compiler-receipt fix: the accepted 72310c49 package plus ONE one-file patch.

    python benchmarks/chastre/prepare_maple.py --output <NEW dir> [--baseline outputs/dependencies/maple_72310c49]
        [--patch benchmarks/chastre/maple_receipt.patch] [--expected-digest <sha256>]

Never edits the accepted baseline, the old snapshots or the live MAPLE checkout (they are only read). The baseline package
digest and `pyproject.toml` hash must equal the accepted 72310c49 values; the distribution metadata is copied and rebound to the
NEW source; the patch must change exactly `case_tools/validators/compiled_case.py`. `candidate_manifest.json` records the original
72310c49 digest, the patch hash, the candidate digest and the one-file difference. Nothing is installed. Nothing here was run by
its author (file-only tools); Codex computes and binds the candidate digest after review.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'phase7d'))
from prepare_candidate import (
    package_manifest,  # phase7d.package_manifest: the accepted manifest algorithm
)

BASELINE_DIGEST = '72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65'
PYPROJECT_SHA256 = 'e6c7da22ee9f4bdd7876747fdfcbf694314cafd9032e3700301522b52b3fe3ae'
CHANGED_FILE = 'case_tools/validators/compiled_case.py'


def prepare(baseline: Path, output: Path, patch: Path, expected_digest: str | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f'candidate directory already exists: {output}')
    baseline, output, patch = baseline.resolve(), output.resolve(), patch.resolve()
    if output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError('the candidate and the baseline must be disjoint directories')
    baseline_package = package_manifest(baseline / 'source' / 'src' / 'maple')
    if baseline_package['digest_sha256'] != BASELINE_DIGEST:
        raise ValueError(f"baseline package {baseline_package['digest_sha256']} is not the accepted 72310c49 snapshot")
    pyproject = hashlib.sha256((baseline / 'source' / 'pyproject.toml').read_bytes()).hexdigest()
    if pyproject != PYPROJECT_SHA256:
        raise ValueError('baseline pyproject.toml differs from the accepted hash')
    source = output / 'source'
    shutil.copytree(baseline / 'source', source)
    shutil.copytree(baseline / 'editable', output / 'editable')
    for item in output.rglob('*'):  # the accepted copy may be read-only: owner-write on the NEW copy only
        item.chmod(item.stat().st_mode | stat.S_IWUSR)
    (output / 'editable' / 'maple-0.0.1.dist-info' / 'direct_url.json').write_text(
        json.dumps({'dir_info': {'editable': True}, 'url': source.as_uri()}) + '\n')
    (output / 'editable' / '__editable__.maple-0.0.1.pth').write_text(str(source / 'src') + '\n')
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(output))
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=source, env=env, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=source, env=env, check=True)
    candidate_package = package_manifest(source / 'src' / 'maple')
    before, after = dict(baseline_package['files']), dict(candidate_package['files'])
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if changed != [CHANGED_FILE]:
        raise ValueError(f'the patch must change exactly {CHANGED_FILE}, it changed {changed}')
    if expected_digest is not None and candidate_package['digest_sha256'] != expected_digest:
        raise ValueError('candidate package does not match the expected patched digest')
    result = {'baseline_package_digest': BASELINE_DIGEST, 'candidate_package_digest': candidate_package['digest_sha256'],
              'changed_package_files': changed, 'baseline': str(baseline), 'candidate': str(output),
              'pyproject_sha256': pyproject, 'patch': str(patch), 'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
              'note': 'Isolated candidate (compiled-case receipt evaluation only), not adopted upstream; bind the digest after review.'}
    (output / 'candidate_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/dependencies/maple_72310c49'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patch', type=Path, default=HERE / 'maple_receipt.patch')
    parser.add_argument('--expected-digest')
    args = parser.parse_args()
    print(json.dumps(prepare(args.baseline, args.output, args.patch, args.expected_digest), indent=2))
