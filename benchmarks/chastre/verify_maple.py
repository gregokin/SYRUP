"""Verify the isolated Chastre MAPLE candidate (72310c49 + one-file receipt patch) before its environment is selected.

The expected candidate package digest is read from benchmarks/chastre/candidate_digest.txt (or MAPLE_SYRUP_CHASTRE_EXPECTED).
While that file still holds the UNBOUND placeholder this verifier REFUSES: no hash is guessed. Checked: package digest, the
accepted pyproject hash, editable metadata rebound to this source, and `candidate_manifest.json` (original 72310c49 baseline,
exactly one changed file, the hash of benchmarks/chastre/maple_receipt.patch, the candidate digest). Nothing here was run by its
author (file-only tools).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'phase7d'))
from prepare_candidate import package_manifest

BASELINE_DIGEST = '72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65'
PYPROJECT_SHA256 = 'e6c7da22ee9f4bdd7876747fdfcbf694314cafd9032e3700301522b52b3fe3ae'
CHANGED_FILE = 'case_tools/validators/compiled_case.py'
PATCH = HERE / 'maple_receipt.patch'


def expected_digest() -> str:
    text = os.environ.get('MAPLE_SYRUP_CHASTRE_EXPECTED') or (HERE / 'candidate_digest.txt').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{64}', text):
        raise ValueError('the Chastre candidate digest is not bound yet (candidate_digest.txt is a placeholder); refusing')
    return text


def verify(root: Path) -> str:
    root = root.resolve()
    source = root / 'source'
    digest = package_manifest(source / 'src' / 'maple')['digest_sha256']
    want = expected_digest()
    if digest != want:
        raise ValueError(f'Chastre MAPLE candidate source mismatch: {digest} != {want}')
    if hashlib.sha256((source / 'pyproject.toml').read_bytes()).hexdigest() != PYPROJECT_SHA256:
        raise ValueError('Chastre MAPLE candidate pyproject.toml mismatch')
    metadata = json.loads((root / 'editable/maple-0.0.1.dist-info/direct_url.json').read_text())
    url = urlparse(metadata['url'])
    if url.scheme != 'file' or Path(unquote(url.path)).resolve() != source:
        raise ValueError('editable metadata does not identify this candidate source')
    pth = (root / 'editable' / '__editable__.maple-0.0.1.pth').read_text().strip()
    if Path(pth).resolve() != (source / 'src').resolve():
        raise ValueError('editable .pth does not identify this candidate source')
    manifest = json.loads((root / 'candidate_manifest.json').read_text())
    if manifest.get('baseline_package_digest') != BASELINE_DIGEST:
        raise ValueError('candidate manifest does not record the accepted 72310c49 baseline')
    if manifest.get('changed_package_files') != [CHANGED_FILE]:
        raise ValueError(f'candidate manifest must record exactly one changed file {CHANGED_FILE}')
    if manifest.get('candidate_package_digest') != digest:
        raise ValueError('candidate manifest digest differs from the actual package')
    if manifest.get('patch_sha256') != hashlib.sha256(PATCH.read_bytes()).hexdigest():
        raise ValueError('candidate manifest patch hash differs from benchmarks/chastre/maple_receipt.patch')
    return digest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(verify(p.parse_args().root))
