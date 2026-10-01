"""Verify the Phase 7e MAPLE candidate before selecting its environment.

The expected digest is read from benchmarks/phase7e/candidate_digest.txt (recorded after the
reproducible build) unless MAPLE_SYRUP_PHASE7E_EXPECTED overrides it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'phase7d'))
from prepare_candidate import package_manifest

EXPECTED_PYPROJECT = 'e6c7da22ee9f4bdd7876747fdfcbf694314cafd9032e3700301522b52b3fe3ae'


def expected_digest() -> str:
    env = os.environ.get('MAPLE_SYRUP_PHASE7E_EXPECTED')
    if env:
        return env
    return (HERE / 'candidate_digest.txt').read_text().strip()


def verify(root: Path) -> str:
    source = root.resolve() / 'source'
    digest = package_manifest(source / 'src' / 'maple')['digest_sha256']
    want = expected_digest()
    if digest != want:
        raise ValueError(f'MAPLE Phase 7e candidate source mismatch: {digest} != {want}')
    if hashlib.sha256((source / 'pyproject.toml').read_bytes()).hexdigest() != EXPECTED_PYPROJECT:
        raise ValueError('MAPLE candidate pyproject.toml mismatch')
    metadata = json.loads((root / 'editable/maple-0.0.1.dist-info/direct_url.json').read_text())
    url = urlparse(metadata['url'])
    if url.scheme != 'file' or Path(unquote(url.path)).resolve() != source:
        raise ValueError('editable metadata does not identify this candidate source')
    return digest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(verify(p.parse_args().root))
