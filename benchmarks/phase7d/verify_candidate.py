"""Verify the accepted MAPLE allocation candidate before selecting its environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from prepare_candidate import package_manifest

EXPECTED_CANDIDATE = '72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65'
EXPECTED_PYPROJECT = 'e6c7da22ee9f4bdd7876747fdfcbf694314cafd9032e3700301522b52b3fe3ae'


def verify(root):
    source = root.resolve() / 'source'
    digest = package_manifest(source / 'src' / 'maple')['digest_sha256']
    if digest != EXPECTED_CANDIDATE:
        raise ValueError(f'MAPLE candidate source mismatch: {digest}')
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
    args = p.parse_args()
    print(verify(args.root))
