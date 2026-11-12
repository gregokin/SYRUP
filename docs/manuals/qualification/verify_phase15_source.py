#!/usr/bin/env python
"""Portable check that a SYRUP checkout carries the exact Phase 1.5 (N15) source qualified by Root.

    python verify_phase15_source.py SYRUP_CHECKOUT_ROOT

Reads `phase15_source_manifest.json` (next to this script: file name -> size and SHA-256 of the 114 files of `src/maple_syrup`), compares the working files with it
(no extra, missing or changed file; bytecode caches ignored) and, when the package is importable, prints the package tree digest of `src/maple_syrup` computed by
`maple_syrup.provenance.source_tree_digest`. Exit 0 and `PASS` only if all file hashes match AND the digest equals the qualified
82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107. It writes nothing. It identifies source bytes only: it is not the whole-phase acceptance."""
import hashlib
import json
import sys
from pathlib import Path

QUALIFIED = "82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1]).resolve() / "src" / "maple_syrup"
    manifest = json.loads((Path(__file__).resolve().parent / "phase15_source_manifest.json").read_text())
    if not root.is_dir():
        raise SystemExit(f"not a SYRUP checkout (no src/maple_syrup): {root}")
    present = {p.name for p in root.iterdir() if p.is_file() and p.suffix == ".py"}
    bad = [n for n in sorted(manifest) if not (root / n).is_file() or (root / n).stat().st_size != manifest[n]["size"]
           or hashlib.sha256((root / n).read_bytes()).hexdigest() != manifest[n]["sha256"]]
    extra = sorted(present - set(manifest))
    print(f"manifest files: {len(manifest)}; changed/missing: {len(bad)}; extra .py: {len(extra)}")
    for n in bad + extra:
        print("  differs:", n)
    ok = not bad and not extra
    try:
        sys.path.insert(0, str(root.parent))
        from maple_syrup.provenance import source_tree_digest
        digest = source_tree_digest(root).digest_sha256
        print("package tree digest:", digest, "(matches the qualified digest)" if digest == QUALIFIED else "(DIFFERS from the qualified digest)")
        ok = ok and digest == QUALIFIED
    except Exception as exc:  # the file-level comparison above is still meaningful without the import
        print("package digest not computed:", type(exc).__name__, exc)
        ok = False
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
