"""Root-review helper (NOT a test, never imported by a guard): PREPARES proposed final-lock bytes and the literal pin block from an actual source directory. It never writes into `tests/fixtures/phase25`, never edits
`p25_source_pins.py`, and never admits anything: the proposal carries status `PROPOSED_NOT_LOCKED`, which the guards refuse. Root alone, after every production writer closes and after reviewing the proposal, runs
`seal` with the exact tree digest it reviewed (an explicit second argument that must equal the proposal's digest) and copies the sealed bytes + literals itself.

    python prepare_lock.py propose <src/maple_syrup> <out_dir>
    python prepare_lock.py seal <proposal.json> <expected_tree_digest> <out_dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import p25_source_pins as pins
import p25_source_profile as prof

PROPOSAL_STATUS = "PROPOSED_NOT_LOCKED"


def _dump(obj: dict) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def propose(src_dir, fixtures=None) -> bytes:
    base = prof.load_base(prof.read_fixture("base169_manifest.json", fixtures), pins)
    prof.load_wh1_proof(prof.read_fixture("wh1_953_input_proof.json", fixtures), pins, base)
    digest, actual, total = prof.actual_listing(src_dir)
    if set(base) - set(actual):
        raise prof.SourceProfileError(f"base modules are missing from the source: {sorted(set(base) - set(actual))}")
    changed = [{"path": p, "base_sha256": base[p], "sha256": actual[p]} for p in sorted(base) if actual[p] != base[p]]
    added = [{"path": p, "sha256": actual[p]} for p in sorted(actual) if p not in base and p != pins.WH1_PATH]
    if pins.WH1_PATH not in actual:
        raise prof.SourceProfileError("the WH1 module is missing from the source")
    return _dump({"schema": prof.LOCK_SCHEMA, "status": PROPOSAL_STATUS, "whole_phase_accepted": False, "git_base": pins.BASE_GIT,
                  "wh1": {"path": pins.WH1_PATH, "input_sha256": pins.WH1_INPUT_SHA256, "sha256": actual[pins.WH1_PATH]},
                  "files": [[p, actual[p]] for p in sorted(actual)], "changed": changed, "added": added, "file_count": len(actual), "total_bytes": total, "tree_digest_sha256": digest})


def seal(proposal: bytes, expected_tree_digest: str) -> bytes:
    d = prof.strict_json(proposal, "proposal")
    if d.get("status") != PROPOSAL_STATUS:
        raise prof.SourceProfileError("only an unsealed proposal can be sealed")
    if d.get("tree_digest_sha256") != expected_tree_digest:
        raise prof.SourceProfileError("the reviewed tree digest is not the proposal's digest")
    d["status"] = prof.LOCK_STATUS
    return _dump(d)


def pin_block(lock: bytes) -> str:
    d = json.loads(lock)
    return "\n".join([f'LOCK_SHA256 = "{hashlib.sha256(lock).hexdigest()}"', f'TREE_DIGEST_SHA256 = "{d["tree_digest_sha256"]}"', f"FILE_COUNT = {d['file_count']}", f"TOTAL_BYTES = {d['total_bytes']}",
                      f"# CHANGED = {len(d['changed'])} names, ADDED = {len(d['added'])} names (+ the WH1 module): update p25_source_pins.CHANGED / ADDED to the lock's lists"])


def main(argv: list) -> int:
    if len(argv) == 4 and argv[1] == "propose":
        out = Path(argv[3]) / "proposed_lock.json"
        data = propose(argv[2])
    elif len(argv) == 5 and argv[1] == "seal":
        out = Path(argv[4]) / "final_lock.sealed.json"
        data = seal(Path(argv[2]).read_bytes(), argv[3])
    else:
        print(__doc__)
        return 2
    if out.exists() or out.parent.name == "phase25" and out.parent.parent.name == "fixtures":
        print(f"refusing to write {out}: an existing file or the shipped fixtures directory", file=sys.stderr)
        return 1
    out.write_bytes(data)
    print(out, hashlib.sha256(data).hexdigest())
    if argv[1] == "seal":
        print(pin_block(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
