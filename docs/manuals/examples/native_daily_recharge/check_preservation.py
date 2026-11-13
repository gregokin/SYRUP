#!/usr/bin/env python
"""File-preservation check for daily_recharge_example.py (no model run, no source tree needed).

Creates sentinel files in a temporary directory, runs the example script against them as an existing --workdir and as an existing --checkpoint-dir, and verifies:
the script refuses (non-zero exit, message names the path), every sentinel byte is identical afterwards (content, names, sizes), and no new file was created.
Prints PASS or exits non-zero."""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir" for p in sorted(root.rglob("*"))}


def run(args):
    return subprocess.run([sys.executable, "-B", str(HERE / "daily_recharge_example.py"), *args], capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        bundle = t / "bundle"
        bundle.mkdir()
        work = t / "work"  # an EXISTING work directory holding a user config, a case and a checkpoint
        (work / "checkpoint_v3").mkdir(parents=True)
        (work / "layered_water.yaml").write_bytes(b"user: edited config\n")
        (work / "checkpoint_v3" / "SEAL.json").write_bytes(b"\x00\x01precious checkpoint bytes")
        (work / "case").mkdir()
        (work / "case" / "case.yaml").write_bytes(b"user case\n")
        elsewhere = t / "ck_elsewhere"  # an EXISTING checkpoint directory requested via --checkpoint-dir
        elsewhere.mkdir()
        (elsewhere / "maple_state.npz").write_bytes(b"live checkpoint payload")
        before = snapshot(t)
        trials = [("existing --workdir", ["--bundle", str(bundle), "--workdir", str(work), "--checkpoint"], str(work)),
                  ("existing --checkpoint-dir", ["--bundle", str(bundle), "--workdir", str(t / "fresh"), "--checkpoint", "--checkpoint-dir", str(elsewhere)], str(elsewhere)),
                  ("--checkpoint-dir without --checkpoint", ["--bundle", str(bundle), "--workdir", str(t / "fresh2"), "--checkpoint-dir", str(t / "newck")], "--checkpoint")]
        for name, args, token in trials:
            r = run(args)
            after = snapshot(t)
            assert r.returncode != 0, (name, "did not refuse")
            assert token in (r.stderr + r.stdout), (name, "message does not name the path", r.stderr)
            assert after == before, (name, "files changed", set(after.items()) ^ set(before.items()))
            print(f"refused ({name}): exit {r.returncode}; all {len(before)} sentinel entries byte-identical")
    print("PASS")


if __name__ == "__main__":
    main()
