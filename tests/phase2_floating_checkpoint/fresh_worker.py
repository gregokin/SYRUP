"""Worker of the fresh-process test: `python fresh_worker.py save|load WORKDIR BACKEND WINDOWED K M`. Each invocation is a NEW interpreter that builds the ORIGINAL case from the same factory; neither side needs the other's
in-memory Pond state."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "phase2_floating_pond")]
from case import AVAIL, IDENT, build, run, snapshot

from maple_syrup import floating_checkpoint as fc


def main(mode, workdir, backend, windowed, k, m):
    work = Path(workdir) / mode
    work.mkdir(parents=True, exist_ok=True)
    ev, st, vert = build(work, backend=backend, windowed=windowed == "1")
    ck = Path(workdir) / "ck"
    if mode == "save":
        st = run(ev, st, int(k))
        fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    else:
        st = fc.load_floating_pond_checkpoint(ck, ev, IDENT, available_bytes_fn=AVAIL)
    st = run(ev, st, int(m))
    snap = snapshot(ev, st, vert)
    np.savez(Path(workdir) / f"{mode}_final.npz", **{k_: v_ for k_, v_ in snap.items()})


if __name__ == "__main__":
    main(*sys.argv[1:7])
