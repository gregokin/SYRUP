"""FRESH-PROCESS restore worker of the CUDA context container (run on Root's GPU by `test_gpu_definitions.py`; author did not run it): the ONLY inputs are the container directory and the static primitive recipe; every
original-case/owner/probe builder is DISABLED (the ACTUAL loaded native bed, built by the child loader, is allowed). `python gpu_fresh_worker.py WORKDIR M DEVICE_ID [windowed]`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path[:0] = [str(HERE), str(T / "phase2_floating_checkpoint"), str(T / "phase2_floating_context_checkpoint"), str(T / "phase2_floating_pond"), str(T / "phase2_floating_held_context")]


def main(workdir: str, m: int, device_id: int, windowed: bool) -> None:
    import gpu_static_case as gsc
    from case import snapshot

    from maple_syrup import floating_context_checkpoint_cuda as ccc
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup import probe

    work = Path(workdir)

    def boom(*a, **k):
        raise AssertionError("original-case/owner/probe machinery must not run on the restore path")

    nfs.initialize_native_floating_soil = boom
    nfs.stage_native_refresh = boom
    probe.build_minimal_state = boom
    ident = json.loads((work / "case_identity.json").read_text())
    ev, st = ccc.load_floating_context_checkpoint_cuda(work / "ctx", gsc.factory(windowed), ident, expected_device_id=device_id)
    for _ in range(int(m)):
        st = ev.step(st, 300.0 / 3.6e6, 0.2).state
    np.savez(work / "restored_final.npz", **snapshot(ev, st, ev.vertical))
    print(json.dumps({"ok": True, "steps": st.counters.n_accepted_steps}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), len(sys.argv) > 4 and sys.argv[4] == "windowed")
