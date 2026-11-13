"""FRESH-PROCESS restore worker of the Dense CUDA floating file (run on Root's GPU by `test_gpu_definitions.py`; author did not run it). It loads ONLY the genuine H0 `/2` record parts (meta + arrays written by the saving
process), the static primitive case recipe of the H2 suite (`static_case`: terrain, grid, forcing, parameters, control -- no bed, owner, probe or provider) and the CUDA file, with every original-case/owner/probe builder
DISABLED, then continues M steps and dumps the physical snapshot.

`python gpu_fresh_worker.py WORKDIR M [windowed]` (the optional third argument selects the native Windowed `/8` restore)"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path[:0] = [str(HERE), str(T / "phase2_floating_checkpoint"), str(T / "phase2_floating_context_checkpoint"), str(T / "phase2_floating_pond"), str(T / "phase2_floating_held_context")]

AVAIL = lambda: 1 << 40


def main(workdir: str, m: int, windowed: bool = False) -> None:
    import static_case as sc
    from case import snapshot
    from maple.core.backend import cupy_module, to_device

    from maple_syrup import floating_checkpoint_cuda as fcc
    from maple_syrup import floating_held_context as fh
    from maple_syrup import floating_pond as fp
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup import probe
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondEvent, PondEventControl
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    cp = cupy_module()
    work = Path(workdir)

    def boom(*a, **k):
        raise AssertionError("original-case/owner/probe machinery must not run on the restore path")

    nfs.initialize_native_floating_soil = boom
    nfs.stage_native_refresh = boom
    probe.build_minimal_state = boom
    meta = json.loads((work / "h0_meta.json").read_text())
    names = json.loads((work / "h0_arrays.json").read_text())
    arrays = {k: np.frombuffer((work / f"h0_{k}.bin").read_bytes(), dtype=np.dtype(dt)).reshape(shape).copy(order="C") for k, (dt, shape) in names.items()}
    record = fh.record_from_parts(meta, arrays, available_bytes_fn=AVAIL)
    ident = meta["identities"]
    ctx = fh.reconstruct_held_context(record, fh.ExpectedHeldIdentities(ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"]), available_bytes_fn=AVAIL)
    vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    bed_ctx = sc.bed_context()
    dctx = replace(bed_ctx, initial_available_fraction=to_device(bed_ctx.initial_available_fraction, cp))
    ctl = PondEventControl(backend="cuda", time=ExperimentalControl(max_dt_s=0.2, min_dt_s=1.0 / 1024.0, max_retries=12, max_steps=50_000), bed_storage="windowed" if windowed else "dense", deep_publication="step")
    ev = PondEvent(dctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, 1000.0, 300.0), sc.topology(), control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, sc.NC), xp=cp), floating_vertical=vert)
    ident_case = json.loads((work / "case_identity.json").read_text())
    st = fcc.load_floating_pond_checkpoint_cuda(work / "ck", ev, ident_case)
    rate = 300.0 / 3.6e6
    for _ in range(int(m)):
        st = ev.step(st, rate, 0.2).state
    np.savez(work / "restored_final.npz", **snapshot(ev, st, vert))
    print(json.dumps({"ok": True, "steps": st.counters.n_accepted_steps}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), len(sys.argv) > 3 and sys.argv[3] == "windowed")
