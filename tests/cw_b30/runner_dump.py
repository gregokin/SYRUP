"""Fresh-process runner driver for the CW-B30 byte comparison (run once per tree with that tree's `src` on the path).

Usage: python runner_dump.py TESTS_ROOT KIND MODE WORK OUT_JSON
MODE: `complete`  quiet completion + dry reset, twice in this process (cold, then warm) -> WORK/complete[_warm]
      `pause`     a wet pause at 4 s -> WORK/paused (+ its checkpoint)
      `resume`    a FRESH process resuming WORK/paused's checkpoint to completion -> WORK/resumed
      `load`      a FRESH process loading the completed handoff checkpoint of WORK/complete (allow_complete)
Writes every array of every output/checkpoint npz (and of the loaded handoff) to WORK/<mode>_arrays.npz, the
normalised summaries and resource figures to OUT_JSON."""
from __future__ import annotations

import json
import resource
import sys
import time
import tracemalloc
from pathlib import Path

tests_root, kind, mode, work, out_json = sys.argv[1:6]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # b30_cases (this tree's test helper; no science)
for sub in ("phase5", "conservative_water"):
    sys.path.insert(0, str(Path(tests_root) / sub))

import numpy as np
from b30_cases import make_hook, run_options, window_options

import maple_syrup
from maple_syrup.event_experiment import run_complete_case

work = Path(work)
opts = run_options if kind == "dense" else window_options
VOLATILE = {"timings", "outputs", "checkpoint_paths", "resumed_from", "last_checkpoint", "loop_transfer_counters"}


def normalise(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in VOLATILE or k.endswith("_wall_s") or k == "syrup_source_sha256":
                continue
            out[k] = normalise(v)
        return out
    if isinstance(value, list):
        return [normalise(v) for v in value]
    return value


def arrays_of(directory: Path, prefix: str, arrays: dict):
    for path in sorted(directory.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            for name in z.files:
                arrays[f"{prefix}/{path.relative_to(directory)}/{name}"] = z[name].copy()


def measured(label, fn, record):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record[label] = {"tracemalloc_peak_bytes": peak, "wall_s": time.perf_counter() - t0,
                     "ru_maxrss_bytes_process_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    return result


resources, summaries, arrays = {}, {}, {}
if mode == "complete":
    for label in ("complete", "complete_warm"):
        run = measured(label, lambda label=label: run_complete_case(
            work / "src" / "case", work / label, case_setup=make_hook(kind, work / f"src_{label}"), **opts()), resources)
        summaries[label] = normalise(run.summary)
        arrays_of(work / label, label, arrays)
elif mode == "pause":
    run = measured("pause", lambda: run_complete_case(work / "src" / "case", work / "paused",
                                                      case_setup=make_hook(kind, work / "src_pause"),
                                                      **opts(pause_at_s=4.0)), resources)
    summaries["pause"] = normalise(run.summary)
    (work / "paused_checkpoint.txt").write_text(run.summary["last_checkpoint"])
    arrays_of(work / "paused", "paused", arrays)
    arrays_of(Path(run.summary["last_checkpoint"]), "paused_checkpoint", arrays)
elif mode == "resume":
    checkpoint = (work / "paused_checkpoint.txt").read_text()
    run = measured("resume", lambda: run_complete_case(work / "src" / "case", work / "resumed",
                                                       case_setup=make_hook(kind, work / "src_resume"),
                                                       resume=checkpoint, **opts()), resources)
    summaries["resume"] = normalise(run.summary)
    arrays_of(work / "resumed", "resumed", arrays)
elif mode == "load":
    from b30_cases import build_case
    from maple.core.backend import array_leaves

    from maple_syrup.checkpoint import load_checkpoint
    from maple_syrup.sediment_event import SedimentEventControl
    from maple_syrup.storm import StormControl

    handoff = work / "complete" / "handoff"
    manifest = json.loads((handoff / "checkpoint.json").read_text())
    storage = "dense" if kind == "dense" else "csr"
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, implementation="array", max_steps=10_000_000),
                                   record_exchange_history=True, phase_storage=storage)
    s, ctx, _schedule = build_case(kind, control, work / "src_load")
    import dataclasses

    ctx = dataclasses.replace(ctx, record_exchange_history=True)
    item = load_checkpoint(handoff, ctx, s["column"], s["sediment"], manifest["identity"], allow_complete=True)
    for i, leaf in enumerate(array_leaves(item)):
        arrays[f"loaded/{i:05d}"] = np.asarray(leaf).copy()
    summaries["load"] = {"type": type(item).__name__, "accounting": normalise(item.accounting)}
np.savez(work / f"{mode}_arrays.npz", **arrays)
Path(out_json).write_text(json.dumps({"maple_syrup": str(Path(maple_syrup.__file__).resolve().parent),
                                      "summaries": summaries, "resources": resources}, sort_keys=True, default=str))
