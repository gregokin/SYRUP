"""Run the COMPLETE event driver on one synthetic case with whatever `maple_syrup` is importable (the parent test
sets PYTHONPATH to the original B15 tree or to this tree) and dump every array and count to an npz/json pair.

Usage: python driver_dump.py TESTS_ROOT CASE BACKEND OUT_PREFIX WORKDIR
Runs: a quiet completion + dry reset (`reference`), a wet pause, a checkpoint save/load and a resumed completion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

tests_root, case, backend, out, work = sys.argv[1:6]
for sub in ("phase5", "conservative_water", "cw_b20"):
    sys.path.insert(0, str(Path(tests_root) / sub))

import numpy as np
from b20_cases import build, host, resolve
from maple.core.backend import array_leaves
from test_exchange_history import IDENTITY, complete

import maple_syrup
from maple_syrup.checkpoint import load_checkpoint, save_checkpoint

work = Path(work)
xp = resolve(backend)
arrays, counts = {}, {"maple_syrup": str(Path(maple_syrup.__file__).resolve().parent)}


def dump(label, outcome):
    leaves = array_leaves(outcome)
    for i, leaf in enumerate(leaves):
        arrays[f"{label}/{i:05d}"] = host(leaf)
    progress = getattr(outcome, "progress", outcome)
    r = progress.result
    counts[label] = {"n_leaves": len(leaves), "type": type(outcome).__name__,
                     **{k: getattr(r, k) for k in ("n_accepted_steps", "n_rejected_attempts", "n_commits",
                                                   "n_graph_changes", "n_forced_commits", "n_maple_water_calls",
                                                   "n_phase_canonicalized", "n_phase_rounding_remnants",
                                                   "max_transport_substeps_used", "n_transport_rejections")},
                     "closure": r.closure(), "accounting": getattr(outcome, "accounting", None)}
    column = getattr(getattr(outcome, "dry_state", r.state).bed, "voxel_column", None)
    if column is not None and hasattr(column, "deep_store"):
        from maple.surface.voxels.windowed_inventory import windowed_to_dense

        arrays[f"{label}/logical_voxels"] = host(windowed_to_dense(column, max_bytes=10**8).mass_kg)


s, ctx, ctl, schedule = build(case, xp, work / "a")
dump("reference", complete(s, ctx, ctl, schedule))
s, ctx, ctl, schedule = build(case, xp, work / "b")
paused = complete(s, ctx, ctl, schedule, checkpoint_callback=lambda p: True)
dump("paused", paused)
save_checkpoint(work / "wet", paused, ctx, s["column"], s["sediment"], IDENTITY)
restored = load_checkpoint(work / "wet", ctx, s["column"], s["sediment"], IDENTITY)
dump("resumed", complete(dict(s, state0=restored.result.state), ctx, ctl, schedule, continuation=restored))
np.savez(out + ".npz", **arrays)
Path(out + ".json").write_text(json.dumps(counts, default=lambda v: host(v).tolist(), sort_keys=True,
                                          allow_nan=False))
