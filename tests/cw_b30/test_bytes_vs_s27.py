"""CW-B30: complete runner outputs in FRESH processes are byte-identical to the immutable S27 baseline source.

Each tree runs `runner_dump.py` in its own subprocesses (its own `src` on the path; the imported package path is
asserted): quiet completion + dry reset (cold and warm in one process), a wet pause with its checkpoint, a resume of
that checkpoint in a fresh process to completion, and a fresh-process load of the completed handoff checkpoint.
Every array of every output and checkpoint npz is compared byte for byte, plus the normalised summaries (timing,
path, output-file-hash and SYRUP-source-digest fields excluded: zip timestamps and the source digest differ by
construction). Resource figures are recorded for the report, not asserted.

CW-F2: the candidate's runner summaries may carry ONE additive key, `restart_resource_bound` (the resume byte-bound
resource record, excluded from every identity), that the older baseline source does not write. It is tolerated only
at the top level of a runner summary, only when the baseline summary lacks it, and only after its exact keys, positive
integer bound, documented resolution, `applies_to` and `identity` texts validate (`validate_restart_bound`). Every
other summary key must remain equal and every raw array comparison is unchanged."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from b30_cases import KINDS

B30_ROOT = Path(__file__).resolve().parents[2]
S27_ROOT = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b27_integrated_frozen")
MANIFEST = Path("/home/okin/SYRUP/agent_handoffs/tasks/conservative_water/cw_b30_baseline_manifest.json")
DUMP = Path(__file__).resolve().parent / "runner_dump.py"
MODES = ("complete", "pause", "resume", "load")


def s27_tree():
    if not MANIFEST.is_file():
        pytest.skip("B30 baseline manifest unavailable")
    manifest = json.loads(MANIFEST.read_text())
    for rel, digest in manifest.items():
        if rel.startswith(("src/", "tests/phase5/", "tests/conservative_water/")):
            path = S27_ROOT / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                pytest.skip(f"S27 baseline changed ({rel}); no comparison claim")
    return S27_ROOT


def run_tree(tree, kind, work):
    maple = os.environ.get("MAPLE_SYRUP_EXPECTED_MAPLE_ROOT", "")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    keep = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
            if p and (not p.rstrip("/").endswith("/src") or (maple and p.startswith(maple)))]
    env["PYTHONPATH"] = os.pathsep.join([*keep, str(tree / "src")])
    work.mkdir(parents=True)
    records = {}
    for mode in MODES:
        out = work / f"{mode}.json"
        proc = subprocess.run([sys.executable, str(DUMP), str(tree / "tests"), kind, mode, str(work), str(out)],
                              env=env, capture_output=True, text=True, timeout=3600, check=False)
        assert proc.returncode == 0, f"{tree.name} {mode}: " + proc.stdout[-3000:] + proc.stderr[-3000:]
        record = json.loads(out.read_text())
        assert Path(record["maple_syrup"]) == tree / "src" / "maple_syrup"
        with np.load(work / f"{mode}_arrays.npz", allow_pickle=False) as z:
            record["arrays"] = {k: z[k].copy() for k in z.files}
        records[mode] = record
    return records


RESTART_BOUND_KEY = "restart_resource_bound"
RESTART_BOUND_APPLIES_TO = ("summed uncompressed payload bytes of a resumed bundle (zip central directories, checked "
                            "before any array is materialized)")
RESTART_BOUND_IDENTITY = "resource control only; excluded from the run and checkpoint identity"
DEFAULT_CHECKPOINT_BYTES = 1024**3
#: resolution text -> whether the record carries `max_host_bytes` (the resident-window runner) or not (the dense runner)
RESTART_BOUND_RESOLUTIONS = {
    "default of the dense runner (1 GiB)": False,
    "caller-explicit": False,
    "resident-window default: equal to the explicit max_host_bytes cap": True,
    "caller-explicit, tightened within max_host_bytes": True,
}


class RestartBoundRecordError(AssertionError):
    """An invalid `restart_resource_bound` record or an untolerated summary difference."""


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_restart_bound(record):
    """Exact validation of one CW-F2 `restart_resource_bound` record (raises RestartBoundRecordError)."""
    base = {"max_checkpoint_uncompressed_bytes", "resolution", "applies_to", "identity"}
    if not isinstance(record, dict):
        raise RestartBoundRecordError(f"restart bound record is not a mapping: {record!r}")
    resolution = record.get("resolution")
    if resolution not in RESTART_BOUND_RESOLUTIONS:
        raise RestartBoundRecordError(f"undocumented restart bound resolution {resolution!r}")
    windowed = RESTART_BOUND_RESOLUTIONS[resolution]
    if set(record) != (base | {"max_host_bytes"} if windowed else base):
        raise RestartBoundRecordError(f"restart bound record keys {sorted(record)} invalid for {resolution!r}")
    bound = record["max_checkpoint_uncompressed_bytes"]
    if not _positive_int(bound):
        raise RestartBoundRecordError(f"restart bound is not a positive integer: {bound!r}")
    if record["applies_to"] != RESTART_BOUND_APPLIES_TO or record["identity"] != RESTART_BOUND_IDENTITY:
        raise RestartBoundRecordError("restart bound applies_to / identity text differs from the documented record")
    if resolution == "default of the dense runner (1 GiB)" and bound != DEFAULT_CHECKPOINT_BYTES:
        raise RestartBoundRecordError("dense default resolution with a bound other than 1 GiB")
    if resolution == "caller-explicit" and bound == DEFAULT_CHECKPOINT_BYTES:
        raise RestartBoundRecordError("caller-explicit resolution with exactly the dense default")
    if windowed:
        host = record["max_host_bytes"]
        if not _positive_int(host):
            raise RestartBoundRecordError(f"max_host_bytes is not a positive integer: {host!r}")
        if resolution.startswith("resident-window default") and bound != host:
            raise RestartBoundRecordError("resident-window default bound differs from max_host_bytes")
        if bound > host:
            raise RestartBoundRecordError("resident-window bound exceeds max_host_bytes")
    return record


def comparable_candidate_summaries(baseline, candidate):
    """The candidate summaries with ONLY a validated additive top-level `restart_resource_bound` removed from a RUNNER
    summary (a mapping with `status` and `identity`) whose baseline summary lacks it. Labels must match; nothing else is removed or tolerated (compare the result with ==)."""
    if set(baseline) != set(candidate):
        raise RestartBoundRecordError(f"summary labels differ: {sorted(baseline)} vs {sorted(candidate)}")
    out = {}
    for label, summary in candidate.items():
        old = baseline[label]
        runner_summary = isinstance(summary, dict) and "status" in summary and "identity" in summary
        if runner_summary and RESTART_BOUND_KEY in summary and not (isinstance(old, dict) and RESTART_BOUND_KEY in old):
            validate_restart_bound(summary[RESTART_BOUND_KEY])
            summary = {k: v for k, v in summary.items() if k != RESTART_BOUND_KEY}
        out[label] = summary
    return out


PROVENANCE_TEXT = ("provenance_json", "metadata")  # MAPLE snapshot provenance text inside checkpoint payloads


def provenance_equal(a, b):
    """Snapshot provenance text: equal as JSON after removing exactly the two fields that differ by construction
    between the trees (the SYRUP source digest in the bound identity, and the per-run temporary deep-store path)."""
    def clean(raw):
        value = json.loads(raw.item() if raw.shape == () else raw.tobytes())
        value.pop("store_dir", None)
        for holder in (value, value.get("provenance", {})):
            if isinstance(holder, dict) and isinstance(holder.get("identity"), dict):
                holder["identity"].pop("syrup_source_sha256", None)
        return value
    return a.dtype.kind in "SU" and b.dtype.kind in "SU" and clean(a) == clean(b)


@pytest.mark.parametrize("kind", KINDS)
def test_fresh_process_outputs_and_restarts_match_s27_bytes(kind, tmp_path):
    s27 = run_tree(s27_tree(), kind, tmp_path / "s27")
    b30 = run_tree(B30_ROOT, kind, tmp_path / "b30")
    resources = {}
    for mode in MODES:
        a, b = s27[mode]["arrays"], b30[mode]["arrays"]
        assert set(a) == set(b) and len(a) > 0, mode
        bad = [k for k in a if not (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                                    and a[k].tobytes() == b[k].tobytes())]
        text = [k for k in bad if k.rsplit("/", 1)[-1] in PROVENANCE_TEXT and provenance_equal(a[k], b[k])]
        bad = [k for k in bad if k not in text]
        assert len(text) <= 2 * len([k for k in a if k.rsplit("/", 1)[-1] in PROVENANCE_TEXT])
        assert not bad, f"{mode}: {len(bad)} arrays differ, first {bad[:5]}"
        assert s27[mode]["summaries"] == comparable_candidate_summaries(s27[mode]["summaries"],
                                                                        b30[mode]["summaries"]), mode
        resources[mode] = {"s27": s27[mode]["resources"], "b30": b30[mode]["resources"]}
    assert s27["complete"]["summaries"]["complete"]["status"] == "complete"
    assert s27["resume"]["summaries"]["resume"]["status"] == "complete"
    assert s27["pause"]["summaries"]["pause"]["status"] == "paused"
    assert s27["load"]["summaries"]["load"]["type"] == "DryHandoff"
    report = os.environ.get("CW_B30_RESOURCE_REPORT")
    if report:
        Path(report).with_name(Path(report).stem + f"_{kind}.json").write_text(json.dumps(resources, indent=1))
