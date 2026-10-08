"""CLI for the original-routine Fortran sediment reference (task gpu_sediment, A2). Nothing here was executed by its author.

    # 1. build (NEW directories only). The hooked build records the actual pre-clip trial; the nohook build is the no-effect gate.
    python benchmarks/legacy_sediment/run_fortran_sediment.py build --build-dir <NEW> [--hooked yes|no] [--variant timing|checked]
    # 2. prepare the binary input from a verified A1 case (RFID / Chastre; Plot 1 uses the existing whole-application ledger)
    python benchmarks/legacy_sediment/run_fortran_sediment.py prepare --case-kind rfid --case outputs/rfid/case \\
        --output-dir <NEW> --end-s 600 --capture-steps 300,600 --snapshot-steps 300,600 --allow-maple-source-change [--hash-tiles]
    # 3. run (a fresh process; completion marker and strict schema parsing are mandatory). The output ROOT may already exist
    #    (and is reused) but `<root>/run` must not.
    python benchmarks/legacy_sediment/run_fortran_sediment.py run --build-dir <B> --prepared <P> --output-dir <ROOT> \\
        [--rehash-tiles --allow-maple-source-change]
    # 4. compare with an A1 output directory (observations; >1 % flags trigger investigation, nothing is fitted)
    python benchmarks/legacy_sediment/run_fortran_sediment.py compare --run <ROOT> --prepared <P> --a1 <A1 dir> --output <NEW report.json>
    # 5. state injection (equation-level, rtol 2e-6 / atol 1e-14) on a captured step
    python benchmarks/legacy_sediment/run_fortran_sediment.py inject --run <ROOT> --prepared <P> --case-kind rfid --case outputs/rfid/case \\
        --step 600 [--depth previous] --output <NEW report.json>

Every output path is validated BEFORE anything is created: it may not equal, lie inside or contain the MAHLERAN reference, the project
src/tests/benchmarks/cases trees, the MAPLE dependency, the case, the build, prepared, input or executable directories; reports are
created exclusively (never overwritten); failed runs keep their partial output. Progress goes to stderr outside the kernel timers.
`prepare` re-hashes the bound XML / rainfall / vegetation / sidecar artifacts after preparation (and optionally the streamed Chastre
tiles once) and records exactly what was checked; `run` repeats the artifact check after the run.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_legacy_sediment as C
import sources as S


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _steps(text: str) -> list[int]:
    return sorted({int(v) for v in text.split(",") if v.strip()}) if text else []


def _protected(case_dir=None, extra=None) -> dict:
    """Protected trees: the project trees, MAHLERAN, MAPLE (A1's `protected_roots`, read-only) plus the case and `extra`."""
    roots: dict = {}
    try:
        from maple_syrup.legacy_driver import protected_roots

        roots.update(protected_roots(Path(case_dir).resolve() if case_dir else Path("/nonexistent-case")))
    except ImportError:  # the A1 modules are only needed for prepare/inject; build/run/compare still protect the reference
        pass
    if case_dir:
        roots["case"] = Path(case_dir).resolve()
    roots.update(extra or {})
    return roots


def _pins(paths) -> dict[str, str]:
    return {str(p): S.sha256_file(p) for p in paths}


def _tiles(case) -> dict:
    digest_fn = getattr(case.verified.case, "persisted_digest", None)
    if digest_fn is None:
        return {"performed": False, "note": "no tile bed in this case kind"}
    digest = digest_fn()
    return {"performed": True, "digest": digest, "equals_bound_digest": digest == case.verified.case.bound_tiles_digest}


def cmd_build(a) -> int:
    record = S.build(a.build_dir, hooked=a.hooked == "yes", variant=a.variant)
    print(json.dumps({k: record[k] for k in ("executable", "executable_sha256", "hooked", "variant", "compiler_version",
                                             "route_sediment_patch_diff_file")}, indent=2))
    return 0


def cmd_prepare(a) -> int:
    from maple_syrup.legacy_case import legacy_case_for

    case_dir = Path(a.case).resolve()
    out = S.new_output_root(a.output_dir, _protected(case_dir))
    if os.path.lexists(out):
        raise S.FortranGlueError(f"refusing to reuse existing prepared directory {out}")
    progress("verifying the case and adapting it")
    case = legacy_case_for(a.case_kind, a.case, allow_maple_source_change=a.allow_maple_source_change,
                           hash_only_tile_verify=a.hash_only_tile_verify, end_s=a.end_s)
    dep = case.verified.maple_dependency
    roots = _protected(case_dir, {"verified MAPLE source": Path(dep.source_root), "verified MAPLE package": Path(dep.package_dir),
                                  **{f"bound input {i}": Path(p) for i, p in enumerate(case.pin_paths)}})
    mah_root = (case.verified.report.get("mahleran") or {}).get("root")
    if mah_root:
        roots["verified MAHLERAN"] = Path(mah_root)
    out = S.new_output_root(out, roots)  # again, with everything the case revealed, BEFORE creating anything
    pins_before = _pins(case.pin_paths)
    caps, snaps = _steps(a.capture_steps), _steps(a.snapshot_steps)
    progress("building the Fortran arrays from the verified case")
    arrays, kwargs = S.inputs_from_legacy_case(case, end_s=a.end_s, iroute=a.iroute, capture_steps=caps, snapshot_steps=snaps)
    out.mkdir(parents=True)
    stats: dict = {}
    progress(f"writing {out / 'input.bin'}")
    digest = S.write_input(out / "input.bin", arrays, progress=progress, stats=stats, **kwargs)
    nr2, nc2 = arrays["aspect"].shape
    pins_after = _pins(case.pin_paths)
    tiles = _tiles(case) if a.hash_tiles else {"performed": False, "note": "not requested (--hash-tiles)"}
    unchanged = pins_before == pins_after and tiles.get("equals_bound_digest", True)
    meta = {"status": "ok" if unchanged else "FAILED: bound artifacts changed during preparation",
            "case_kind": a.case_kind, "case": str(case_dir), "n_steps": len(kwargs["rates_mm_s"]), "iroute": a.iroute,
            "nr2": nr2, "nc2": nc2, "active_cells": int(arrays["n_active"]), "capture_steps": caps, "snapshot_steps": snaps,
            "input_sha256": digest, "input_stats": stats, "xml_sha256": kwargs["xml"]["xml_sha256"],
            "selector_in_xml": kwargs["xml"]["selector_in_xml"],
            "selector_used": "Crank-Nicolson (2); the RFID XML has none and the native diagnostic used Euler (1)",
            "depth_convention": "original: post-infiltration d(1); SYRUP A1 default: previous step's depth (quantify by injection)",
            "terminal_storage_cells": int(((np.asarray(arrays["aspect"]) == 0) & (np.asarray(arrays["active"]) == 1)
                                           & (np.asarray(arrays["outlet"]) == 0)).sum()),
            "case_pins": pins_before, "case_pins_unchanged_after_prepare": pins_before == pins_after,
            "tiles_after_prepare": tiles, "allow_maple_source_change": bool(a.allow_maple_source_change),
            "hash_only_tile_verify": bool(a.hash_only_tile_verify),
            "peak_rss_kib_prepare": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "case_record_hydrology": case.record.get("hydrology_parameters")}
    S.exclusive_text(out / "meta.json", json.dumps(meta, indent=2, default=str) + "\n", roots)
    print(json.dumps({k: meta[k] for k in ("status", "n_steps", "input_sha256", "input_stats", "peak_rss_kib_prepare")}, indent=2))
    return 0 if unchanged else 1


def _expected(meta: dict, hooked: bool) -> dict:
    return {"n_steps": meta["n_steps"], "iroute": meta["iroute"], "nr2": meta["nr2"], "nc2": meta["nc2"], "hooked": hooked,
            "capture_steps": meta["capture_steps"], "snapshot_steps": meta["snapshot_steps"], "active_cells": meta["active_cells"]}


def cmd_run(a) -> int:
    build = json.loads((Path(a.build_dir) / "build.json").read_text())
    prepared = Path(a.prepared).resolve()
    meta = json.loads((prepared / "meta.json").read_text())
    if meta.get("status") != "ok":
        raise S.FortranGlueError(f"the prepared input is not usable: {meta.get('status')}")
    input_path = prepared / "input.bin"
    if S.sha256_file(input_path) != meta["input_sha256"]:
        raise S.FortranGlueError("input.bin does not hash to the prepared digest")
    exe = Path(build["executable"]).resolve()
    protected = _protected(meta["case"], {"build directory": Path(a.build_dir).resolve(), "prepared directory": prepared,
                                          "input directory": input_path.parent, "executable directory": exe.parent})
    root = S.new_output_root(a.output_dir, protected)
    if os.path.lexists(root / "run"):
        raise S.FortranGlueError(f"refusing: {root / 'run'} already exists")
    root.mkdir(parents=True, exist_ok=True)
    expected = _expected(meta, build["hooked"])
    progress(f"running the Fortran reference ({meta['n_steps']} steps) in {root / 'run'}")
    record = S.run_once(exe, input_path, root / "run", expected=expected, build_record=build, extra_protected=protected)
    light = {k: v for k, v in record.items() if k not in ("ledger", "water_steps", "counts", "maps", "captures", "snapshots")}
    if record.get("status") != "complete":
        S.exclusive_text(root / "run_failed.json", json.dumps({**light, "build": build, "expected": expected}, indent=2, default=str)
                         + "\n", protected)
        print(json.dumps(light, indent=2, default=str), file=sys.stderr)
        return 1
    pins_after = _pins(Path(p) for p in meta["case_pins"])
    pins_ok = pins_after == meta["case_pins"]
    tiles = {"performed": False, "note": "not requested (--rehash-tiles)"}
    if a.rehash_tiles:
        from maple_syrup.legacy_case import legacy_case_for

        tiles = _tiles(legacy_case_for(meta["case_kind"], meta["case"], allow_maple_source_change=a.allow_maple_source_change,
                                       hash_only_tile_verify=True))
    status = "complete" if pins_ok and tiles.get("equals_bound_digest", True) else "FAILED: bound artifacts changed during the run"
    S.exclusive_text(root / "run_record.json", json.dumps({
        **light, "status": status, "build": build, "expected": expected, "input_sha256": meta["input_sha256"],
        "case_pins_checked_after_run": {"performed": True, "unchanged": pins_ok, "files": len(pins_after)},
        "tiles_after_run": tiles, "peak_rss_kib_report": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
        indent=2, default=str) + "\n", protected)
    print(json.dumps({"status": status, **{k: light[k] for k in ("loop_s", "diag_s", "capture_s", "kernel_s", "max_rss_kib",
                                                                    "process_wall_s")}}, indent=2))
    return 0 if status == "complete" else 1


def _reload(root: Path, meta: dict) -> dict:
    """Re-read a finished run from disk with the SAME strict loader `run_once` uses (no re-execution)."""
    info = json.loads((Path(root) / "run_record.json").read_text())
    if info.get("status") != "complete":
        raise S.FortranGlueError(f"the run is not complete: {info.get('status')}")
    rec = S.load_run_outputs(Path(root) / "run", _expected(meta, info["build"]["hooked"]))
    rec.update({k: info[k] for k in ("loop_s", "diag_s", "capture_s", "kernel_s", "process_wall_s", "max_rss_kib")})
    return rec


def cmd_compare(a) -> int:
    prepared = Path(a.prepared).resolve()
    meta = json.loads((prepared / "meta.json").read_text())
    protected = _protected(meta["case"], {"prepared directory": prepared, "run directory": Path(a.run).resolve()})
    report = C.compare_runs(_reload(Path(a.run), meta), a.a1, nr2=meta["nr2"], nc2=meta["nc2"])
    S.exclusive_text(a.output, json.dumps(report, indent=2, default=str) + "\n", protected)
    print(json.dumps({"flags": report["flags"], "first_positive_pickup_step": report["first_positive_pickup_step"]}, indent=2))
    return 0


def cmd_inject(a) -> int:
    from maple_syrup.legacy_case import legacy_case_for

    prepared = Path(a.prepared).resolve()
    meta = json.loads((prepared / "meta.json").read_text())
    protected = _protected(meta["case"], {"prepared directory": prepared, "run directory": Path(a.run).resolve()})
    rec = _reload(Path(a.run), meta)
    case = legacy_case_for(a.case_kind, a.case, allow_maple_source_change=a.allow_maple_source_change,
                           hash_only_tile_verify=a.hash_only_tile_verify)
    af, dx_mm = S._f(rec["result"], "AF_KG_PER_MM"), S._f(rec["result"], "DX_MM")
    density = S._f(rec["result"], "DENSITY_G_CM3")  # the widened kind-4 value the original used (not the nominal XML 2.65)
    engine, _ = C.build_engine(case.graph, case.sediment, case.grid, case.vegetation, case.holdings_kg)
    report = C.injection_check(rec["captures"][a.step], engine, nr2=meta["nr2"], nc2=meta["nc2"], af=af, dx_mm=dx_mm,
                               density_g_cm3=density, depth=a.depth)
    report["xml_density_g_cm3_nominal"] = S.xml_sediment_block(case.record["xml"]["xml_path"],
                                                               case.record["xml"]["xml_sha256"])["particle_density_g_cm3"]
    if a.output:
        S.exclusive_text(a.output, json.dumps(report, indent=2) + "\n", protected)
    print(json.dumps(report, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--build-dir", required=True)
    b.add_argument("--hooked", choices=("yes", "no"), default="yes")
    b.add_argument("--variant", choices=tuple(S.BUILD_FLAGS), default="timing")
    b.set_defaults(fn=cmd_build)
    q = sub.add_parser("prepare")
    q.add_argument("--case-kind", choices=("rfid", "chastre"), required=True)
    q.add_argument("--case", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--end-s", type=float, default=None)
    q.add_argument("--iroute", type=int, choices=(2, 5), default=5)
    q.add_argument("--capture-steps", default="")
    q.add_argument("--snapshot-steps", default="")
    q.add_argument("--allow-maple-source-change", action="store_true")
    q.add_argument("--hash-only-tile-verify", action="store_true")
    q.add_argument("--hash-tiles", action="store_true", help="stream-hash every Chastre tile once after preparation")
    q.set_defaults(fn=cmd_prepare)
    r = sub.add_parser("run")
    r.add_argument("--build-dir", required=True)
    r.add_argument("--prepared", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--rehash-tiles", action="store_true", help="stream-hash every Chastre tile once after the run")
    r.add_argument("--allow-maple-source-change", action="store_true")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("--run", required=True)
    c.add_argument("--prepared", required=True)
    c.add_argument("--a1", required=True)
    c.add_argument("--output", required=True)
    c.set_defaults(fn=cmd_compare)
    i = sub.add_parser("inject")
    i.add_argument("--run", required=True)
    i.add_argument("--prepared", required=True)
    i.add_argument("--case-kind", choices=("rfid", "chastre"), required=True)
    i.add_argument("--case", required=True)
    i.add_argument("--step", type=int, required=True)
    i.add_argument("--depth", choices=("post_infiltration", "previous"), default="post_infiltration")
    i.add_argument("--output", default=None)
    i.add_argument("--allow-maple-source-change", action="store_true")
    i.add_argument("--hash-only-tile-verify", action="store_true")
    i.set_defaults(fn=cmd_inject)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except S.FortranGlueError as exc:
        print(f"legacy sediment reference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
