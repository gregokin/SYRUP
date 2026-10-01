"""Audit saved whole-program runs; does not execute either model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from prepare_mahleran import sha

from maple_syrup.case_import import verify_plot1_case
from maple_syrup.rainfall import parse_legacy_rainfall_file
from maple_syrup.routing import plot1_routing_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads((p / "execution.json").read_text()) for p in (args.run, args.repeat)
    ]
    for root, record in zip((args.run, args.repeat), records):
        assert record["returncode"] == 0 and record["completion_marker"]
        assert all(
            record[k]
            for k in ("input_unchanged", "reference_unchanged", "prepared_unchanged")
        )
        for name, data in record["outputs"].items():
            assert sha(root / name) == data["sha256"], name
    files = [
        "SedChange.dat",
        "aspct001.asc",
        "depth001.asc",
        "detac001.asc",
        "dschg001.asc",
        "hydro001.dat",
        "ksat_001.asc",
        "neter001.asc",
        "pave.asc",
        "seddisch001.dat",
        "sedtr001.asc",
        "sedtr001.dat",
        "theta001.asc",
        "veloc001.asc",
    ]
    for name in files:
        assert sha(args.run / "Output" / name) == sha(args.repeat / "Output" / name), (
            name
        )
    hydro, sediment, classes = [
        np.loadtxt(args.run / "Output" / name)
        for name in ("hydro001.dat", "sedtr001.dat", "seddisch001.dat")
    ]
    for values, columns in ((hydro, 12), (sediment, 16), (classes, 7)):
        assert values.shape == (5400, columns) and np.isfinite(values).all()
        assert np.array_equal(values[:, 0], np.arange(1, 5401))
    log = (args.run / "stdout.log").read_text(errors="replace")
    applied = np.array(
        re.findall(
            r"Starting iteration\s+(\d+) rain intensity:\s+([\d.]+) time step:\s+([\d.]+)",
            log,
        ),
        dtype=float,
    )
    assert applied.shape == (5400, 3)
    assert np.array_equal(applied[:, 0], hydro[:, 0]) and np.all(applied[:, 2] == 1)
    case = verify_plot1_case(args.case)
    graph = plot1_routing_graph(case.fields, case.report)
    aspect = np.loadtxt(args.run / "Output/aspct001.asc", skiprows=6)[1:-1, 1:-1][::-1]
    assert np.array_equal(aspect, graph.aspect)
    raw_depth_mm = (
        parse_legacy_rainfall_file(args.run / "Input/input_p1/p1_01_08_06.dat").depth_m(
            0, 5400
        )
        * 1000
    )
    args.output.mkdir(parents=True, exist_ok=False)
    np.savetxt(
        args.output / "applied_rainfall.csv",
        np.column_stack((applied[:, 0] - 1, applied[:, 0], applied[:, 1])),
        delimiter=",",
        header="start_s,end_s,logged_applied_rain_mm_h",
        comments="",
    )
    np.savetxt(
        args.output / "outlet_series_si.csv",
        np.column_stack(
            (hydro[:, 0], hydro[:, 2] * 1e-9, sediment[:, 1], classes[:, 1:])
        ),
        delimiter=",",
        header="time_s,water_m3_s,sediment_kg_s,"
        + ",".join(f"class_{i}_kg_s" for i in range(1, 7)),
        comments="",
    )
    np.save(args.output / "routing_aspect_maple_order.npy", aspect.astype(np.int8))
    report = {
        "status": "completed_repeatable_reference; independent_review_pending",
        "run": str(args.run.resolve()),
        "repeat": str(args.repeat.resolve()),
        "executable_sha256": records[0]["executable_sha256"],
        "duration_s": 5400,
        "steps": 5400,
        "dt_s": 1,
        "wall_s": [r["wall_s"] for r in records],
        "peak_child_rss_kib": [r["peak_child_rss_kib"] for r in records],
        "identical_numeric_files": files,
        "routing_matches_imported_syrup_cells": int(aspect.size),
        "routing_conversion": "crop exterior ring, reverse north-first rows to MAPLE south-first",
        "input_schedule_depth_mm": raw_depth_mm,
        "logged_applied_depth_mm": float(np.sum(applied[:, 1] * applied[:, 2]) / 3600),
        "water_export_m3_from_rounded_output": float(hydro[:, 2].sum() * 1e-9),
        "peak_outlet_m3_s": float(hydro[:, 2].max() * 1e-9),
        "peak_outlet_time_s": float(hydro[np.argmax(hydro[:, 2]), 0]),
        "sediment_export_kg_from_rounded_output": float(sediment[:, 1].sum()),
        "class_export_kg_from_rounded_output": classes[:, 1:].sum(axis=0).tolist(),
        "limitations": [
            "Stock outputs are rounded, not a full-precision conservation ledger.",
            "Logged rainfall is printed to two decimals mm/h; save original forcing too.",
            "Legacy end-of-rain-file read warnings persist; logged subsequent rainfall is zero.",
            "Full-program timing includes initialization, output and sediment; no speed ratio to SYRUP.",
            "SYRUP frozen elevation/routing runner not yet implemented or executed.",
            "depth001 and veloc001 are synchronous fields at the maximum outlet discharge, not per-cell storm maxima or final fields.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
