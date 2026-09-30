"""Run unchanged MAHLERAN water routines on controlled MAPLE Plot1 inputs.

Requires PYTHONPATH=src:tests/phase4 (reuse the existing Fortran toolchain
and source-hash harness). This is a bounded benchmark, not the MAHLERAN
application or a production SYRUP solver. Only dry initial Plot1 is supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from fortran_reference import (
    REFERENCE_SOURCES,
    _to_full,
    legacy_step,
    locate_toolchain,
    reference_hashes,
    watched_listing,
)

from maple_syrup.case_import import _refuse_output, verify_plot1_case
from maple_syrup.column_experiment import plot1_parameters
from maple_syrup.provenance import capture_syrup_provenance, source_tree_digest
from maple_syrup.rainfall import parse_legacy_rainfall_file
from maple_syrup.routing import plot1_routing_graph

REPO = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).with_name("reference_storm_driver.f90")
MARKER = "SYRUP_COUPLED_AUDIT_COMPLETE"
MAX_STEPS = 100_000
SOURCES = (
    "src/Program_Control/shared_data.f90",
    "src/Subroutines_Water/ff_type8.for",
    "src/Subroutines_Water/infilt.for",
    "src/Subroutines_Water/route_water.for",
    "src/Subroutines_Water/update_water_flow.for",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(case_dir: Path, output_dir: Path, dt: float, end: float) -> dict:
    if not (math.isfinite(dt) and dt > 0 and math.isfinite(end) and end > 0):
        raise ValueError("dt and end must be finite and positive")
    ratio = end / dt
    if (
        not math.isfinite(ratio)
        or ratio > MAX_STEPS
        or ratio != round(ratio)
        or ratio < 1
    ):
        raise ValueError(
            f"end must be an exact multiple of dt, with 1..{MAX_STEPS} steps"
        )
    n = round(ratio)
    if float(np.float32(dt)) != dt:
        raise ValueError(
            "dt must be exactly representable in the original default REAL kind"
        )
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_relative_to(case_dir.resolve()):
        raise ValueError("output must be a new directory outside the bound case")
    v = verify_plot1_case(case_dir)
    root = Path(v.report["recipe"]["mahleran_root"]).resolve()
    _refuse_output(
        output_dir,
        {
            "MAHLERAN": root,
            "MAPLE": v.maple_dependency.source_root,
            "SYRUP source": REPO / "src",
            "recipe": Path(v.report["recipe"]["recipe_path"]).parent,
        },
    )
    before = reference_hashes(root)
    if before != REFERENCE_SOURCES:
        raise RuntimeError(
            "original MAHLERAN sources differ from the audited reference hashes"
        )
    listing = watched_listing(root)
    g = plot1_routing_graph(v.fields, v.report)
    if not np.all(g.active) or np.any(v.case.water.depth_m != 0):
        raise ValueError(
            "this reference driver supports fully active, initially dry Plot1 only"
        )
    p, parameter_record = plot1_parameters(v.report, v.fields)
    schedule = parse_legacy_rainfall_file(v.rainfall_path)
    if schedule.provenance.sha256 != v.report["rainfall"]["sha256"]:
        raise RuntimeError("rainfall changed after case verification")
    for edge in schedule.edges_s[(schedule.edges_s > 0) & (schedule.edges_s < end)]:
        if float(edge / dt) != round(float(edge / dt)):
            raise ValueError("fixed reference steps must land on every forcing knot")
    tc = locate_toolchain()
    if tc is None:
        raise RuntimeError(
            "gfortran unavailable; set MAPLE_SYRUP_GFORTRAN and flags as in docs/phase4/routing.md"
        )
    zero = np.zeros(g.shape)
    legacy = legacy_step(
        g,
        old_flow_depth_m=zero,
        depth_start_m=zero,
        old_discharge_m2_s=zero,
        old_inflow_m2_s=zero,
        dt_s=dt,
        ring_export_full=v.fields["legacy_full_rainfall_scaling"] < 0,
    )
    fields = [
        legacy.aspect,
        legacy.rmask,
        legacy.slope,
        legacy.ff,
        _to_full(p["ksat_m_per_s"] * 1000, 0.0),
        _to_full(p["suction_m"] * 1000, 0.0),
        _to_full(p["pavement_cover_fraction"] * 0.01, 0.0),
        _to_full(p["drainage_parameter"], 0.0),
        _to_full(p["theta_sat"], 0.4),
        _to_full(p["initial_theta"], 0.25),
        _to_full(p["initial_theta"] * p["soil_thickness_m"] * 1000, 0.0),
        _to_full(p["theta_sat"] * p["soil_thickness_m"] * 1000, 1.0),
        _to_full(p["rainfall_scale"], 0.0),
        _to_full(g.outlet.astype(float), 0.0),
        _to_full(g.active.astype(float), 0.0),
    ]
    record = {
        "status": "started",
        "scope": "unchanged infilt/route_water/update_water_flow in a controlled water-only driver; NOT the MAHLERAN application",
        "dt_s": dt,
        "end_s": end,
        "n_steps": n,
        "max_history_rows": MAX_STEPS,
        "case_identity_sha256": v.binding["maple_case_identity_sha256"],
        "case_binding": v.binding,
        "graph_sha256": g.input_sha256,
        "rainfall_sha256": schedule.provenance.sha256,
        "forcing_convention": schedule.provenance.convention,
        "parameters": parameter_record,
        "syrup_provenance": capture_syrup_provenance(),
        "maple_provenance": v.maple_provenance,
        "source_sha256": before,
        "driver_sha256": sha(DRIVER),
        "writer_sha256": sha(__file__),
        "helper_sha256": sha(REPO / "tests/phase4/fortran_reference.py"),
        "commands": [],
        "departures_from_application": [
            "controlled exact rainfall intervals, not set_rain_xml switching",
            "deterministic mean Ksat, no stochastic draw/XML setup",
            "prescribed common parameters in FP64; legacy infilt lambda literals remain original precision",
            "no sediment/chemistry; zero dummy sediment arrays for update_water_flow",
            "accumulate_flow omitted: method5 does not use add/dup",
            "literal stale qin and legacy root bracket retained; diagnostics do not correct them",
        ],
    }
    output_dir.mkdir(parents=True)
    build = output_dir / "build"
    build.mkdir()
    inp = output_dir / "input.dat"
    with inp.open("x") as f:
        f.write(
            f"{legacy.aspect.shape[0]} {legacy.aspect.shape[1]} {g.n_active} {n} {dt:.17e} {legacy.dx_mm:.17e}\n"
        )
        for row in legacy.order:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
        for i in range(legacy.aspect.shape[0]):
            for j in range(legacy.aspect.shape[1]):
                f.write(
                    " ".join(
                        str(int(a[i, j]))
                        if k in (0, 13, 14)
                        else format(a[i, j], ".17e")
                        for k, a in enumerate(fields)
                    )
                    + "\n"
                )
        for k in range(n):
            f.write(format(schedule.rate_after_m_per_s(k * dt) * 1000, ".17e") + "\n")
    record["input_sha256"] = sha(inp)
    record_path = output_dir / "reference_summary.json"

    def save():
        record_path.write_text(json.dumps(record, indent=2, default=str) + "\n")

    def command(args):
        record["commands"].append(args)
        save()
        environment = os.environ.copy()
        if tc.run_library_path:
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                [*tc.run_library_path, environment.get("LD_LIBRARY_PATH", "")]
            )
        result = subprocess.run(
            args, cwd=build, env=environment, capture_output=True, text=True, timeout=300, check=False
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result

    try:
        base = [
            tc.compiler,
            *tc.flags,
            "-std=legacy",
            "-fcheck=all",
            "-ffixed-line-length-none",
            "-ffree-line-length-none",
            f"-I{build}",
            f"-J{build}",
        ]
        record["compiler_version"] = command(
            [tc.compiler, "--version"]
        ).stdout.splitlines()[0]
        objects = []
        for source in [*[root / name for name in SOURCES], DRIVER]:
            obj = build / (source.stem + ".o")
            command([*base, "-c", str(source), "-o", str(obj)])
            objects.append(str(obj))
        executable = build / "reference_storm"
        command([*base, *objects, *tc.link_flags, "-o", str(executable)])
        hist, final = output_dir / "hydrograph.dat", output_dir / "final.dat"
        start = time.perf_counter()
        result = command([str(executable), str(inp), str(hist), str(final)])
        record["run_wall_s"] = time.perf_counter() - start
        (output_dir / "stdout.log").write_text(result.stdout)
        (output_dir / "stderr.log").write_text(result.stderr)
        if not hist.is_file() or not hist.read_text().endswith(MARKER + "\n"):
            raise RuntimeError(
                "original routines stopped before completion (STOP may return status zero)"
            )
        data = np.loadtxt(hist, skiprows=1, max_rows=n)
        data = np.atleast_2d(data)
        columns = hist.read_text().splitlines()[0].split()
        final_data = np.atleast_2d(np.loadtxt(final))
        if data.shape != (n, len(columns)) or final_data.shape != (g.n_active, 5):
            raise RuntimeError("unexpected reference output shape")
        if not np.all(np.isfinite(data)) or not np.all(np.isfinite(final_data)):
            raise RuntimeError("nonfinite reference result")
        expected_rain = (
            schedule.depth_m(0.0, end) * float(p["rainfall_scale"].sum()) * g.dx_m**2
        )
        if not math.isclose(
            float(data[-1, 6]), expected_rain, rel_tol=1e-10, abs_tol=1e-13
        ):
            raise RuntimeError(
                "reference forcing integral differs from supplied schedule"
            )
        if reference_hashes(root) != before or watched_listing(root) != listing:
            raise RuntimeError("original reference tree changed during benchmark")
        if (
            sha(DRIVER) != record["driver_sha256"]
            or sha(__file__) != record["writer_sha256"]
        ):
            raise RuntimeError("benchmark source changed during execution")
        if sha(REPO / "tests/phase4/fortran_reference.py") != record["helper_sha256"]:
            raise RuntimeError("reference helper changed during execution")
        for label, package, provenance in (
            ("maple_syrup", Path(record["syrup_provenance"]["package_dir"]), record["syrup_provenance"]),
            ("maple", v.maple_dependency.package_dir, record["maple_provenance"]),
        ):
            current = source_tree_digest(package).digest_sha256
            if current != provenance["package_source_digest"]["digest_sha256"]:
                raise RuntimeError(f"{label} package changed during benchmark")
        record["build_scratch_listing"] = sorted(p.name for p in build.iterdir())
        record.update(
            status="complete",
            final=dict(zip(columns, data[-1].tolist(), strict=True)),
            peak_outlet_m3_s=float(data[:, 1].max()),
            peak_time_s=float(data[np.argmax(data[:, 1]), 0]),
            budget_minus_stale_minus_closure_m3=float(
                data[-1, 7] - data[-1, 8] - data[-1, 9]
            ),
            output_sha256={p.name: sha(p) for p in [hist, final]},
            reference_tree_unchanged=True,
        )
        save()
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        save()
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--end-s", type=float, default=5400.0)
    args = parser.parse_args()
    result = run(args.case_dir, args.output_dir, args.dt_s, args.end_s)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "dt_s",
                    "end_s",
                    "final",
                    "peak_outlet_m3_s",
                    "peak_time_s",
                    "budget_minus_stale_minus_closure_m3",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
