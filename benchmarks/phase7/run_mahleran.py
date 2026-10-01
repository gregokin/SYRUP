"""Execute the prepared whole MAHLERAN Plot1 program in a new isolated run."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import time
from pathlib import Path

from prepare_mahleran import inventory, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout-s", type=float, default=180)
    args = p.parse_args()
    prepared, build, output = (
        args.prepared.resolve(),
        args.build.resolve(),
        args.output.resolve(),
    )
    manifest = json.loads((prepared / "benchmark_manifest.json").read_text())
    build_info = json.loads((build / "build.json").read_text())
    original = Path(manifest["reference_root"])
    if (
        inventory(prepared) != manifest["prepared_sha256"]
        or inventory(original) != manifest["original_sha256"]
    ):
        raise ValueError("prepared/reference inputs changed")
    if (
        build_info["status"] != "built"
        or sha(build / "mahleran") != build_info["executable_sha256"]
    ):
        raise ValueError("build incomplete or executable changed")
    for name, expected in build_info["source_sha256"].items():
        if sha(prepared / name) != expected:
            raise ValueError("build source differs from prepared copy")
    if output.exists() or any(
        output.is_relative_to(root) for root in (original, prepared, build)
    ):
        raise ValueError("new isolated run directory required")
    output.mkdir(parents=True)
    shutil.copytree(prepared / "Input", output / "Input")
    shutil.copy2(prepared / "mahleran_input.xml", output / "mahleran_input.xml")
    (output / "Output").mkdir()
    inputs = {
        str(f.relative_to(output)): sha(f) for f in output.rglob("*") if f.is_file()
    }
    command = [str(build / "mahleran")]
    report = {
        "command": command,
        "cwd": str(output),
        "input_sha256": inputs,
        "executable_sha256": sha(build / "mahleran"),
        "build_manifest": str(build / "build.json"),
        "status": "started",
    }
    start = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        with (
            (output / "stdout.log").open("w") as stdout,
            (output / "stderr.log").open("w") as stderr,
        ):
            result = subprocess.run(
                command,
                cwd=output,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout_s,
                check=False,
            )
        report["returncode"] = result.returncode
        text = (output / "stdout.log").read_text(errors="replace")
        report["completion_marker"] = "run completed" in text
        report["status"] = (
            "program_completed_pending_output_audit"
            if result.returncode == 0 and report["completion_marker"]
            else "failed"
        )
    except subprocess.TimeoutExpired:
        report["status"] = "timeout"
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        report.update(
            wall_s=time.perf_counter() - start,
            cpu_user_s=after.ru_utime - before.ru_utime,
            cpu_system_s=after.ru_stime - before.ru_stime,
            peak_child_rss_kib=after.ru_maxrss,
            input_unchanged=all(
                sha(output / name) == digest for name, digest in inputs.items()
            ),
            reference_unchanged=inventory(original) == manifest["original_sha256"],
            prepared_unchanged=inventory(prepared) == manifest["prepared_sha256"],
        )
        report["outputs"] = {
            str(f.relative_to(output)): {"sha256": sha(f), "bytes": f.stat().st_size}
            for f in sorted(output.rglob("*"))
            if f.is_file() and str(f.relative_to(output)) not in inputs
        }
        (output / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("outputs", "input_sha256")},
            indent=2,
        )
    )
    if report["status"] != "program_completed_pending_output_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
