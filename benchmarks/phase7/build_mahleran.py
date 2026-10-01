"""Build all MAHLERAN application sources selected by its original makefile."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from prepare_mahleran import inventory, sha


def build(source, output, *, checked=True):
    source, output = source.resolve(), output.resolve()
    manifest = json.loads((source / "benchmark_manifest.json").read_text())
    if inventory(source) != manifest["prepared_sha256"]:
        raise ValueError("prepared source does not match its manifest")
    if output.exists() or output.is_relative_to(source):
        raise ValueError("new build directory outside prepared source required")
    output.mkdir(parents=True)
    makefile = source / "nbproject/Makefile-Release.mk"
    selected = re.findall(
        r"-o \$\{OBJECTDIR\}/.+?\.o (src/.+?)\s*$", makefile.read_text(), re.MULTILINE
    )
    selected = [name.replace("\\ ", " ") for name in selected]
    if len(selected) != len(set(selected)) or not selected:
        raise ValueError("ambiguous original build source list")
    # Legacy comments contain non-UTF8 bytes; inspect ASCII module names
    # through a lossless single-byte decode, compile original bytes unchanged.
    texts = {name: (source / name).read_text(encoding="latin-1") for name in selected}
    modules = {}
    for name, text in texts.items():
        for module in re.findall(
            r"^\s*module\s+(?!procedure\b)(\w+)", text, re.IGNORECASE | re.MULTILINE
        ):
            if module.lower() in modules:
                raise ValueError("duplicate module provider")
            modules[module.lower()] = name
    dependencies = {
        name: {
            modules[m.lower()]
            for m in re.findall(r"^\s*use\s+(\w+)", text, re.IGNORECASE | re.MULTILINE)
            if m.lower() in modules and modules[m.lower()] != name
        }
        for name, text in texts.items()
    }
    ordered = []
    while len(ordered) < len(selected):
        ready = [
            name
            for name in selected
            if name not in ordered and dependencies[name].issubset(ordered)
        ]
        if not ready:
            raise ValueError("module dependency cycle")
        ordered.extend(ready)
    compiler = os.environ.get("MAPLE_SYRUP_GFORTRAN", "gfortran")
    flags = shlex.split(os.environ.get("MAPLE_SYRUP_GFORTRAN_FLAGS", ""))
    flags += [
        "-O2",
        "-std=legacy",
        "-ffree-line-length-none",
        "-ffixed-line-length-none",
        "-fallow-argument-mismatch",
    ]
    if checked:
        flags += ["-fcheck=all", "-fbacktrace"]
    ldflags = shlex.split(os.environ.get("MAPLE_SYRUP_GFORTRAN_LDFLAGS", ""))
    report = {
        "source_root": str(source),
        "source_sha256": {name: sha(source / name) for name in selected},
        "makefile_sha256": sha(makefile),
        "compiler": compiler,
        "compiler_version": subprocess.check_output(
            [compiler, "--version"], text=True
        ).splitlines()[0],
        "flags": flags,
        "ldflags": ldflags,
        "checked": checked,
        "commands": [],
        "status": "building",
    }
    start = time.perf_counter()
    objects = []
    try:
        for i, name in enumerate(ordered):
            obj = output / f"{i:03d}.o"
            cmd = [compiler, *flags, "-c", str(source / name), "-o", str(obj)]
            result = subprocess.run(
                cmd,
                cwd=output,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            (output / f"{i:03d}.stdout").write_text(result.stdout)
            (output / f"{i:03d}.stderr").write_text(result.stderr)
            report["commands"].append(
                {"source": name, "command": cmd, "returncode": result.returncode}
            )
            result.check_returncode()
            objects.append(obj)
        cmd = [
            compiler,
            *flags,
            *map(str, objects),
            *ldflags,
            "-o",
            str(output / "mahleran"),
        ]
        result = subprocess.run(
            cmd, cwd=output, capture_output=True, text=True, timeout=120, check=False
        )
        (output / "link.stdout").write_text(result.stdout)
        (output / "link.stderr").write_text(result.stderr)
        report["commands"].append({"command": cmd, "returncode": result.returncode})
        result.check_returncode()
        report["status"] = "built"
        report["executable_sha256"] = sha(output / "mahleran")
    finally:
        report["wall_s"] = time.perf_counter() - start
        report["sources_unchanged"] = inventory(source) == manifest["prepared_sha256"]
        (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "source_files": len(selected),
                "wall_s": report["wall_s"],
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unchecked", action="store_true")
    args = p.parse_args()
    build(args.source, args.output, checked=not args.unchecked)
