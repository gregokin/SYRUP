"""Original-MAHLERAN Fortran reference for the legacy sediment benchmark: pins, isolated sources, build, input/output, run guards.

Task gpu_sediment A2. Nothing here was executed by its author (file-only tools); Codex builds and runs.

What the reference IS. The ORIGINAL MAHLERAN 1.2.3 routines (water: `infilt`, `route_water`, `update_water_flow`, `ff_type8`;
sediment: `route_sediment_xml` with its callees, `flow_distrib`, `update_sediment_flow`) linked unchanged against the original
`shared_data` and `parameters_from_xml` modules, driven by `legacy_sediment_driver.f90`, which contains state loading, the call
order of MAHLERAN_storm_xml.f90 and read-only accounting, and no erosion/transport equation. Only TWO derived inputs exist:

* an ISOLATED COPY of `route_sediment_xml.f90` with (1) the dry-cell splash call pair replaced by zeroing of that cell's detachment
  and deposition rates (the same change as benchmarks/phase7/no_splash.patch; wet-cell raindrop detachment is untouched) and,
  for the HOOKED build only, (2) a read-only recorder immediately before the original clip: it stores the actual unclipped trial
  depth and the clip factor in `syrup_hook` arrays that no original routine reads. The unhooked build (patch 1 only) exists to
  prove by a hook-on/hook-off bitwise comparison that the recorder changes nothing;
* `syrup_derived_constants`, a subroutine GENERATED here from VERBATIM line ranges of the original
  `src/Subroutines_In_out/initialize_values_xml.f90` (density, hz, spa..hs and nutrient copies, dtdx, sigma, p_par, dstar_const, the
  nmax_* limits, spa/1200, diameter, settling_vel and the splash distributions). Each range is asserted by its first and last line
  and the whole file by its SHA-256; nothing is retyped.

Type fidelity. `shared_data.f90` declares most globals by implicit typing, i.e. default REAL (kind 4): dt, dx, dx_m, density, sigma,
dstar_const, radius, diameter, viscosity, settling_vel, ustar, d50 and the widened constants pi, fourth; re, ke, p_par and v_soil
are kind 8 (shared_kind_probe.json, compiled evidence). The harness uses the module unchanged and never passes `-fdefault-real-8`;
values stored into those variables are narrowed exactly as in the application. The SYRUP FP64 laws therefore differ from this
reference by REAL-precision effects, a precision CATEGORY (smooth-equation probes use rtol 2e-6, atol 1e-14), not by a bug.

The sediment routing selector is 2 (Crank-Nicolson), the matched SYRUP convention; the RFID XML has no selector and the native
diagnostic run used Euler (1): that run is NOT this reference. Chemistry and marker-in-cell are absent (the driver stubs
`route_markers_xml` with an error stop and sets MiC = 0). Water keeps the original stale-inflow/bracket behaviour; its budget is
REPORTED, never claimed conservative.

Binary protocol (little-endian). INPUT `SYRFSED1`, int32 header[14] = (version 1, endian 0x01020304, nr2, nc2, ncell1, nsteps,
nclass 6, iroute 2|5, sediment selector 2, inf_model 1, inf_type 1, KE model, n_capture, n_snapshot), float64 header[5] = (dt s,
dx mm, dx m, particle density g/cm3, active_layer_sensitivity mm), then tagged blocks (8-byte tag, int64 count, payload; arrays in
Fortran/column-major order, first index fastest) in `INPUT_BLOCKS` order, then `SYREOF01` with count 0 and nothing after.
OUTPUT files start with `SYRFOUT1` / `SYRFCAP1` / `SYRFSNP1`, hold tagged float64 (int for COUNTS/TERMINAL) blocks and end with the
same trailer. Completion is `result.txt` ending in SYRUP_LEGACY_SEDIMENT_COMPLETE (a legacy STOP returns status 0).
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "rfid"))
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
import fortran_reference as fr
import fortran_timing as ft

DRIVER = HERE / "legacy_sediment_driver.f90"
HOOK_MODULE = HERE / "syrup_hook.f90"
WALK_PROBE = HERE / "walk_probe.f90"
MAGIC_IN, MAGIC_OUT, MAGIC_CAP, MAGIC_SNAP = b"SYRFSED1", b"SYRFOUT1", b"SYRFCAP1", b"SYRFSNP1"
TRAILER_TAG = b"SYREOF01"
ENDIAN_MARKER = 0x01020304
MARKER = "SYRUP_LEGACY_SEDIMENT_COMPLETE"
N_CLASSES = 6
LEDGER_NAMES = ("pickup_kg", "deposition_active_kg", "deposition_pit_kg", "deposition_ring_kg", "deposition_inactive_kg",
                "effective_clip_source_kg", "old_mobile_kg", "new_mobile_kg", "mobile_terminal_kg", "cn_export_kg",
                "endpoint_export_kg", "outlet_flux_kg_s", "erased_deposition_kg_not_measured")
INT_TAGS = {"COUNTS  ": "<i8", "TERMINAL": "<i4"}

# SHA-256 of the ORIGINAL sources (from agent_handoffs/tasks/gpu_sediment/fortran_source_pins.json; copied here so the shipped
# benchmark does not depend on an ignored task path). A different hash is a refused build.
PINS = {
    "src/Program_Control/shared_data.f90": "0d05a3b38bad24b40833c530900dc1e061cd90cef187b4e6bce2e1e1d233d36f",
    "src/Program_Control/parameters_from_xml.f90": "cf2907218772ffb9225b5551cd5cd395451e88b2a9074eeb6aa14c9de75b6873",
    "src/Subroutines_In_out/initialize_values_xml.f90": "ab7f4ff9e04826b40b3dcdbe5911d1e2b574b7d1a218949ab10b60e6dfc19fa6",
    "src/Subroutines_Water/ff_type8.for": "0dc87ac00acc28a7b04b4e4a57e4251569a9dd127db06880f3df3fbe5b50b796",
    "src/Subroutines_Water/infilt.for": "f67ae7a741be20d80ce041a4510cf65abe66993409e2b6974b857f713f279b8e",
    "src/Subroutines_Water/route_water.for": "cd3c906ec35f181d108ed09fd3636be72578e759192a9f08a61b6f3395462188",
    "src/Subroutines_Water/update_water_flow.for": "fca3dd601611333865df996d7d73bd62da0592810726073ccd61945ae333ed92",
    "src/Subroutines_Sediment/route_sediment_xml.f90": "c4ff42a8a0f5acf86708199082ebec348483b0583bc804b052c2ba025e44a7a4",
    "src/Subroutines_Sediment/flow_distrib.for": "18a3eaf48ab6b1e81aacb970530d2e8fe214bde796696f21572110165ecef47f",
    "src/Subroutines_Sediment/flow_detachment.for": "e5c2e51f233a114914b32b5674aa89a6db78fc5a9d5570a018a8f7483ba19d50",
    "src/Subroutines_Sediment/raindrop_detachment.for": "e30435007b94875882a4f2718afc28d58a12673d904bb267ae7d834f7cf59746",
    "src/Subroutines_Sediment/diffuse_flow_transport.for": "896c4977d2829df7d40378bfde3d2668ad015213d298cfd179b46ba0318f9ae3",
    "src/Subroutines_Sediment/conc_flow_transport.for": "40756820a0049a8eaf09de2cf0b0429e9da37e9254be9b401ded4a12238362e4",
    "src/Subroutines_Sediment/suspended_transport.for": "37914dd2b8d98b9dfbb2933060d7a9912749619bb5a0070fa52f7326058e0ea0",
    "src/Subroutines_Sediment/update_sediment_flow.for": "58d6ba11b6fa93cbce2953aae7ff793583777685a824a3ec5e6f6a3452627408",
}
# compile order (modules first); route_sediment_xml and the derived-constants subroutine are generated into the build dir
ORIGINAL_ORDER = (
    "src/Program_Control/shared_data.f90", "src/Program_Control/parameters_from_xml.f90",
)
AFTER_HOOK_ORDER = (
    "src/Subroutines_Water/ff_type8.for", "src/Subroutines_Water/infilt.for", "src/Subroutines_Water/route_water.for",
    "src/Subroutines_Water/update_water_flow.for", "src/Subroutines_Sediment/flow_detachment.for",
    "src/Subroutines_Sediment/raindrop_detachment.for", "src/Subroutines_Sediment/diffuse_flow_transport.for",
    "src/Subroutines_Sediment/conc_flow_transport.for", "src/Subroutines_Sediment/suspended_transport.for",
    "src/Subroutines_Sediment/flow_distrib.for", "src/Subroutines_Sediment/update_sediment_flow.for",
)
ROUTE_SEDIMENT = "src/Subroutines_Sediment/route_sediment_xml.f90"
INITIALIZE = "src/Subroutines_In_out/initialize_values_xml.f90"
# (first line, last line, first-line text, last-line text) of the VERBATIM ranges taken from initialize_values_xml.f90
EXTRACT_RANGES = (
    (134, 137, "density = particle_density", "hz = active_layer_sensitivity"),
    (157, 173, "do i = 1, 6", "enddo"),
    (206, 207, "dtdx = dt / dx", "dt2dx = dt / (2. * dx)"),
    (209, 219, "!   sigma is relative density in kg/m^3",
     "dstar_const = (((sigma - 1.0d0) * 9.81d0) / (viscosity ** 2)) ** (1.d0 / 3.d0)"),
    (243, 324, "do phi = 1, 6", "enddo"),
)
BASE_FLAGS = ("-std=legacy", "-ffixed-line-length-none", "-ffree-line-length-none")
BUILD_FLAGS = {"timing": ("-O2",), "checked": ("-O0", "-fcheck=all", "-fbacktrace")}
INPUT_BLOCKS = ("ASPECT  ", "RMASK   ", "SLOPE   ", "FF      ", "KSAT    ", "PSI     ", "PAVE    ", "DRAINPAR", "THETASAT",
                "THETA   ", "CUMINF  ", "STMAX   ", "SCALE   ", "OUTLET  ", "ACTIVE  ", "VEG     ", "SEDPROPN", "ORDER   ",
                "RATES   ", "CAPSTEPS", "SNAPSTEP", "SEDPARAM")
# reference-equation probe tolerances (REAL kind-4 constants and globals in the original): predeclared, never fitted afterwards
SMOOTH_RTOL, SMOOTH_ATOL = 2.0e-6, 1.0e-14
WALK_RTOL, WALK_ATOL = 1.0e-12, 1.0e-14  # flow_distrib uses double expressions on exactly representable dx_m, dt
MAX_STEPS = 1_000_000


class FortranGlueError(RuntimeError):
    pass


class FortranGlueUnsupported(FortranGlueError):
    """The requested case needs a feature this glue does not reproduce (it is refused, never approximated)."""


def sha256_file(path) -> str:
    """Streamed SHA-256 (the Chastre input can be a gigabyte: never read it whole)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def toolchain():
    return fr.locate_toolchain()


def require_toolchain():
    """The configured toolchain, or None when NONE is configured (callers may skip only then). A configured toolchain that does
    not answer `--version` is a hard failure, not a skip."""
    tc = toolchain()
    if tc is None:
        return None
    try:
        done = subprocess.run([tc.compiler, "--version"], capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FortranGlueError(f"the configured Fortran compiler {tc.compiler} does not run: {exc}") from exc
    if done.returncode != 0:
        raise FortranGlueError(f"the configured Fortran compiler {tc.compiler} failed --version")
    return tc


# --- pinned originals, extraction, patches ----------------------------------------------------------------------------
def pinned_hashes(root: Path = fr.MAHLERAN_ROOT) -> dict[str, str]:
    return {rel: sha256_file(Path(root) / rel) for rel in PINS}


def check_pins(root: Path = fr.MAHLERAN_ROOT) -> dict[str, str]:
    hashes = pinned_hashes(root)
    bad = {k: v for k, v in hashes.items() if v != PINS[k]}
    if bad:
        raise FortranGlueError(f"original sources differ from the audited revision: {bad}")
    return hashes


def generate_constants_source(initialize_text: str) -> str:
    """The `syrup_derived_constants` subroutine from verbatim line ranges (asserted). Raises on any mismatch."""
    lines = initialize_text.splitlines()
    chunks = []
    for first, last, first_text, last_text in EXTRACT_RANGES:
        if len(lines) < last or lines[first - 1].strip() != first_text or lines[last - 1].strip() != last_text:
            raise FortranGlueError(f"initialize_values_xml.f90 lines {first}-{last} are not the expected original text")
        chunks.append("\n".join(lines[first - 1:last]))
    head = ("! GENERATED by benchmarks/legacy_sediment/sources.py from verbatim line ranges of\n"
            "! src/Subroutines_In_out/initialize_values_xml.f90 (ranges: " + ", ".join(f"{a}-{b}" for a, b, _, _ in EXTRACT_RANGES) + ").\n"
            "subroutine syrup_derived_constants\n"
            "use shared_data\n"
            "use parameters_from_xml, only: particle_density, active_layer_sensitivity, Raindrop_detachment_a_parameter_size, &\n"
            "   Raindrop_detachment_b_parameter_size, Raindrop_detachment_c_parameter_size, "
            "Raindrop_detachment_d_parameter_size, &\n"
            "   Raindrop_detachment_max_parameter_size, Particulate_bound_NH4, Particulate_bound_NO3, Particulate_bound_TN, &\n"
            "   Particulate_bound_TP, Particulate_bound_IC, Particulate_bound_TC\n"
            "implicit none\n"
            "integer :: i, k\n"
            "double precision :: spsum\n")
    return head + "\n!---- verbatim original lines ----\n".join(["", *chunks])[1:] + "\nend subroutine syrup_derived_constants\n"


_SPLASH = re.compile(r"^([ \t]*)call raindrop_detachment[ \t]*\r?\n[ \t]*call splash_transport[ \t]*\r?$", re.MULTILINE)
_USE = re.compile(r"^use parameters_from_xml[ \t]*$", re.MULTILINE)
_CLIP = re.compile(r"^([ \t]*)if \(d_soil \(phi, 2, i, j\)\.lt\.0\.d0\) then", re.MULTILINE)


def patch_route_sediment(text: str, *, hooked: bool) -> str:
    """The isolated copy of route_sediment_xml.f90. Anchors must match exactly once, otherwise the build is refused."""
    if len(_SPLASH.findall(text)) != 1:
        raise FortranGlueError("route_sediment_xml.f90: the dry-cell splash call pair was not found exactly once")
    indent_note = ("! SYRUP BENCHMARK ONLY: no dry-cell splash or its associated pickup (as benchmarks/phase7/no_splash.patch).\n"
                   "! The wet-cell raindrop detachment above is retained without modification; rates are cleared, the mobile\n"
                   "! inventory (d_soil/q_soil) never is.\n"
                   "            detach_soil (:, im, jm) = 0.0d0\n"
                   "            depos_soil (:, im, jm) = 0.0d0")
    out = _SPLASH.sub(lambda m: indent_note, text, count=1)
    if hooked:
        if len(_USE.findall(out)) != 1 or len(_CLIP.findall(out)) != 1:
            raise FortranGlueError("route_sediment_xml.f90: the hook anchors were not found exactly once")
        out = _USE.sub("use parameters_from_xml\nuse syrup_hook", out, count=1)
        out = _CLIP.sub(lambda m: (f"{m.group(1)}syrup_trial (phi, i, j) = d_soil (phi, 2, i, j)   ! SYRUP read-only recorder\n"
                                   f"{m.group(1)}syrup_clip_factor (phi, i, j) = 1.d0 + 0.5d0 * dt / dx * v_soil (phi, i, j)\n"
                                   f"{m.group(1)}syrup_hook_called = .true.\n"
                                   f"{m.group(0)}"), out, count=1)
    return out


# --- build ------------------------------------------------------------------------------------------------------------
def _compile(commands, build_dir: Path, logs: list) -> None:
    for command in commands:
        done = subprocess.run(command, cwd=build_dir, capture_output=True, text=True, timeout=1800, check=False)
        logs.append({"command": command, "returncode": done.returncode, "stderr": done.stderr[-4000:]})
        if done.returncode != 0:
            raise FortranGlueError(f"command failed ({done.returncode}): {shlex.join(command)}\n{done.stdout}\n{done.stderr}")


def build(build_dir, *, hooked: bool = True, variant: str = "timing", root: Path = fr.MAHLERAN_ROOT,
          extra_protected: dict | None = None) -> dict:
    """Compile the driver into the NEW `build_dir`. Refuses changed originals, an unsafe destination and any compiler failure
    (a configured-toolchain build failure is a FAILURE, never a skip). Returns (and saves) the build record."""
    if variant not in BUILD_FLAGS:
        raise ValueError(f"variant must be one of {tuple(BUILD_FLAGS)}")
    tc = require_toolchain()
    if tc is None:
        raise FortranGlueError("no Fortran toolchain is configured (MAPLE_SYRUP_GFORTRAN or gfortran on PATH)")
    root = Path(root)
    hashes = check_pins(root)
    listing = fr.watched_listing(root)
    build_dir = ft.safe_destination(build_dir, extra_protected)
    build_dir.mkdir(parents=True, exist_ok=False)
    src = build_dir / "generated"
    src.mkdir()
    original_route = (root / ROUTE_SEDIMENT).read_text(encoding="latin-1")
    patched = patch_route_sediment(original_route, hooked=hooked)
    (src / "route_sediment_xml.f90").write_text(patched, encoding="latin-1")
    # the exact change as a unified diff (evidence, not only hashes): it must contain only the splash pair and the read-only hook
    diff_text = "".join(difflib.unified_diff(original_route.splitlines(keepends=True), patched.splitlines(keepends=True),
                                             fromfile="original/" + ROUTE_SEDIMENT, tofile="isolated/route_sediment_xml.f90"))
    (build_dir / "route_sediment_xml.patch.diff").write_text(diff_text, encoding="latin-1")
    constants = generate_constants_source((root / INITIALIZE).read_text(encoding="latin-1"))
    (src / "syrup_derived_constants.f90").write_text(constants, encoding="latin-1")
    flags = [*tc.flags, *BASE_FLAGS, *BUILD_FLAGS[variant]]
    base = [tc.compiler, *flags, f"-J{build_dir}", f"-I{build_dir}"]
    ordered = [root / p for p in ORIGINAL_ORDER] + [HOOK_MODULE] + [root / p for p in AFTER_HOOK_ORDER] + [
        src / "route_sediment_xml.f90", src / "syrup_derived_constants.f90", DRIVER]
    objects, commands, logs = [], [], []
    for source in ordered:
        obj = build_dir / f"{len(objects):02d}_{source.stem}.o"
        commands.append([*base, "-c", str(source), "-o", str(obj)])
        objects.append(str(obj))
    name = "legacy_sediment_driver_" + ("hooked" if hooked else "nohook")
    exe = build_dir / name
    commands.append([tc.compiler, *flags, *tc.link_flags, *objects, "-o", str(exe)])
    start = time.perf_counter()
    _compile(commands, build_dir, logs)
    version = subprocess.run([tc.compiler, "--version"], capture_output=True, text=True, check=False).stdout.splitlines()
    record = {
        "variant": variant, "hooked": hooked, "executable": str(exe), "executable_sha256": sha256_file(exe),
        "compiler": tc.compiler, "compiler_version": version[0] if version else "", "flags": flags,
        "link_flags": list(tc.link_flags), "commands": logs, "original_sha256": hashes,
        "route_sediment_patched_sha256": sha256_file(src / "route_sediment_xml.f90"),
        "route_sediment_patch_diff_file": str(build_dir / "route_sediment_xml.patch.diff"),
        "route_sediment_patch_diff_sha256": sha256_file(build_dir / "route_sediment_xml.patch.diff"),
        "derived_constants_sha256": sha256_file(src / "syrup_derived_constants.f90"),
        "driver_sha256": sha256_file(DRIVER), "hook_module_sha256": sha256_file(HOOK_MODULE),
        "extract_ranges": [list(r[:2]) for r in EXTRACT_RANGES], "build_wall_s": time.perf_counter() - start,
        "type_note": "shared_data used unchanged (implicit default REAL globals, no -fdefault-real-8); see shared_kind_probe.json",
        "reference_tree_unchanged": pinned_hashes(root) == hashes and fr.watched_listing(root) == listing,
    }
    if not record["reference_tree_unchanged"]:
        raise FortranGlueError("the MAHLERAN reference tree changed during the build")
    (build_dir / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def build_walk_probe(build_dir, *, root: Path = fr.MAHLERAN_ROOT, extra_protected: dict | None = None) -> dict:
    """The original flow_distrib probe (shared_data + flow_distrib + `walk_probe.f90`)."""
    tc = require_toolchain()
    if tc is None:
        raise FortranGlueError("no Fortran toolchain is configured")
    hashes = check_pins(root)
    build_dir = ft.safe_destination(build_dir, extra_protected)
    build_dir.mkdir(parents=True, exist_ok=False)
    flags = [*tc.flags, *BASE_FLAGS, "-O0", "-fcheck=all", "-fbacktrace"]
    base = [tc.compiler, *flags, f"-J{build_dir}", f"-I{build_dir}"]
    sources = [Path(root) / "src/Program_Control/shared_data.f90", Path(root) / "src/Subroutines_Sediment/flow_distrib.for", WALK_PROBE]
    objects, commands, logs = [], [], []
    for source in sources:
        obj = build_dir / f"{len(objects):02d}_{source.stem}.o"
        commands.append([*base, "-c", str(source), "-o", str(obj)])
        objects.append(str(obj))
    exe = build_dir / "walk_probe"
    commands.append([tc.compiler, *flags, *tc.link_flags, *objects, "-o", str(exe)])
    _compile(commands, build_dir, logs)
    record = {"executable": str(exe), "executable_sha256": sha256_file(exe), "compiler": tc.compiler, "flags": flags,
              "original_sha256": hashes, "probe_sha256": sha256_file(WALK_PROBE), "commands": logs}
    (build_dir / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run_walk_probe(executable, workdir, *, aspect_full: np.ndarray, dx_m: float, dt: float, calls: list[tuple]) -> np.ndarray:
    """Run the original flow_distrib on `aspect_full` (north-first with ring, ints 0..4). `calls` = (im, jm, phi, detach, travel_dist,
    nsteps) with 1-based Fortran indices. Returns depos `(nr2, nc2, 6)`. Needs exactly representable `dx_m` and `dt`."""
    workdir = ft.safe_destination(workdir)
    workdir.mkdir(parents=True)
    nr2, nc2 = aspect_full.shape
    lines = [f"{nr2} {nc2} {dx_m!r} {dt!r}"] + [" ".join(str(int(v)) for v in row) for row in aspect_full] + [str(len(calls))]
    lines += [f"{im} {jm} {phi} {det!r} {td!r} {ns}" for im, jm, phi, det, td, ns in calls]
    (workdir / "input.txt").write_text("\n".join(lines) + "\n")
    tc = toolchain()
    env = os.environ.copy()
    if tc is not None and tc.run_library_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*tc.run_library_path, env.get("LD_LIBRARY_PATH", "")])
    done = subprocess.run([str(executable), str(workdir / "input.txt"), str(workdir / "out.txt")], cwd=workdir,
                          capture_output=True, text=True, timeout=300, check=False, env=env)
    out = workdir / "out.txt"
    if done.returncode != 0 or not out.is_file() or not out.read_text().rstrip().endswith("SYRUP_WALK_PROBE_COMPLETE"):
        raise FortranGlueError(f"walk probe failed ({done.returncode}): {done.stderr[-2000:]}")
    values = np.array([float(t) for ln in out.read_text().splitlines()[:-1] for t in ln.split()])
    if values.size != nr2 * nc2 * N_CLASSES or not np.all(np.isfinite(values)):
        raise FortranGlueError("walk probe output has the wrong size or non-finite values")
    return values.reshape(nc2, nr2, N_CLASSES).transpose(1, 0, 2)


# --- inputs -----------------------------------------------------------------------------------------------------------
def _fbytes(a, dtype) -> bytes:
    return np.asarray(a).astype(dtype, copy=False).tobytes(order="F")


def _block(tag: str, payload: bytes, count: int) -> bytes:
    assert len(tag) == 8
    return tag.encode("ascii") + struct.pack("<q", count) + payload


def full_north_first(interior_south_first: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """`(ny, nx)` MAPLE south-first interior -> `(ny + 2, nx + 2)` legacy north-first with a ring (the same layout as
    `fortran_timing._full`)."""
    return ft._full(np.asarray(interior_south_first, dtype=np.float64), fill)


def xml_sediment_block(xml_path, expected_sha256: str) -> dict[str, Any]:
    """Wet-law parameters read from the actual XML (hash-bound): the 30 doubles of SEDPARAM and the scalars."""
    from maple_syrup.case_import import _xml_record, _xml_value, parse_mahleran_xml

    xml_path = Path(xml_path)
    if sha256_file(xml_path) != expected_sha256:
        raise FortranGlueError(f"{xml_path} does not hash to the bound {expected_sha256}")
    xml = parse_mahleran_xml(xml_path)
    classes = [f"phi_{k}" for k in range(1, 7)]

    def per_class(tag):
        children = _xml_record(xml, tag)["children"]
        if any(c not in children for c in classes):
            raise FortranGlueError(f"<{tag}> lacks a class")
        return [float(children[c]) for c in classes]

    sed = (per_class("Raindrop_detachment_a_parameter_size") + per_class("Raindrop_detachment_b_parameter_size")
           + per_class("Raindrop_detachment_c_parameter_size") + per_class("Raindrop_detachment_d_parameter_size")
           + per_class("Raindrop_detachment_max_parameter_size"))
    selector = _xml_record(xml, "sediment-routing_solution_method", required=False)
    return {"sedparam": sed, "particle_density_g_cm3": float(_xml_value(xml, "particle_density")),
            "active_layer_sensitivity_mm": float(_xml_value(xml, "active_layer_sensitivity")),
            "ke_model": int(_xml_value(xml, "KE_model_type")), "time_step_s": float(_xml_value(xml, "time_step")),
            "selector_in_xml": None if selector is None else selector["value"], "xml_sha256": expected_sha256}


def hydrology_arrays(graph, column, rainfall_scale, soil_water_m, theta0: float) -> dict:
    """The legacy water arrays through the SAME function the RFID water driver uses (`fortran_timing.common_arrays`). Only the
    attributes it reads are provided: graph, column parameters, rainfall scale, initial theta and soil water."""
    if getattr(column, "model", None) != "fixed_ksat":
        raise FortranGlueUnsupported("only the fixed_ksat infiltration model (original infilt model 1) is reproduced by this glue")
    shim = SimpleNamespace(graph=graph, params=column, soil0=np.asarray(soil_water_m, dtype=np.float64),
                           host={"rainfall_scale": np.asarray(rainfall_scale, dtype=np.float64),
                                 "initial_theta": np.full(tuple(graph.shape), float(theta0))})
    return ft.common_arrays(shim)


def write_input(path, arrays: dict, *, rates_mm_s, sediment_fractions, vegetation_percent, xml: dict, iroute: int = 5,
                dt_s: float = 1.0, capture_steps=(), snapshot_steps=(), progress=None, stats: dict | None = None) -> str:
    """Write the NEW binary input as a stream; returns its SHA-256 (of the exact bytes written). Validates everything before
    writing (finite, non-negative, REAL-exact dt and dx, bounded steps, capture/snapshot steps inside the window). `progress(msg)`
    is called at a bounded cadence; `stats` receives bytes written and the largest block copy."""
    rates = np.asarray(rates_mm_s, dtype=np.float64)
    if rates.ndim != 1 or not 1 <= rates.size <= MAX_STEPS or not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise FortranGlueError("rates must be a bounded finite non-negative 1-D array")
    ft.validate_run_request(int(rates.size), dt_s, 1, iroute)
    nr2, nc2 = arrays["aspect"].shape
    dx_mm, dx_m = float(arrays["dx_mm"]), float(arrays["dx_mm"]) / 1000.0
    if float(np.float32(dx_mm)) != dx_mm:
        raise FortranGlueError("dx (mm) must be exactly representable in default REAL")
    frac = np.asarray(sediment_fractions, dtype=np.float64)
    veg = np.asarray(vegetation_percent, dtype=np.float64)
    if frac.shape != (N_CLASSES, nr2, nc2) or veg.shape != (nr2, nc2):
        raise FortranGlueError(f"fractions {frac.shape} / vegetation {veg.shape} do not match the {nr2} x {nc2} grid")
    if not (np.all(np.isfinite(frac)) and np.all(frac >= 0.0) and np.all(np.isfinite(veg))):
        raise FortranGlueError("fractions and vegetation must be finite (fractions >= 0)")
    caps, snaps = sorted({int(s) for s in capture_steps}), sorted({int(s) for s in snapshot_steps})
    if len(caps) > 8 or len(snaps) > 16 or any(not 1 <= s <= rates.size for s in (*caps, *snaps)):
        raise FortranGlueError("at most 8 capture and 16 snapshot steps, each within 1..n_steps")
    sed = np.asarray(xml["sedparam"], dtype=np.float64)
    if sed.size != 30 or not np.all(np.isfinite(sed)):
        raise FortranGlueError("SEDPARAM must be 30 finite doubles")
    for name in ("rmask", "slope", "ff", "ksat", "psi", "pave", "drain_par", "theta_sat", "theta", "cum_inf", "stmax", "scale"):
        a = np.asarray(arrays[name])
        if a.shape != (nr2, nc2) or not np.all(np.isfinite(a)):
            raise FortranGlueError(f"array {name} has the wrong shape or non-finite values")
    order = np.asarray(arrays["order"])
    header_i = struct.pack("<14i", 1, ENDIAN_MARKER, nr2, nc2, order.shape[0], rates.size, N_CLASSES, iroute, 2, 1, 1,
                           int(xml["ke_model"]), len(caps), len(snaps))
    header_d = struct.pack("<5d", float(dt_s), dx_mm, dx_m, float(xml["particle_density_g_cm3"]),
                           float(xml["active_layer_sensitivity_mm"]))
    f8, i4 = "<f8", "<i4"
    # (tag, array, dtype): each payload is converted, written and hashed ONE AT A TIME (peak extra memory = one block copy)
    blocks = [
        ("ASPECT  ", arrays["aspect"], i4), ("RMASK   ", arrays["rmask"], f8), ("SLOPE   ", arrays["slope"], f8),
        ("FF      ", arrays["ff"], f8), ("KSAT    ", arrays["ksat"], f8), ("PSI     ", arrays["psi"], f8),
        ("PAVE    ", arrays["pave"], f8), ("DRAINPAR", arrays["drain_par"], f8), ("THETASAT", arrays["theta_sat"], f8),
        ("THETA   ", arrays["theta"], f8), ("CUMINF  ", arrays["cum_inf"], f8), ("STMAX   ", arrays["stmax"], f8),
        ("SCALE   ", arrays["scale"], f8), ("OUTLET  ", arrays["outlet"], i4), ("ACTIVE  ", arrays["active"], i4),
        ("VEG     ", veg, f8), ("SEDPROPN", frac, f8), ("ORDER   ", order, i4), ("RATES   ", rates, f8),
        ("CAPSTEPS", np.array(caps, dtype=np.int32), i4), ("SNAPSTEP", np.array(snaps, dtype=np.int32), i4),
        ("SEDPARAM", sed, f8),
    ]
    assert tuple(b[0] for b in blocks) == INPUT_BLOCKS
    digest = hashlib.sha256()
    written, largest = 0, 0

    def emit(handle, data: bytes) -> None:
        nonlocal written
        handle.write(data)
        digest.update(data)
        written += len(data)

    log = progress or (lambda message: None)
    with Path(path).open("xb") as handle:
        emit(handle, MAGIC_IN + header_i + header_d)
        for index, (tag, array, dtype) in enumerate(blocks):
            payload = _fbytes(array, dtype)
            largest = max(largest, len(payload))
            emit(handle, _block(tag, payload, int(np.asarray(array).size)))
            del payload
            if index % 6 == 5 or index == len(blocks) - 1:
                log(f"input block {index + 1}/{len(blocks)} ({tag.strip()}) written, {written / 2**20:.1f} MiB")
        emit(handle, _block("SYREOF01", b"", 0))
    if stats is not None:
        stats.update(bytes_written=written, largest_block_bytes=largest)
    return digest.hexdigest()


def inputs_from_legacy_case(case, *, end_s: float | None = None, iroute: int = 5, capture_steps=(), snapshot_steps=()):
    """`(arrays, kwargs)` for `write_input` from an A1 `LegacyCase` (attributes used: graph, column, initial_storm,
    rainfall_scale, schedule, vegetation, holdings_kg, record, kind). The LegacyCase is only read. Plot 1 (infiltration model 2,
    pavement) is refused here: its sediment reference is the existing whole-application ledger or state injection."""
    if getattr(case, "kind", None) not in ("rfid", "chastre"):
        raise FortranGlueUnsupported("the original-routine glue supports RFID and Chastre (fixed_ksat); Plot 1 uses the "
                                     "existing whole-application reference ledger (compare_legacy_sediment.plot1_golden)")
    hyd = case.record.get("hydrology_parameters")
    if not hyd or "theta0" not in hyd:
        raise FortranGlueError("the case record carries no initial theta0")
    if np.any(np.asarray(case.initial_storm.depth_m) != 0.0):
        raise FortranGlueUnsupported("the glue starts from zero surface water")
    arrays = hydrology_arrays(case.graph, case.column, case.rainfall_scale, case.initial_storm.soil_water_m, hyd["theta0"])
    xml = xml_sediment_block(case.record["xml"]["xml_path"], case.record["xml"]["xml_sha256"])
    if xml["time_step_s"] != 1.0:
        raise FortranGlueUnsupported("the original fixed step is 1 s")
    end = float(case.end_s if end_s is None else end_s)
    if not (math.isfinite(end) and end >= 1.0 and float(end).is_integer()):
        raise FortranGlueError("the window must be a whole number of seconds")
    n = int(end)
    rates = [case.schedule.rate_after_m_per_s(float(k)) * 1.0e3 for k in range(n)]
    hold = np.asarray(case.holdings_kg, dtype=np.float64)
    total = hold.sum(axis=-1)
    fractions = np.where((total > 0.0)[..., None], hold / np.where(total > 0.0, total, 1.0)[..., None], 0.0)
    frac_full = np.stack([full_north_first(fractions[..., k], 0.0) for k in range(N_CLASSES)])
    veg_full = full_north_first(np.asarray(case.vegetation, dtype=np.float64) * 100.0, 0.0)
    kwargs = {"rates_mm_s": rates, "sediment_fractions": frac_full, "vegetation_percent": veg_full, "xml": xml, "iroute": iroute,
              "capture_steps": capture_steps, "snapshot_steps": snapshot_steps}
    return arrays, kwargs


# --- outputs ----------------------------------------------------------------------------------------------------------
def read_tagged(path, magic: bytes, schema=None) -> dict[str, np.ndarray]:
    """Parse a tagged output file strictly: magic, whole blocks, trailer with count 0 and NOTHING after it. DUPLICATE tags are
    refused always. With a `schema` (see `ledger_schema` etc.: ordered `(tag, count, dtype, role)`), the tags must be exactly the
    schema's, in order, with exactly the declared counts, and every block must satisfy its role (`nonneg`: finite and >= 0;
    `finite`: finite, negatives allowed; `flag`: 0 or 1). Arrays are read-only views of the file's bytes (no per-block copy)."""
    data = Path(path).read_bytes()
    if data[:8] != magic:
        raise FortranGlueError(f"{path}: bad magic {data[:8]!r}")
    offset, out = 8, {}
    while True:
        if offset + 16 > len(data):
            raise FortranGlueError(f"{path}: truncated before the trailer")
        tag, count = data[offset:offset + 8], struct.unpack_from("<q", data, offset + 8)[0]
        offset += 16
        if tag == TRAILER_TAG:
            if count != 0 or offset != len(data):
                raise FortranGlueError(f"{path}: bad trailer or trailing bytes")
            break
        name = tag.decode("ascii")
        if name in out:
            raise FortranGlueError(f"{path}: duplicate block {name.strip()}")
        dtype = INT_TAGS.get(name, "<f8")
        size = np.dtype(dtype).itemsize * count
        if count < 0 or offset + size > len(data):
            raise FortranGlueError(f"{path}: block {name.strip()} is truncated")
        out[name] = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        offset += size
    if schema is not None:
        expected_tags = [s[0] for s in schema]
        if list(out) != expected_tags:
            raise FortranGlueError(f"{path}: blocks {[t.strip() for t in out]} differ from the schema {[t.strip() for t in expected_tags]}")
        for tag, count, dtype, role in schema:
            a = out[tag]
            if a.size != count or a.dtype != np.dtype(dtype):
                raise FortranGlueError(f"{path}: block {tag.strip()} has {a.size} x {a.dtype}, expected {count} x {dtype}")
            if role == "flag":
                if not np.all((a == 0) | (a == 1)):
                    raise FortranGlueError(f"{path}: block {tag.strip()} is not a 0/1 flag array")
            elif a.dtype.kind == "f":
                if not np.all(np.isfinite(a)) or (role == "nonneg" and np.any(a < 0.0)):
                    raise FortranGlueError(f"{path}: block {tag.strip()} is non-finite" + (" or negative" if role == "nonneg" else ""))
            elif role == "nonneg" and np.any(a < 0):
                raise FortranGlueError(f"{path}: block {tag.strip()} is negative")
    return out


def ledger_schema(n_steps: int) -> list:
    return [("LEDGER  ", 13 * N_CLASSES * n_steps, "<f8", "nonneg"), ("WATER   ", 4 * n_steps, "<f8", "nonneg"),
            ("COUNTS  ", 7 * n_steps, "<i8", "nonneg")]


def maps_schema(nr2: int, nc2: int) -> list:
    g, g6 = nr2 * nc2, N_CLASSES * nr2 * nc2
    return [("CUMDET  ", g6, "<f8", "nonneg"), ("CUMDEP  ", g6, "<f8", "nonneg"), ("CUMCLIP ", g6, "<f8", "nonneg"),
            ("MOBILE  ", g6, "<f8", "nonneg"), ("DEPTH_MM", g, "<f8", "nonneg"), ("VELOC_MM", g, "<f8", "nonneg"),
            ("SOILW_MM", g, "<f8", "nonneg"), ("DISCH_MM", g, "<f8", "nonneg"), ("TERMINAL", g, "<i4", "flag")]


def capture_schema(nr2: int, nc2: int) -> list:
    """The ORIGINAL negative trial depth (`POST_TRL`) is the only field allowed to be negative."""
    g, g6 = nr2 * nc2, N_CLASSES * nr2 * nc2
    fields = [("PRE_D1  ", g), ("PRE_V   ", g), ("PRE_R2  ", g), ("PRE_DOLD", g), ("PRE_VSED", g6), ("PRE_DS1 ", g6),
              ("PRE_QS1 ", g6), ("PRE_QIN1", g6), ("POST_DET", g6), ("POST_DEP", g6), ("POST_VS ", g6), ("POST_DS2", g6),
              ("POST_QS2", g6), ("POST_QI2", g6), ("POST_TRL", g6), ("POST_FAC", g6)]
    return [(tag, count, "<f8", "finite" if tag == "POST_TRL" else "nonneg") for tag, count in fields]


def snapshot_schema(nr2: int, nc2: int) -> list:
    return [("DEPTH_MM", nr2 * nc2, "<f8", "nonneg"), ("MOBILE  ", N_CLASSES * nr2 * nc2, "<f8", "nonneg")]


def read_result(path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FortranGlueError("result.txt is missing: no completion marker (a legacy STOP returns status 0)")
    text = path.read_text().splitlines()
    if not text or text[-1].strip() != MARKER:
        raise FortranGlueError("result.txt lacks the completion marker (a legacy STOP returns status 0)")
    out = {}
    for line in text[:-1]:
        key, value = line.split()
        out[key] = value
    return out


def _f(result, key) -> float:
    return float(result[key].replace("D", "E"))


def run_once(executable, input_path, workdir, *, expected: dict, build_record: dict | None = None, timeout_s: float = 6 * 3600.0,
             extra_protected: dict | None = None) -> dict:
    """One fresh process in the NEW `workdir`. `expected` = {n_steps, iroute, nr2, nc2, hooked, capture_steps, snapshot_steps,
    active_cells}. Qualifies ONLY if: exit 0; the completion marker; steps/iroute/grid/active/hook flag as requested; the binary
    files parse strictly with the expected sizes and finite values (cumulative maps >= 0, mobile >= 0); every requested capture
    and snapshot file exists and no other exists; executable, input and original-source hashes unchanged across the run."""
    executable, input_path = Path(executable).resolve(), Path(input_path).resolve()
    check_pins()  # the originals must equal the audited pins whether or not a build record is supplied
    guard_before = {"executable": sha256_file(executable), "input": sha256_file(input_path), "originals": pinned_hashes()}
    if build_record is not None and (guard_before["executable"] != build_record["executable_sha256"]
                                     or guard_before["originals"] != build_record["original_sha256"]):
        raise FortranGlueError("the executable or the original sources differ from the build record; refusing to run")
    protected = {"executable directory": executable.parent, "input directory": input_path.parent, **(extra_protected or {})}
    workdir = ft.safe_destination(workdir, protected)
    workdir.mkdir(parents=True, exist_ok=False)
    tc = toolchain()
    env = os.environ.copy()
    if tc is not None and tc.run_library_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*tc.run_library_path, env.get("LD_LIBRARY_PATH", "")])
    start = time.perf_counter()
    with (workdir / "stdout.log").open("wb") as so, (workdir / "stderr.log").open("wb") as se:
        proc = subprocess.Popen([str(executable), str(input_path), str(workdir)], cwd=workdir, stdout=so, stderr=se, env=env)
        try:
            _, status, usage = ft._wait_with_timeout(proc, timeout_s)
        except TimeoutError:
            return {"status": "timeout"}
    wall = time.perf_counter() - start
    code = os.waitstatus_to_exitcode(status)
    record: dict[str, Any] = {"returncode": code, "process_wall_s": wall, "max_rss_kib": int(usage.ru_maxrss),
                              "stdout_tail": (workdir / "stdout.log").read_text(errors="replace")[-2000:],
                              "stderr_tail": (workdir / "stderr.log").read_text(errors="replace")[-2000:]}
    guard_after = {"executable": sha256_file(executable), "input": sha256_file(input_path), "originals": pinned_hashes()}
    if guard_before != guard_after:
        return {**record, "status": "failed", "reason": "executable, input or original sources changed during the run"}
    if code != 0:
        return {**record, "status": "failed", "reason": f"nonzero exit status {code}"}
    try:
        outputs = load_run_outputs(workdir, expected)
    except (FortranGlueError, KeyError, ValueError, OSError) as exc:  # the partial output stays in `workdir` as evidence
        return {**record, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    result = outputs["result"]
    record.update(
        status="complete", **outputs, loop_s=_f(result, "LOOP_SECONDS"), diag_s=_f(result, "DIAG_SECONDS"),
        capture_s=_f(result, "CAPTURE_SECONDS"), kernel_s=_f(result, "KERNEL_SECONDS"),
        output_sha256={p: sha256_file(workdir / p) for p in ("ledger.bin", "final_maps.bin", "result.txt")})
    return record


def load_run_outputs(workdir, expected: dict) -> dict[str, Any]:
    """Strictly load and validate a finished run directory (used by `run_once` and by later reloads, so both apply identical
    checks): completion marker, request echo, exact file set, exact per-file schemas (tags, order, counts, dtypes, finite and
    sign roles; no duplicates or unknown blocks) for the ledger, the final maps and EVERY requested capture and snapshot."""
    workdir = Path(workdir)
    result = read_result(workdir / "result.txt")
    n, nr2, nc2 = expected["n_steps"], expected["nr2"], expected["nc2"]
    if int(result["STEPS"]) != n or int(result["IROUTE"]) != expected["iroute"] or int(result["NR2"]) != nr2 \
            or int(result["NC2"]) != nc2 or int(result["ACTIVE_CELLS"]) != expected["active_cells"]:
        raise FortranGlueError("result.txt disagrees with the request")
    if (result["HOOK_CALLED"].strip() == "T") != bool(expected["hooked"]):
        raise FortranGlueError("the pre-clip hook flag does not match the build")
    for key in ("LOOP_SECONDS", "DIAG_SECONDS", "CAPTURE_SECONDS", "KERNEL_SECONDS", "RAIN_M3", "EXPORT_M3", "SURFACE_M3",
                "SOIL_M3", "DRAIN_M3", "AF_KG_PER_MM", "DX_MM", "DT_S", "DENSITY_G_CM3"):
        if key not in result or not math.isfinite(_f(result, key)):
            raise FortranGlueError(f"result.txt lacks a finite {key}")
    files = sorted(p.name for p in workdir.iterdir())
    want = sorted({"ledger.bin", "final_maps.bin", "result.txt", "stdout.log", "stderr.log",
                   *(f"capture_{s:06d}.bin" for s in expected["capture_steps"]),
                   *(f"snapshot_{s:06d}.bin" for s in expected["snapshot_steps"])})
    if files != want:
        raise FortranGlueError(f"output files {files} differ from the expected {want}")
    led = read_tagged(workdir / "ledger.bin", MAGIC_OUT, ledger_schema(n))
    maps = read_tagged(workdir / "final_maps.bin", MAGIC_OUT, maps_schema(nr2, nc2))
    captures = {s: read_tagged(workdir / f"capture_{s:06d}.bin", MAGIC_CAP, capture_schema(nr2, nc2)) for s in expected["capture_steps"]}
    snapshots = {s: read_tagged(workdir / f"snapshot_{s:06d}.bin", MAGIC_SNAP, snapshot_schema(nr2, nc2))
                 for s in expected["snapshot_steps"]}
    return {"result": result, "ledger": led["LEDGER  "].reshape((13, N_CLASSES, n), order="F").transpose(2, 0, 1),
            "water_steps": led["WATER   "].reshape((4, n), order="F").T, "counts": led["COUNTS  "].reshape((7, n), order="F").T,
            "maps": maps, "captures": captures, "snapshots": snapshots}


def new_output_root(root, protected: dict | None = None) -> Path:
    """Validate a requested output ROOT before anything nested is created: it may already exist (and be reused) or be created,
    but must not equal, lie inside or contain any protected tree (reference, project trees, case, MAPLE, build, prepared, input
    and executable directories passed in `protected`). Returns the resolved path; creates nothing."""
    resolved = Path(os.path.realpath(root))
    for label, tree in {**ft.PROTECTED, **(protected or {})}.items():
        r = Path(os.path.realpath(tree))
        if resolved == r or resolved.is_relative_to(r) or r.is_relative_to(resolved):
            raise FortranGlueError(f"refusing output root {resolved}: it is, lies inside or contains the {label} tree {r}")
    return resolved


def exclusive_text(path, text: str, protected: dict | None = None) -> Path:
    """Write a report/summary file that must be NEW (exclusive create), outside the protected trees. No overwrite."""
    resolved = new_output_root(Path(path).resolve().parent, protected) / Path(path).name
    if os.path.lexists(resolved):
        raise FortranGlueError(f"refusing to overwrite existing {resolved}")
    with resolved.open("x") as handle:
        handle.write(text)
    return resolved


def fortran_grid_to_syrup(a: np.ndarray, nr2: int, nc2: int) -> np.ndarray:
    """A flat Fortran-order block of `(nr2, nc2)` or `(6, nr2, nc2)` -> SYRUP `(ny, nx)` / `(ny, nx, 6)` (interior only, row 0 =
    south)."""
    a = np.asarray(a)
    if a.size == nr2 * nc2:
        return a.reshape((nr2, nc2), order="F")[1:-1, 1:-1][::-1, :].copy()
    if a.size == N_CLASSES * nr2 * nc2:
        cube = a.reshape((N_CLASSES, nr2, nc2), order="F")[:, 1:-1, 1:-1][:, ::-1, :]
        return np.moveaxis(cube, 0, -1).copy()
    raise FortranGlueError("unexpected block size")


def fortran_grid_full(a: np.ndarray, nr2: int, nc2: int) -> np.ndarray:
    """Whole `(6, nr2, nc2)` / `(nr2, nc2)` Fortran block with the ring (legacy north-first orientation)."""
    a = np.asarray(a)
    shape = (nr2, nc2) if a.size == nr2 * nc2 else (N_CLASSES, nr2, nc2)
    return a.reshape(shape, order="F").copy()
