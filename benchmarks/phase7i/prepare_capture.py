"""Prepare, but do not build or run, an ISOLATED derivative of the heterogeneous whole-MAHLERAN Plot1 source with
read-only diagnostic hooks that capture the sampled conductivity realization and the applied forcing at full precision.

    python benchmarks/phase7i/prepare_capture.py \\
        --parent outputs/phase7/mahleran_plot1_no_splash_linux --output outputs/phase7i/mahleran_capture_source

The parent is the Linux derivative that produced `outputs/phase7/mahleran_fixed_no_splash` (the no-splash patch and
the allocated-array diagnostic guard, original XML with `finalInfiltrationRateDistribution = normal`). The ONLY file
changed is `src/Program_Control/MAHLERAN_storm_xml.f90`: four one-line `call syrup_hydro_capture (n)` insertions in the
storm driver and one appended subroutine group that READS shared state and writes new files in the output folder. No
equation, random draw, forcing, time step or state update is touched (`remove_hooks` reproduces the parent file
byte for byte, which the tests and `prepare` both check). Build and run reuse benchmarks/phase7/build_mahleran.py and
run_mahleran.py unchanged; the manifest keeps the parent's `original_sha256`/`reference_root` so the run tool still
binds the untouched reference tree.

Captured (all FULL PRECISION text, `ES25.16E3`; Fortran `(i, j)` arrays, see capture_data.py):
  syrup_hydro_static.txt  after setup, before step 1: ksat, psi, theta_sat, theta, cum_inf, stmax, drain_par, pave, slope,
                          ff, rmask, aspect, order, initial d/q and the scalar configuration (dt, dx, iroute, ...)
  syrup_hydro_steps.txt   one row per iteration: the rainfall rate APPLIED in that step (read before infilt), outlet
                          discharge two ways (the model's single-precision q_plot and a double-precision sum over the
                          same outlet cells), the conservative Crank-Nicolson face export, and storage/drainage sums
  syrup_hydro_final.txt   synchronous depth/velocity at the model's own strict-greater peak (single-precision logic of
                          output_hydro_data_xml.f90 489-505, replicated) AND at the double-precision peak, the model's own
                          dmax/vmax for cross-checking, and the final state
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase7"))
from prepare_mahleran import inventory, sha

STORM = Path("src/Program_Control/MAHLERAN_storm_xml.f90")
# `mahleran_input.xml` of the heterogeneous no-splash run (execution.json input_sha256): normal conductivity draw.
EXPECTED_XML_SHA256 = "863052542397774ece9bf0c666466bbd2d178b7590f9d8d7fab03214e3495c4c"
HOOK_MARKER = "! ===== SYRUP hydrology-qualification capture (phase7i): read-only diagnostic ====="
HOOK_CALL = "call syrup_hydro_capture"

INSERTIONS = (
    # the actual parent line has NO leading blanks; the leading newline anchors it as a whole line
    ("\nwrite (6, *) ' About to start Mahleran Storm, iout = ', iout\n", 0),
    ("   write (6, 9999) iter, rval * 3600., dt, t, Julian, istart, q_plot * dx, sed_plot\n", 1),
    ("   call output_hydro_data_xml\n", 2),
    ("enddo\nclose (51)\n\n!EVA2016", 3),
)

HOOK_SOURCE = f'''

{HOOK_MARKER}
! Reads shared_data; writes only NEW files syrup_hydro_*.txt in the output folder (status = 'new').
! stage 0: static setup state    stage 1: before infilt (applied rain)
! stage 2: after output_hydro_data_xml, before update_water_flow    stage 3: after the last step.
subroutine syrup_hydro_capture (a_stage)
use shared_data
use parameters_from_xml, only: output_folder, output_folder_length
implicit none
integer :: a_stage
integer :: a_i, a_j, a_n1, a_n2, a_u, a_nr1, a_nc1
integer, save :: a_u_steps = -1
integer, save :: a_iter_s = 0, a_iter_d = 0, a_nrec = 0
real, save :: a_qmax_s
double precision, save :: a_qmax_d, a_rval, a_sum_r2, a_max_r2
double precision, allocatable, save :: a_d2_s (:,:), a_v_s (:,:), a_d2_d (:,:), a_v_d (:,:), a_work (:,:)
double precision :: a_qd, a_q1d, a_sum_d2, a_sum_inf, a_sum_drn, a_sum_exc, a_max_d, a_max_v
double precision :: a_cn, a_dd, a_dv
logical :: a_out, a_act
select case (a_stage)
case (0)
   a_n1 = size (ksat, 1)
   a_n2 = size (ksat, 2)
   if (a_n1 /= nr2 .or. a_n2 /= nc2) error stop 'syrup capture: unexpected array shape'
   ! every array the hook reads or passes must have the same (nr2, nc2) extent (dmax/vmax included)
   if (size (psi, 1) /= a_n1 .or. size (psi, 2) /= a_n2 .or. size (theta_sat, 1) /= a_n1 .or. &
       size (theta_sat, 2) /= a_n2 .or. size (theta, 1) /= a_n1 .or. size (theta, 2) /= a_n2 .or. &
       size (cum_inf, 1) /= a_n1 .or. size (cum_inf, 2) /= a_n2 .or. size (cum_drain, 1) /= a_n1 .or. &
       size (cum_drain, 2) /= a_n2 .or. size (stmax, 1) /= a_n1 .or. size (stmax, 2) /= a_n2 .or. &
       size (drain_par, 1) /= a_n1 .or. size (drain_par, 2) /= a_n2 .or. size (pave, 1) /= a_n1 .or. &
       size (pave, 2) /= a_n2 .or. size (slope, 1) /= a_n1 .or. size (slope, 2) /= a_n2 .or. &
       size (ff, 1) /= a_n1 .or. size (ff, 2) /= a_n2 .or. size (rmask, 1) /= a_n1 .or. &
       size (rmask, 2) /= a_n2 .or. size (aspect, 1) /= a_n1 .or. size (aspect, 2) /= a_n2 .or. &
       size (v, 1) /= a_n1 .or. size (v, 2) /= a_n2 .or. size (excess, 1) /= a_n1 .or. &
       size (excess, 2) /= a_n2 .or. size (r2, 1) /= a_n1 .or. size (r2, 2) /= a_n2 .or. &
       size (dmax, 1) /= a_n1 .or. size (dmax, 2) /= a_n2 .or. size (vmax, 1) /= a_n1 .or. &
       size (vmax, 2) /= a_n2 .or. size (d, 2) /= a_n1 .or. size (d, 3) /= a_n2 .or. &
       size (q, 2) /= a_n1 .or. size (q, 3) /= a_n2 .or. size (order, 2) /= 3) &
      error stop 'syrup capture: model arrays do not share the (nr2, nc2) extent'
   allocate (a_d2_s (a_n1, a_n2), a_v_s (a_n1, a_n2), a_d2_d (a_n1, a_n2), a_v_d (a_n1, a_n2), a_work (a_n1, a_n2))
   a_d2_s = 0.0d0
   a_v_s = 0.0d0
   a_d2_d = 0.0d0
   a_v_d = 0.0d0
   a_qmax_s = real (-999.999d0)
   a_qmax_d = -999.999d0
   open (newunit = a_u, file = output_folder (1:output_folder_length) // 'syrup_hydro_static.txt', &
         status = 'new', action = 'write')
   write (a_u, '(a)') 'SYRUP_HYDRO_CAPTURE_V1 static'
   call syrup_wr_si (a_u, 'nr', nr)
   call syrup_wr_si (a_u, 'nc', nc)
   call syrup_wr_si (a_u, 'nr1', nr1)
   call syrup_wr_si (a_u, 'nc1', nc1)
   call syrup_wr_si (a_u, 'nr2', nr2)
   call syrup_wr_si (a_u, 'nc2', nc2)
   call syrup_wr_si (a_u, 'ncell1', ncell1)
   call syrup_wr_si (a_u, 'nit', nit)
   call syrup_wr_si (a_u, 'ndirn', ndirn)
   call syrup_wr_si (a_u, 'iroute', iroute)
   call syrup_wr_si (a_u, 'ff_type', ff_type)
   call syrup_wr_si (a_u, 'inf_type', inf_type)
   call syrup_wr_si (a_u, 'inf_model', inf_model)
   call syrup_wr_si (a_u, 'rain_type', rain_type)
   call syrup_wr_sd (a_u, 'dt_s', dble (dt))
   call syrup_wr_sd (a_u, 'dx_mm', dble (dx))
   call syrup_wr_sd (a_u, 'dy_mm', dble (dy))
   call syrup_wr_sd (a_u, 'ksat_mod', ksat_mod)
   call syrup_wr_sd (a_u, 'psi_mod', psi_mod)
   call syrup_wr_sd (a_u, 'rval_initial_mm_s', dble (rval))
   call syrup_wr_sd (a_u, 'stormlength_s', dble (stormlength))
   call syrup_wr_d2 (a_u, 'ksat', ksat, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'psi', psi, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'theta_sat', theta_sat, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'theta', theta, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'cum_inf', cum_inf, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'cum_drain_initial_mm', cum_drain, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'stmax', stmax, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'drain_par', drain_par, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'pave', pave, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'slope', slope, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'ff', ff, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'rmask', rmask, a_n1, a_n2)
   call syrup_wr_i2 (a_u, 'aspect', aspect, a_n1, a_n2)
   call syrup_wr_i2 (a_u, 'order', order, size (order, 1), size (order, 2))
   a_work = d (1, :, :)
   call syrup_wr_d2 (a_u, 'd_initial_mm', a_work, a_n1, a_n2)
   a_work = q (1, :, :)
   call syrup_wr_d2 (a_u, 'q_initial_mm2_s', a_work, a_n1, a_n2)
   write (a_u, '(a)') 'SYRUP_HYDRO_CAPTURE_COMPLETE static'
   close (a_u)
   open (newunit = a_u_steps, file = output_folder (1:output_folder_length) // 'syrup_hydro_steps.txt', &
         status = 'new', action = 'write')
   write (a_u_steps, '(a)') 'SYRUP_HYDRO_CAPTURE_V1 steps'
   write (a_u_steps, '(a)') 'columns iter t_s rval_applied_mm_s sum_r2_active_mm_s max_r2_active_mm_s ' // &
      'q_plot_single_mm2_s q_outlet_double_mm2_s cn_export_step_m3 surface_sum_mm soil_sum_mm drain_sum_mm ' // &
      'excess_sum_mm_s max_depth_mm max_velocity_mm_s'
case (1)
   a_rval = dble (rval)
   a_sum_r2 = 0.0d0
   a_max_r2 = 0.0d0
   do a_i = 2, nr
      do a_j = 2, nc
         if (rmask (a_i, a_j) .ge. 0.0d0) then
            a_sum_r2 = a_sum_r2 + r2 (a_i, a_j)
            a_max_r2 = max (a_max_r2, r2 (a_i, a_j))
         endif
      enddo
   enddo
case (2)
   a_qd = 0.0d0
   a_q1d = 0.0d0
   a_sum_d2 = 0.0d0
   a_sum_inf = 0.0d0
   a_sum_drn = 0.0d0
   a_sum_exc = 0.0d0
   a_max_d = 0.0d0
   a_max_v = 0.0d0
   do a_i = 2, nr
      do a_j = 2, nc
         a_act = rmask (a_i, a_j) .ge. 0.0d0
         if (.not. a_act) cycle
         a_sum_d2 = a_sum_d2 + d (2, a_i, a_j)
         a_sum_inf = a_sum_inf + cum_inf (a_i, a_j)
         a_sum_drn = a_sum_drn + cum_drain (a_i, a_j)
         a_sum_exc = a_sum_exc + excess (a_i, a_j)
         a_max_d = max (a_max_d, d (2, a_i, a_j))
         a_max_v = max (a_max_v, v (a_i, a_j))
         a_out = (aspect (a_i, a_j) .eq. 1 .and. rmask (a_i - 1, a_j) .lt. 0.0d0) .or. &
                 (aspect (a_i, a_j) .eq. 2 .and. rmask (a_i, a_j + 1) .lt. 0.0d0) .or. &
                 (aspect (a_i, a_j) .eq. 3 .and. rmask (a_i + 1, a_j) .lt. 0.0d0) .or. &
                 (aspect (a_i, a_j) .eq. 4 .and. rmask (a_i, a_j - 1) .lt. 0.0d0)
         if (a_out) then
            a_qd = a_qd + q (2, a_i, a_j)
            a_q1d = a_q1d + q (1, a_i, a_j)
         endif
      enddo
   enddo
   a_cn = 0.5d0 * dble (dt) * dble (dx) * (a_q1d + a_qd) * 1.0d-9
   write (a_u_steps, '(i8,1x,13(ES25.16E3,1x))') iter, dble (iter) * dble (dt), a_rval, a_sum_r2, a_max_r2, &
      dble (q_plot), a_qd, a_cn, a_sum_d2, a_sum_inf, a_sum_drn, a_sum_exc, a_max_d, a_max_v
   a_nrec = a_nrec + 1
   if (q_plot .gt. a_qmax_s) then
      a_qmax_s = q_plot
      a_iter_s = iter
      a_d2_s = d (2, :, :)
      a_v_s = v
   endif
   if (a_qd .gt. a_qmax_d) then
      a_qmax_d = a_qd
      a_iter_d = iter
      a_d2_d = d (2, :, :)
      a_v_d = v
   endif
case (3)
   write (a_u_steps, '(a)') 'SYRUP_HYDRO_CAPTURE_COMPLETE steps'
   close (a_u_steps)
   a_n1 = size (ksat, 1)
   a_n2 = size (ksat, 2)
   a_nr1 = nr1
   a_nc1 = nc1
   a_dd = 0.0d0
   a_dv = 0.0d0
   do a_i = 1, a_nr1
      do a_j = 1, a_nc1
         a_dd = max (a_dd, abs (dmax (a_i, a_j) - a_d2_s (a_i, a_j)))
         a_dv = max (a_dv, abs (vmax (a_i, a_j) - a_v_s (a_i, a_j)))
      enddo
   enddo
   open (newunit = a_u, file = output_folder (1:output_folder_length) // 'syrup_hydro_final.txt', &
         status = 'new', action = 'write')
   write (a_u, '(a)') 'SYRUP_HYDRO_CAPTURE_V1 final'
   call syrup_wr_si (a_u, 'n_steps_written', a_nrec)
   call syrup_wr_si (a_u, 'peak_iter_single', a_iter_s)
   call syrup_wr_si (a_u, 'peak_iter_double', a_iter_d)
   call syrup_wr_sd (a_u, 'peak_q_single_mm2_s', dble (a_qmax_s))
   call syrup_wr_sd (a_u, 'peak_q_double_mm2_s', a_qmax_d)
   call syrup_wr_sd (a_u, 'max_abs_model_dmax_minus_capture_mm', a_dd)
   call syrup_wr_sd (a_u, 'max_abs_model_vmax_minus_capture_mm_s', a_dv)
   call syrup_wr_d2 (a_u, 'd_peak_single_mm', a_d2_s, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'v_peak_single_mm_s', a_v_s, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'd_peak_double_mm', a_d2_d, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'v_peak_double_mm_s', a_v_d, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'model_dmax_mm', dmax, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'model_vmax_mm_s', vmax, a_n1, a_n2)
   a_work = d (1, :, :)
   call syrup_wr_d2 (a_u, 'd_final_mm', a_work, a_n1, a_n2)
   a_work = q (1, :, :)
   call syrup_wr_d2 (a_u, 'q_final_mm2_s', a_work, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'v_final_mm_s', v, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'cum_inf_final_mm', cum_inf, a_n1, a_n2)
   call syrup_wr_d2 (a_u, 'cum_drain_final_mm', cum_drain, a_n1, a_n2)
   write (a_u, '(a)') 'SYRUP_HYDRO_CAPTURE_COMPLETE final'
   close (a_u)
case default
   error stop 'syrup capture: unknown stage'
end select
return
end

subroutine syrup_wr_si (a_u, a_name, a_val)
implicit none
integer :: a_u, a_val
character (len = *) :: a_name
write (a_u, '(a,1x,a,1x,i0)') 'scalar', trim (a_name), a_val
return
end

subroutine syrup_wr_sd (a_u, a_name, a_val)
implicit none
integer :: a_u
double precision :: a_val
character (len = *) :: a_name
write (a_u, '(a,1x,a,1x,ES25.16E3)') 'scalar', trim (a_name), a_val
return
end

subroutine syrup_wr_d2 (a_u, a_name, a_arr, a_n1, a_n2)
implicit none
integer :: a_u, a_n1, a_n2, a_i, a_j
character (len = *) :: a_name
double precision :: a_arr (a_n1, a_n2)
write (a_u, '(a,1x,a,1x,a,2(1x,i0))') 'array', trim (a_name), 'd', a_n1, a_n2
do a_i = 1, a_n1
   write (a_u, '(*(ES25.16E3,1x))') (a_arr (a_i, a_j), a_j = 1, a_n2)
enddo
return
end

subroutine syrup_wr_i2 (a_u, a_name, a_arr, a_n1, a_n2)
implicit none
integer :: a_u, a_n1, a_n2, a_i, a_j
character (len = *) :: a_name
integer :: a_arr (a_n1, a_n2)
write (a_u, '(a,1x,a,1x,a,2(1x,i0))') 'array', trim (a_name), 'i', a_n1, a_n2
do a_i = 1, a_n1
   write (a_u, '(*(i0,1x))') (a_arr (a_i, a_j), a_j = 1, a_n2)
enddo
return
end
'''


def insert_hooks(text: str) -> str:
    """The parent storm driver with the four hook calls and the appended hook source. Each anchor must occur
    exactly once; the original text is otherwise untouched."""
    if HOOK_MARKER in text or HOOK_CALL in text:
        raise ValueError("the parent already contains the capture hooks")
    out = text
    for anchor, stage in INSERTIONS:
        if out.count(anchor) != 1:
            raise ValueError(f"expected exactly one occurrence of the hook anchor {anchor.strip()!r}, found {out.count(anchor)}")
        call = f"   {HOOK_CALL} ({stage})\n"
        if stage == 3:  # the call goes between the loop's `enddo` and `close (51)`
            out = out.replace(anchor, anchor.replace("enddo\n", "enddo\n" + call, 1), 1)
        else:  # the call goes on the line after the anchor
            out = out.replace(anchor, anchor + call, 1)
    return out + HOOK_SOURCE


def remove_hooks(text: str) -> str:
    """Inverse of `insert_hooks`: drop the inserted call lines and the appended source."""
    head, marker, _ = text.partition(HOOK_SOURCE)
    if not marker:
        raise ValueError("hook source not found")
    keep = [ln for ln in head.splitlines(keepends=True) if HOOK_CALL not in ln]
    return "".join(keep)


def prepare(parent: Path, output: Path, *, expected_xml_sha256: str | None = EXPECTED_XML_SHA256,
            parent_build: Path | None = None) -> dict:
    parent, output = parent.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("a new output directory outside the parent tree is required")
    manifest = json.loads((parent / "benchmark_manifest.json").read_text())
    if inventory(parent) != manifest["prepared_sha256"]:
        raise ValueError("the parent no longer matches its own manifest")
    if expected_xml_sha256 is not None and sha(parent / "mahleran_input.xml") != expected_xml_sha256:
        raise ValueError("the parent XML is not the heterogeneous (normal-conductivity) case of the earlier run")
    xml = (parent / "mahleran_input.xml").read_text(encoding="latin-1")
    if 'finalInfiltrationRateDistribution value="normal"' not in xml:
        raise ValueError("finalInfiltrationRateDistribution is not 'normal': not the heterogeneous case")
    text = (parent / STORM).read_bytes().decode("latin-1")
    patched = insert_hooks(text)
    if remove_hooks(patched) != text:
        raise ValueError("hook insertion is not exactly reversible; refusing")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shutil.copytree(parent, stage, dirs_exist_ok=True)
        (stage / STORM).write_bytes(patched.encode("latin-1"))
        patch = "".join(difflib.unified_diff(text.splitlines(True), patched.splitlines(True),
                                             fromfile="a/" + str(STORM), tofile="b/" + str(STORM)))
        (stage / "syrup_hydro_capture.patch").write_text(patch, encoding="latin-1")
        after = inventory(stage)
        changed = sorted(name for name in manifest["prepared_sha256"] if after.get(name) != manifest["prepared_sha256"][name])
        if changed != [str(STORM)] or set(after) != set(manifest["prepared_sha256"]):
            raise ValueError(f"unexpected derivative differences: {changed}")
        if inventory(parent) != manifest["prepared_sha256"]:
            raise ValueError("the parent changed during preparation")
        same_as_build = None
        if parent_build is not None:
            earlier = json.loads(Path(parent_build).read_text())["source_sha256"]
            same_as_build = all(sha(stage / name) == digest for name, digest in earlier.items() if name != str(STORM))
            if not same_as_build:
                raise ValueError("a compiled source other than the storm driver differs from the earlier build")
        record = dict(manifest)
        record.update(
            status="prepared with read-only capture hooks; not built or run by the preparation",
            parent_root=str(parent), parent_prepared_sha256=manifest["prepared_sha256"], copy_root=str(output),
            prepared_sha256=after, changed_vs_parent=changed, hydrology_capture=True,
            hook_scope=("four `call syrup_hydro_capture (n)` lines in MAHLERAN_storm_xml.f90 plus appended read-only "
                        "subroutines; writes new syrup_hydro_{static,steps,final}.txt; no equation, RNG, forcing, "
                        "step or state change"),
            patch_sha256=hashlib.sha256(patch.encode("latin-1")).hexdigest(),
            script_sha256=sha(__file__), expected_xml_sha256=expected_xml_sha256,
            unchanged_files_match_earlier_build=same_as_build,
            earlier_heterogeneous_run="outputs/phase7/mahleran_fixed_no_splash (rounded outputs to be compared by hash)",
        )
        (stage / "benchmark_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", type=Path, default=Path("outputs/phase7/mahleran_plot1_no_splash_linux"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-build", type=Path, default=Path("outputs/phase7/build_checked_linux/build.json"),
                        help="earlier build record; every other compiled source must hash identically")
    args = parser.parse_args()
    record = prepare(args.parent, args.output, parent_build=args.parent_build if args.parent_build.is_file() else None)
    print(json.dumps({"copy_root": record["copy_root"], "changed_vs_parent": record["changed_vs_parent"],
                      "patch_sha256": record["patch_sha256"],
                      "unchanged_files_match_earlier_build": record["unchanged_files_match_earlier_build"]}, indent=2))


if __name__ == "__main__":
    main()
