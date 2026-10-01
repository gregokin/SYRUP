"""Prepare an isolated diagnostic-only MAHLERAN derivative; no physics change."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase7"))
from prepare_mahleran import inventory

parent = Path("outputs/phase7/mahleran_deterministic_ksat").resolve()
out = Path("outputs/phase7b/mahleran_ledger_source_v3").resolve()
if out.exists():
    raise SystemExit("new output required")
manifest = json.loads((parent / "benchmark_manifest.json").read_text())
assert inventory(parent) == manifest["prepared_sha256"]
shutil.copytree(parent, out)
shared = out / "src/Program_Control/shared_data.f90"
s = shared.read_text(encoding="latin-1")
s = s.replace("module shared_data\nsave\n", "module shared_data\nsave\n! SYRUP audit only: clipping diagnostic (mm equivalent depth by class).\ndouble precision :: syrup_clip_step(6) = 0.0d0\n", 1)
shared.write_text(s, encoding="latin-1")
route = out / "src/Subroutines_Sediment/route_sediment_xml.f90"
s = route.read_text(encoding="latin-1")
s = s.replace("data sdirin / 1, 0, -1, 0, 0, -1, 0, 1 /", "data sdirin / 1, 0, -1, 0, 0, -1, 0, 1 /\n\nsyrup_clip_step = 0.0d0", 1)
needle = "if (d_soil (phi, 2, i, j).lt.0.d0) then"
assert s.count(needle) == 1
s = s.replace(needle, needle + "\n               syrup_clip_step(phi) = syrup_clip_step(phi) - d_soil(phi,2,i,j) &\n                    * (1.d0 + 0.5d0 * dt / dx * v_soil(phi,i,j))", 1)
needle = "if (d_soil (phi, 2, im, jm).lt.0) then"
assert s.count(needle) == 1
s = s.replace(needle, needle + "\n                  syrup_clip_step(phi) = syrup_clip_step(phi) - d_soil(phi,2,im,jm)", 1)
route.write_text(s, encoding="latin-1")
storm = out / "src/Program_Control/MAHLERAN_storm_xml.f90"
s = storm.read_text(encoding="latin-1")
assert s.count("   call route_sediment_xml\n") == 1
s = s.replace("   call route_sediment_xml\n", "   call route_sediment_xml\n   call syrup_audit_sediment\n", 1)
s += '''

! SYRUP diagnostic hook; original fields are READ ONLY here.
subroutine syrup_audit_sediment
use shared_data
use parameters_from_xml, only: output_folder, output_folder_length
implicit none
integer :: a_i, a_j, a_c
logical :: a_active, a_outlet
double precision :: a_factor, a_fluxfactor
double precision :: a_pick(6), a_dep(6), a_ring(6), a_old(6), a_new(6)
double precision :: a_cn(6), a_endpoint(6), a_cellflux(6), a_clip(6), a_resid(6)
a_factor = dx * dy * density * 1.0d-6
a_fluxfactor = dx * density * 1.0d-6 * dt
a_pick=0.d0; a_dep=0.d0; a_ring=0.d0; a_old=0.d0; a_new=0.d0
a_cn=0.d0; a_endpoint=0.d0; a_cellflux=0.d0
a_clip = syrup_clip_step * a_factor
do a_i=1,nr2
 do a_j=1,nc2
  a_active = a_i>=2.and.a_i<=nr.and.a_j>=2.and.a_j<=nc
  if (a_active) a_active=rmask(a_i,a_j)>=0.d0
  if (.not.a_active) then
   a_ring=a_ring+depos_soil(:,a_i,a_j)*dt*a_factor
   cycle
  endif
  a_pick=a_pick+detach_soil(:,a_i,a_j)*dt*a_factor
  a_dep=a_dep+depos_soil(:,a_i,a_j)*dt*a_factor
  a_old=a_old+d_soil(:,1,a_i,a_j)*a_factor
  a_new=a_new+d_soil(:,2,a_i,a_j)*a_factor
  a_cellflux=a_cellflux+0.5d0*a_fluxfactor*(qsedin(:,2,a_i,a_j)+qsedin(:,1,a_i,a_j) &
            -q_soil(:,2,a_i,a_j)-q_soil(:,1,a_i,a_j))
  a_outlet=.false.
  select case(aspect(a_i,a_j))
  case(1)
   a_outlet=rmask(a_i-1,a_j)<0.d0
  case(2)
   a_outlet=rmask(a_i,a_j+1)<0.d0
  case(3)
   a_outlet=rmask(a_i+1,a_j)<0.d0
  case(4)
   a_outlet=rmask(a_i,a_j-1)<0.d0
  end select
  if (a_outlet) then
   a_cn=a_cn+0.5d0*a_fluxfactor*(q_soil(:,1,a_i,a_j)+q_soil(:,2,a_i,a_j))
   a_endpoint=a_endpoint+a_fluxfactor*q_soil(:,2,a_i,a_j)
  endif
 enddo
enddo
a_resid=a_new-a_old-(a_pick-a_dep)-a_cellflux-a_clip
if (iter==1) then
 open(988,file=output_folder(1:output_folder_length)//'syrup_sediment_ledger.dat',status='replace')
 write(988,'(a)') '# iter t_s class pickup_kg deposition_active_kg deposition_outside_active_kg effective_clip_source_kg old_mobile_kg new_mobile_kg cn_export_kg endpoint_export_kg net_cell_flux_kg algebra_residual_kg internal_flux_residual_kg'
endif
do a_c=1,6
 write(988,'(i8,1x,es24.16,1x,i2,11(1x,es24.16))') iter,dble(iter)*dt,a_c, &
 a_pick(a_c),a_dep(a_c),a_ring(a_c),a_clip(a_c),a_old(a_c),a_new(a_c),a_cn(a_c), &
 a_endpoint(a_c),a_cellflux(a_c),a_resid(a_c),a_cellflux(a_c)+a_cn(a_c)
enddo
flush(988)
end subroutine syrup_audit_sediment
'''
storm.write_text(s, encoding="latin-1")
manifest.update(copy_root=str(out), status="diagnostic-only sediment ledger derivative, not yet executed", parent_root=str(parent), parent_prepared_sha256=inventory(parent), prepared_sha256=inventory(out), diagnostic_scope="read-only post-route pool/source/face audit plus additive counter immediately before existing clipping; original state expressions unchanged")
(out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"prepared": str(out), "changed": [str(p.relative_to(out)) for p in [shared, route, storm]], "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
