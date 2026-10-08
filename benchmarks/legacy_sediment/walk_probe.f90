! Tiny probe of the ORIGINAL flow_distrib (task gpu_sediment, A2). Glue only: state and calls, no physics.
!
! input text: nr2 nc2 dx_m dt
!             aspect(i, j) rows i = 1..nr2 (north-first, ring included)
!             ncalls
!             im jm phi detach travel_dist_ave nsteps        (ncalls lines; each call ADDS to depos_soil, like the original)
! output text: depos_soil(phi, i, j) for i, j, phi=1..6 (ES25.16E3), then SYRUP_WALK_PROBE_COMPLETE
!
! shared_data keeps its implicit default-REAL globals: dx, dx_m and dt are narrowed to kind 4, so use exactly representable
! values (0.5, 1, 2 ...). A legacy STOP returns status 0: completion is the marker, not the exit status.
program walk_probe
use shared_data
implicit none
integer :: u_in, u_out, sg_i, sg_j, sg_k, sg_n, sg_c, sg_nst, sg_ios
double precision :: sg_det, sg_td, sg_dxm, sg_dt
character(len=4096) :: sg_inp, sg_outp
call get_command_argument(1, sg_inp)
call get_command_argument(2, sg_outp)
open(newunit=u_in, file=trim(sg_inp), status='old', action='read')
read(u_in, *) nr2, nc2, sg_dxm, sg_dt
dx_m = real(sg_dxm); dt = real(sg_dt); dx = dx_m * 1.0d3; dy = dx
if (dble(dx_m) /= sg_dxm .or. dble(dt) /= sg_dt) error stop 'dx_m/dt not exactly representable in default REAL'
allocate(aspect(nr2, nc2), detach_soil(6, nr2, nc2), depos_soil(6, nr2, nc2))
detach_soil = 0.d0; depos_soil = 0.d0
do sg_i = 1, nr2
   read(u_in, *) (aspect(sg_i, sg_j), sg_j = 1, nc2)
enddo
read(u_in, *) sg_n
do sg_k = 1, sg_n
   read(u_in, *, iostat=sg_ios) im, jm, sg_c, sg_det, sg_td, sg_nst
   if (sg_ios /= 0) error stop 'bad call line'
   if (sg_c < 1 .or. sg_c > 6 .or. im < 1 .or. im > nr2 .or. jm < 1 .or. jm > nc2) error stop 'call index out of range'
   phi = sg_c
   detach_soil(phi, im, jm) = sg_det
   call flow_distrib(sg_td, sg_nst)
   detach_soil(phi, im, jm) = 0.d0
enddo
close(u_in)
open(newunit=u_out, file=trim(sg_outp), status='new', action='write')
do sg_j = 1, nc2
   do sg_i = 1, nr2
      write(u_out, '(6(ES25.16E3,1X))') (depos_soil(sg_c, sg_i, sg_j), sg_c = 1, 6)
   enddo
enddo
write(u_out, '(A)') 'SYRUP_WALK_PROBE_COMPLETE'
close(u_out)
end program walk_probe
