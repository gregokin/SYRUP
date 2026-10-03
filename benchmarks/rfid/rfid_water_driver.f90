! EXPERIMENTAL water-only timing driver for the RFID_2014 benchmark (task rfid_timing).
!
! Links the UNCHANGED original MAHLERAN routines infilt.for, route_water.for (iroute 2 = Newton-Crank-Nicolson, the native
! method; iroute 5 = bisection-Crank-Nicolson, the method the SYRUP solvers reproduce), update_water_flow.for, ff_type8.for and
! the original shared_data module. It contains NO routing, infiltration or friction equation: it only reads the common input
! arrays, applies the forcing, calls the originals in the order of MAHLERAN_storm_xml.f90 100-104 / 171 (infilt, route_water,
! update_water_flow; accumulate_flow is omitted because methods 2 and 5 do not use it), and keeps cheap per-step accounting.
!
! TIMING REGION = the step loop only (system_clock). Inside it: forcing assignment, infilt, route_water, update_water_flow
! (which also copies the six dummy sediment classes of the original, disclosed), a finite/non-negative check of the new depth
! and discharge, running totals of rain / export / peak outlet discharge, and a report row every `nrep` steps held in memory.
! Outside it: all text input (forcing and arrays are read before the timer starts), allocation, and every output file.
!
! A STOP inside an original routine returns status 0; the history file therefore ends with a completion marker that the
! caller must check. Any non-finite or negative state ends the run with a NONZERO status (error stop); it is reported, never
! repaired. The originals' known stale-inflow / bracket behaviour is retained, not corrected.
!
! usage: rfid_water_driver input.dat history.dat final.dat   (stdout: RFID_LOOP_SECONDS, RFID_STEPS)
program rfid_water_driver
use shared_data
use, intrinsic :: ieee_arithmetic
implicit none
integer :: au_i, au_j, au_k, au_n, au_in, au_out, au_final, au_flag, au_act, au_nrep, au_nrow, au_row
real(8) :: au_dt, au_dx, au_scale_rate, au_rain, au_export, au_oldout, au_peak_q, au_peak_t, au_qnow, au_tloop
integer(8) :: au_c0, au_c1, au_crate
real(8), allocatable :: au_rate(:), au_rows(:, :), au_scale(:, :)
logical, allocatable :: au_outlet(:, :), au_active(:, :)
character(len=4096) :: au_path, au_hist, au_grid
call get_command_argument(1, au_path)
call get_command_argument(2, au_hist)
call get_command_argument(3, au_grid)
open(newunit=au_in, file=trim(au_path), status='old')
read(au_in, *) nr2, nc2, ncell1, au_n, au_dt, au_dx, iroute, au_nrep
nr = nr2 - 1; nc = nc2 - 1; nr1 = nr; nc1 = nc; ncell = nr * nc
dt = real(au_dt); dx = real(au_dx); dy = dx; dtdx = dt / dx
if (dble(dt) /= au_dt .or. dble(dx) /= au_dx) error stop 'inexact default REAL dt/dx'
if (iroute /= 2 .and. iroute /= 5) error stop 'iroute must be 2 or 5'
if (au_nrep < 1) error stop 'report cadence must be >= 1 step'
ff_type = 1; inf_model = 1; inf_type = 1; ndirn = 4; ksat_mod = 1.d0; psi_mod = 1.d0
allocate(order(ncell1, 3), aspect(nr2, nc2), rmask(nr2, nc2), slope(nr2, nc2), ff(nr2, nc2))
allocate(d(2, nr2, nc2), q(2, nr2, nc2), qin(2, nr2, nc2), v(nr2, nc2), v1(nr2, nc2), qsum(nr2, nc2))
allocate(excess(nr2, nc2), cum_inf(nr2, nc2), cum_inf_p(nr2, nc2), cum_drain(nr2, nc2), stmax(nr2, nc2))
allocate(theta(nr2, nc2), theta_sat(nr2, nc2), ksat(nr2, nc2), psi(nr2, nc2), pave(nr2, nc2), r2(nr2, nc2))
allocate(drain_par(nr2, nc2), t_ponding(nr2, nc2), au_scale(nr2, nc2), au_outlet(nr2, nc2), au_active(nr2, nc2))
allocate(d_soil(6, 2, nr2, nc2), q_soil(6, 2, nr2, nc2), qsedin(6, 2, nr2, nc2), v_soil(6, nr2, nc2))
d = 0; q = 0; qin = 0; v = 0; v1 = 0; qsum = 0; excess = 0; cum_inf_p = 0; cum_drain = 0; r2 = 0; t_ponding = -9999
d_soil = 0; q_soil = 0; qsedin = 0; v_soil = 0
do au_k = 1, ncell1
   read(au_in, *) order(au_k, :)
enddo
do au_i = 1, nr2
   do au_j = 1, nc2
      read(au_in, *) aspect(au_i, au_j), rmask(au_i, au_j), slope(au_i, au_j), ff(au_i, au_j), &
         ksat(au_i, au_j), psi(au_i, au_j), pave(au_i, au_j), drain_par(au_i, au_j), &
         theta_sat(au_i, au_j), theta(au_i, au_j), cum_inf(au_i, au_j), stmax(au_i, au_j), &
         au_scale(au_i, au_j), au_flag, au_act
      au_outlet(au_i, au_j) = au_flag == 1; au_active(au_i, au_j) = au_act == 1
   enddo
enddo
allocate(au_rate(au_n))
do au_k = 1, au_n
   read(au_in, *) au_rate(au_k)
enddo
close(au_in)
au_nrow = (au_n + au_nrep - 1) / au_nrep   ! ceil: a row every nrep steps AND one at the final step (exactly once)
allocate(au_rows(max(au_nrow, 1), 7))
au_rain = 0; au_export = 0; au_peak_q = 0.d0; au_peak_t = 0; au_row = 0  ! strict '>' from an initial dry 0, as the SYRUP driver

call system_clock(au_c0, au_crate)
do iter = 1, au_n
   r2 = au_rate(iter) * au_scale
   au_rain = au_rain + sum(r2, mask=au_active) * dble(dt) * dble(dx)**2 * 1.d-9
   call infilt
   au_oldout = sum(q(1, :, :), mask=au_outlet)
   call route_water
   if (.not. all(ieee_is_finite(d(2, :, :)))) error stop 'nonfinite depth'
   if (.not. all(ieee_is_finite(q(2, :, :)))) error stop 'nonfinite discharge'
   if (any(d(2, :, :) < 0.d0) .or. any(q(2, :, :) < 0.d0)) error stop 'negative depth or discharge'
   au_qnow = sum(q(2, :, :), mask=au_outlet) * dble(dx) * 1.d-9
   au_export = au_export + .5d0 * dble(dt) * dble(dx) * (au_oldout + sum(q(2, :, :), mask=au_outlet)) * 1.d-9
   if (au_qnow > au_peak_q) then
      au_peak_q = au_qnow; au_peak_t = dble(iter) * dble(dt)
   endif
   if (mod(iter, au_nrep) == 0 .or. iter == au_n) then
      if (au_row >= au_nrow) error stop 'report row overflow'
      au_row = au_row + 1
      au_rows(au_row, 1) = dble(iter) * dble(dt)
      au_rows(au_row, 2) = au_export
      au_rows(au_row, 3) = sum(d(2, :, :), mask=au_active) * dble(dx)**2 * 1.d-9
      au_rows(au_row, 4) = sum(cum_inf, mask=au_active) * dble(dx)**2 * 1.d-9
      au_rows(au_row, 5) = sum(cum_drain, mask=au_active) * dble(dx)**2 * 1.d-9
      au_rows(au_row, 6) = au_rain
      au_rows(au_row, 7) = au_qnow
   endif
   call update_water_flow
enddo
call system_clock(au_c1)
au_tloop = dble(au_c1 - au_c0) / dble(au_crate)

write(*, '(A,ES25.16E3)') 'RFID_LOOP_SECONDS ', au_tloop
write(*, '(A,I0)') 'RFID_STEPS ', au_n
write(*, '(A,I0)') 'RFID_IROUTE ', iroute
open(newunit=au_out, file=trim(au_hist), status='replace')
write(au_out, '(A)') 'time_s export_m3 surface_m3 soil_m3 drain_m3 rain_m3 outlet_m3_s'
do au_k = 1, au_row
   write(au_out, '(7(ES25.16E3,1X))') au_rows(au_k, :)
enddo
write(au_out, '(A,ES25.16E3,1X,ES25.16E3)') 'PEAK ', au_peak_q, au_peak_t
write(au_out, '(A)') 'SYRUP_RFID_DRIVER_COMPLETE'
close(au_out)
open(newunit=au_final, file=trim(au_grid), status='replace')
do au_i = 2, nr
   do au_j = 2, nc
      if (au_active(au_i, au_j)) write(au_final, '(2(I0,1X),3(ES25.16E3,1X))') au_i, au_j, d(1, au_i, au_j) * 1.d-3, &
         cum_inf(au_i, au_j) * 1.d-3, q(1, au_i, au_j) * 1.d-6
   enddo
enddo
close(au_final)
end program
