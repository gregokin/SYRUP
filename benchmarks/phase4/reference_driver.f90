! MAPLE-SYRUP Phase 4b test driver for the ORIGINAL, UNMODIFIED MAHLERAN
! 1.2.3 subroutine route_water (src/Subroutines_Water/route_water.for),
! flow-routing_solution_method (iroute) = 5, friction_factor_type = 1.
!
! This program contains NO routing equations. It loads one step of legacy
! state -- millimetre and second units, legacy north-first 1-based indices
! over the full grid including the boundary ring -- into module shared_data,
! calls route_water exactly once, and writes d(2), q(2), qin(2) and v for
! every cell. The output file is opened only after route_water returns, and
! its last line is the completion marker. route_water ends the program with
! STOP on its error paths (possibly with exit status 0); the harness treats a
! missing marker as an incomplete run whatever the exit status.
!
! Usage: syrup_route_water_driver INPUT OUTPUT
!
! INPUT (list-directed):
!   nrows_full ncols_full ncell1 iroute ff_type
!   dt_s dx_mm
!   ncell1 lines: i j level          (the routing order, upstream first)
!   nrows_full*ncols_full lines, i outer, j inner:
!     aspect rmask slope ff d1_mm q1_mm2_s qin1_mm2_s excess_mm_s
!
! dt and dx are default-REAL variables in shared_data (implicit typing in the
! module), so both must be exactly representable in that kind.
program syrup_route_water_driver
   use shared_data
   implicit none
   character(len=4096) :: in_path, out_path
   integer :: u_in, u_out, nrows_full, ncols_full, ncells, method, friction_type
   integer :: i, j, k, asp
   double precision :: dt_in, dx_in, rm, sl, fr, d1, q1, qin1, ex

   if (command_argument_count() /= 2) error stop 'usage: syrup_route_water_driver INPUT OUTPUT'
   call get_command_argument(1, in_path)
   call get_command_argument(2, out_path)

   open (newunit=u_in, file=trim(in_path), status='old', action='read')
   read (u_in, *) nrows_full, ncols_full, ncells, method, friction_type
   read (u_in, *) dt_in, dx_in
   if (method /= 5) error stop 'driver: only flow-routing_solution_method 5 is supported'
   if (friction_type /= 1) error stop 'driver: only friction_factor_type 1 is supported'
   if (nrows_full < 3 .or. ncols_full < 3 .or. ncells < 1) error stop 'driver: invalid sizes'
   if (ncells > (nrows_full - 2) * (ncols_full - 2)) error stop 'driver: too many cells'

   nr2 = nrows_full
   nc2 = ncols_full
   nr = nr2 - 1
   nc = nc2 - 1
   nr1 = nr
   nc1 = nc
   ncell = nr1 * nc1
   ncell1 = ncells
   iroute = method
   ff_type = friction_type
   ndirn = 4
   iter = 1
   dt = real(dt_in, kind(dt))
   dx = real(dx_in, kind(dx))
   dy = dx
   if (dble(dt) /= dt_in) error stop 'driver: dt is not exactly representable in the shared_data REAL kind'
   if (dble(dx) /= dx_in) error stop 'driver: dx is not exactly representable in the shared_data REAL kind'
   dtdx = dt / dx

   allocate (order(ncell1, 3), aspect(nr2, nc2), rmask(nr2, nc2), slope(nr2, nc2), ff(nr2, nc2))
   allocate (v(nr2, nc2), excess(nr2, nc2), d(2, nr2, nc2), q(2, nr2, nc2), qin(2, nr2, nc2))
   do k = 1, ncell1
      read (u_in, *) order(k, 1), order(k, 2), order(k, 3)
   end do
   do i = 1, nr2
      do j = 1, nc2
         read (u_in, *) asp, rm, sl, fr, d1, q1, qin1, ex
         aspect(i, j) = asp
         rmask(i, j) = rm
         slope(i, j) = sl
         ff(i, j) = fr
         ! Level 1 is the post-infilt state route_water reads; level 2 starts
         ! as update_water_flow leaves it and is overwritten for routed cells.
         d(1, i, j) = d1
         d(2, i, j) = d1
         q(1, i, j) = q1
         q(2, i, j) = q1
         qin(1, i, j) = qin1
         qin(2, i, j) = qin1
         excess(i, j) = ex
         v(i, j) = 0.0d0
      end do
   end do
   close (u_in)

   call route_water

   open (newunit=u_out, file=trim(out_path), status='replace', action='write')
   write (u_out, '(A,1X,I0,1X,I0)') 'REAL_STORAGE_BITS_DT_DX', storage_size(dt), storage_size(dx)
   write (u_out, '(A,1X,I0,1X,I0,1X,I0)') 'GRID', nr2, nc2, ncell1
   do i = 1, nr2
      do j = 1, nc2
         write (u_out, '(I0,1X,I0,4(1X,ES26.17E3))') i, j, d(2, i, j), q(2, i, j), qin(2, i, j), v(i, j)
      end do
   end do
   write (u_out, '(A)') 'SYRUP_ROUTE_WATER_DRIVER_COMPLETE'
   close (u_out)
end program syrup_route_water_driver
