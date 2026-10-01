! Driver calling the ORIGINAL MAHLERAN flow_distrib on a small synthetic grid (Phase 7e equivalence test).
! Reads: nr2 nc2 nclass dx_m dt ; aspect(nr2,nc2) row-major ; then per (im jm phi) lines: detach travel_dist nsteps
! Writes depos_soil(phi, im, jm) for every cell/class after all calls. No physics law is executed.
program flow_distrib_driver
  use shared_data
  implicit none
  integer :: unit_in, unit_out, i, j, k, n_calls, c, nclass, nst, status
  double precision :: det, tdist
  character(len=2048) :: input_path, output_path
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  open(newunit=unit_in, file=trim(input_path), status='old', action='read')
  open(newunit=unit_out, file=trim(output_path), status='new', action='write')
  read(unit_in, *) nr2, nc2, nclass, dx_m, dt
  dx = dx_m * 1.0d3; dy = dx
  allocate(aspect(nr2, nc2), detach_soil(6, nr2, nc2), depos_soil(6, nr2, nc2))
  detach_soil = 0.0d0; depos_soil = 0.0d0
  do i = 1, nr2
    read(unit_in, *) (aspect(i, j), j = 1, nc2)
  enddo
  read(unit_in, *) n_calls
  do c = 1, n_calls
    read(unit_in, *, iostat=status) im, jm, phi, det, tdist, nst
    if (status /= 0) stop 1
    detach_soil(phi, im, jm) = det
    call flow_distrib(tdist, nst)
  enddo
  do i = 1, nr2
    do j = 1, nc2
      write(unit_out, '(*(ES25.16E3,1X))') (depos_soil(k, i, j), k = 1, nclass)
    enddo
  enddo
  write(unit_out, '(A)') 'SYRUP_FLOW_DISTRIB_DRIVER_COMPLETE'
  close(unit_in); close(unit_out)
end program
