! Read-only diagnostic storage for the ORIGINAL route_sediment_xml Crank-Nicolson solve (task gpu_sediment, A2).
!
! `syrup_trial` receives the ACTUAL unclipped trial mobile depth d_soil(phi,2,i,j) [mm] and `syrup_clip_factor` the
! factor (1 + 0.5 dt v_soil / dx) of the original clipping branch, recorded by the hooked copy of route_sediment_xml
! immediately BEFORE the original `if (d_soil .lt. 0)` clip. Nothing here is read by any scientific routine, so the hook cannot
! change a result (checked by the hook-on versus hook-off tiny gate). In a build WITHOUT the hook the arrays stay zero and
! `syrup_hook_called` stays .false. (the driver reports it).
module syrup_hook
  double precision, allocatable :: syrup_trial(:, :, :), syrup_clip_factor(:, :, :)
  logical :: syrup_hook_called = .false.
end module syrup_hook
