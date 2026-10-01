! Branch/side-effect test of the actual route_sediment_xml routine.
! Physics callees are explicit test doubles; this is NOT a storm benchmark.
module branch_counts
  integer :: rain_calls=0, splash_calls=0, diffuse_calls=0
end module
program check_branches
  use shared_data
  use parameters_from_xml
  use branch_counts
  implicit none
  integer :: wet
  nr=2; nr1=2; nr2=4; nc=2; nc1=2; nc2=4; ncell1=1
  allocate(rmask(4,4), r2(4,4), d(2,4,4), v(4,4), slope(4,4), aspect(4,4), order(1,3))
  allocate(sed_propn(6,4,4), detach_soil(6,4,4), depos_soil(6,4,4), v_soil(6,4,4))
  allocate(d_soil(6,2,4,4), q_soil(6,2,4,4), qsedin(6,2,4,4), add_soil(6,4,4))
  allocate(amm_q(6,4,4), nit_q(6,4,4), TN_q(6,4,4), TP_q(6,4,4), IC_q(6,4,4), TC_q(6,4,4))
  allocate(sed_tot(4,4), detach_tot(4,4), depos_tot(4,4), z_change(4,4))
  allocate(amm_tot(4,4), nit_tot(4,4), TN_tot(4,4), TP_tot(4,4), IC_tot(4,4), TC_tot(4,4))
  rmask=-1; rmask(2,2)=1; r2=0; r2(2,2)=0.01d0
  v=0; slope=0.01d0; aspect=0; order(1,:)=[2,2,1]
  sed_propn=1.d0/6.d0; diameter=0.001d0
  dx=500.d0; dt=1; dtdx=dt/dx; density=2.65d0; MiC=0
  sediment_routing_solution_method=2
  p_ammonium=0; p_nitrate=0; p_TN=0; p_TP=0; p_IC=0; p_TC=0
  do wet=0,1
    rain_calls=0; splash_calls=0; diffuse_calls=0
    d=0; d(:,2,2)=real(wet,8)*0.1d0
    detach_soil=7; depos_soil=9 ! deliberately stale rate outputs
    v_soil=0; d_soil=0; d_soil(:,1,2,2)=1.d0
    q_soil=0; qsedin=0; add_soil=0
    sed_tot=0; detach_tot=0; depos_tot=0; z_change=0
    amm_tot=0; nit_tot=0; TN_tot=0; TP_tot=0; IC_tot=0; TC_tot=0
    call route_sediment_xml
    write(*,'(4(i0,1x),3(es24.16,1x))') wet,rain_calls,splash_calls,diffuse_calls, &
       detach_soil(1,2,2),depos_soil(1,2,2),d_soil(1,2,2,2)
  enddo
end program
subroutine raindrop_detachment
  use shared_data
  use branch_counts
  rain_calls=rain_calls+1
  detach_soil(:,im,jm)=2.d0
end subroutine
subroutine splash_transport
  use shared_data
  use branch_counts
  splash_calls=splash_calls+1
  depos_soil(:,im,jm)=2.d0
end subroutine
subroutine diffuse_flow_transport
  use shared_data
  use branch_counts
  diffuse_calls=diffuse_calls+1
  depos_soil(phi,im,jm)=0.5d0
end subroutine
subroutine flow_detachment
  error stop 'unexpected flow-detachment branch'
end subroutine
subroutine suspended_transport
  error stop 'unexpected suspension branch'
end subroutine
subroutine conc_flow_transport
  error stop 'unexpected concentrated branch'
end subroutine
subroutine route_markers_xml
  error stop 'marker mode outside this benchmark'
end subroutine
