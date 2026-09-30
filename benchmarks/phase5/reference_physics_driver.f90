! Original MAHLERAN equation calls; no production sediment routing.
! flow_distrib is replaced only by an argument recorder below, to observe
! the length supplied by each unmodified transport law before redistribution.
module capture_distance
  implicit none
  double precision :: last_distance = -1d0
end module
program reference_physics
  use shared_data
  use parameters_from_xml, only: KE_model_type
  use capture_distance
  implicit none
  integer :: unit_in, unit_out, status, row, mode, n
  double precision :: h_m, speed_m_s, rain_m_s, slope_in, veg_percent, median_m
  double precision :: fractions(6), rain_rate(6), flow_rate(6), distances(6,3), speeds(6,3)
  character(len=2048) :: input_path, output_path
  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  open(newunit=unit_in,file=trim(input_path),status='old',action='read')
  open(newunit=unit_out,file=trim(output_path),status='new',action='write')
  allocate(slope(3,3),d(2,3,3),v(3,3),r2(3,3),veg(3,3))
  allocate(sed_propn(6,3,3),detach_soil(6,3,3),depos_soil(6,3,3),sed_temp(6,3,3))
  allocate(v_soil(6,3,3),raindrop_detach_tot(3,3),flow_detach_tot(3,3))
  im=2; jm=2; iter=1; dt=1.; density=2.65; hz=1.52e-6
  sigma=(1000.d0*density-1000.d0)/1000.d0
  excess_density=1000.*(density-1.)
  bagnold_density_scale=(excess_density/1650.)**(-0.5)
  p_par=-2.d0/pi
  diameter=2.d0*radius
  hs=1000.
  spa=[4.25e-5,8.07e-4,5.1e-4,8.07e-4,8.07e-5,8.49e-6]/1.2e3
  spb=[1.2,1.08,0.79,0.75,0.75,0.5]
  spc=[0.23,0.21,0.11,1.06,0.1,0.1]
  grav_propn=0.d0
  nmax_diffuse_flow=20; nmax_conc_flow=200; nmax_susp_flow=1000
  read(unit_in,*) n
  do row=1,n
    read(unit_in,*,iostat=status) h_m,speed_m_s,rain_m_s,slope_in,veg_percent,median_m,mode,fractions
    if(status/=0) stop 1
    slope=slope_in; d=h_m*1000.d0; v=speed_m_s*1000.d0
    r2=rain_m_s*1000.d0; veg=veg_percent; d50=median_m
    sed_propn=0.d0; sed_propn(:,2,2)=fractions
    detach_soil=0.d0; depos_soil=0.d0; sed_temp=0.d0; v_soil=0.d0
    raindrop_detach_tot=0.d0; flow_detach_tot=0.d0
    KE_model_type=mode
    call raindrop_detachment
    rain_rate=detach_soil(:,2,2)*density ! legacy mm/s * g/cm3 -> kg/m2/s
    ustar=sqrt(9.81d-3*d(1,2,2)*slope_in)
    call flow_detachment
    flow_rate=detach_soil(:,2,2)*density
    do phi=1,6
      last_distance=-1.d0; v_soil(phi,2,2)=-1.d0
      call diffuse_flow_transport
      distances(phi,1)=last_distance; speeds(phi,1)=v_soil(phi,2,2)*1.d-3
      last_distance=-1.d0; v_soil(phi,2,2)=-1.d0
      call conc_flow_transport
      distances(phi,2)=last_distance; speeds(phi,2)=v_soil(phi,2,2)*1.d-3
      last_distance=-1.d0; v_soil(phi,2,2)=-1.d0
      call suspended_transport
      distances(phi,3)=last_distance; speeds(phi,3)=v_soil(phi,2,2)*1.d-3
    enddo
    write(unit_out,'(*(ES25.16E3,1X))') rain_rate,flow_rate,distances,speeds
  enddo
  write(unit_out,'(A)') 'SYRUP_SEDIMENT_PHYSICS_COMPLETE'
  close(unit_in); close(unit_out)
end program
subroutine flow_distrib(distance,nsteps)
  use capture_distance
  implicit none
  double precision, intent(in) :: distance
  integer, intent(in) :: nsteps
  last_distance=distance
end subroutine
