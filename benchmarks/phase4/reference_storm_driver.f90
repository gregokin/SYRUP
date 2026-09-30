! Controlled dry-initial Plot1 water-only benchmark. Original routines unchanged.
! Diagnostics below measure the original CN closure; they never change state.
! Input/row bounds and provenance are enforced by storm_reference.py.
program coupled_audit
use shared_data
use, intrinsic :: ieee_arithmetic
implicit none
integer :: au_i,au_j,au_k,au_n,au_in,au_out,au_final,au_flag,au_act
real(8) :: au_dt,au_dx,au_rate,au_rain,au_export,au_oldout,au_store0,au_store,au_drain,au_gain
real(8) :: au_c,au_closure_sum,au_closure_max,au_eps
integer :: au_bracket_count
real(8), allocatable :: au_rhs(:,:),au_closure(:,:),au_bracket(:,:),au_closure_tol(:,:)
real(8), allocatable :: au_scale(:,:)
logical, allocatable :: au_outlet(:,:),au_active(:,:)
character(len=4096) :: au_path,au_hist,au_grid
call get_command_argument(1,au_path)
call get_command_argument(2,au_hist)
call get_command_argument(3,au_grid)
open(newunit=au_in,file=trim(au_path),status='old')
read(au_in,*) nr2,nc2,ncell1,au_n,au_dt,au_dx
nr=nr2-1;nc=nc2-1;nr1=nr;nc1=nc;ncell=nr*nc
dt=real(au_dt);dx=real(au_dx);dy=dx;dtdx=dt/dx
if(dble(dt)/=au_dt.or.dble(dx)/=au_dx) error stop 'inexact default REAL dt/dx'
iroute=5;ff_type=1;inf_model=2;inf_type=2;ndirn=4;ksat_mod=1.d0;psi_mod=1.d0
allocate(order(ncell1,3),aspect(nr2,nc2),rmask(nr2,nc2),slope(nr2,nc2),ff(nr2,nc2))
allocate(d(2,nr2,nc2),q(2,nr2,nc2),qin(2,nr2,nc2),v(nr2,nc2),v1(nr2,nc2),qsum(nr2,nc2))
allocate(excess(nr2,nc2),cum_inf(nr2,nc2),cum_inf_p(nr2,nc2),cum_drain(nr2,nc2),stmax(nr2,nc2))
allocate(theta(nr2,nc2),theta_sat(nr2,nc2),ksat(nr2,nc2),psi(nr2,nc2),pave(nr2,nc2),r2(nr2,nc2))
allocate(drain_par(nr2,nc2),t_ponding(nr2,nc2),au_scale(nr2,nc2),au_outlet(nr2,nc2),au_active(nr2,nc2))
allocate(d_soil(6,2,nr2,nc2),q_soil(6,2,nr2,nc2),qsedin(6,2,nr2,nc2),v_soil(6,nr2,nc2))
allocate(au_rhs(nr2,nc2),au_closure(nr2,nc2),au_bracket(nr2,nc2),au_closure_tol(nr2,nc2))
d=0;q=0;qin=0;v=0;v1=0;qsum=0;excess=0;cum_inf_p=0;cum_drain=0;r2=0;t_ponding=-9999
d_soil=0;q_soil=0;qsedin=0;v_soil=0
 do au_k=1,ncell1
 read(au_in,*) order(au_k,:)
 enddo
 do au_i=1,nr2
 do au_j=1,nc2
 read(au_in,*) aspect(au_i,au_j),rmask(au_i,au_j),slope(au_i,au_j),ff(au_i,au_j), &
 ksat(au_i,au_j),psi(au_i,au_j),pave(au_i,au_j),drain_par(au_i,au_j), &
 theta_sat(au_i,au_j),theta(au_i,au_j),cum_inf(au_i,au_j),stmax(au_i,au_j), &
 au_scale(au_i,au_j),au_flag,au_act
 au_outlet(au_i,au_j)=au_flag==1;au_active(au_i,au_j)=au_act==1
 enddo
 enddo
au_rain=0;au_export=0;au_gain=0;au_closure_sum=0;au_closure_max=0;au_bracket_count=0
au_c=.5d0*dble(dt)/dble(dx);au_eps=epsilon(1.d0)
au_store0=(sum(d(1,:,:),mask=au_active)+sum(cum_inf,mask=au_active))*dble(dx)**2*1.d-9
open(newunit=au_out,file=trim(au_hist),status='replace')
write(au_out,'(A)') 'time_s outlet_m3_s export_m3 surface_m3 soil_m3 drain_m3 rain_m3 residual_m3 stale_gain_m3 closure_sum_m3 max_closure_m bracket_end_count'
 do iter=1,au_n
 read(au_in,*) au_rate
 r2=au_rate*au_scale
 au_rain=au_rain+sum(r2,mask=au_active)*dble(dt)*dble(dx)**2*1.d-9
 call infilt
 au_oldout=sum(q(1,:,:),mask=au_outlet)
 au_gain=au_gain+.5d0*dble(dt)*dble(dx)*(sum(qin(1,:,:),mask=au_active)- &
 sum(q(1,:,:),mask=au_active.and..not.au_outlet))*1.d-9
 call route_water
 if(.not.all(ieee_is_finite(d))) error stop 'nonfinite depth'
 if(.not.all(ieee_is_finite(q))) error stop 'nonfinite discharge'
 if(.not.all(ieee_is_finite(cum_inf))) error stop 'nonfinite soil water'
 if(.not.all(ieee_is_finite(excess))) error stop 'nonfinite rainfall excess'
 if(any(d<0.d0).or.any(q<0.d0).or.any(cum_inf<0.d0).or.any(excess<0.d0)) &
     error stop 'negative water state or flux in original routines'
 ! Original equation residual, signed in storage-depth units (mm).
 au_rhs=d(1,:,:)+excess*dble(dt)+au_c*(qin(1,:,:)+qin(2,:,:)-q(1,:,:))
 au_closure=d(2,:,:)+au_c*q(2,:,:)-au_rhs
 au_closure_sum=au_closure_sum+sum(au_closure,mask=au_active)*dble(dx)**2*1.d-9
 au_closure_max=max(au_closure_max,maxval(abs(au_closure),mask=au_active))
 au_bracket=100.d0*(d(1,:,:)+excess)
 where(au_bracket==0.d0) au_bracket=.5d0
 ! Endpoint proximity alone is ambiguous near zero. Count only material
 ! closure errors beyond the legacy 1e-8 mm root tolerance and roundoff.
 au_closure_tol=1.d-8*(1.d0+1.5d0*au_c*sqrt(78480.d0*slope/ff)*sqrt(max(d(2,:,:),0.d0))) &
     +64.d0*au_eps*max(abs(au_rhs),tiny(1.d0))
 au_bracket_count=au_bracket_count+count(au_active.and.abs(d(2,:,:)-au_bracket)<=1.d-8 &
     .and.abs(au_closure)>au_closure_tol)
 au_export=au_export+.5d0*dble(dt)*dble(dx)*(au_oldout+sum(q(2,:,:),mask=au_outlet))*1.d-9
 au_drain=sum(cum_drain,mask=au_active)*dble(dx)**2*1.d-9
 au_store=(sum(d(2,:,:),mask=au_active)+sum(cum_inf,mask=au_active))*dble(dx)**2*1.d-9
 write(au_out,'(12(ES25.16E3,1X))') dble(iter)*dble(dt), &
 sum(q(2,:,:),mask=au_outlet)*dble(dx)*1.d-9,au_export, &
 sum(d(2,:,:),mask=au_active)*dble(dx)**2*1.d-9,sum(cum_inf,mask=au_active)*dble(dx)**2*1.d-9, &
 au_drain,au_rain,au_store+au_export+au_drain-au_store0-au_rain,au_gain, &
 au_closure_sum,au_closure_max*1.d-3,dble(au_bracket_count)
 call update_water_flow
 enddo
write(au_out,'(A)') 'SYRUP_COUPLED_AUDIT_COMPLETE'
close(au_out);close(au_in)
open(newunit=au_final,file=trim(au_grid),status='replace')
 do au_i=2,nr
 do au_j=2,nc
 write(au_final,'(2(I0,1X),3(ES25.16E3,1X))') au_i,au_j,d(1,au_i,au_j)*1.d-3, &
 cum_inf(au_i,au_j)*1.d-3,q(1,au_i,au_j)*1.d-6
 enddo
 enddo
close(au_final)
end program
