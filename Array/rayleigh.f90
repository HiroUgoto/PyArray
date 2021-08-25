program rayleigh
  implicit none
  real(8), allocatable :: vp(:), vs(:), rho(:), thick(:), depth(:)
  real(8), allocatable :: co(:), hv(:)
  real(8) :: freq, omega, c, fmax, pi
  real(8) :: hv_peak_freq
  integer :: nlay, nfreq
  integer :: ifreq, inum, i
  character(10) :: dum
  character(2) :: cnum

  open(11, file = "input.dat")

  read(11,*) nlay
  allocate(vp(nlay),vs(nlay),rho(nlay),thick(nlay),depth(nlay))

  do i = 1, nlay-1
     read(11,*) vs(i), vp(i), rho(i), depth(i)
  end do
  read(11,*) vs(nlay), vp(nlay), rho(nlay)

  close(11)

  thick(1) = depth(1)
  do i = 2, nlay-1
     thick(i) = depth(i) - depth(i-1)
  end do



  open(21, file = "result/phase_velocity.dat")
  open(22, file = "result/hv.dat")

  fmax = 10.d0
  nfreq = 200

  allocate(co(nfreq))
  allocate(hv(nfreq))

  pi = 2.d0*acos(0.d0)
  c = vs(nlay)

  call search_fondamantal_mode_omega(nlay,vp,vs,rho,thick,fmax,nfreq,co,hv,hv_peak_freq)

  do ifreq = 1, nfreq
     freq = fmax * dble(ifreq)/dble(nfreq)
     write(21,*) freq, co(ifreq)
     write(22,*) freq, hv(ifreq)
  end do
  close(21)
  close(22)

end program rayleigh


!-----------------------------------------------------------------!
subroutine search_fondamantal_mode_omega(nlay,vp,vs,rho,thick,fmax,nfreq,co,hv,hv_peak_freq)
  implicit none
  integer, intent(in) :: nlay, nfreq
  real(8), intent(in) :: vp(nlay), vs(nlay), rho(nlay), thick(nlay-1)
  real(8), intent(out) :: co(nfreq), hv(nfreq), hv_peak_freq
  real(8), intent(in) :: fmax
  complex(8) :: B(2,2)
  real(8), allocatable :: copt(:), freq(:), hvopt(:)
  real(8) :: detB
  real(8) :: hvt, hv_max
  real(8) :: omega, k, f
  real(8) :: c, cmax, cmin, vs_max, vs_min, dc
  real(8) :: cmax2, cmin2, dc2
  real(8) :: pi
  real(8) :: detBopt
  integer :: nc, ic, i, ic2
  integer :: nf, ifreq


  vs_max = vs(nlay)
  vs_min = vs(1)

  do i = 1, nlay
     if(vs_max < vs(i)) vs_max = vs(i)
     if(vs_min > vs(i)) vs_min = vs(i)
  end do

  cmax = vs_max * 0.999d0
  cmin = vs_min * 0.70d0

  nf = nfreq

  nc = 5000
  pi = 2.d0*acos(0.d0)

  allocate(copt(nf),freq(nf),hvopt(nf))

  hv_max = 0.d0
  do ifreq = 1, nfreq
     freq(ifreq) = fmax * dble(ifreq)/dble(nfreq)
     omega = 2.d0*pi*freq(ifreq)
!     write(*,*) freq(ifreq)

     dc =  (cmax-cmin)/dble(nc)

     do ic = 1, nc
        c = cmin + dc*dble(ic-1)
        k = omega / c

        call set_B_matrix(nlay,vp,vs,rho,thick,omega,k,B)
        detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)
        hvt = abs(B(1,2)) / abs(B(1,1))

        if(ic == 1) then
           copt(ifreq) = c
           detBopt = detB
           hvopt(ifreq) = hvt
        else
           if(detBopt * detB < 0.d0) then
              cmin2 = cmin + dc*dble(ic-2)
              cmax2 = cmin + dc*dble(ic-1)
              dc2 = (cmax2-cmin2)/dble(10)

              do ic2 = 1, 10
                 c = cmin2 + dc2*dble(ic2)
                 k = omega / c

                 call set_B_matrix(nlay,vp,vs,rho,thick,omega,k,B)
                 detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)
                 hvt = abs(B(1,2)) / abs(B(1,1))

!                 write(*,*) ic2, c, detBopt, detB

                 if(detBopt * detB < 0.d0) then
                    c = c - dc2
                    call set_B_matrix(nlay,vp,vs,rho,thick,omega,k,B)
                    detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)
                    hvt = abs(B(1,2)) / abs(B(1,1))

!                    write(*,*) c, detBopt, detB
                    copt(ifreq) = c
                    hvopt(ifreq) = hvt
                    exit
                 end if
              end do
              exit
           end if
        end if
     end do

     write(*,*) freq(ifreq), copt(ifreq), hvopt(ifreq)

     if(hvopt(ifreq) > hv_max) then
        hv_max = hvopt(ifreq)
        hv_peak_freq = freq(ifreq)
     end if
  end do


  write(*,*) hv_peak_freq, hv_max

  do ifreq = 1, nfreq
     co(ifreq) = copt(ifreq)
     hv(ifreq) = hvopt(ifreq)
  end do


end subroutine search_fondamantal_mode_omega

!-----------------------------------------------------------------!
subroutine search_fondamantal_mode_simple(nlay,vp,vs,rho,thick,nfreq,freq,co,hv)
  use ieee_arithmetic
  implicit none
  integer, intent(in) :: nlay, nfreq
  real(8), intent(in) :: vp(nlay), vs(nlay), rho(nlay), thick(nlay-1)
  real(8), intent(in) :: freq(nfreq)
  real(8), intent(out) :: co(nfreq), hv(nfreq)
  complex(8) :: B(2,2)
  real(8), allocatable :: copt(:), hvopt(:)
  real(8) :: detB
  real(8) :: omega, k, f
  real(8) :: c, cmax, cmin, vs_max, vs_min, vp_min, dc
  real(8) :: cmax2, cmin2, dc2
  real(8) :: pi
  real(8) :: detBopt
  integer :: nc, ic, i, ic2, ic_flag
  integer :: nf, ifreq

  ! vs_max = vs(nlay)
  ! vs_min = vs(1)
  vs_max = maxval(vs)
  vs_min = minval(vs)

  ! do i = 1, nlay
  !    if(vs_max < vs(i)) vs_max = vs(i)
  !    if(vs_min > vs(i)) vs_min = vs(i)
  ! end do

  cmax = vs_max * 0.999d0
  cmin = vs_min * 0.90d0

  nf = nfreq

  nc = 5000
  pi = 2.d0*acos(0.d0)

  allocate(copt(nf),hvopt(nf))

  do ifreq = 1, nfreq
     omega = 2.d0*pi*freq(ifreq)
     dc =  (cmax-cmin)/dble(nc)

     ic_flag = 0
     do ic = 1, nc
        c = cmin + dc*dble(ic-1)
        k = omega / c

        call set_B_matrix(nlay,vp,vs,rho,thick,omega,k,B)
        detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)

        if(ic_flag == 0) then
          if(ieee_is_finite(detB)) then
            ic_flag = 1
            copt(ifreq) = c
            detBopt = detB
!            write(*,*) detBopt
          end if
        else
           if(detBopt * detB < 0.d0) then
              cmin2 = cmin + dc*dble(ic-2)
              cmax2 = cmin + dc*dble(ic-1)
              dc2 = (cmax2-cmin2)/dble(10)

!              write(*,*) freq(ifreq), cmin2, cmax2

              do ic2 = 1, 10
                 c = cmin2 + dc2*dble(ic2)
                 k = omega / c

                 call set_B_matrix(nlay,vp,vs,rho,thick,omega,k,B)
                 detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)

                 if(detBopt * detB < 0.d0) then
                    c = c - dc2
                    call set_B_matrix(nlay,vp,vs,rho,thick,omega,k,B)
                    detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)
                    hvopt(ifreq) = abs(B(1,2)) / abs(B(1,1))
                    copt(ifreq) = c
                    exit
                 end if
              end do
              exit
           end if
        end if
     end do

!     write(*,*) freq(ifreq), copt(ifreq)
  end do

  do ifreq = 1, nfreq
     co(ifreq) = copt(ifreq)
     hv(ifreq) = hvopt(ifreq)
  end do


end subroutine search_fondamantal_mode_simple

!-----------------------------------------------------------------!
subroutine search_fondamantal_mode(nlay,vp,vs,rho,thick,nfreq,freq,co,hv)
  use ieee_arithmetic
  implicit none
  integer, intent(in) :: nlay, nfreq
  real(8), intent(in) :: vp(nlay), vs(nlay), rho(nlay), thick(nlay-1)
  real(8), intent(in) :: freq(nfreq)
  real(8), intent(out) :: co(nfreq), hv(nfreq)
  complex(8) :: B(2,2)
  real(8), allocatable :: copt(:), hvopt(:)
  real(8) :: detB
  real(8) :: omega, k, f
  real(8) :: c, cmax, cmin, vs_max, vs_min, vp_min, dc
  real(8) :: cmax2, cmin2, dc2
  real(8) :: pi
  real(8) :: detBopt
  integer :: nc, ic, i, ic2, ic_flag
  integer :: nf, ifreq
  integer :: npart, ip


  vs_max = vs(nlay)
  vs_min = vs(1)

  do i = 1, nlay
     if(vs_max < vs(i)) vs_max = vs(i)
     if(vs_min > vs(i)) vs_min = vs(i)
  end do

  cmax = vs_max * 0.999d0
  cmin = vs_min * 0.90d0

  nf = nfreq

  nc = 5000
  pi = 2.d0*acos(0.d0)

  allocate(copt(nf),hvopt(nf))

  do ifreq = 1, nfreq
     omega = 2.d0*pi*freq(ifreq)
     dc =  (cmax-cmin)/dble(nc)

     ic_flag = 0
     npart = nlay
     do ic = 1, nc
        c = cmin + dc*dble(ic-1)
        k = omega / c

        call set_B_matrix_partial(nlay,npart,vp,vs,rho,thick,omega,k,B)
        detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)

        if(ic_flag == 0) then
          if(ieee_is_finite(detB)) then
            ic_flag = 1
            copt(ifreq) = c
            detBopt = detB
!            write(*,*) c,detBopt
          else
            do ip = npart-1, 2, -1
              call set_B_matrix_partial(nlay,ip,vp,vs,rho,thick,omega,k,B)
              detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)
              if(ieee_is_finite(detB)) then
                ic_flag = 1
                copt(ifreq) = c
                detBopt = detB
                npart = ip
!                write(*,*) npart,c,detBopt
                exit
              end if
            end do
          end if
        else
           if(detBopt * detB < 0.d0) then
              cmin2 = cmin + dc*dble(ic-2)
              cmax2 = cmin + dc*dble(ic-1)
              dc2 = (cmax2-cmin2)/dble(10)

!              write(*,*) freq(ifreq), cmin2, cmax2

              do ic2 = 1, 10
                 c = cmin2 + dc2*dble(ic2)
                 k = omega / c

                 call set_B_matrix_partial(nlay,npart,vp,vs,rho,thick,omega,k,B)
                 detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)

                 if(detBopt * detB < 0.d0) then
                    c = c - dc2
                    call set_B_matrix_partial(nlay,npart,vp,vs,rho,thick,omega,k,B)
                    detB = B(1,1)*B(2,2) - B(1,2)*B(2,1)
                    hvopt(ifreq) = abs(B(1,2)) / abs(B(1,1))
                    copt(ifreq) = c
                    exit
                 end if
              end do
              exit
           end if
        end if
     end do
!     write(*,*) freq(ifreq), copt(ifreq)
  end do

  do ifreq = 1, nfreq
     co(ifreq) = copt(ifreq)
     hv(ifreq) = hvopt(ifreq)
  end do

end subroutine search_fondamantal_mode


!-----------------------------------------------------------------!
subroutine set_B_matrix(nlay,vp,vs,rho,thick,omega,k,B)
  implicit none
  integer, intent(in) :: nlay
  real(8), intent(in) :: vp(nlay), vs(nlay), rho(nlay), thick(nlay-1)
  real(8), intent(in) :: omega, k
  complex(8), intent(out) :: B(2,2)
  real(8) :: P(4,4)
  complex(8) :: F(4,4), Ball(4,4)
  real(8) :: Bmax


  call total_propagator_matrix(nlay,vp,vs,rho,thick,omega,k,P)
  call set_F_matrix(vp(nlay),vs(nlay),rho(nlay),omega,k,F)

  Ball = matmul(F,P)

  B(1,1) = Ball(3,1)
  B(1,2) = Ball(3,2)
  B(2,1) = Ball(4,1)
  B(2,2) = Ball(4,2)

  Bmax = abs(B(1,1))
  Bmax = max(Bmax,abs(B(1,2)))
  Bmax = max(Bmax,abs(B(2,1)))
  Bmax = max(Bmax,abs(B(2,2)))

  B = B/Bmax

end subroutine set_B_matrix

!-----------------------------------------------------------------!
subroutine set_B_matrix_partial(nlay,n,vp,vs,rho,thick,omega,k,B)
  implicit none
  integer, intent(in) :: nlay, n
  real(8), intent(in) :: vp(nlay), vs(nlay), rho(nlay), thick(nlay-1)
  real(8), intent(in) :: omega, k
  complex(8), intent(out) :: B(2,2)
  real(8) :: P(4,4)
  complex(8) :: F(4,4), Ball(4,4)
  real(8) :: Bmax


  call total_propagator_matrix(n,vp(1:n),vs(1:n),rho(1:n),thick(1:n-1),omega,k,P)
  call set_F_matrix(vp(n),vs(n),rho(n),omega,k,F)

  Ball = matmul(F,P)

  B(1,1) = Ball(3,1)
  B(1,2) = Ball(3,2)
  B(2,1) = Ball(4,1)
  B(2,2) = Ball(4,2)

  Bmax = abs(B(1,1))
  Bmax = max(Bmax,abs(B(1,2)))
  Bmax = max(Bmax,abs(B(2,1)))
  Bmax = max(Bmax,abs(B(2,2)))

  B = B/Bmax

end subroutine set_B_matrix_partial

!-----------------------------------------------------------------!
!--------------------------------------------------------!
! Aki and Richards (2002); Eq.(7.56)                     !
!--------------------------------------------------------!
subroutine set_F_matrix(a,b,rho,omega,k,F)
  implicit none
  real(8), intent(in) :: a, b, rho, omega, k
  complex(8), intent(out) :: F(4,4)
  complex(8) :: coeff
  complex(8) :: gamma, nu, im
  real(8) :: mu

  im = cmplx(0.d0,1.d0)
  mu = rho * b*b

  if(k >= omega/a) then
     gamma = sqrt(k**2 - (omega/a)**2)
  else
     gamma = -im*sqrt((omega/a)**2 - k**2)
  end if

  if(k >= omega/b) then
     nu = sqrt(k**2 - (omega/b)**2)
  else
     nu = -im*sqrt((omega/b)**2 - k**2)
  end if

  coeff = b / (2.d0*a*mu*gamma*nu*omega)

  F(1,1) = 2.d0*b*mu*k*gamma*nu
  F(1,2) = -b*mu*nu*(k**2 + nu**2)
  F(1,3) = -b*k*nu
  F(1,4) = b*gamma*nu

  F(2,1) = -a*mu*gamma*(k**2 + nu**2)
  F(2,2) = 2.d0*a*mu*k*gamma*nu
  F(2,3) = a*gamma*nu
  F(2,4) = -a*k*gamma

  F(3,1) = 2.d0*b*mu*k*gamma*nu
  F(3,2) = b*mu*nu*(k**2 + nu**2)
  F(3,3) = b*k*nu
  F(3,4) = b*gamma*nu

  F(4,1) = -a*mu*gamma*(k**2 + nu**2)
  F(4,2) = -2.d0*a*mu*k*gamma*nu
  F(4,3) = -a*gamma*nu
  F(4,4) = -a*k*gamma

  F = coeff*F

end subroutine set_F_matrix

!-----------------------------------------------------------------!
subroutine total_propagator_matrix(nlay,vp,vs,rho,thick,omega,k,P)
  implicit none
  integer, intent(in) :: nlay
  real(8), intent(in) :: vp(nlay), vs(nlay), rho(nlay), thick(nlay-1)
  real(8), intent(in) :: omega, k
  real(8), intent(out) :: P(4,4)
  real(8) :: P0(4,4)
  integer :: il

  P = 0.d0
  P(1,1) = 1.d0
  P(2,2) = 1.d0
  P(3,3) = 1.d0
  P(4,4) = 1.d0

  do il = 1, nlay-1
     call propagator_matrix(vp(il),vs(il),rho(il),thick(il),omega,k,P0)
     P = matmul(P0,P)
  end do

end subroutine total_propagator_matrix

!-----------------------------------------------------------------!
!--------------------------------------------------------!
! Aki and Richards (2002); Eq.(7.45)                     !
!--------------------------------------------------------!
subroutine propagator_matrix(a,b,rho,h,omega,k,P)
  implicit none
  real(8), intent(in) :: a, b, rho, h, omega, k
  real(8), intent(out) :: P(4,4)
  real(8) :: Pg(4,4), Pn(4,4)
  real(8) :: mu
  real(8) :: gamma, nu, nu2, ro2

  nu2 = k**2 - (omega/b)**2
  mu = rho * b*b
  ro2 = rho * omega**2

  Pg = 0.d0
  Pn = 0.d0

  if(k >= omega/a) then
     gamma = sqrt(k**2 - (omega/a)**2)

     Pg(1,1) = 2.d0*k**2*(sinh(0.5d0*gamma*h))**2
     Pg(1,4) = (sinh(0.5d0*gamma*h))**2
     Pg(2,1) = -2.d0*gamma*sinh(gamma*h)
     Pg(2,2) = -(k**2 + nu2)*(sinh(0.5d0*gamma*h))**2
     Pg(2,4) = -gamma*sinh(gamma*h)
     Pg(3,1) = 4.d0*k**2*gamma*sinh(gamma*h)
     if(abs(gamma) < 1.d-9) then
        Pg(1,2) = (k**2+nu2) * h
        Pg(1,3) = k**2 * h
        Pg(4,2) = -(k**2 + nu2)**2 * h
     else
        Pg(1,2) = (k**2+nu2) * sinh(gamma*h) / gamma
        Pg(1,3) = k**2 * sinh(gamma*h) / gamma
        Pg(4,2) = -(k**2 + nu2)**2 * sinh(gamma*h)/gamma
     end if

  else
     gamma = -sqrt((omega/a)**2 - k**2)

     Pg(1,1) = -2.d0*k**2*(sin(0.5d0*gamma*h))**2
     Pg(1,4) = -(sin(0.5d0*gamma*h))**2
     Pg(2,1) =  2.d0*gamma*sin(gamma*h)
     Pg(2,2) = (k**2 + nu2)*(sin(0.5d0*gamma*h))**2
     Pg(2,4) = gamma*sin(gamma*h)
     Pg(3,1) = -4.d0*k**2*gamma*sin(gamma*h)
     if(abs(gamma) < 1.d-9) then
        Pg(1,2) = (k**2+nu2) * h
        Pg(1,3) = k**2 * h
        Pg(4,2) = -(k**2 + nu2)**2 * h
     else
        Pg(1,2) = (k**2+nu2) * sin(gamma*h) / gamma
        Pg(1,3) = k**2 * sin(gamma*h) / gamma
        Pg(4,2) = -(k**2 + nu2)**2 * sin(gamma*h) / gamma
     end if

  end if

  if(k >= omega/b) then
     nu = sqrt(k**2 - (omega/b)**2)

     Pn(1,1) = -(k**2 + nu2)*(sinh(0.5d0*nu*h))**2
     Pn(1,2) = -2.d0*nu*sinh(nu*h)
     Pn(1,3) = -nu*sinh(nu*h)
     Pn(1,4) = -(sinh(0.5d0*nu*h))**2
     Pn(2,2) = 2*k**2*(sinh(0.5d0*nu*h))**2
     Pn(4,2) = 4.d0*k**2*nu*sinh(nu*h)
     if(abs(nu) < 1.d-9) then
        Pn(2,1) = (k**2+nu2) * h
        Pn(2,4) = k**2 * h
        Pn(3,1) = -(k**2+nu2)**2 * h
     else
        Pn(2,1) = (k**2+nu2) * sinh(nu*h) / nu
        Pn(2,4) = k**2 * sinh(nu*h) / nu
        Pn(3,1) = -(k**2+nu2)**2 * sinh(nu*h) / nu
     end if

  else
     nu = -sqrt((omega/b)**2 - k**2)

     Pn(1,1) = (k**2 + nu2)*(sin(0.5d0*nu*h))**2
     Pn(1,2) =  2.d0*nu*sin(nu*h)
     Pn(1,3) =  nu*sin(nu*h)
     Pn(1,4) = (sin(0.5d0*nu*h))**2
     Pn(2,2) = -2*k**2*(sin(0.5d0*nu*h))**2
     Pn(4,2) = -4.d0*k**2*nu*sin(nu*h)
     if(abs(nu) < 1.d-9) then
        Pn(2,1) = (k**2+nu2) * h
        Pn(2,4) = k**2 * h
        Pn(3,1) = -(k**2+nu2)**2 * h
     else
        Pn(2,1) = (k**2+nu2) * sin(nu*h) / nu
        Pn(2,4) = k**2 * sin(nu*h) / nu
        Pn(3,1) = -(k**2+nu2)**2 * sin(nu*h) / nu
     end if

  end if

  P(1,1) = 1.d0 + 2.d0*mu/ro2 * (Pg(1,1) + Pn(1,1))
  P(3,3) = P(1,1)

  P(1,2) = k*mu/ro2 * (Pg(1,2) + Pn(1,2))
  P(4,3) = -P(1,2)

  P(1,3) = 1.d0/ro2 * (Pg(1,3) + Pn(1,3))

  P(1,4) = 2.d0*k/ro2 * (Pg(1,4) + Pn(1,4))
  P(2,3) = -P(1,4)

  P(2,1) = k*mu/ro2 * (Pg(2,1) + Pn(2,1))
  P(3,4) = -P(2,1)

  P(2,2) = 1.d0 + 2.d0*mu/ro2 * (Pg(2,2) + Pn(2,2))
  P(4,4) = P(2,2)

  P(2,4) = 1.d0/ro2 * (Pg(2,4) + Pn(2,4))

  P(3,1) = mu**2/ro2 * (Pg(3,1) + Pn(3,1))

  P(3,2) = 2.d0*mu**2*(k**2+nu2)*P(1,4)
  P(4,1) = -P(3,2)

  P(4,2) = mu**2/ro2 * (Pg(4,2) + Pn(4,2))

end subroutine propagator_matrix
