import numpy as np
import scipy.optimize
import scipy.integrate
# from . import rayleigh_fortran
import matplotlib.pyplot as plt
import sys
import warnings

#--------------------------------------------------------#
# Aki and Richards (2002); Eq.(7.45)                     #
#--------------------------------------------------------#
def propagator_matrix(vp,vs,rho,h,omega,k):
    nu2 = k**2 - (omega/vs)**2
    mu = rho * vs*vs
    ro2 = rho * omega**2

    # Pg = np.zeros((4,4),dtype=np.complex128)
    # Pn = np.zeros((4,4),dtype=np.complex128)
    # P  = np.zeros((4,4),dtype=np.complex128)
    Pg = np.zeros((4,4))
    Pn = np.zeros((4,4))
    P  = np.zeros((4,4))

    if k >= omega/vp:
        gamma = np.sqrt(k**2 - (omega/vp)**2)

        Pg[0,0] = 2*k**2 * (np.sinh(0.5*gamma*h))**2
        Pg[0,3] = (np.sinh(0.5*gamma*h))**2
        Pg[1,0] = -2*gamma*np.sinh(gamma*h)
        Pg[1,1] = -(k**2 + nu2)*(np.sinh(0.5*gamma*h))**2
        Pg[1,3] = -gamma*np.sinh(gamma*h)
        Pg[2,0] = 4*k**2 * gamma*np.sinh(gamma*h)
        if np.abs(gamma) < 1e-7:
            Pg[0,1] = (k**2+nu2)*h
            Pg[0,2] = k**2 * h
            Pg[3,1] = -(k**2+nu2)**2 * h
        else:
            Pg[0,1] = (k**2+nu2)* np.sinh(gamma*h)/gamma
            Pg[0,2] = k**2 * np.sinh(gamma*h)/gamma
            Pg[3,1] = -(k**2+nu2)**2 * np.sinh(gamma*h)/gamma

    else:
        gamma = -np.sqrt((omega/vp)**2 - k**2)

        Pg[0,0] = -2*k**2 * (np.sin(0.5*gamma*h))**2
        Pg[0,3] = -(np.sin(0.5*gamma*h))**2
        Pg[1,0] = 2*gamma*np.sin(gamma*h)
        Pg[1,1] = (k**2 + nu2)*(np.sin(0.5*gamma*h))**2
        Pg[1,3] = gamma*np.sin(gamma*h)
        Pg[2,0] = -4*k**2 * gamma*np.sin(gamma*h)
        if np.abs(gamma) < 1e-7:
            Pg[0,1] = (k**2+nu2)*h
            Pg[0,2] = k**2 * h
            Pg[3,1] = -(k**2+nu2)**2 * h
        else:
            Pg[0,1] = (k**2+nu2)* np.sin(gamma*h)/gamma
            Pg[0,2] = k**2 * np.sin(gamma*h)/gamma
            Pg[3,1] = -(k**2+nu2)**2 * np.sin(gamma*h)/gamma

    if k >= omega/vs:
        nu = np.sqrt(k**2 - (omega/vs)**2)

        Pn[0,0] = -(k**2 + nu2) * (np.sinh(0.5*nu*h))**2
        Pn[0,1] = -2*nu*np.sinh(nu*h)
        Pn[0,2] = -nu*np.sinh(nu*h)
        Pn[0,3] = -(np.sinh(0.5*nu*h))**2
        Pn[1,1] = 2*k**2 * (np.sinh(0.5*nu*h))**2
        Pn[3,1] = 4*k**2 * nu*np.sinh(nu*h)
        if np.abs(nu) < 1e-7:
            Pn[1,0] = (k**2+nu2)*h
            Pn[1,3] = k**2 * h
            Pn[2,0] = -(k**2+nu2)**2 * h
        else:
            Pn[1,0] = (k**2+nu2)* np.sinh(nu*h)/nu
            Pn[1,3] = k**2 * np.sinh(nu*h)/nu
            Pn[2,0] = -(k**2+nu2)**2 * np.sinh(nu*h)/nu

    else:
        nu = -np.sqrt((omega/vs)**2 - k**2)

        Pn[0,0] = (k**2 + nu2) * (np.sin(0.5*nu*h))**2
        Pn[0,1] = 2*nu*np.sin(nu*h)
        Pn[0,2] = nu*np.sin(nu*h)
        Pn[0,3] = (np.sin(0.5*nu*h))**2
        Pn[1,1] = -2*k**2 * (np.sin(0.5*nu*h))**2
        Pn[3,1] = -4*k**2 * nu*np.sin(nu*h)
        if np.abs(nu) < 1e-7:
            Pn[1,0] = (k**2+nu2)*h
            Pn[1,3] = k**2 * h
            Pn[2,0] = -(k**2+nu2)**2 * h
        else:
            Pn[1,0] = (k**2+nu2)* np.sin(nu*h)/nu
            Pn[1,3] = k**2 * np.sin(nu*h)/nu
            Pn[2,0] = -(k**2+nu2)**2 * np.sin(nu*h)/nu

    P[0,0] = 1 + 2*mu/ro2 * (Pg[0,0] + Pn[0,0])
    P[2,2] = P[0,0]

    P[0,1] = k*mu/ro2 * (Pg[0,1] + Pn[0,1])
    P[3,2] = -P[0,1]

    P[0,2] = 1/ro2 * (Pg[0,2] + Pn[0,2])

    P[0,3] = 2*k/ro2 * (Pg[0,3] + Pn[0,3])
    P[1,2] = -P[0,3]

    P[1,0] = k*mu/ro2 * (Pg[1,0] + Pn[1,0])
    P[2,3] = -P[1,0]

    P[1,1] = 1 + 2*mu/ro2 * (Pg[1,1] + Pn[1,1])
    P[3,3] = P[1,1]

    P[1,3] = 1/ro2 * (Pg[1,3] + Pn[1,3])
    P[2,0] = mu**2/ro2 * (Pg[2,0] + Pn[2,0])

    P[2,1] = 2*mu**2*(k**2+nu2)*P[0,3]
    P[3,0] = -P[2,1]

    P[3,1] = mu**2/ro2 * (Pg[3,1] + Pn[3,1])

    return P

#--------------------------------------------------------#
def propagator_matrix_d(vp,vs,rho,h,omega,k):
    nu2 = k**2 - (omega/vs)**2
    mu = rho * vs*vs
    ro2 = rho * omega**2

    # Pg = np.zeros((4,4),dtype=np.complex128)
    # Pn = np.zeros((4,4),dtype=np.complex128)
    # P  = np.zeros((4,4),dtype=np.complex128)
    Pg = np.zeros((4,4))
    Pn = np.zeros((4,4))
    P  = np.zeros((4,4))

    if k >= omega/vp:
        gamma = np.sqrt(k**2 - (omega/vp)**2)

        Pg[0,0] = 2*k**2 * 2*0.5*gamma*np.sinh(0.5*gamma*h)*np.cosh(0.5*gamma*h)
        Pg[0,3] = 2*0.5*gamma*np.sinh(0.5*gamma*h)*np.cosh(0.5*gamma*h)
        Pg[1,0] = -2*gamma * gamma*np.cosh(gamma*h)  
        Pg[1,1] = -(k**2 + nu2)* 2*0.5*gamma*np.sinh(0.5*gamma*h)*np.cosh(0.5*gamma*h)
        Pg[1,3] = -gamma * gamma*np.cosh(gamma*h)
        Pg[2,0] = 4*k**2 * gamma * gamma*np.cosh(gamma*h)

        Pg[0,1] = (k**2+nu2) * np.cosh(gamma*h)
        Pg[0,2] = k**2 * np.cosh(gamma*h)
        Pg[3,1] = -(k**2+nu2)**2 * np.cosh(gamma*h)
        
    else:
        gamma = -np.sqrt((omega/vp)**2 - k**2)

        Pg[0,0] = -2*k**2 * 2*0.5*gamma*np.sin(0.5*gamma*h)*np.cos(0.5*gamma*h)
        Pg[0,3] = -2*0.5*gamma*np.sin(0.5*gamma*h)*np.cos(0.5*gamma*h)
        Pg[1,0] = 2*gamma * gamma*np.cos(gamma*h)
        Pg[1,1] = (k**2 + nu2) * 2*0.5*gamma*np.sin(0.5*gamma*h)*np.cos(0.5*gamma*h)
        Pg[1,3] = gamma * gamma*np.cos(gamma*h)
        Pg[2,0] = -4*k**2 * gamma * gamma*np.cos(gamma*h)

        Pg[0,1] = (k**2+nu2)* np.cos(gamma*h)
        Pg[0,2] = k**2 * np.cos(gamma*h)
        Pg[3,1] = -(k**2+nu2)**2 * np.cos(gamma*h)

    if k >= omega/vs:
        nu = np.sqrt(k**2 - (omega/vs)**2)

        Pn[0,0] = -(k**2 + nu2) * 2*0.5*nu*np.sinh(0.5*nu*h)*np.cosh(0.5*nu*h)
        Pn[0,1] = -2*nu * nu*np.cosh(nu*h)
        Pn[0,2] = -nu * nu*np.cosh(nu*h)
        Pn[0,3] = -2*0.5*nu*np.sinh(0.5*nu*h)*np.cosh(0.5*nu*h)
        Pn[1,1] = 2*k**2 * 2*0.5*nu*np.sinh(0.5*nu*h)*np.cosh(0.5*nu*h)
        Pn[3,1] = 4*k**2 * nu * nu*np.cosh(nu*h)

        Pn[1,0] = (k**2+nu2)* np.cosh(nu*h)
        Pn[1,3] = k**2 * np.cosh(nu*h)
        Pn[2,0] = -(k**2+nu2)**2 * np.cosh(nu*h)

    else:
        nu = -np.sqrt((omega/vs)**2 - k**2)

        Pn[0,0] = (k**2 + nu2) * 2*0.5*nu*np.sin(0.5*nu*h)*np.cos(0.5*nu*h)
        Pn[0,1] = 2*nu * nu*np.cos(nu*h)
        Pn[0,2] = nu * nu*np.cos(nu*h)
        Pn[0,3] = 2*0.5*nu*np.sin(0.5*nu*h)*np.cos(0.5*nu*h)
        Pn[1,1] = -2*k**2 * 2*0.5*nu*np.sin(0.5*nu*h)*np.cos(0.5*nu*h)
        Pn[3,1] = -4*k**2 * nu * nu*np.cos(nu*h)

        Pn[1,0] = (k**2+nu2)* np.cos(nu*h)
        Pn[1,3] = k**2 * np.cos(nu*h)
        Pn[2,0] = -(k**2+nu2)**2 * np.cos(nu*h)

    P[0,0] = 2*mu/ro2 * (Pg[0,0] + Pn[0,0])
    P[2,2] = P[0,0]

    P[0,1] = k*mu/ro2 * (Pg[0,1] + Pn[0,1])
    P[3,2] = -P[0,1]

    P[0,2] = 1/ro2 * (Pg[0,2] + Pn[0,2])

    P[0,3] = 2*k/ro2 * (Pg[0,3] + Pn[0,3])
    P[1,2] = -P[0,3]

    P[1,0] = k*mu/ro2 * (Pg[1,0] + Pn[1,0])
    P[2,3] = -P[1,0]

    P[1,1] = 2*mu/ro2 * (Pg[1,1] + Pn[1,1])
    P[3,3] = P[1,1]

    P[1,3] = 1/ro2 * (Pg[1,3] + Pn[1,3])
    P[2,0] = mu**2/ro2 * (Pg[2,0] + Pn[2,0])

    P[2,1] = 2*mu**2*(k**2+nu2)*P[0,3]
    P[3,0] = -P[2,1]

    P[3,1] = mu**2/ro2 * (Pg[3,1] + Pn[3,1])

    return P

#--------------------------------------------------------#
# Aki and Richards (2002); Eq.(7.56)                     #
#--------------------------------------------------------#
def set_F_matrix(vp,vs,rho,omega,k):
    warnings.simplefilter('ignore')
    
    im = 0 + 1j
    mu = rho * vs*vs

    if k >= omega/vp:
        gamma = np.sqrt(k**2 - (omega/vp)**2)
    else:
        gamma = -im*np.sqrt((omega/vp)**2 - k**2)

    if k >= omega/vs:
        nu = np.sqrt(k**2 - (omega/vs)**2)
    else:
        nu = -im*np.sqrt((omega/vs)**2 - k**2)

    coeff = vs / (2*vp*mu*gamma*nu*omega)
    F = np.empty((4,4),dtype=np.complex128)

    F[0,0] = 2*vs*mu*k*gamma*nu
    F[0,1] = -vs*mu*nu*(k**2 + nu**2)
    F[0,2] = -vs*k*nu
    F[0,3] = vs*gamma*nu

    F[1,0] = -vp*mu*gamma*(k**2 + nu**2)
    F[1,1] = 2*vp*mu*k*gamma*nu
    F[1,2] = vp*gamma*nu
    F[1,3] = -vp*k*gamma

    F[2,0] = 2*vs*mu*k*gamma*nu
    F[2,1] = vs*mu*nu*(k**2 + nu**2)
    F[2,2] = vs*k*nu
    F[2,3] = vs*gamma*nu

    F[3,0] = -vp*mu*gamma*(k**2 + nu**2)
    F[3,1] = -2*vp*mu*k*gamma*nu
    F[3,2] = -vp*gamma*nu
    F[3,3] = -vp*k*gamma

    F = coeff*F
    return F

#--------------------------------------------------------#
# Aki and Richards (2002); Eq.(7.55)                     #
#--------------------------------------------------------#
def set_Fz_matrix(vp,vs,rho,z,omega,k):
    im = 0 + 1j
    mu = rho * vs*vs

    if k >= omega/vp:
        gamma = np.sqrt(k**2 - (omega/vp)**2)
    else:
        gamma = -im*np.sqrt((omega/vp)**2 - k**2)

    if k >= omega/vs:
        nu = np.sqrt(k**2 - (omega/vs)**2)
    else:
        nu = -im*np.sqrt((omega/vs)**2 - k**2)

    coeff = 1.0 / omega

    F = np.empty((4,4),dtype=np.complex128)
    F[0,0] = vp*k
    F[0,1] = vs*nu
    F[0,2] = vp*k
    F[0,3] = vs*nu

    F[1,0] = vp*gamma
    F[1,1] = vs*k
    F[1,2] = -vp*gamma
    F[1,3] = -vs*k

    F[2,0] = -2*vp*mu*k*gamma
    F[2,1] = -vs*mu*(k**2 + nu**2)
    F[2,2] = 2*vp*mu*k*gamma
    F[2,3] = vs*mu*(k**2 + nu**2)

    F[3,0] = -vp*mu*(k**2 + nu**2)
    F[3,1] = -2*vs*mu*k*nu
    F[3,2] = -vp*mu*(k**2 + nu**2)
    F[3,3] = -2*vs*mu*k*nu

    Ld = np.array([np.exp(-gamma*z),np.exp(-nu*z),np.exp(gamma*z),np.exp(nu*z)],dtype=np.complex128)
    L = np.diag(Ld)

    F = coeff*F @ L
    return F

#--------------------------------------------------------#
def set_dFz_matrix(vp,vs,rho,z,omega,k):
    im = 0 + 1j
    mu = rho * vs*vs

    if k >= omega/vp:
        gamma = np.sqrt(k**2 - (omega/vp)**2)
    else:
        gamma = -im*np.sqrt((omega/vp)**2 - k**2)

    if k >= omega/vs:
        nu = np.sqrt(k**2 - (omega/vs)**2)
    else:
        nu = -im*np.sqrt((omega/vs)**2 - k**2)

    coeff = 1.0 / omega

    F = np.empty((4,4),dtype=np.complex128)
    F[0,0] = vp*k
    F[0,1] = vs*nu
    F[0,2] = vp*k
    F[0,3] = vs*nu

    F[1,0] = vp*gamma
    F[1,1] = vs*k
    F[1,2] = -vp*gamma
    F[1,3] = -vs*k

    F[2,0] = -2*vp*mu*k*gamma
    F[2,1] = -vs*mu*(k**2 + nu**2)
    F[2,2] = 2*vp*mu*k*gamma
    F[2,3] = vs*mu*(k**2 + nu**2)

    F[3,0] = -vp*mu*(k**2 + nu**2)
    F[3,1] = -2*vs*mu*k*nu
    F[3,2] = -vp*mu*(k**2 + nu**2)
    F[3,3] = -2*vs*mu*k*nu

    Ld = np.array([-gamma*np.exp(-gamma*z),-nu*np.exp(-nu*z),gamma*np.exp(gamma*z),nu*np.exp(nu*z)],dtype=np.complex128)
    L = np.diag(Ld)

    F = coeff*F @ L
    return F

#--------------------------------------------------------#
def total_propagator_matrix(nlay,vp,vs,rho,thick,omega,k):
    P = np.identity(4,dtype=np.complex128)

    for i in range(0,nlay-1):
        P0 = propagator_matrix(vp[i],vs[i],rho[i],thick[i],omega,k)
        P = np.matmul(P0,P)

    F = set_F_matrix(vp[nlay-1],vs[nlay-1],rho[nlay-1],omega,k)
    P = np.matmul(F,P)

    return P

#--------------------------------------------------------#
def set_B_matrix(nlay,vp,vs,rho,thick,omega,k):
    P = total_propagator_matrix(nlay,vp,vs,rho,thick,omega,k)

    B = np.empty((2,2),dtype=np.complex128)
    B[0,0] = P[2,0]
    B[0,1] = P[2,1]
    B[1,0] = P[3,0]
    B[1,1] = P[3,1]
    return B

#--------------------------------------------------------#
def search_phase_velocity(nlay,vp,vs,rho,thick,omega,nmode=1,nc=1000):
    def detB_func(c,nlay,vp,vs,rho,thick,omega):
        k = omega/c
        B = set_B_matrix(nlay,vp,vs,rho,thick,omega,k)
        detB = B[0,0]*B[1,1] - B[0,1]*B[1,0]
        return detB

    vsmax = np.max(vs)

    cmax = vsmax*0.999
    cmin = np.min(vs)*0.90
    dc = (cmax-cmin)/nc

    copt = np.ones(nmode) * vsmax 
    hvopt = np.zeros(nmode)

    for im in range(nmode):
        k = omega/vsmax
        B = set_B_matrix(nlay,vp,vs,rho,thick,omega,k)
        hvopt[im] = abs(B[0,1])/abs(B[0,0])

    ic_flag = 0
    im = 0
    for ic in range(nc):
        c = cmin + dc*ic
        k = omega/c

        B = set_B_matrix(nlay,vp,vs,rho,thick,omega,k)
        detB = B[0,0]*B[1,1] - B[0,1]*B[1,0]

        if ic_flag == 0:
            if np.isfinite(detB):
                ic_flag = 1
                # copt = c
                detB_check = detB
        else:
            if detB_check * detB < 0.0:
                args = (nlay,vp,vs,rho,thick,omega)
                ans = scipy.optimize.root_scalar(detB_func,args=args,bracket=[c-dc,c],method='brentq')

                c = ans.root
                k = omega/c
                B = set_B_matrix(nlay,vp,vs,rho,thick,omega,k)

                hvopt[im] = abs(B[0,1])/abs(B[0,0])
                copt[im] = c
                
                detB_check = detB
                im += 1
                if im >= nmode:
                    break

    return copt,hvopt

#--------------------------------------------------------#
def search_fundamental_mode_py(nlay,vp,vs,rho,thick,fmax=20,nfreq=100,print_flag=False,plot_flag=False):
    freq = np.linspace(fmax/nfreq,fmax,nfreq)
    copt = np.zeros_like(freq)
    hvopt = np.zeros_like(freq)

    if print_flag:
        print("------------------------------------")

    for i,f in enumerate(freq):
        omega = 2*np.pi*f
        copt[i],hvopt[i] = search_phase_velocity(nlay,vp,vs,rho,thick,omega)

        if print_flag:
            print("  {0:.2f} Hz, {1:.1f} m/s".format(f,copt[i]))


    if plot_flag:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax2.set_yscale("log")

        ax1.set_xlim([0,fmax])
        ax2.set_xlim([0,fmax])

        ax1.plot(freq,copt,color='k',lw=1)
        ax2.plot(freq,hvopt,color='k',lw=1)

        ax1.set_ylabel("phase velocity [m/s]")
        ax2.set_ylabel("ellipse ratio")
        ax2.set_xlabel("frequency [Hz]")

        plt.subplots_adjust()
        plt.show()

    return freq,copt,hvopt

#--------------------------------------------------------#
def search_phase_velocities_py(nlay,vp,vs,rho,thick,fmax=20,nmode=2,nfreq=100,print_flag=False,plot_flag=False):
    freq = np.linspace(fmax/nfreq,fmax,nfreq)
    copt = np.zeros((nfreq,nmode))
    hvopt = np.zeros((nfreq,nmode))

    if print_flag:
        print("------------------------------------")

    for i,f in enumerate(freq):
        omega = 2*np.pi*f
        copt[i,:],hvopt[i,:] = search_phase_velocity(nlay,vp,vs,rho,thick,omega,nmode)

        if print_flag:
            print("  {0:.2f} Hz, {1:.1f} m/s".format(f,copt[i,0]))

    if plot_flag:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax2.set_yscale("log")

        ax1.set_xlim([0,fmax])
        ax2.set_xlim([0,fmax])

        for im in range(nmode):
            ax1.plot(freq,copt[:,im],color='k',lw=1)
            ax2.plot(freq,hvopt[:,im],color='k',lw=1)

        ax1.set_ylabel("phase velocity [m/s]")
        ax2.set_ylabel("ellipse ratio")
        ax2.set_xlabel("frequency [Hz]")

        plt.subplots_adjust()
        plt.show()

    return freq,copt,hvopt

#--------------------------------------------------------#
def search_fundamental_mode(nlay,vp,vs,rho,thick,fmax=20,nfreq=100,print_flag=False,plot_flag=False):
    freq = np.linspace(fmax/nfreq,fmax,nfreq)
    c,hv = rayleigh_fortran.search_fondamantal_mode(vp,vs,rho,thick,freq,nlay)

    return freq,c,hv

#--------------------------------------------------------#
def search_fundamental_mode_simple(nlay,vp,vs,rho,thick,freq):
    c,hv = rayleigh_fortran.search_fondamantal_mode_simple(vp,vs,rho,thick,freq,nlay)

    return c,hv

#####################################################################################################################
#--------------------------------------------------------#
def set_r(nlay,vp,vs,rho,thick,z,r0,omega,k):
    r = np.copy(r0)

    depth = 0.0
    for i in range(nlay-1):
        if z <= depth + thick[i]:
            P0 = propagator_matrix(vp[i],vs[i],rho[i],z-depth,omega,k)
            r = P0 @ r
            return np.real(r)
        else:
            P0 = propagator_matrix(vp[i],vs[i],rho[i],thick[i],omega,k)
            r = P0 @ r
            depth += thick[i]

    F = set_F_matrix(vp[-1],vs[-1],rho[-1],omega,k)
    w = F @ r
    w[2],w[3] = 0.0,0.0 

    Fz = set_Fz_matrix(vp[-1],vs[-1],rho[-1],z-depth,omega,k)
    r = Fz @ w

    return np.real(r)

#--------------------------------------------------------#
def set_dr(nlay,vp,vs,rho,thick,z,r0,omega,k):
    r = np.copy(r0)

    depth = 0.0
    for i in range(nlay-1):
        if z <= depth + thick[i]:
            dP0 = propagator_matrix_d(vp[i],vs[i],rho[i],z-depth,omega,k)
            r = dP0 @ r
            return np.real(r)
        else:
            P0 = propagator_matrix(vp[i],vs[i],rho[i],thick[i],omega,k)
            r = P0 @ r
            depth += thick[i]

    F = set_F_matrix(vp[-1],vs[-1],rho[-1],omega,k)
    w = F @ r
    w[2],w[3] = 0.0,0.0 

    dFz = set_dFz_matrix(vp[-1],vs[-1],rho[-1],z-depth,omega,k)
    r = dFz @ w

    return np.real(r)

#--------------------------------------------------------#
def set_r12(nlay,vp,vs,rho,thick,z,r0,omega,k):
    r = set_r(nlay,vp,vs,rho,thick,z,r0,omega,k)
    return r[0],r[1]

def set_r12_dr12(nlay,vp,vs,rho,thick,z,r0,omega,k):
    r = set_r(nlay,vp,vs,rho,thick,z,r0,omega,k)
    dr = set_dr(nlay,vp,vs,rho,thick,z,r0,omega,k)
    return r[0],r[1],dr[0],dr[1]

#--------------------------------------------------------#
def calc_group_velocity(nlay,vp,vs,rho,thick,omega,c):
    def calc_I1(z,rho_i,rlambda_i,rmu_i,nlay,vp,vs,rho,thick,r0,omega,k):
        r1,r2 = set_r12(nlay,vp,vs,rho,thick,z,r0,omega,k)
        I1 = rho_i * (r1**2 + r2**2)
        return I1

    def calc_I2(z,rho_i,rlambda_i,rmu_i,nlay,vp,vs,rho,thick,r0,omega,k):
        r1,r2 = set_r12(nlay,vp,vs,rho,thick,z,r0,omega,k)
        I2 = (rlambda_i + 2*rmu_i) * r1**2 + rmu_i * r2**2
        return I2

    def calc_I3(z,rho_i,rlambda_i,rmu_i,nlay,vp,vs,rho,thick,r0,omega,k):
        r1,r2,dr1,dr2 = set_r12_dr12(nlay,vp,vs,rho,thick,z,r0,omega,k)
        I3 = rlambda_i * r1*dr2 - rmu_i * r2*dr1
        return I3

    k = omega/c
    zmax = 2*np.pi/k * 10    # zmax = n x wave_length

    # set r0 vector 
    B = set_B_matrix(nlay,vp,vs,rho,thick,omega,k)
    r1_z0 = np.real(-0.5*(B[0,1]/B[0,0] + B[1,1]/B[1,0])) 
    r0 = np.array([r1_z0,1.0,0.0,0.0],dtype=np.complex128)

    # integrate I1, I2, I3
    I1,I2,I3 = 0.0,0.0,0.0

    depth = 0.0
    for i in range(nlay-1):
        rho_i,vs_i,vp_i = rho[i],vs[i],vp[i]
        rmu_i = rho_i * vs_i**2
        rlambda_i = rho_i * vp_i**2 - 2*rmu_i

        z0 = depth
        z1 = depth + thick[i]

        args = (rho_i,rlambda_i,rmu_i,nlay,vp,vs,rho,thick,r0,omega,k)
        I1_i,_ = scipy.integrate.quad(calc_I1,z0,z1,args=args)
        I2_i,_ = scipy.integrate.quad(calc_I2,z0,z1,args=args)
        I3_i,_ = scipy.integrate.quad(calc_I3,z0,z1,args=args)

        I1 += I1_i
        I2 += I2_i
        I3 += I3_i

        depth += thick[i]

    if zmax > depth:
        rho_i,vs_i,vp_i = rho[-1],vs[-1],vp[-1]
        rmu_i = rho_i * vs_i**2
        rlambda_i = rho_i * vp_i**2 - 2*rmu_i

        z0 = depth + 1e-5
        z1 = zmax

        args = (rho_i,rlambda_i,rmu_i,nlay,vp,vs,rho,thick,r0,omega,k)
        I1_i,_ = scipy.integrate.quad(calc_I1,z0,z1,args=args)
        I2_i,_ = scipy.integrate.quad(calc_I2,z0,z1,args=args)
        I3_i,_ = scipy.integrate.quad(calc_I3,z0,z1,args=args)

        I1 += I1_i
        I2 += I2_i
        I3 += I3_i

    I1 = I1 * 0.5
    I2 = I2 * 0.5
    U = (I2 + I3/(2.0*k)) / (c*I1)
    # print("{0:.3e}, {1:.3e}, {2:.3e}, {3:.1f}".format(I1,I2,I3,U))

    return U,I1,I2,I3

#--------------------------------------------------------#
def calc_resp(nlay,vp,vs,rho,thick,omega,c,nmode):
    U,I1,I2,I3 = calc_group_velocity(nlay,vp,vs,rho,thick,omega,c)
    resp = 1 / (4.0*omega*U*I1)
    return resp, U

#--------------------------------------------------------#
def calc_medium_response_py(nlay,vp,vs,rho,thick,fmax=20,nmode=1,nfreq=100,print_flag=False,plot_flag=False):
    freq = np.linspace(fmax/nfreq,fmax,nfreq)
    U = np.zeros((nfreq,nmode))
    resp = np.zeros((nfreq,nmode))

    if print_flag:
        print("------------------------------------")

    for i,f in enumerate(freq):
        omega = 2*np.pi*f
        c,_ = search_phase_velocity(nlay,vp,vs,rho,thick,omega,nmode)

        for im in range(nmode):
            resp[i,im],U[i,im] = calc_resp(nlay,vp,vs,rho,thick,omega,c[im],nmode)

        if print_flag:
            print("  {0:.2f} Hz, {1:.1f} m/s, {2:.2e}".format(f,U[i,0],resp[i,0]))

    if plot_flag:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        # ax2.set_yscale("log")

        ax1.set_xlim([0,fmax])
        ax2.set_xlim([0,fmax])

        for im in range(nmode):
            ax1.plot(freq,U[:,im],color='k',lw=1)
            ax2.plot(freq,resp[:,im],color='k',lw=1)

        ax1.set_ylabel("group velocity [m/s]")
        ax2.set_ylabel("medium response")
        ax2.set_xlabel("frequency [Hz]")

        plt.subplots_adjust()
        plt.show()


    return freq,resp,U
