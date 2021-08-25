import numpy as np
import scipy.optimize
from . import rayleigh_fortran
import matplotlib.pyplot as plt
import sys

#--------------------------------------------------------#
# Aki and Richards (2002); Eq.(7.45)                     #
#--------------------------------------------------------#
def propagator_matrix(vp,vs,rho,h,omega,k):
    nu2 = k**2 - (omega/vs)**2
    mu = rho * vs*vs
    ro2 = rho * omega**2

    Pg = np.zeros((4,4),dtype=np.complex_)
    Pn = np.zeros((4,4),dtype=np.complex_)
    P  = np.zeros((4,4),dtype=np.complex_)

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
# Aki and Richards (2002); Eq.(7.56)                     #
#--------------------------------------------------------#
def set_F_matrix(vp,vs,rho,omega,k):
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
    F = np.empty((4,4),dtype=np.complex_)

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
def total_propagator_matrix(nlay,vp,vs,rho,thick,omega,k):
    P = np.identity(4,dtype=np.complex_)

    for i in range(0,nlay-1):
        P0 = propagator_matrix(vp[i],vs[i],rho[i],thick[i],omega,k)
        P = np.matmul(P0,P)

    F = set_F_matrix(vp[nlay-1],vs[nlay-1],rho[nlay-1],omega,k)
    P = np.matmul(F,P)

    return P

#--------------------------------------------------------#
def set_B_matrix(nlay,vp,vs,rho,thick,omega,k):
    P = total_propagator_matrix(nlay,vp,vs,rho,thick,omega,k)

    B = np.empty((2,2),dtype=np.complex_)
    B[0,0] = P[2,0]
    B[0,1] = P[2,1]
    B[1,0] = P[3,0]
    B[1,1] = P[3,1]
    return B

#--------------------------------------------------------#
def set_nonlinear_function(k,nlay,vp,vs,rho,thick,omega):
#    B = set_B_matrix(nlay,vp,vs,rho,thick,omega,k)
    B = rayleigh_fortran.set_b_matrix(vp,vs,rho,thick,omega,k,nlay)
    det = B[0,0]*B[1,1] - B[0,1]*B[1,0]

    return np.abs(det)*np.sign(np.real(det))

#--------------------------------------------------------#
def root_finding(func,xmin,xmax,args):
    xc = np.linspace(xmin,xmax,200)

    x0 = xmin
    f0 = func(xmin,*args)
    for x in xc[1:]:
        f = func(x,*args)
        if f0*f < 0.0:
#            xp = scipy.optimize.newton(func,0.5*(x0+x),args=args)
            xp = scipy.optimize.bisect(func,x,x0,args=args)
            return xp
        else:
            x0 = x
    return xmax

#--------------------------------------------------------#
def search_fundamental_mode(nlay,vp,vs,rho,thick,fmax=20,nfreq=100,print_flag=False,plot_flag=False):
    freq = np.linspace(fmax/nfreq,fmax,nfreq)
    c,hv = rayleigh_fortran.search_fondamantal_mode(vp,vs,rho,thick,freq,nlay)

    # cmax = np.max(vs)*0.999
    # cmin = np.min(vs)*0.7
    # c = np.zeros_like(freq)
    # hv = np.zeros_like(freq)

    # if print_flag:
    #     print("------------------------------------")
    #
    # omega = 2*np.pi*freq[0]
    # kmax = omega/cmax
    # kmin = omega/cmin
    # for i,f in enumerate(freq):
    #     omega = 2*np.pi*f
    #     k = root_finding(set_nonlinear_function,kmin,kmax,args=(nlay,vp,vs,rho,thick,omega))
    #     B = set_B_matrix(nlay,vp,vs,rho,thick,omega,k)
    #     c[i] = np.real(omega/k)
    #     hv[i] = np.abs(B[0,1]) / np.abs(B[0,0])
    #     if print_flag:
    #         print("  {:4.2f}[Hz]  {:4.0f}[m/s]".format(f,c[i]))
    #     kmax = omega/(c[i]*1.0)
    #     kmin = omega/cmin


    if plot_flag:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax2.set_yscale("log")

        ax1.plot(freq,c,color='k',lw=1)
        ax2.plot(freq,hv,color='k',lw=1)

        ax1.set_ylabel("phase velocity [m/s]")
        ax2.set_ylabel("ellipse ratio")
        ax2.set_xlabel("frequency [Hz]")

        plt.subplots_adjust()
        plt.show()

    return freq,c,hv

#--------------------------------------------------------#
def search_fundamental_mode_simple(nlay,vp,vs,rho,thick,freq):
    c,hv = rayleigh_fortran.search_fondamantal_mode_simple(vp,vs,rho,thick,freq,nlay)

    return c,hv
