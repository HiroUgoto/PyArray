import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings


#--------------------------------------------------------#
def set_T_matrix(vs,rho,thick,h,omega):
    T = np.zeros((2,2),dtype=np.complex128)

    cmp0 = np.sqrt((1 + 2j*h[0])) 
    cmp1 = np.sqrt((1 + 2j*h[1])) 

    r = (vs[0]*rho[0]*cmp0) / (vs[1]*rho[1]*cmp1)

    ep = np.exp( 1j*omega*thick/(vs[0]*cmp0))
    em = np.exp(-1j*omega*thick/(vs[0]*cmp1))
    cp = 0.5*(1+r)
    cm = 0.5*(1-r)

    T[0,0],T[0,1] = cp*ep, cm*em
    T[1,0],T[1,1] = cm*ep, cp*em

    return T

#--------------------------------------------------------#
def haskell_matrix(nlay,vs,rho,thick,h,omega):
    R = np.identity(2,dtype=np.complex128)

    if np.isscalar(h):
        for i in range(0,nlay-1):
            T = set_T_matrix(vs[i:i+2],rho[i:i+2],thick[i],([h,h]),omega)
            R = np.matmul(T,R)
    else:
        for i in range(0,nlay-1):
            T = set_T_matrix(vs[i:i+2],rho[i:i+2],thick[i],h[i:i+2],omega)
            R = np.matmul(T,R)

    return R

#--------------------------------------------------------#
def calc_resp(nlay,vs,rho,thick,h,omega):
    R = haskell_matrix(nlay,vs,rho,thick,h,omega)
    resp = 2.0 / (R[0,0]+R[0,1])
    return resp

#--------------------------------------------------------#
def calc_transfer_function_py(nlay,vs,rho,thick,h=0.0005,fmax=20,nfreq=500,abs=True,print_flag=False,plot_flag=False):
    freq = np.linspace(fmax/nfreq,fmax,nfreq)
    resp = np.zeros(nfreq,dtype=np.complex128)

    if print_flag:
        print("------------------------------------")

    for i,f in enumerate(freq):
        omega = 2*np.pi*f
        resp[i] = calc_resp(nlay,vs,rho,thick,h,omega)

        if print_flag:
            print("  {0:.2f} Hz, {1:.2f}".format(f,np.abs(resp[i])))

    if plot_flag:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # ax1.set_yscale("log")

        ax1.set_xlim([0,fmax])
        ax1.plot(freq,np.abs(resp),color='k',lw=1)

        ax1.set_ylabel("transfer function")
        ax1.set_xlabel("frequency [Hz]")

        plt.subplots_adjust()
        plt.show()

    if abs:
        return freq,np.abs(resp)
    else:   
        return freq,resp