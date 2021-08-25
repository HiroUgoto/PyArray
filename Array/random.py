import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from . import io

#-----------------------------------------------------------------#
def von_Karman_1D(L,dx,a=1.0,std=0.1,kappa=0.5):
    def psdf1d(eps2,a,kappa,omega):
        g0 = scipy.special.gamma(kappa)
        g1 = scipy.special.gamma(kappa+0.5)
        return 2*np.sqrt(np.pi)*g1*eps2*a / (g0*(1+(a*omega)**2))**(kappa+0.5)

    eps = std

    k = math.ceil(math.log2(L/dx))
    n = 2**k
    nl = int(L//dx)

    x = np.linspace(0,n*dx,num=n,endpoint=False)
    m = 2*np.pi*x/(n*dx)

    psd = psdf1d(eps**2,a/dx,kappa,m[1:n//2+1]) / n
    phase = np.random.uniform(0,2*np.pi,n//2)

    cp = np.zeros(n,dtype=np.complex_)
    cp[1:n//2+1] = np.sqrt(psd)*np.exp(1j*phase)
    cp[n//2:] = np.conj(cp[n//2:0:-1])

    y = np.fft.fft(cp)
    return x[:nl],np.real(y[:nl])

#-----------------------------------------------------------------#
def model_random(model,bottom=1000,a=5.0,dx=1.0,std=0.1,plot_flag=False):
    nlay,vs,vp,rho,thick = io.parse_model(model)

    depth = 0
    vs_new,vp_new,rho_new,thick_new = np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0)
    for i in range(nlay-1):
        depth += thick[i]
        x,r1 = von_Karman_1D(thick[i],dx,a=a,std=std)
        x,r2 = von_Karman_1D(thick[i],dx,a=a,std=std)
        x,r3 = von_Karman_1D(thick[i],dx,a=a,std=std)

        n = len(x)
        thick_new = np.append(thick_new,np.ones(n)*thick[i]/n)
        vs_new = np.append(vs_new,vs[i]*(np.ones(n) + r1))
        vp_new = np.append(vp_new,vp[i]*(np.ones(n) + r2))
        rho_new = np.append(rho_new,rho[i]*(np.ones(n) + r3))

    thick_bottom = bottom-depth
    x,r1 = von_Karman_1D(thick_bottom,dx,a=a,std=std)
    x,r2 = von_Karman_1D(thick_bottom,dx,a=a,std=std)
    x,r3 = von_Karman_1D(thick_bottom,dx,a=a,std=std)

    n = len(x)
    thick_new = np.append(thick_new,np.ones(n)*thick_bottom/n)
    vs_new = np.append(vs_new,vs[-1]*(np.ones(n) + r1))
    vp_new = np.append(vp_new,vp[-1]*(np.ones(n) + r2))
    rho_new = np.append(rho_new,rho[-1]*(np.ones(n) + r3))
    thick_new = np.delete(thick_new,-1)

    depth = np.ones(1) * thick_new[0]
    for t in thick_new[1:]:
        depth = np.append(depth,depth[-1]+t)

    if plot_flag:
        plt.figure()
        plt.ylim(depth[-1],0)
        plt.plot(vs_new,depth,lw=1,color='b')
        plt.plot(vp_new,depth,lw=1,color='r')
        plt.plot(rho_new,depth,lw=1,color='g')
        plt.show()

    return {"nlay":len(vs_new), "vs":vs_new, "vp":vp_new, "density":rho_new, "thick":thick_new, "depth":depth}
