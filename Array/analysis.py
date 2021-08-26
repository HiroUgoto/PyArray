import numpy as np
import matplotlib.pyplot as plt
import scipy.special, scipy.optimize
from . import signal
from . import rayleigh
import sys

#-----------------------------------------------------------------#
def segment_selection(param,data,print_flag=True):
    fs = param["sampling_frequency"]
    tseg = param["segment_length"]
    mseg = param["max_number_segment"]

    nt = int(tseg*fs)
    nseg = len(data[0])//nt
    ns = len(data)

    v_list = []
    rms_stack = np.zeros(2*nseg)

    for d in data:
        v_raw = np.resize(d,[nseg,nt]).copy()
        v_raw2 = np.resize(np.roll(d,int(nt/2)),[nseg,nt]).copy()
        v = signal.detrend(np.vstack((v_raw,v_raw2)))
        v_list += [signal.cos_taper(v)]
        rms_stack += signal.rms(v)

    nseg = nseg*2
    total_rms = rms_stack.mean()/ns
    rms_index = np.argsort(rms_stack)

    total_rms = rms_stack.mean()/ns
    selected_rms = rms_stack[rms_index[0:mseg]].mean()/ns

    segment_data = []
    for v in v_list:
        sv = v[rms_index[0:mseg],:].copy()
        segment_data += [sv]

    if print_flag:
        print("------------------------------------")
        print("segment selection")
        print("+ selected segment :",mseg,"/",nseg)
        print("+ rms ratio        :",selected_rms,"/",total_rms)
        print("------------------------------------")

    return segment_data

def segment_selection_3d(param,data,print_flag=True):
    fs = param["sampling_frequency"]
    tseg = param["segment_length"]
    mseg = param["max_number_segment"]

    ud,ns,ew = data[0]

    nt = int(tseg*fs)
    nseg = len(ud)//nt
    ns = 3

    v_list = []
    rms_stack = np.zeros(2*nseg)

    for d in [ud,ns,ew]:
        v_raw = np.resize(d,[nseg,nt]).copy()
        v_raw2 = np.resize(np.roll(d,int(nt/2)),[nseg,nt]).copy()
        v = signal.detrend(np.vstack((v_raw,v_raw2)))
        v_list += [signal.cos_taper(v)]
        rms_stack += signal.rms(v)

    nseg = nseg*2
    total_rms = rms_stack.mean()/ns
    rms_index = np.argsort(rms_stack)

    total_rms = rms_stack.mean()/ns
    selected_rms = rms_stack[rms_index[0:mseg]].mean()/ns

    segment_data = []
    for v in v_list:
        sv = v[rms_index[0:mseg],:].copy()
        segment_data += [sv]

    if print_flag:
        print("------------------------------------")
        print("segment selection")
        print("+ selected segment :",mseg,"/",nseg)
        print("+ rms ratio        :",selected_rms,"/",total_rms)
        print("------------------------------------")

    return segment_data

#-----------------------------------------------------------------#
def spac_coeff(param,segment_data,plot_flag=True):
    fs = param["sampling_frequency"]
    ns = len(segment_data)

    nt = len(segment_data[0][0,:])
    nyq = int(nt/2)
    freq = np.fft.fftfreq(nt,d=1/fs)

    df = freq[1]-freq[0]
    nw = int(param["band_width"]/df)

    X = []
    for v in segment_data:
        F = signal.fourier_spectrum(v)
        X += [signal.smoothing(F,nw)]

    S00 = np.mean(np.abs(np.conj(X[0])*X[0]),axis=0)

    rho = np.zeros(nt)
    for Xc in X[1:]:
        Srr = np.mean(np.abs(np.conj(Xc)*Xc),axis=0)
        S0r = np.mean(np.conj(X[0])*Xc,axis=0)
        rho += np.real(S0r)/np.sqrt(S00*Srr)
    rho = signal.smoothing(rho/(ns-1),nw)

    if plot_flag:
        plt.figure()
        plt.xlabel("frequency (Hz)")
        plt.ylabel("SPAC coefficient")
        plt.plot(freq[0:nyq],rho[0:nyq])
        plt.grid()
        plt.show()

    return freq[0:nyq],rho[0:nyq]

#-----------------------------------------------------------------#
def cca_coeff(param,segment_data,plot_flag=True,print_flag=True):
    fs = param["sampling_frequency"]
    mseg = param["max_number_segment"]
    ns = len(segment_data)

    nt = len(segment_data[0][0,:])
    nyq = int(nt/2)
    freq = np.fft.fftfreq(nt,d=1/fs)

    df = freq[1]-freq[0]
    nw = int(param["band_width"]/df)

    # ----- CCA coefficient ---- #
    k = ((ns-1)-1)//2
    k_list = np.arange(-k,k+1)
    Phy = np.zeros((len(k_list),len(k_list)),dtype=np.complex_)
    for i,s in enumerate(param["site"][1:]):
        theta_rad = -s["theta"]/180*np.pi
        Phy[i,:] = np.exp((0-1j)*k_list*theta_rad) / (2*np.pi)
    PhyH = np.transpose(np.conj(Phy))
    Phy_inv = np.dot(np.linalg.inv(np.dot(PhyH,Phy)),PhyH)

    X = []
    for v in segment_data:
        F = signal.fourier_spectrum(v)
        X += [signal.smoothing(F,nw)]
    S00 = np.mean(np.abs(np.conj(X[0])*X[0]),axis=0)

    n0 = np.where(k_list==0)[0][0]
    n1 = np.where(k_list==1)[0][0]
    Z0 = np.zeros((mseg,nt),dtype=np.complex_)
    Z1 = np.zeros((mseg,nt),dtype=np.complex_)
    for i,Xc in enumerate(X[1:]):
        Z0 += Xc * Phy_inv[n0,i]
        Z1 += Xc * Phy_inv[n1,i]

    G00 = np.mean(np.abs(np.conj(Z0)*Z0),axis=0)
    G11 = np.mean(np.abs(np.conj(Z1)*Z1),axis=0)
    coeff = signal.smoothing(G00/G11,nw)

    # ----- NS ratio ---- #
    S0r = np.zeros(nt,dtype=np.complex_)
    Srr = np.zeros(nt)
    for Xc in X[1:]:
        Srr += np.mean(np.abs(np.conj(Xc)*Xc),axis=0)
        S0r += np.mean(np.conj(X[0])*Xc,axis=0)
    Srr = Srr/(ns-1)
    S0r = S0r/(ns-1)

    coh = np.abs(S0r)/np.sqrt(S00*Srr)
    rho = np.abs(S0r)/S00

    A = -rho**2
    B = rho**2/coh**2 - 2*rho**2 - 1/(ns-1)
    C = rho**2*(1/coh**2 - 1)
    ns_ratio = signal.smoothing((-B-np.sqrt(B**2-2*A*C))/(2*A),nw)

    if plot_flag:
        plt.figure()
        plt.plot(freq[0:nyq],coeff[0:nyq])
        plt.show()

        plt.yscale("log")
        plt.plot(freq[0:nyq],ns_ratio[0:nyq])
        plt.show()

    freq_max = freq[np.amin(np.where(ns_ratio[0:nyq]>=1.0))]
    freq_min = freq[np.argmin(ns_ratio[0:nyq])]
    param["freq_max"] = freq_max
    if print_flag:
        print("Resolve frequency range")
        print("+ frequency gives minimum NSratio [Hz]    :",freq_min)
        print("+ frequency not exceeding NSratio 1.0 [Hz]:",freq_max)
        print("------------------------------------")

    return freq[0:nyq],coeff[0:nyq],ns_ratio[0:nyq],freq_min

#-----------------------------------------------------------------#
def spac_phase_velocity(param,freq,spac_coeff,fmin=0.2,fmax=20.0,plot_flag=True):
    def spac_function(kr,coeff):
        return scipy.special.jv(0,kr) - coeff

    r = param["r"]
    freq_spac = []
    velocity_spac = []

    for f,coeff in zip(freq,spac_coeff):
        if f <= fmin:
            continue
        if f <= fmax:
            if spac_function(0,coeff)*spac_function(2.405,coeff) >= 0:
                print("+ maximum frequency for SPAC [Hz]:",f)
                break
            kr = scipy.optimize.brentq(spac_function,0,2.405,args=(coeff,))
            c = (2*np.pi*f*r)/kr
            freq_spac += [f]
            velocity_spac += [c]
        else:
            break

    if plot_flag:
        plt.figure()
        plt.xlabel("frequency (Hz)")
        plt.ylabel("phase velocity (m/s)")
        plt.plot(freq_spac,velocity_spac)
        plt.grid()
        plt.show()

    return np.array(freq_spac), np.array(velocity_spac)

#-----------------------------------------------------------------#
def cca_phase_velocity(param,freq,cca_coeff,fmax=20.0,plot_flag=True):
    def cca_function(kr,coeff):
        return (scipy.special.jv(0,kr))**2 - coeff*(scipy.special.jv(1,kr))**2

    r = param["r"]
    freq_cca = []
    velocity_cca = []

    for f,coeff in zip(freq,cca_coeff):
        if f <= fmax:
            if cca_function(0,coeff)*cca_function(2.405,coeff) >= 0:
                print("+ maximum frequency for CCA [Hz]:",f)
                break
            kr = scipy.optimize.brentq(cca_function,0,2.405,args=(coeff,))
            c = (2*np.pi*f*r)/kr
            freq_cca += [f]
            velocity_cca += [c]
        else:
            break

    if plot_flag:
        plt.figure()
        plt.plot(freq_cca,velocity_cca)
        plt.show()

    return np.array(freq_cca), np.array(velocity_cca)

#-----------------------------------------------------------------#
def cca_eps_phase_velocity(param,freq,cca_coeff,ns_ratio,fmax=20.0,plot_flag=True):
    def cca_function(kr,coeff,eps):
        return ((scipy.special.jv(0,kr))**2 + eps) - coeff*((scipy.special.jv(1,kr))**2 + eps)

    r = param["r"]
    ns = len(param["site"])-1

    freq_cca = []
    velocity_cca = []

    for f,coeff,eps in zip(freq,cca_coeff,ns_ratio):
        if f <= fmax:
            if cca_function(0,coeff,eps/ns)*cca_function(2.405,coeff,eps/ns) >= 0:
                print("+ maximum frequency for CCA [Hz]:",f)
                break
            kr = scipy.optimize.brentq(cca_function,0,2.405,args=(coeff,eps/ns))
            c = (2*np.pi*f*r)/kr
            freq_cca += [f]
            velocity_cca += [c]
        else:
            break

    if plot_flag:
        plt.figure()
        plt.plot(freq_cca,velocity_cca)
        plt.show()

    return np.array(freq_cca), np.array(velocity_cca)

#-----------------------------------------------------------------#
def connect_phase_velocity(freq_list,pv_list,freq_range,plot_flag=True):
    n = len(freq_range)
    for i in range(0,n):
        id0 = np.abs(freq_list[i] - freq_range[i][0]).argmin()
        id1 = np.abs(freq_list[i] - freq_range[i][1]).argmin()
        if i == 0:
            freq = freq_list[i][id0:id1]
            vel = pv_list[i][id0:id1]
        else:
            freq = np.append(freq,freq_list[i][id0:id1])
            vel = np.append(vel,pv_list[i][id0:id1])

    ind = np.argsort(freq)
    freq_con = freq[ind]
    vel_con = vel[ind]

    if plot_flag:
        plt.figure()
        for i in range(0,n):
            plt.plot(freq_list[i],pv_list[i],color='k',lw=1)
        plt.plot(freq_con,vel_con,color='r',lw=2)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("phase velocity (m/s)")
        plt.grid()
        plt.show()

    return freq_con, vel_con

#-----------------------------------------------------------------#
def model_phase_velocity_py(model,fmax=20,print_flag=False,plot_flag=False):
    freq,c,hv = rayleigh.search_fundamental_mode_py(model["nlay"],model["vp"],model["vs"],model["density"],model["thick"],fmax=fmax,print_flag=print_flag,plot_flag=plot_flag)

    return freq, c, hv

#-----------------------------------------------------------------#
def model_phase_velocity(model,fmax=20,print_flag=False,plot_flag=False):
    freq,c,hv = rayleigh.search_fundamental_mode(model["nlay"],model["vp"],model["vs"],model["density"],model["thick"],fmax=fmax,print_flag=print_flag,plot_flag=plot_flag)

    return freq, c, hv

#-----------------------------------------------------------------#
def compare_phase_velocity(freq0,vel0,freq1,vel1):
        plt.figure()
        plt.xlabel("frequency (Hz)")
        plt.ylabel("phase_velocity (m/s)")
        plt.scatter(freq0,vel0,marker='.',color='k')
        plt.plot(freq1,vel1,color='r')
        plt.grid()
        plt.show()


#-----------------------------------------------------------------#
def hv_spactra(param,segment_data,plot_flag=True):
    fs = param["sampling_frequency"]
    ns = len(segment_data)

    nt = len(segment_data[0][0,:])
    nyq = int(nt/2)
    freq = np.fft.fftfreq(nt,d=1/fs)

    df = freq[1]-freq[0]
    nw = int(param["band_width"]/df)

    X = []
    for v in segment_data:
        F = signal.fourier_spectrum(v)
        X += [signal.smoothing(F,nw)]

    UD = np.mean(np.abs(np.conj(X[0])*X[0]),axis=0)
    H1 = np.mean(np.abs(np.conj(X[1])*X[1]),axis=0)
    H2 = np.mean(np.abs(np.conj(X[2])*X[2]),axis=0)

    hv = signal.smoothing(np.sqrt(H1+H2)/np.sqrt(UD),nw)

    if plot_flag:
        plt.figure()
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(freq[0:nyq],hv[0:nyq])
        plt.show()

    return freq[0:nyq],hv[0:nyq]
