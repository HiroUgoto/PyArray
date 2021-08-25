import numpy as np
import scipy.signal

def detrend(data):
    v = scipy.signal.detrend(data)
    return v

def cos_taper(data):
    taper_data = []
    for line in data:
        w = scipy.signal.tukey(len(line),0.5)
        taper_data += [w*line]
    return np.array(taper_data)

def rms(data):
    r = np.mean(data**2,axis=-1)
    return r

def fourier_spectrum(data):
    F = np.fft.fft(data)
    return F

def smoothing(fourier_data,nw):
    if nw%2 == 0:
        nwt = nw + 1
    else:
        nwt = nw

    nw2 = int((nwt-1)/2)
    w = scipy.signal.parzen(nwt)

    if fourier_data.ndim == 1:
        line = fourier_data
        a = np.r_[line[nw2:0:-1],line,line[0],line[-1:-nw2:-1]]
        smooth_data = np.convolve(w/w.sum(),a,mode='valid')
    else:
        smooth_data = []
        for line in fourier_data:
            a = np.r_[line[nw2:0:-1],line,line[0],line[-1:-nw2:-1]]
            smooth_data += [np.convolve(w/w.sum(),a,mode='valid')]

    return np.array(smooth_data)
