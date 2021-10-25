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

def bandpass(data,fs,low,high):
    wave = detrend(data)

    dt = 1.0/fs
    w = np.fft.fft(wave)
    freq = np.fft.fftfreq(len(wave),d=dt)
    df = freq[1] - freq[0]

    nt = 10
    low0  = max(0,low - nt*df)
    high0 = high + nt*df

    w2 = np.ones_like(w)
    for (i,f) in enumerate(freq):
        if abs(f) < low0:
            w2[i] = 0.0 + 0.0j
        elif abs(f) < low:
            w2[i] = w[i] * (abs(f)-low0)/(low-low0)
        elif abs(f) <= high:
            w2[i] = w[i]
        elif abs(f) <= high0:
            w2[i] = w[i] * (abs(f)-high0)/(high-high0)
        else:
            w2[i] = 0.0 + 0.0j

    wave_int = np.real(np.fft.ifft(w2))
    return wave_int
