import random
from typing import List

import numpy as np
from scipy.signal import sosfiltfilt, welch

from .config import FS_EXTRACT, RESP_BAND, HR_BAND, BP_ORDER, GLOBAL_LAG_CLIP, SEED


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)

def butter_sos(lo, hi, fs, order=BP_ORDER):
    nyq = fs*0.5
    lo2 = max(1e-3, lo/nyq); hi2 = min(0.999, hi/nyq)
    if hi2 <= lo2 + 1e-3: return None
    from scipy.signal import butter
    return butter(order, [lo2, hi2], btype='bandpass', output='sos')

def bandpass(x, fs, band):
    if x is None or fs is None: return x
    x = np.asarray(x, np.float32).ravel()
    sos = butter_sos(band[0], band[1], fs)
    if sos is None: return x
    padlen = 3 * sos.shape[0]
    if x.size <= padlen: return x
    try:
        return sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return x

def zscore(x):
    x = np.asarray(x, np.float32)
    m = float(np.mean(x)); s = float(np.std(x))
    s = 1.0 if (not np.isfinite(s) or s < 1e-6) else s
    return (x - m) / s

def resample_uniform(t, x, fs):
    t = np.asarray(t, np.float32); x = np.asarray(x, np.float32)
    if t.size < 3: return None, None, None
    t_u = np.arange(t[0], t[-1] + 1e-6, 1.0/fs, dtype=np.float32)
    x_u = np.interp(t_u, t, x).astype(np.float32)
    return t_u, x_u, fs

def gcc_phat_linear(x, y, fs):
    x = np.asarray(x, np.float32).ravel(); y = np.asarray(y, np.float32).ravel()
    n = int(len(x)+len(y)-1); nfft=1
    while nfft < n: nfft <<= 1
    X = np.fft.rfft(x, nfft); Y = np.fft.rfft(y, nfft)
    R = X*np.conj(Y); R /= (np.abs(R)+1e-12)
    cc = np.fft.irfft(R, nfft)
    lags = np.arange(-len(y)+1, len(x))
    cc = np.concatenate([cc[-(len(y)-1):], cc[:len(x)]])
    i = int(np.argmax(cc))
    return float(lags[i]/fs)

def estimate_rr_bpm(sig, fs):
    sig = zscore(bandpass(sig, fs, RESP_BAND))
    if sig is None or len(sig) < int(fs*6): return np.nan
    nper = max(128, int(fs*8)); nover = nper//2
    f, P = welch(sig, fs=fs, nperseg=nper, noverlap=nover)
    mb = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    if np.count_nonzero(mb) < 3: return np.nan
    fb = f[mb]; f0 = float(fb[np.argmax(P[mb])])
    return f0*60.0

def estimate_hr_bpm(sig, fs):
    sig = zscore(bandpass(sig, fs, HR_BAND))
    if sig is None or len(sig) < int(fs*4): return np.nan
    nper = max(128, int(fs*4)); nover = nper//2
    f, P = welch(sig, fs=fs, nperseg=nper, noverlap=nover)
    mb = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    if np.count_nonzero(mb) < 3: return np.nan
    fb = f[mb]; f0 = float(fb[np.argmax(P[mb])])
    return f0*60.0

def align_common_time(tu, Xlist, tg, g, fs):
    t0 = max(float(tu[0]), float(tg[0])); t1 = min(float(tu[-1]), float(tg[-1]))
    if t1 <= t0 + (1.0/fs)*4: return tu, Xlist, g
    tC = np.arange(t0, t1+1e-6, 1.0/fs, dtype=np.float32)
    Xc = [np.interp(tC, tu, x).astype(np.float32) for x in Xlist]
    gc = np.interp(tC, tg, g).astype(np.float32)
    return tC, Xc, gc

def subject_split(subjects: List[int], ratios=(0.7,0.15,0.15), seed=SEED):
    subs = sorted(set(subjects))
    rng = np.random.RandomState(seed); rng.shuffle(subs)
    n = len(subs); n_tr=int(n*ratios[0]); n_va=int(n*ratios[1])
    return set(subs[:n_tr]), set(subs[n_tr:n_tr+n_va]), set(subs[n_tr+n_va:])

def estimate_global_lag(ts, dW, dY, dD, gt_t, gt_resp, fs=FS_EXTRACT) -> float:
    tu, wU, _ = resample_uniform(ts, dW, fs)
    _,  yU, _ = resample_uniform(ts, dY, fs)
    _,  dU, _ = resample_uniform(ts, dD, fs)
    if tu is None: return 0.0
    cU = (zscore(bandpass(wU, fs, RESP_BAND)) +
          zscore(bandpass(yU, fs, RESP_BAND)) +
          zscore(bandpass(dU, fs, RESP_BAND))) / 3.0
    tg, gU, _ = resample_uniform(gt_t, gt_resp, fs)
    if tg is None: return 0.0
    gU = zscore(bandpass(gU, fs, RESP_BAND))
    try:
        lag = float(np.clip(gcc_phat_linear(cU, gU, fs), -GLOBAL_LAG_CLIP, GLOBAL_LAG_CLIP))
    except Exception:
        lag = 0.0
    return lag
