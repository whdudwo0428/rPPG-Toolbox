# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch, find_peaks

from .config import FS_MODEL, BPM_MIN_PROM, RESP_BAND


def _finite_fill(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, np.float32)
    idx = np.where(np.isfinite(x))[0]
    x[:idx[0]] = x[idx[0]]
    x[idx[-1] + 1:] = x[idx[-1]]
    bad = ~np.isfinite(x)
    if np.any(bad):
        x[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], x[~bad])
    return x


def zscore(x, eps=1e-6):
    x = _finite_fill(np.asarray(x, np.float32))
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < eps:
        y = x - mu
    else:
        y = (x - mu) / (sd + eps)
    y[~np.isfinite(y)] = 0.0
    return y.astype(np.float32)


# ---------------- Filters ----------------

def butter_bandpass(x, fs, lo, hi, order=4):
    x = _finite_fill(np.asarray(x, np.float32))
    lo = max(lo, 1e-5)
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    y = filtfilt(b, a, x, method='gust')
    y[~np.isfinite(y)] = 0.0
    return y.astype(np.float32)


def butter_lowpass(x, fs, fc, order=4):
    x = _finite_fill(np.asarray(x, np.float32))
    nyq = 0.5 * fs
    b, a = butter(order, fc / nyq, btype='low')
    y = filtfilt(b, a, x, method='gust')
    y[~np.isfinite(y)] = 0.0
    return y.astype(np.float32)


def rr_bandpass_z(x, fs, lo=RESP_BAND[0], hi=RESP_BAND[1]):
    return zscore(butter_bandpass(x, fs, lo, hi))


def env_rr(x, fs, lo=RESP_BAND[0], hi=RESP_BAND[1]):
    x_bp = butter_bandpass(x, fs, lo, hi)
    mag = np.abs(hilbert(x_bp))
    return zscore(mag)


def rr_subband_env(x, fs, lo, hi):
    x_bp = butter_bandpass(x, fs, lo, hi)
    mag = np.abs(hilbert(x_bp))
    return zscore(mag)


# ---------------- BPM / metrics ----------------
def welch_psd_rr_bpm(x, fs=FS_MODEL, lo=RESP_BAND[0], hi=RESP_BAND[1], min_prom=BPM_MIN_PROM):
    x = _finite_fill(np.asarray(x, np.float32))
    if len(x) < fs * 5:  # too short
        return float('nan')
    fr, p = welch(x, fs=fs, nperseg=min(1024, max(64, 2 ** int(np.floor(np.log2(len(x)))))))
    m = (fr >= lo) & (fr <= hi)
    if not np.any(m):
        return float('nan')
    fr = fr[m]
    p = p[m]

    # absolute prominence (current default)
    peaks, props = find_peaks(p, prominence=float(min_prom))
    if peaks.size == 0:
        return float('nan')
    f0 = fr[peaks[np.argmax(p[peaks])]]
    return float(f0 * 60.0)


def hit_rate_bpm(pred, gt, tol_bpm=2.0, fs=FS_MODEL, lo=RESP_BAND[0], hi=RESP_BAND[1]):
    pb = welch_psd_rr_bpm(pred, fs, lo, hi)
    gb = welch_psd_rr_bpm(gt, fs, lo, hi)
    if np.isnan(pb) or np.isnan(gb):
        return 0.0
    return float(abs(pb - gb) <= tol_bpm)


# ---------------- corr@soft-best-lag ----------------
def corr_soft_bestlag(pred, gt, fs=FS_MODEL, lag_s=0.5, beta=8.0, eps=1e-8):
    x = _finite_fill(np.asarray(pred, np.float32))
    y = _finite_fill(np.asarray(gt, np.float32))
    L = min(len(x), len(y))
    if L < 16:
        return 0.0, 0.0
    x = x[:L]
    y = y[:L]
    max_lag = int(round(lag_s * fs))
    if max_lag < 1:
        c = np.corrcoef(x, y)[0, 1]
        c = float(0.0 if not np.isfinite(c) else c)
        return c, 0.0

    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int32)
    ncc = []
    for l in lags:
        if l < 0:
            xs = x[:l]
            ys = y[-l:]
        elif l > 0:
            xs = x[l:]
            ys = y[:-l]
        else:
            xs = x
            ys = y
        if len(xs) < 8:
            ncc.append(0.0)
            continue
        c = np.dot(xs, ys) / (len(xs) + eps)
        ncc.append(c)
    ncc = np.asarray(ncc, np.float32)
    ex = np.exp(beta * (ncc - np.max(ncc)))
    w = ex / (np.sum(ex) + eps)
    corr_soft = float(np.sum(w * ncc))
    lag_exp = float(np.sum(w * lags) / fs)
    return corr_soft, lag_exp


# ---------------- Alignment helpers ----------------
def align_scale_np(pred, gt, eps=1e-6):
    """Return scaled prediction and a_hat so that pred_aligned ≈ a_hat * pred best-matches gt."""
    p = _finite_fill(np.asarray(pred, np.float32))
    g = _finite_fill(np.asarray(gt, np.float32))
    den = float(np.dot(p, p) + eps)
    num = float(np.dot(p, g))
    a_hat = num / den if den > 0 else 0.0
    return (a_hat * p).astype(np.float32), float(a_hat)


def global_sign_and_lag(ref, sig, fs=FS_MODEL, max_lag_s=4.0):
    """Estimate sign and lag to align sig to ref (use cross-correlation in RR band)."""
    ref = rr_bandpass_z(ref, fs)
    sig = rr_bandpass_z(sig, fs)
    L = min(len(ref), len(sig))
    ref = ref[:L]
    sig = sig[:L]
    max_lag = int(round(max_lag_s * fs))
    lags = np.arange(-max_lag, max_lag + 1)
    best_c = -1e9
    best_l = 0
    best_s = 1.0
    for s in (1.0, -1.0):
        xs = s * sig
        # quick NCC via direct dots
        for l in lags:
            if l < 0:
                a = xs[:l]
                b = ref[-l:]
            elif l > 0:
                a = xs[l:]
                b = ref[:-l]
            else:
                a = xs
                b = ref
            if len(a) < 32:
                continue
            c = float(np.dot(a, b) / (len(a) + 1e-6))
            if c > best_c:
                best_c, best_l, best_s = c, l, s
    return best_s, int(best_l)
