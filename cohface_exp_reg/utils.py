# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch


def zscore(x, eps=1e-8):
    x = np.asarray(x)
    mu = np.nanmean(x)
    sd = np.nanstd(x) + eps
    return (x - mu) / sd

def butter_bandpass(x, fs, lo, hi, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def butter_lowpass(x, fs, fc, order=4):
    b, a = butter(order, fc/(fs/2), btype='low')
    return filtfilt(b, a, x)

def env_rr(x, fs, lo=0.08, hi=0.60):
    # RR 대역 통과 후 Hilbert magnitude → z
    x_bp = butter_bandpass(x, fs, lo, hi)
    mag = np.abs(hilbert(x_bp))
    return zscore(mag)

def rr_bandpass_z(x, fs, lo=0.08, hi=0.60):
    return zscore(butter_bandpass(x, fs, lo, hi))

def rr_subband_env(x, fs, lo, hi):
    x_bp = butter_bandpass(x, fs, lo, hi)
    mag = np.abs(hilbert(x_bp))
    return zscore(mag)

def welch_psd_rr_bpm(x, fs, lo=0.08, hi=0.60):
    # 대역제한 PSD 피크 주파수를 bpm으로
    nper = int(8*fs)  # 안정적 추정을 위해 기본 8초
    if nper > len(x): nper = max(256, len(x)//2)
    f, pxx = welch(x, fs=fs, nperseg=nper, window='hann', noverlap=nper//2)
    mask = (f>=lo) & (f<=hi)
    if not np.any(mask):
        return np.nan
    f_sel = f[mask]
    p_sel = pxx[mask]
    idx = np.nanargmax(p_sel)
    return f_sel[idx] * 60.0

def hit_rate_bpm(pred, gt, tol_bpm=2.0, fs=256, lo=0.08, hi=0.60):
    # pred/gt는 파형; 윈도우 전체로 하나의 bpm 추정
    pb = welch_psd_rr_bpm(pred, fs, lo, hi)
    gb = welch_psd_rr_bpm(gt, fs, lo, hi)
    if np.isnan(pb) or np.isnan(gb):
        return 0.0
    return float(abs(pb-gb) <= tol_bpm)

# ---------------- corr@soft-best-lag (numpy reference) ----------------
def corr_soft_bestlag(pred, gt, fs=256, lag_s=0.5, beta=6.0, mask=None, eps=1e-8):
    """
    pred, gt: 1D arrays (same length after mask)
    mask: optional boolean mask
    1) 정규화 교차상관을 ±lag 내에서 계산
    2) softmax(beta·ncc) 가중합 → soft-best-lag 상관값, 기대 지연(초) 반환
    """
    x = np.asarray(pred).astype(np.float32)
    y = np.asarray(gt).astype(np.float32)
    if mask is not None:
        m = (mask.astype(bool))
        x = x[m]; y = y[m]
    L = len(x)
    if L < 8:
        return 0.0, 0.0
    x = (x - x.mean())/(x.std()+eps)
    y = (y - y.mean())/(y.std()+eps)

    maxlag = int(round(lag_s*fs))
    lags = np.arange(-maxlag, maxlag+1)
    ncc = []
    for k in lags:
        if k < 0:
            xs = x[:k]
            ys = y[-k:]
        elif k > 0:
            xs = x[k:]
            ys = y[:-k]
        else:
            xs = x; ys = y
        if len(xs) < 8:
            ncc.append(0.0); continue
        c = np.dot(xs, ys) / (len(xs)+eps)
        ncc.append(c)
    ncc = np.array(ncc, dtype=np.float32)
    # softmax
    ex = np.exp(beta*(ncc - np.max(ncc)))
    w = ex / (np.sum(ex)+eps)
    corr_soft = float(np.sum(w*ncc))
    lag_exp  = float(np.sum(w*lags)/fs)
    return corr_soft, lag_exp
