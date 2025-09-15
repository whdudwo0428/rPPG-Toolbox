# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch, find_peaks

from .config import FS_MODEL, BPM_MIN_PROM


# 안전 z-score
def _finite_fill(x):
    x = np.asarray(x, np.float32)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, np.float32)
    idx = np.where(np.isfinite(x))[0]
    x[:idx[0]] = x[idx[0]]
    x[idx[-1] + 1:] = x[idx[-1]]
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x


def butter_bandpass(x, fs, lo, hi, order=4):
    x = _finite_fill(x)
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(x) <= padlen + 1:
        return np.zeros_like(x, np.float32)
    y = filtfilt(b, a, x)
    y[~np.isfinite(y)] = 0.0
    y = np.clip(y, -1e3, 1e3).astype(np.float32)  # ▼ 클리핑 후 캐스팅
    return y


def butter_lowpass(x, fs, fc, order=4):
    x = _finite_fill(x)
    b, a = butter(order, fc / (fs / 2), btype="low")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(x) <= padlen + 1:
        return np.zeros_like(x, np.float32)
    y = filtfilt(b, a, x)
    y[~np.isfinite(y)] = 0.0
    y = np.clip(y, -1e3, 1e3).astype(np.float32)  # ▼ 동일
    return y


def zscore(x, eps=1e-6):
    x = np.asarray(x, np.float32)
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros_like(x, np.float32)
    mu = np.nanmean(x[mask])
    sd = np.nanstd(x[mask])
    if not np.isfinite(sd) or sd < eps:
        y = x - mu  # ▼ 분산이 너무 작으면 ‘평균만 제거’로 대체
    else:
        y = (x - mu) / (sd + eps)
    y[~np.isfinite(y)] = 0.0
    return y.astype(np.float32)


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


def welch_psd_rr_bpm(x, fs=FS_MODEL, band=(0.08, 0.60), min_prom=None):
    """
    RR 대역 PSD의 1st peak → bpm
    - detrend='constant', nperseg는 2의 거듭제곱으로 자동
    - min_prom: prominence 임계; 기본 config.BPM_MIN_PROM
    """
    if min_prom is None: min_prom = BPM_MIN_PROM
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all() or x.size < 16:
        return np.nan

    # Welch-PSD 계산
    n = int(2 ** int(np.ceil(np.log2(max(256, min(8192, len(x)))))))
    f, Pxx = welch(x - np.mean(x), fs=fs, nperseg=n // 2, noverlap=n // 4, detrend='constant')
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return np.nan
    p = Pxx[m]
    fr = f[m]
    if not np.isfinite(p).all():
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 상대 prominence 모드: median 기준 정규화 후 prominence 적용 ---
    if 0.0 < float(min_prom) < 1.0:
        base = float(np.median(p) + 1e-6)
        p_norm = (p - base) / base
        peaks, _ = find_peaks(p_norm, prominence=float(min_prom))
        if peaks.size == 0:
            return np.nan
        f0 = fr[peaks[np.argmax(p_norm[peaks])]]
        return float(f0 * 60.0)

    # --- 절대 prominence 모드(현행 동작) ---
    peaks, _ = find_peaks(p, prominence=float(min_prom))  # 현재 구현 그대로  :contentReference[oaicite:6]{index=6}
    if peaks.size == 0:
        return np.nan
    f0 = fr[peaks[np.argmax(p[peaks])]]
    return float(f0 * 60.0)


def hit_rate_bpm(pred, gt, tol_bpm=2.0, fs=256, lo=0.08, hi=0.60):
    # pred/gt는 파형; 윈도우 전체로 하나의 bpm 추정
    pb = welch_psd_rr_bpm(pred, fs, lo, hi)
    gb = welch_psd_rr_bpm(gt, fs, lo, hi)
    if np.isnan(pb) or np.isnan(gb):
        return 0.0
    return float(abs(pb - gb) <= tol_bpm)


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
        x = x[m]
        y = y[m]
    L = len(x)
    if L < 8:
        return 0.0, 0.0
    x = (x - x.mean()) / (x.std() + eps)
    y = (y - y.mean()) / (y.std() + eps)

    maxlag = int(round(lag_s * fs))
    lags = np.arange(-maxlag, maxlag + 1)
    ncc = []
    for k in lags:
        if k < 0:
            xs = x[:k]
            ys = y[-k:]
        elif k > 0:
            xs = x[k:]
            ys = y[:-k]
        else:
            xs = x
            ys = y
        if len(xs) < 8:
            ncc.append(0.0)
            continue
        c = np.dot(xs, ys) / (len(xs) + eps)
        ncc.append(c)
    ncc = np.array(ncc, dtype=np.float32)
    # softmax
    ex = np.exp(beta * (ncc - np.max(ncc)))
    w = ex / (np.sum(ex) + eps)
    corr_soft = float(np.sum(w * ncc))
    lag_exp = float(np.sum(w * lags) / fs)
    return corr_soft, lag_exp
