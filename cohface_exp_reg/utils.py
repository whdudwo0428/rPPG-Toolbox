# -*- coding: utf-8 -*-
import os

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch, find_peaks, sosfiltfilt

from .config import FS_MODEL, BPM_MIN_PROM


# ---------------- Robust helpers ----------------

def _finite_fill(x: np.ndarray) -> np.ndarray:
    """Fill NaN/Inf by edge-hold + linear interp; ensure finite float64 during processing."""
    x = np.asarray(x)
    if x.dtype.kind != 'f':
        x = x.astype(np.float64, copy=False)
    else:
        x = x.astype(np.float64, copy=True)

    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=np.float32)

    idx = np.where(np.isfinite(x))[0]
    # edge hold
    x[:idx[0]] = x[idx[0]]
    x[idx[-1] + 1:] = x[idx[-1]]
    # interior linear interp
    bad = ~np.isfinite(x)
    if np.any(bad):
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    # guard final
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def robust_clip(x: np.ndarray, clip_std: float = 8.0, abs_max: float = 1e5) -> np.ndarray:
    """Clip by mean±k*std and absolute guard; return finite float32."""
    x = _finite_fill(x)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd == 0.0:
        lo, hi = mu - 10.0, mu + 10.0
    else:
        lo, hi = mu - clip_std * sd, mu + clip_std * sd
    x = np.clip(x, lo, hi)
    x = np.clip(x, -abs_max, abs_max).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def zscore(x, eps=1e-6, clip_std: float = 8.0):
    """Z-score with robust clipping to avoid overflow."""
    x = _finite_fill(x)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < eps:
        y = x - mu
    else:
        y = (x - mu) / (sd + eps)
    # clamp extreme z
    y = np.clip(y, -clip_std, clip_std)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32)


# ---------------- Filters (safe: use SOS when possible) ----------------

def butter_bandpass(x, fs, lo, hi, order=4):
    """Stable bandpass using SOS + filtfilt; fully finite in/out."""
    x = robust_clip(x, abs_max=1e5)  # restrict dynamic range before filtering
    lo = max(lo, 1e-5)
    nyq = 0.5 * fs
    wn = [lo / nyq, hi / nyq]
    try:
        sos = butter(order, wn, btype='band', output='sos')
        y = sosfiltfilt(sos, x.astype(np.float64))
    except Exception:
        # fallback to (b,a) if SOS unavailable
        b, a = butter(order, wn, btype='band')
        y = filtfilt(b, a, x.astype(np.float64), method='gust')
    y = _finite_fill(y)
    return y.astype(np.float32)


def butter_lowpass(x, fs, fc, order=4):
    x = robust_clip(x, abs_max=1e5)
    nyq = 0.5 * fs
    wn = fc / nyq
    try:
        sos = butter(order, wn, btype='low', output='sos')
        y = sosfiltfilt(sos, x.astype(np.float64))
    except Exception:
        b, a = butter(order, wn, btype='low')
        y = filtfilt(b, a, x.astype(np.float64), method='gust')
    y = _finite_fill(y)
    return y.astype(np.float32)


def rr_bandpass_z(x, fs, lo=0.08, hi=0.60):
    return zscore(butter_bandpass(x, fs, lo, hi))


def env_rr(x, fs, lo=0.08, hi=0.60):
    x_bp = butter_bandpass(x, fs, lo, hi)
    mag = np.abs(hilbert(x_bp))
    return zscore(mag)


def rr_subband_env(x, fs, lo, hi):
    x_bp = butter_bandpass(x, fs, lo, hi)
    mag = np.abs(hilbert(x_bp))
    return zscore(mag)


# ---------------- BPM / metrics ----------------

def _subbin_parabolic(fr, p, i0):
    """Parabolic (quadratic) interpolation around peak index i0 → refined freq (Hz)."""
    if not (0 < i0 < len(p) - 1):
        return float(fr[i0])
    y1, y2, y3 = p[i0 - 1], p[i0], p[i0 + 1]
    denom = (y1 - 2.0 * y2 + y3)
    if denom == 0:
        return float(fr[i0])
    delta = 0.5 * (y1 - y3) / denom  # ∈ [-0.5, 0.5] approx
    return float(fr[i0] + delta * (fr[1] - fr[0]))

def welch_psd_rr_bpm(x, fs=FS_MODEL, lo=0.08, hi=0.60, min_prom=BPM_MIN_PROM):
    """
    RR-band Welch PSD → BPM.
    - 기본: prominence>=min_prom인 피크 중 최댓값 선택
    - 피크 없으면: 대역 argmax 폴백(환경변수 BPM_FALLBACK_ARGMAX=1)
    - 분해능 개선: nfft 업샘플링(BPM_NFFT_UP, 기본 1), 서브-빈 포물선 보간(BPM_SUBBIN_QUAD=1)
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < fs * 5:  # 너무 짧으면 신뢰도 낮음
        return float('nan')

    # 세분화 옵션
    up = max(1, int(os.getenv("BPM_NFFT_UP", "1")))         # e.g., 1/2/4
    use_subbin = int(os.getenv("BPM_SUBBIN_QUAD", "1")) != 0
    use_fallback = int(os.getenv("BPM_FALLBACK_ARGMAX", "1")) != 0

    # nperseg: 파형 길이에 맞춰 64~1024의 2의 거듭제곱
    nseg = min(1024, max(64, 1 << int(np.floor(np.log2(len(x))))))
    nfft = int(nseg * up)

    # Welch
    try:
        fr, p = welch(x, fs=fs, nperseg=nseg, nfft=nfft)
    except TypeError:
        # SciPy가 nfft를 지원하지 않는 구버전 대비
        fr, p = welch(x, fs=fs, nperseg=nseg)

    m = (fr >= lo) & (fr <= hi)
    if not np.any(m):
        return float('nan')
    fr, p = fr[m], p[m]

    peaks, props = find_peaks(p, prominence=float(min_prom))
    if peaks.size > 0:
        i0 = peaks[np.argmax(p[peaks])]
        f0 = _subbin_parabolic(fr, p, i0) if use_subbin else float(fr[i0])
        return float(f0 * 60.0)

    if use_fallback and p.size > 0:
        i0 = int(np.argmax(p))
        f0 = _subbin_parabolic(fr, p, i0) if use_subbin else float(fr[i0])
        return float(f0 * 60.0)

    return float('nan')


# ---------------- corr@soft-best-lag ----------------

def corr_soft_bestlag(pred, gt, fs=FS_MODEL, lag_s=0.5, beta=8.0, eps=1e-8):
    x = zscore(pred)  # normalize to remove scale issues
    y = zscore(gt)
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
            xs, ys = x[:l], y[-l:]
        elif l > 0:
            xs, ys = x[l:], y[:-l]
        else:
            xs, ys = x, y
        if len(xs) < 8:
            ncc.append(0.0)
            continue
        xs = zscore(xs)
        ys = zscore(ys)
        c = float(np.dot(xs, ys) / (len(xs) + eps))
        if not np.isfinite(c):
            c = 0.0
        ncc.append(c)
    ncc = np.array(ncc, dtype=np.float32)
    ex = np.exp(beta * (ncc - np.max(ncc)))
    w = ex / (np.sum(ex) + eps)
    corr_soft = float(np.sum(w * ncc))
    lag_exp = float(np.sum(w * lags) / fs)
    return corr_soft, lag_exp


# ---------------- Alignment helpers ----------------
def global_sign_and_lag(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: float = 4.0, eps: float = 1e-8):
    """
    두 신호 x(특징)와 y(타깃)에 대해, 상관이 최대가 되도록 y의 전역 부호(+1/-1)와
    정수 샘플 라그(양수= y를 뒤로, 음수= y를 앞으로)를 추정한다.

    반환:
        sgn (int): +1 또는 -1
        best_lag (int): 샘플 단위 라그 (양수면 y를 +lag만큼 뒤로 시프트)
    """
    # 안전 전처리: 유한화 + z-score
    x = zscore(_finite_fill(x))
    y = zscore(_finite_fill(y))

    L = int(min(len(x), len(y)))
    if L < 16 or not np.any(np.isfinite(x)) or not np.any(np.isfinite(y)):
        return 1, 0

    max_lag = int(round(float(max_lag_s) * float(fs)))
    # 창 길이보다 과도한 라그는 제한
    max_lag = int(max(0, min(max_lag, L - 8)))

    def _best_corr_and_lag(xz, yz):
        best_c, best_l = -1e9, 0
        for l in range(-max_lag, max_lag + 1):
            if l < 0:
                xs, ys = xz[:l], yz[-l:]
            elif l > 0:
                xs, ys = xz[l:], yz[:-l]
            else:
                xs, ys = xz, yz
            if len(xs) < 8:
                continue
            # 창 내부 재정규화(수치 안정)
            xs = zscore(xs)
            ys = zscore(ys)
            c = float(np.dot(xs, ys) / (len(xs) + eps))
            if not np.isfinite(c):
                c = 0.0
            if c > best_c:
                best_c, best_l = c, l
        return best_c, best_l

    c_pos, l_pos = _best_corr_and_lag(x, y)
    c_neg, l_neg = _best_corr_and_lag(x, -y)

    if c_neg > c_pos:
        return -1, int(l_neg)
    else:
        return +1, int(l_pos)


def align_scale_np(pred, gt, eps=1e-6):
    """Return scaled prediction and a_hat so that pred_aligned ≈ a_hat * pred best-matches gt."""
    p = _finite_fill(pred).astype(np.float64)
    g = _finite_fill(gt).astype(np.float64)
    den = float(np.dot(p, p) + eps)
    num = float(np.dot(p, g))
    a_hat = num / den if den > 0 else 0.0
    y = (a_hat * p).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, float(a_hat)
