# -*- coding: utf-8 -*-
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert, welch, find_peaks, sosfiltfilt

from .config import FS_MODEL, BPM_MIN_PROM
from .config import RUNS_DIR


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


def _minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (max(hi - lo, eps))


def _minmax11(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y = _minmax01(x, eps)
    return y * 2.0 - 1.0


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
    up = max(1, int(os.getenv("BPM_NFFT_UP", "1")))  # e.g., 1/2/4
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


def corr_soft_bestlag_torch(pred: torch.Tensor, gt: torch.Tensor, fs: float,
                            lag_s: float = 2.0, beta: float = 8.0, eps: float = 1e-6):
    """
    Differentiable soft-best-lag correlation.
    pred, gt: [B,T] 또는 [T] (float/half 모두 허용, 내부에서 float32로 연산)
    반환: (corr_soft:[B], lag_soft:[B, seconds])
    """
    if pred.dim() == 1:
        pred = pred[None, :]
    if gt.dim() == 1:
        gt = gt[None, :]

    P = pred.float()
    G = gt.float()

    # z-normalize
    P = P - P.mean(dim=1, keepdim=True)
    P = P / (P.std(dim=1, keepdim=True) + eps)
    G = G - G.mean(dim=1, keepdim=True)
    G = G / (G.std(dim=1, keepdim=True) + eps)

    max_lag = int(round(float(lag_s) * float(fs)))
    if max_lag < 1:
        C = (P * G).mean(dim=1)
        return C, torch.zeros_like(C)

    vals = []
    for l in range(-max_lag, max_lag + 1):
        if l < 0:
            x = P[:, :l]
            y = G[:, -l:]
        elif l > 0:
            x = P[:, l:]
            y = G[:, :-l]
        else:
            x, y = P, G
        c = (x * y).mean(dim=1)  # [B]
        vals.append(c)

    C = torch.stack(vals, dim=1)  # [B, 2*L+1]
    Cmax = C.max(dim=1, keepdim=True).values
    W = torch.softmax(beta * (C - Cmax), dim=1)  # 안정적 softmax
    corr_soft = (W * C).sum(dim=1)  # [B]
    lag_idx = torch.arange(-max_lag, max_lag + 1, device=C.device, dtype=C.dtype)
    lag_soft = (W * lag_idx[None, :]).sum(dim=1) / float(fs)
    return corr_soft, lag_soft


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


# ==== (신규) 결과 집계 유틸 =============
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _pivot_append(csv_path: Path, run_col: str, rows: dict):
    """
    'metric'을 행 index, run명을 열로 하는 wide CSV를 유지.
    - 없으면 생성: header = ['metric', run_col]
    - 있으면 읽어서 metric 행 union 후 run_col 열을 우측에 추가
    """
    table = OrderedDict()  # metric -> {run_name: value}
    if csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            rdr = csv.reader(f)
            try:
                header = next(rdr)
            except StopIteration:
                header = ['metric']
            run_cols = header[1:]
            for r in rdr:
                metric = r[0]
                table[metric] = {c: v for c, v in zip(run_cols, r[1:])}

    # 신규 열 주입
    for k, v in rows.items():
        table.setdefault(k, {})
        table[k][run_col] = v

    # header 재구성
    all_metrics = list(table.keys())
    all_runs = set()
    for d in table.values(): all_runs.update(d.keys())
    all_runs = sorted(all_runs)

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric'] + all_runs)
        for m in all_metrics:
            w.writerow([m] + [table[m].get(r, "") for r in all_runs])


def append_exp_results(run_dir: str,
                       run_name: str,
                       metrics: dict,
                       settings: dict,
                       timings: dict,
                       by_win: dict = None):
    """
    exp_results/{run_name}.json 저장 + exp_results/summary.csv (wide pivot) 누적 갱신
    - rows(행)는 metric 이름, 열은 run_name
    - by_win: 윈도우/스트라이드별 요약(dict), CSV에는 JSON 문자열로 1셀에 기록
    """
    exp_dir = Path(RUNS_DIR) / "exp_results"
    _ensure_dir(exp_dir)

    payload = {
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "timings": timings,
        "settings": settings,
        "by_win": by_win or {},
    }
    with open(exp_dir / f"{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    flat = OrderedDict()
    # 핵심 지표
    for k in ("val_corr", "val_corr_bestlag", "val_rr_bpm_mae",
              "test_corr", "test_corr_bestlag", "test_rr_bpm_mae",
              "val_mse", "val_mae", "test_mse", "test_mae"):
        if k in metrics: flat[k] = metrics[k]
    # 동작시간
    for k in ("time_total_s", "time_data_s", "time_train_s", "time_eval_val_s", "time_eval_test_s"):
        if k in timings: flat[k] = timings[k]
    # 설정 요약
    flat["wins_stride"] = settings.get("wins_stride", "")
    flat["pad_mode"] = settings.get("pad_mode", "")
    flat["include_tail"] = settings.get("include_tail", "")
    # 윈도우별 성능/시간
    if by_win:
        flat["by_win_json"] = json.dumps(by_win, ensure_ascii=False)

    _pivot_append(exp_dir / "summary.csv", run_name, flat)
