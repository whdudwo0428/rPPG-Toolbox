# -*- coding: utf-8 -*-
"""
COHFACE (v3.0, GT-aligned, phase-corrected, per-panel lag, robust gating)

왼쪽: 비디오 + 랜드마크 + 타이머 + 지터
오른쪽: 2x2 패널 (dY, dD, dW, dC vs GT)
 - 각 패널: corr | lagcorr | dlag | RR | p
 - GT는 "원시 GT(점선)" + "위상보정된 GT(연녹 실선)" 동시 표기
핵심:
 - 전역 래그 ±1.0s (GCC-PHAT) + 미세 스윕(±0.2s, 5ms) + EMA(β=0.35)
 - 패널별 국소 래그 스윕(±0.30s)로 lagcorr 계산, dlag는 글로벌 기준 Δlag
 - 학습/추론 도메인 통일(resample→bandpass→zscore)
 - 지터 모니터링(L/R/N): 최근 구간 frame-to-frame step std(px)
 - 게이트(학습/표시): coh≥0.40 & lagcorr≥0.50 & p<0.05 & jitter<0.50px
 - dC_opt = ridge(+EMA) 가중합 → HP(0.03Hz) + soft-clip
 - polarity 1회 고정(워밍업) 후 과거 버퍼 재부호
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2, h5py, numpy as np, mediapipe as mp
from collections import deque
from scipy.signal import welch, coherence

# ---------------- Paths ----------------
VIDEO = "/mnt/hdd18t/rppg_dataset/raw/cohface/1/0/data.mkv"
H5    = "/mnt/hdd18t/rppg_dataset/raw/cohface/1/0/data.hdf5"
SHOW_GT = os.path.exists(H5)

# ---------------- View/Run Params ----------------
VIEW_W, VIEW_H = 1920, 900
LEFT_RATIO = 0.40
SPAN_SEC   = 60.0
HIS_SEC    = 180.0
PROC_W     = 640
POSE_EVERY_N = 2
BG = (16,16,16)
HUD_COL = (200,230,230)
COLORS = {  # BGR
    "dY": (230,230,230),   # white
    "dD": (120,220,120),   # light green
    "dW": (120,200,255),   # cyan
    "dC": (220,220, 80),   # yellow
    "GT": (90, 220, 90),   # corrected GT
    "GT_RAW": (60, 160, 60) # raw GT (dotted)
}

# (검출/스무딩)
EMA_BETA_LMK = 0.7
MAX_LMK_JUMP = 40.0

# (호흡 대역/필터 & 스펙트럼)
RESP_BAND      = (0.08, 0.60)  # 5–36 bpm
RESP_ORDER     = 4
RR_BAND        = (0.08, 0.60)
MIN_BAND_BINS  = 3
SPEC_WIN       = 12.0          # 모든 지표 창 12s 고정
SNR_EXCLUDE_K  = 1
SNR_FLOOR_MIN  = 1e-6

# (Z-정규화 시각화)
VIZ_MODE        = "zfix"  # zfix | zwin | raw
ZFIX_WARMUP_SEC = 25.0
Z_CLIP          = 3.0

# (학습형 결합특징)
RIDGE_ALPHA   = 1e-3
W_EMA_BETA    = 0.8

# (게이트 기준)
GATE_COH      = 0.40
GATE_LAGCORR  = 0.50
GATE_P        = 0.05
JITTER_SPIKE  = 0.50  # px (L/R/N 중 하나라도 초과하면 학습 정지)

# (Global/Local lag)
GLOBAL_LAG_SEC       = None
GLOBAL_LAG_READY_T   = 8.0
GLOBAL_LAG_TEXT      = ""
LAG_MAX_SEC          = 1.00   # v2.9: 0.50 → v3.0: 1.00
LAG_FINE_RANGE       = 0.20   # ±0.20s
LAG_FINE_STEP        = 0.002
LAG_EMA_BETA         = 0.35
PANEL_FINE_RANGE     = 0.60

# (표시)
MA_WIN = 7
TIMER_POS = (40, VIEW_H - 30)

# ---------------- State ----------------
ts = deque(); dY = deque(); dD = deque(); dW = deque(); dC = deque(); dC_opt = deque()
y0 = d0 = w0 = None
L_EMA = R_EMA = N_EMA = None
prev_L = prev_R = prev_N = None
WC = None
ZFIX = {"dY": None, "dD": None, "dW": None, "dC": None, "GT": None}
# 지터 측정용
hist_L = deque(maxlen=40); hist_R = deque(maxlen=40); hist_N = deque(maxlen=40)

# ---- Polarity fix (워밍업 1회) ----
POL_SIGN      = {"dY": None, "dD": None, "dW": None}
POL_READY_SEC = 10.0
POL_APPLIED   = False

# ---------------- Utils ----------------
def put(img, txt, xy, s=0.72, col=(230,230,230)):
    x, y = xy
    cv2.putText(img, txt, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, s, col, 1, cv2.LINE_AA)

def soft_clip_z(z, clip=3.0):
    z = np.asarray(z, np.float32)
    return clip * np.tanh(z / (clip/1.5))

def moving_average(x, k):
    if k <= 1 or len(x) < k: return np.asarray(x, np.float32)
    if k % 2 == 0: k += 1
    pad = k//2
    xx = np.pad(np.asarray(x, np.float32), (pad,pad), mode="edge")
    w = np.ones(k)/k
    return np.convolve(xx, w, mode="valid")

def resample_uniform(t, x, fs=None):
    t = np.asarray(t, np.float32); x = np.asarray(x, np.float32)
    if len(t) < 3: return None, None, None
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 1e-6: return None, None, None
    if fs is None: fs = 1.0/dt
    t_u = np.arange(t[0], t[-1] + 1e-6, 1.0/fs, dtype=np.float32)
    x_u = np.interp(t_u, t, x).astype(np.float32)
    return t_u, x_u, fs

def butter_sos(lo, hi, fs, order=4):
    nyq = fs*0.5
    lo2 = max(1e-3, lo/nyq); hi2 = min(0.999, hi/nyq)
    if hi2 <= lo2 + 1e-3: return None
    from scipy.signal import butter
    return butter(order, [lo2,hi2], btype='bandpass', output='sos')

def butter_hp_sos(cut, fs, order=2):
    nyq = fs*0.5
    w = max(1e-4, cut/nyq)
    from scipy.signal import butter
    return butter(order, w, btype='highpass', output='sos')

def bandpass(x, fs, band=RESP_BAND, order=RESP_ORDER):
    if x is None or fs is None: return x
    x = np.asarray(x, np.float32).ravel()
    sos = butter_sos(band[0], band[1], fs, order)
    if sos is None: return x
    from scipy.signal import sosfiltfilt
    padlen = 3 * (sos.shape[0] if hasattr(sos, "shape") else 4)
    if x.size <= padlen: return x
    try:
        return sosfiltfilt(sos, x).astype(np.float32)
    except ValueError:
        return x

def highpass(x, fs, cut=0.03, order=2):
    if x is None or fs is None: return x
    x = np.asarray(x, np.float32).ravel()
    sos = butter_hp_sos(cut, fs, order)
    from scipy.signal import sosfiltfilt
    padlen = 3 * (sos.shape[0] if hasattr(sos, "shape") else 2)
    if x.size <= padlen: return x
    try:
        return sosfiltfilt(sos, x).astype(np.float32)
    except ValueError:
        return x

def zscore(x):
    x = np.asarray(x, np.float32)
    mu = float(np.mean(x)); sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-6: sd = 1.0
    return (x - mu)/sd

def seg_params(L, fs):
    target = int(fs*SPEC_WIN)
    nper = max(16, min(target, max(16, L//2)))
    nover = min(nper-1, nper//2)
    return nper, nover

def safe_fmt(v, fmt="{:.2f}"):
    return fmt.format(v) if v is not None and np.isfinite(v) else "--"

def r_to_p(r, n):
    if not np.isfinite(r) or n is None or n < 4: return None
    den = max(1e-9, 1.0 - r*r)
    t = r * np.sqrt(max(1.0, (n-2))/den)
    from math import erf, sqrt
    Phi = 0.5*(1.0 + erf(abs(t)/sqrt(2.0)))
    p = 2.0*(1.0 - Phi)
    return float(max(1e-9, min(1.0, p)))

def gcc_phat_linear(x, y, fs):
    x = np.asarray(x, np.float32).ravel()
    y = np.asarray(y, np.float32).ravel()
    n = int(len(x)+len(y)-1)
    nfft = 1
    while nfft < n: nfft <<= 1
    X = np.fft.rfft(x, nfft); Y = np.fft.rfft(y, nfft)
    R = X * np.conj(Y); R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, nfft)
    lag_samp = np.arange(-len(y)+1, len(x))
    cc = np.concatenate([cc[-(len(y)-1):], cc[:len(x)]])
    i = int(np.argmax(cc))
    return float(lag_samp[i] / fs)

def delta_lag_window(global_lag, window_lag, clip=None):
    if clip is None:
        clip = PANEL_FINE_RANGE   # 예: 0.60
    if global_lag is None or not np.isfinite(global_lag):
        return float(np.clip(window_lag, -clip, clip))
    return float(np.clip(window_lag - global_lag, -clip, clip))

def draw_dotted_polyline(img, pts, color, gap=6):
    # pts: (N,2) int array
    if pts is None or len(pts) < 2: return
    for i in range(0, len(pts)-1, gap):
        p1 = tuple(pts[i]); p2 = tuple(pts[min(i+1, len(pts)-1)])
        cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

# ---------------- GT ----------------
gt_t, gt_resp = None, None
if SHOW_GT:
    try:
        with h5py.File(H5, "r") as f:
            gt_resp = np.asarray(f["respiration"][:], np.float32)
            gt_t    = np.asarray(f["time"][:], np.float32)
    except Exception as e:
        print("[GT disabled]", e); SHOW_GT = False

def estimate_rr_from_gt(t0, t1):
    if not SHOW_GT or gt_t is None: return np.nan
    m = (gt_t >= t0) & (gt_t <= t1)
    if np.count_nonzero(m) < 8: return np.nan
    tu, gu, fs = resample_uniform(gt_t[m], gt_resp[m])
    if tu is None: return np.nan
    gu = bandpass(zscore(gu), fs, RR_BAND, RESP_ORDER)
    L = len(gu); nper, nover = seg_params(L, fs)
    if nper == 0 or L < nper: return np.nan
    f, P = welch(gu, fs=fs, nperseg=nper, noverlap=nover)
    mb = (f >= RR_BAND[0]) & (f <= RR_BAND[1])
    if np.count_nonzero(mb) < MIN_BAND_BINS: return np.nan
    fb = f[mb]; Pb = P[mb]
    f0 = float(fb[np.argmax(Pb)])
    return f0 * 60.0

# ---------------- Domain helpers ----------------
def unify_domain(tw, sig):
    tu, su, fs = resample_uniform(tw, sig)
    if tu is None: return None, None, None
    su = zscore(bandpass(su, fs, RESP_BAND, RESP_ORDER))
    return tu, su, fs

def phase_corrected_gt(tw, base_lag=0.0, local_adj=0.0):
    if not SHOW_GT: return None, None, None
    tg = gt_t + (base_lag + local_adj)
    mg = (tg >= tw[0]) & (tg <= tw[-1])
    if not np.any(mg): return None, None, None
    tu = np.linspace(tw[0], tw[-1], max(2, int((tw[-1]-tw[0]) / (np.median(np.diff(tw))+1e-9))), dtype=np.float32)
    gA = np.interp(tu, tg[mg], gt_resp[mg]).astype(np.float32)
    return tu, gA, (tu.size/(tw[-1]-tw[0]+1e-9))

def fine_sweep_lag(xU, gU, fs, base_lag, sweep=0.20):
    # xU, gU: 이미 도메인 통일(zscore+bandpass)된 1D
    if gU is None: return base_lag, np.nan
    lags = np.arange(-sweep, sweep+1e-9, LAG_FINE_STEP, dtype=np.float32)
    best = base_lag; bestcorr = -1.0
    for dl in lags:
        sh = int(np.round(dl*fs))
        if sh > 0:
            xs = xU[:-sh]; gs = gU[sh:]
        elif sh < 0:
            xs = xU[-sh:]; gs = gU[:sh]
        else:
            xs = xU; gs = gU
        if len(xs) < 16: continue
        r = np.corrcoef(xs, gs)[0,1]
        if np.isfinite(r) and abs(r) > bestcorr:
            bestcorr = abs(r); best = base_lag + dl
    return float(np.clip(best, -LAG_MAX_SEC, LAG_MAX_SEC)), float(bestcorr)

# ---------------- Global lag ----------------
GLOBAL_LAG_EMA_BETA   = 0.08   # 느린 추종
GLOBAL_LAG_RECALC_SEC = 6.0    # 최소 재추정 간격
GLOBAL_LAG_LAST_T     = 0.0

def learn_global_lag_adaptive():
    """최근 12s 창에서 주기적 재추정(게이트 조건 충족 시) + 느린 EMA."""
    global GLOBAL_LAG_SEC, GLOBAL_LAG_TEXT, GLOBAL_LAG_LAST_T
    if not SHOW_GT or len(ts) < 64: return
    t_np = np.asarray(ts, np.float32)
    t_now = t_np[-1]
    if (t_now - GLOBAL_LAG_LAST_T) < GLOBAL_LAG_RECALC_SEC: return

    # 최근 12s 창
    t1 = t_now; t0 = max(t_np[0], t1 - max(12.0, SPEC_WIN))
    m  = (t_np >= t0)
    if np.count_nonzero(m) < 64: return
    tw = t_np[m]; sw = np.asarray(dC, np.float32)[m]

    # 통일 도메인 (rPPG 쪽 기준 시간축 tu 생성)
    tu, su, fs = resample_uniform(tw, sw)
    if tu is None:
        return
    su = bandpass(zscore(su), fs, RESP_BAND, RESP_ORDER)

    # 동 구간 GT → 반드시 tu로 '재보간'
    mg = (gt_t >= t0) & (gt_t <= t1)
    if not np.any(mg):
        return
    tg_u, g_u, _ = resample_uniform(gt_t[mg], gt_resp[mg], fs)
    if tg_u is None:
        return
    gu = np.interp(tu, tg_u, g_u).astype(np.float32)  # ★ 길이=tu와 정확히 일치
    gu = bandpass(zscore(gu), fs, RESP_BAND, RESP_ORDER)

    # 길이/유효성 가드
    L = len(su)
    if L != len(gu) or L < 16:
        m = min(L, len(gu))
        if m < 16:
            return
        su = su[-m:]
        gu = gu[-m:]

    # 품질(간단 게이트)
    nper = max(16, int(fs * 8.0))
    if len(su) < nper:
        return
    from scipy.signal import coherence
    f, Cxy = coherence(su, gu, fs=fs, nperseg=nper, noverlap=nper // 2)
    Cxy = np.nan_to_num(Cxy, nan=0.0, posinf=0.0, neginf=0.0)  # ★ divide NaN 방지
    mb = (f >= RR_BAND[0]) & (f <= RR_BAND[1])
    coh_val = float(np.mean(Cxy[mb])) if np.any(mb) else 0.0

    r0 = float(np.corrcoef(su, gu)[0, 1])
    p0 = r_to_p(r0, len(su)) or 1.0
    if not (coh_val >= 0.45 and abs(r0) >= 0.25 and p0 < 0.05):
        return

    # 래그 추정 & EMA
    try:
        lag_est = float(np.clip(gcc_phat_linear(su, gu, fs), -LAG_MAX_SEC, LAG_MAX_SEC))
    except Exception:
        return
    if GLOBAL_LAG_SEC is None:
        GLOBAL_LAG_SEC = lag_est
    else:
        GLOBAL_LAG_SEC = float((1.0 - GLOBAL_LAG_EMA_BETA) * GLOBAL_LAG_SEC + GLOBAL_LAG_EMA_BETA * lag_est)

    GLOBAL_LAG_LAST_T = t_now
    GLOBAL_LAG_TEXT = f"GLOBAL LAG {GLOBAL_LAG_SEC:+.02f}s (EMA, FS={fs:.1f}Hz)"

# ---------------- Metrics ----------------
def band_metrics(tw, sw):
    if not SHOW_GT or tw is None or len(tw) < 8: return {}
    t1 = tw[-1]; t0 = max(tw[0], t1 - SPEC_WIN)
    m = (tw >= t0)
    if np.count_nonzero(m) < 8: return {}
    tw = tw[m].astype(np.float32)
    sw = np.asarray(sw, np.float32)[m]
    tu, su, fs = resample_uniform(tw, sw)
    if tu is None: return {}
    su = zscore(bandpass(su, fs, RESP_BAND, RESP_ORDER))

    # raw GT (표시/진단용)
    tg_raw = gt_t
    mg_raw = (tg_raw >= tw[0]) & (tg_raw <= tw[-1])
    if not np.any(mg_raw): return {}
    gu_raw = np.interp(tu, tg_raw[mg_raw], gt_resp[mg_raw]).astype(np.float32)
    gu_raw = zscore(bandpass(gu_raw, fs, RESP_BAND, RESP_ORDER))

    # global-lag 적용 GT (정합/지표용)
    tg = gt_t + (GLOBAL_LAG_SEC or 0.0)
    mg = (tg >= tw[0]) & (tg <= tw[-1])
    if not np.any(mg): return {}
    gu = np.interp(tu, tg[mg], gt_resp[mg]).astype(np.float32)
    gu = zscore(bandpass(gu, fs, RESP_BAND, RESP_ORDER))

    L = len(su)
    if L < 16: return {}

    # zero-shift corr & 전역 기준 dlag
    corr0 = float(np.corrcoef(su, gu)[0,1])
    try:
        lag_lin = gcc_phat_linear(su, gu, fs)
    except Exception:
        lag_lin = 0.0
    dlag = delta_lag_window(GLOBAL_LAG_SEC, lag_lin, clip=0.5)

    # 패널 국소 래그(전역 주변 미세 스윕)
    adj_lag, lagcorr_abs = fine_sweep_lag(su, gu, fs, base_lag=0.0, sweep=PANEL_FINE_RANGE)
    lagcorr = float(lagcorr_abs)

    # 대역 코히런스
    nper = max(16, int(fs*8.0)); nover = nper//2
    if L < nper:
        coh_val = np.nan
    else:
        f, Cxy = coherence(su, gu, fs=fs, nperseg=nper, noverlap=nover)
        Cxy = np.nan_to_num(Cxy, nan=0.0, posinf=0.0, neginf=0.0)
        mb = (f >= RR_BAND[0]) & (f <= RR_BAND[1])
        coh_val = float(np.nanmean(Cxy[mb])) if np.any(mb) else np.nan

    rr_bpm = estimate_rr_from_gt(t0, t1)
    pval = r_to_p(corr0, L)

    # 패널 품질 게이트 판단
    good_gate = (np.isfinite(coh_val) and (coh_val >= GATE_COH)
                 and np.isfinite(lagcorr) and (lagcorr >= GATE_LAGCORR)
                 and (pval is not None and pval < GATE_P))

    return dict(
        corr=corr0, lagcorr=lagcorr, dlag=dlag, rr_bpm=rr_bpm, n=L, p=pval, fs=fs,
        tu=tu, su=su, gu=gu, gu_raw=gu_raw,
        coh=coh_val, adj_lag=float(adj_lag), gate=bool(good_gate)
    )


# ---------------- ZFIX 기준 산출 ----------------
def try_update_zfix():
    if ZFIX["dY"] is not None: return
    if len(ts) < 3: return
    t_np = np.asarray(ts, np.float32)
    t1 = t_np[-1]; t0 = t1 - ZFIX_WARMUP_SEC
    if t_np[0] > t0: return
    def ref_mu_sd(t, x):
        m = t >= t0; tw = t[m]; sw = np.asarray(x, np.float32)[m]
        tu, su, fs = resample_uniform(tw, sw)
        if tu is None: return None
        su = bandpass(su, fs, RESP_BAND, RESP_ORDER)
        mu, sd = float(np.mean(su)), float(np.std(su)+1e-9)
        return (mu, sd)
    ZFIX["dY"] = ref_mu_sd(t_np, dY)
    ZFIX["dD"] = ref_mu_sd(t_np, dD)
    ZFIX["dW"] = ref_mu_sd(t_np, dW)
    ZFIX["dC"] = ref_mu_sd(t_np, dC)
    if SHOW_GT and gt_t is not None:
        mgt = (gt_t >= t0) & (gt_t <= t1)
        if np.any(mgt):
            tu, gu, fs = resample_uniform(gt_t[mgt], gt_resp[mgt])
            if tu is not None:
                gu = bandpass(gu, fs, RESP_BAND, RESP_ORDER)
                ZFIX["GT"] = (float(np.mean(gu)), float(np.std(gu)+1e-9))

# ---------------- Drawing helpers ----------------
def draw_panel_to(right_img, rect, title, t_seq, sig_seq, color):
    x1,y1,x2,y2 = rect
    panel = right_img[y1:y2, x1:x2]; panel[:] = BG
    if len(t_seq) < 2: return
    t_arr = np.asarray(t_seq, np.float32)
    s_arr = np.asarray(sig_seq, np.float32)
    t1 = t_arr[-1]
    span = min(SPAN_SEC, (t_arr[-1]-t_arr[0]) if len(t_arr)>1 else SPAN_SEC)
    t0 = t1 - span
    m = t_arr >= t0
    if np.count_nonzero(m) < 4: return
    tw = t_arr[m].astype(np.float32); sw = s_arr[m].astype(np.float32)
    if MA_WIN > 1:
        sw = moving_average(sw, MA_WIN); tw = tw[-len(sw):]

    stats = band_metrics(tw, sw) if SHOW_GT else {}

    # 도메인 통일
    tu, su, fs = resample_uniform(tw, sw)
    if tu is not None:
        su = zscore(bandpass(su, fs, RESP_BAND, RESP_ORDER))
        sw = np.interp(tw, tu, su).astype(np.float32)

    # GT(raw/보정)
    gv_raw = None
    gv_corr = None
    if SHOW_GT and stats:
        # raw (정규화된 원시 GT)
        gv_raw = stats["gu_raw"]

        # 패널 보정 GT = global_lag + adj_lag
        dlag_local = float(np.clip(stats.get("adj_lag", 0.0), -PANEL_FINE_RANGE, PANEL_FINE_RANGE))
        tg_corr = gt_t + (GLOBAL_LAG_SEC or 0.0) + dlag_local
        mg = (tg_corr >= tw[0]) & (tg_corr <= tw[-1])
        if np.any(mg):
            gv_corr = np.interp(tw, tg_corr[mg], gt_resp[mg]).astype(np.float32)
            tu2, gu2, fs2 = resample_uniform(tw, gv_corr)
            if tu2 is not None:
                gu2 = zscore(bandpass(gu2, fs2, RESP_BAND, RESP_ORDER))
                gv_corr = np.interp(tw, tu2, gu2).astype(np.float32)
        # raw 정규화
        tu3, gu3, fs3 = resample_uniform(tw, np.interp(tw, gt_t, gt_resp))
        if tu3 is not None:
            gu3 = zscore(bandpass(gu3, fs3, RESP_BAND, RESP_ORDER))
            gv_raw = np.interp(tw, tu3, gu3).astype(np.float32)

    # 시각화 스케일링
    sw_v   = soft_clip_z(sw, Z_CLIP)
    gv_v   = soft_clip_z(gv_corr, Z_CLIP) if gv_corr is not None else None
    gv0_v  = soft_clip_z(gv_raw, Z_CLIP)  if gv_raw  is not None else None

    y_min, y_max = -Z_CLIP, Z_CLIP
    ph,pw = panel.shape[:2]; margin = 10; h = ph - 40; w = pw
    cv2.rectangle(panel, (0,30), (w-1,ph-1), (40,40,40), 1)

    def to_xy(tvec, yvec):
        xs = (margin + (w-2*margin) * np.clip((tvec - t0) / max(1e-6, span), 0,1)).astype(int)
        norm = (np.clip(yvec, y_min, y_max) - y_min) / (y_max - y_min + 1e-9)
        ys = (ph-1 - margin - (h-2*margin) * np.clip(norm,0,1)).astype(int)
        return xs, ys

    for val, label_t in [(-Z_CLIP, f"{-Z_CLIP:.0f}"), (0.0,"0"), (Z_CLIP, f"+{Z_CLIP:.0f}")]:
        y_pos = int(ph-1 - margin - (h-2*margin) * ((val-y_min)/(y_max-y_min)))
        cv2.line(panel, (0,y_pos), (6,y_pos), (140,140,140), 1)
        cv2.putText(panel, label_t, (8, y_pos+4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175,175,175), 1, cv2.LINE_AA)

    xs, ys = to_xy(tw, sw_v)
    if len(xs) >= 2:
        cv2.polylines(panel, [np.stack([xs, ys],1)], False, color, 2, cv2.LINE_AA)
    # GT corrected (실선)
    if gv_v is not None:
        xs2, ys2 = to_xy(tw, gv_v)
        cv2.polylines(panel, [np.stack([xs2, ys2],1)], False, COLORS["GT"], 1, cv2.LINE_AA)
    # GT raw (점선)
    if gv0_v is not None:
        xs3, ys3 = to_xy(tw, gv0_v)
        draw_dotted_polyline(panel, np.stack([xs3, ys3],1), COLORS["GT_RAW"], gap=8)

    # HUD
    if stats:
        coh_v = stats.get("coh")
        show_dlag = (np.isfinite(stats.get("dlag", np.nan))
                     and np.isfinite(coh_v) and (coh_v >= 0.45)
                     and (abs(stats.get("corr", 0)) >= 0.25))
        dlag_txt = safe_fmt(stats["dlag"], "{:+.02f}") + "s" if show_dlag else "--"

        gate_txt = "●" if stats.get("gate") else "○"  # ✅ 게이트 점

        line = (f"coh={safe_fmt(coh_v)}  "
                f"corr={safe_fmt(stats.get('corr'), '{:+.2f}')}  "
                f"lagcorr={safe_fmt(stats.get('lagcorr'), '{:.2f}')}  "
                f"dlag={dlag_txt}  "
                f"RR={safe_fmt(stats.get('rr_bpm'), '{:.1f}')} bpm  "
                f"p{('<' + safe_fmt(stats.get('p'), '{:.003f}')) if stats.get('p') is not None else '--'}  "
                f"{gate_txt}")  # ✅ 끝에 ●/○ 추가

        put(panel, line, (8, 18), 0.48, (200, 200, 200))

# ---------------- Polarity (워밍업 1회) ----------------
def try_update_polarity_once():
    global POL_APPLIED
    if not SHOW_GT or POL_SIGN["dY"] is not None: return
    if len(ts) < 8: return
    t_np = np.asarray(ts, np.float32)
    t1 = t_np[-1]; t0 = t1 - POL_READY_SEC
    if t_np[0] > t0: return
    m = (t_np >= t0); tw = t_np[m]
    Y = np.asarray(dY, np.float32)[m]; D = np.asarray(dD, np.float32)[m]; Wv = np.asarray(dW, np.float32)[m]

    tu, yU, fs = resample_uniform(tw, Y)
    if tu is None: return
    _, dU, _ = resample_uniform(tw, D, fs)
    _, wU, _ = resample_uniform(tw, Wv, fs)
    yU = zscore(bandpass(yU, fs, RESP_BAND, RESP_ORDER))
    dU = zscore(bandpass(dU, fs, RESP_BAND, RESP_ORDER))
    wU = zscore(bandpass(wU, fs, RESP_BAND, RESP_ORDER))
    # 글로벌 래그 적용한 GT
    tg = gt_t + (GLOBAL_LAG_SEC or 0.0)
    mg = (tg >= tw[0]) & (tg <= tw[-1])
    if not np.any(mg): return
    gU = np.interp(tu, tg[mg], gt_resp[mg]).astype(np.float32)
    gU = zscore(bandpass(gU, fs, RESP_BAND, RESP_ORDER))

    def sgn(a, b):
        r = np.corrcoef(a, b)[0,1]; return +1.0 if r >= 0 else -1.0
    POL_SIGN["dY"] = sgn(yU, gU)
    POL_SIGN["dD"] = sgn(dU, gU)
    POL_SIGN["dW"] = sgn(wU, gU)

    if not POL_APPLIED and POL_SIGN["dY"] is not None:
        def re_sign(dq, s): tmp = [v*s for v in dq]; dq.clear(); dq.extend(tmp)
        re_sign(dY, POL_SIGN["dY"]); re_sign(dD, POL_SIGN["dD"]); re_sign(dW, POL_SIGN["dW"])
        POL_APPLIED = True

# ---------------- MediaPipe Pose ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------- Video ----------------
cap = cv2.VideoCapture(VIDEO); assert cap.isOpened(), "영상 열기 실패"
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# ---------------- Main loop ----------------
win_name = "COHFACE v3.0 (minimal HUD, GT-aligned, phase-corrected)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, VIEW_W, VIEW_H)

prev_ms = 0.0; frame_idx = 0
lag_ema = None

while True:
    ok, frame = cap.read()
    if not ok: break
    H, W = frame.shape[:2]
    ms = cap.get(cv2.CAP_PROP_POS_MSEC) or (prev_ms + 1000.0/max(1.0,fps))
    t  = ms/1000.0
    if t < prev_ms/1000.0: t = prev_ms/1000.0 + 1.0/max(1.0,fps)
    prev_ms = ms; frame_idx += 1
    vis = frame.copy()

    # ---- Pose (격프레임 계산) ----
    if (frame_idx % POSE_EVERY_N) == 0:
        rgb_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               (PROC_W, int(H*PROC_W/W)), interpolation=cv2.INTER_AREA)
        res = pose.process(rgb_small)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            h_s, w_s = rgb_small.shape[:2]; sx, sy = W/float(w_s), H/float(h_s)
            def to_px(pt): return np.array([pt.x*w_s*sx, pt.y*h_s*sy], np.float32)
            L = to_px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
            R = to_px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            N = to_px(lm[mp_pose.PoseLandmark.NOSE])
            visL = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility
            visR = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility
            visN = lm[mp_pose.PoseLandmark.NOSE].visibility
            if min(visL, visR, visN) > 0.5:
                if prev_L is not None:
                    if (np.linalg.norm(L - prev_L) > MAX_LMK_JUMP or
                        np.linalg.norm(R - prev_R) > MAX_LMK_JUMP or
                        np.linalg.norm(N - prev_N) > MAX_LMK_JUMP):
                        pass
                    else:
                        L_EMA = L if L_EMA is None else EMA_BETA_LMK*L_EMA + (1-EMA_BETA_LMK)*L
                        R_EMA = R if R_EMA is None else EMA_BETA_LMK*R_EMA + (1-EMA_BETA_LMK)*R
                        N_EMA = N if N_EMA is None else EMA_BETA_LMK*N_EMA + (1-EMA_BETA_LMK)*N
                else:
                    L_EMA, R_EMA, N_EMA = L.copy(), R.copy(), N.copy()
                prev_L, prev_R, prev_N = L.copy(), R.copy(), N.copy()
                # 지터 기록(프레임 간 이동량)
                hist_L.append(L_EMA.copy()); hist_R.append(R_EMA.copy()); hist_N.append(N_EMA.copy())

    valid = (L_EMA is not None and R_EMA is not None and N_EMA is not None)
    if valid:
        if y0 is None or d0 is None or w0 is None:
            y0 = float((L_EMA[1] + R_EMA[1]) / 2.0)
            v = R_EMA - L_EMA; wvec = N_EMA - L_EMA
            v2 = float(np.dot(v, v)) + 1e-12
            tproj = float(np.clip(np.dot(wvec, v)/v2, 0.0, 1.0))
            foot = L_EMA + tproj * v
            d_init = v[0]*wvec[1] - v[1]*wvec[0]
            d0 = float(d_init/(np.sqrt(v2)+1e-12))
            w0 = float(np.linalg.norm(R_EMA - L_EMA))

        y_now = float((L_EMA[1] - y0) + (R_EMA[1] - y0))/2.0  # 평균 중심 기준
        v = R_EMA - L_EMA; wvec = N_EMA - L_EMA
        v2 = float(np.dot(v, v)) + 1e-12
        tproj = float(np.clip(np.dot(wvec, v)/v2, 0.0, 1.0))
        foot = L_EMA + tproj * v
        d_now = (v[0]*wvec[1] - v[1]*wvec[0])/(np.sqrt(v2)+1e-12) - d0
        w_now = float(np.linalg.norm(R_EMA - L_EMA)) - w0

        dY_t = y_now
        dD_t = d_now
        dW_t = w_now

        # Polarity 적용
        try_update_polarity_once()
        if POL_SIGN["dY"] is not None:
            dY_t *= POL_SIGN["dY"]; dD_t *= POL_SIGN["dD"]; dW_t *= POL_SIGN["dW"]

        ts.append(t); dY.append(dY_t); dD.append(dD_t); dW.append(dW_t)

        # dC = 최근 표준화 z로 합성
        def z_last(buf):
            arr = np.asarray(buf, np.float32)
            if arr.size < 5: return 0.0
            seg = arr[-min(200, len(arr)):]
            sd = float(np.std(seg)) or 1.0; mu = float(np.mean(seg))
            return float((arr[-1] - mu) / sd)
        dC.append((z_last(dY) + z_last(dD) + z_last(dW)) / 3.0)

        # ----- 학습/추론 도메인 일치 dC_opt -----
        if SHOW_GT and len(ts) >= int(SPEC_WIN*fps*0.5):
            t_np = np.asarray(ts, np.float32)
            mwin = t_np >= (t_np[-1] - SPEC_WIN)
            tw = t_np[mwin]; Yw = np.asarray(dY, np.float32)[mwin]
            Dw = np.asarray(dD, np.float32)[mwin]; Ww = np.asarray(dW, np.float32)[mwin]
            tu, yU, fs = resample_uniform(tw, Yw)
            if tu is not None:
                _, dU, _ = resample_uniform(tw, Dw, fs)
                _, wU, _ = resample_uniform(tw, Ww, fs)
                yU = zscore(bandpass(yU, fs, RESP_BAND, RESP_ORDER))
                dU = zscore(bandpass(dU, fs, RESP_BAND, RESP_ORDER))
                wU = zscore(bandpass(wU, fs, RESP_BAND, RESP_ORDER))
                # global GT
                tg = gt_t + (GLOBAL_LAG_SEC or 0.0)
                mg = (tg >= tw[0]) & (tg <= tw[-1])
                if np.any(mg):
                    gU = np.interp(tu, tg[mg], gt_resp[mg]).astype(np.float32)
                    gU = zscore(bandpass(gU, fs, RESP_BAND, RESP_ORDER))

                    # 게이트(품질 + 지터)
                    def jitter_of(hist):
                        if len(hist) < 3: return 0.0
                        d = np.diff(np.stack(hist,0), axis=0)  # (k-1,2)
                        step = np.linalg.norm(d, axis=1)
                        return float(np.std(step))
                    jL = jitter_of(hist_L); jR = jitter_of(hist_R); jN = jitter_of(hist_N)
                    good_jitter = (max(jL, jR, jN) < JITTER_SPIKE)

                    # 패널 품질
                    su = (yU + dU + wU)/3.0
                    # lagcorr for gate
                    adj_lag, lagcorr_abs = fine_sweep_lag(su, gU, fs, base_lag=0.0, sweep=PANEL_FINE_RANGE)
                    # corr/p
                    corr0 = float(np.corrcoef(su, gU)[0,1]); pval = r_to_p(corr0, len(su))
                    # coherence
                    nper = max(16, int(fs*8.0)); nover = nper//2
                    coh_val = np.nan
                    if len(su) >= nper:
                        f, Cxy = coherence(su, gU, fs=fs, nperseg=nper, noverlap=nover)
                        Cxy = np.nan_to_num(Cxy, nan=0.0, posinf=0.0, neginf=0.0)
                        mb = (f >= RR_BAND[0]) & (f <= RR_BAND[1])
                        coh_val = float(np.nanmean(Cxy[mb])) if np.any(mb) else np.nan

                    good = (np.isfinite(coh_val) and coh_val>=GATE_COH and lagcorr_abs>=GATE_LAGCORR
                            and (pval is not None and pval<GATE_P) and good_jitter)

                    # 가중치 학습
                    if good:
                        X = np.stack([yU, dU, wU], 1); Yt = gU
                        XtX = X.T @ X; XtX[np.diag_indices_from(XtX)] += RIDGE_ALPHA
                        try:
                            w_r = np.linalg.solve(XtX, X.T @ Yt)
                            w_r = w_r/(np.linalg.norm(w_r)+1e-9)
                            WC = w_r if WC is None else (W_EMA_BETA*WC + (1.0-W_EMA_BETA)*w_r)
                        except Exception:
                            pass

                # 추론: 마지막 점 가중합
                if WC is not None:
                    val = float(WC[0]*yU[-1] + WC[1]*dU[-1] + WC[2]*wU[-1])
                else:
                    val = 0.0
                # HP + soft-clip for display stability
                dC_opt.append(val)
        else:
            dC_opt.append(0.0)

        # 히스토리 유지
        while len(ts) and (ts[-1] - ts[0] > HIS_SEC):
            ts.popleft(); dY.popleft(); dD.popleft(); dW.popleft(); dC.popleft(); dC_opt.popleft()

        # draw overlay
        pL = tuple(np.round(L_EMA).astype(int)); pR = tuple(np.round(R_EMA).astype(int))
        pN = tuple(np.round(N_EMA).astype(int))
        pF = tuple(np.round(foot).astype(int))
        cv2.circle(vis, pL, 6, (40,230,255), -1, cv2.LINE_AA)
        cv2.circle(vis, pR, 6, (40,230,255), -1, cv2.LINE_AA)
        cv2.circle(vis, pN, 6, (180,200,80), -1, cv2.LINE_AA)
        cv2.line(vis, pL, pR, (230,230,230), 3, cv2.LINE_AA)
        cv2.line(vis, pN, pF, (120,200,120), 2, cv2.LINE_AA)
        put(vis, f"dY:{dY_t:+.2f}px  dD:{dD_t:+.2f}px  dW:{dW_t:+.2f}px", (12,28), 0.9, HUD_COL)
        if WC is not None:
            put(vis, f"w=[{WC[0]:+.2f},{WC[1]:+.2f},{WC[2]:+.2f}]  (ridge+EMA)", (12,52), 0.7, HUD_COL)
    else:
        put(vis, "Waiting shoulders + nose...", (12,28), 0.8, (80,180,255))

    # ---- jitter HUD ----
    def jitter_val(hist):
        if len(hist) < 3: return 0.0
        d = np.diff(np.stack(hist,0), axis=0)
        step = np.linalg.norm(d, axis=1)
        return float(np.std(step))
    jL = jitter_val(hist_L); jR = jitter_val(hist_R); jN = jitter_val(hist_N)
    put(vis, f"jitter L/R/N  ≈  {jL:.2f}px / {jR:.2f}px / {jN:.2f}px", (12, 84), 0.55, (190,210,210))

    # ---- plumbing ----
    try_update_zfix()
    learn_global_lag_adaptive()

    # ---- Compose view ----
    left_w = int(VIEW_W * LEFT_RATIO); right_w = VIEW_W - left_w
    left = cv2.resize(vis, (left_w, VIEW_H), interpolation=cv2.INTER_AREA)
    put(left, f"{t:.1f}s", TIMER_POS, 1.0, (210,210,210))
    if GLOBAL_LAG_TEXT:
        put(left, GLOBAL_LAG_TEXT, (12, 110), 0.55, (180,220,220))

    right = np.full((VIEW_H, right_w, 3), BG, np.uint8)
    Rw, Rh = right_w, VIEW_H; gap = 8
    cell_w = (Rw - 3*gap)//2; cell_h = (Rh - 3*gap)//2
    cells = [
        (gap, gap, gap+cell_w, gap+cell_h),                          # dY
        (gap, 2*gap+cell_h, gap+cell_w, 2*gap+2*cell_h),             # dD
        (2*gap+cell_w, gap, 2*gap+2*cell_w, gap+cell_h),             # dW
        (2*gap+cell_w, 2*gap+cell_h, 2*gap+2*cell_w, 2*gap+2*cell_h) # dC
    ]

    tnp = np.asarray(ts, np.float32)
    draw_panel_to(right, cells[0], "dY vs GT", tnp, np.asarray(dY, np.float32), COLORS["dY"])
    draw_panel_to(right, cells[1], "dD vs GT", tnp, np.asarray(dD, np.float32), COLORS["dD"])
    draw_panel_to(right, cells[2], "dW vs GT", tnp, np.asarray(dW, np.float32), COLORS["dW"])

    # dC (기본) + dC_opt(얇게, HP+soft-clip)
    draw_panel_to(right, cells[3], "dC vs GT", tnp, np.asarray(dC, np.float32), COLORS["dC"])
    x1,y1,x2,y2 = cells[3]; panel = right[y1:y2, x1:x2]
    if len(ts) > 8:
        t_arr = np.asarray(ts, np.float32); s_arr = np.asarray(dC_opt, np.float32)
        t1v = t_arr[-1]; span = min(SPAN_SEC, (t_arr[-1]-t_arr[0])); t0v = t1v - span
        m = t_arr >= t0v
        if np.count_nonzero(m) >= 4:
            tw = t_arr[m]; sw = s_arr[m]
            if MA_WIN > 1:
                sw = moving_average(sw, MA_WIN); tw = tw[-len(sw):]
            tu, su, fs = resample_uniform(tw, sw)
            if tu is not None:
                su = highpass(su, fs, cut=0.03, order=2)  # drift 억제
                su = soft_clip_z(su, Z_CLIP)
                sw = np.interp(tw, tu, su).astype(np.float32)
            if VIZ_MODE == "zfix" and ZFIX.get("dC") is not None:
                mu, sd = ZFIX["dC"]; sw_v = (sw - mu)/(sd+1e-9)
            elif VIZ_MODE == "zwin":
                sw_v = zscore(sw)
            else:
                sw_v = sw
            sw_v = soft_clip_z(sw_v, Z_CLIP)
            ph,pw = panel.shape[:2]; margin=10; h=ph-40; w=pw
            y_min, y_max = -Z_CLIP, Z_CLIP
            xs = (margin + (w-2*margin) * np.clip((tw - t0v)/max(1e-6, span), 0,1)).astype(int)
            norm = (np.clip(sw_v, y_min, y_max) - y_min) / (y_max-y_min + 1e-9)
            ys = (ph-1 - margin - (h-2*margin) * np.clip(norm,0,1)).astype(int)
            if len(xs) >= 2:
                cv2.polylines(panel, [np.stack([xs, ys],1)], False, (180,255,255), 1, cv2.LINE_AA)
                put(panel, "dC_opt (ridge+EMA, HP, phase-corr)", (12, panel.shape[0]-8), 0.5, (180,255,255))

    canvas = np.hstack([left, right])
    cv2.imshow(win_name, canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

pose.close()
cap.release()
cv2.destroyAllWindows()
