# -*- coding: utf-8 -*-
"""
COHFACE (v2.7a, GT-aligned minimal HUD)
- 왼쪽: 비디오 + 랜드마크 + 타이머
- 오른쪽: 2x2 패널 (dY, dD, dW, dC vs GT)
"""

import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2, h5py, numpy as np, mediapipe as mp
from collections import deque
from scipy.signal import welch, coherence

# ---------------- Paths ----------------
VIDEO = "/mnt/hdd18t/rppg_dataset/raw/cohface/1/0/data.mkv"
H5 = "/mnt/hdd18t/rppg_dataset/raw/cohface/1/0/data.hdf5"
SHOW_GT = os.path.exists(H5)

# ---------------- View/Run Params ----------------
VIEW_W, VIEW_H = 1920, 900
LEFT_RATIO = 0.40
SPAN_SEC = 60.0
HIS_SEC = 180.0
PROC_W = 640
POSE_EVERY_N = 2
BG = (16, 16, 16)
HUD_COL = (200, 230, 230)
COLORS = {  # BGR (OpenCV)
    "dY": (230, 230, 230),  # 흰색
    "dD": (120, 220, 120),  # 연녹
    "dW": (120, 200, 255),  # 청록
    "dC": (220, 220, 80),  # 연노랑
}

# (검출/스무딩)
EMA_BETA_LMK = 0.7
MAX_LMK_JUMP = 40.0

# (호흡 대역/필터 & 스펙트럼)
RESP_BAND = (0.08, 0.60)  # 5–36 bpm
RESP_ORDER = 4
RR_BAND = (0.08, 0.60)
MIN_BAND_BINS = 3
SPEC_WIN = 12.0
SNR_EXCLUDE_K = 1
SNR_FLOOR_MIN = 1e-6

# (Z-정규화)
VIZ_MODE = "zfix"  # zfix | zwin | raw
ZFIX_WARMUP_SEC = 25.0
Z_CLIP = 3.0

# (학습형 결합특징)
RIDGE_ALPHA = 1e-3
W_EMA_BETA = 0.8
GATE_COH = 0.50
GATE_CORR = 0.20

# (전역 지연)
GLOBAL_LAG_SEC = None
GLOBAL_LAG_READY_T = 25.0
GLOBAL_LAG_TEXT = ""

# (표시)
MA_WIN = 7
TIMER_POS = (40, VIEW_H - 30)

# ---------------- State ----------------
ts = deque();
dY = deque();
dD = deque();
dW = deque();
dC = deque();
dC_opt = deque()
y0 = d0 = w0 = None
L_EMA = R_EMA = N_EMA = None
prev_L = prev_R = prev_N = None
WC = None
ZFIX = {"dY": None, "dD": None, "dW": None, "dC": None, "GT": None}

# ---- Polarity fix (워밍업 1회) ----
POL_SIGN = {"dY": None, "dD": None, "dW": None}
POL_READY_SEC = 10.0  # 워밍업 길이(초)
POL_APPLIED = False


def try_update_polarity_once():
    # 워밍업 구간(마지막 10s)에서 dY/dD/dW의 부호를 GT에 맞춰 한 번만 고정
    if not SHOW_GT or POL_SIGN["dY"] is not None:
        return
    if len(ts) < 8:
        return

    t_np = np.asarray(ts, np.float32)
    t1 = t_np[-1];
    t0 = t1 - POL_READY_SEC
    if t_np[0] > t0:
        return

    m = (t_np >= t0)
    tw = t_np[m]
    Y = np.asarray(dY, np.float32)[m]
    D = np.asarray(dD, np.float32)[m]
    Wv = np.asarray(dW, np.float32)[m]

    tu, yU, fs = resample_uniform(tw, Y)
    if tu is None:
        return
    _, dU, _ = resample_uniform(tw, D, fs)
    _, wU, _ = resample_uniform(tw, Wv, fs)

    tg = gt_t if GLOBAL_LAG_SEC is None else (gt_t + GLOBAL_LAG_SEC)
    mg = (tg >= tw[0]) & (tg <= tw[-1])
    if not np.any(mg):
        return
    gU = np.interp(tu, tg[mg], gt_resp[mg]).astype(np.float32)

    def prep(z):
        return zscore(bandpass(z, fs, RESP_BAND, RESP_ORDER))

    yU, dU, wU, gU = map(prep, [yU, dU, wU, gU])

    def sgn(a, b):
        r = np.corrcoef(a, b)[0, 1]
        return +1.0 if r >= 0 else -1.0

    POL_SIGN["dY"] = sgn(yU, gU)
    POL_SIGN["dD"] = sgn(dU, gU)
    POL_SIGN["dW"] = sgn(wU, gU)

    # --- 확정 순간, 과거 버퍼도 재부호 (deque는 슬라이스 X) ---
    global POL_APPLIED
    if not POL_APPLIED and POL_SIGN["dY"] is not None:
        def re_sign(dq, s):
            tmp = [v * s for v in dq]  # 리스트로 새로 만든 뒤
            dq.clear()
            dq.extend(tmp)  # 덮어쓰기

        re_sign(dY, POL_SIGN["dY"])
        re_sign(dD, POL_SIGN["dD"])
        re_sign(dW, POL_SIGN["dW"])
        POL_APPLIED = True


# ---------------- Utils ----------------
def put(img, txt, xy, s=0.72, col=(230, 230, 230)):
    x, y = xy;
    cv2.putText(img, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, s, col, 1, cv2.LINE_AA)


def soft_clip_z(z, clip=3.0):
    # tanh로 부드럽게 클립 (형상 보존)
    z = np.asarray(z, np.float32)
    return clip * np.tanh(z / (clip / 1.5))


def gcc_phat_linear(x, y, fs):
    # 선형 상관 기반 GCC-PHAT (제로패딩 n = len(x)+len(y)-1)
    x = np.asarray(x, np.float32).ravel()
    y = np.asarray(y, np.float32).ravel()
    n = int(len(x) + len(y) - 1)
    nfft = 1
    while nfft < n:
        nfft <<= 1
    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, nfft)
    # 선형 상관의 유효 구간 추출
    lag_samp = np.arange(-len(y) + 1, len(x))
    cc = np.concatenate([cc[-(len(y) - 1):], cc[:len(x)]])
    i = int(np.argmax(cc))
    return float(lag_samp[i] / fs)


def delta_lag_window(global_lag, window_lag, clip=0.5):
    if global_lag is None or not np.isfinite(global_lag):
        return float(np.clip(window_lag, -clip, clip))
    return float(np.clip(window_lag - global_lag, -clip, clip))


def moving_average(x, k):
    if k <= 1 or len(x) < k: return x
    if k % 2 == 0: k += 1
    pad = k // 2
    xx = np.pad(np.asarray(x, np.float32), (pad, pad), mode="edge")
    w = np.ones(k) / k
    return np.convolve(xx, w, mode="valid")


def resample_uniform(t, x, fs=None):
    t = np.asarray(t, np.float32);
    x = np.asarray(x, np.float32)
    if len(t) < 3: return None, None, None
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 1e-6: return None, None, None
    if fs is None: fs = 1.0 / dt
    t_u = np.arange(t[0], t[-1] + 1e-6, 1.0 / fs, dtype=np.float32)
    x_u = np.interp(t_u, t, x).astype(np.float32)
    return t_u, x_u, fs


def butter_sos(lo, hi, fs, order=4):
    nyq = fs * 0.5
    lo2 = max(1e-3, lo / nyq);
    hi2 = min(0.999, hi / nyq)
    if hi2 <= lo2 + 1e-3: return None
    from scipy.signal import butter
    return butter(order, [lo2, hi2], btype='bandpass', output='sos')


def bandpass(x, fs, band=RESP_BAND, order=RESP_ORDER):
    if x is None or fs is None: return x
    x = np.asarray(x, np.float32).ravel()
    sos = butter_sos(band[0], band[1], fs, order)
    if sos is None: return x
    from scipy.signal import sosfiltfilt
    padlen = 3 * sos.shape[0]
    if x.size <= padlen: return x
    try:
        return sosfiltfilt(sos, x).astype(np.float32)
    except ValueError:
        return x


def zscore(x):
    x = np.asarray(x, np.float32)
    mu = float(np.mean(x));
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-6: sd = 1.0
    return (x - mu) / sd


def seg_params(L, fs):
    target = int(fs * SPEC_WIN)
    nper = max(16, min(target, max(16, L // 2)))
    nover = min(nper - 1, nper // 2)
    return nper, nover


def safe_fmt(v, fmt="{:.2f}"):
    return fmt.format(v) if v is not None and np.isfinite(v) else "--"


def r_to_p(r, n):
    if not np.isfinite(r) or n is None or n < 4: return None
    den = max(1e-9, 1.0 - r * r)
    t = r * np.sqrt(max(1.0, (n - 2)) / den)
    from math import erf, sqrt
    Phi = 0.5 * (1.0 + erf(abs(t) / sqrt(2.0)))
    p = 2.0 * (1.0 - Phi)
    return float(max(1e-9, min(1.0, p)))


def gcc_phat(x, y, fs):
    n = 1
    L = len(x) + len(y)
    while n < L: n <<= 1
    X = np.fft.rfft(x, n);
    Y = np.fft.rfft(y, n)
    R = X * np.conj(Y);
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n)
    cc = np.concatenate((cc[-(len(x) - 1):], cc[:len(y)]))
    lags = np.arange(-len(x) + 1, len(y))
    i = int(np.argmax(cc))
    return float(lags[i] / fs)


# ---------------- GT ----------------
gt_t, gt_resp = None, None
if SHOW_GT:
    try:
        with h5py.File(H5, "r") as f:
            gt_resp = np.asarray(f["respiration"][:], np.float32)
            gt_t = np.asarray(f["time"][:], np.float32)
    except Exception as e:
        print("[GT disabled]", e);
        SHOW_GT = False


def estimate_rr_from_gt(t0, t1):
    if not SHOW_GT or gt_t is None: return np.nan
    m = (gt_t >= t0) & (gt_t <= t1)
    if np.count_nonzero(m) < 8: return np.nan
    tu, gu, fs = resample_uniform(gt_t[m], gt_resp[m])
    if tu is None: return np.nan
    gu = bandpass(zscore(gu), fs, RR_BAND, RESP_ORDER)
    L = len(gu);
    nper, nover = seg_params(L, fs)
    if nper == 0 or L < nper: return np.nan
    f, P = welch(gu, fs=fs, nperseg=nper, noverlap=nover)
    mb = (f >= RR_BAND[0]) & (f <= RR_BAND[1])
    if np.count_nonzero(mb) < MIN_BAND_BINS: return np.nan
    fb = f[mb];
    Pb = P[mb]
    f0 = float(fb[np.argmax(Pb)])
    return f0 * 60.0


# ---------------- Alignment helpers ----------------
GLOBAL_LAG_SEC = None
GLOBAL_LAG_TEXT = ""


def learn_global_lag_once():
    global GLOBAL_LAG_SEC, GLOBAL_LAG_TEXT
    if GLOBAL_LAG_SEC is not None: return
    if len(ts) < 16 or not SHOW_GT: return
    t_np = np.asarray(ts, np.float32)
    t1 = t_np[-1]
    if t1 < GLOBAL_LAG_READY_T: return
    t0 = max(t_np[0], t1 - GLOBAL_LAG_READY_T)
    m = (t_np >= t0)
    if np.count_nonzero(m) < 64: return
    tw = t_np[m];
    sw = np.asarray(dC, np.float32)[m]
    mgt = (gt_t >= t0) & (gt_t <= t1)
    if not np.any(mgt): return
    tu, su, fs = resample_uniform(tw, sw)
    _, gu, _ = resample_uniform(gt_t[mgt], gt_resp[mgt], fs)
    if tu is None or su is None or gu is None: return
    su = bandpass(zscore(su), fs, RESP_BAND, RESP_ORDER)
    gu = bandpass(zscore(gu), fs, RESP_BAND, RESP_ORDER)
    try:
        lag = gcc_phat_linear(su, gu, fs)
        GLOBAL_LAG_SEC = float(lag)
        GLOBAL_LAG_TEXT = f"GLOBAL LAG {lag:+.02f}s (GCC-PHAT@{int(GLOBAL_LAG_READY_T)}s)"
    except Exception:
        GLOBAL_LAG_SEC = 0.0
        GLOBAL_LAG_TEXT = "GLOBAL LAG +0.00s (fallback)"


# ---------------- Metrics ----------------
def band_metrics(tw, sw):
    if not SHOW_GT or tw is None or len(tw) < 8:
        return {}
    t1 = tw[-1];
    t0 = max(tw[0], t1 - SPEC_WIN)
    m = (tw >= t0)
    if np.count_nonzero(m) < 8:
        return {}
    tw = tw[m].astype(np.float32);
    sw = np.asarray(sw, np.float32)[m]

    tg = gt_t if GLOBAL_LAG_SEC is None else (gt_t + GLOBAL_LAG_SEC)
    mg = (tg >= t0) & (tg <= t1)
    if not np.any(mg):
        return {}

    tu, su, fs = resample_uniform(tw, sw)
    gg = np.interp(tu, tg[mg], gt_resp[mg]).astype(np.float32)

    su = zscore(bandpass(su, fs, RESP_BAND, RESP_ORDER))
    gu = zscore(bandpass(gg, fs, RESP_BAND, RESP_ORDER))
    L = len(su)
    if L < 16:
        return {}

    corr = float(np.corrcoef(su, gu)[0, 1])
    try:
        lag_lin = gcc_phat_linear(su, gu, fs)
    except Exception:
        lag_lin = 0.0
    dlag = delta_lag_window(GLOBAL_LAG_SEC, lag_lin, clip=0.5)

    # Welch
    nper = max(16, int(fs * 8.0))
    nover = nper // 2
    if L < nper:
        coh_val = np.nan
    else:
        f, Cxy = coherence(su, gu, fs=fs, nperseg=nper, noverlap=nover)
        # NaN/Inf 방지 가드
        if not np.all(np.isfinite(Cxy)):
            Cxy = np.nan_to_num(Cxy, nan=0.0, posinf=0.0, neginf=0.0)
        mb = (f >= RR_BAND[0]) & (f <= RR_BAND[1])
        coh_val = float(np.nanmean(Cxy[mb])) if np.any(mb) else np.nan

    rr_bpm = estimate_rr_from_gt(t0, t1)
    pval = r_to_p(corr, L)

    return dict(corr=corr, dlag=dlag, coh=coh_val, rr_bpm=rr_bpm, n=L, p=pval)


# ---------------- ZFIX 기준 산출 ----------------
def try_update_zfix():
    if ZFIX["dY"] is not None: return
    if len(ts) < 3: return
    t_np = np.asarray(ts, np.float32)
    t1 = t_np[-1];
    t0 = t1 - ZFIX_WARMUP_SEC
    if t_np[0] > t0: return

    def ref_mu_sd(t, x):
        m = t >= t0
        tw = t[m];
        sw = np.asarray(x, np.float32)[m]
        tu, su, fs = resample_uniform(tw, sw)
        if tu is None: return None
        su = bandpass(su, fs, RESP_BAND, RESP_ORDER)
        mu, sd = float(np.mean(su)), float(np.std(su) + 1e-9)
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
                ZFIX["GT"] = (float(np.mean(gu)), float(np.std(gu) + 1e-9))


# ---------------- Drawing helpers ----------------
def draw_panel_to(right_img, rect, title, t_seq, sig_seq, color):
    x1, y1, x2, y2 = rect
    panel = right_img[y1:y2, x1:x2];
    panel[:] = BG
    if len(t_seq) < 2:
        return

    t_arr = np.asarray(t_seq, np.float32)
    s_arr = np.asarray(sig_seq, np.float32)

    t1 = t_arr[-1]
    span = min(SPAN_SEC, (t_arr[-1] - t_arr[0]) if len(t_arr) > 1 else SPAN_SEC)
    t0 = t1 - span
    m = t_arr >= t0
    if np.count_nonzero(m) < 4:
        return
    tw = t_arr[m].astype(np.float32);
    sw = s_arr[m].astype(np.float32)
    if MA_WIN > 1:
        sw = moving_average(sw, MA_WIN);
        tw = tw[-len(sw):]

    stats = band_metrics(tw, sw) if SHOW_GT else {}

    # 표준화(표시 전용)
    tu, su, fs = resample_uniform(tw, sw)
    if tu is not None:
        su = zscore(bandpass(su, fs, RESP_BAND, RESP_ORDER))
        sw = np.interp(tw, tu, su).astype(np.float32)
    if SHOW_GT:
        tg = gt_t if GLOBAL_LAG_SEC is None else (gt_t + GLOBAL_LAG_SEC)
        mg = (tg >= tw[0]) & (tg <= tw[-1])
        gv = np.interp(tw, tg[mg], gt_resp[mg]).astype(np.float32) if np.any(mg) else None
        if gv is not None:
            tu2, gu2, fs2 = resample_uniform(tw, gv)
            if tu2 is not None:
                gu2 = zscore(bandpass(gu2, fs2, RESP_BAND, RESP_ORDER))
                gv = np.interp(tw, tu2, gu2).astype(np.float32)
    else:
        gv = None

    # 시각화 스케일링: 소프트 클립
    sw_v = soft_clip_z(sw, Z_CLIP)
    gv_v = soft_clip_z(gv, Z_CLIP) if gv is not None else None

    y_min, y_max = -Z_CLIP, Z_CLIP
    ph, pw = panel.shape[:2];
    margin = 10;
    h = ph - 40;
    w = pw
    cv2.rectangle(panel, (0, 30), (w - 1, ph - 1), (40, 40, 40), 1)

    def to_xy(tvec, yvec):
        xs = (margin + (w - 2 * margin) * np.clip((tvec - t0) / max(1e-6, span), 0, 1)).astype(int)
        norm = (np.clip(yvec, y_min, y_max) - y_min) / (y_max - y_min + 1e-9)
        ys = (ph - 1 - margin - (h - 2 * margin) * np.clip(norm, 0, 1)).astype(int)
        return xs, ys

    for val, label_t in [(-Z_CLIP, f"{-Z_CLIP:.0f}"), (0.0, "0"), (Z_CLIP, f"+{Z_CLIP:.0f}")]:
        y_pos = int(ph - 1 - margin - (h - 2 * margin) * ((val - y_min) / (y_max - y_min)))
        cv2.line(panel, (0, y_pos), (6, y_pos), (140, 140, 140), 1)
        cv2.putText(panel, label_t, (8, y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 175, 175), 1, cv2.LINE_AA)

    xs, ys = to_xy(tw, sw_v)
    if len(xs) >= 2:
        cv2.polylines(panel, [np.stack([xs, ys], 1)], False, color, 2, cv2.LINE_AA)
    if gv_v is not None:
        xs2, ys2 = to_xy(tw, gv_v)
        cv2.polylines(panel, [np.stack([xs2, ys2], 1)], False, (80, 220, 80), 1, cv2.LINE_AA)

    show_dlag = (stats and np.isfinite(stats.get("dlag", np.nan))
                 and (stats.get("coh", 0) >= 0.45) and (abs(stats.get("corr", 0)) >= 0.25))
    dlag_txt = safe_fmt(stats["dlag"], "{:+.02f}") + "s" if show_dlag else "--"

    # --- 간결 HUD (coh / corr / Δlag / RR / p) ---
    if stats:
        line = (f"coh={safe_fmt(stats['coh'])}  "
                f"corr={safe_fmt(stats['corr'], '{:+.2f}')}  "
                f"dlag={dlag_txt}  "
                f"RR={safe_fmt(stats['rr_bpm'], '{:.1f}')} bpm  "
                f"p{('<' + safe_fmt(stats['p'], '{:.003f}')) if stats['p'] is not None else '--'}")
        put(panel, line, (8, 18), 0.50, (200, 200, 200))

    x_cur = int(10 + (panel.shape[1] - 20) * ((t1 - t0) / max(1e-6, span)))
    cv2.line(panel, (x_cur, 30), (x_cur, panel.shape[0] - 1), (180, 180, 80), 1, cv2.LINE_AA)

    key = title.split()[0]
    put(panel, key, (panel.shape[1] - 32, panel.shape[0] - 8), 0.55, color)


# ---------------- MediaPipe Pose ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------- Video ----------------
cap = cv2.VideoCapture(VIDEO);
assert cap.isOpened(), "영상 열기 실패"
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# ---------------- Main loop ----------------
win_name = "COHFACE v2.7a (minimal HUD, GT-aligned)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, VIEW_W, VIEW_H)

prev_ms = 0.0;
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    H, W = frame.shape[:2]

    ms = cap.get(cv2.CAP_PROP_POS_MSEC) or (prev_ms + 1000.0 / max(1.0, fps))
    t = ms / 1000.0
    if t < prev_ms / 1000.0: t = prev_ms / 1000.0 + 1.0 / max(1.0, fps)
    prev_ms = ms
    frame_idx += 1

    vis = frame.copy()

    if (frame_idx % POSE_EVERY_N) == 0:
        rgb_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               (PROC_W, int(H * PROC_W / W)), interpolation=cv2.INTER_AREA)
        res = pose.process(rgb_small)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            h_s, w_s = rgb_small.shape[:2];
            sx, sy = W / float(w_s), H / float(h_s)


            def to_px(pt):
                return np.array([pt.x * w_s * sx, pt.y * h_s * sy], np.float32)


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
                        L_EMA = L if L_EMA is None else EMA_BETA_LMK * L_EMA + (1 - EMA_BETA_LMK) * L
                        R_EMA = R if R_EMA is None else EMA_BETA_LMK * R_EMA + (1 - EMA_BETA_LMK) * R
                        N_EMA = N if N_EMA is None else EMA_BETA_LMK * N_EMA + (1 - EMA_BETA_LMK) * N
                else:
                    L_EMA, R_EMA, N_EMA = L.copy(), R.copy(), N.copy()
                prev_L, prev_R, prev_N = L.copy(), R.copy(), N.copy()

    valid = L_EMA is not None and R_EMA is not None and N_EMA is not None
    if valid:
        if y0 is None or d0 is None or w0 is None:
            y0 = float((L_EMA[1] + R_EMA[1]) / 2.0)
            v = R_EMA - L_EMA;
            wvec = N_EMA - L_EMA
            v2 = float(np.dot(v, v)) + 1e-12;
            tproj = float(np.clip(np.dot(wvec, v) / v2, 0.0, 1.0))
            foot = L_EMA + tproj * v
            d_init = v[0] * wvec[1] - v[1] * wvec[0]
            d0 = float(d_init / (np.sqrt(v2) + 1e-12))
            w0 = float(np.linalg.norm(R_EMA - L_EMA))

        y_now = float((L_EMA[1] + R_EMA[1]) / 2.0)
        v = R_EMA - L_EMA;
        wvec = N_EMA - L_EMA
        v2 = float(np.dot(v, v)) + 1e-12;
        tproj = float(np.clip(np.dot(wvec, v) / v2, 0.0, 1.0))
        foot = L_EMA + tproj * v
        d_now = (v[0] * wvec[1] - v[1] * wvec[0]) / (np.sqrt(v2) + 1e-12)
        w_now = float(np.linalg.norm(R_EMA - L_EMA))

        dY_t = y_now - y0
        dD_t = d_now - d0
        dW_t = w_now - w0

        # --- 워밍업 1회 polarity 고정 시도 + 부호 적용 ---
        try_update_polarity_once()
        if POL_SIGN["dY"] is not None:
            dY_t *= POL_SIGN["dY"]
            dD_t *= POL_SIGN["dD"]
            dW_t *= POL_SIGN["dW"]

        # 이제 부호가 정렬된 값을 버퍼에 저장
        ts.append(t)
        dY.append(dY_t)
        dD.append(dD_t)
        dW.append(dW_t)


        def z_last(buf):
            arr = np.asarray(buf, np.float32)
            if arr.size < 5: return 0.0
            seg = arr[-min(200, len(arr)):]
            sd = float(np.std(seg)) or 1.0
            mu = float(np.mean(seg))
            return float((arr[-1] - mu) / sd)


        dC.append((z_last(dY) + z_last(dD) + z_last(dW)) / 3.0)

        # dC_opt 업데이트 게이트 + Ridge+EMA
        if SHOW_GT and len(ts) >= 32:
            t_np = np.asarray(ts, np.float32)
            m = t_np >= (t_np[-1] - SPEC_WIN)
            stats = band_metrics(t_np[m], np.asarray(dC, np.float32)[m])
            if stats and (stats.get("coh", 0) >= 0.30) and (stats.get("p", 1.0) < 0.05):
                tw = t_np[m]
                Y = np.asarray(dY, np.float32)[m];
                D = np.asarray(dD, np.float32)[m];
                Wv = np.asarray(dW, np.float32)[m]
                tu, yU, fs = resample_uniform(tw, Y)
                _, dU, _ = resample_uniform(tw, D, fs)
                _, wU, _ = resample_uniform(tw, Wv, fs)
                if GLOBAL_LAG_SEC is None:
                    tg = gt_t;
                    gg = gt_resp
                else:
                    tg = gt_t + GLOBAL_LAG_SEC;
                    gg = gt_resp
                mgt = (tg >= tw[0]) & (tg <= tw[-1])
                if tu is not None and np.any(mgt):
                    gU = np.interp(tu, tg[mgt], gg[mgt]).astype(np.float32)
                    yU = zscore(bandpass(yU, fs, RESP_BAND, RESP_ORDER))
                    dU = zscore(bandpass(dU, fs, RESP_BAND, RESP_ORDER))
                    wU = zscore(bandpass(wU, fs, RESP_BAND, RESP_ORDER))
                    gU = zscore(bandpass(gU, fs, RESP_BAND, RESP_ORDER))
                    X = np.stack([yU, dU, wU], 1);
                    Yt = gU
                    XtX = X.T @ X;
                    XtX[np.diag_indices_from(XtX)] += RIDGE_ALPHA
                    try:
                        w_r = np.linalg.solve(XtX, X.T @ Yt)
                        w_r = w_r / (np.linalg.norm(w_r) + 1e-9)
                        WC = w_r if WC is None else (W_EMA_BETA * WC + (1.0 - W_EMA_BETA) * w_r)
                    except Exception:
                        pass


        # 학습과 동일한 도메인(표준화)에서 합성
        def z_now(buf):
            arr = np.asarray(buf, np.float32)
            if arr.size < 5: return 0.0
            seg = arr[-min(200, len(arr)):]
            mu, sd = float(np.mean(seg)), float(np.std(seg)) or 1.0
            return float((arr[-1] - mu) / sd)


        if WC is not None:
            zY, zD, zW = z_now(dY), z_now(dD), z_now(dW)
            dC_opt.append(float(WC[0] * zY + WC[1] * zD + WC[2] * zW))
        else:
            dC_opt.append(0.0)

        while len(ts) and (ts[-1] - ts[0] > HIS_SEC):
            ts.popleft();
            dY.popleft();
            dD.popleft();
            dW.popleft();
            dC.popleft();
            dC_opt.popleft()

        pL = tuple(np.round(L_EMA).astype(int));
        pR = tuple(np.round(R_EMA).astype(int))
        pN = tuple(np.round(N_EMA).astype(int));
        pF = tuple(np.round(foot).astype(int))
        cv2.circle(vis, pL, 6, (40, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, pR, 6, (40, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, pN, 6, (180, 200, 80), -1, cv2.LINE_AA)
        cv2.line(vis, pL, pR, (230, 230, 230), 3, cv2.LINE_AA)
        cv2.line(vis, pN, pF, (120, 200, 120), 2, cv2.LINE_AA)
        put(vis, f"dY:{dY_t:+.2f}px  dD:{dD_t:+.2f}px  dW:{dW_t:+.2f}px", (12, 28), 0.9, HUD_COL)
        if WC is not None:
            put(vis, f"w=[{WC[0]:+.2f},{WC[1]:+.2f},{WC[2]:+.2f}]  (ridge+EMA)", (12, 52), 0.7, HUD_COL)
    else:
        put(vis, "Waiting shoulders + nose...", (12, 28), 0.8, (80, 180, 255))

    try_update_zfix()
    learn_global_lag_once()

    left_w = int(VIEW_W * LEFT_RATIO)
    right_w = VIEW_W - left_w
    left = cv2.resize(vis, (left_w, VIEW_H), interpolation=cv2.INTER_AREA)

    put(left, f"{t:.1f}s", TIMER_POS, 1.0, (210, 210, 210))
    if GLOBAL_LAG_TEXT:
        put(left, GLOBAL_LAG_TEXT, (12, 80), 0.6, (180, 220, 220))

    right = np.full((VIEW_H, right_w, 3), BG, np.uint8)
    Rw, Rh = right_w, VIEW_H
    gap = 8
    cell_w = (Rw - 3 * gap) // 2
    cell_h = (Rh - 3 * gap) // 2
    cells = [
        (gap, gap, gap + cell_w, gap + cell_h),  # dY
        (gap, 2 * gap + cell_h, gap + cell_w, 2 * gap + 2 * cell_h),  # dD
        (2 * gap + cell_w, gap, 2 * gap + 2 * cell_w, gap + cell_h),  # dW
        (2 * gap + cell_w, 2 * gap + cell_h, 2 * gap + 2 * cell_w, 2 * gap + 2 * cell_h),  # dC
    ]

    tnp = np.asarray(ts, np.float32)
    draw_panel_to(right, cells[0], "dY vs GT", tnp, np.asarray(dY, np.float32), COLORS["dY"])
    draw_panel_to(right, cells[1], "dD vs GT", tnp, np.asarray(dD, np.float32), COLORS["dD"])
    draw_panel_to(right, cells[2], "dW vs GT", tnp, np.asarray(dW, np.float32), COLORS["dW"])
    draw_panel_to(right, cells[3], "dC vs GT", tnp, np.asarray(dC, np.float32), COLORS["dC"])

    # dC_opt (얇게)
    x1, y1, x2, y2 = cells[3];
    panel = right[y1:y2, x1:x2]
    if len(ts) > 8:
        t_arr = np.asarray(ts, np.float32);
        s_arr = np.asarray(dC_opt, np.float32)
        t1 = t_arr[-1];
        span = min(SPAN_SEC, (t_arr[-1] - t_arr[0]))
        t0 = t1 - span;
        m = t_arr >= t0
        if np.count_nonzero(m) >= 4:
            tw = t_arr[m];
            sw = s_arr[m]
            if MA_WIN > 1:
                sw = moving_average(sw, MA_WIN);
                tw = tw[-len(sw):]
            tu, su, fs = resample_uniform(tw, sw)
            if tu is not None:
                from numpy import interp

                su = zscore(bandpass(su, fs, RESP_BAND, RESP_ORDER))
                sw = np.interp(tw, tu, su).astype(np.float32)
            if VIZ_MODE == "zfix" and ZFIX.get("dC") is not None:
                mu, sd = ZFIX["dC"];
                sw_v = (sw - mu) / (sd + 1e-9)
            elif VIZ_MODE == "zwin":
                sw_v = zscore(sw)
            else:
                sw_v = sw
            ph, pw = panel.shape[:2];
            margin = 10;
            h = ph - 40;
            w = pw
            y_min, y_max = -Z_CLIP, Z_CLIP
            xs = (margin + (w - 2 * margin) * np.clip((tw - t0) / max(1e-6, span), 0, 1)).astype(int)
            norm = (np.clip(sw_v, y_min, y_max) - y_min) / (y_max - y_min + 1e-9)
            ys = (ph - 1 - margin - (h - 2 * margin) * np.clip(norm, 0, 1)).astype(int)
            if len(xs) >= 2:
                cv2.polylines(panel, [np.stack([xs, ys], 1)], False, (180, 255, 255), 1, cv2.LINE_AA)
                put(panel, "dC_opt (ridge+EMA)", (12, panel.shape[0] - 8), 0.5, (180, 255, 255))

    canvas = np.hstack([left, right])
    cv2.imshow(win_name, canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

pose.close()
cap.release()
cv2.destroyAllWindows()
