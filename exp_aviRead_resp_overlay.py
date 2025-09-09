# aviRead_resp_overlay_clean.py  (최종: 로그 없음 / RR trend / peak fallback / 들숨·날숨)
import cv2, numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, welch, find_peaks, hilbert

# =========================
# 기본 설정
# =========================
VIDEO = "dataset/UBFC-rPPG/subject1/vid.avi"
GT_TXT = "dataset/UBFC-rPPG/subject1/ground_truth.txt"

WIDTH_SCALE = 1.2
SEC_WINDOW_INIT = 8
FS_BVP_FALLBACK = 60

RESP_BAND_INIT = [0.10, 0.50]   # RR 탐색 대역(Hz) — 고정
VIS_BAND = [0.05, 1.00]         # 화면 표시 대역(Hz)

ALPHA_RR  = 0.20                 # RR EMA
ALPHA_QLT = 0.30                 # 품질(Q) EMA

# RR 추세 표시 설정
TREND_SPAN_SEC = 60              # 마지막 60초
TREND_Y_MIN = 6.0                # brpm
TREND_Y_MAX = 30.0               # brpm

# =========================
# 텍스트/레이아웃 유틸
# =========================
def text_h(scale=0.6, thickness=1):
    (_, h), bl = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return h, bl

def put(img, txt, org, scale=0.6, color=(230,230,230), thick=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_title(img, rect, title, scale=0.55):
    x,y,w,h = rect
    th,_ = text_h(scale)
    put(img, title, (x+12, y+10+th), scale, (200,200,200))

def fmt_time_by_mode(frame_idx, fps, mode=0):
    # mode=0 -> "12.34s"  /  mode=1 -> "00:12.34"
    t = frame_idx / max(fps, 1e-6)
    if mode == 0:
        return f"{t:5.2f}s"
    m = int(t // 60)
    s = t - 60*m
    return f"{m:02d}:{s:05.2f}"

# =========================
# 필터
# =========================
def butter_band(fs, lo=0.1, hi=0.5, order=3):
    nyq = 0.5*fs
    lo = max(1e-4, lo)
    hi = min(nyq-1e-4, hi)
    b,a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return b,a

# =========================
# 파형/PSD/Trend 그리기
# =========================
def draw_wave(img, rect, v, title=None, auto_scale=False):
    x,y,w,h = rect
    cv2.rectangle(img, (x,y), (x+w-1,y+h-1), (90,90,90), 1)
    if title: draw_title(img, rect, title, 0.55)

    th,_ = text_h(0.55)
    top  = y + 10 + th + 6
    bot  = y + h - 10
    plot_h = max(1, bot - top)

    # 기준선
    gy = int(top + 0.5*plot_h)
    cv2.line(img, (x+12, gy), (x+w-12, gy), (70,70,70), 1, cv2.LINE_AA)

    if v.size >= 2:
        vv = v.copy()
        if auto_scale:
            q1, q2 = np.quantile(vv, [0.10, 0.90])
            amp = max(1e-6, (q2 - q1)*0.75)
            vv = np.clip((vv - np.median(vv))/amp, -1.0, 1.0)
        else:
            vv = np.clip(vv, -2, 2)/2

        xs = np.linspace(x+12, x+w-12, vv.size).astype(np.int32)
        ys = (top + (0.5 - vv*0.5)*plot_h).astype(np.int32)
        cv2.polylines(img, [np.column_stack([xs,ys]).reshape(-1,1,2)], False, (235,235,235), 2, cv2.LINE_AA)

def draw_psd_and_peak(img, rect, sig, fs, vis_band, rr_band):
    """
    Welch PSD를 그리고 rr_band(예: 0.10-0.50Hz) 내 prominence 피크를 찾는다.
    Returns: (fpk, rr_brpm, peak_power, peak_ratio, warmup_flag)
    """
    x,y,w,h = rect
    cv2.rectangle(img, (x,y), (x+w-1,y+h-1), (90,90,90), 1)
    draw_title(img, rect, f"Resp PSD (Welch) {rr_band[0]:.2f}-{rr_band[1]:.2f} Hz", 0.60)

    n = len(sig)
    if n < int(1.5*fs):
        return np.nan, np.nan, 0.0, 0.0, True

    # 해상도 보강
    nper = int(min(n, max(3*fs, min(8*fs, int(0.9*n)))))
    nover = int(0.5*nper)
    nfft  = max(2048, 4*int(2**np.ceil(np.log2(max(64, nper)))))
    f, Pxx = welch(sig - np.mean(sig), fs=fs, nperseg=nper, noverlap=nover, window="hann", nfft=nfft)
    Pxx = np.maximum(Pxx, 1e-18)

    fmin_vis, fmax_vis = vis_band
    fmin_rr,  fmax_rr  = rr_band
    i0, i1 = np.searchsorted(f, fmin_vis), np.searchsorted(f, fmax_vis)
    j0, j1 = np.searchsorted(f, fmin_rr),  np.searchsorted(f, fmax_rr)

    # 표시용 보간(탐색은 원 스펙트럼 사용)
    f_vis, P_vis = f[i0:i1], Pxx[i0:i1]
    if f_vis.size >= 2:
        fx = np.linspace(fmin_vis, fmax_vis, 600)
        Pi = np.interp(fx, f_vis, P_vis)
        Pn = Pi / (Pi.max() + 1e-12)
        th_title,_ = text_h(0.60); th_tick,_ = text_h(0.48)
        top = y + 10 + th_title + 8
        bot = y + h - (10 + th_tick + 14)
        ph  = max(1, bot - top)
        xs = np.linspace(x+12, x+w-12, fx.size).astype(np.int32)
        ys = (top + (1 - Pn)*ph).astype(np.int32)
        cv2.polylines(img, [np.column_stack([xs,ys]).reshape(-1,1,2)], False, (215,185,200), 2, cv2.LINE_AA)

    # --- RR 대역 피크 탐색 (피크 없을 때 백업: 무게중심 + ACF) ---
    f_rr, P_rr = f[j0:j1], Pxx[j0:j1]
    if f_rr.size == 0:
        return np.nan, np.nan, 0.0, 0.0, False

    prom_base = max(np.median(P_rr) * 0.4, 1e-12)
    peaks, _ = find_peaks(P_rr, prominence=prom_base)

    def _fallback_rr_by_centroid(fr, Pr):
        w = Pr - Pr.min()
        if np.sum(w) <= 0:
            return np.nan
        f_c = np.sum(fr * w) / np.sum(w)
        return 60.0 * f_c

    def _fallback_rr_by_acf(sig_in, fs, lo, hi):
        try:
            b, a = butter_band(fs, lo, hi, 3)
            s = filtfilt(b, a, sig_in, method="gust")
        except Exception:
            s = sig_in
        s = s - np.mean(s)
        acf = np.correlate(s, s, mode="full")[len(s) - 1:]
        lag_lo = int(np.floor(fs / hi))
        lag_hi = int(np.ceil(fs / lo))
        lag_hi = min(lag_hi, len(acf))
        if lag_hi - lag_lo < 3:
            return np.nan
        k = lag_lo + np.argmax(acf[lag_lo:lag_hi])
        return 60.0 * (fs / max(1, k))

    if peaks.size:
        k = peaks[np.argmax(P_rr[peaks])]
        fpk = float(f_rr[k]); rr = 60.0 * fpk
        peak_power = float(P_rr[k])
        band_power = float(np.trapz(P_rr, f_rr)) + 1e-12
        peak_ratio = float(np.clip(peak_power / band_power, 0.0, 1.0))
    else:
        rr_c = _fallback_rr_by_centroid(f_rr, P_rr)
        rr_a = _fallback_rr_by_acf(sig, fs, fmin_rr, fmax_rr)
        candidates = [x for x in (rr_c, rr_a) if (x is not None and np.isfinite(x))]
        if len(candidates) == 0:
            return np.nan, np.nan, 0.0, 0.0, False
        rr = float(np.mean(candidates)); fpk = rr / 60.0
        peak_power = float(np.max(P_rr))
        band_power = float(np.trapz(P_rr, f_rr)) + 1e-12
        peak_ratio = float(np.clip(peak_power / band_power, 0.0, 1.0))

    # 수직선 + 라벨
    th_title,_ = text_h(0.60); th_tick,_ = text_h(0.48)
    top = y + 10 + th_title + 8
    bot = y + h - (10 + th_tick + 14)
    xp = int(x+12 + (fpk - fmin_vis)/(fmax_vis - fmin_vis) * (w-24))
    cv2.line(img, (xp, top), (xp, bot), (140,170,255), 1, cv2.LINE_AA)
    put(img, f"{fpk:.2f} Hz ({rr:.1f} brpm)", (x+w-12-220, top-6), 0.55, (170,200,255))

    # 눈금
    ticks = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    px = x + 12 + (ticks - fmin_vis) / (fmax_vis - fmin_vis) * (w - 24)
    min_gap = 85; last = -1e9
    for hz, xx in zip(ticks, px):
        xx = int(xx)
        cv2.line(img, (xx, bot), (xx, bot+6), (120,120,120), 1, cv2.LINE_AA)
        label = f"{hz:.1f}Hz/{int(60*hz)}brpm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        if xx - last < min_gap: continue
        put(img, label, (xx - tw//2, bot + th + 6), 0.48, (180,180,180))
        last = xx

    return fpk, rr, peak_power, peak_ratio, False

def draw_rr_trend(img, rect, ts, rr_inst, rr_ema, t_now, span_sec, y_min=6.0, y_max=30.0):
    """RR 추세(instant/EMA)를 최근 span_sec 동안 시계열로 표시"""
    x,y,w,h = rect
    cv2.rectangle(img, (x,y), (x+w-1,y+h-1), (90,90,90), 1)
    draw_title(img, rect, f"RR trend (last {int(span_sec)} s)", 0.55)

    th,_ = text_h(0.55)
    top  = y + 10 + th + 14   # 제목과 겹침 방지
    bot  = y + h - 18
    left = x + 12
    right= x + w - 12
    plot_h = max(1, bot - top)

    # 표시 구간 선택
    t0 = t_now - span_sec
    t = np.asarray(ts, dtype=np.float32)
    m = (t >= t0) & (t <= t_now)
    if not np.any(m): return
    t = t[m]; r1 = np.asarray(rr_inst, dtype=np.float32)[m]
    r2 = np.asarray(rr_ema,  dtype=np.float32)[m]

    # 보이는 구간에서 y 스케일 보완
    r_all = np.concatenate([r1[np.isfinite(r1)], r2[np.isfinite(r2)]], axis=0)
    if r_all.size >= 5:
        q1, q9 = np.percentile(r_all, [5,95])
        pad = max(1.0, 0.2*(q9-q1))
        y_min = min(y_min, q1 - pad)
        y_max = max(y_max, q9 + pad)
        if y_max - y_min < 5:
            mid = 0.5*(y_min + y_max)
            y_min, y_max = mid-2.5, mid+2.5

    # 정상 범위(12–20 brpm) 음영
    norm_lo, norm_hi = 12.0, 20.0
    lo = max(y_min, norm_lo); hi = min(y_max, norm_hi)
    if hi > lo:
        y_lo = int(top + (1 - (lo - y_min)/(y_max - y_min)) * plot_h)
        y_hi = int(top + (1 - (hi - y_min)/(y_max - y_min)) * plot_h)
        overlay = img.copy()
        cv2.rectangle(overlay, (left, y_hi), (right, y_lo), (80,80,80), -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    # 격자 + 눈금
    for v in [6,12,18,24,30]:
        if v < y_min or v > y_max: continue
        gy = int(top + (1 - (v - y_min)/(y_max - y_min)) * plot_h)
        cv2.line(img, (left, gy), (right, gy), (70,70,70), 1, cv2.LINE_AA)
        put(img, f"{v} brpm", (left, gy-4), 0.48, (140,140,140))

    def to_xy(tt, rr):
        rr = np.clip(rr, y_min, y_max)
        xs = (left + (tt - t0)/span_sec * (right-left)).astype(np.int32)
        ys = (top  + (1 - (rr - y_min)/(y_max-y_min)) * plot_h).astype(np.int32)
        return xs, ys

    # instant(회색), EMA(주황)
    if np.any(np.isfinite(r1)):
        xs, ys = to_xy(t[np.isfinite(r1)], r1[np.isfinite(r1)])
        if xs.size >= 2:
            cv2.polylines(img, [np.column_stack([xs,ys]).reshape(-1,1,2)], False, (200,200,200), 1, cv2.LINE_AA)
    if np.any(np.isfinite(r2)):
        xs, ys = to_xy(t[np.isfinite(r2)], r2[np.isfinite(r2)])
        if xs.size >= 2:
            cv2.polylines(img, [np.column_stack([xs,ys]).reshape(-1,1,2)], False, (255,200,150), 2, cv2.LINE_AA)

    # 우상단 현재값 라벨
    rr_now  = r2[np.isfinite(r2)][-1] if np.any(np.isfinite(r2)) else np.nan
    rr_now2 = r1[np.isfinite(r1)][-1] if np.any(np.isfinite(r1)) else np.nan
    label = f"EMA:{rr_now:.1f} brpm" if np.isfinite(rr_now) else "EMA:--"
    if np.isfinite(rr_now2): label += f"  inst:{rr_now2:.1f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    put(img, label, (right - tw, top - 6), 0.55, (230,230,230))

def draw_gauge(img, rect, mag_norm):
    x,y,w,h = rect
    cv2.rectangle(img, (x,y), (x+w-1,y+h-1), (90,90,90), 1)
    put(img, "quality (Q)", (x+8, y+20), 0.52, (200,200,200))
    cv2.rectangle(img, (x+12, y+30), (x+w-12, y+h-16), (90,90,90), 1)
    mag = float(np.clip(mag_norm, 0, 1))
    fill = int((h-46) * mag)
    cv2.rectangle(img, (x+13, y+h-17-fill), (x+w-13, y+h-17), (255,180,180), -1)

# =========================
# 들숨/날숨 판정
# =========================
def estimate_resp_phase(resp_sig, fs, flip=False):
    """
    resp_sig(밴드패스·정규화 파형)에서 현재가 들숨/날숨인지 판정.
    Returns: label('INHALE'|'EXHALE'|'WARMUP'), inhale_flag(0/1 or nan), phase_pct[0..1)
    """
    if resp_sig.size < max(8, int(0.5*fs)):
        return "WARMUP", np.nan, 0.0

    d = np.gradient(resp_sig).astype(np.float32)
    k = max(3, int(0.15*fs))  # 150ms 평활
    if k > 1:
        ker = np.ones(k, np.float32)/k
        d = np.convolve(d, ker, mode="same")
    inhale = (d[-1] >= 0.0)
    if flip: inhale = not inhale

    try:
        phi = np.unwrap(np.angle(hilbert(resp_sig))).astype(np.float32)
        phase_pct = float((phi[-1]/(2*np.pi)) % 1.0)
    except Exception:
        phase_pct = 0.0

    return ("INHALE" if inhale else "EXHALE"), (1.0 if inhale else 0.0), phase_pct

# =========================
# 데이터 로드
# =========================
with open(GT_TXT, "r") as f:
    lines = [ln.strip() for ln in f if ln.strip()]
bvp = np.fromstring(lines[0], sep=" ", dtype=np.float32)
hr  = np.fromstring(lines[1], sep=" ", dtype=np.float32) if len(lines) > 1 else None
bvp = bvp[np.isfinite(bvp)]
if bvp.size and np.std(bvp) > 1e-6:
    bvp = (bvp - bvp.mean()) / (bvp.std() + 1e-8)

cap = cv2.VideoCapture(VIDEO); assert cap.isOpened(), "영상 열기 실패"
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = max(1e-6, n_frames / fps)

fs_bvp = int(round(bvp.size / duration)) if bvp.size else FS_BVP_FALLBACK
fs_bvp = FS_BVP_FALLBACK if not (30 <= fs_bvp <= 240) else fs_bvp

# =========================
# 버퍼/상태
# =========================
sec_window = SEC_WINDOW_INIT
buf_len = max(1, int(sec_window * fs_bvp))
buf = np.zeros(buf_len, np.float32)
bvp_idx_prev = 0

resp_band = RESP_BAND_INIT[:]
bpf = butter_band(fs_bvp, resp_band[0], resp_band[1], 3)

rr_ema, qlt_ema = None, 0.0
time_mode = 0           # 0: "12.34s", 1: "MM:SS.ss"
auto_scale = False
show_help  = False      # H로 토글
phase_flip = False      # P로 반전

# RR 추세 버퍼
trend_t = deque()
trend_rr = deque()
trend_rr_ema = deque()

i = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok: break

        if WIDTH_SCALE != 1.0:
            H,W = frame.shape[:2]
            frame = cv2.resize(frame, (int(W*WIDTH_SCALE), int(H*WIDTH_SCALE)), interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]

        # ----- 버퍼 업데이트 -----
        bvp_idx = int(min(bvp.size, round((i+1)/fps * fs_bvp)))
        if bvp_idx > bvp_idx_prev and bvp.size:
            new = bvp[bvp_idx_prev:bvp_idx]; shift = new.size
            if shift >= buf_len: buf[:] = new[-buf_len:]
            else:               buf = np.roll(buf, -shift); buf[-shift:] = new
            bvp_idx_prev = bvp_idx
        filled = min(bvp_idx_prev, buf_len)
        sig = buf[-filled:] if filled > 0 else buf[:1]

        # 표준화 + 호흡대역 필터
        wave = (sig - sig.mean())/(sig.std()+1e-8) if sig.std()>1e-6 else sig
        bpf = butter_band(fs_bvp, resp_band[0], resp_band[1], 3)
        resp = filtfilt(*bpf, sig, method="gust") if sig.std()>1e-6 and sig.size>10 else sig
        resp_n = (resp - resp.mean())/(resp.std()+1e-8) if resp.std()>1e-6 else resp

        # ----- 하단 패널 -----
        pad = 18
        gauge_w = 110
        wave_h, resp_h, fft_h, trend_h = 120, 120, 240, 120
        panel_h = wave_h + resp_h + fft_h + trend_h + pad*5
        panel = np.full((panel_h, W, 3), (16,16,16), np.uint8)

        draw_gauge(panel, (pad, pad, gauge_w, panel_h - pad*2), qlt_ema)

        cx = pad + gauge_w + pad
        cw = W - cx - pad
        y  = pad

        draw_wave(panel, (cx, y, cw, wave_h), wave, "BVP (recent window)", auto_scale)
        y += wave_h + pad
        draw_wave(panel, (cx, y, cw, resp_h), resp_n, "Resp-band 0.10-0.50 Hz", auto_scale)
        y += resp_h + pad

        fpk, rr, peak, peak_ratio, warmup = draw_psd_and_peak(panel, (cx, y, cw, fft_h), sig, fs_bvp, VIS_BAND, resp_band)
        y += fft_h + pad

        # ----- EMA/품질 업데이트 -----
        if np.isfinite(rr):
            rr_ema = rr if rr_ema is None else (1-ALPHA_RR)*rr_ema + ALPHA_RR*rr
        if peak_ratio > 0:
            qlt_ema = (1-ALPHA_QLT)*qlt_ema + ALPHA_QLT*peak_ratio

        # ----- 들숨/날숨 판정 -----
        phase_label, inhale_flag, phase_pct = estimate_resp_phase(resp_n, fs_bvp, flip=phase_flip)

        # ----- RR 추세 버퍼 업데이트 -----
        t_now = i / fps
        trend_t.append(t_now)
        trend_rr.append(rr if np.isfinite(rr) else np.nan)
        trend_rr_ema.append(rr_ema if rr_ema is not None else np.nan)
        while len(trend_t) and (t_now - trend_t[0] > TREND_SPAN_SEC):
            trend_t.popleft(); trend_rr.popleft(); trend_rr_ema.popleft()

        # ----- RR trend 그리기 -----
        draw_rr_trend(panel, (cx, y, cw, trend_h), list(trend_t), list(trend_rr), list(trend_rr_ema),
                      t_now, TREND_SPAN_SEC, TREND_Y_MIN, TREND_Y_MAX)

        # ========== 상태바(2줄) ==========
        def _fmt(x, spec="{:.1f}"):
            return (spec.format(x) if (x is not None and np.isfinite(x)) else "--")

        def put_fit(img, text, org, max_w, init_scale, color, thick=1, min_scale=0.52):
            scale = init_scale
            while True:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                if tw <= max_w or scale <= min_scale: break
                scale -= 0.02
            put(img, text, org, scale, color, thick)

        left_x = cx; right_x = W - 12
        max_w1 = right_x - left_x; max_w2 = right_x - left_x

        th1,_ = text_h(0.86)
        th2,_ = text_h(0.70)
        footer_h = th1 + th2 + 40
        footer = np.full((footer_h, W, 3), (12,12,12), np.uint8)

        # line1: 시간 + HR_gt
        hr_val = None
        if hr is not None and hr.size:
            j = min(int(round(i)), hr.size-1)
            hr_val = float(hr[j])
        time_txt = fmt_time_by_mode(i, fps, time_mode)
        line1 = f"t:{time_txt}   HR_gt:{_fmt(hr_val)} bpm"
        put_fit(footer, line1, (left_x, 16 + th1), max_w1, 0.86, (230,230,230))

        # line2: RR / phase / fs_bvp / Q / peak
        fpk_disp = fpk if (fpk is not None and np.isfinite(fpk)) else ((rr/60.0) if (rr is not None and np.isfinite(rr)) else None)
        rr_disp  = (60.0*fpk_disp) if (fpk_disp is not None and np.isfinite(fpk_disp)) else None
        line2 = (
            f"RR:{_fmt(rr, '{:.1f}')}/{_fmt(rr_ema, '{:.1f}')} brpm"
            f"   phase:{phase_label}"
            f"   fs_bvp:{fs_bvp}Hz"
            f"   Q:{_fmt(qlt_ema, '{:.2f}')}"
            f"   peak:{_fmt(fpk_disp, '{:.2f}')} Hz ({_fmt(rr_disp, '{:.1f}')} brpm)"
            f"{'   WARMUP' if warmup else ''}"
        )
        put_fit(footer, line2, (left_x, 26 + th1 + th2), max_w2, 0.70, (170,200,255))

        # 도움말(토글)
        if show_help:
            help_h = 72
            help_box = np.full((help_h, W, 3), (8,8,8), np.uint8)
            put(help_box, "Keys: 2=2s  8=8s  1=12s  A:auto-scale  T:time  P:phase-flip  H:hide",
                (12, 28), 0.58, (200,200,200))
            put(help_box, "Trend: gray=instant, orange=EMA. If peak is '--', extend window or stabilize.",
                (12, 56), 0.58, (180,180,180))
            spacer = np.full((8, W, 3), (8,8,8), np.uint8)
            footer = np.vstack([footer, help_box, spacer])
        else:
            spacer = np.full((12, W, 3), (8,8,8), np.uint8)
            footer = np.vstack([footer, spacer])

        # 최종 캔버스
        canvas = np.vstack([frame, panel, footer])
        cv2.imshow("video + RR (ESC to exit)", canvas)

        # 키 핸들링
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == 27: break  # ESC
        if key in (ord('t'), ord('T')): time_mode = 1 - time_mode
        if key in (ord('p'), ord('P')): phase_flip = not phase_flip
        if key in (ord('2'), ord('8'), ord('1')):
            if key == ord('2'): sec_window = 2
            elif key == ord('8'): sec_window = 8
            elif key == ord('1'): sec_window = 12
            buf_len = max(1, int(sec_window * fs_bvp))
            buf = np.zeros(buf_len, np.float32); bvp_idx_prev = 0
        if key in (ord('a'), ord('A')): auto_scale = not auto_scale
        if key in (ord('h'), ord('H')): show_help = not show_help

        i += 1

finally:
    cap.release(); cv2.destroyAllWindows()
