# -*- coding: utf-8 -*-
"""
COHFACE V0_exp_delay (Auto Runner, common-time alignment, single-file)
- 실행 인자 없이 'Run'만으로 전체(1~10, 0~3) 자동 수행
- 각 클립 CSV(pairs/), summary.csv, overall_stats.json 생성
"""

import os, sys, csv, json, time
import numpy as np
import cv2, h5py
from collections import deque
from scipy.signal import welch, find_peaks

# ---------------- User defaults ----------------
DATA_ROOT = "/mnt/hdd18t/rppg_dataset/raw/cohface"
AUTO_SUBJECTS = list(range(1, 41))     # 1~10
AUTO_SESSIONS = list(range(0, 4))      # 0~3
OUT_DIR_ROOT  = "./exp_delay_out"
SAVE_PAIRS_CSV = True

# DSP/추정 파라미터
FS_RESAMP = 256.0            # resample fs(Hz)
RESP_BAND = (0.08, 0.60)     # 5–36 bpm
RESP_ORDER = 4
LAG_CLIP = 0.5               # 전역 래그 클리핑(±s)

# 헤드리스/백엔드
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# --------------- MediaPipe (pose) ---------------
import mediapipe as mp
mp_pose = mp.solutions.pose

EMA_BETA_LMK = 0.7
MAX_LMK_JUMP = 40.0
POSE_EVERY_N = 2
PROC_W = 640

# ---------------- DSP utils ----------------
def resample_uniform(t, x, fs=FS_RESAMP):
    t = np.asarray(t, np.float32); x = np.asarray(x, np.float32)
    if t.size < 3: return None, None, None
    t_u = np.arange(t[0], t[-1] + 1e-6, 1.0/fs, dtype=np.float32)
    x_u = np.interp(t_u, t, x).astype(np.float32)
    return t_u, x_u, fs

def butter_sos(lo, hi, fs, order=RESP_ORDER):
    from scipy.signal import butter
    nyq = fs*0.5
    lo2 = max(1e-3, lo/nyq); hi2 = min(0.999, hi/nyq)
    if hi2 <= lo2 + 1e-3: return None
    return butter(order, [lo2,hi2], btype="bandpass", output="sos")

def bandpass(x, fs, band=RESP_BAND, order=RESP_ORDER):
    if x is None or fs is None: return x
    from scipy.signal import sosfiltfilt
    sos = butter_sos(band[0], band[1], fs, order)
    if sos is None: return x
    x = np.asarray(x, np.float32).ravel()
    padlen = 3 * sos.shape[0]
    if x.size <= padlen: return x
    try:
        return sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return x

def zscore(x):
    x = np.asarray(x, np.float32)
    m = float(np.mean(x)); s = float(np.std(x))
    if not np.isfinite(s) or s < 1e-6: s = 1.0
    return (x - m)/s

def gcc_phat_linear(x, y, fs):
    x = np.asarray(x, np.float32).ravel(); y = np.asarray(y, np.float32).ravel()
    n = int(len(x)+len(y)-1); nfft=1
    while nfft < n: nfft <<= 1
    X = np.fft.rfft(x, nfft); Y = np.fft.rfft(y, nfft)
    R = X * np.conj(Y); R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, nfft)
    lag_samp = np.arange(-len(y)+1, len(x))
    cc = np.concatenate([cc[-(len(y)-1):], cc[:len(x)]])
    i = int(np.argmax(cc))
    return float(lag_samp[i] / fs)

def estimate_rr_from_gt(gt_t, gt_resp):
    if gt_t.size < 32: return np.nan, np.nan
    tu, gu, fs = resample_uniform(gt_t, gt_resp, FS_RESAMP)
    gu = bandpass(zscore(gu), fs)
    L = len(gu);
    if L < int(fs*8): return np.nan, np.nan
    nper = max(128, int(fs*8)); nover = nper//2
    f, P = welch(gu, fs=fs, nperseg=nper, noverlap=nover)
    mb = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    if np.count_nonzero(mb) < 3: return np.nan, np.nan
    fb = f[mb]; f0 = float(fb[np.argmax(P[mb])])
    rr_bpm = f0*60.0; period = 1.0/max(1e-6, f0)
    return rr_bpm, period

def wrap_phase_deg(deg):
    return (deg + 180.0) % 360.0 - 180.0

# ---------------- Pose → dW/dY/dD ----------------
def extract_displacements(video_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"영상 열기 실패: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_ms = 0.0; frame_idx = 0

    L_EMA = R_EMA = N_EMA = None
    prev_L = prev_R = prev_N = None
    y0 = d0 = w0 = None

    ts, dW, dY, dD = [], [], [], []

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break
            H, W = frame.shape[:2]
            ms = cap.get(cv2.CAP_PROP_POS_MSEC) or (prev_ms + 1000.0/max(1.0,fps))
            t = ms/1000.0
            if t < prev_ms/1000.0: t = prev_ms/1000.0 + 1.0/max(1.0,fps)
            prev_ms = ms; frame_idx += 1

            if (frame_idx % POSE_EVERY_N) != 0:
                continue

            rgb_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   (PROC_W, int(H*PROC_W/W)), interpolation=cv2.INTER_AREA)
            res = pose.process(rgb_small)
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            h_s, w_s = rgb_small.shape[:2]
            sx, sy = W/float(w_s), H/float(h_s)
            def to_px(pt):
                return np.array([pt.x*w_s*sx, pt.y*h_s*sy], np.float32)
            L = to_px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
            R = to_px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            N = to_px(lm[mp_pose.PoseLandmark.NOSE])
            visL = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility
            visR = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility
            visN = lm[mp_pose.PoseLandmark.NOSE].visibility
            if min(visL, visR, visN) <= 0.5:
                continue

            if prev_L is not None:
                if (np.linalg.norm(L - prev_L) > MAX_LMK_JUMP or
                    np.linalg.norm(R - prev_R) > MAX_LMK_JUMP or
                    np.linalg.norm(N - prev_N) > MAX_LMK_JUMP):
                    prev_L, prev_R, prev_N = L.copy(), R.copy(), N.copy()
                    continue
                else:
                    L_EMA = L if L_EMA is None else EMA_BETA_LMK*L_EMA + (1-EMA_BETA_LMK)*L
                    R_EMA = R if R_EMA is None else EMA_BETA_LMK*R_EMA + (1-EMA_BETA_LMK)*R
                    N_EMA = N if N_EMA is None else EMA_BETA_LMK*N_EMA + (1-EMA_BETA_LMK)*N
            else:
                L_EMA, R_EMA, N_EMA = L.copy(), R.copy(), N.copy()
            prev_L, prev_R, prev_N = L.copy(), R.copy(), N.copy()

            if y0 is None or d0 is None or w0 is None:
                y0 = float((L_EMA[1] + R_EMA[1]) / 2.0)
                v = R_EMA - L_EMA; wvec = N_EMA - L_EMA
                v2 = float(np.dot(v, v)) + 1e-12
                d_init = v[0]*wvec[1] - v[1]*wvec[0]
                d0 = float(d_init/(np.sqrt(v2)+1e-12))
                w0 = float(np.linalg.norm(R_EMA - L_EMA))

            y_now = float((L_EMA[1] + R_EMA[1]) / 2.0)
            v = R_EMA - L_EMA; wvec = N_EMA - L_EMA
            v2 = float(np.dot(v, v)) + 1e-12
            d_now = (v[0]*wvec[1] - v[1]*wvec[0])/(np.sqrt(v2)+1e-12)
            w_now = float(np.linalg.norm(R_EMA - L_EMA))

            dY.append(y_now - y0)
            dD.append(d_now - d0)
            dW.append(w_now - w0)
            ts.append(t)

    return np.asarray(ts, np.float32), np.asarray(dW, np.float32), np.asarray(dY, np.float32), np.asarray(dD, np.float32)

# ---------------- GT & Alignment ----------------
def load_gt(h5_path):
    with h5py.File(h5_path, "r") as f:
        gt_resp = np.asarray(f["respiration"][:], np.float32)
        gt_t    = np.asarray(f["time"][:], np.float32)
    return gt_t, gt_resp

def estimate_global_lag(ts, dW, dY, dD, gt_t, gt_resp, clip=LAG_CLIP):
    tu, wU, fs = resample_uniform(ts, dW, FS_RESAMP)
    _,  yU, _  = resample_uniform(ts, dY, fs)
    _,  dU, _  = resample_uniform(ts, dD, fs)
    cU = (zscore(bandpass(wU, fs)) + zscore(bandpass(yU, fs)) + zscore(bandpass(dU, fs)))/3.0
    tg, gU, _  = resample_uniform(gt_t, gt_resp, fs)
    gU = zscore(bandpass(gU, fs))
    try:
        lag = float(np.clip(gcc_phat_linear(cU, gU, fs), -clip, clip))
    except Exception:
        lag = 0.0
    return lag

def align_common_time(tu, wU, yU, dU, tg, gU, fs):
    """tu, tg(정렬 적용 후)를 공통 시간축 tC로 맞춰 길이 동일하게 반환"""
    t0 = max(float(tu[0]), float(tg[0]))
    t1 = min(float(tu[-1]), float(tg[-1]))
    if t1 - t0 < 5.0 / fs:
        # 겹침이 너무 작으면 실패 → 원본 반환
        return tu, wU, yU, dU, tg, gU, False
    tC = np.arange(t0, t1 + 1e-6, 1.0/fs, dtype=np.float32)
    def interp(t, x): return np.interp(tC, t, x).astype(np.float32)
    wC = interp(tu, wU); yC = interp(tu, yU); dC = interp(tu, dU); gC = interp(tg, gU)
    return tC, wC, yC, dC, tC, gC, True

def sign_align(sig_u, g_u):
    r = np.corrcoef(sig_u, g_u)[0,1]
    return sig_u if (not np.isfinite(r) or r >= 0) else (-sig_u)

# ---------------- Peak pairing ----------------
def pair_peaks(t_sig, sig_u, t_gt, g_u, period, fs):
    """t_sig와 t_gt는 같은 시간축이어도 무방"""
    min_dist = max(1, int(fs * 0.6*period))
    prom_sig = max(0.1, 0.15*np.std(sig_u))
    prom_gt  = max(0.1, 0.20*np.std(g_u))
    pk_gt, _ = find_peaks(g_u, distance=min_dist, prominence=prom_gt)
    pk_s , _ = find_peaks(sig_u, distance=min_dist, prominence=prom_sig)
    if pk_gt.size == 0 or pk_s.size == 0:
        return dict(n=0, pairs=[])
    tg = t_gt[pk_gt]; ts = t_sig[pk_s]
    pairs = []
    for i, tgi in enumerate(tg):
        j = int(np.argmin(np.abs(ts - tgi)))
        dt = float(ts[j] - tgi)
        if abs(dt) <= 0.5*period:
            ph = wrap_phase_deg(360.0 * dt / max(1e-6, period))
            pairs.append((i, tgi, ts[j], dt, ph))
    return dict(n=len(pairs), pairs=pairs)

def summarize_pairs(pairs):
    if pairs["n"] == 0:
        return dict(n=0, dt_med=np.nan, dt_iqr=np.nan, ph_med=np.nan, ph_iqr=np.nan)
    arr = np.asarray(pairs["pairs"], np.float32)  # (i, t_gt, t_sig, dt, ph)
    dt = arr[:,3]; ph = arr[:,4]
    def iqr(x):
        q1, q3 = np.percentile(x, [25, 75])
        return float(q3 - q1)
    return dict(
        n=int(arr.shape[0]),
        dt_med=float(np.median(dt)), dt_iqr=iqr(dt),
        ph_med=float(np.median(ph)), ph_iqr=iqr(ph)
    )

# ---------------- Experiment (one clip) ----------------
def run_experiment(video_path, h5_path, out_csv=None, verbose=True):
    if verbose: print(f"[run] {video_path}")
    print("[1/4] 영상에서 dW/dY/dD 추출 중 ...")
    ts, dW, dY, dD = extract_displacements(video_path)

    print("[2/4] GT(H5) 로드 중 ...")
    gt_t, gt_resp = load_gt(h5_path)

    print("[3/4] 전역 래그 추정 ...")
    lag = estimate_global_lag(ts, dW, dY, dD, gt_t, gt_resp, clip=LAG_CLIP)
    print(f"    → GLOBAL LAG {lag:+.3f}s (applied)")

    rr_bpm, period = estimate_rr_from_gt(gt_t, gt_resp)
    if not np.isfinite(period) or period <= 0: period = 4.0

    print("[4/4] 밴드패스+zscore → 피크 매칭 ...")
    tu, wU, fs = resample_uniform(ts, dW, FS_RESAMP)
    _,  yU, _  = resample_uniform(ts, dY, fs)
    _,  dU, _  = resample_uniform(ts, dD, fs)
    wU = zscore(bandpass(wU, fs)); yU = zscore(bandpass(yU, fs)); dU = zscore(bandpass(dU, fs))
    tg, gU, _  = resample_uniform(gt_t + lag, gt_resp, fs); gU = zscore(bandpass(gU, fs))

    # ---- 공통 시간축 정렬(길이 동일화) ----
    tS, wS, yS, dS, tG, gS, ok_align = align_common_time(tu, wU, yU, dU, tg, gU, fs)
    if not ok_align:
        # 겹침이 아주 작아도 최소 보간으로 맞춤 (tg 범위 내 tu 마스크)
        mask = (tu >= tg[0]) & (tu <= tg[-1])
        if not np.any(mask):  # 마지막 가드
            raise RuntimeError("신호/GT 겹침 구간이 너무 짧습니다.")
        tS = tu[mask]; wS = wU[mask]; yS = yU[mask]; dS = dU[mask]
        gS = np.interp(tS, tg, gU).astype(np.float32); tG = tS

    # 부호 정렬(반전 방지)
    wS = sign_align(wS, gS); yS = sign_align(yS, gS); dS = sign_align(dS, gS)

    P_w = pair_peaks(tS, wS, tG, gS, period, fs); S_w = summarize_pairs(P_w)
    P_y = pair_peaks(tS, yS, tG, gS, period, fs); S_y = summarize_pairs(P_y)
    P_d = pair_peaks(tS, dS, tG, gS, period, fs); S_d = summarize_pairs(P_d)

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sig","idx_gt","t_gt","t_sig","delta_t_sec","phase_deg"])
            for sig, P in [("dW", P_w), ("dY", P_y), ("dD", P_d)]:
                for (i, tgt, tsig, dt, ph) in P["pairs"]:
                    w.writerow([sig, int(i), float(tgt), float(tsig), float(dt), float(ph)])

    duration = float(ts[-1]-ts[0]) if ts.size else 0.0
    # 정렬된 GT로 피크 수 집계
    pk_gt_cnt = len(find_peaks(gS, distance=int(fs*0.6*period), prominence=max(0.1,0.2*np.std(gS)))[0])

    print("\n================ Result (whole-clip) ================")
    print(f"Video: {video_path}")
    print(f"H5   : {h5_path}")
    print(f"Duration: {duration:.1f}s   Samples: {len(tS)}   fs(resampled): {fs:.2f}Hz")
    print(f"Global lag applied: {lag:+.3f}s")
    print(f"RR(bpm)={rr_bpm if np.isfinite(rr_bpm) else np.nan:.2f}  Period={period:.2f}s  Peaks_GT={pk_gt_cnt}")
    print(f" dW: n={S_w['n']:<3d}  Δt_med={S_w['dt_med']:+.3f}s  IQR={S_w['dt_iqr']:.3f}s   phase_med={S_w['ph_med']:+.1f}°  IQR={S_w['ph_iqr']:.1f}°")
    print(f" dY: n={S_y['n']:<3d}  Δt_med={S_y['dt_med']:+.3f}s  IQR={S_y['dt_iqr']:.3f}s   phase_med={S_y['ph_med']:+.1f}°  IQR={S_y['ph_iqr']:.1f}°")
    print(f" dD: n={S_d['n']:<3d}  Δt_med={S_d['dt_med']:+.3f}s  IQR={S_d['dt_iqr']:.3f}s   phase_med={S_d['ph_med']:+.1f}°  IQR={S_d['ph_iqr']:.1f}°")
    print("=====================================================\n")

    return dict(
        rr_bpm=float(rr_bpm) if np.isfinite(rr_bpm) else np.nan,
        period=float(period),
        lag_sec=float(lag),
        summary=dict(dW=S_w, dY=S_y, dD=S_d)
    )

# ---------------- Batch helpers ----------------
VIDEO_PREFER = ("data.mkv", "data.avi")
H5_CANDIDATES = ("data.hdf5", "data.h5")

def pick_existing(base, names):
    for n in names:
        p = os.path.join(base, n)
        if os.path.exists(p): return p
    return None

def nanmean(vs):
    arr = np.asarray(vs, float)
    return float(np.nanmean(arr)) if arr.size else np.nan
def nanmedian(vs):
    arr = np.asarray(vs, float)
    return float(np.nanmedian(arr)) if arr.size else np.nan

# ---------------- Auto main ----------------
def auto_run():
    assert os.path.isdir(DATA_ROOT), f"COHFACE 루트가 존재하지 않습니다: {DATA_ROOT}"
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_DIR_ROOT, f"auto_{ts_tag}")
    os.makedirs(out_dir, exist_ok=True)
    pairs_dir = os.path.join(out_dir, "pairs")
    if SAVE_PAIRS_CSV: os.makedirs(pairs_dir, exist_ok=True)

    rows = []; run_count = 0
    for s in AUTO_SUBJECTS:
        s_dir = os.path.join(DATA_ROOT, str(s))
        if not os.path.isdir(s_dir):
            print(f"[skip] subject {s} 없음")
            continue
        for k in AUTO_SESSIONS:
            k_dir = os.path.join(s_dir, str(k))
            if not os.path.isdir(k_dir):
                print(f"[skip] s={s} session {k} 없음")
                continue
            vid = pick_existing(k_dir, VIDEO_PREFER)
            h5  = pick_existing(k_dir, H5_CANDIDATES)
            if not vid or not h5:
                print(f"[skip] s={s} k={k} (video/h5 없음)")
                continue

            out_csv = os.path.join(pairs_dir, f"s{s}_k{k}.csv") if SAVE_PAIRS_CSV else None
            res = run_experiment(vid, h5, out_csv=out_csv, verbose=False)
            sm = res["summary"]
            rows.append(dict(
                subject=s, session=k, video=vid, h5=h5,
                rr_bpm=res["rr_bpm"], lag_sec=res["lag_sec"],
                n_w=sm["dW"]["n"], dt_w=sm["dW"]["dt_med"], iqr_dt_w=sm["dW"]["dt_iqr"],
                ph_w=sm["dW"]["ph_med"], iqr_ph_w=sm["dW"]["ph_iqr"],
                n_y=sm["dY"]["n"], dt_y=sm["dY"]["dt_med"], iqr_dt_y=sm["dY"]["dt_iqr"],
                ph_y=sm["dY"]["ph_med"], iqr_ph_y=sm["dY"]["ph_iqr"],
                n_d=sm["dD"]["n"], dt_d=sm["dD"]["dt_med"], iqr_dt_d=sm["dD"]["dt_iqr"],
                ph_d=sm["dD"]["ph_med"], iqr_ph_d=sm["dD"]["ph_iqr"],
            ))
            run_count += 1

    # summary.csv
    if rows:
        sum_csv = os.path.join(out_dir, "summary.csv")
        keys = list(rows[0].keys())
        with open(sum_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[saved] summary → {sum_csv}")
    else:
        print("[done] 실행 가능한 샘플이 없어 summary.csv를 만들지 않았습니다.")
        return

    # overall stats
    def collect(col): return [r[col] for r in rows if r[col] is not None]
    stats = dict(
        count=run_count,
        rr_bpm_mean = nanmean(collect("rr_bpm")),
        lag_sec_mean= nanmean(collect("lag_sec")),
        dt_w_mean   = nanmean(collect("dt_w")), ph_w_mean = nanmean(collect("ph_w")),
        dt_y_mean   = nanmean(collect("dt_y")), ph_y_mean = nanmean(collect("ph_y")),
        dt_d_mean   = nanmean(collect("dt_d")), ph_d_mean = nanmean(collect("ph_d")),
        dt_w_med    = nanmedian(collect("dt_w")), ph_w_med = nanmedian(collect("ph_w")),
        dt_y_med    = nanmedian(collect("dt_y")), ph_y_med = nanmedian(collect("ph_y")),
        dt_d_med    = nanmedian(collect("dt_d")), ph_d_med = nanmedian(collect("ph_d")),
    )
    stats_json = os.path.join(out_dir, "overall_stats.json")
    with open(stats_json, "w") as f: json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[saved] overall stats → {stats_json}")

    print("\n===== OVERALL =====")
    print(f"runs={stats['count']}  rr_bpm_mean={stats['rr_bpm_mean']:.2f}  lag_mean={stats['lag_sec_mean']:+.3f}s")
    print(f"dW: Δt_mean={stats['dt_w_mean']:+.3f}s (med {stats['dt_w_med']:+.3f}s)  "
          f"ϕ_mean={stats['ph_w_mean']:+.1f}° (med {stats['ph_w_med']:+.1f}°)")
    print(f"dY: Δt_mean={stats['dt_y_mean']:+.3f}s (med {stats['dt_y_med']:+.3f}s)  "
          f"ϕ_mean={stats['ph_y_mean']:+.1f}° (med {stats['ph_y_med']:+.1f}°)")
    print(f"dD: Δt_mean={stats['dt_d_mean']:+.3f}s (med {stats['dt_d_med']:+.3f}s)  "
          f"ϕ_mean={stats['ph_d_mean']:+.1f}° (med {stats['ph_d_med']:+.1f}°)")
    print("=========================================\n")

# ---------------- Entry ----------------
if __name__ == "__main__":
    auto_run()