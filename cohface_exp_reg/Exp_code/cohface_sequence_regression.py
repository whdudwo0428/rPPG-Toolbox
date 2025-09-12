# -*- coding: utf-8 -*-
"""
COHFACE 시계열 회귀 단일 스크립트 (dW/dY/dD → RR/HR 파형 회귀)
- Mediapipe로 dW/dY/dD 추출(캐시) → RR/HR 밴드 전처리
- subject-wise split(train/val/test)로 누수 방지
- LSTM/GRU 시퀀스 모델로 멀티태스크 회귀(Resp 파형 + ECG/PPG 파형)
- RR/HR(bpm) 추정 및 상관/RMSE 평가, 모델/결과 저장

필요 패키지: mediapipe, numpy, scipy, h5py, opencv-python, torch, tqdm
"""

import os, sys, time, json, math, random, glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2, h5py
from scipy.signal import butter, sosfiltfilt, welch
from scipy.signal import find_peaks
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- 사용자 설정 --------------------
DATA_ROOT = os.environ.get("COHFACE_ROOT", "/mnt/hdd18t/rppg_dataset/raw/cohface")
OUT_DIR_ROOT = "./seqreg_out"
CACHE_DIR = "cohface_exp_reg/cache_cohface_feats"
os.makedirs(OUT_DIR_ROOT, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)

# 모델/학습
MODEL_TYPE = "LSTM"   # "LSTM" or "GRU"
HIDDEN = 128
LAYERS = 2
BIDIR = True
LR = 3e-4
EPOCHS = 25
BATCH = 32
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# 신호 처리
FS_EXTRACT = 256.0   # 추출/정렬 resample fs
FS_MODEL   = 64.0    # 학습용 downsample fs (경량)
RESP_BAND = (0.08, 0.60)  # 5–36 bpm
HR_BAND   = (0.8, 3.0)    # 48–180 bpm
BP_ORDER  = 4
GLOBAL_LAG_CLIP = 0.5     # ±s

# 샘플 윈도우
WIN_SEC = 8.0
STRIDE_SEC = 2.0

# Mediapipe 설정 (헤드리스)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
import mediapipe as mp
mp_pose = mp.solutions.pose
EMA_BETA_LMK = 0.7
MAX_LMK_JUMP = 40.0
POSE_EVERY_N = 2
PROC_W = 640

# -----------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# -------------------- DSP 유틸 --------------------
def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
    t = np.asarray(t, np.float32); x = np.asarray(x, np.float32)
    if t.size < 3: return None, None, None
    t_u = np.arange(t[0], t[-1] + 1e-6, 1.0/fs, dtype=np.float32)
    x_u = np.interp(t_u, t, x).astype(np.float32)
    return t_u, x_u, fs

def butter_sos(lo, hi, fs, order=BP_ORDER):
    nyq = fs * 0.5
    lo2 = max(1e-3, lo/nyq); hi2 = min(0.999, hi/nyq)
    if hi2 <= lo2 + 1e-3: return None
    return butter(order, [lo2, hi2], btype='bandpass', output='sos')

def bandpass(x, fs, band):
    if x is None or fs is None: return x
    sos = butter_sos(band[0], band[1], fs)
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
    s = 1.0 if (not np.isfinite(s) or s < 1e-6) else s
    return (x - m) / s

def gcc_phat_linear(x, y, fs):
    x = np.asarray(x, np.float32).ravel(); y = np.asarray(y, np.float32).ravel()
    n = int(len(x) + len(y) - 1); nfft = 1
    while nfft < n: nfft <<= 1
    X = np.fft.rfft(x, nfft); Y = np.fft.rfft(y, nfft)
    R = X * np.conj(Y); R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, nfft)
    lags = np.arange(-len(y) + 1, len(x))
    cc = np.concatenate([cc[-(len(y)-1):], cc[:len(x)]])
    i = int(np.argmax(cc))
    return float(lags[i] / fs)

def estimate_rr_bpm(sig, fs):
    sig = zscore(bandpass(sig, fs, RESP_BAND))
    if sig is None or len(sig) < int(fs*6): return np.nan
    nper = max(128, int(fs*8)); nover = nper//2
    f, P = welch(sig, fs=fs, nperseg=nper, noverlap=nover)
    mb = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    if np.count_nonzero(mb) < 3: return np.nan
    fb = f[mb]; f0 = float(fb[np.argmax(P[mb])])
    return f0 * 60.0

def estimate_hr_bpm(sig, fs):
    sig = zscore(bandpass(sig, fs, HR_BAND))
    if sig is None or len(sig) < int(fs*4): return np.nan
    nper = max(128, int(fs*4)); nover = nper//2
    f, P = welch(sig, fs=fs, nperseg=nper, noverlap=nover)
    mb = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    if np.count_nonzero(mb) < 3: return np.nan
    fb = f[mb]; f0 = float(fb[np.argmax(P[mb])])
    return f0 * 60.0

# -------------------- COHFACE IO --------------------
def load_h5(h5_path) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        get = lambda k: np.asarray(f[k][:], np.float32) if k in f else None
        time = get("time")
        resp = get("respiration")
        # ECG/PPG 후보 키 자동 탐색
        ecg = get("ecg") if "ecg" in keys else (get("pulse") if "pulse" in keys else (get("ppg") if "ppg" in keys else None))
    return dict(time=time, respiration=resp, cardio=ecg)

def pick_files(subj_dir):
    vids = [p for p in [os.path.join(subj_dir, s, "data.mkv") for s in ["0","1","2","3"]] if os.path.exists(p)]
    h5s  = [p for p in [os.path.join(subj_dir, s, "data.hdf5") for s in ["0","1","2","3"]] if os.path.exists(p)]
    pairs = []
    for s in ["0","1","2","3"]:
        v = os.path.join(subj_dir, s, "data.mkv")
        h = os.path.join(subj_dir, s, "data.hdf5")
        if os.path.exists(v) and os.path.exists(h):
            pairs.append((int(os.path.basename(subj_dir)), int(s), v, h))
    return pairs

# -------------------- Mediapipe → dW/dY/dD --------------------
def extract_displacements(video_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            t  = ms/1000.0
            if t < prev_ms/1000.0: t = prev_ms/1000.0 + 1.0/max(1.0,fps)
            prev_ms = ms; frame_idx += 1
            if (frame_idx % POSE_EVERY_N) != 0: continue

            rgb_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   (PROC_W, int(H*PROC_W/W)), interpolation=cv2.INTER_AREA)
            res = pose.process(rgb_small)
            if not res.pose_landmarks: continue
            lm = res.pose_landmarks.landmark
            h_s, w_s = rgb_small.shape[:2]; sx, sy = W/float(w_s), H/float(h_s)
            def to_px(pt): return np.array([pt.x*w_s*sx, pt.y*h_s*sy], np.float32)
            L = to_px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
            R = to_px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            N = to_px(lm[mp_pose.PoseLandmark.NOSE])
            visL = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility
            visR = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility
            visN = lm[mp_pose.PoseLandmark.NOSE].visibility
            if min(visL, visR, visN) <= 0.5: continue

            if prev_L is not None:
                if (np.linalg.norm(L-prev_L) > MAX_LMK_JUMP or
                    np.linalg.norm(R-prev_R) > MAX_LMK_JUMP or
                    np.linalg.norm(N-prev_N) > MAX_LMK_JUMP):
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

    cap.release()
    return np.asarray(ts, np.float32), np.asarray(dW, np.float32), np.asarray(dY, np.float32), np.asarray(dD, np.float32)

def estimate_global_lag_from_composite(ts, dW, dY, dD, gt_t, gt_resp, fs=FS_EXTRACT) -> float:
    tu, wU, _ = resample_uniform(ts, dW, fs)
    _,  yU, _ = resample_uniform(ts, dY, fs)
    _,  dU, _ = resample_uniform(ts, dD, fs)
    cU = (zscore(bandpass(wU, fs, RESP_BAND)) +
          zscore(bandpass(yU, fs, RESP_BAND)) +
          zscore(bandpass(dU, fs, RESP_BAND))) / 3.0
    tg, gU, _ = resample_uniform(gt_t, gt_resp, fs)
    gU = zscore(bandpass(gU, fs, RESP_BAND))
    if cU is None or gU is None: return 0.0
    try:
        lag = float(np.clip(gcc_phat_linear(cU, gU, fs), -GLOBAL_LAG_CLIP, GLOBAL_LAG_CLIP))
    except Exception:
        lag = 0.0
    return lag

def align_common_time(tu, Xlist, tg, g, fs) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    t0 = max(float(tu[0]), float(tg[0]))
    t1 = min(float(tu[-1]), float(tg[-1]))
    if t1 <= t0 + (1.0/fs)*4:  # 4샘플 미만이면 실패
        return tu, Xlist, g
    tC = np.arange(t0, t1 + 1e-6, 1.0/fs, dtype=np.float32)
    Xc = [np.interp(tC, tu, x).astype(np.float32) for x in Xlist]
    gc = np.interp(tC, tg, g).astype(np.float32)
    return tC, Xc, gc

# -------------------- 캐시 I/O --------------------
def cache_path(subject:int, sess:int) -> str:
    return os.path.join(CACHE_DIR, f"s{subject}_k{sess}.npz")

def ensure_cached(subject:int, sess:int, vid_path:str, h5_path:str) -> Optional[Dict]:
    path = cache_path(subject, sess)
    if os.path.exists(path):
        try:
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        except Exception:
            pass  # 재생성
    # 새로 추출
    ts, dW, dY, dD = extract_displacements(vid_path)
    gt = load_h5(h5_path)
    gt_t, resp, cardio = gt["time"], gt["respiration"], gt["cardio"]
    if ts is None or resp is None or gt_t is None: return None
    lag = estimate_global_lag_from_composite(ts, dW, dY, dD, gt_t, resp, fs=FS_EXTRACT)
    # 전역 래그 적용 후 resample(학습 fs)
    tu, wU, _ = resample_uniform(ts, dW, FS_EXTRACT)
    _,  yU, _ = resample_uniform(ts, dY, FS_EXTRACT)
    _,  dU, _ = resample_uniform(ts, dD, FS_EXTRACT)
    tg, gU, _ = resample_uniform(gt_t + lag, resp, FS_EXTRACT)
    # 공통 시간축
    tC, [wC, yC, dC], gC = align_common_time(tu, [wU, yU, dU], tg, gU, FS_EXTRACT)
    # 학습 fs로 다운샘플
    tM, wM, _ = resample_uniform(tC, wC, FS_MODEL)
    _,  yM, _ = resample_uniform(tC, yC, FS_MODEL)
    _,  dM, _ = resample_uniform(tC, dC, FS_MODEL)
    _,  gM, _ = resample_uniform(tC, gC, FS_MODEL)
    # cardio(ECG/PPG) 있으면 정렬/다운샘플
    cM = None
    if cardio is not None and gt_t is not None:
        tg2, cU, _ = resample_uniform(gt_t + lag, cardio, FS_EXTRACT)
        _, cM, _ = resample_uniform(tC, np.interp(tC, tg2, cU), FS_MODEL)
    # 저장
    out = dict(t=tM, dW=wM, dY=yM, dD=dM, g_resp=gM, g_cardio=cM, lag=lag,
               subject=subject, session=sess, vid=vid_path, h5=h5_path)
    np.savez_compressed(path, **out)
    return out

# -------------------- 데이터셋 생성 --------------------
def build_index() -> List[Tuple[int,int,str,str]]:
    subjects = sorted([int(d) for d in os.listdir(DATA_ROOT) if d.isdigit()])
    pairs = []
    for s in subjects:
        sdir = os.path.join(DATA_ROOT, str(s))
        for k in [0,1,2,3]:
            v = os.path.join(sdir, str(k), "data.mkv")
            h = os.path.join(sdir, str(k), "data.hdf5")
            if os.path.exists(v) and os.path.exists(h):
                pairs.append((s, k, v, h))
    return pairs

def subject_split(subjects: List[int], ratios=(0.7,0.15,0.15)):
    subs = sorted(set(subjects))
    rng = np.random.RandomState(SEED)
    rng.shuffle(subs)
    n = len(subs); n_train = int(n*ratios[0]); n_val = int(n*ratios[1])
    train = subs[:n_train]; val = subs[n_train:n_train+n_val]; test = subs[n_train+n_val:]
    return set(train), set(val), set(test)

def make_windows(sig_len:int, fs:float, win_sec:float, stride_sec:float):
    win = int(round(win_sec*fs)); st = int(round(stride_sec*fs))
    for s in range(0, max(1, sig_len - win + 1), st):
        e = s + win
        if e <= sig_len: yield s, e

class CohfaceSeqDataset(Dataset):
    def __init__(self, entries: List[Dict], split: str):
        self.rows = []
        for E in entries:
            t = E["t"]; w = E["dW"]; y = E["dY"]; d = E["dD"]; g = E["g_resp"]; c = E["g_cardio"]
            if t is None or g is None: continue
            # 특징: RR/HR 밴드 + dC (각 밴드)
            dC = (w + y + d) / 3.0
            w_rr = zscore(bandpass(w, FS_MODEL, RESP_BAND)); y_rr = zscore(bandpass(y, FS_MODEL, RESP_BAND)); d_rr = zscore(bandpass(d, FS_MODEL, RESP_BAND)); c_rr = zscore(bandpass(dC, FS_MODEL, RESP_BAND))
            w_hr = zscore(bandpass(w, FS_MODEL, HR_BAND));   y_hr = zscore(bandpass(y, FS_MODEL, HR_BAND));   d_hr = zscore(bandpass(d, FS_MODEL, HR_BAND));   c_hr = zscore(bandpass(dC, FS_MODEL, HR_BAND))
            X = np.stack([w_rr,y_rr,d_rr,c_rr, w_hr,y_hr,d_hr,c_hr], axis=1).astype(np.float32)  # [T,8]
            y_rr_tgt = zscore(bandpass(g, FS_MODEL, RESP_BAND)).astype(np.float32)
            if c is not None:
                y_hr_tgt = zscore(bandpass(c, FS_MODEL, HR_BAND)).astype(np.float32)
            else:
                y_hr_tgt = None

            for s,e in make_windows(len(t), FS_MODEL, WIN_SEC, STRIDE_SEC):
                xin = X[s:e]
                yrr = y_rr_tgt[s:e]
                if y_hr_tgt is not None:
                    yhr = y_hr_tgt[s:e]
                else:
                    yhr = None
                if xin.shape[0] == int(WIN_SEC*FS_MODEL):
                    self.rows.append(dict(X=xin, y_rr=yrr, y_hr=yhr,
                                          subject=E["subject"], session=E["session"]))
        # 정보
        self.split = split
        self.fs = FS_MODEL
        self.win_len = int(WIN_SEC*FS_MODEL)

    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        X = torch.from_numpy(r["X"])              # [T,8]
        y_rr = torch.from_numpy(r["y_rr"]).unsqueeze(-1)  # [T,1]
        if r["y_hr"] is not None:
            y_hr = torch.from_numpy(r["y_hr"]).unsqueeze(-1)  # [T,1]
            mask_hr = torch.ones_like(y_hr)
        else:
            y_hr = torch.zeros_like(y_rr)
            mask_hr = torch.zeros_like(y_rr)  # 없으면 loss 제외
        return X, torch.cat([y_rr, y_hr], dim=-1), mask_hr, r["subject"], r["session"]

# -------------------- 모델 --------------------
class SeqRegressor(nn.Module):
    def __init__(self, input_dim=8, hidden=HIDDEN, layers=LAYERS, bidir=BIDIR, cell="LSTM"):
        super().__init__()
        if cell.upper() == "GRU":
            self.rnn = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        else:
            self.rnn = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        h_out = hidden * (2 if bidir else 1)
        self.head = nn.Linear(h_out, 2)  # [rr, hr] 두 채널
    def forward(self, x):
        y, _ = self.rnn(x)        # [B,T,H*dir]
        y = self.head(y)          # [B,T,2]
        return y

def corrcoef_torch(x, y, eps=1e-8):
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (x*y).sum(dim=1)
    den = torch.sqrt((x.square().sum(dim=1) + eps)*(y.square().sum(dim=1) + eps))
    return (num / den).mean()

# -------------------- 학습 루프 --------------------
def train_loop(model, train_loader, val_loader, out_dir):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = {"val_loss": 1e9}
    patience = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for X, Y, mask_hr, *_ in train_loader:
            X = X.to(DEVICE).float()
            Y = Y.to(DEVICE).float()
            mask_hr = mask_hr.to(DEVICE).float()
            pred = model(X)  # [B,T,2]
            # 손실: MSE + (1-corr) for RR channel
            mse_rr = ((pred[:,:,0] - Y[:,:,0])**2).mean()
            mse_hr = ((((pred[:,:,1] - Y[:,:,1])**2) * mask_hr.squeeze(-1)).sum()
                      / (mask_hr.sum() + 1e-6))
            # 상관 보정(모양 맞추기)
            corr_rr = corrcoef_torch(pred[:,:,0], Y[:,:,0])
            loss = mse_rr + 0.5*mse_hr + 0.2*(1.0 - corr_rr)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        # validation
        model.eval()
        with torch.no_grad():
            va_loss = 0.0; corr_rr_all = []
            for X, Y, mask_hr, *_ in val_loader:
                X = X.to(DEVICE).float(); Y = Y.to(DEVICE).float(); mask_hr = mask_hr.to(DEVICE).float()
                P = model(X)
                mse_rr = ((P[:,:,0] - Y[:,:,0])**2).mean()
                mse_hr = ((((P[:,:,1] - Y[:,:,1])**2) * mask_hr.squeeze(-1)).sum()
                          / (mask_hr.sum() + 1e-6))
                corr_rr = corrcoef_torch(P[:,:,0], Y[:,:,0]).item()
                va_loss += (mse_rr + 0.5*mse_hr).item()
                corr_rr_all.append(corr_rr)
            va_loss /= max(1, len(val_loader))
            corr_rr_mean = float(np.mean(corr_rr_all)) if corr_rr_all else 0.0

        print(f"[{epoch:02d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_corrRR={corr_rr_mean:.3f}")

        # early stopping
        if va_loss + (1.0 - corr_rr_mean)*0.1 < best["val_loss"]:
            best.update(val_loss=va_loss, corr_rr=corr_rr_mean, epoch=epoch)
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break
    return best

# -------------------- 평가 --------------------
def evaluate(model, loader) -> Dict[str, float]:
    model = model.to(DEVICE).eval()
    all_corr_rr, all_rmse_rr = [], []
    all_corr_hr, all_rmse_hr = [], []
    rr_pred_bpm, rr_gt_bpm = [], []
    hr_pred_bpm, hr_gt_bpm = [], []

    with torch.no_grad():
        for X, Y, mask_hr, *_ in loader:
            X = X.to(DEVICE).float(); Y = Y.to(DEVICE).float(); mask_hr = mask_hr.to(DEVICE).float()
            P = model(X)  # [B,T,2]
            # RR
            corr_rr = corrcoef_torch(P[:,:,0], Y[:,:,0]).item()
            rmse_rr = torch.sqrt(((P[:,:,0]-Y[:,:,0])**2).mean()).item()
            all_corr_rr.append(corr_rr); all_rmse_rr.append(rmse_rr)
            # HR(있을 때만)
            if mask_hr.sum() > 0:
                B = X.shape[0]
                # bpm 추정(윈도우별)
                for b in range(B):
                    y_gt_rr = Y[b,:,0].cpu().numpy()
                    y_pr_rr = P[b,:,0].cpu().numpy()
                    rr_gt_bpm = estimate_rr_bpm(y_gt_rr, FS_MODEL); rr_pr_bpm = estimate_rr_bpm(y_pr_rr, FS_MODEL)
                    if np.isfinite(rr_gt_bpm) and np.isfinite(rr_pr_bpm):
                        rr_gt_bpm = float(rr_gt_bpm); rr_pr_bpm = float(rr_pr_bpm)
                        rr_pred_bpm.append(rr_pr_bpm); rr_gt_bpm.append(rr_gt_bpm)

                # HR 상관/오차
                # mask 적용
                m = mask_hr.squeeze(-1) > 0.5
                if m.any():
                    gt_hr = Y[:,:,1][m].cpu()
                    pr_hr = P[:,:,1][m].cpu()
                    corr = np.corrcoef(pr_hr.numpy(), gt_hr.numpy())[0,1] if gt_hr.numel() > 10 else np.nan
                    rmse = torch.sqrt(((P[:,:,1]-Y[:,:,1])**2 * mask_hr.squeeze(-1)).sum()
                                      /(mask_hr.sum()+1e-6)).item()
                    if np.isfinite(corr): all_corr_hr.append(float(corr))
                    all_rmse_hr.append(rmse)

    def safemean(v): return float(np.mean(v)) if len(v) else float("nan")
    metrics = dict(
        corr_rr=safemean(all_corr_rr), rmse_rr=safemean(all_rmse_rr),
        corr_hr=safemean(all_corr_hr), rmse_hr=safemean(all_rmse_hr),
        rr_bpm_mae = safemean([abs(a-b) for a,b in zip(rr_pred_bpm, rr_gt_bpm)]) if rr_pred_bpm else float("nan")
    )
    return metrics

# -------------------- 메인 파이프라인 --------------------
def main():
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(os.path.join(OUT_DIR_ROOT, f"{MODEL_TYPE.lower()}_{ts_tag}"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    # 1) 인덱스 & 캐시 생성
    pairs = build_index()
    if not pairs:
        print(f"[ERR] COHFACE 경로 확인: {DATA_ROOT}")
        sys.exit(1)

    subj_ids = [s for s,_,_,_ in pairs]
    trS, vaS, teS = subject_split(subj_ids, ratios=(0.7,0.15,0.15))
    print(f"[split] train={len(trS)} val={len(vaS)} test={len(teS)} subjects")

    cache_entries = []
    for s, k, v, h in tqdm(pairs, desc="Caching/Extracting"):
        E = ensure_cached(s, k, v, h)
        if E is not None:
            cache_entries.append(E)

    # split 적용
    train_entries = [E for E in cache_entries if E["subject"] in trS]
    val_entries   = [E for E in cache_entries if E["subject"] in vaS]
    test_entries  = [E for E in cache_entries if E["subject"] in teS]

    # 2) Dataset/Dataloader
    ds_tr = CohfaceSeqDataset(train_entries, split="train")
    ds_va = CohfaceSeqDataset(val_entries,   split="val")
    ds_te = CohfaceSeqDataset(test_entries,  split="test")
    print(f"[windows] train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)} (win={WIN_SEC}s, stride={STRIDE_SEC}s)")

    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=2)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=2)

    # 3) 모델
    model = SeqRegressor(input_dim=8, hidden=HIDDEN, layers=LAYERS, bidir=BIDIR, cell=MODEL_TYPE)

    # 4) 학습
    best = train_loop(model, dl_tr, dl_va, out_dir)
    print(f"[best] epoch={best.get('epoch')}  val_loss={best.get('val_loss'):.4f}  val_corrRR={best.get('corr_rr'):.3f}")

    # best 로드 후 평가
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=DEVICE))
    metrics = {
        "val": evaluate(model, dl_va),
        "test": evaluate(model, dl_te),
        "config": dict(MODEL_TYPE=MODEL_TYPE, HIDDEN=HIDDEN, LAYERS=LAYERS, BIDIR=BIDIR,
                       FS_EXTRACT=FS_EXTRACT, FS_MODEL=FS_MODEL,
                       RESP_BAND=RESP_BAND, HR_BAND=HR_BAND,
                       WIN_SEC=WIN_SEC, STRIDE_SEC=STRIDE_SEC,
                       LR=LR, EPOCHS=EPOCHS, BATCH=BATCH, PATIENCE=PATIENCE, SEED=SEED)
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[saved] metrics.json")

if __name__ == "__main__":
    main()
