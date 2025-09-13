# -*- coding: utf-8 -*-
import os, glob, json
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any

from .config import FS_MODEL, RESP_BAND, RR_WIN_LIST, STRIDE_FRAC, FIXED_STRIDE
from .utils import zscore, rr_bandpass_z, env_rr, rr_subband_env, butter_lowpass

class CohfaceSeqDataset(Dataset):
    """
    RR-only 16채널 입력 생성
    cache npz: keys = t, dW, dY, dD, dD_perp(optional), resp(or g_resp)
    """
    def __init__(self, cache_dir, subjects=None, sessions=None):
        self.cache_dir = cache_dir
        self.items = []
        for p in sorted(glob.glob(os.path.join(cache_dir, "s*_k*.npz"))):
            self.items.append(p)
        self.X, self.Y, self.L = self._build()

    # ---------------- 16채널 구성 ----------------
    def _build_features(self, rec: Dict[str, np.ndarray]):
        fs = FS_MODEL
        lo, hi = RESP_BAND
        dW = rec["dW"].astype(np.float32)
        dY = rec["dY"].astype(np.float32)
        dD = rec.get("dD_perp", rec["dD"]).astype(np.float32)

        W0 = np.median(dW) + 1e-6
        Wslow = butter_lowpass(dW, fs, fc=0.03)
        w_rel_raw  = dW / W0 - 1.0
        y_norm_raw = dY / (Wslow + 1e-6)
        d_norm_raw = dD / (Wslow + 1e-6)
        dw_rel_raw = np.gradient(dW) * fs / W0  # d/dt dW / W0

        # RR 원파형 (4)
        w_rr  = rr_bandpass_z(w_rel_raw, fs, lo, hi)
        y_rr  = rr_bandpass_z(y_norm_raw, fs, lo, hi)
        d_rr  = rr_bandpass_z(d_norm_raw, fs, lo, hi)
        dw_rr = rr_bandpass_z(dw_rel_raw, fs, lo, hi)

        # 위상 불변 엔벨로프 (4)
        env_w  = env_rr(w_rel_raw, fs, lo, hi)
        env_y  = env_rr(y_norm_raw, fs, lo, hi)
        env_d  = env_rr(d_norm_raw, fs, lo, hi)
        env_dw = env_rr(dw_rel_raw, fs, lo, hi)

        # 결합/서브밴드 (4) — raw bandpass끼리 곱 → RR-BP → z
        from .utils import butter_bandpass
        w_bp_raw = butter_bandpass(w_rel_raw, fs, lo, hi)
        y_bp_raw = butter_bandpass(y_norm_raw, fs, lo, hi)
        d_bp_raw = butter_bandpass(d_norm_raw, fs, lo, hi)
        cross_wy_rr = rr_bandpass_z(w_bp_raw * y_bp_raw, fs, lo, hi)
        cross_wd_rr = rr_bandpass_z(w_bp_raw * d_bp_raw, fs, lo, hi)
        env_low_y   = rr_subband_env(y_norm_raw, fs, lo=0.08, hi=0.25)
        env_high_y  = rr_subband_env(y_norm_raw, fs, lo=0.25, hi=0.60)

        # 느린 컨텍스트 (4) — RR 대역 금지
        w_trend = zscore(butter_lowpass(w_rel_raw, fs, fc=0.2))
        snr_rr_hint = np.zeros_like(w_trend, dtype=np.float32)
        corr_hint_wy= np.zeros_like(w_trend, dtype=np.float32)
        corr_hint_wd= np.zeros_like(w_trend, dtype=np.float32)

        X = np.stack([
            w_rr, y_rr, d_rr, dw_rr,
            env_w, env_y, env_d, env_dw,
            cross_wy_rr, cross_wd_rr, env_low_y, env_high_y,
            w_trend, snr_rr_hint, corr_hint_wy, corr_hint_wd
        ], axis=-1).astype(np.float32)
        return X

    def _build(self):
        Xs, Ys, Ls = [], [], []
        for p in self.items:
            rec = dict(np.load(p, allow_pickle=True))
            t = rec["t"].astype(np.float32)
            # resp 키 호환 (기존 g_resp 지원)
            resp = rec.get("resp", rec.get("g_resp")).astype(np.float32)
            y = rr_bandpass_z(resp, FS_MODEL, *RESP_BAND).astype(np.float32)
            X = self._build_features(rec)
            Xs.append(X); Ys.append(y[:,None]); Ls.append(len(y))
        return Xs, Ys, Ls

    def _gen_windows(self, L):
        fs = FS_MODEL
        for w in RR_WIN_LIST:
            T = int(round(w*fs))
            stride = int(round(FIXED_STRIDE*fs)) if FIXED_STRIDE is not None else max(1, int(round(w*fs*STRIDE_FRAC)))
            for s in range(0, max(1, L-T+1), stride):
                yield T, s, s+T

    def __len__(self):
        # 세션 수 반환 (실제 학습은 iter_windows로 윈도우 생성)
        return len(self.items)

    def iter_windows(self):
        for i in range(len(self.items)):
            X = self.X[i]; Y = self.Y[i][:,0]
            L = len(Y)
            for T, a, b in self._gen_windows(L):
                yield (i, a, b, T)