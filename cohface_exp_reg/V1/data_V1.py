# -*- coding: utf-8 -*-
import glob
import os
from typing import Dict

import numpy as np
from torch.utils.data import Dataset

from .config import FS_MODEL, RESP_BAND, RR_WIN_LIST, STRIDE_FRAC, FIXED_STRIDE, W_TREND_FC
from .utils import butter_bandpass
from .utils import zscore, rr_bandpass_z, env_rr, rr_subband_env, butter_lowpass


class CohfaceSeqDataset(Dataset):
    """
    RR-only 16채널 입력 생성
    cache npz: keys = t, dW, dY, dD, dD_perp(optional), resp(or g_resp)
    """
    def __init__(self, cache_dir, subjects=None, sessions=None):
        self.cache_dir = cache_dir
        self.items = []
        for p in sorted(glob.glob(os.path.join(cache_dir, "s??_k??.npz"))):
            self.items.append(p)
        self.X, self.Y, self.L = self._build()

    # ---------------- 16채널 구성 ----------------
    def _build_features(self, rec: Dict[str, np.ndarray]):
        fs = FS_MODEL
        lo, hi = RESP_BAND
        dW = rec["dW"].astype(np.float32)
        dY = rec["dY"].astype(np.float32)
        dD = rec.get("dD_perp", rec["dD"]).astype(np.float32)

        W0 = max(np.median(dW), 1e-3)
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

        # --- 결합/서브밴드 (4) — raw bandpass끼리 곱 → RR-BP → z ---
        w_bp_raw = butter_bandpass(w_rel_raw, fs, lo, hi)
        y_bp_raw = butter_bandpass(y_norm_raw, fs, lo, hi)
        d_bp_raw = butter_bandpass(d_norm_raw, fs, lo, hi)

        # ▶ 폭주 방지: 곱을 안전 구간으로 클램프
        _MAX_PROD = 50.0
        prod_wy = np.clip(w_bp_raw * y_bp_raw, -_MAX_PROD, _MAX_PROD)
        prod_wd = np.clip(w_bp_raw * d_bp_raw, -_MAX_PROD, _MAX_PROD)

        cross_wy_rr = rr_bandpass_z(prod_wy, fs, lo, hi)
        cross_wd_rr = rr_bandpass_z(prod_wd, fs, lo, hi)

        env_low_y = rr_subband_env(y_norm_raw, fs, lo=0.08, hi=0.25)
        env_high_y = rr_subband_env(y_norm_raw, fs, lo=0.25, hi=0.60)

        # 느린 컨텍스트 (4) — RR 대역 금지
        w_trend = zscore(butter_lowpass(w_rel_raw, fs, fc=W_TREND_FC))  # 기존 0.2 → W_TREND_FC
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
        min_T = int(min(RR_WIN_LIST) * FS_MODEL)  # 예: 24s → 6144
        for p in self.items:
            rec = dict(np.load(p, allow_pickle=True))
            # ① resp 로드
            resp = rec.get("resp", rec.get("g_resp")).astype(np.float32)
            # ② 타깃: 밴드패스만 적용하고 평균만 제거(표준화 X)
            y = butter_bandpass(resp, FS_MODEL, *RESP_BAND).astype(np.float32)
            y = y - float(np.nanmean(y))
            # ③ 입력 특징
            X = self._build_features(rec)  # [Tx, 16]
            # ④ 길이 정합 + 세션 최소 길이 필터
            L = min(len(X), len(y))
            if L < min_T:  # 최소 윈도우보다 짧은 세션은 제외
                continue
            Xs.append(X[:L])
            Ys.append(y[:L, None])
            Ls.append(L)
        return Xs, Ys, Ls

    def _gen_windows(self, L):
        fs = FS_MODEL
        for w in RR_WIN_LIST:
            T = int(round(w * fs))
            if L < T:  # ★ 세션이 더 짧으면 해당 윈도우 스킵
                continue
            stride = int(round(FIXED_STRIDE * fs)) if FIXED_STRIDE is not None else max(1, int(round(
                w * fs * STRIDE_FRAC)))
            for s in range(0, L - T + 1, stride):
                yield T, s, s + T

    def __len__(self):
        # 세션 수 반환 (실제 학습은 iter_windows로 윈도우 생성)
        return len(self.items)

    def iter_windows(self):
        for i in range(len(self.items)):
            X = self.X[i]
            Y = self.Y[i][:, 0]
            L = len(Y)
            for T, a, b in self._gen_windows(L):
                # ▼ 타깃 분산 너무 작으면 학습/평가 가치 없음 → 스킵
                if np.nanstd(Y[a:b]) < 1e-3:  # 임계값은 데이터 분포에 맞게 1e-3~1e-2
                    continue
                yield (i, a, b, T)