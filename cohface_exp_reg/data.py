# -*- coding: utf-8 -*-
import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    FS_MODEL, RR_WIN_LIST, STRIDE_FRAC, FIXED_STRIDE, W_TREND_FC,
    ENABLE_PREALIGN, PREALIGN_MAX_LAG
)
from .utils import (
    zscore, rr_bandpass_z, env_rr, rr_subband_env,
    butter_lowpass, global_sign_and_lag,
    robust_clip, _finite_fill
)


def _safe_norm(num: np.ndarray, denom: np.ndarray, base: float, floor_frac: float = 0.3) -> np.ndarray:
    """
    Normalize 'num/denom' with a robust floor on denom:
      denom = max(denom, floor_frac * base)
    where base is typically median(dW).
    """
    num = _finite_fill(num)
    denom = _finite_fill(denom)
    floor = max(1e-6, float(base) * float(floor_frac))
    denom = np.maximum(denom, floor)
    y = num / (denom + 1e-6)
    # guard against extreme outliers before any filtering
    y = robust_clip(y, clip_std=8.0, abs_max=1e6)
    return y.astype(np.float32)


class CohfaceSeqDataset(Dataset):
    """Build 16-ch RR-only features from cached npz files and provide sliding windows."""

    def __init__(self, cache_dir: str, subset: str = "train", seed: int = 42, floor_frac: float = 0.5):
        super().__init__()
        assert subset in {"train", "val", "test"}
        np.random.seed(seed)
        files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if len(files) == 0:
            raise FileNotFoundError(f"No npz found under {cache_dir}")

        def _split_key(p):
            b = os.path.basename(p)
            nums = [int(x) for x in re.findall(r"\d+", b)]
            return 10 * nums[0] + (nums[1] if len(nums) > 1 else 0)

        # validate npz keys
        pool = []
        for p in files:
            try:
                with np.load(p) as z:
                    if not all(k in z for k in ("t", "dW", "dY", "dD", "resp")):
                        continue
                    pool.append(p)
            except Exception:
                continue
        files = pool
        keys = [_split_key(p) for p in files]
        uniq = sorted(set(keys))
        uniq_train = [k for i, k in enumerate(uniq) if i % 5 not in (3, 4)]  # 60%
        uniq_val = [k for i, k in enumerate(uniq) if i % 5 == 3]  # 20%
        uniq_test = [k for i, k in enumerate(uniq) if i % 5 == 4]  # 20%

        def _sel(ks):
            return [p for p, k in zip(files, keys) if k in ks]

        if subset == "train":
            files = _sel(uniq_train)
        elif subset == "val":
            files = _sel(uniq_val)
        else:
            files = _sel(uniq_test)
        self.files = files

        # Preload and optionally pre-align
        self.sessions = []  # list of dicts with 'X' (T,16), 'Y' (T,)
        for p in self.files:
            with np.load(p) as z:
                t = z["t"].astype(np.float32)
                dW = z["dW"].astype(np.float32)
                dY = z["dY"].astype(np.float32)
                dD = (z["dD_perp"].astype(np.float32)
                      if "dD_perp" in z else z["dD"].astype(np.float32))
                resp = z["resp"].astype(np.float32)

            # robust bases
            W0 = float(np.median(_finite_fill(dW))) + 1e-6
            Wslow = butter_lowpass(dW, FS_MODEL, fc=W_TREND_FC) + 1e-6
            # 모니터링: 바닥 적용 비율(너무 높으면 품질 이슈 의심)
            _floor = max(1e-6, W0 * floor_frac)
            _hit_ratio = float(np.mean(Wslow < _floor))
            if _hit_ratio > 0.10:
                print(f"[warn] W_slow floor hit ratio={_hit_ratio:.1%} (>10%) — {os.path.basename(p)}")

            # normalized sources with denom floor
            w_rel = dW / W0 - 1.0
            y_n = _safe_norm(dY, Wslow, base=W0, floor_frac=floor_frac)
            d_n = _safe_norm(dD, Wslow, base=W0, floor_frac=floor_frac)
            # derivative of dW (centered), then robustify
            dw = np.gradient(dW).astype(np.float32) / (W0 + 1e-6)
            dw = robust_clip(dw)

            # Optional pre-alignment of RESP against y_n
            if ENABLE_PREALIGN:
                sgn, lag = global_sign_and_lag(y_n, resp, fs=FS_MODEL, max_lag_s=PREALIGN_MAX_LAG)
                if lag < 0:
                    resp = np.concatenate([resp[-lag:], np.zeros((-lag,), np.float32)])
                elif lag > 0:
                    resp = np.concatenate([np.zeros((lag,), np.float32), resp[:-lag]])
                resp = float(sgn) * resp

            # 16ch stack (each branch internally robustified + z-score)
            w_rr = rr_bandpass_z(w_rel, FS_MODEL)
            y_rr = rr_bandpass_z(y_n, FS_MODEL)
            d_rr = rr_bandpass_z(d_n, FS_MODEL)
            dw_rr = rr_bandpass_z(dw, FS_MODEL)

            env_w = env_rr(w_rel, FS_MODEL)
            env_y = env_rr(y_n, FS_MODEL)
            env_d = env_rr(d_n, FS_MODEL)
            env_dw = env_rr(dw, FS_MODEL)

            cross_wy_rr = rr_bandpass_z(w_rr * y_rr, FS_MODEL)
            cross_wd_rr = rr_bandpass_z(w_rr * d_rr, FS_MODEL)

            env_low_y = rr_subband_env(y_n, FS_MODEL, lo=0.08, hi=0.25)
            env_high_y = rr_subband_env(y_n, FS_MODEL, lo=0.25, hi=0.60)

            # slow trend context (kept outside RR band)
            from .utils import butter_lowpass as _lp
            w_trend = zscore(_lp(w_rel, FS_MODEL, fc=0.2))

            # session-level hints (scalar)
            def _snr(sig):
                x = rr_bandpass_z(sig, FS_MODEL)
                p = np.abs(np.fft.rfft(x)) ** 2
                m = p.max() + 1e-6
                return float(np.clip((m - np.median(p)) / m, 0, 1))

            snr_hint = _snr(w_rel)
            corr_hint_wy = float(abs(np.corrcoef(w_rr, y_rr)[0, 1])) if len(w_rr) > 8 else 0.0
            corr_hint_wd = float(abs(np.corrcoef(w_rr, d_rr)[0, 1])) if len(w_rr) > 8 else 0.0

            # final stack (ensure finite)
            X16 = np.stack([
                w_rr, y_rr, d_rr, dw_rr,
                env_w, env_y, env_d, env_dw,
                cross_wy_rr, cross_wd_rr, env_low_y, env_high_y,
                w_trend,
                np.full_like(w_rr, snr_hint, dtype=np.float32),
                np.full_like(w_rr, corr_hint_wy, dtype=np.float32),
                np.full_like(w_rr, corr_hint_wd, dtype=np.float32)
            ], axis=-1).astype(np.float32)
            X16 = _finite_fill(X16)

            # target RR waveform (z) for loss/eval
            Y = rr_bandpass_z(resp, FS_MODEL)

            self.sessions.append({"X": X16, "Y": Y})

        # build window indices
        self.idxs = []  # list of (sess_id, start, length)
        strides = []
        for win_s in RR_WIN_LIST:
            T = int(round(win_s * FS_MODEL))
            if FIXED_STRIDE is None:
                stride = int(round(T * STRIDE_FRAC))
            else:
                stride = int(round(FIXED_STRIDE * FS_MODEL))
            strides.append((T, stride))
        for sid, s in enumerate(self.sessions):
            L = len(s["Y"])
            for T, stride in strides:
                if L < T:
                    continue
                for st in range(0, L - T + 1, stride):
                    self.idxs.append((sid, st, T))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i: int):
        sid, st, T = self.idxs[i]
        X = self.sessions[sid]["X"][st:st + T]  # [T,16]
        Y = self.sessions[sid]["Y"][st:st + T]  # [T]
        X = torch.from_numpy(X)  # [T,16]
        Y = torch.from_numpy(Y).unsqueeze(-1)  # [T,1]
        return X, Y
