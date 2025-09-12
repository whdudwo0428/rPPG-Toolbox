import glob
import os
from typing import List, Dict, Optional

import numpy as np
from torch.utils.data import Dataset

from .utils import bandpass, zscore
from .config import (CACHE_DIR, FS_MODEL,
                     RESP_BAND, HR_BAND,
                     HR_WIN_LIST as CFG_HR_WINS,
                     RR_WIN_LIST as CFG_RR_WINS,
                     STRIDE_FRAC as CFG_STRIDE_FRAC,
                     FIXED_STRIDE as CFG_FIXED_STRIDE)


def _make_windows(sig_len:int, fs:float, win_sec:float, stride_sec:float):
    win = int(round(win_sec*fs)); st = int(round(stride_sec*fs))
    for s in range(0, max(1, sig_len - win + 1), st):
        e = s + win
        if e <= sig_len:
            yield s, e, win

def load_all_entries() -> List[Dict]:
    rows=[]
    for path in sorted(glob.glob(os.path.join(CACHE_DIR, "s*_k*.npz"))):
        Z = np.load(path, allow_pickle=True)
        rows.append({k: Z[k] for k in Z.files})
    return rows


class CohfaceSeqDataset(Dataset):
    """
    - 입력: [T,8] = [w_rr,y_rr,d_rr,c_rr, w_hr,y_hr,d_hr,c_hr]
    - 타깃: [T,2] = [gt_rr, gt_hr]; 마스크: [T,2] (rr, hr 각각)
    - 멀티스케일: RR 윈도우(예: 32,64), HR 윈도우(예: 8,16)
      * RR 샘플 → mask_rr=1, mask_hr=0
      * HR 샘플 → mask_rr=0, mask_hr=1 (HR GT 없으면 생성 안함)
    """
    def __init__(
        self,
        entries: List[Dict],
        split: str,
        rr_win_list: Optional[List[float]] = None,
        hr_win_list: Optional[List[float]] = None,
        stride_frac: Optional[float] = None,
        fixed_stride: Optional[float] = None,
    ):
        self.rows=[]
        RR_WINS = list(rr_win_list) if rr_win_list else list(CFG_RR_WINS)
        HR_WINS = list(hr_win_list) if hr_win_list else list(CFG_HR_WINS)
        STRIDE_FRAC = float(stride_frac) if (stride_frac is not None) else float(CFG_STRIDE_FRAC)
        FIXED_STRIDE = float(fixed_stride) if (fixed_stride is not None) else CFG_FIXED_STRIDE

        for E in entries:
            t = E["t"]; w=E["dW"]; y=E["dY"]; d=E["dD"]; g=E["g_resp"]; c=E.get("g_cardio", None)
            if t is None or g is None:
                continue

            dC = (w+y+d)/3.0
            # RR/HR 대역 분리 특징
            w_rr = zscore(bandpass(w, FS_MODEL, RESP_BAND)); y_rr = zscore(bandpass(y, FS_MODEL, RESP_BAND)); d_rr = zscore(bandpass(d, FS_MODEL, RESP_BAND)); c_rr = zscore(bandpass(dC, FS_MODEL, RESP_BAND))
            w_hr = zscore(bandpass(w, FS_MODEL, HR_BAND));   y_hr = zscore(bandpass(y, FS_MODEL, HR_BAND));   d_hr = zscore(bandpass(d, FS_MODEL, HR_BAND));   c_hr = zscore(bandpass(dC, FS_MODEL, HR_BAND))
            X = np.stack([w_rr,y_rr,d_rr,c_rr, w_hr,y_hr,d_hr,c_hr], axis=1).astype(np.float32)  # [T,8]
            y_rr_tgt = zscore(bandpass(g, FS_MODEL, RESP_BAND)).astype(np.float32)
            y_hr_tgt = zscore(bandpass(c, FS_MODEL, HR_BAND)).astype(np.float32) if c is not None else None

            S = len(t); fs = FS_MODEL

            def stride_for(win):
                # 고정 stride가 주어지면 그것을 사용, 아니면 윈도우 비율
                if FIXED_STRIDE is not None and FIXED_STRIDE > 0:
                    return FIXED_STRIDE
                return max(1.0, float(win) * float(STRIDE_FRAC))

            # RR 샘플
            for win_sec in RR_WINS:
                st = stride_for(win_sec)
                for s,e,_ in _make_windows(S, fs, win_sec, st):
                    xin = X[s:e]; yrr = y_rr_tgt[s:e]
                    if xin.shape[0] == int(round(win_sec*fs)):
                        yhr = y_hr_tgt[s:e] if y_hr_tgt is not None else np.zeros_like(yrr)
                        mask_rr = np.ones_like(yrr, dtype=np.float32)
                        mask_hr = np.zeros_like(yrr, dtype=np.float32)
                        self.rows.append(dict(
                            X=xin, y_rr=yrr, y_hr=yhr, m_rr=mask_rr, m_hr=mask_hr,
                            T=int(xin.shape[0]), subject=int(E["subject"]), session=int(E["session"])
                        ))

            # HR 샘플 (HR GT 없으면 skip)
            if y_hr_tgt is not None:
                for win_sec in HR_WINS:
                    st = stride_for(win_sec)
                    for s,e,_ in _make_windows(S, fs, win_sec, st):
                        xin = X[s:e]; yrr = y_rr_tgt[s:e]; yhr = y_hr_tgt[s:e]
                        if xin.shape[0] == int(round(win_sec*fs)) and yhr.shape[0] == int(round(win_sec*fs)):
                            mask_rr = np.zeros_like(yrr, dtype=np.float32)
                            mask_hr = np.ones_like(yhr, dtype=np.float32)
                            self.rows.append(dict(
                                X=xin, y_rr=yrr, y_hr=yhr, m_rr=mask_rr, m_hr=mask_hr,
                                T=int(xin.shape[0]), subject=int(E["subject"]), session=int(E["session"])
                            ))

        self.split = split
        self.fs = FS_MODEL

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        r = self.rows[i]
        X = torch.from_numpy(r["X"])                       # [T,8]
        y_rr = torch.from_numpy(r["y_rr"]).unsqueeze(-1)   # [T,1]
        y_hr = torch.from_numpy(r["y_hr"]).unsqueeze(-1)   # [T,1]
        m_rr = torch.from_numpy(r["m_rr"]).unsqueeze(-1)   # [T,1]
        m_hr = torch.from_numpy(r["m_hr"]).unsqueeze(-1)   # [T,1]
        Y = torch.cat([y_rr, y_hr], dim=-1)                # [T,2]
        M = torch.cat([m_rr, m_hr], dim=-1)                # [T,2]
        return X, Y, M, r["subject"], r["session"]

    @property
    def lengths(self):
        # 버킷 샘플러용 길이(T)
        return [int(r.get("T", r["X"].shape[0])) for r in self.rows]


def pad_collate(batch):
    """
    서로 다른 길이(T)가 섞인 배치를 패딩 + 마스크로 정리합니다.
      - X: [B, T_max, 8]
      - Y: [B, T_max, 2]
      - M: [B, T_max, 2]  (rr/hr 유효 마스크)
      - pad_mask: [B, T_max, 1]  (패딩 마스크: 유효=1, 패딩=0)
    """
    import torch
    Xs, Ys, Ms, subs, sess = zip(*batch)
    lens = [x.shape[0] for x in Xs]
    T = max(lens); B = len(batch)
    Xp = torch.zeros((B, T, Xs[0].shape[1]), dtype=torch.float32)
    Yp = torch.zeros((B, T, Ys[0].shape[1]), dtype=torch.float32)
    Mp = torch.zeros((B, T, Ms[0].shape[1]), dtype=torch.float32)
    Pm = torch.zeros((B, T, 1), dtype=torch.float32)
    for i,(x,y,m) in enumerate(zip(Xs,Ys,Ms)):
        L = x.shape[0]
        Xp[i,:L,:] = x; Yp[i,:L,:] = y; Mp[i,:L,:] = m; Pm[i,:L,0] = 1.0
    return Xp, Yp, Mp, Pm, np.array(subs), np.array(sess)
