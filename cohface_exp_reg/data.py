
import os, glob, numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from config import CACHE_DIR, FS_MODEL, WIN_SEC, STRIDE_SEC, HR_WIN_LIST, RR_WIN_LIST, STRIDE_FRAC, FIXED_STRIDE
from utils import bandpass, zscore
from config import RESP_BAND, HR_BAND

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
    - 멀티스케일: RR_WIN_LIST(예: 32,64), HR_WIN_LIST(예: 8,16)
      * RR윈도우 샘플에서는 mask_rr=1, mask_hr=0
      * HR윈도우 샘플에서는 mask_rr=0, mask_hr=1 (HR GT 미존재시 자동 0)
    - 서로 다른 길이를 패딩 없이 반환하고, collate_fn에서 패딩/마스킹을 처리합니다.
    """
    def __init__(self, entries: List[Dict], split: str):
        self.rows=[]
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

            # 멀티스케일 윈도우 생성
            S = len(t); fs = FS_MODEL
            # stride 정책
            def stride_for(win):
                return STRIDE_SEC if (STRIDE_SEC and STRIDE_SEC>0) else (win*STRIDE_FRAC)
            # RR 샘플
            for win_sec in RR_WIN_LIST:
                st = stride_for(win_sec)
                for s,e,_ in _make_windows(S, fs, win_sec, st):
                    xin = X[s:e]; yrr = y_rr_tgt[s:e]
                    # RR 샘플에서는 RR만 사용, HR은 마스크 0 / 값은 dummy
                    if xin.shape[0] == int(round(win_sec*fs)):
                        if y_hr_tgt is not None:
                            yhr = y_hr_tgt[s:e]
                        else:
                            yhr = np.zeros_like(yrr)
                        mask_rr = np.ones_like(yrr, dtype=np.float32)
                        mask_hr = np.zeros_like(yrr, dtype=np.float32)
                        self.rows.append(dict(X=xin, y_rr=yrr, y_hr=yhr, 
                                              m_rr=mask_rr, m_hr=mask_hr,
                                              subject=int(E["subject"]), session=int(E["session"])))
            # HR 샘플
            for win_sec in HR_WIN_LIST:
                st = stride_for(win_sec)
                for s,e,_ in _make_windows(S, fs, win_sec, st):
                    xin = X[s:e]; yrr = y_rr_tgt[s:e]
                    # HR 샘플에서는 HR만 사용, RR은 마스크 0
                    if y_hr_tgt is None:
                        # HR GT가 없으면 생성 안함
                        continue
                    yhr = y_hr_tgt[s:e]
                    if xin.shape[0] == int(round(win_sec*fs)) and yhr.shape[0] == int(round(win_sec*fs)):
                        mask_rr = np.zeros_like(yrr, dtype=np.float32)
                        mask_hr = np.ones_like(yhr, dtype=np.float32)
                        self.rows.append(dict(X=xin, y_rr=yrr, y_hr=yhr, 
                                              m_rr=mask_rr, m_hr=mask_hr,
                                              subject=int(E["subject"]), session=int(E["session"])))

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
    T = max(lens)
    B = len(batch)
    Xp = torch.zeros((B, T, Xs[0].shape[1]), dtype=torch.float32)
    Yp = torch.zeros((B, T, Ys[0].shape[1]), dtype=torch.float32)
    Mp = torch.zeros((B, T, Ms[0].shape[1]), dtype=torch.float32)
    Pm = torch.zeros((B, T, 1), dtype=torch.float32)
    for i,(x,y,m) in enumerate(zip(Xs,Ys,Ms)):
        L = x.shape[0]
        Xp[i,:L,:] = x
        Yp[i,:L,:] = y
        Mp[i,:L,:] = m
        Pm[i,:L,0] = 1.0
    return Xp, Yp, Mp, Pm, torch.tensor(subs), torch.tensor(sess)
