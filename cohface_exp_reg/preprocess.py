# -*- coding: utf-8 -*-
"""
전처리/캐시: 비디오→(dW,dY,dD,dD_perp), H5→resp, 공통 시간축 리샘플(256 Hz)
출력: cache_dir/s{subject}_k{session}.npz  (keys: t,dW,dY,dD,dD_perp,resp)
"""
import os, glob, h5py, numpy as np
from scipy.interpolate import interp1d
from .pose_backend import extract_displacements
from .config import FS_RESAMP

def _resample(ts, xs, fs=256.0):
    if len(ts) < 2:
        return None, None
    t0, t1 = ts[0], ts[-1]
    T = int(round((t1 - t0) * fs)) + 1
    t_new = np.linspace(t0, t1, T)
    out = []
    for x in xs:
        f = interp1d(ts, x, kind='linear', fill_value="extrapolate", bounds_error=False)
        out.append(f(t_new))
    return t_new, out

def process_session(video_path, h5_path, out_path):
    if os.path.exists(out_path):
        return out_path
    ts, dW, dY, dD, dD_perp = extract_displacements(video_path)
    with h5py.File(h5_path, "r") as f:
        gt_t = np.asarray(f["time"][:]).astype(np.float32)
        resp = np.asarray(f["respiration"][:]).astype(np.float32)
    # 공통 resample
    t1, [w1,y1,d1,dp1] = _resample(ts, [dW,dY,dD,dD_perp], fs=FS_RESAMP)
    t2, [r1]           = _resample(gt_t, [resp], fs=FS_RESAMP)
    # 시간 교집합
    t0 = max(t1[0], t2[0]); tE = min(t1[-1], t2[-1])
    mask1 = (t1>=t0)&(t1<=tE)
    mask2 = (t2>=t0)&(t2<=tE)
    # 같은 길이로 자르기
    L = min(mask1.sum(), mask2.sum())
    t  = t1[mask1][:L]
    dW = w1[mask1][:L]; dY = y1[mask1][:L]; dD = d1[mask1][:L]; dD_perp = dp1[mask1][:L]
    resp = r1[mask2][:L]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, t=t, dW=dW, dY=dY, dD=dD, dD_perp=dD_perp, resp=resp)
    return out_path
