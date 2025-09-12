import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from pose_backend import make_pose_landmarker, extract_displacements
from . import config
from .config import DATA_ROOT, CACHE_DIR, FS_EXTRACT, FS_MODEL
from .utils import resample_uniform, align_common_time, estimate_global_lag


def load_h5(h5_path) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        def get(k): return np.asarray(f[k][:], np.float32) if k in f else None
        time = get("time")
        resp = get("respiration")
        ecg  = get("ecg") if "ecg" in keys else (get("pulse") if "pulse" in keys else (get("ppg") if "ppg" in keys else None))
    return dict(time=time, respiration=resp, cardio=ecg)

def build_index() -> List[Tuple[int,int,str,str]]:
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"COHFACE_ROOT not found: {DATA_ROOT}")
    subs = sorted([int(d) for d in os.listdir(DATA_ROOT) if d.isdigit()])
    pairs=[]
    for s in subs:
        d = os.path.join(DATA_ROOT, str(s))
        for k in [0,1,2,3]:
            v = os.path.join(d, str(k), "data.mkv")
            h = os.path.join(d, str(k), "data.hdf5")
            if os.path.exists(v) and os.path.exists(h): pairs.append((s,k,v,h))
    return pairs

def cache_path(subject:int, sess:int) -> str:
    return os.path.join(CACHE_DIR, f"s{subject}_k{sess}.npz")

def process_one(subject:int, sess:int, vid_path:str, h5_path:str, pose_backend:str, pose_handle):
    out_path = cache_path(subject, sess)
    if os.path.exists(out_path): return out_path  # 이미 처리됨

    # 1) 비디오 → dW/dY/dD
    ts, dW, dY, dD = extract_displacements(vid_path, pose_backend, pose_handle)

    # 2) H5 → GT
    gt = load_h5(h5_path)
    gt_t, resp, cardio = gt["time"], gt["respiration"], gt["cardio"]
    if ts is None or resp is None or gt_t is None: return None

    # 3) 전역 래그
    lag = estimate_global_lag(ts, dW, dY, dD, gt_t, resp, fs=FS_EXTRACT)

    # 4) 공통 시간축 정렬 (FS_EXTRACT)
    tu, wU, _ = resample_uniform(ts, dW, FS_EXTRACT)
    _,  yU, _ = resample_uniform(ts, dY, FS_EXTRACT)
    _,  dU, _ = resample_uniform(ts, dD, FS_EXTRACT)
    tg, gU, _ = resample_uniform(gt_t + lag, resp, FS_EXTRACT)
    tC, [wC,yC,dC], gC = align_common_time(tu, [wU,yU,dU], tg, gU, FS_EXTRACT)

    # 5) 모델 샘플레이트로 다운샘플
    tM, wM, _ = resample_uniform(tC, wC, FS_MODEL)
    _,  yM, _ = resample_uniform(tC, yC, FS_MODEL)
    _,  dM, _ = resample_uniform(tC, dC, FS_MODEL)
    _,  gM, _ = resample_uniform(tC, gC, FS_MODEL)

    cM = None
    if cardio is not None and gt_t is not None:
        tg2, cU, _ = resample_uniform(gt_t + lag, cardio, FS_EXTRACT)
        if tg2 is not None:
            cA = np.interp(tC, tg2, cU).astype(np.float32)
            _, cM, _ = resample_uniform(tC, cA, FS_MODEL)

    np.savez_compressed(out_path, t=tM, dW=wM, dY=yM, dD=dM, g_resp=gM, g_cardio=cM,
                        lag=lag, subject=subject, session=sess, vid=vid_path, h5=h5_path)
    return out_path

def extract_and_cache(root, subject, session, out_dir, fs, resp_band):
    # 호출 시점 설정 오버라이드
    config.DATA_ROOT = root
    config.CACHE_DIR = out_dir
    config.FS_EXTRACT = fs
    config.FS_MODEL = fs
    config.RESP_BAND = tuple(resp_band)

    vid_path = os.path.join(root, str(subject), str(session), "data.mkv")
    h5_path  = os.path.join(root, str(subject), str(session), "data.hdf5")
    pose_backend, pose_handle = make_pose_landmarker(use_gpu=True)
    try:
        return process_one(subject, session, vid_path, h5_path, pose_backend, pose_handle)
    finally:
        if pose_backend == "solutions":
            try: pose_handle.close()
            except: pass

def main():
    pose_backend, pose_handle = make_pose_landmarker(use_gpu=True)
    pairs = build_index()
    print(f"[extract] total sessions: {len(pairs)}")
    done=0
    for s,k,v,h in tqdm(pairs, desc="Extracting"):
        p = process_one(s,k,v,h, pose_backend, pose_handle)
        if p: done+=1
    print(f"[extract] cached: {done}/{len(pairs)}")

    if pose_backend == "solutions":
        try: pose_handle.close()
        except: pass

if __name__ == "__main__":
    main()
