# cohface_exp_reg/run_extract_all.py
# -*- coding: utf-8 -*-
"""
유연 탐색 + 다양한 preprocess 시그니처 자동호출 버전
- subject/session: 01/00 형식과 1/0 형식 모두 지원
- 비디오 확장자: .mkv/.mp4/.avi  (data.mkv 우선)
- 라벨  확장자: .hdf5/.h5       (data.hdf5 우선)
- preprocess 모듈의 함수 시그니처를 자동 판별하여 호출:
  1) 경로기반: process_session(video, label, out, [fs], [resp_band], ...)
  2) 루트기반: extract_and_cache(root, subj, sess, out, [fs], [resp_band], ...)
"""

import argparse
import glob
import inspect
import os
from typing import List, Tuple, Optional

from tqdm import tqdm

from . import preprocess
from .config import DATA_ROOT, CACHE_DIR, FS_RESAMP, RESP_BAND


# -------------------------- 유틸: 입력 파싱/확장 -------------------------- #
def _expand(spec: Optional[str]) -> Optional[List[int]]:
    """'1-5,7,10-12' -> [1,2,3,4,5,7,10,11,12]; None이면 None 유지"""
    if spec is None or spec == "":
        return None
    out: List[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


# -------------------------- 데이터 탐색(유연) ----------------------------- #
def _find_session_files(root: str, s: int, k: int) -> Optional[Tuple[str, str, str, str]]:
    """
    - subject 디렉토리: f"{s:02d}" 또는 f"{s}"
    - session 디렉토리: f"{k:02d}" 또는 f"{k}"
    - 비디오: *.mkv | *.mp4 | *.avi (data.mkv 우선)
    - 라벨 : *.hdf5 | *.h5          (data.hdf5 우선)
    둘 다 존재하는 경우 (sd, kd, v_path, h_path) 반환, 없으면 None
    """
    cand_s = [f"{s:02d}", f"{s}"]
    cand_k = [f"{k:02d}", f"{k}"]
    for sd in cand_s:
        for kd in cand_k:
            base = os.path.join(root, sd, kd)
            if not os.path.isdir(base):
                continue
            vids: List[str] = []
            for pat in ("*.mkv", "*.mp4", "*.avi"):
                vids.extend(glob.glob(os.path.join(base, pat)))
            labs: List[str] = []
            for pat in ("*.hdf5", "*.h5"):
                labs.extend(glob.glob(os.path.join(base, pat)))
            if vids and labs:
                v_pref = os.path.join(base, "data.mkv")
                h_pref = os.path.join(base, "data.hdf5")
                v = v_pref if os.path.exists(v_pref) else sorted(vids)[0]
                h = h_pref if os.path.exists(h_pref) else sorted(labs)[0]
                return sd, kd, v, h
    return None


def _autoscan(root: str) -> Tuple[List[int], List[int]]:
    """
    루트에서 subject 디렉토리와 공통 session 집합(교집합)을 자동 탐색
    - 제로패딩/비패딩 혼재 가능 → 디렉토리명 숫자만 추출
    """
    subs: List[int] = []
    sess_map = {}
    for d in sorted(os.listdir(root), key=lambda x: (len(x), x)):
        if d.isdigit():
            sdir = os.path.join(root, d)
            if not os.path.isdir(sdir):
                continue
            sid = int(d)
            subs.append(sid)
            sessions = set()
            for x in os.listdir(sdir):
                if x.isdigit():
                    kdir = os.path.join(sdir, x)
                    if not os.path.isdir(kdir):
                        continue
                    vids = any(glob.glob(os.path.join(kdir, ext)) for ext in ("*.mkv", "*.mp4", "*.avi"))
                    labs = any(glob.glob(os.path.join(kdir, ext)) for ext in ("*.hdf5", "*.h5"))
                    if vids and labs:
                        sessions.add(int(x))
            sess_map[sid] = sorted(sessions)
    common_sessions = sorted(set.intersection(*[set(v) for v in sess_map.values()])) if sess_map else []
    return subs, common_sessions


# -------------------------- preprocess 호출 어댑터 ------------------------ #
def _call_preprocess_paths(fn, video: str, label: str, out_path: str, fs: float, resp_band: Tuple[float, float]) -> bool:
    """
    process_session(video, label, out_path, [fs], [resp_band], **kw) 시그니처 지원
    """
    sig = inspect.signature(fn)
    kwargs = {}
    if "fs" in sig.parameters:
        kwargs["fs"] = fs
    if "resp_band" in sig.parameters:
        kwargs["resp_band"] = resp_band
    try:
        # 가장 표준: (video, label, out, **kwargs)
        fn(video, label, out_path, **kwargs)
        return True
    except TypeError:
        # 최소 시그니처: (video, label, out)
        fn(video, label, out_path)
        return True


def _call_preprocess_root(fn, root: str, subj: int, sess: int, out_dir: str, fs: float, resp_band: Tuple[float, float]) -> bool:
    """
    extract_and_cache(root, subj, sess, out, [fs], [resp_band]) 시그니처 지원
    """
    sig = inspect.signature(fn)
    kwargs = {}
    if "fs" in sig.parameters:
        kwargs["fs"] = fs
    if "resp_band" in sig.parameters:
        kwargs["resp_band"] = resp_band
    try:
        fn(root, subj, sess, out_dir, **kwargs)
        return True
    except TypeError:
        fn(root, subj, sess, out_dir, fs, resp_band)
        return True


def _dispatch_preprocess(root: str, s: int, k: int, v: str, h: str,
                         out_dir: str, fs: float, resp_band: Tuple[float, float]) -> bool:
    """
    preprocess 모듈에서 사용 가능한 함수를 자동 선택하여 호출
    우선순위:
      1) 경로 기반: process_session / process_one / run_one
      2) 루트 기반: extract_and_cache / extract_session / run_one
    """
    # 1) 경로 기반 후보
    for name in ("process_session", "process_one", "run_one"):
        fn = getattr(preprocess, name, None)
        if callable(fn):
            try:
                out_path = os.path.join(out_dir, f"s{int(s):02d}_k{int(k):02d}.npz")
                return _call_preprocess_paths(fn, v, h, out_path, fs, resp_band)
            except Exception as e:
                print(f"[warn] {name}(paths) 실패: {e}")
    # 2) 루트 기반 후보
    for name in ("extract_and_cache", "extract_session", "run_one"):
        fn = getattr(preprocess, name, None)
        if callable(fn):
            try:
                return _call_preprocess_root(fn, root, s, k, out_dir, fs, resp_band)
            except Exception as e:
                print(f"[warn] {name}(root) 실패: {e}")
    raise RuntimeError("preprocess 모듈에 호출 가능한 추출 함수가 없습니다. "
                       "예: process_session(video, label, out[, fs, resp_band]) 또는 "
                       "extract_and_cache(root, subject, session, out[, fs, resp_band])")


# -------------------------- 메인 --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=DATA_ROOT)
    ap.add_argument("--subjects", type=str, default=None, help="예) '1-40' 또는 '1-10,12'")
    ap.add_argument("--sessions", type=str, default=None, help="예) '0-3' 또는 '0,2'")
    ap.add_argument("--out", type=str, default=CACHE_DIR)
    ap.add_argument("--fs", type=float, default=FS_RESAMP, help=f"resample fs (default: {FS_RESAMP})")
    ap.add_argument("--resp_lo", type=float, default=RESP_BAND[0])
    ap.add_argument("--resp_hi", type=float, default=RESP_BAND[1])
    args = ap.parse_args()

    root = args.root
    out = args.out
    fs = float(args.fs)
    resp_band = (float(args.resp_lo), float(args.resp_hi))
    os.makedirs(out, exist_ok=True)

    Ss = _expand(args.subjects)
    Ks = _expand(args.sessions)
    if Ss is None or Ks is None:
        subs_all, sess_common = _autoscan(root)
        if Ss is None:
            Ss = subs_all
        if Ks is None:
            Ks = sess_common
        if Ss and Ks:
            print(f"[auto] subjects={Ss[0]}-{Ss[-1]}  sessions={Ks}")

    # 페어링
    pairs: List[Tuple[int, int, str, str, str, str]] = []
    miss_samples: List[Tuple[int, int]] = []
    for s in Ss or []:
        for k in Ks or []:
            hit = _find_session_files(root, s, k)
            if hit is not None:
                sd, kd, v, h = hit
                pairs.append((s, k, sd, kd, v, h))
            else:
                if len(miss_samples) < 5:
                    miss_samples.append((s, k))

    print(f"[extract] total sessions = {len(pairs)}")
    if len(pairs) == 0:
        print(f"[hint] root='{root}' 에서 탐색 실패. 예시 미스: {miss_samples}")
        try:
            lvl1 = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])[:10]
            print(f"[hint] 1단계 디렉토리 예시(10개): {lvl1}")
            if lvl1:
                first = os.path.join(root, lvl1[0])
                lvl2 = sorted([d for d in os.listdir(first) if os.path.isdir(os.path.join(first, d))])[:10]
                print(f"[hint] '{lvl1[0]}' 하위(10개): {lvl2}")
        except Exception as e:
            print(f"[hint] 디렉토리 프리뷰 실패: {e}")

    done = 0
    for (s, k, sd, kd, v, h) in tqdm(pairs, desc="Extract"):
        try:
            ok = _dispatch_preprocess(root, s, k, v, h, out, fs, resp_band)
            if ok:
                done += 1
        except Exception as e:
            print(f"[ERR] s={s}, k={k}: {e}")

    print(f"[extract] cached: {done}/{len(pairs)}")


if __name__ == "__main__":
    main()