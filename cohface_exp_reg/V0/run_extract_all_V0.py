# run_extract_all.py
import argparse
import os

from tqdm import tqdm

from . import preprocess
from . import config
from .config import DATA_ROOT, CACHE_DIR, FS_RESAMP, RESP_BAND


def parse_range(s):
    """ '1-5,7,10-12' -> [1,2,3,4,5,7,10,11,12] """
    if s is None or s == "":
        return None
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if "-" in tok:
            a,b = tok.split("-",1)
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(tok))
    return sorted(set(out))

def scan_all_subjects_sessions(root):
    subs = []
    sess_map = {}
    for d in sorted(os.listdir(root), key=lambda x: (len(x), x)):
        if d.isdigit():
            sdir = os.path.join(root, d)
            if not os.path.isdir(sdir): continue
            subs.append(int(d))
            sess = []
            for x in os.listdir(sdir):
                if x.isdigit():
                    if os.path.isfile(os.path.join(sdir, x, "data.mkv")) and \
                       (os.path.isfile(os.path.join(sdir, x, "data.hdf5")) or os.path.isfile(os.path.join(sdir, x, "data.h5"))):
                        sess.append(int(x))
            sess_map[int(d)] = sorted(sess)
    # 공통 세션(보통 0~3)
    common_sessions = sorted(set.intersection(*[set(v) for v in sess_map.values()])) if sess_map else []
    return subs, common_sessions

def call_preprocess(root, subj, sess, out_dir, fs, resp_band):
    """preprocess 모듈의 다양한 함수 이름을 안전 호출"""
    candidates = [
        "extract_and_cache",     # (root, subject, session, out_dir, fs, resp_band)
        "extract_session",
        "process_one",
        "run_one",
    ]
    for name in candidates:
        fn = getattr(preprocess, name, None)
        if callable(fn):
            return fn(root, subj, sess, out_dir, fs, resp_band)
    raise RuntimeError("preprocess 모듈에 호출 가능한 추출 함수가 없습니다. "
                       "extract_and_cache(root, subject, session, out_dir, fs, resp_band) 형태를 구현해주세요.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="COHFACE root (default: config.DATA_ROOT)")
    ap.add_argument("--subjects", default=None, help="예) 1-40 또는 1-10,12")
    ap.add_argument("--sessions", default=None, help="예) 0-3 또는 0,2")
    ap.add_argument("--out", default=None, help="캐시 출력 디렉토리 (default: config.CACHE_DIR)")
    ap.add_argument("--fs", type=float, default=None, help="resample fs (default: config.FS_RESAMP)")
    ap.add_argument("--resp_lo", type=float, default=None)
    ap.add_argument("--resp_hi", type=float, default=None)
    args = ap.parse_args()

    root = args.root or config.DATA_ROOT
    out = args.out or config.CACHE_DIR
    fs = args.fs or config.FS_RESAMP
    resp_band = (args.resp_lo, args.resp_hi) if args.resp_lo is not None else config.RESP_BAND
    lo   = args.resp_lo if args.resp_lo is not None else RESP_BAND[0]
    hi   = args.resp_hi if args.resp_hi is not None else RESP_BAND[1]

    os.makedirs(out, exist_ok=True)

    subs = parse_range(args.subjects)
    sess = parse_range(args.sessions)
    if subs is None or sess is None:
        # 인자 없으면 전체 스캔
        subs_all, sess_common = scan_all_subjects_sessions(root)
        if subs is None: subs = subs_all
        if sess  is None: sess  = sess_common
        print(f"[auto] subjects={subs[0]}-{subs[-1]}  sessions={sess}")

    print(f"[extract] total sessions: {len(subs)*len(sess)}")
    for s in tqdm(subs, desc="Extracting"):
        for u in sess:
            try:
                call_preprocess(root, s, u, out, fs, resp_band)
            except Exception as e:
                print(f"[ERR] s={s}, u={u}: {e}")

if __name__ == "__main__":
    main()
