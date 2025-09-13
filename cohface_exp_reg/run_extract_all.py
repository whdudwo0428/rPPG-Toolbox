# -*- coding: utf-8 -*-
import argparse
import os

from tqdm import tqdm

from .config import DATA_ROOT, CACHE_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=DATA_ROOT)
    ap.add_argument("--subjects", type=str, default="1-40")
    ap.add_argument("--sessions", type=str, default="0-3")
    ap.add_argument("--out", type=str, default=CACHE_DIR)
    args = ap.parse_args()

    def _expand(spec):
        parts = []
        for tok in spec.split(","):
            if "-" in tok:
                a,b = tok.split("-")
                parts.extend(list(range(int(a), int(b)+1)))
            else:
                parts.append(int(tok))
        return parts

    Ss = _expand(args.subjects)
    Ks = _expand(args.sessions)

    pairs = []
    for s in Ss:
        for k in Ks:
            v = os.path.join(args.root, f"{s:02d}", f"{k:02d}", "data.mkv")
            h = os.path.join(args.root, f"{s:02d}", f"{k:02d}", "data.hdf5")
            if os.path.exists(v) and os.path.exists(h):
                pairs.append((s,k,v,h))
    print(f"[extract] total sessions = {len(pairs)}")

    from .preprocess import process_session
    done = 0
    for (s,k,v,h) in tqdm(pairs, desc="Extract"):
        out = os.path.join(args.out, f"s{s:02d}_k{k:02d}.npz")
        res = process_session(v, h, out)
        if res: done += 1
    print(f"[extract] cached: {done}/{len(pairs)}")

if __name__ == "__main__":
    main()
