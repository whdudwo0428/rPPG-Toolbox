# -*- coding: utf-8 -*-
"""
best_model.pt 재평가 (FULL) + 테스트 세션 4개 오버레이 플롯 저장
- test 전 윈도우 평가: evaluate() 재사용 → metrics_test_full.json
- 플롯: GT vs Pred(aligned) 4장 PNG (plots/)
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DEVICE, FS_MODEL
from .data import CohfaceSeqDataset
from .models import SeqRegressor
from .sampler import BucketBatchSampler
from .train import evaluate
from .utils import align_scale_np, _minmax01, _minmax11


def parse_bucket_bs(s: str):
    mp = {}
    for kv in (s or "").split(","):
        kv = kv.strip()
        if not kv:
            continue
        L, b = kv.split(":")
        mp[int(L)] = int(b)
    return mp


def _pick_sessions(ds, n=4):
    # 데이터셋이 세션 단위 메타를 제공한다고 가정 (ds.files / ds.sessions)
    picks = []
    for sid in range(len(getattr(ds, "files", []))):
        picks.append(sid)
        if len(picks) >= n:
            break
    if not picks:
        picks = list(range(min(n, len(ds))))
    return picks


def _plot_one_session(model, X, G, out_png, fs=FS_MODEL, vis_norm: str = "none"):
    with torch.no_grad():
        P = model(torch.from_numpy(X).unsqueeze(0).to(DEVICE).float())  # [1,T,1]
        P = P.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)

    P_aligned, _a = align_scale_np(P, G)

    # 플롯 전용 정규화 (지표에는 영향 없음)
    Gb = _apply_vis_norm(G, vis_norm)
    Pb = _apply_vis_norm(P_aligned, vis_norm)

    t = np.arange(len(G)) / float(fs)
    plt.figure(figsize=(10, 3))
    plt.plot(t, Gb, label=("GT" if vis_norm == "none" else f"GT({vis_norm})"), linewidth=1.0)
    plt.plot(t, Pb, label=("Pred(aligned)" if vis_norm == "none" else f"Pred(aligned,{vis_norm})"),
             linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("RR (a.u.)")
    plt.title(os.path.basename(out_png).replace(".png", ""))
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _zscore_np(x, eps=1e-8):
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + eps)


def _apply_vis_norm(arr: np.ndarray, mode: str) -> np.ndarray:
    """플롯용 정규화(지표에는 영향 없음)"""
    if mode == "none":
        return arr
    if mode == "zscore":
        return _zscore_np(arr)
    if mode == "minmax01":
        if _minmax01 is not None:
            return _minmax01(arr)
        # 안전 폴백
        lo, hi = float(np.min(arr)), float(np.max(arr))
        return (arr - lo) / max(hi - lo, 1e-8)
    if mode == "minmax11":
        if _minmax11 is not None:
            return _minmax11(arr)
        # 안전 폴백: [0,1] → [-1,1]
        lo, hi = float(np.min(arr)), float(np.max(arr))
        y = (arr - lo) / max(hi - lo, 1e-8)
        return y * 2.0 - 1.0
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bidir", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bucket_bs", type=str, default="10240:4,5120:8")
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--pin_memory", type=int, default=1)
    ap.add_argument("--n_plots", type=int, default=4)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--vis_norm", type=str, default="none",
                    choices=["none", "minmax01", "minmax11", "zscore"],
                    help="플롯용 정규화(지표 계산에는 영향 없음)")
    args = ap.parse_args()

    out_root = args.outdir or os.path.dirname(args.model) or "."
    os.makedirs(out_root, exist_ok=True)
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Dataset & FULL-window loader
    ds = CohfaceSeqDataset(args.cache, subset="test")
    bucket = parse_bucket_bs(args.bucket_bs)
    ld = DataLoader(
        ds,
        batch_sampler=BucketBatchSampler(ds, bucket, shuffle=False),
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    # Model load (weights-only 안전 모드 지원)
    try:
        state = torch.load(args.model, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(args.model, map_location=DEVICE)
    model = SeqRegressor(in_dim=16, hidden=args.hidden, layers=args.layers,
                         bidir=bool(args.bidir), dropout=args.dropout).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # === FULL test metrics (전 창) + 4-session overlays (샘플) ===
    m_test = evaluate(model, ld)
    picks = _pick_sessions(ds, n=args.n_plots)
    saved = []
    for sid in picks:
        sess = ds.sessions[sid]
        X = sess["X"]
        G = sess["Y"]
        # 중앙 40s(or 20s) 클립
        Tprefer = int(40 * FS_MODEL) if len(G) >= int(40 * FS_MODEL) else int(20 * FS_MODEL)
        st = max(0, (len(G) - Tprefer) // 2)
        ed = st + Tprefer
        png = os.path.join(plot_dir, f"test_session{sid:02d}_[{os.path.basename(ds.files[sid])}].png")
        _plot_one_session(model, X[st:ed], G[st:ed], png, fs=FS_MODEL, vis_norm=args.vis_norm)
        saved.append(png)
    # === 하나의 JSON으로 합치기 ===
    combined = dict(m_test)
    combined.update({
        "plots_saved_to": plot_dir,
        "plotted_sessions": [os.path.basename(ds.files[sid]) for sid in picks],
        "plot_files": [os.path.basename(p) for p in saved],
        # 평가 설정 요약(재현성 기록)
        "bpm_fallback_argmax": int(os.getenv("BPM_FALLBACK_ARGMAX", "1")),
        "bpm_subbin_quad": int(os.getenv("BPM_SUBBIN_QUAD", "1")),
        "bpm_nfft_up": int(os.getenv("BPM_NFFT_UP", "1")),
    })
    out_json = os.path.join(out_root, "metrics_test.json")
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print("[test-full+plots]", json.dumps(combined, ensure_ascii=False))


if __name__ == "__main__":
    main()