# -*- coding: utf-8 -*-
"""
재평가/재플로팅 유틸
- 저장된 best_model.pt 로드 → val/test 평가 → 플롯 생성
- runs/exp_results/ (long CSV, wide CSV, per-run JSON) 동일 포맷으로 append
사용 예:
python -m cohface_exp_reg.run_eval_best \
  --cache cohface_exp_reg/cache_cohface_feats \
  --model runs/<EXP_NAME>/model_rronly_time/<TS>/best_model.pt \
  --n_plots 6 --vis_norm minmax01
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DEVICE, FS_MODEL, RR_WIN_LIST, STRIDE_FRAC, STRIDE_FRAC_LIST, FIXED_STRIDE, \
    FIXED_STRIDE_LIST, WINDOW_PAD_MODE, WINDOW_INCLUDE_TAIL
from .data import CohfaceSeqDataset
from .models import SeqRegressor
from .sampler import BucketBatchSampler
from .train import evaluate
from .utils import align_scale_np, welch_psd_rr_bpm, append_exp_results


def _save_plots(model, loader, out_dir: Path, n_plots=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for X, Y in loader:
        with torch.no_grad():
            P = model(X.to(DEVICE).float()).squeeze(-1).cpu().numpy()
        G = Y.squeeze(-1).numpy()
        for b in range(P.shape[0]):
            pb, gb = P[b].astype(np.float32), G[b].astype(np.float32)
            pa, _ = align_scale_np(pb, gb)
            # 정규화
            lo, hi = float(np.min(pa)), float(np.max(pa))
            pa_n = pa if hi - lo < 1e-8 else (pa - lo) / (hi - lo)
            lo, hi = float(np.min(gb)), float(np.max(gb))
            gb_n = gb if hi - lo < 1e-8 else (gb - lo) / (hi - lo)
            t = np.arange(len(pa)) / FS_MODEL
            plt.figure(figsize=(10, 3))
            plt.plot(t, gb_n, label="GT")
            plt.plot(t, pa_n, label="Pred(aligned)")

            # BPM 텍스트
            def bpm(x):
                v = welch_psd_rr_bpm(x, FS_MODEL)
                return np.nan if v is None else float(v)

            plt.title(f"GT={bpm(gb):.2f} bpm | Pred={bpm(pa):.2f} bpm")
            plt.xlabel("Time (s)")
            plt.ylabel("norm. amp")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(out_dir / f"test_{saved:02d}.png", dpi=120)
            plt.close()
            saved += 1
            if saved >= n_plots: return


def parse_bucket_bs(s: str):
    mp = {}
    for kv in (s or "").split(","):
        kv = kv.strip()
        if not kv:
            continue
        L, b = kv.split(":")
        mp[int(L)] = int(b)
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True, help="path to best_model.pt")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bidir", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bucket_bs", type=str, default=os.getenv("BUCKET_BS", "2560:12,3840:10,5120:8,7680:5,10240:4"))
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--pin_memory", type=int, default=1)
    ap.add_argument("--n_plots", type=int, default=4)
    args = ap.parse_args()

    # 데이터셋/로더
    t0 = time.perf_counter()
    val_set = CohfaceSeqDataset(args.cache, subset="val")
    test_set = CohfaceSeqDataset(args.cache, subset="test")
    bucket_map = parse_bucket_bs(args.bucket_bs)
    val_loader = DataLoader(val_set, batch_sampler=BucketBatchSampler(val_set, bucket_map, shuffle=False),
                            num_workers=args.num_workers, pin_memory=bool(args.pin_memory))
    test_loader = DataLoader(test_set, batch_sampler=BucketBatchSampler(test_set, bucket_map, shuffle=False),
                             num_workers=args.num_workers, pin_memory=bool(args.pin_memory))
    time_data = time.perf_counter() - t0

    # 모델
    model = SeqRegressor(input_dim=16, hidden=args.hidden, num_layers=args.layers,
                         bidir=bool(args.bidir), dropout=args.dropout).to(DEVICE)
    sd = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(sd)

    # 출력 디렉토리: 모델 경로 기준
    mpath = Path(args.model).resolve()
    base = mpath.parent  # .../model_rronly_time/<TS>
    out = base / f"reval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    # 평가
    t1 = time.perf_counter()
    val_metrics = evaluate(model, val_loader)
    tval = time.perf_counter() - t1
    t2 = time.perf_counter()
    test_metrics = evaluate(model, test_loader)
    ttest = time.perf_counter() - t2

    # 저장
    with (out / "metrics.json").open("w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2, ensure_ascii=False)
    _save_plots(model, test_loader, plots, n_plots=args.n_plots)

    # 집계(동일 포맷)
    settings = {
        "wins": RR_WIN_LIST,
        "stride_frac": STRIDE_FRAC,
        "stride_frac_list": STRIDE_FRAC_LIST,
        "fixed_stride": FIXED_STRIDE,
        "fixed_stride_list": FIXED_STRIDE_LIST,
        "pad_mode": WINDOW_PAD_MODE,
        "include_tail": WINDOW_INCLUDE_TAIL,
        "reval_from": str(args.model)
    }
    timings = {
        "time_total_s": round(time_data + tval + ttest, 3),
        "time_data_s": round(time_data, 3),
        "time_eval_val_s": round(tval, 3),
        "time_eval_test_s": round(ttest, 3),
    }
    flat_metrics = {
        "val_corr": val_metrics.get("corr"),
        "val_corr_bestlag": val_metrics.get("corr_bestlag"),
        "val_rr_bpm_mae": val_metrics.get("rr_bpm_mae"),
        "val_mse": val_metrics.get("mse"), "val_mae": val_metrics.get("mae"),
        "test_corr": test_metrics.get("corr"),
        "test_corr_bestlag": test_metrics.get("corr_bestlag"),
        "test_rr_bpm_mae": test_metrics.get("rr_bpm_mae"),
        "test_mse": test_metrics.get("mse"), "test_mae": test_metrics.get("mae"),
    }
    # run_name: 원래 run + "+reval"
    run_name = base.parent.parent.name + "+" + base.name  # <EXP_NAME> + <TS>
    run_name = run_name + "+reval"
    append_exp_results(
        run_dir=str(out),
        run_name=run_name,
        metrics=flat_metrics,
        settings={"wins_stride": f"{RR_WIN_LIST}@{(STRIDE_FRAC_LIST or [STRIDE_FRAC])}",
                  "pad_mode": settings["pad_mode"], "include_tail": settings["include_tail"]},
        timings=timings,
        by_win={}
    )


if __name__ == "__main__":
    main()
