# -*- coding: utf-8 -*-
"""
학습+평가+플롯 일체화 실행기
- (중첩 디렉토리) runs/<EXP_NAME>/model_rronly_time/<timestamp>/{best_model.pt, metrics.json, plots/*.png}
- runs/exp_results/ 에 (1) wide pivot CSV(summary.csv), (2) long CSV(exp_results.csv), (3) per-run JSON 저장
- 윈도우/스트라이드별(by-win) 성능·시간도 JSON으로 함께 기록
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import RUNS_DIR, DEVICE, LR, FS_MODEL, RR_WIN_LIST, STRIDE_FRAC, STRIDE_FRAC_LIST, FIXED_STRIDE, \
    FIXED_STRIDE_LIST, WINDOW_PAD_MODE, WINDOW_INCLUDE_TAIL
from .data import CohfaceSeqDataset
from .models import SeqRegressor
from .sampler import BucketBatchSampler
from .train import train_loop, evaluate
from .utils import align_scale_np, welch_psd_rr_bpm, append_exp_results  # append_exp_results 신규


# -------------------- helpers --------------------
def parse_bucket_bs(s: str):
    mp = {}
    for kv in (s or "").split(","):
        kv = kv.strip()
        if not kv: continue
        L, b = kv.split(":")
        mp[int(L)] = int(b)
    return mp


def append_exp_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if need_header: w.writeheader()
        w.writerow(row)


def _exp_name() -> str:
    """윈도우/스트라이드로 사람이 읽기 쉬운 실험 이름"""
    wins = RR_WIN_LIST
    if STRIDE_FRAC_LIST:
        s = [float(x) for x in STRIDE_FRAC_LIST]
        part = ",".join(f"{int(w)}@{int(100 * f)}" for w, f in zip(wins, s))
    elif STRIDE_FRAC is not None and STRIDE_FRAC != "":
        part = ",".join(f"{int(w)}@{int(100 * float(STRIDE_FRAC))}" for w in wins)
    elif FIXED_STRIDE_LIST:
        s = [float(x) for x in FIXED_STRIDE_LIST]
        part = ",".join(f"{int(w)}@{float(ss):.2f}s" for w, ss in zip(wins, s))
    elif FIXED_STRIDE not in (None, ""):
        part = ",".join(f"{int(w)}@{float(FIXED_STRIDE):.2f}s" for w in wins)
    else:
        part = ",".join(str(int(w)) for w in wins)
    run_group = os.getenv("RUN_GROUP", "lstm_rronly")
    return f"{run_group}__ws[{','.join(str(int(w)) for w in wins)}]__st[{part}]"


def _save_plots(model, test_loader, out_dir: Path, n_plots=4, vis_norm="minmax01"):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for X, Y in test_loader:
        with torch.no_grad():
            P = model(X.to(DEVICE).float()).squeeze(-1).cpu().numpy()  # [B,T]
        G = Y.squeeze(-1).numpy()
        B = P.shape[0]
        for b in range(B):
            pb = P[b].astype(np.float32)
            gb = G[b].astype(np.float32)
            pa, _ = align_scale_np(pb, gb)

            # 보기 좋게 정규화
            def norm(x):
                lo, hi = float(np.min(x)), float(np.max(x))
                if hi - lo <= 1e-8: return x
                return (x - lo) / (hi - lo)

            t = np.arange(len(pa)) / FS_MODEL
            plt.figure(figsize=(10, 3))
            plt.plot(t, norm(gb), label="GT")
            plt.plot(t, norm(pa), label="Pred(aligned)", alpha=0.9)

            # BPM 텍스트
            def bpm(x):
                v = welch_psd_rr_bpm(x, FS_MODEL)
                return np.nan if v is None else float(v)

            plt.title(f"GT={bpm(gb):.2f} bpm | Pred={bpm(pa):.2f} bpm")
            plt.xlabel("Time (s)")
            plt.ylabel("norm. amp")
            plt.legend(loc="upper right")
            png = out_dir / f"test_{saved:02d}.png"
            plt.tight_layout()
            plt.savefig(png, dpi=120)
            plt.close()
            saved += 1
            if saved >= n_plots: return


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "50")))
    ap.add_argument("--lr", type=float, default=float(os.getenv("LR", str(LR))))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bidir", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bucket_bs", type=str, default=os.getenv("BUCKET_BS", "2560:12,3840:10,5120:8,7680:5,10240:4"))
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--pin_memory", type=int, default=1)
    args = ap.parse_args()

    # ── dataset ────────────────────────────────────────────────────────
    t_data0 = time.perf_counter()
    train_set = CohfaceSeqDataset(args.cache, subset="train")
    val_set = CohfaceSeqDataset(args.cache, subset="val")
    test_set = CohfaceSeqDataset(args.cache, subset="test")
    bucket_map = parse_bucket_bs(args.bucket_bs)

    train_loader = DataLoader(
        train_set, batch_sampler=BucketBatchSampler(train_set, bucket_map, shuffle=True),
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory),
    )
    val_loader = DataLoader(
        val_set, batch_sampler=BucketBatchSampler(val_set, bucket_map, shuffle=False),
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory),
    )
    test_loader = DataLoader(
        test_set, batch_sampler=BucketBatchSampler(test_set, bucket_map, shuffle=False),
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory),
    )
    time_data = time.perf_counter() - t_data0

    # ── model/opt ────────────────────────────────────────────────────────
    model = SeqRegressor(in_dim=16, hidden=args.hidden, layers=args.layers,
                         bidir=bool(args.bidir), dropout=args.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── directories ─────────────────────────────────────────────────────
    exp_name = _exp_name()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(RUNS_DIR) / exp_name / "model_rronly_time"
    run_dir = run_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── train ────────────────────────────────────────────────────────────
    t_train0 = time.perf_counter()
    model = train_loop(model, opt, train_loader, val_loader, epochs=args.epochs, device=DEVICE)
    time_train = time.perf_counter() - t_train0

    # ── eval (val/test) ─────────────────────────────────────────────────
    t_val0 = time.perf_counter()
    val_metrics = evaluate(model, val_loader)
    time_val = time.perf_counter() - t_val0

    t_test0 = time.perf_counter()
    test_metrics = evaluate(model, test_loader)
    time_test = time.perf_counter() - t_test0

    # ── plots ───────────────────────────────────────────────────────────
    _save_plots(model, test_loader, plots_dir, n_plots=4, vis_norm="minmax01")

    # ── save (best_model + metrics.json) ────────────────────────────────
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    with (run_dir / "metrics.json").open("w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2, ensure_ascii=False)

    # ── timings & settings 기록 ─────────────────────────────────────────
    time_total = time_data + time_train + time_val + time_test
    settings = {
        "wins": RR_WIN_LIST,
        "stride_frac": STRIDE_FRAC,
        "stride_frac_list": STRIDE_FRAC_LIST,
        "fixed_stride": FIXED_STRIDE,
        "fixed_stride_list": FIXED_STRIDE_LIST,
        "pad_mode": WINDOW_PAD_MODE,
        "include_tail": WINDOW_INCLUDE_TAIL,
        "bucket_bs": args.bucket_bs,
        "hidden": args.hidden, "layers": args.layers, "bidir": bool(args.bidir),
        "dropout": args.dropout, "lr": args.lr, "epochs": args.epochs,
    }
    timings = {
        "time_total_s": round(time_total, 3),
        "time_data_s": round(time_data, 3),
        "time_train_s": round(time_train, 3),
        "time_eval_val_s": round(time_val, 3),
        "time_eval_test_s": round(time_test, 3),
    }

    # ── long CSV(세로)  + per-run CSV ───────────────────────────────────
    row = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "exp_name": exp_name,
        "wins": settings["wins"],
        "stride_frac": settings["stride_frac"], "stride_frac_list": settings["stride_frac_list"],
        "fixed_stride": settings["fixed_stride"], "fixed_stride_list": settings["fixed_stride_list"],
        "pad_mode": settings["pad_mode"], "include_tail": settings["include_tail"],
        "bucket_bs": settings["bucket_bs"], "hidden": settings["hidden"],
        "layers": settings["layers"], "bidir": settings["bidir"], "dropout": settings["dropout"],
        "lr": settings["lr"], "epochs": settings["epochs"],
        "val_mse": val_metrics.get("mse"), "val_mae": val_metrics.get("mae"),
        "val_corr": val_metrics.get("corr"), "val_corr_bestlag": val_metrics.get("corr_bestlag"),
        "val_rr_bpm_mae": val_metrics.get("rr_bpm_mae"), "val_hit@±2bpm": val_metrics.get("hit@±2bpm"),
        "test_mse": test_metrics.get("mse"), "test_mae": test_metrics.get("mae"),
        "test_corr": test_metrics.get("corr"), "test_corr_bestlag": test_metrics.get("corr_bestlag"),
        "test_rr_bpm_mae": test_metrics.get("rr_bpm_mae"), "test_hit@±2bpm": test_metrics.get("hit@±2bpm"),
        "time_total_s": timings["time_total_s"],
        "time_data_s": timings["time_data_s"], "time_train_s": timings["time_train_s"],
        "time_eval_val_s": timings["time_eval_val_s"], "time_eval_test_s": timings["time_eval_test_s"],
    }
    append_exp_csv(Path(RUNS_DIR) / "exp_results" / "exp_results.csv", row)
    append_exp_csv(run_dir / "metrics_row.csv", row)

    # ── wide CSV(가로 pivot) + per-run JSON  ────────────────────────────
    flat_metrics = {
        "val_corr": val_metrics.get("corr"),
        "val_corr_bestlag": val_metrics.get("corr_bestlag"),
        "val_rr_bpm_mae": val_metrics.get("rr_bpm_mae"),
        "val_mse": val_metrics.get("mse"),
        "val_mae": val_metrics.get("mae"),
        "test_corr": test_metrics.get("corr"),
        "test_corr_bestlag": test_metrics.get("corr_bestlag"),
        "test_rr_bpm_mae": test_metrics.get("rr_bpm_mae"),
        "test_mse": test_metrics.get("mse"),
        "test_mae": test_metrics.get("mae"),
    }
    # evaluate()가 length별 집계를 지원하지 않는 경우를 대비해 빈 dict로 전달
    by_win = {}  # 필요 시 train.evaluate를 확장해 길이별 기록
    append_exp_results(
        run_dir=str(run_dir),
        run_name=run_dir.name,
        metrics=flat_metrics,
        settings={
            "wins_stride": f"{','.join(str(int(w)) for w in RR_WIN_LIST)}@{(STRIDE_FRAC_LIST or [STRIDE_FRAC])}",
            "pad_mode": settings["pad_mode"],
            "include_tail": settings["include_tail"],
        },
        timings=timings,
        by_win=by_win,
    )


if __name__ == "__main__":
    main()
