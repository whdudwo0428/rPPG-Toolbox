# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import RUNS_DIR, DEVICE, LR
from .data import CohfaceSeqDataset
from .models import SeqRegressor
from .sampler import BucketBatchSampler
from .train import train_loop, evaluate, save_run


def parse_bucket_bs(s: str):
    mp = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        k, v = tok.split(":")
        mp[int(k)] = int(v)
    return mp


def append_csv(p: Path, row: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    need_header = (not p.exists()) or (p.stat().st_size == 0)
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if need_header: w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--bidir", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--pin_memory", type=int, default=1)
    ap.add_argument("--bucket_bs", type=str, default="2560:12,3840:10,5120:8,7680:5,10240:4")
    args = ap.parse_args()

    t0 = time.perf_counter()
    t0w = time.time()

    bucket = parse_bucket_bs(args.bucket_bs)
    train_set = CohfaceSeqDataset(args.cache, subset="train")
    val_set = CohfaceSeqDataset(args.cache, subset="val")
    test_set = CohfaceSeqDataset(args.cache, subset="test")

    tr = DataLoader(train_set, batch_sampler=BucketBatchSampler(train_set.length_buckets, bucket, shuffle=True),
                    num_workers=args.num_workers, pin_memory=bool(args.pin_memory))
    va = DataLoader(val_set, batch_sampler=BucketBatchSampler(val_set.length_buckets, bucket, shuffle=False),
                    num_workers=args.num_workers, pin_memory=bool(args.pin_memory))
    te = DataLoader(test_set, batch_sampler=BucketBatchSampler(test_set.length_buckets, bucket, shuffle=False),
                    num_workers=args.num_workers, pin_memory=bool(args.pin_memory))

    model = SeqRegressor(input_dim=16, hidden=args.hidden, num_layers=args.layers,
                         bidir=bool(args.bidir), dropout=args.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model = train_loop(model, opt, tr, va, epochs=args.epochs, device=DEVICE)
    mv = evaluate(model, va)
    mt = evaluate(model, te)

    run_group = os.getenv("RUN_GROUP", "base_rronly")
    tag = f"base_rronly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(RUNS_DIR) / run_group / tag
    save_run(str(run_dir), model, {"val": mv, "test": mt})

    durations = {"time_total_sec": round(time.perf_counter() - t0, 2),
                 "start_time": datetime.fromtimestamp(t0w).strftime("%Y-%m-%d %H:%M:%S"),
                 "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (run_dir / "durations.json").write_text(json.dumps(durations, indent=2, ensure_ascii=False))

    row = {
        "run_dir": str(run_dir), "run_group": run_group, "tag": tag, **durations,
        "val_corr_bestlag": mv.get("corr_bestlag"), "test_corr_bestlag": mt.get("corr_bestlag"),
        "test_rr_bpm_mae": mt.get("rr_bpm_mae"), "test_hit@±2bpm": mt.get("hit@±2bpm"),
    }
    append_csv(Path(RUNS_DIR) / "exp_results.csv", row)
    append_csv(run_dir / "metrics_row.csv", row)


if __name__ == "__main__":
    main()
