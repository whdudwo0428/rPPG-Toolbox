# -*- coding: utf-8 -*-
"""
GRU Runner — LSTM 스타일로 통일 (V1.5)
- CohfaceSeqDataset(subset='train'/'val')
- train/val 모두 BucketBatchSampler 사용
- 모델은 현 repo의 SeqRegressor(LSTM 백본) 인터페이스 유지
"""
import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from .config import RUNS_DIR, DEVICE, LR
from .data import CohfaceSeqDataset
from .models import SeqRegressor
from .sampler import BucketBatchSampler
from .train import train_loop, evaluate, save_run


def parse_bucket_bs(s: str):
    """
    "10240:4,5120:8" -> {10240:4, 5120:8}
    """
    out = {}
    for kv in s.split(','):
        k, v = kv.split(':')
        out[int(k)] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--bidir', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--bucket_bs', type=str, default='10240:4,5120:8')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--pin_memory', type=int, default=1)
    args = ap.parse_args()

    # Dataset
    train_set = CohfaceSeqDataset(args.cache, subset='train')
    val_set = CohfaceSeqDataset(args.cache, subset='val')

    # Bucket sampler config
    bucket = parse_bucket_bs(args.bucket_bs)

    train_loader = DataLoader(
        train_set,
        batch_sampler=BucketBatchSampler(train_set, bucket, shuffle=True),
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=BucketBatchSampler(val_set, bucket, shuffle=False),
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    # Model (LSTM backbone; interface parity)
    model = SeqRegressor(
        in_dim=16,
        hidden=args.hidden,
        layers=args.layers,
        bidir=bool(args.bidir),
        dropout=args.dropout,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model = train_loop(model, opt, train_loader, val_loader, epochs=args.epochs, device=DEVICE)

    val_metrics = evaluate(model, val_loader)

    tag = f"gru_rronly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, tag)
    save_run(run_dir, model, {"val": val_metrics})
    print("[metrics]", {"val": val_metrics})


if __name__ == '__main__':
    main()