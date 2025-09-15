# -*- coding: utf-8 -*-
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

    train_set = CohfaceSeqDataset(args.cache, subset='train')
    val_set = CohfaceSeqDataset(args.cache, subset='val')

    bucket = parse_bucket_bs(args.bucket_bs)
    train_loader = DataLoader(
        train_set,
        batch_sampler=BucketBatchSampler(train_set, bucket, shuffle=True),
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    model = SeqRegressor(
        in_dim=16, hidden=args.hidden, layers=args.layers,
        bidir=bool(args.bidir), dropout=args.dropout
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model = train_loop(model, opt, train_loader, val_loader, epochs=args.epochs, device=DEVICE)

    # final eval on val
    val_metrics = evaluate(model, val_loader)

    tag = f"lstm_rronly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, tag)
    save_run(run_dir, model, {"val": val_metrics})
    print("[metrics]", {"val": val_metrics})


if __name__ == '__main__':
    main()
