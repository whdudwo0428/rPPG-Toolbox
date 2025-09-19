# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cohface_exp_reg.config import (CACHE_DIR, RUNS_DIR, DEVICE, LR, EPOCHS, BUCKET_BS)
from cohface_exp_reg.V2.data import CohfaceSeqDataset
from cohface_exp_reg.models import SeqRegressor
from cohface_exp_reg.train import make_batch, evaluate, train_loop, save_run


def parse_bucket(spec: str):
    mp = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        L, b = tok.split(":")
        mp[int(L)] = int(b)
    return mp


def build_loaders_by_session(ds, bucket_bs=None, num_workers=0, pin_memory=0, seed=42):
    rng = np.random.default_rng(seed)
    windows = list(ds.iter_windows())  # (session_i, a, b, T)

    # 세션별 묶기
    sess_to_wins = {}
    for (i, a, b, T) in windows:
        sess_to_wins.setdefault(i, []).append((i, a, b, T))

    sess_ids = sorted(sess_to_wins.keys())
    rng.shuffle(sess_ids)

    n = len(sess_ids)
    ntr = max(1, int(round(n * 0.8)))
    nv = max(1, int(round(n * 0.1)))
    tr_ids = set(sess_ids[:ntr])
    va_ids = set(sess_ids[ntr:ntr + nv])
    te_ids = set(sess_ids[ntr + nv:])

    def _mk_loaders(id_set):
        bylen = {}
        for i in id_set:
            for it in sess_to_wins[i]:
                T = it[-1]
                bylen.setdefault(T, []).append(it)

        bs_map = parse_bucket(bucket_bs)
        loaders = []
        for T, group in bylen.items():
            loaders.append(DataLoader(
                group,
                batch_size=bs_map.get(T, 64),
                shuffle=True,
                collate_fn=lambda idxs: make_batch(ds, idxs),
                num_workers=num_workers,
                pin_memory=bool(pin_memory),
                persistent_workers=bool(num_workers > 0),
                prefetch_factor=(4 if num_workers > 0 else None),
            ))
        return loaders

    return _mk_loaders(tr_ids), _mk_loaders(va_ids), _mk_loaders(te_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default=CACHE_DIR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--bidir", type=int, default=1)  # 양방향 권장
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bucket_bs", type=str, default=BUCKET_BS)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--pin_memory", type=int, default=1)
    args = ap.parse_args()

    ds = CohfaceSeqDataset(args.cache)

    # 세션 기반 분할 + 길이 버킷 로더
    tr_loaders, val_loaders, te_loaders = build_loaders_by_session(
        ds,
        bucket_bs=args.bucket_bs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        seed=42,
    )

    # Early-Stop: 모든 Val 로더 평균 corr_bestlag
    vdict = {f"val_{k}": ld for k, ld in enumerate(val_loaders)} or {"val_fallback": tr_loaders[0]}

    model = SeqRegressor(
        input_dim=16,
        hidden=args.hidden,
        layers=args.layers,
        cell='gru',
        bidir=bool(args.bidir),
        dropout=args.dropout
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model = train_loop(model, optimizer, tr_loaders, vdict, epochs=args.epochs)

    def _avg_metrics(loaders):
        if not loaders:
            return {}
        outs = [evaluate(model, ld) for ld in loaders]
        keys = outs[0].keys()
        return {k: float(np.mean([o[k] for o in outs])) for k in keys}

    metrics = {
        "val": _avg_metrics(val_loaders) if val_loaders else _avg_metrics(tr_loaders),
        "test": _avg_metrics(te_loaders) if te_loaders else _avg_metrics([tr_loaders[-1]]),
    }
    print("[metrics]", json.dumps(metrics, indent=2, ensure_ascii=False))

    tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"gru_rronly_{tag}")
    save_run(run_dir, model, metrics)


if __name__ == "__main__":
    main()