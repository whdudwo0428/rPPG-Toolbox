# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (CACHE_DIR, RUNS_DIR, DEVICE, LR, EPOCHS, BUCKET_BS)
from .data import CohfaceSeqDataset
from .models import SeqRegressor
from .train import make_batch, evaluate, train_loop, save_run


def parse_bucket(bs_str):
    mp = {}
    for tok in bs_str.split(","):
        L, b = tok.split(":")
        mp[int(L)] = int(b)
    return mp

def build_loaders(ds, batch=64):
    # 윈도우를 길이별로 묶어 DataLoader 리스트를 만든다.
    windows = list(ds.iter_windows())
    # 길이별로 그룹화
    bylen = {}
    for (i,a,b,T) in windows:
        bylen.setdefault(T, []).append((i,a,b,T))
    loaders = []
    for T, group in bylen.items():
        # 같은 길이끼리 한 로더로 (batch는 bucket 지정 없으면 기본)
        # bucket map을 일관성 있게 사용하려면 config의 BUCKET_BS를 파싱
        loaders.append(DataLoader(group, batch_size=parse_bucket(BUCKET_BS).get(T, batch),
                                  shuffle=True, collate_fn=lambda idxs: make_batch(ds, idxs)))
    return loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default=CACHE_DIR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bidir", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    print(f"[device] {DEVICE}")
    ds = CohfaceSeqDataset(args.cache)
    loaders = build_loaders(ds, batch=64)

    # 80/10/10 split (길이 버킷 그룹 단위)
    n = len(loaders)
    ntr = max(1, int(round(n*0.8)))
    nv  = max(1, int(round(n*0.1)))
    tr_loaders = loaders[:ntr]
    val_loader = loaders[ntr:ntr+nv]
    te_loader  = loaders[ntr+nv:]

    # ── 변경 ①: Early-Stop은 기존대로 "첫 번째 val 그룹"을 사용
    vdict = {"val": val_loader[0] if val_loader else tr_loaders[0]}

    model = SeqRegressor(input_dim=16, hidden=args.hidden, layers=args.layers,
                         cell='lstm', bidir=bool(args.bidir), dropout=args.dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = train_loop(model, optim, tr_loaders, vdict, epochs=args.epochs)

    # ── 변경 ②: 최종 리포트는 모든 val/test 그룹 평가 후 "평균"으로 집계
    def _avg_metrics(loaders):
        if not loaders:
            return {}
        outs = [evaluate(model, ld) for ld in loaders]
        keys = outs[0].keys()
        return {k: float(np.mean([o[k] for o in outs])) for k in keys}

    metrics = {}
    metrics["val"]  = _avg_metrics(val_loader) if val_loader else _avg_metrics(tr_loaders)
    metrics["test"] = _avg_metrics(te_loader)  if te_loader  else _avg_metrics([tr_loaders[-1]])
    print("[metrics]", json.dumps(metrics, indent=2, ensure_ascii=False))

    # 저장
    tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"lstm_rronly_{tag}")
    save_run(run_dir, model, metrics)

if __name__ == "__main__":
    main()