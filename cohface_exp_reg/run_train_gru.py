# -*- coding: utf-8 -*-
import os, argparse, time, json, numpy as np, torch
from torch.utils.data import DataLoader
from .config import (CACHE_DIR, RUNS_DIR, DEVICE, FS_MODEL, LR, EPOCHS, BATCH,
                     RR_WIN_LIST, BUCKET_BS)
from .data import CohfaceSeqDataset
from .models import SeqRegressor
from .train import make_batch, evaluate, train_loop, save_run

def parse_bucket(bs_str):
    mp = {}
    for tok in bs_str.split(","):
        L, b = tok.split(":")
        mp[int(L)] = int(b)
    return mp

def build_loaders(ds, batch=64, bucket_bs=BUCKET_BS):
    # 윈도우를 길이별로 묶어 DataLoader 리스트를 만든다.
    windows = list(ds.iter_windows())
    bylen = {}
    for (i,a,b,T) in windows:
        bylen.setdefault(T, []).append((i,a,b,T))
    loaders = []
    bs_map = parse_bucket(bucket_bs)
    for T, group in bylen.items():
        loaders.append(DataLoader(group, batch_size=bs_map.get(T, batch),
                                  shuffle=True, collate_fn=lambda idxs: make_batch(ds, idxs)))
    return loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default=CACHE_DIR)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--bidir", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--bucket_bs", type=str, default=BUCKET_BS)
    args = ap.parse_args()

    if args.epochs is not None:   from . import config as _C; _C.EPOCHS = args.epochs
    if args.lr is not None:       from . import config as _C; _C.LR = args.lr

    print(f"[device] {DEVICE}")
    ds = CohfaceSeqDataset(args.cache)
    loaders = build_loaders(ds, batch=BATCH, bucket_bs=args.bucket_bs)
    # 80/10/10 split (by loader groups)
    n = len(loaders)
    ntr = max(1, int(round(n*0.8)))
    nv  = max(1, int(round(n*0.1)))
    tr_loaders = loaders[:ntr]
    val_loader = loaders[ntr:ntr+nv]
    te_loader  = loaders[ntr+nv:]

    vdict = {"val": val_loader[0] if val_loader else tr_loaders[0]}
    tdict = {"test": te_loader[0] if te_loader else tr_loaders[-1]}

    model = SeqRegressor(input_dim=16, hidden=args.hidden, layers=args.layers,
                         cell='gru', bidir=bool(args.bidir), dropout=args.dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr or LR)

    model = train_loop(model, optim, tr_loaders, vdict, epochs=(args.epochs or EPOCHS))
    # 최종 평가
    metrics = {}
    metrics["val"]  = vdict and evaluate(model, vdict["val"])
    metrics["test"] = tdict and evaluate(model, tdict["test"])
    print("[metrics]", json.dumps(metrics, indent=2, ensure_ascii=False))

    # 저장
    tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"gru_rronly_{tag}")
    save_run(run_dir, model, metrics)

if __name__ == "__main__":
    main()
