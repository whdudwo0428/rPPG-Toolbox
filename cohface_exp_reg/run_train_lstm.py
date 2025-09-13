# -*- coding: utf-8 -*-
import os, argparse, time, json, numpy as np, torch
from torch.utils.data import DataLoader
from .config import (CACHE_DIR, RUNS_DIR, DEVICE, FS_MODEL, LR, EPOCHS, BATCH,
                     RR_WIN_LIST, BUCKET_BS)
from .data import CohfaceSeqDataset
from .models import SeqRegressor
from .sampler import LengthBucketBatchSampler
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
    # 간단 분할: 앞 80% train, 다음 10% val, 마지막 10% test (세션 윈도우 단위)
    n = len(loaders)
    ntr = max(1, int(round(n*0.8)))
    nv  = max(1, int(round(n*0.1)))
    tr_loaders = loaders[:ntr]
    val_loader = loaders[ntr:ntr+nv]
    te_loader  = loaders[ntr+nv:]
    vdict = {"val": val_loader[0] if val_loader else tr_loaders[0]}
    tdict = {"test": te_loader[0] if te_loader else tr_loaders[-1]}

    model = SeqRegressor(input_dim=16, hidden=args.hidden, layers=args.layers,
                         cell='lstm', bidir=bool(args.bidir), dropout=args.dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = train_loop(model, optim, tr_loaders, vdict, epochs=args.epochs)
    # 최종 평가
    metrics = {}
    metrics["val"]  = vdict and evaluate(model, vdict["val"])
    metrics["test"] = tdict and evaluate(model, tdict["test"])
    print("[metrics]", json.dumps(metrics, indent=2, ensure_ascii=False))

    # 저장
    tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"lstm_rronly_{tag}")
    save_run(run_dir, model, metrics)

if __name__ == "__main__":
    main()
