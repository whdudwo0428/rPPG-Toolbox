# 실행은 모듈 모드 필수:  python -m cohface_exp_reg.run_train_lstm  [args...]
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from . import config
from .data import load_all_entries, CohfaceSeqDataset, pad_collate
from .models import SeqRegressor
from .sampler import LengthBucketBatchSampler, parse_bucket_bs
from .train import train_loop, evaluate
from .utils import set_seed, subject_split


def parse_float_list(s):
    if s is None:
        return None
    try:
        vals = [float(x) for x in str(s).split(",") if str(x).strip() != ""]
        return vals if len(vals) > 0 else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="cache dir (npz) override")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None, help="fallback batch_size (when no bucket_bs)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--cuda", type=int, default=None, help="-1 for CPU, else CUDA index")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bidir", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hr_wins", default=None, help="comma sep, e.g. 8,16")
    ap.add_argument("--rr_wins", default=None, help="comma sep, e.g. 32,64")
    ap.add_argument("--stride_frac", type=float, default=None)
    ap.add_argument("--fixed_stride", type=float, default=None)
    ap.add_argument("--bucket_bs", default="16384:2,8192:4,4096:16,2048:32",
                    help='length:batch pairs, empty("") to disable bucketing')
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # ---- config overrides (CLI 우선) ----
    FS_MODEL = getattr(config, "FS_MODEL", getattr(config, "FS_RESAMP", 256))
    SEED = args.seed if args.seed is not None else getattr(config, "SEED", getattr(config, "SPLIT_SEED", 42))

    if args.cache is not None:
        config.CACHE_DIR = args.cache
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH = args.batch_size
    if args.lr is not None:
        config.LR = args.lr

    # 멀티스케일 파라미터
    if not hasattr(config, "HR_WIN_LIST"):
        config.HR_WIN_LIST = [8.0, 16.0]
    if not hasattr(config, "RR_WIN_LIST"):
        config.RR_WIN_LIST = [32.0, 64.0]
    if not hasattr(config, "STRIDE_FRAC"):
        config.STRIDE_FRAC = 0.25
    if not hasattr(config, "FIXED_STRIDE"):
        config.FIXED_STRIDE = None

    cli_hr = parse_float_list(args.hr_wins)
    cli_rr = parse_float_list(args.rr_wins)
    if cli_hr is not None:
        config.HR_WIN_LIST = cli_hr
    if cli_rr is not None:
        config.RR_WIN_LIST = cli_rr
    if args.stride_frac is not None:
        config.STRIDE_FRAC = float(args.stride_frac)
    if args.fixed_stride is not None:
        config.FIXED_STRIDE = float(args.fixed_stride)

    # 디바이스 설정
    device = getattr(config, "DEVICE", "cpu")
    if args.cuda is not None:
        device = "cpu" if args.cuda < 0 else f"cuda:{args.cuda}"
        config.DEVICE = device

    print(f"[cfg] device={device}  cache_dir={getattr(config,'CACHE_DIR','cohface_exp_reg/cache_cohface_feats')}")
    print(f"[cfg] HR_WIN_LIST={config.HR_WIN_LIST}  RR_WIN_LIST={config.RR_WIN_LIST}  stride_frac={config.STRIDE_FRAC}  fixed_stride={config.FIXED_STRIDE}")
    print(f"[cfg] bucket_bs={args.bucket_bs}")

    set_seed(SEED)

    # ---- load cache entries & subject split ----
    entries = load_all_entries()
    subs = [int(e["subject"]) for e in entries]
    trS, vaS, teS = subject_split(subs, ratios=(0.7, 0.15, 0.15), seed=SEED)

    train_entries = [E for E in entries if int(E["subject"]) in trS]
    val_entries   = [E for E in entries if int(E["subject"]) in vaS]
    test_entries  = [E for E in entries if int(E["subject"]) in teS]

    ds_tr = CohfaceSeqDataset(train_entries, "train",
                              rr_win_list=config.RR_WIN_LIST, hr_win_list=config.HR_WIN_LIST,
                              stride_frac=config.STRIDE_FRAC, fixed_stride=config.FIXED_STRIDE)
    ds_va = CohfaceSeqDataset(val_entries, "val",
                              rr_win_list=config.RR_WIN_LIST, hr_win_list=config.HR_WIN_LIST,
                              stride_frac=config.STRIDE_FRAC, fixed_stride=config.FIXED_STRIDE)
    ds_te = CohfaceSeqDataset(test_entries, "test",
                              rr_win_list=config.RR_WIN_LIST, hr_win_list=config.HR_WIN_LIST,
                              stride_frac=config.STRIDE_FRAC, fixed_stride=config.FIXED_STRIDE)

    # ---- DataLoaders: bucket sampler or fixed batch size ----
    pin = device.startswith("cuda")
    use_bucket = isinstance(args.bucket_bs, str) and args.bucket_bs.strip() != ""
    if use_bucket:
        bs_map = parse_bucket_bs(args.bucket_bs)
        dl_tr = DataLoader(ds_tr, batch_sampler=LengthBucketBatchSampler(ds_tr.lengths, bs_map, shuffle=True),
                           num_workers=args.num_workers, pin_memory=pin, collate_fn=pad_collate)
        dl_va = DataLoader(ds_va, batch_sampler=LengthBucketBatchSampler(ds_va.lengths, bs_map, shuffle=False),
                           num_workers=args.num_workers, pin_memory=pin, collate_fn=pad_collate)
        dl_te = DataLoader(ds_te, batch_sampler=LengthBucketBatchSampler(ds_te.lengths, bs_map, shuffle=False),
                           num_workers=args.num_workers, pin_memory=pin, collate_fn=pad_collate)
    else:
        bs = int(getattr(config, "BATCH", 64))
        if args.batch_size is not None:
            bs = args.batch_size
        dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,
                           num_workers=args.num_workers, pin_memory=pin, collate_fn=pad_collate)
        dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False,
                           num_workers=args.num_workers, pin_memory=pin, collate_fn=pad_collate)
        dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False,
                           num_workers=args.num_workers, pin_memory=pin, collate_fn=pad_collate)

    # ---- Model ----
    model = SeqRegressor(cell="LSTM", hidden=args.hidden, layers=args.layers,
                         bidir=bool(args.bidir), dropout=args.dropout)

    # ---- Tag / run dir ----
    tag = (
        f"lstm_h{args.hidden}x{args.layers}_bi{int(bool(args.bidir))}_do{args.dropout:g}_"
        f"mscale_hr{'-'.join(str(int(w)) for w in config.HR_WIN_LIST)}_"
        f"rr{'-'.join(str(int(w)) for w in config.RR_WIN_LIST)}_"
        f"{'bkt'+args.bucket_bs.replace(',','-').replace(':','x') if use_bucket else 'bs'+str(getattr(config,'BATCH',args.batch_size or 64))}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )

    # ---- Train ----
    out_dir, best = train_loop(model, dl_tr, dl_va, tag)

    # ---- Reload best & evaluate ----
    ckpt = os.path.join(out_dir, "best_model.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    metrics = {
        "val":  evaluate(model, dl_va, FS_MODEL),
        "test": evaluate(model, dl_te, FS_MODEL),
        "best": best
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[saved]", os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
