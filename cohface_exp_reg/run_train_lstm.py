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
    if not s: return None
    try:
        return [float(x) for x in str(s).split(",")]
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)  # (검증/테스트용 기본값)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--cuda", type=int, default=None, help="-1=cpu, 0/1=GPU index")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bidir", type=int, default=1, help="1=BiRNN, 0=Uni")
    ap.add_argument("--dropout", type=float, default=0.0)
    # 멀티스케일 오버라이드
    ap.add_argument("--hr_wins", default=None, help="예: 8,16")
    ap.add_argument("--rr_wins", default=None, help="예: 32,64")
    ap.add_argument("--stride_frac", type=float, default=None)
    ap.add_argument("--fixed_stride", type=float, default=None)
    # 버킷별 배치 사이즈
    ap.add_argument("--bucket_bs", default="16384:2,8192:4,4096:16,2048:32")
    args = ap.parse_args()

    # ===== config 심볼 안전하게 해석 =====
    FS_MODEL = getattr(config, "FS_MODEL", getattr(config, "FS_RESAMP", 256))
    SEED     = getattr(config, "SEED", getattr(config, "SPLIT_SEED", 42))

    # 동적 오버라이드
    if args.cache:      config.CACHE_DIR = args.cache
    if args.epochs:     config.EPOCHS = args.epochs
    if args.batch_size: config.BATCH = args.batch_size
    if args.lr:         config.LR = args.lr

    # 멀티스케일 키 없으면 기본 주입
    if not hasattr(config, "HR_WIN_LIST"): config.HR_WIN_LIST = [8.0, 16.0]
    if not hasattr(config, "RR_WIN_LIST"): config.RR_WIN_LIST = [32.0, 64.0]
    if not hasattr(config, "STRIDE_FRAC"): config.STRIDE_FRAC = 0.25
    if not hasattr(config, "FIXED_STRIDE"): config.FIXED_STRIDE = None

    device = getattr(config, "DEVICE", "cpu")
    if args.cuda is not None:
        device = "cpu" if args.cuda < 0 else f"cuda:{args.cuda}"
        config.DEVICE = device  # train/eval 내부에서 사용
    print(f"[cfg] device={device}  cache_dir={getattr(config,'CACHE_DIR','cohface_exp_reg/cache_cohface_feats')}")
    print(f"[cfg] HR_WIN_LIST={config.HR_WIN_LIST}  RR_WIN_LIST={config.RR_WIN_LIST}  stride_frac={config.STRIDE_FRAC}  fixed_stride={config.FIXED_STRIDE}")
    print(f"[cfg] bucket_bs={args.bucket_bs}")

    set_seed(SEED)

    entries = load_all_entries()
    subs = [int(e["subject"]) for e in entries]
    trS, vaS, teS = subject_split(subs, ratios=(0.7,0.15,0.15), seed=SEED)

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

    bs_map = parse_bucket_bs(args.bucket_bs)
    pin = device.startswith("cuda")

    dl_tr = DataLoader(ds_tr, batch_sampler=LengthBucketBatchSampler(ds_tr.lengths, bs_map, shuffle=True),
                       num_workers=4, pin_memory=pin, collate_fn=pad_collate)
    dl_va = DataLoader(ds_va, batch_sampler=LengthBucketBatchSampler(ds_va.lengths, bs_map, shuffle=False),
                       num_workers=4, pin_memory=pin, collate_fn=pad_collate)
    dl_te = DataLoader(ds_te, batch_sampler=LengthBucketBatchSampler(ds_te.lengths, bs_map, shuffle=False),
                       num_workers=4, pin_memory=pin, collate_fn=pad_collate)

    model = SeqRegressor(cell="LSTM", hidden=args.hidden, layers=args.layers,
                         bidir=bool(args.bidir), dropout=args.dropout)

    tag = (
        f"lstm_h{args.hidden}x{args.layers}_bi{int(bool(args.bidir))}_do{args.dropout:g}_"
        f"mscale_hr{'-'.join(str(int(w)) for w in config.HR_WIN_LIST)}_"
        f"rr{'-'.join(str(int(w)) for w in config.RR_WIN_LIST)}_"
        f"bkt{args.bucket_bs.replace(',','-').replace(':','x')}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )

    out_dir, best = train_loop(model, dl_tr, dl_va, tag)

    ckpt = os.path.join(out_dir, "best_model.pt")
    state = torch.load(ckpt, map_location="cpu")
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
