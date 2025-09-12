
import os, time, json
import torch
from torch.utils.data import DataLoader

import config  # 동적 오버라이드 위해 모듈 자체 import
from config import DEVICE as CFG_DEVICE, CACHE_DIR, FS_MODEL, BATCH, SEED
from data import load_all_entries, CohfaceSeqDataset, pad_collate
from models import SeqRegressor
from train import train_loop, evaluate
from utils import set_seed, subject_split

import argparse

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
    ap.add_argument("--batch_size", type=int, default=None)
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
    args = ap.parse_args()

    # ===== 동적 오버라이드 (config 값 갱신) =====
    if args.cache:      config.CACHE_DIR = args.cache
    if args.epochs:     config.EPOCHS = args.epochs
    if args.batch_size: config.BATCH = args.batch_size
    if args.lr:         config.LR = args.lr
    if args.hr_wins:    config.HR_WIN_LIST = parse_float_list(args.hr_wins) or config.HR_WIN_LIST
    if args.rr_wins:    config.RR_WIN_LIST = parse_float_list(args.rr_wins) or config.RR_WIN_LIST
    if args.stride_frac is not None: config.STRIDE_FRAC = float(args.stride_frac)
    if args.fixed_stride is not None: config.FIXED_STRIDE = float(args.fixed_stride)

    device = CFG_DEVICE
    if args.cuda is not None:
        device = "cpu" if args.cuda < 0 else f"cuda:{args.cuda}"
        config.DEVICE = device  # train/eval 내부 일관성을 위해 동기화
    print(f"[cfg] device={device}  cache_dir={config.CACHE_DIR}")
    print(f"[cfg] HR_WIN_LIST={config.HR_WIN_LIST}  RR_WIN_LIST={config.RR_WIN_LIST}  stride_frac={config.STRIDE_FRAC}  fixed_stride={config.FIXED_STRIDE}")

    set_seed(SEED)

    entries = load_all_entries()  # config.CACHE_DIR 사용
    subs = [int(e["subject"]) for e in entries]
    trS, vaS, teS = subject_split(subs, ratios=(0.7,0.15,0.15), seed=SEED)

    train_entries = [E for E in entries if int(E["subject"]) in trS]
    val_entries   = [E for E in entries if int(E["subject"]) in vaS]
    test_entries  = [E for E in entries if int(E["subject"]) in teS]

    ds_tr = CohfaceSeqDataset(train_entries, "train")
    ds_va = CohfaceSeqDataset(val_entries, "val")
    ds_te = CohfaceSeqDataset(test_entries, "test")

    pin = device.startswith("cuda")
    dl_tr = DataLoader(ds_tr, batch_size=config.BATCH, shuffle=True,  num_workers=4, pin_memory=pin, drop_last=True, collate_fn=pad_collate)
    dl_va = DataLoader(ds_va, batch_size=config.BATCH, shuffle=False, num_workers=4, pin_memory=pin, collate_fn=pad_collate)
    dl_te = DataLoader(ds_te, batch_size=config.BATCH, shuffle=False, num_workers=4, pin_memory=pin, collate_fn=pad_collate)

    model = SeqRegressor(cell="GRU", hidden=args.hidden, layers=args.layers,
                         bidir = bool(args.bidir), dropout = args.dropout)
    tag = f"gru_h{args.hidden}x{args.layers}_bi{int(bool(args.bidir))}_do{args.dropout:g}_"           f"mscale_hr{'-'.join(str(int(w)) for w in config.HR_WIN_LIST)}_"           f"rr{'-'.join(str(int(w)) for w in config.RR_WIN_LIST)}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir, best = train_loop(model, dl_tr, dl_va, tag)

    # load best & evaluate
    ckpt = os.path.join(out_dir, "best_model.pt")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    metrics = {
        "val": evaluate(model, dl_va, FS_MODEL),
        "test": evaluate(model, dl_te, FS_MODEL),
        "best": best
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[saved]", os.path.join(out_dir, "metrics.json"))

if __name__ == "__main__":
    main()
