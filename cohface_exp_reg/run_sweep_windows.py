# -*- coding: utf-8 -*-
"""
윈도우/스트라이드 스윕 실행기
- 단일/2개/3개 윈도우 조합 × stride{0.10,0.20,0.25,0.50,0.80}
- 각 실험: 환경변수(RR_WIN_LIST, STRIDE_FRAC or STRIDE_FRAC_LIST, BUCKET_BS 등) 세팅 후
  run_train_lstm 모듈 실행 → runs/ 하위에 결과 저장
- 끝나면 summary.csv 생성(각 run의 val/test corr_bestlag, rr_bpm_mae 등)
"""
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

W_CAND = [10, 15, 20, 30]
S_CAND = [0.10, 0.20, 0.25, 0.50, 0.80]  # stride fraction
MAX_COMB = 3  # 1~3 윈도우 조합

PROJ = Path(__file__).resolve().parents[0]
RUNS = Path(os.getenv("RUNS_DIR", str(PROJ / "runs")))
CACHE = os.getenv("CACHE_DIR", str(PROJ / "cache_cohface_feats"))


def bucket_for_windows(ws):
    # 길이(샘플) -> 배치사이즈 매핑 문자열
    mp = {}
    for w in ws:
        T = int(round(w * 256))
        if T >= 10240:
            mp[T] = 4
        elif T >= 7680:
            mp[T] = 5
        elif T >= 5120:
            mp[T] = 8
        elif T >= 3840:
            mp[T] = 10
        else:
            mp[T] = 12
    items = sorted(mp.items())
    return ",".join([f"{k}:{v}" for k, v in items])


def run_one(wins, stride_frac):
    wins_str = ",".join(str(int(w)) for w in wins)
    bmap = bucket_for_windows(wins)
    tag = f"sweep_ws[{wins_str}]_sf[{int(stride_frac * 100)}]"
    env = os.environ.copy()
    env["RR_WIN_LIST"] = wins_str
    env["STRIDE_FRAC"] = str(stride_frac)
    env["FIXED_STRIDE_LIST"] = ""  # 비사용
    env["STRIDE_FRAC_LIST"] = ""  # 윈도우마다 다른 stride 쓰려면 "0.25,0.5" 등 지정
    env["WINDOW_PAD_MODE"] = env.get("WINDOW_PAD_MODE", "edge")
    env["WINDOW_INCLUDE_TAIL"] = env.get("WINDOW_INCLUDE_TAIL", "1")
    env["BUCKET_BS"] = bmap

    cmd = [
        sys.executable, "-m", "cohface_exp_reg.run_train_lstm",
        "--cache", CACHE,
        "--epochs", os.getenv("EPOCHS", "50"),
        "--lr", os.getenv("LR", "5e-4"),
        "--hidden", "128", "--layers", "2", "--bidir", "1", "--dropout", "0.1",
        "--bucket_bs", bmap,
        "--num_workers", os.getenv("NUM_WORKERS", "12"),
        "--pin_memory", "1",
    ]
    print("[run]", tag, "|", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def collect_summary():
    rows = []
    for pj in RUNS.glob("lstm_rronly_*"):
        m = pj / "metrics.json"
        if not m.exists(): continue
        try:
            js = json.loads(m.read_text())
            row = {
                "run_dir": str(pj),
                "val_corr_bestlag": js.get("val", {}).get("corr_bestlag"),
                "test_corr_bestlag": js.get("test", {}).get("corr_bestlag"),
                "test_rr_bpm_mae": js.get("test", {}).get("rr_bpm_mae"),
                "val_hit@±2bpm": js.get("val", {}).get("hit@±2bpm"),
                "test_hit@±2bpm": js.get("test", {}).get("hit@±2bpm"),
            }
            rows.append(row)
        except Exception:
            pass
    import csv
    out = RUNS / f"sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_dir"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print("[summary]", out)


def main():
    # 1-window + 2-window + 3-window 조합
    all_combs = []
    for k in range(1, MAX_COMB + 1):
        all_combs.extend(itertools.combinations(W_CAND, k))
    for wins in all_combs:
        for sf in S_CAND:
            run_one(wins, sf)
    collect_summary()


if __name__ == "__main__":
    main()
