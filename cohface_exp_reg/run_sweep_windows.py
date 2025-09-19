# -*- coding: utf-8 -*-
"""
V3.0 — 윈도우/스트라이드 전수 스윕 실행기 (단일/2윈도우/3윈도우)
- Window:  {10, 15, 20, 30, 40} (초)
- Stride:  {0.10, 0.20, 0.25, 0.50} (비율)
- 2윈도우:  짧은{10,15,20} × 긴{30,40}
- 3윈도우:  (짧은{10,15}) + (중간=20 고정) + (긴{30,40})

각 조합마다 run_train_lstm.py 호출 → 결과는 개별 run 디렉토리 + runs/exp_results/ 에 누적
- exp_results.csv  : long 포맷(모든 설정/메트릭 평탄화)
- summary.csv      : wide 요약(핵심 지표만)
- <run>.json       : 1런 요약 JSON
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# ── 경로 기본값 ───────────────────────────────────────────────────────────
PKG_ROOT = Path(__file__).resolve().parent
RUNS = Path(os.getenv("RUNS_DIR", str(PKG_ROOT / "runs")))
EXP_DIR = RUNS / "exp_results"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ── 스윕 대상 ─────────────────────────────────────────────────────────────
WIN_ALL = [10, 15, 20, 30, 40]
STRIDE_FR = [0.10, 0.20, 0.25, 0.50]

SHORT_FOR_PAIR = [10, 15, 20]
LONG_FOR_PAIR = [30, 40]

SHORT_FOR_TRIPLE = [10, 15]
MID_FIXED = 20
LONG_FOR_TRIPLE = [30, 40]

# 윈도우→(길이 샘플수, 권장 배치) 매핑 (fs=256 Hz)
BUCKET_MAP = {10: (2560, 12), 15: (3840, 10), 20: (5120, 8), 30: (7680, 5), 40: (10240, 4)}


# ── 유틸 ───────────────────────────────────────────────────────────────────
def _bucket_bs_string(wins: List[int]) -> str:
    return ",".join(f"{BUCKET_MAP[w][0]}:{BUCKET_MAP[w][1]}" for w in wins)


def _as_str_list(x: List[Any]) -> str:
    return ",".join(str(v) for v in x)


def _build_env(wins: List[int], strides: List[float]) -> Dict[str, str]:
    env = os.environ.copy()
    env["RR_WIN_LIST"] = _as_str_list(wins)
    env["STRIDE_FRAC_LIST"] = _as_str_list(strides)  # 윈도우별 비율 지정
    # 빈/충돌 방지: 절대초는 사용 안 함
    env.pop("FIXED_STRIDE", None)
    env.pop("FIXED_STRIDE_LIST", None)
    # STRIDE_FRAC는 건드리지 말 것(빈 문자열 덮어쓰기 방지)
    env["BUCKET_BS"] = _bucket_bs_string(wins)
    return env


def _latest_run_dir(group: str) -> Path | None:
    """runs/ 아래에서 group 프리픽스가 들어간 최신 디렉토리 탐색."""
    if not RUNS.exists():
        return None
    cands = [p for p in RUNS.iterdir() if p.is_dir() and p.name.startswith(group)]
    if not cands:
        # 프리픽스가 없는 구현(기존 스크립트)이면 전체 중 최신 반환(추정)
        cands = [p for p in RUNS.iterdir() if p.is_dir()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _flatten(prefix: str, obj: Dict[str, Any], out: Dict[str, Any]):
    for k, v in obj.items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(kk, v, out)
        else:
            out[kk] = v


def _try_load_metrics_json(run_dir: Path) -> Dict[str, Any]:
    cands = []
    if (run_dir / "metrics.json").exists():
        cands.append(run_dir / "metrics.json")
    cands.extend(run_dir.rglob("metrics.json"))
    if not cands:
        return {}
    mj = max(cands, key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(mj.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_csv(csv_path: Path, row: Dict[str, Any], key_field: str):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        import csv
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    # 중복키 갱신
    new_rows, found = [], False
    for r in rows:
        if r.get(key_field, "") == str(row.get(key_field, "")):
            new_rows.append({**r, **{k: str(v) for k, v in row.items()}})
            found = True
        else:
            new_rows.append(r)
    if not found:
        new_rows.append({k: str(v) for k, v in row.items()})
    # 필드 통일
    fieldnames = sorted({k for rr in new_rows for k in rr.keys()})
    import csv
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rr in new_rows:
            w.writerow({k: rr.get(k, "") for k in fieldnames})


def _run_one(wins: List[int], strides: List[float], run_group: str):
    tag = f"{run_group}__ws[{_as_str_list(wins)}]__sf[{_as_str_list([int(s * 100) for s in strides])}]"
    print(f"[RUN] {tag}")
    py = sys.executable
    cmd = [
        py, "-m", "cohface_exp_reg.run_train_lstm",
        "--cache", "cohface_exp_reg/cache_cohface_feats",
        "--epochs", os.getenv("EPOCHS", "50"),
        "--lr", os.getenv("LR", "5e-4"),
        "--hidden", os.getenv("HIDDEN", "128"),
        "--layers", os.getenv("LAYERS", "2"),
        "--bidir", os.getenv("BIDIR", "1"),
        "--dropout", os.getenv("DROPOUT", "0.1"),
        "--bucket_bs", _bucket_bs_string(wins),
        "--num_workers", os.getenv("NUM_WORKERS", "12"),
        "--pin_memory", os.getenv("PIN_MEMORY", "1"),
    ]
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, env=_build_env(wins, strides))
    elapsed = time.perf_counter() - t0
    print(f"[DONE] rc={rc} | {elapsed:.1f}s | {tag}")

    run_group = os.getenv("RUN_GROUP", run_group)
    run_dir = _latest_run_dir(run_group) or Path("")
    metrics = _try_load_metrics_json(run_dir)

    base: Dict[str, Any] = {
        "key": str(run_dir),
        "run_dir": str(run_dir),
        "run_group": run_group,
        "windows": _as_str_list(wins),
        "strides_frac": _as_str_list(strides),
        "bucket_bs": _bucket_bs_string(wins),
        "elapsed_sec": round(elapsed, 3),
        "window_pad_mode": os.getenv("WINDOW_PAD_MODE", "edge"),
        "window_include_tail": os.getenv("WINDOW_INCLUDE_TAIL", "1"),
    }
    flat: Dict[str, Any] = {}
    if metrics:
        _flatten("", metrics, flat)
    row = {**base, **flat}

    # 저장
    json_path = EXP_DIR / f"{run_dir.name or tag}.json"
    _save_json(json_path, row)
    _append_csv(EXP_DIR / "exp_results.csv", row, key_field="key")

    # summary(wide)
    summary = {
        "key": row["key"],
        "run_dir": row["run_dir"],
        "windows": row["windows"],
        "strides_frac": row["strides_frac"],
        "elapsed_sec": row["elapsed_sec"],
    }
    for k in [
        "val.corr", "val.corr_bestlag", "val.rr_bpm_mae", "val.hit_at_±2bpm",
        "test.corr", "test.corr_bestlag", "test.rr_bpm_mae", "test.hit_at_±2bpm",
    ]:
        if k in row:
            summary[k] = row[k]
    _append_csv(EXP_DIR / "summary.csv", summary, key_field="key")


def main():
    group = os.getenv("RUN_GROUP", "lstm_rronly")
    print(f"[RUN_GROUP] {group}")
    print(f"[OUTPUT] runs dir = {RUNS}")
    print(f"[EXP_RESULTS] {EXP_DIR}")

    # 1) 단일 윈도우
    for w in WIN_ALL:
        for s in STRIDE_FR:
            _run_one([w], [s], group)

    # 2) 2윈도우 (짧은{10,15,20} × 긴{30,40})
    for s_win in SHORT_FOR_PAIR:
        for l_win in LONG_FOR_PAIR:
            for s_s in STRIDE_FR:
                for l_s in STRIDE_FR:
                    _run_one([s_win, l_win], [s_s, l_s], group)

    # 3) 3윈도우 (짧은{10,15} + 20 고정 + 긴{30,40})
    for sh in SHORT_FOR_TRIPLE:
        for lg in LONG_FOR_TRIPLE:
            for s0 in STRIDE_FR:
                for s1 in STRIDE_FR:
                    for s2 in STRIDE_FR:
                        _run_one([sh, MID_FIXED, lg], [s0, s1, s2], group)

    print(f"[ALL DONE] runs dir: {RUNS}")


if __name__ == "__main__":
    main()
