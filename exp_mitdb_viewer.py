# -*- coding: utf-8 -*-
"""
MIT-BIH ECG 뷰어 (주석 시각화 + 라벨 카운트 + 라벨별 줌인)
사용법:
  python exp_mitdb_viewer.py            # 기본: record=100, 전체 플롯 + 통계
  python exp_mitdb_viewer.py 102 V      # 예: 102번 레코드의 첫 V(PVC) 주변 줌인
"""

# ---- 0) matplotlib 백엔드 폴백(최상단) ----
import os
for be in ("TkAgg", "Qt5Agg", "Agg"):  # TkAgg를 우선순위로
    try:
        os.environ["MPLBACKEND"] = be
        import matplotlib
        matplotlib.use(be, force=True)
        import matplotlib.pyplot as plt  # noqa
        print(f"[matplotlib] backend = {be}")
        break
    except Exception as e:
        print(f"[matplotlib] backend {be} failed:", e)

import sys
from pathlib import Path
from collections import Counter
import numpy as np
import wfdb


# ---- 1) 경로 설정: 프로젝트/dataset/MIT-BIH-A 심볼릭 링크 기준 ----
ROOT = Path(__file__).resolve().parents[0]
MITDB = ROOT / "dataset" / "MIT-BIH-A"  # 심볼릭 링크가 여기로 연결돼 있어야 함

# ---- 2) 유틸: 레코드 로드 ----
def load_record(record_id: str):
    rec_path = MITDB / record_id
    # record(물리단위 mV) + annotation(atr)
    rec = wfdb.rdrecord(str(rec_path))
    ann = wfdb.rdann(str(rec_path), "atr")
    return rec, ann


# ---- 3) 채널 인덱스 선택(MLII, Vx 우선) ----
def pick_channels(rec):
    names = [s.upper() for s in rec.sig_name]
    # 우선순위: MLII(또는 II), 그 다음 V5/V2/V1 …
    primary_candidates = ["MLII", "II", "V5", "V2", "V1"]
    secondary_candidates = ["V5", "V2", "V1", "III", "AVF", "AVL", "AVR"]

    def find_first(cands):
        for c in cands:
            if c in names:
                return names.index(c)
        return 0  # fallback

    ch1 = find_first(primary_candidates)
    # 두 번째는 ch1과 다른 채널로
    ch2 = None
    for c in secondary_candidates:
        if c in names and names.index(c) != ch1:
            ch2 = names.index(c)
            break
    if ch2 is None:
        ch2 = 1 if rec.n_sig > 1 else ch1
    return ch1, ch2


# ---- 4) 전체 구간 플롯(주석 별표 표시) ----
def plot_full(rec, ann, save_path=None, title_prefix="MITDB"):
    fs = rec.fs
    n = rec.sig_len
    t = np.arange(n) / fs

    ch1, ch2 = pick_channels(rec)
    sig1 = rec.p_signal[:, ch1]
    sig2 = rec.p_signal[:, ch2]
    name1 = rec.sig_name[ch1]
    name2 = rec.sig_name[ch2]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t, sig1, linewidth=0.7)
    axes[0].set_ylabel(f"{name1}\nmV")
    axes[0].set_title(f"{title_prefix} {rec.record_name}")

    axes[1].plot(t, sig2, linewidth=0.7)
    axes[1].set_ylabel(f"{name2}\nmV")
    axes[1].set_xlabel("time/second")

    # 주석 위치 표시(윗 채널 위에 빨간 별)
    ann_t = ann.sample / fs
    # 별이 파형과 겹치지 않도록 약간 위로 올린 기준선
    y_base = np.nanmedian(sig1) + 0.8 * np.nanstd(sig1)
    axes[0].plot(ann_t, np.full_like(ann_t, y_base), "r*", markersize=4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"[saved] {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ---- 5) 라벨 카운트 & 샘플 프린트 ----
def print_annotation_stats(ann, rec_fs):
    cnt = Counter(ann.symbol)
    total = sum(cnt.values())
    print(f"[annotations] total={total}")
    for k, v in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k:>3} : {v}")
    # 앞쪽 10개 예시
    print("\n[first 10] time(s)  symbol")
    for s, sym in list(zip(ann.sample, ann.symbol))[:10]:
        print(f"  {s/rec_fs:8.3f}   {sym}")


# ---- 6) 특정 라벨 주변 줌인 플롯 ----
def plot_zoom_around_label(rec, ann, target_symbol="V", win_sec=8.0, save_path=None):
    fs = rec.fs
    ch1, ch2 = pick_channels(rec)
    sig1 = rec.p_signal[:, ch1]
    sig2 = rec.p_signal[:, ch2]
    name1 = rec.sig_name[ch1]
    name2 = rec.sig_name[ch2]

    # 첫 번째 target_symbol 위치 찾기
    idx = next((i for i, s in enumerate(ann.symbol) if s == target_symbol), None)
    if idx is None:
        print(f"[zoom] '{target_symbol}' 라벨이 없습니다.")
        return
    center = ann.sample[idx]
    half = int(win_sec * fs / 2)
    a = max(0, center - half)
    b = min(rec.sig_len, center + half)

    t = np.arange(a, b) / fs

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, sig1[a:b], linewidth=1.0)
    axes[0].axvline(center / fs, color="r", linestyle="--", linewidth=1)
    axes[0].set_ylabel(f"{name1}\nmV")
    axes[0].set_title(f"Zoom around '{target_symbol}' @ {center/fs:.2f}s")

    axes[1].plot(t, sig2[a:b], linewidth=1.0)
    axes[1].axvline(center / fs, color="r", linestyle="--", linewidth=1)
    axes[1].set_ylabel(f"{name2}\nmV")
    axes[1].set_xlabel("time/second")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180)
        print(f"[saved] {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ---- 7) main ----
def main():
    # 인자: record_id [label]
    record_id = sys.argv[1] if len(sys.argv) >= 2 else "100"
    label = sys.argv[2] if len(sys.argv) >= 3 else None

    if not MITDB.exists():
        raise FileNotFoundError(f"MITDB not found: {MITDB} (심볼릭 링크 확인)")

    rec, ann = load_record(record_id)
    print(rec.sig_name, rec.fs, len(ann.sample))
    print_annotation_stats(ann, rec.fs)

    # 전체 플롯
    plot_full(
        rec,
        ann,
        save_path=ROOT / f"mitdb_{record_id}_full.png",
        title_prefix="MITDB",
    )

    # 특정 라벨 줌인(요청 시)
    if label:
        plot_zoom_around_label(
            rec,
            ann,
            target_symbol=label,
            win_sec=8.0,
            save_path=ROOT / f"mitdb_{record_id}_zoom_{label}.png",
        )


if __name__ == "__main__":
    main()