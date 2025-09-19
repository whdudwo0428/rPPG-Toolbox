# -*- coding: utf-8 -*-
"""
V1.6 — Flexible windows/stride + padding policy
- RR_WIN_LIST: "10,15,20,30" 등 가변 윈도우(초)
- STRIDE_FRAC or STRIDE_FRAC_LIST: 전체 공통 또는 윈도우별(예: "0.25" 또는 "0.1,0.2,0.25")
- WINDOW_PAD_MODE: none|zero|edge|reflect  (테일 패딩)
- WINDOW_INCLUDE_TAIL: 1이면 마지막 불완전 윈도우를 패딩 포함해 생성
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
PROJ_ROOT = str(Path(__file__).resolve().parents[0])
DATA_ROOT = os.getenv("COHFACE_ROOT", "/home/youruser/dataset/cohface")
CACHE_DIR = os.getenv("CACHE_DIR", str(Path(PROJ_ROOT) / "cache_cohface_feats"))
RUNS_DIR  = os.getenv("RUNS_DIR",  str(Path(PROJ_ROOT) / "runs"))

ASSETS_DIR  = str(Path(PROJ_ROOT) / "assets")
MP_TASK_PATH = os.getenv("MP_TASK_PATH", str(Path(ASSETS_DIR) / "pose_landmarker_full.task"))

# ── Sampling ──────────────────────────────────────────────────────────────
FS_EXTRACT = int(float(os.getenv("FS_EXTRACT", "256")))
FS_RESAMP  = FS_EXTRACT
FS_MODEL   = FS_EXTRACT

# ── Bands ─────────────────────────────────────────────────────────────────
RESP_BAND = (
    float(os.getenv("RESP_LO", "0.08")),
    float(os.getenv("RESP_HI", "0.60")),
)

# ── Windows / stride (flexible) ───────────────────────────────────────────
def _parse_float_list(s: str):
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out

# 윈도우(초). 기본은 기존과 동일하게 "20,40" 유지
RR_WIN_LIST = _parse_float_list(os.getenv("RR_WIN_LIST", "20,40"))
if not RR_WIN_LIST:
    RR_WIN_LIST = [20.0, 40.0]

# stride: 공통 비율(0~1) 또는 윈도우별 리스트
STRIDE_FRAC       = float(os.getenv("STRIDE_FRAC", "0.25"))
STRIDE_FRAC_LIST  = _parse_float_list(os.getenv("STRIDE_FRAC_LIST", ""))  # 옵션

# 고정 stride(초) — 지정 시 비율 대신 절대초 적용; 리스트도 허용
FIXED_STRIDE      = os.getenv("FIXED_STRIDE", "")
FIXED_STRIDE_LIST = _parse_float_list(os.getenv("FIXED_STRIDE_LIST", ""))
if FIXED_STRIDE in ("", "None", "none"):
    FIXED_STRIDE = None
else:
    FIXED_STRIDE = float(FIXED_STRIDE)

# 테일 패딩 정책
WINDOW_PAD_MODE     = os.getenv("WINDOW_PAD_MODE", "none").lower()  # none|zero|edge|reflect
WINDOW_INCLUDE_TAIL = int(os.getenv("WINDOW_INCLUDE_TAIL", "1"))    # 1: 마지막 불완전 창 포함

# 256 Hz 기준 길이별 기본 배치(휴리스틱)
# 10s=2560, 15s=3840, 20s=5120, 30s=7680, 40s=10240
BUCKET_BS = os.getenv("BUCKET_BS", "2560:12,3840:10,5120:8,7680:5,10240:4")

# ── Loss / metrics ────────────────────────────────────────────────────────
LOSS_MODE    = os.getenv("LOSS_MODE", "corr").lower()
PHASE_LAMBDA = float(os.getenv("PHASE_LAMBDA", "0.3"))
PHASE_BETA   = float(os.getenv("PHASE_BETA", "8.0"))
LAG_MAX_S    = float(os.getenv("LAG_MAX_S", "2.0"))
BPM_MIN_PROM = float(os.getenv("BPM_MIN_PROM", "3.0"))
SNR_HIT_BPM  = int(os.getenv("SNR_HIT_BPM", "2"))

# Optional pre-align
ENABLE_PREALIGN  = int(os.getenv("ENABLE_PREALIGN", "0"))
PREALIGN_MAX_LAG = float(os.getenv("PREALIGN_MAX_LAG", "4.0"))

# Trend
W_TREND_FC = float(os.getenv("W_TREND_FC", "0.05"))

# ── Training ──────────────────────────────────────────────────────────────
LR       = float(os.getenv("LR", "1e-3"))
EPOCHS   = int(os.getenv("EPOCHS", "50"))
PATIENCE = int(os.getenv("PATIENCE", "6"))
SEED     = int(os.getenv("SPLIT_SEED", "42"))

# device
CUDA_INDEX = int(os.getenv("CUDA_INDEX", "0"))
try:
    import torch
    DEVICE = f"cuda:{CUDA_INDEX}" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"