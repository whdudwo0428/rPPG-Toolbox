# -*- coding: utf-8 -*-
"""
V3.0 — Flexible windows/stride + padding + full training/loss knobs

- RR_WIN_LIST: "10,15,20,30,40" 등 가변 윈도우(초)
- STRIDE_FRAC 또는 STRIDE_FRAC_LIST: 전체 공통/윈도우별 비율(빈 문자열/None 허용)
- FIXED_STRIDE(_LIST): 절대초 스트라이드(비율 대신)
- WINDOW_PAD_MODE: none|zero|edge|reflect (테일 패딩)
- WINDOW_INCLUDE_TAIL: 마지막 불완전 창 포함 여부
- RESP_BAND, LOSS_MODE, PHASE_* , LAG_MAX_S, ENABLE_PREALIGN, MP_TASK_PATH 등
  기존 코드가 참조하던 심볼 복구/유지
"""

import os
from pathlib import Path

CONFIG_VERSION = "V3.0"

# ── Paths ─────────────────────────────────────────────────────────────────
PROJ_ROOT = str(Path(__file__).resolve().parents[0])
DATA_ROOT = os.getenv("COHFACE_ROOT", "/home/youruser/dataset/cohface")
CACHE_DIR = os.getenv("CACHE_DIR", str(Path(PROJ_ROOT) / "cache_cohface_feats"))
RUNS_DIR = os.getenv("RUNS_DIR", str(Path(PROJ_ROOT) / "runs"))

ASSETS_DIR = str(Path(PROJ_ROOT) / "assets")
MP_TASK_PATH = os.getenv("MP_TASK_PATH", str(Path(ASSETS_DIR) / "pose_landmarker_full.task"))

# ── Sampling ──────────────────────────────────────────────────────────────
FS_EXTRACT = int(float(os.getenv("FS_EXTRACT", "256")))
FS_RESAMP = FS_EXTRACT
FS_MODEL = FS_EXTRACT  # 모델 입력 샘플링레이트(기존 코드 호환)

# ── Bands ─────────────────────────────────────────────────────────────────
RESP_BAND = (
    float(os.getenv("RESP_LO", "0.08")),
    float(os.getenv("RESP_HI", "0.60")),
)


# ── Helpers ───────────────────────────────────────────────────────────────
def _parse_float_list(s: str):
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _opt_float(env_name: str, default=None):
    """빈 문자열/None/'none'을 허용하는 float 파서"""
    raw = os.getenv(env_name, None if default is None else str(default))
    if raw is None:
        return default
    raw_l = str(raw).strip().lower()
    if raw_l in ("", "none", "null"):
        return default
    try:
        return float(raw)
    except Exception:
        return default


# ── Windows / stride (flexible) ───────────────────────────────────────────
# 윈도우(초) — 기본은 README의 V3 프리셋과 동일(예: "15,30")
RR_WIN_LIST = _parse_float_list(os.getenv("RR_WIN_LIST", "15,30"))
if not RR_WIN_LIST:
    RR_WIN_LIST = [15.0, 30.0]

# stride: 공통 비율(0~1) 또는 윈도우별 리스트; 빈 문자열이면 None
STRIDE_FRAC = _opt_float("STRIDE_FRAC", default=None)  # 스윕에서 "" 들어와도 안전
STRIDE_FRAC_LIST = _parse_float_list(os.getenv("STRIDE_FRAC_LIST", "")) or None

# 고정 stride(초) — 지정 시 비율 대신 절대초 적용; 리스트도 허용
FIXED_STRIDE = _opt_float("FIXED_STRIDE", default=None)
FIXED_STRIDE_LIST = _parse_float_list(os.getenv("FIXED_STRIDE_LIST", "")) or None

# 테일 패딩 정책
WINDOW_PAD_MODE = os.getenv("WINDOW_PAD_MODE", "edge").lower()  # none|zero|edge|reflect
WINDOW_INCLUDE_TAIL = int(os.getenv("WINDOW_INCLUDE_TAIL", "1"))  # 1: 마지막 불완전 창 포함

# 256 Hz 기준 길이별 기본 배치(휴리스틱)
# 10s=2560, 15s=3840, 20s=5120, 30s=7680, 40s=10240
BUCKET_BS = os.getenv("BUCKET_BS", "2560:12,3840:10,5120:8,7680:5,10240:4")

# ── Loss / metrics ────────────────────────────────────────────────────────
LOSS_MODE = os.getenv("LOSS_MODE", "corr").lower()  # 'mse'|'corr' 등
PHASE_LAMBDA = float(os.getenv("PHASE_LAMBDA", "0.3"))
PHASE_BETA = float(os.getenv("PHASE_BETA", "10.0"))
LAG_MAX_S = float(os.getenv("LAG_MAX_S", "1.5"))

# BPM/PSD 옵션(README와 일치)
BPM_MIN_PROM = float(os.getenv("BPM_MIN_PROM", "3.0"))
BPM_FALLBACK_ARGMAX = int(os.getenv("BPM_FALLBACK_ARGMAX", "1"))
BPM_SUBBIN_QUAD = int(os.getenv("BPM_SUBBIN_QUAD", "1"))
BPM_NFFT_UP = int(os.getenv("BPM_NFFT_UP", "4"))
SNR_HIT_BPM = int(os.getenv("SNR_HIT_BPM", "2"))

# Optional pre-align (전역 사전정렬)
ENABLE_PREALIGN = int(os.getenv("ENABLE_PREALIGN", "0"))
PREALIGN_MAX_LAG = float(os.getenv("PREALIGN_MAX_LAG", "4.0"))

# 느린 트렌드 컷오프(게이팅/드리프트 억제)
W_TREND_FC = float(os.getenv("W_TREND_FC", "0.05"))

# ── Training ──────────────────────────────────────────────────────────────
LR = float(os.getenv("LR", "5e-4"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
PATIENCE = int(os.getenv("PATIENCE", "6"))
SEED = int(os.getenv("SPLIT_SEED", "42"))

# 디바이스
CUDA_INDEX = int(os.getenv("CUDA_INDEX", "0"))
try:
    import torch

    DEVICE = f"cuda:{CUDA_INDEX}" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# 러닝 그룹 접두(스윕 태그 구성용)
RUN_GROUP = os.getenv("RUN_GROUP", "lstm_rronly")
