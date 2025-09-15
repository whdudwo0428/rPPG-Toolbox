# -*- coding: utf-8 -*-
"""
V1 config (RR-only):
- RR 전용 멀티스케일 윈도우(메인: 32,64s / 서브: 24,48,96s)
- 손실: RR_MSE + PHASE_LAMBDA*(1 - corr@soft-best-lag)
- 입력: 16채널(RR-only), HR 관련 항목 제거
- 디바이스: 3060 Ti 고정 의도 → CUDA_VISIBLE_DEVICES=0, DEVICE=cuda:0 (가용시)
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────────────────────
PROJ_ROOT = str(Path(__file__).resolve().parents[0])
DATA_ROOT = os.getenv("COHFACE_ROOT", "/dataset/cohface")
CACHE_DIR = os.getenv("CACHE_DIR", str(Path(PROJ_ROOT) / "cache_cohface_feats"))
RUNS_DIR  = os.getenv("RUNS_DIR",  str(Path(PROJ_ROOT) / "runs"))

# ──────────────────────────────────────────────────────────────────────────────
# 샘플레이트/주파수대역
# ──────────────────────────────────────────────────────────────────────────────
FS_EXTRACT = int(float(os.getenv("FS_EXTRACT", "256")))   # 추출(리샘플) 기준
FS_RESAMP  = FS_EXTRACT
FS_MODEL   = FS_EXTRACT

# RR 대역(Hz): 0.08–0.60 → 4.8–36 bpm
RESP_BAND = (
    float(os.getenv("RESP_LO", "0.08")),
    float(os.getenv("RESP_HI", "0.60")),
)

# ──────────────────────────────────────────────────────────────────────────────
# 윈도우/스트라이드 (RR-only)
# ──────────────────────────────────────────────────────────────────────────────
def _parse_floats(envv, default):
    if envv is None or len(envv.strip()) == 0:
        return default
    return [float(x) for x in envv.split(",")]

# 윈도우/스트라이드
RR_WIN_LIST = _parse_floats(os.getenv("RR_WIN_LIST", ""), [20.0, 40.0])
STRIDE_FRAC = float(os.getenv("STRIDE_FRAC", "0.25"))
FIXED_STRIDE = os.getenv("FIXED_STRIDE", "")
FIXED_STRIDE = None if FIXED_STRIDE in ("", "None", "none") else float(FIXED_STRIDE)

# 256 Hz 기준 길이: 20=5120, 40=10240 → 긴 시퀀스는 작은 배치
BUCKET_BS = os.getenv("BUCKET_BS", "10240:4,5120:8")

# 손실 진폭 패널티 계수(필요 시 튜닝)
AMP_LAMBDA = float(os.getenv("AMP_LAMBDA", "5e-2"))

# 느린 컨텍스트 컷오프(Hz): RR(≥0.08)보다 확 낮게
W_TREND_FC = float(os.getenv("W_TREND_FC", "0.05"))

# ──────────────────────────────────────────────────────────────────────────────
# 손실/지표 하이퍼
# ──────────────────────────────────────────────────────────────────────────────
PHASE_LAMBDA = float(os.getenv("PHASE_LAMBDA", "0.2"))  # 권장 0.1~0.3
PHASE_BETA   = float(os.getenv("PHASE_BETA", "8.0"))
LAG_MAX_S    = float(os.getenv("LAG_MAX_S", "0.5"))
BPM_MIN_PROM = float(os.getenv("BPM_MIN_PROM", "3.0")) #RR bpm 검출 민감도(낮을수록 피크 더 쉽게 검출)
SNR_HIT_BPM  = int(os.getenv("SNR_HIT_BPM", "2"))  # hit@±2 bpm

# ──────────────────────────────────────────────────────────────────────────────
# 학습 하이퍼
# ──────────────────────────────────────────────────────────────────────────────
LR      = float(os.getenv("LR", "1e-3"))
EPOCHS  = int(os.getenv("EPOCHS", "50"))
BATCH   = int(os.getenv("BATCH", "64"))
PATIENCE= int(os.getenv("PATIENCE", "10"))
SEED    = int(os.getenv("SPLIT_SEED", "42"))

# ──────────────────────────────────────────────────────────────────────────────
# Mediapipe/장치
# ──────────────────────────────────────────────────────────────────────────────
MP_TASK_PATH = os.getenv("MP_TASK_PATH", str(Path(PROJ_ROOT)/"assets"/"pose_landmarker_full.task"))
MEDIAPIPE_GL_BACKEND = os.getenv("MEDIAPIPE_GL_BACKEND", "egl")
MEDIAPIPE_USE_GPU = os.getenv("MEDIAPIPE_USE_GPU", "1")  # Tasks면 GPU, 없으면 CPU 폴백

CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
CUDA_INDEX = int(os.getenv("CUDA_INDEX", "0"))
try:
    import torch
    DEVICE = f"cuda:{CUDA_INDEX}" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# 기타
# ──────────────────────────────────────────────────────────────────────────────
SPLIT_SEED = SEED

# 보장
assert isinstance(RR_WIN_LIST, list) and len(RR_WIN_LIST) >= 2
