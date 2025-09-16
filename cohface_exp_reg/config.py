# -*- coding: utf-8 -*-
"""
V1.5 config (RR-only, fixed windows 20s/40s)
- Windows: 20, 40 (seconds)
- Stride: win * STRIDE_FRAC (default 0.25)
- Loss: selectable via LOSS_MODE
  * 'phase' : SI-MSE + PHASE_LAMBDA * spectral phase loss
  * 'corr'  : MSE on z-score + PHASE_LAMBDA * (1 - corr@soft-best-lag)
- Optional pre-alignment: ENABLE_PREALIGN=1 to estimate global sign/lag per session
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJ_ROOT = str(Path(__file__).resolve().parents[0])
DATA_ROOT = os.getenv("COHFACE_ROOT", "/home/youruser/dataset/cohface")
CACHE_DIR = os.getenv("CACHE_DIR", str(Path(PROJ_ROOT) / "cache_cohface_feats"))
RUNS_DIR = os.getenv("RUNS_DIR", str(Path(PROJ_ROOT) / "runs"))

# Mediapipe Tasks model (.task) path (pose_backend.py에서 import)
ASSETS_DIR = str(Path(PROJ_ROOT) / "assets")
MP_TASK_PATH = os.getenv("MP_TASK_PATH", str(Path(ASSETS_DIR) / "pose_landmarker_full.task"))

# ──────────────────────────────────────────────────────────────────────────────
# Sampling / bands
# ──────────────────────────────────────────────────────────────────────────────
FS_EXTRACT = int(float(os.getenv("FS_EXTRACT", "256")))
FS_RESAMP = FS_EXTRACT
FS_MODEL = FS_EXTRACT

# RR band (Hz) ≈ 4.8–36 bpm
RESP_BAND = (
    float(os.getenv("RESP_LO", "0.08")),
    float(os.getenv("RESP_HI", "0.60")),
)

# ──────────────────────────────────────────────────────────────────────────────
# Windows / stride (RR-only)
# ──────────────────────────────────────────────────────────────────────────────
RR_WIN_LIST = [20.0, 40.0]  # fixed
STRIDE_FRAC = float(os.getenv("STRIDE_FRAC", "0.25"))
FIXED_STRIDE = os.getenv("FIXED_STRIDE", "")
FIXED_STRIDE = None if FIXED_STRIDE in ("", "None", "none") else float(FIXED_STRIDE)

# 256 Hz length: 20s→5120, 40s→10240
BUCKET_BS = os.getenv("BUCKET_BS", "10240:4,5120:8")

# ──────────────────────────────────────────────────────────────────────────────
# Loss / metrics
# ──────────────────────────────────────────────────────────────────────────────
LOSS_MODE = os.getenv("LOSS_MODE", "corr").lower()  # default: 'corr'
PHASE_LAMBDA = float(os.getenv("PHASE_LAMBDA", "0.3"))
PHASE_BETA = float(os.getenv("PHASE_BETA", "8.0"))
LAG_MAX_S = float(os.getenv("LAG_MAX_S", "2.0"))  # ±2.0s default; can widen
BPM_MIN_PROM = float(os.getenv("BPM_MIN_PROM", "3.0"))
SNR_HIT_BPM = int(os.getenv("SNR_HIT_BPM", "2"))  # hit@±2 bpm

# Optional session-level pre-alignment (sign + lag)
ENABLE_PREALIGN = int(os.getenv("ENABLE_PREALIGN", "0"))  # 0=off, 1=on
PREALIGN_MAX_LAG = float(os.getenv("PREALIGN_MAX_LAG", "4.0"))  # seconds

# Slow trend LPF cutoff used in data normalization
W_TREND_FC = float(os.getenv("W_TREND_FC", "0.05"))

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
LR = float(os.getenv("LR", "1e-3"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
PATIENCE = int(os.getenv("PATIENCE", "6"))
SEED = int(os.getenv("SPLIT_SEED", "42"))

# device
CUDA_INDEX = int(os.getenv("CUDA_INDEX", "0"))
try:
    import torch

    DEVICE = f"cuda:{CUDA_INDEX}" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

assert isinstance(RR_WIN_LIST, list) and len(RR_WIN_LIST) == 2 and RR_WIN_LIST == [20.0, 40.0]
