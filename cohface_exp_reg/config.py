# -*- coding: utf-8 -*-
"""
하위호환 보장용 통합 config:
- 과거/현재 모듈이 요구하는 모든 심볼(이름)을 한 곳에서 정의
- 환경변수로 쉽게 오버라이드
- 멀티스케일 윈도우(HR/RR) + 추출/학습 공통 샘플레이트 + Mediapipe/장치 설정 포함
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 경로/디렉터리
# ──────────────────────────────────────────────────────────────────────────────
PROJ_ROOT = str(Path(__file__).resolve().parents[1])

# 데이터 루트 (환경변수 COHFACE_ROOT로 오버라이드)
DATA_ROOT = os.getenv("COHFACE_ROOT", f"{PROJ_ROOT}/dataset/cohface")

# 캐시/실행 결과
CACHE_DIR = os.getenv("CACHE_DIR", "cohface_exp_reg/cache_cohface_feats")
RUNS_DIR  = os.getenv("RUNS_DIR",  "cohface_exp_reg/runs")

# 자산(포즈 .task)
ASSETS_DIR = os.getenv("ASSETS_DIR", os.path.join(PROJ_ROOT, "cohface_exp_reg", "assets"))

# ──────────────────────────────────────────────────────────────────────────────
# 샘플레이트(추출/모델)
# ──────────────────────────────────────────────────────────────────────────────
def _to_int(s, default):
    try:
        return int(float(str(s)))
    except Exception:
        return int(default)

FS_RESAMP  = _to_int(os.getenv("FS_RESAMP",  "256"), 256)   # 공통 리샘플
FS_EXTRACT = _to_int(os.getenv("FS_EXTRACT", FS_RESAMP), FS_RESAMP)  # 추출 파이프라인
FS_MODEL   = _to_int(os.getenv("FS_MODEL",   FS_RESAMP), FS_RESAMP)  # 모델 입력

# ──────────────────────────────────────────────────────────────────────────────
# 대역/필터
# ──────────────────────────────────────────────────────────────────────────────
def _parse_pair(env_val, default_pair):
    if env_val:
        try:
            a, b = [float(x) for x in env_val.split(",")]
            return (a, b)
        except Exception:
            pass
    return tuple(default_pair)

RESP_BAND = _parse_pair(os.getenv("RESP_BAND"), (0.08, 0.60))  # RR (5–36 bpm)
HR_BAND   = _parse_pair(os.getenv("HR_BAND"),   (0.7, 3.0))    # HR (42–180 bpm)
BP_ORDER  = _to_int(os.getenv("BP_ORDER", "4"), 4)

# ──────────────────────────────────────────────────────────────────────────────
# 윈도우/스트라이드 (단일/멀티스케일 하위호환)
# ──────────────────────────────────────────────────────────────────────────────
def _parse_list(env_val, default_list):
    if not env_val:
        return list(default_list)
    try:
        return [float(x.strip()) for x in env_val.split(",") if x.strip() != ""]
    except Exception:
        return list(default_list)

# (과거 단일 윈도우용 – 필요 시 여전히 사용 가능)
WIN_SEC    = float(os.getenv("WIN_SEC", "8.0"))
STRIDE_SEC = float(os.getenv("STRIDE_SEC", "2.0"))

# (기본 멀티스케일)
HR_WIN_LIST = _parse_list(os.getenv("HR_WIN_LIST"), [8.0, 16.0])
RR_WIN_LIST = _parse_list(os.getenv("RR_WIN_LIST"), [32.0, 64.0])

# stride 정책: FIXED_STRIDE가 존재하면 우선, 없으면 STRIDE_FRAC(윈도우 비율)
STRIDE_FRAC  = float(os.getenv("STRIDE_FRAC", "0.25"))
_fix = os.getenv("FIXED_STRIDE", "").strip()
FIXED_STRIDE = float(_fix) if _fix not in ("", None) else None

# 정렬(전역 래그) 한계
LAG_MAX_S        = float(os.getenv("LAG_MAX_S", "0.50"))
GLOBAL_LAG_CLIP  = LAG_MAX_S  # alias (하위호환)

# ──────────────────────────────────────────────────────────────────────────────
# 학습 기본값
# ──────────────────────────────────────────────────────────────────────────────
SPLIT_SEED = _to_int(os.getenv("SPLIT_SEED", "42"), 42)
SEED       = _to_int(os.getenv("SEED", str(SPLIT_SEED)), SPLIT_SEED)

LR       = float(os.getenv("LR", "1e-3"))
EPOCHS   = _to_int(os.getenv("EPOCHS", "50"), 50)
BATCH    = _to_int(os.getenv("BATCH", "64"), 64)
PATIENCE = _to_int(os.getenv("PATIENCE", "6"), 6)

# ──────────────────────────────────────────────────────────────────────────────
# Mediapipe / Tasks / 장치
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_TASK   = os.path.join(ASSETS_DIR, "pose_landmarker_full.task")
MP_TASK_PATH   = os.getenv("MP_TASK_PATH", DEFAULT_TASK)
POSE_TASK_PATH = MP_TASK_PATH   # pose_backend에서 이 이름을 사용할 수 있어 alias 제공

MEDIAPIPE_USE_GPU    = os.getenv("MEDIAPIPE_USE_GPU", "1") not in ("0", "false", "False", "no", "NO")
MEDIAPIPE_GL_BACKEND = os.getenv("MEDIAPIPE_GL_BACKEND", "egl")

try:
    import torch
    CUDA_INDEX = _to_int(os.getenv("CUDA_INDEX", "0"), 0)   # 3060 Ti = 0 가정
    DEVICE     = f"cuda:{CUDA_INDEX}" if torch.cuda.is_available() else "cpu"
except Exception:
    CUDA_INDEX = 0
    DEVICE     = "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# 하위호환 보강(과거 코드가 다른 이름을 썼어도 안전)
# ──────────────────────────────────────────────────────────────────────────────
# 이미 동일 이름으로 위에 정의되어 있으므로 추가 alias는 필요치 않지만,
# 혹시 import * 로 덮어쓰기를 겪는 경우를 대비해 존재 보장만 체크합니다.
assert DATA_ROOT is not None
assert CACHE_DIR is not None
assert RUNS_DIR  is not None
assert FS_RESAMP and FS_EXTRACT and FS_MODEL
assert RESP_BAND and HR_BAND
assert isinstance(HR_WIN_LIST, list) and isinstance(RR_WIN_LIST, list)
