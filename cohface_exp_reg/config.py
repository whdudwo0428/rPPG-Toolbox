
import os
from pathlib import Path

# === 프로젝트 루트 ===
PROJ_ROOT = str(Path(__file__).resolve().parents[1])

# === 데이터 루트 (심볼릭 링크 기본). 환경변수 COHFACE_ROOT로 override 가능 ===
DATA_ROOT = os.getenv(
    "COHFACE_ROOT",
    "/home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface"
)

# === 출력 디렉토리 (프로젝트 루트 기준 상대경로) ===
CACHE_DIR = os.getenv("CACHE_DIR", "cohface_exp_reg/cache_cohface_feats")
RUNS_DIR  = os.getenv("RUNS_DIR",  "cohface_exp_reg/runs")

# === 신호 처리 공통 ===
RESP_BAND = (0.08, 0.60)  # Hz (RR 5–36 bpm)
HR_BAND   = (0.7, 3.0)    # Hz (HR 42–180 bpm; 필요시 사용)
BP_ORDER  = 4             # Butterworth bandpass order

FS_RESAMP = 256           # 기준 샘플레이트
FS_EXTRACT = FS_RESAMP    # 호환 별칭
FS_MODEL   = FS_RESAMP    # 모델 입력 샘플레이트

LAG_MAX_S       = 0.50    # 전역 래그 최대 절대값(초)
GLOBAL_LAG_CLIP = LAG_MAX_S  # 호환 별칭

# === 윈도우 설정 ===
# 단일값과의 호환을 유지하면서 멀티스케일을 기본 제공
def _parse_float_list(env, default):
    if env is None or env.strip() == "":
        return list(default)
    toks = [t.strip() for t in env.split(",")]
    try:
        vals = [float(t) for t in toks]
        return [v for v in vals if v > 0]
    except Exception:
        return list(default)

WIN_SEC    = float(os.getenv("WIN_SEC", "8.0"))          # 과거 호환
STRIDE_SEC = float(os.getenv("STRIDE_SEC", "2.0"))       # 과거 호환

# 멀티스케일(기본: HR={8,16}s, RR={32,64}s). 쉼표로 오버라이드 가능: HR_WIN_LIST="6,10,14"
HR_WIN_LIST = _parse_float_list(os.getenv("HR_WIN_LIST"), [8.0, 16.0])
RR_WIN_LIST = _parse_float_list(os.getenv("RR_WIN_LIST"), [32.0, 64.0])
# stride는 각 윈도우의 일정 비율을 기본 사용(25%), 환경변수로 고정값도 허용
STRIDE_FRAC = float(os.getenv("STRIDE_FRAC", "0.25"))
FIXED_STRIDE = os.getenv("FIXED_STRIDE", "").strip()
FIXED_STRIDE = float(FIXED_STRIDE) if FIXED_STRIDE not in ("", None) else None

# === 학습 하이퍼파라미터(기본값; run_train_*에서 동적 오버라이드 가능) ===
LR       = float(os.getenv("LR", "1e-3"))
EPOCHS   = int(os.getenv("EPOCHS", "50"))
BATCH    = int(os.getenv("BATCH", "64"))
PATIENCE = int(os.getenv("PATIENCE", "6"))
SPLIT_SEED = 42
SEED = SPLIT_SEED  # 호환 별칭

# === Mediapipe Tasks 모델 경로/옵션 ===
ASSETS_DIR      = os.path.join(PROJ_ROOT, "cohface_exp_reg", "assets")
DEFAULT_TASK    = os.path.join(ASSETS_DIR, "pose_landmarker_full.task")
POSE_TASK_PATH  = os.getenv("MP_TASK_PATH", DEFAULT_TASK)

MEDIAPIPE_USE_GPU    = os.getenv("MEDIAPIPE_USE_GPU", "1") not in ("0", "false", "False")
MEDIAPIPE_GL_BACKEND = os.getenv("MEDIAPIPE_GL_BACKEND", "egl")

# === Torch / CUDA 장치 기본 ===
try:
    import torch
    CUDA_INDEX_DEFAULT = int(os.getenv("CUDA_INDEX", "0"))  # 3060 Ti를 0번으로
    DEVICE = f"cuda:{CUDA_INDEX_DEFAULT}" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
