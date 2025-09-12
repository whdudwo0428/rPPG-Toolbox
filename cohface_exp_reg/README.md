# COHFACE Respiration & Heart Rate Regression (cohface\_exp\_reg)

> **의도(환경/GPU 정책)**
>
> * **학습은 무조건 3060 Ti**(`cuda:0`)에 붙도록.
> * **추출은** Mediapipe **Tasks .task**가 있으면 **GPU**, 없으면 **자동 CPU 폴백**.
> * **T400은 디스플레이**가 쓰고 있어도 상관 없음. `CUDA_VISIBLE_DEVICES=0`이면 연산은 3060 Ti로만 간다.

---

## 무엇을 하나요?

COHFACE 전체 데이터에서 **어깨·코 기반 변위(dY, dD, dW)** 를 추출하고, **밴드패스 + 정렬(z-score/GCC-PHAT)** 처리된 시퀀스로 \*\*호흡(RR)\*\*과 **심박(HR)** **동시 회귀(멀티헤드)** 를 학습/평가합니다.
**멀티스케일 윈도우**(기본: HR=8·16 s / RR=32·64 s)로 샘플을 구성하고, **헤드별 마스크**로 손실·지표를 분리 계산합니다. GPU가 있으면 **학습은 3060 Ti**로, 추출은 **Tasks 모델 유무에 따라 GPU/CPU**로 자동 선택합니다.

---

## 폴더 구조

```
cohface_exp_reg/
  config.py
  utils.py
  pose_backend.py
  preprocess.py
  data.py
  models.py
  train.py
  run_extract_all.py
  run_train_lstm.py
  run_train_gru.py
  assets/pose_landmarker_full.task     # ← 직접 다운로드
  cache_cohface_feats/                 # ← 자동 생성
  runs/                                # ← 자동 생성
```

---

## 요구사항

* Python 3.10
* PyTorch CUDA (학습 가속)
* Mediapipe (`pip install mediapipe`)
* 데이터(예시 경로):

  ```
  /home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface/<subject>/<session>/
    ├─ data.mkv
    └─ data.hdf5
  ```

---

## 0) 한 번만: 포즈 모델(.task) 다운로드

```bash
mkdir -p cohface_exp_reg/assets
wget -O cohface_exp_reg/assets/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

---

## 1) 환경 변수 (세션마다)

```bash
# 3060 Ti 고정
export CUDA_VISIBLE_DEVICES=0
export CUDA_INDEX=0

# Mediapipe GPU (Tasks 우선)
export MEDIAPIPE_GL_BACKEND=egl
export MEDIAPIPE_USE_GPU=1
export MP_TASK_PATH="$PWD/cohface_exp_reg/assets/pose_landmarker_full.task"

# (선택) 데이터/캐시 위치 오버라이드
export COHFACE_ROOT="/home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface"
export CACHE_DIR="cohface_exp_reg/cache_cohface_feats"
export RUNS_DIR="cohface_exp_reg/runs"

# (선택) 멀티스케일/stride 오버라이드
export HR_WIN_LIST="8,16"      # HR 윈도우(초)
export RR_WIN_LIST="32,64"     # RR 윈도우(초)
export STRIDE_FRAC="0.25"      # 각 윈도우의 25%로 슬라이딩
# export FIXED_STRIDE="2.0"    # 고정 stride(초)로 쓰고 싶을 때

# 메모리 단편화 완화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> 참고: 단일 윈도우 사용이 필요하면 `WIN_SEC`/`STRIDE_SEC` 인자나 환경변수를 그대로 사용할 수 있습니다(하위호환).

---

## 2) 특징 추출/전처리 캐시

```bash
python cohface_exp_reg/run_extract_all.py \
  --root /home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface \
  --subjects 1-40 --sessions 0-3 \
  --out cohface_exp_reg/cache_cohface_feats \
  --fs 256 --resp_lo 0.08 --resp_hi 0.60
```

* 로그 예시:

  ```
  [extract] DATA_ROOT=/.../dataset/cohface
  [extract] MP_TASK_PATH=/.../assets/pose_landmarker_full.task  exists=True
  [extract] MEDIAPIPE_USE_GPU=True
  [pose] tasks.PoseLandmarker (GPU) 사용: /.../pose_landmarker_full.task
  ...
  ```

* 캐시 파일은 아래 경로에 생성됩니다:

  ```
  cohface_exp_reg/cache_cohface_feats/s<subject>_k<session>.npz(.json)
  ```

---

## 3) 학습 (LSTM / GRU)

### LSTM

```bash
python cohface_exp_reg/run_train_lstm.py \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --lr 1e-3 --cuda 0 \
  --hidden 128 --layers 2 --bidir 1 --dropout 0.1 \
  --hr_wins 8,16 --rr_wins 32,64 --stride_frac 0.25 \
  --bucket_bs "16384:2,8192:4,4096:16,2048:32"
```

### GRU

```bash
python cohface_exp_reg/run_train_gru.py \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --batch_size 64 --lr 1e-3 --cuda 0 \
  --hidden 256 --layers 3 --bidir 0 --dropout 0.0 \
  --hr_wins 8,16 --rr_wins 32,64 --stride_frac 0.25
```

* 장치 로그 예:

  ```
  [device] Using cuda:0 (NVIDIA GeForce RTX 3060 Ti)
  ```

* 출력:

  ```
  cohface_exp_reg/runs/<model>_<timestamp>/
    best_model.pt
    best.json
    metrics.json
    train_log.csv (선택)
    plots/ (선택)
  ```

---

## 내부 동작 요약

* **추출**: 포즈(코/좌우 어깨) → `dY/dD/dW` → 리샘플(256 Hz) → 밴드패스(기본 RR: 0.08–0.60 Hz, HR: 0.7–3.0 Hz) → z-score →
  GCC-PHAT 전역 래그(±0.5 s 제한)로 GT 정렬 → 캐시 저장.

* **학습 입력**: `[w_rr,y_rr,d_rr,c_rr, w_hr,y_hr,d_hr,c_hr]` 총 **8채널**.
  여기서 `c_*`는 `(dW+dY+dD)/3`의 합성 채널.

* **멀티스케일/멀티헤드**:

  * RR 윈도우(기본 32, 64 s)로 생성된 샘플은 **RR 헤드만** 손실/지표에 반영(`mask_rr=1, mask_hr=0`).
  * HR 윈도우(기본 8, 16 s)로 생성된 샘플은 **HR 헤드만** 반영(`mask_rr=0, mask_hr=1`).
  * 서로 다른 길이가 **같은 배치**에 섞여도 **패딩/마스크 collate**로 안전 처리.

* **손실/지표**:

  * `loss = MSE_RR + 0.5×MSE_HR + 0.2×(1 - corr_RR)`
  * RR/HR 각각 마스크된 MSE와 상관을 계산.
  * RR bpm(PSD 피크) 추정은 RR 유효 마스크 구간에서만 계산.

---

## 자주 바꾸는 설정 (`config.py` 발췌)

```python
DATA_ROOT = "/home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface"

FS_RESAMP = 256
RESP_BAND = (0.08, 0.60)   # RR (5–36 bpm)
HR_BAND   = (0.7, 3.0)     # HR (42–180 bpm)
LAG_MAX_S = 0.50

# 하위호환(단일 윈도우)
WIN_SEC, STRIDE_SEC = 8.0, 2.0

# 멀티스케일(기본값)
HR_WIN_LIST = [8.0, 16.0]
RR_WIN_LIST = [32.0, 64.0]
STRIDE_FRAC = 0.25    # 윈도우 길이의 25%
# FIXED_STRIDE = None # (초)로 고정 stride를 쓰고 싶을 때 설정

SPLIT_SEED = 42
```

> 주의: 환경변수(`HR_WIN_LIST`, `RR_WIN_LIST`, `STRIDE_FRAC`, `FIXED_STRIDE`)로도 오버라이드 가능합니다.
> 단일 윈도우 실험은 `--win_sec/--stride_sec` 인자로 진행 가능합니다.

---

## 트러블슈팅

* **로그에 `solutions.pose (CPU) 사용`**
  → `.task` 경로가 잘못되었거나 `MP_TASK_PATH` 미설정.

  ```bash
  echo $MP_TASK_PATH
  ls -l "$MP_TASK_PATH"
  ```

* **T400로 잡힘**
  → 학습 실행 전에 항상 `export CUDA_VISIBLE_DEVICES=0`로 고정.

* **OOM(메모리 부족)**
  → `--batch_size` 감소, `--stride_frac` ↑(예: 0.5), HR/RR 윈도우 개수 축소, `--hidden`/`--layers` 감소, `num_workers` 2–4로 조절.

---

## PyCharm 런 설정(요약)

* **Script**: `cohface_exp_reg/run_extract_all.py`
  **Parameters**:

  ```
  --root /home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface \
  --subjects 1-10 --sessions 0-3 \
  --out cohface_exp_reg/cache_cohface_feats \
  --fs 256 --resp_lo 0.08 --resp_hi 0.60
  ```

  **Working directory**: `/home/gongjae/PycharmProjects/rPPG-Toolbox`
  **Env**:

  ```
  CUDA_VISIBLE_DEVICES=0; MEDIAPIPE_GL_BACKEND=egl; MEDIAPIPE_USE_GPU=1;
  MP_TASK_PATH=/home/gongjae/PycharmProjects/rPPG-Toolbox/cohface_exp_reg/assets/pose_landmarker_full.task;
  PYTHONUNBUFFERED=1
  ```

* **Script**: `cohface_exp_reg/run_train_lstm.py` (또는 `run_train_gru.py`)
  **Parameters(예)**:

  ```
  --cache cohface_exp_reg/cache_cohface_feats --epochs 30 --batch_size 64 \
  --hr_wins 8,16 --rr_wins 32,64 --stride_frac 0.25 --cuda 0
  ```

  **Working directory**: 프로젝트 루트
  **Env**: `CUDA_VISIBLE_DEVICES=0; PYTHONUNBUFFERED=1`
