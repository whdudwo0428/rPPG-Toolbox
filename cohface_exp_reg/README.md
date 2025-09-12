# 환경 변수 세팅 (3060 Ti 우선)

> **의도**
>
> * 학습은 **무조건 3060 Ti**(`cuda:0`)에 붙도록.
> * 추출은 `.task`가 있으면 GPU 사용, 없으면 자동 CPU 폴백.
> * **T400은 디스플레이**가 쓰고 있어도 상관 없음. `CUDA_VISIBLE_DEVICES=0`이면 연산은 3060 Ti로만 간다.

---

## COHFACE Respiration Regression (cohface\_exp\_reg)

### 무엇을 하나요?

COHFACE 전체 데이터에서 **어깨·코 기반 변위(dY, dD, dW)** 를 추출하고, **밴드패스+정렬(z-score/GCC-PHAT)** 처리된 시퀀스로 **호흡(RR/HR 대역) 회귀**를 학습/평가합니다.
GPU가 있으면 **학습은 3060 Ti**로, 추출은 **Tasks 모델이 있을 때 GPU**로 자동 이동합니다(없으면 CPU 폴백).

---

### 폴더 구조

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

### 요구사항

* Python 3.10
* PyTorch CUDA (학습 가속)
* Mediapipe (`pip install mediapipe`)
* 데이터:

  ```
  /home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface/<subject>/<session>/
    ├─ data.mkv
    └─ data.hdf5
  ```

---

### 0) 한 번만: 포즈 모델 다운로드

```bash
mkdir -p cohface_exp_reg/assets
wget -O cohface_exp_reg/assets/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

---

### 1) 환경 변수 (세션마다)

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_INDEX=0
export MEDIAPIPE_GL_BACKEND=egl
export MEDIAPIPE_USE_GPU=1
export MP_TASK_PATH="$PWD/cohface_exp_reg/assets/pose_landmarker_full.task"
# (선택) 멀티스케일/stride 오버라이드도 가능
export HR_WIN_LIST="8,16"
export RR_WIN_LIST="32,64"
export STRIDE_FRAC="0.25"    # 각 윈도우의 25% 슬라이딩
# export FIXED_STRIDE="2.0"  # 고정 stride를 원하면 설정
```

---

### 2) 특징 추출/전처리 캐시

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

캐시는 아래에 생성:

```
cohface_exp_reg/cache_cohface_feats/<subject>_<session>.npz/.json
```

---

### 3) 학습 (LSTM / GRU)

**LSTM**

```bash
python cohface_exp_reg/run_train_lstm.py \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --batch_size 64 --lr 1e-3 --cuda 0 \
  --hidden 128 --layers 2 --bidir 1 --dropout 0.1 \
  --hr_wins 8,16 --rr_wins 32,64 --stride_frac 0.25
```

**GRU**

```bash
python cohface_exp_reg/run_train_gru.py \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --batch_size 64 --lr 1e-3 --cuda 0 \
  --hidden 256 --layers 3 --bidir 0 --dropout 0.0 \
  --hr_wins 8,16 --rr_wins 32,64 --stride_frac 0.25
```

* 장치 로그:

  ```
  [device] Using cuda:0 (NVIDIA GeForce RTX 3060 Ti)
  ```

출력:

```
cohface_exp_reg/runs/<model>_<timestamp>/
  best_model.pt, metrics.json, train_log.csv, plots/
```

---

### 내부 동작 요약

* **추출**: 포즈(코/좌우어깨) → dY/dD/dW → 리샘플(256 Hz) → 밴드패스(0.08–0.60 Hz) → z-score →
  GCC-PHAT 전역 래그(±0.5 s 제한)로 GT 정렬 → 캐시 저장.
* **학습**: `[dY_bpz,dD_bpz,dW_bpz,dC_bpz]` → LSTM/GRU → 타깃 `gt_bpz` 회귀.
  평가: MSE, 상관계수, RR 추정(PSD 피크) 등.

---

### 자주 바꾸는 설정 (`config.py`)

```python
DATA_ROOT = "/home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface"
FS_RESAMP = 256
RESP_BAND = (0.08, 0.60)
WIN_SEC, STRIDE_SEC = 8.0, 2.0
LAG_MAX_S = 0.50
SPLIT_SEED = 42
```

---

### 트러블슈팅

* **로그에 `solutions.pose (CPU) 사용`**
  → `.task` 경로가 잘못되었거나 `MP_TASK_PATH` 미설정.

  ```
  echo $MP_TASK_PATH
  ls -l "$MP_TASK_PATH"
  ```
* **T400로 잡힘**
  → 학습 실행 전에 항상 `export CUDA_VISIBLE_DEVICES=0`로 고정.
* **OOM**
  → `--batch_size` 감소, `--win_sec` 6–8 s, `num_workers` 2–4로 조절.

---

## PyCharm 런 설정(요약)

* **Script**: `cohface_exp_reg/run_extract_all.py`
  **Parameters**:

  ```
  --root /home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface --subjects 1-10 --sessions 0-3 --out cohface_exp_reg/cache_cohface_feats --fs 256 --resp_lo 0.08 --resp_hi 0.60
  ```

  **Working directory**: `/home/gongjae/PycharmProjects/rPPG-Toolbox`
  **Env**:

  ```
  CUDA_VISIBLE_DEVICES=0; MEDIAPIPE_GL_BACKEND=egl; MEDIAPIPE_USE_GPU=1; MP_TASK_PATH=/home/gongjae/PycharmProjects/rPPG-Toolbox/cohface_exp_reg/assets/pose_landmarker_full.task; PYTHONUNBUFFERED=1
  ```
* **Script**: `cohface_exp_reg/run_train_lstm.py` (또는 `run_train_gru.py`)
  **Parameters**: `--cache cohface_exp_reg/cache_cohface_feats --epochs 30 --batch_size 64 --win_sec 8 --stride_sec 2 --cuda 0`
  **Working directory**: 프로젝트 루트
  **Env**: `CUDA_VISIBLE_DEVICES=0; PYTHONUNBUFFERED=1`

