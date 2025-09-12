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
  __init__.py
  config.py
  utils.py
  pose_backend.py
  preprocess.py
  data.py
  models.py
  train.py
  sampler.py
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
# export FIXED_STRIDE="2.0"    # 고정 stride(초)로 쓸 때

# 메모리 단편화 완화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> 참고: 단일 윈도우 사용이 필요하면 `WIN_SEC`/`STRIDE_SEC` 인자나 환경변수를 그대로 사용할 수 있습니다(하위호환).

---

## 2) 특징 추출/전처리 캐시

```bash
python -m cohface_exp_reg.run_extract_all \
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
python -m cohface_exp_reg.run_train_lstm \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --lr 1e-3 --cuda 0 \
  --hidden 128 --layers 2 --bidir 1 --dropout 0.1 \
  --hr_wins 8,16 --rr_wins 32,64 --stride_frac 0.25 \
  --bucket_bs "16384:2,8192:4,4096:16,2048:32"
```

### GRU

```bash
python -m cohface_exp_reg.run_train_gru \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --lr 1e-3 --cuda 0 \
  --hidden 256 --layers 3 --bidir 0 --dropout 0.0 \
  --hr_wins 8,16 --rr_wins 32,64 --stride_frac 0.25 \
  --bucket_bs "16384:2,8192:4,4096:16,2048:32"
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

---

---

# Methods: COHFACE 기반 RR·HR 동시 회귀 파이프라인

> **결론(요약)**
> 본 파이프라인은 **포즈 기반 변위(dY, dD, dW)** 를 추출·정렬·정규화한 뒤, **멀티스케일 윈도우( RR=32·64s / HR=8·16s )** 로 샘플링하여 **멀티헤드 RNN(SeqRegressor: LSTM/GRU)** 으로 **RR·HR을 동시 회귀**합니다. **헤드별 마스크**로 손실·지표를 분리하고, **AMP + 길이 버킷 배치**로 64초 윈도우도 OOM 없이 학습합니다.

---

## 1) 데이터 & 실행 환경

**근거**

* 원천: COHFACE `(<subject>/<session>/data.mkv, data.hdf5)`
* 장치 정책: **3060 Ti 고정(`cuda:0`)**, Mediapipe **Tasks .task**가 있으면 **GPU**, 없으면 **CPU 폴백**
* 공통 샘플레이트: `FS_RESAMP = FS_MODEL = 256 Hz`

**추가설명**

* 환경변수로 루트/캐시/윈도우/스트라이드 오버라이드 가능: `COHFACE_ROOT, CACHE_DIR, RUNS_DIR, HR_WIN_LIST, RR_WIN_LIST, STRIDE_FRAC, FIXED_STRIDE …`
* 실행은 **모듈 모드**: `python -m cohface_exp_reg.run_extract_all`, `python -m cohface_exp_reg.run_train_lstm` 등

---

## 2) 전처리(세션 단위)

**결론**
포즈→변위 추출 → 256 Hz 리샘플 → **밴드패스(RR/HR)** → **z-score** → **GCC-PHAT 전역 래그(±0.5 s) 보정** → **NPZ 캐시** 저장.

**근거(메소드)**

1. **포즈 추정**: Mediapipe Pose (Tasks 우선, 실패 시 solutions)
2. **변위 산출**:

   * `dY` (코의 수직 변위), `dD` (양 어깨 간 거리), `dW` (상체 폭/가로 변위)
   * 합성 채널 `dC = (dW + dY + dD)/3`
3. **리샘플**: 비디오 기준 시계열을 `256 Hz` 균일 격자로 보정
4. **대역 분리 & 정규화**:

   * **RR 대역** $[0.08, 0.60] Hz$, **HR 대역** $[0.7, 3.0] Hz$ 밴드패스
   * 각 채널 **z-score**로 표준화
5. **전역 시차 보정**: 입력 특징(예: `dC`)과 GT(RR/HR) 간 **GCC-PHAT**로 전역 지연 $\hat{\tau}$ 추정, $|\tau|\le0.5$ s 에서 **GT를 $-\hat{\tau}$** 만큼 시프트
6. **캐시**: `cohface_exp_reg/cache_cohface_feats/s<subject>_k<session>.npz`

   * 포함: `t, dW, dY, dD, g_resp, g_cardio(선택), subject, session`

**추가설명**

* z-score는 스케일·오프셋 민감도를 제거해 **학습 안정화**
* GCC-PHAT 정렬은 **입력–GT 위상 불일치**를 제거해 **상관/회귀 성능** 보존

---

## 3) 특징 구성 & 샘플링(멀티스케일 윈도우)

**결론**
입력은 **8채널** $[w_rr, y_rr, d_rr, c_rr, w_hr, y_hr, d_hr, c_hr]$ 로 구성하고, **RR/HR에 최적화된 윈도우 길이**로 각각 샘플링합니다.

**근거(메소드)**

* **채널 구성**

  * RR용: `w_rr, y_rr, d_rr, c_rr` = $\text{bandpass}_{RR}(\cdot)$ 후 z-score
  * HR용: `w_hr, y_hr, d_hr, c_hr` = $\text{bandpass}_{HR}(\cdot)$ 후 z-score
* **윈도우/스트라이드 정책**

  * RR: **32, 64s**, HR: **8, 16s** (기본)
  * **stride =** `FIXED_STRIDE`(있으면 고정) **else** `STRIDE_FRAC × win` (기본 0.25 → 75% 중첩)
* **타깃 & 마스크(멀티헤드)**

  * RR 윈도우: 타깃 `[gt_rr, dummy_hr]`, **mask** `[1, 0]`
  * HR 윈도우: 타깃 `[dummy_rr, gt_hr]`, **mask** `[0, 1]`
  * 세션에 `g_cardio`가 없으면 **HR 윈도우 생성 스킵**

**추가설명**

* 윈도우 길이별 **정보 대역**을 존중(RR은 긴 윈도우, HR은 짧은 윈도우)
* 마스크로 **헤드별 손실·지표를 정확히 분리**하여 학습 누수 방지

---

## 4) 배치 구성(메모리 안전)

**결론**
**LengthBucketBatchSampler + pad\_collate** 로 길이별로 다른 배치 크기를 적용, 64 s 윈도우도 OOM 없이 처리.

**근거(메소드)**

* 길이 버킷: 예) `16384:2, 8192:4, 4096:16, 2048:32` (256 Hz 기준 64/32/16/8 s)
* 동일 길이끼리 묶여 **패딩 낭비 최소화**, 긴 시퀀스는 **작은 BS**
* `pad_collate`: `[B, T_max, ·]` 패딩 + `pad_mask` 생성

**추가설명**

* **AMP(autocast+GradScaler)** 를 병행해 VRAM·연산량을 추가 절감

---

## 5) 모델(SeqRegressor) & 목적함수

**결론**
**양방향 RNN(LSTM/GRU)** 에서 시계열 회귀로 **\[B, T, 2]** (RR, HR)를 출력하고, **마스크 기반 손실**로 학습합니다.

**근거(메소드)**

* **아키텍처**: `cell ∈ {LSTM, GRU}`, `hidden`, `layers`, `bidir`, `dropout`
* **출력**: $\hat{\mathbf{y}} \in \mathbb{R}^{B\times T\times 2}$ (채널 0=RR, 1=HR)
* **손실**

  $$
  \mathcal{L} = \underbrace{\text{MSE}(\hat{y}_{RR}, y_{RR}; m_{RR})}_{\text{RR만}} \;+\;
               0.5\cdot\underbrace{\text{MSE}(\hat{y}_{HR}, y_{HR}; m_{HR})}_{\text{HR만}} \;+\;
               0.2\cdot(1-\rho_{RR})
  $$

  * $\text{MSE}(\cdot;\,m)$: **마스크된 구간만 평균**
  * $\rho_{RR}$: RR 채널의 **마스크된 상관계수**
* **최적화**: Adam(+AMP), 조기 종료(옵션), best ckpt 저장

**추가설명**

* RR 상관을 보조항으로 두어 **위상 정합·파형 품질**을 함께 유도
* 손실 가중치(0.5, 0.2)는 실험적으로 조정 가능

---

## 6) 평가 & 리포팅

**결론**
**마스크 적용 MSE·상관**(RR/HR)과 **RR bpm 오차**(밴드제한 PSD 피크 기반)를 기록합니다.

**근거(메소드)**

* **RR/HR 회귀 품질**: 마스크된 MSE, Pearson r
* **RR bpm**: RR 예측/GT를 **유효 구간 마스크**로 잘라 **대역 제한 PSD**에서 피크 주파수 → **bpm** 변환, **MAE** 집계

**추가설명**

* 출력 구조: `runs/<tag>/best_model.pt, best.json, metrics.json (val/test)`
* 태그에 **윈도우/버킷/모델 하이퍼파라미터**가 포함되어 재현성 확보

---

## 7) 전체 흐름(ASCII)

```
[Raw COHFACE] ──▶ Pose(MP Tasks/CPU fallback) ──▶ dY,dD,dW
                       │
                       └─▶ 256 Hz resample ─▶ bandpass(RR/HR) ─▶ z-score
                                           └▶ GCC-PHAT global lag (±0.5s) → shift GT
                                                         │
                                   [cache: s{subject}_k{session}.npz]
                                                         │
                     ┌────────────── data.py ────────────────┐
                     │  build 8ch (RR 4 + HR 4)              │
                     │  windows: RR(32,64s), HR(8,16s)       │
                     │  stride: FIXED_STRIDE or STRIDE_FRAC  │
                     │  mask per head (RR/HR)                │
                     └────────────────────────────────────────┘
                                      │
        LengthBucket (e.g., 16384:2, 8192:4, 4096:16, 2048:32)
                                      │
                        pad_collate (X,Y,M,pad_mask)
                                      │
                   SeqRegressor (LSTM/GRU, AMP enabled)
                                      │
           loss = MSE_RR + 0.5*MSE_HR + 0.2*(1 - corr_RR)
                                      │
                         best_model.pt / metrics.json
                                      │
                    eval: RR/HR MSE & r, RR bpm (PSD-peak)
```

---

## 8) 구현 디테일 & 재현성 팁

**결론**
**상대 임포트 고정 + 모듈 모드 실행**으로 환경 의존성을 제거하고, **버킷·AMP**로 64 s 안정화, **환경변수**로 손쉽게 실험 변형이 가능합니다.

**근거**

* 내부 모듈 임포트: `from . import config` (전 파일), 실행은 `python -m cohface_exp_reg.<module>`
* 버킷 샘플러 + AMP: 64 s 윈도우도 **BS=2** 수준에서 원활
* 설정 오버라이드: `HR_WIN_LIST, RR_WIN_LIST, STRIDE_FRAC, FIXED_STRIDE, COHFACE_ROOT, CACHE_DIR …`

**추가설명**

* HR GT(`g_cardio`)가 없는 세션은 **HR 윈도우 자동 스킵** → RR 학습만 진행
* RR·HR 윈도우는 **한 배치 안에 섞일 수 있으나**, 마스크로 손실/지표가 분리되어 안전

---

## 9) 한눈에 파라미터(핵심만)

* **주파수 & 필터**:

  * RR: 0.08–0.60 Hz, HR: 0.7–3.0 Hz (BP 차수/설계는 구현 기본값)
  * 전역 래그 탐색: ±0.5 s
* **멀티스케일 윈도우**: RR=32·64 s, HR=8·16 s

  * stride: `FIXED_STRIDE`(초) **or** `STRIDE_FRAC×win`(기본 0.25)
* **배치**: 버킷 맵 예) `16384:2, 8192:4, 4096:16, 2048:32`
* **손실**: `MSE_RR + 0.5·MSE_HR + 0.2·(1 - corr_RR)`
* **평가**: RR/HR MSE·r, **RR bpm MAE(PSD 피크)**

