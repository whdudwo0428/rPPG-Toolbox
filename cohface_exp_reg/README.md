# COHFACE RR-only Regression — **V1.5 (updated)**

20·40s, corr-loss, **test 지표/플롯/Eval 스크립트 추가**

## 0) 결론(요약)

* **입력**: RR 전용 **16채널** (256 Hz, Mediapipe Pose 파생: w,y,d,dw + envelope/interaction/subband/slow-context)
* **윈도우**: **20 s, 40 s** (stride = win×0.25 또는 `FIXED_STRIDE` 고정초)
* **손실(고정)**: **z-MSE + λ·(1 − corr\@soft-best-lag)** (`λ=0.3`, 라그 ±2.0 s, softmax 온도 `β=8.0`)
* **평가**: **스케일·부호 정렬 후** MSE/MAE, corr, corr\@soft-best-lag, **Welch-BPM(0.08–0.60 Hz)**
* **장치**: 학습 `cuda:0`(3060 Ti), 추출은 Mediapipe **Tasks .task 있으면 GPU** / 없으면 **CPU 폴백**
* **옵션**: 세션 전역 **부호/라그 사전정렬**(`ENABLE_PREALIGN=1`, `±4 s`) 지원

### 이번 업데이트 핵심(요청 반영)

1. **`metrics.json`에 `val` + `test` 모두 저장**
2. **BPM NaN 방지**: Welch 피크 미검출 시 **대역 argmax 폴백**(기본 ON, `BPM_FALLBACK_ARGMAX=1`)
3. **플롯 저장**: `best_model.pt`로 **GT vs Pred(스케일 정렬)** 오버레이 **테스트 4세션 PNG** 저장 스크립트 추가
4. **런너 정리**: `run_train_gru.py`, `run_base_model.py`를 **LSTM 런너 형식**으로 통일(길이 버킷·단일 DataLoader, `--cell` 제거)

---

## 1) 변경 로그 (V1.5 updated)

* **평가 저장**: 학습 종료 시 `val`과 **`test` 지표 동시 저장** (`runs/<tag>/metrics.json`)
* **BPM NaN 개선**: `utils.welch_psd_rr_bpm()`에 **prominence 피크 없을 때 argmax 폴백** (환경변수로 on/off)
* **오버레이 플롯**: `run_eval_best.py`로 **테스트 4세션** *GT vs Pred(aligned)* PNG 저장
* **데이터로더 일관화**: LSTM/GRU/Base 러너 모두 **BucketBatchSampler(5120/10240 분리)** — 길이 혼배치 스택 오류 제거
* **문서/스크립트**: 실행 예시·프리셋·트러블슈팅 갱신

---

## 2) 폴더 구조

```
cohface_exp_reg/
  config.py
  utils.py                  # 필터/정규화/상관/정렬/PSD-BPM(폴백 포함)
  pose_backend.py
  preprocess.py
  data.py                   # 16ch 스택 + 20/40s 윈도우
  models.py                 # SeqRegressor(LSTM)
  sampler.py                # 길이별 버킷(5120/10240)
  train.py                  # 학습/평가 루프(corr-loss, 정렬 평가)
  run_extract_all.py        # 특징 추출 캐시 생성
  run_train_lstm.py         # 학습 엔트리포인트 (val+test 저장)
  run_train_gru.py          # LSTM형 러너(이름만 GRU)
  run_base_model.py         # LSTM형 베이스 러너(--cell 제거)
  run_eval_best.py          # best_model.pt로 test 평가 + 플롯 저장(신규)
  runs/                     # 자동: 모델/메트릭/플롯 저장
  cache_cohface_feats/      # 자동: npz 캐시 (t,dW,dY,dD[,dD_perp],resp)
  assets/pose_landmarker_full.task
```

---

## 3) 요구사항 & 설치

* Python 3.10
* PyTorch (CUDA 권장)
* SciPy, NumPy, Mediapipe(Tasks 권장), OpenCV
* **플롯 저장용**: `matplotlib`

```bash
pip install torch scipy numpy mediapipe opencv-python matplotlib
```

---

## 4) 모델 입력(16채널) — 핵심만 재강조

* RR 대역(0.08–0.60 Hz) **원파형 4 + 엔벨로프 4 + 상호/서브밴드 4 + 느린 컨텍스트 4 = 16ch**
* 타깃 `resp`는 **RR-bandpass 후 per-window z-score** → **z-MSE**로 학습, **정렬 후** 지표 계산

채널 순서(고정):

```
[ w_rr, y_rr, d_rr, dw_rr,
  env_w, env_y, env_d, env_dw,
  cross_wy_rr, cross_wd_rr, env_low_y, env_high_y,
  w_trend, snr_rr_hint, corr_hint_wy, corr_hint_wd ]
```

---

## 5) 파이프라인 요약

```
Video → Pose(Mediapipe) → dW/dY/dD_perp (axis_v2)
→ Resample 256Hz → 16ch RR-only stack
→ Windows (20s,40s), stride=win×0.25
→ SeqRegressor(LSTM, bi/2L, H=128) → RR waveform (B,T,1)
→ Loss: z-MSE + λ·(1 − corr@soft-best-lag, ±2.0s, β=8)
→ Eval: (scale·sign) aligned MSE/MAE, corr, corr@bestlag, Welch-BPM(+fallback)
```

---

## 6) 모델(SeqRegressor, LSTM)

* 입력 `[B,T,16]` → **Bi-LSTM(2L, H=128, drop=0.1)** → `Linear–SiLU–Linear` → `[B,T,1]`
* RNN은 저주파·부드러운 파형 근사에 유리, **양방향**로 위상 정합 안정화

---

## 7) 손실/지표(고정: corr 모드)

### 7.1 손실

* **Main**: per-window **z-MSE**
* **Aux**: **λ·(1 − corr\@soft-best-lag)**, 라그 `±2.0 s`, 온도 `β=8.0`
  → 전역 지연/위상 오차에 강건

### 7.2 지표(정렬 후)

* **scale-align MSE/MAE**, **corr**, **corr\@soft-best-lag**
* **RR BPM(Welch PSD)**: 0.08–0.60 Hz, `BPM_MIN_PROM`(기본 3.0)
* **NaN 방지(업데이트)**: prominence 피크가 없을 때 **대역 argmax 폴백**

  * ON/OFF: `BPM_FALLBACK_ARGMAX={1|0}`, 기본 **1(사용)**

---

## 8) 설정/환경 변수

```bash
# ==== 고정 의도(권장 프리셋) ====
export LOSS_MODE=corr
export PHASE_LAMBDA=0.3
export LAG_MAX_S=2.0
export PHASE_BETA=8.0

# ==== 윈도우/스트라이드 ====
export RR_WIN_LIST="20,40"
export STRIDE_FRAC="0.25"
# export FIXED_STRIDE="2.0"  # (선택) 고정 stride(초)

# ==== 경로 ====
export COHFACE_ROOT="/path/to/cohface"
export CACHE_DIR="cohface_exp_reg/cache_cohface_feats"
export RUNS_DIR="cohface_exp_reg/runs"

# ==== 장치/스레드 ====
export CUDA_VISIBLE_DEVICES=0
export CUDA_INDEX=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TORCH_NUM_THREADS=16

# ==== Mediapipe (추출용) ====
export MEDIAPIPE_GL_BACKEND=egl
export MEDIAPIPE_USE_GPU=1
export MP_TASK_PATH="$PWD/cohface_exp_reg/assets/pose_landmarker_full.task"

# ==== BPM/컨텍스트 ====
export BPM_MIN_PROM=3.0              # (필요시 2.0~2.5로 낮추면 유효 창↑)
export BPM_FALLBACK_ARGMAX=1         # 피크 없을 때 argmax 폴백 사용
export SNR_HIT_BPM=2
export W_TREND_FC=0.05
```

```
# ==== (선택) 전역 부호/라그 사전정렬 ====
# export ENABLE_PREALIGN=1
# export PREALIGN_MAX_LAG=4.0
```

---

## 9) 실행

### 9.1 포즈 모델(.task) 다운로드

```bash
mkdir -p cohface_exp_reg/assets
wget -O cohface_exp_reg/assets/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

### 9.2 특징 추출 & 캐시

```bash
python -m cohface_exp_reg.run_extract_all \
  --root /path/to/cohface \
  --subjects 1-40 --sessions 0-3 \
  --out cohface_exp_reg/cache_cohface_feats
```

* 생성: `cache_cohface_feats/s<subject>_k<session>.npz`
* 키: `t, dW, dY, dD, dD_perp, resp` (`dD_perp` 없으면 `dD` 대체 허용)

### 9.3 학습 (LSTM; **val+test 저장**)

```bash
python -m cohface_exp_reg.run_train_lstm \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --lr 1e-3 \
  --hidden 128 --layers 2 --bidir 1 --dropout 0.1 \
  --num_workers 12 --pin_memory 1
```

* 결과: `runs/lstm_rronly_<timestamp>/{best_model.pt, metrics.json}`
  (`metrics.json`에 **val + test** 동시 기록)

---

## 10) 데이터셋/스플릿

* 캐시 파일의 **subject/session 키**로 **60/20/20** (train/val/test) 분할
* 재현성: `SPLIT_SEED=42`
* 세부 제어 필요 시 `data.py`의 key 리스트를 고정 리스트로 대체

---

## 11) 평가/플롯(체크포인트 재사용)

### 11.1 `best_model.pt`로 **테스트 지표 + 오버레이 4장** 저장

```bash
# 분해능 개선 옵션(추천)
export BPM_FALLBACK_ARGMAX=1     # 폴백 사용
export BPM_SUBBIN_QUAD=1         # 서브-빈 보간
export BPM_NFFT_UP=4             # 격자 4배 세분화 (2~4 권장)

BEST=cohface_exp_reg/runs/lstm_rronly_20250915_172517/best_model.pt

python -m cohface_exp_reg.run_eval_best \
  --cache cohface_exp_reg/cache_cohface_feats \
  --model "$BEST" \
  --hidden 128 --layers 2 --bidir 1 --dropout 0.1 \
  --n_plots 4
```

* 출력:
# - FULL 지표: {run_dir}/metrics_test_full.json   (전 창 evaluate 결과)
# - 샘플 플롯: {run_dir}/plots/test_sessionXX_[s**_k**.npz].png

> 특정 사람(세션)을 지정하고 싶으면 `run_eval_best.py`의 선택 로직을 `sX_kY` 필터로 바꿔드릴 수 있습니다.

---

## 12) 트러블슈팅

* **`stack expects each tensor to be equal size`**
  → 검증/테스트도 **BucketBatchSampler** 사용(5120과 10240 혼배치 금지)
* **`rr_bpm_mae`/`hit@±2bpm`가 NaN**

  1. `export BPM_MIN_PROM=2.0~2.5`로 낮춰 **유효 피크↑**
  2. `export BPM_FALLBACK_ARGMAX=1`(기본)로 **argmax 폴백** 유지
* **상관 낮고 BPM만 좋아 보임**
  → prominence 높아 피크 강제 선택 가능성 → `BPM_MIN_PROM` 낮추기(2.0\~2.5)
* **val corr 불안정**
  → `ENABLE_PREALIGN=1` 또는 `LAG_MAX_S=3.0~4.0`
* **OOM**
  → `--bucket_bs '10240:2,5120:4'` 등 배치 축소, `num_workers/pin_memory` 조절
* **환경변수 오염 주의**
  → `W_TREND_FC`와 `MP_TASK_PATH` **서로 다른 줄**에 설정

---

## 13) 확장 로드맵

1. **멀티헤드**(파형 + BPM 동시 예측)
2. **Cross-window consistency**(위상/주파수 스무딩 정규화)
3. **Transformer/TCN 백본**(40s 상한 향상)
4. **Domain Augmentation**(밝기/헤드모션/스케일)
5. **Robust PSD Loss**(대역 파워 분포 매칭)
6. **Noise Gating**(SNR 힌트 기반 가중 손실)
7. **Self-Supervised Pretraining**(마스크 복원/상관 대조)

---

## 14) 재현성/로깅

* 저장: `runs/<tag>/{best_model.pt, metrics.json}`

  * **metrics.json**: `val.*`, `test.*`
* 로그: `train_main(z-MSE)`, `train_aux(corr-loss)`, `train_total`, `val.*`
* Early-stop: `val.corr_bestlag` 기준
* Seed: `SPLIT_SEED=42` (필요 시 torch/cuda/dataloader 추가 고정)

---

## 15) 라이선스

* COHFACE 데이터 정책, Mediapipe / PyTorch / SciPy 라이선스 준수

---

### 빠른 프리셋 (한 줄)

```bash
export LOSS_MODE=corr; export PHASE_LAMBDA=0.3; export LAG_MAX_S=2.0; export PHASE_BETA=8.0; \
export RR_WIN_LIST="20,40"; export STRIDE_FRAC=0.25; \
export BPM_MIN_PROM=3.0; export BPM_FALLBACK_ARGMAX=1; \
export SNR_HIT_BPM=2; export W_TREND_FC=0.05
```