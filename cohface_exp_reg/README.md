# COHFACE RR-only Regression — **V1.5** (20·40s, corr-loss)

## 0) 결론(요약)

* **입력**: RR 전용 **16채널** 시계열(256 Hz), Mediapipe Pose에서 파생 (w,y,d,dw + envelope/interaction/subband/slow-context)
* **윈도우**: **20 s, 40 s** (stride = win×0.25, 또는 `FIXED_STRIDE` 고정초)
* **손실(고정)**: **z-MSE + λ·(1 − corr\@soft-best-lag)**, `λ=0.3`, 라그 범위 `±2.0 s`, softmax 온도 `β=8.0`
* **평가**: **스케일·부호 정렬 후** MSE/MAE, corr, corr\@soft-best-lag, Welch-BPM (0.08–0.60 Hz)
* **장치 정책**: 학습 `cuda:0`(3060 Ti), 추출은 Mediapipe **Tasks .task가 있으면 GPU** / 없으면 **CPU 폴백**
* **옵션**: 세션 전역 **부호/라그 사전정렬**(`ENABLE_PREALIGN=1`, `±4 s`) 지원

**의도와 기대 효과**

* corr-loss로 **위상/지연 불일치**를 직접 최소화 → \*\*단창(20s)\*\*에서도 **위상 정합/상관**을 안정적으로 끌어올림
* per-window **z-MSE**로 **스케일 의존성 제거** → 진폭 붕괴/정렬계수 포화 방지
* **정렬 후 평가**로 지표 신뢰도↑(과거 음수 corr 고착/아티팩트 제거)

---

## 1) 변경 로그 (V1.5)

* 손실을 **corr 모드로 고정**(z-MSE + λ·(1 − corr\@soft-best-lag), `LAG_MAX_S=2.0`)
* 평가단에서 **스케일·부호 정렬 후** corr/MSE 계산
* 16채널 **RR-only** 입력 스택 정의 및 식/순서 고정
* 윈도우 **20·40s** 고정 + 버킷 배치(5120/10240 샘플 길이)
* **옵션** 전역 부호/라그 **사전정렬(ENABLE\_PREALIGN)** 추가

---

## 2) 폴더 구조

```
cohface_exp_reg/
  config.py                 # 하이퍼/경로/손실 모드/라그 범위 등
  utils.py                  # 필터/정규화/상관/정렬/PSD-BPM
  pose_backend.py           # (기존) Mediapipe 연동
  preprocess.py             # (기존) 시그널 캐싱 파이프라인
  data.py                   # 16채널 스택 + 20/40s 윈도우 데이터셋
  models.py                 # SeqRegressor(LSTM)
  sampler.py                # 길이별 버킷 배치(5120/10240)
  train.py                  # 학습/평가 루프(corr-loss, 정렬 평가)
  run_extract_all.py        # 특징 추출 캐시 생성
  run_train_lstm.py         # 학습 엔트리포인트
  runs/                     # 자동 생성: 모델/메트릭 저장
  cache_cohface_feats/      # 자동 생성: npz 캐시 (t,dW,dY,dD[,dD_perp],resp)
  assets/pose_landmarker_full.task  # 직접 다운로드
```

---

## 3) 요구사항 & 설치

* Python 3.10
* PyTorch (CUDA 권장)
* SciPy, NumPy
* Mediapipe (Tasks 권장; 없으면 solutions CPU 폴백)

```bash
pip install torch scipy numpy mediapipe opencv-python
```

---

## 4) 모델 입력(16채널) 설계 — **세부 정의**

### 4.1 원천 시그널 (Pose 파생; 256 Hz resample)

* `dW`: 좌우 어깨 폭 (pixel or norm)
* `dY`: 코–어깨중점 수직 방향 위치
* `dD_perp`: 코–어깨중점 벡터를 **느린 어깨축** `θ_slow`에 **수직** 투영한 성분
* `dw`: `dW`의 시간 미분(중심차분), 상대스케일

**정규화 스칼라/트렌드**

* 폭 기준: `W0 = median(dW)`
* 느린 폭: `W_slow = LPF(dW, fc=W_TREND_FC≈0.05 Hz)`

**무차원/상대 신호**

* `w_rel = dW / W0 - 1`
* `y_norm = dY / (W_slow + ε)`
* `d_norm = dD_perp / (W_slow + ε)`
* `dw_rel = d/dt(dW) / (W0 + ε)`

### 4.2 RR 대역 처리 & 보조 피처

RR 대역: 0.08–0.60 Hz (≈ 4.8–36 bpm)

* **A: 원파형 (RR-BP → z)**
  `w_rr, y_rr, d_rr, dw_rr = z( BP_RR(w_rel/y_norm/d_norm/dw_rel) )`
* **B: 엔벨로프 (|Hilbert| → z)**
  `env_w, env_y, env_d, env_dw = z( |Hilbert( BP_RR(x) )| )`
* **C: 상호작용/서브밴드**
  `cross_wy_rr = z( BP_RR( w_rr * y_rr ) )`
  `cross_wd_rr = z( BP_RR( w_rr * d_rr ) )`
  `env_low_y  = z( |Hilbert( BP(y_norm, 0.08–0.25) )| )`
  `env_high_y = z( |Hilbert( BP(y_norm, 0.25–0.60) )| )`
* **D: 느린 컨텍스트(예측 대역 제외)**
  `w_trend = z( LPF(w_rel, fc≈0.2 Hz) )`
  `snr_rr_hint`: RR 대역 SNR 힌트(0–1 정규화)
  `corr_hint_wy/wd`: `w_rr`와 `y_rr/d_rr` 간 상관 절대값

### 4.3 최종 16채널 스택(순서 고정)

```
[ w_rr, y_rr, d_rr, dw_rr,
  env_w, env_y, env_d, env_dw,
  cross_wy_rr, cross_wd_rr, env_low_y, env_high_y,
  w_trend, snr_rr_hint, corr_hint_wy, corr_hint_wd ]
```

### 4.4 타깃(라벨)

* `resp`(호흡 벨트/신뢰 신호) → **RR-bandpass → per-window z-score**
  → 학습 시 **z-MSE**, 평가 시는 **정렬 후 지표**에 사용

---

## 5) 방법론(메소드) 구성

### 5.1 파이프라인 개요

```
Video → Pose(Mediapipe) → dW/dY/dD_perp (axis_v2)
 → Resample 256Hz → 16ch RR-only stack
 → Sliding Windows (20s,40s) with stride=win×0.25
 → SeqRegressor(LSTM, bi/2L, H=128) → RR waveform (B,T,1)
 → Loss: z-MSE + λ·(1 − corr@soft-best-lag, ±2.0s, β=8)
 → Eval: (scale·sign) aligned MSE/MAE, corr, corr@bestlag, Welch-BPM
```

### 5.2 핵심 기법

* **axis\_v2 정규화**: 폭 기준/느린 폭 기반 **무차원화** + \*\*수직축 투영(dD\_perp)\*\*로 자세/roll 변화에 강건
* **RR-only 다채널 설계**: 원파형+엔벨로프+상호작용+서브밴드+느린 컨텍스트 → **호흡 주기/진폭/상호위상** 포착
* **corr-loss**: 라그(±2.0s) 내 **soft best-lag 상관**을 직접 최대화 → **전역 지연/위상 오차**에 강건
* **정렬 기반 평가**: 지표 산출 전에 **스케일·부호 정렬**로 **평가 편향 제거**
* **버킷 배치**: 길이 5120(20s)/10240(40s) 별 배치 규모 분리 → **OOM 방지 & 처리량↑**

---

## 6) 모델 아키텍처

### 6.1 SeqRegressor (LSTM)

* 입력: `[B, T, 16]`
* 본체: **Bi-LSTM** (`layers=2`, `hidden=128`, `dropout=0.1`)
* 헤드: `Linear → SiLU → Linear`
* 출력: `[B, T, 1]` (RR 파형)

**선정 근거**

* RR은 장주기/저주파 경향 → **RNN**이 **부드러운 시계열** 근사 및 **의존성** 보존에 유리
* **양방향**으로 과거·미래 문맥 동시 활용 → **위상 정합** 안정화

> (옵션) GRU/LN-LSTM/LayerNorm 등으로 대체 가능. CNN/TCN/Transformer로 확장 시, corr-loss 그대로 적용 가능.

---

## 7) 손실 및 지표 — **corr 모드 고정**

### 7.1 손실

* **Main**: per-window **z-MSE**
  `p̂ = z(p); ĝ = z(g);  L_main = MSE(p̂, ĝ)`
* **Aux**: **λ·(1 − corr\@soft-best-lag)**
  라그 범위 `±LAG_MAX_S(=2.0s)` 내 정규화된 상관을 **softmax(β=8)** 가중 평균
  `L_aux = λ·(1 − Σ_ℓ softmax(β·corr_ℓ)·corr_ℓ)`

> 추천값: `λ=0.3`, `β=8.0`, `LAG_MAX_S=2.0`
> 진동이 크면 `λ∈[0.25, 0.35]`, `β∈[6, 12]` 조정

### 7.2 평가 지표 (정렬 후)

* **scale-align MSE/MAE**: `â = (p·g)/(p·p+ε)`, `p* = â·p` 후 MSE/MAE
* **corr**: `corr(p*, g)`
* **corr\@soft-best-lag**: 상동, 라그 `±2.0s`
* **RR BPM**: Welch PSD(0.08–0.60 Hz), prominence=`BPM_MIN_PROM`(기본 3.0)
* 보고: `scale_a_hat_mean`, `valid_bpm_windows`, `num_windows_scored`

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
export BPM_MIN_PROM=3.0
export SNR_HIT_BPM=2
export W_TREND_FC=0.05

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
* 키: `t, dW, dY, dD, dD_perp, resp` (없으면 `dD_perp`→`dD` 대체 허용)

### 9.3 학습 (LSTM)

```bash
python -m cohface_exp_reg.run_train_lstm \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --lr 1e-3 \
  --hidden 128 --layers 2 --bidir 1 --dropout 0.1 \
  --bucket_bs "10240:4,5120:8" \
  --num_workers 12 --pin_memory 1
```

* 결과: `runs/lstm_rronly_<timestamp>/{best_model.pt, metrics.json}`

---

## 10) 데이터셋/스플릿 정책

* 캐시 파일명에서 **subject/session 키**를 추출해 **60/20/20** (train/val/test) 분할
* 동일 규칙으로 **재현성 유지**(기본 `SPLIT_SEED=42`)

> 세부 스플릿을 통제하려면 `data.py`의 key 함수/리스트를 고정 리스트로 대체하세요.

---

## 11) 트러블슈팅

* **상관은 낮고 BPM만 좋아 보임**: Welch prominence가 높아 **피크 강제 선택**된 경우 → `BPM_MIN_PROM↓(2.0~2.5)`
* **val corr가 불안정**: `ENABLE_PREALIGN=1`로 전역 부호/라그 정렬 or `LAG_MAX_S↑(3.0~4.0)`
* **OOM**: 40s 배치 축소(`bucket_bs`에서 10240의 배치 감소), `pin_memory/num_workers` 조절
* **메트릭 편향 감지**: `scale_a_hat_mean`이 극단적(±100↑)이면 입력 스케일링/진폭 붕괴 의심 → z-MSE 유지 권장

---

## 12) 한계 & 리스크

* **라벨 품질/동기화**: COHFACE 세션 간 **전역 라그/부호** 불일치 가능 → `ENABLE_PREALIGN` 또는 `LAG_MAX_S` 확대 필요
* **짧은 창의 한계**: 20s에서 **극저주파(≤0.1 Hz)** 추정은 제한적 → 40s 비중 증가 또는 멀티창 앙상블 고려
* **피처 의존성**: Mediapipe 추출 품질에 민감(조명/헤드운동) → `snr_rr_hint`/`corr_hint_*`로 완화하되, 실패 세션 필터링 파이프라인 고려

---

## 13) 확장 로드맵(추가 능력)

1. **멀티해드 학습**: 파형 + BPM 동시 예측(보조 분기) → **BPM 안정화**
2. **Cross-window consistency**: 인접 윈도우 간 위상/주파수 **스무딩 정규화**
3. **Transformer/TCN 백본**: 장거리 의존성↑, 40s 성능 상한 향상
4. **Domain Augmentation**: 밝기/헤드모션/스케일 perturbation → **일반화↑**
5. **Robust PSD Loss**: RR 대역 파워 분포 매칭(earth mover’s distance 등)
6. **Noise Gating**: `snr_rr_hint` 기반 가중 손실(가중치↓)로 **저품질 구간 영향 완화**
7. **Self-Supervised Pretraining**: 마스크 복원/상관 대조로 **표본 효율↑**

---

## 14) 재현성/로깅

* 저장: `runs/<tag>/{best_model.pt, metrics.json}`
* 로그: `train_main(z-MSE)`, `train_aux(corr-loss)`, `train_total`, `val.*`
* Early-stop: `val.corr_bestlag` 기준
* Seed: `SPLIT_SEED=42`(필요 시 DataLoader/torch/cuda 추가 고정)

---

## 15) 라이선스/인용

* COHFACE 데이터 사용 정책 준수
* Mediapipe / PyTorch / SciPy 라이선스 준수

---

### 빠른 프리셋 (한 줄)

```bash
export LOSS_MODE=corr; export PHASE_LAMBDA=0.3; export LAG_MAX_S=2.0; export PHASE_BETA=8.0; \
export RR_WIN_LIST="20,40"; export STRIDE_FRAC=0.25; \
export BPM_MIN_PROM=3.0; export SNR_HIT_BPM=2; export W_TREND_FC=0.05
```

---