# COHFACE RR-only Regression (V1)

## 결론(요약)
- **RR 전용 16채널 입력**으로 재설계
- **멀티스케일 윈도우**: 메인 32·64s, 서브 24·48·96s (느린/빠른 호흡 커버)
- **손실**: `MSE_RR + λ*(1 - corr@soft-best-lag)` (λ=0.3 기본)
- **장치 정책**: 학습은 `cuda:0`(3060 Ti), 추출은 Mediapipe **Tasks .task가 있으면 GPU** / 없으면 **CPU 폴백**

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
  sampler.py
  train.py
  run_extract_all.py
  run_train_lstm.py
  runs/                   # 자동 생성
  cache_cohface_feats/    # 자동 생성
  assets/pose_landmarker_full.task  # 직접 다운로드
```

---

## 요구사항
- Python 3.10
- PyTorch (CUDA 권장)
- SciPy, NumPy
- Mediapipe (Tasks 권장; 미존재 시 solutions CPU 폴백)

```bash
pip install torch scipy numpy mediapipe opencv-python
```

---

## 0) 포즈 모델(.task) 다운로드
```bash
mkdir -p cohface_exp_reg/assets
wget -O cohface_exp_reg/assets/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

---

## 1) 환경 변수 (세션마다 권장)
```bash
# ==== GPU 고정 ====
export CUDA_VISIBLE_DEVICES=0
export CUDA_INDEX=0

# ==== Mediapipe (추출에만 영향; 학습엔 영향 없음) ====
export MEDIAPIPE_GL_BACKEND=egl
export MEDIAPIPE_USE_GPU=1
export MP_TASK_PATH="$PWD/cohface_exp_reg/assets/pose_landmarker_full.task"

# ==== CPU 스레드 최적화 ====
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TORCH_NUM_THREADS=16

# ==== 경로/윈도우 ====
export COHFACE_ROOT="/home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface"
export CACHE_DIR="cohface_exp_reg/cache_cohface_feats"
export RUNS_DIR="cohface_exp_reg/runs"
export RR_WIN_LIST="20,40"
export STRIDE_FRAC="0.25"     # (A안) 1/4 stride
# export FIXED_STRIDE="2.0"   # (B안) 고정 2초 stride

# ==== 손실/지표 하이퍼 ====
export PHASE_LAMBDA="0.2"
export AMP_LAMBDA="5e-2"      # 5e-4 ~ 2e-3 사이 튜닝
export PHASE_BETA="8.0"
export LAG_MAX_S="0.5"
export BPM_MIN_PROM="3.0"   # bpm 피크 민감도(완화)
export W_TREND_FC=0.05      # 상황 따라 0.03~0.07 사이 미세 튜닝 가능
```

---

## 2) 특징 추출 & 캐시
```bash
python -m cohface_exp_reg.run_extract_all \
  --root /home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface \
  --subjects 1-40 --sessions 0-3 \
  --out cohface_exp_reg/cache_cohface_feats
```
- 캐시: `cache_cohface_feats/s<subject>_k<session>.npz`
- 키: `t, dW, dY, dD, dD_perp, resp`

> **dD_perp**: 느린 어깨축(`θ_slow`)에 **수직**인 성분으로 산출.

---

## 3) 학습 (LSTM, RR-only)
1) LSTM
```bash
python -m cohface_exp_reg.run_train_lstm \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --lr 1e-3 \
  --hidden 128 --layers 2 --bidir 1 --dropout 0.1 \
  --bucket_bs "10240:4,5120:8" \
  --num_workers 12 --pin_memory 1
```
- 출력: `runs/lstm_rronly_<timestamp>/best_model.pt, metrics.json`
- 지표: `mse, mae, corr, corr_bestlag, rr_bpm_mae, hit@±2bpm`

2) GRU
```bash
python -m cohface_exp_reg.run_train_gru \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --lr 1e-3 \
  --hidden 256 --layers 3 --bidir 1 --dropout 0.1 \
  --bucket_bs "10240:4,5120:8" \
  --num_workers 12 --pin_memory 1
```
- 출력: `runs/gru_rronly_<timestamp>/best_model.pt, metrics`.json`
- 지표: `mse, mae, corr, corr_bestlag, rr_bpm_mae, hit@±2bpm`
---

## 4) 입력 16채널(순서 고정)
```
[ w_rr, y_rr, d_rr, dw_rr,
  env_w, env_y, env_d, env_dw,
  cross_wy_rr, cross_wd_rr, env_low_y, env_high_y,
  w_trend, snr_rr_hint, corr_hint_wy, corr_hint_wd ]
```
- 정의는 문서 본문(요청안)과 동일: RR 대역 필터/엔벨로프/곱/서브밴드/느린 컨텍스트.

---

## 내부 동작 요약
- 비디오→Mediapipe Pose→(dW,dY,dD,dD_perp) 추출
- 공통 256 Hz로 리샘플, resp와 시간축 정렬
- 16채널 특징 구성(RR-only)
- 멀티스케일 윈도우: **20, 40s**, stride=윈도우×0.25(기본)
- RNN(양방향 LSTM)로 RR 파형 회귀
- **손실**: `MSE + λ*(1 - corr@soft-best-lag)`

---

## 윈도우 조합 근거
- **20s**: ≥25 bpm엔 ≥8–9주기, ≥30 bpm엔 10주기
- **40s**: 32↔64 사이 중간 스케일로 파형 정합 보조.

---

## 트러블슈팅
- `.task` 미인식 → `MP_TASK_PATH` 확인
- CPU 폴백 강제 → `export MEDIAPIPE_USE_GPU=0`
- VRAM OOM → `RR_WIN_LIST`에서 96s 제거 또는 `BUCKET_BS`에서 24576:1 유지

---

# 모델 상세 (V1, RR-only SeqRegressor)

## 1) 전체 개요

**결론**

* **입력**: RR 전용 **16채널** 시계열(256 Hz)
* **샘플링**: 멀티스케일 윈도우 **메인 32·64s**, **서브 24·48·96s** (stride = win×0.25, 또는 `FIXED_STRIDE`)
* **모델**: RNN(SeqRegressor; LSTM/GRU), 출력 **\[B, T, 1]** = RR 파형
* **손실(기본)**: **SI-MSE + λ·Phase(주파수-영역 위상) 손실**, λ=0.2 권장
* **지표**: scale-align MSE/MAE, corr, **corr\@soft-best-lag**, RR bpm MAE, **hit@±2 bpm**

**근거**

* RR 파형은 **위상/지연** 민감 → **위상 불일치**를 직접 줄이는 학습이 필요
* 다양한 호흡 속도 분포를 커버하려면 **멀티스케일 윈도우**가 유리
* 스케일(진폭) 차가 큰 환경에서 **SI-MSE**가 파형 정합을 안정화

**추가설명**

* 학습 장치는 **cuda:0(3060 Ti 고정)**, 추출은 Mediapipe **Tasks .task 있으면 GPU/없으면 CPU** 폴백.

---

## 2) 좌표계/정규화 (axis\_v2 요지)

**결론**

* **dD\_perp**: 어깨선의 **느린 축(θ\_slow)** 에 **수직**인 코의 투영 성분을 사용해 자세/roll 변화에 강건
* 폭 기준 **무차원화**: `W0 = median(dW)`, `Wslow = LPF(dW, fc≈0.03 Hz)`

**근거/정의**

* $\theta(t)=\mathrm{atan2}(SR_y-SL_y, SR_x-SL_x)$ → unwrap → LPF → $\theta_{\text{slow}}$
* $v_{\perp}(t)=(-\sin\theta_{\text{slow}}, \cos\theta_{\text{slow}})$, $M=\tfrac{SL+SR}{2}$
* $dD_{\perp}(t) = (N(t)-M(t))\cdot v_{\perp}(t)$

**추가설명**

* 캐시에 `dD_perp`가 없으면 **임시로 dD 대체** 가능(성능 저하 가능).

---

## 3) 16채널 입력 스택 (순서 고정)

**결론**

* **A(원파형, RR-BP→z)**: `w_rr, y_rr, d_rr, dw_rr`
* **B(엔벨로프, |Hilbert|→z)**: `env_w, env_y, env_d, env_dw`
* **C(결합/서브밴드)**: `cross_wy_rr, cross_wd_rr, env_low_y(0.08–0.25), env_high_y(0.25–0.60)`
* **D(느린 컨텍스트, RR 금지)**: `w_trend(LPF≈0.2 Hz→z), snr_rr_hint, corr_hint_wy, corr_hint_wd`

**근거/연산자**

* $\text{BP}_{RR}(x)=\text{bandpass}(x,0.08\!-\!0.60\text{Hz})$, $z(\cdot)=\text{z-score}$
* `cross_*`: \*\*RR-BP(raw)\*\*끼리 곱 → 다시 RR-BP → z (위상/상호작용 포착)
* `snr_rr_hint, corr_hint_*`: 윈도우 내 상수 채널(추정/힌트)

**원본 파생 신호**

* $w_{\text{rel}}=dW/W_0-1,\quad y_{\text{norm}}=dY/(W_{\text{slow}}+\varepsilon),\quad d_{\text{norm}}=dD_{\perp}/(W_{\text{slow}}+\varepsilon),\quad \dot{w}_{\text{rel}}=\tfrac{d}{dt}dW/W_0$

---

## 4) 윈도우/샘플러

**결론**

* RR\_WIN\_LIST: **\[24, 32, 48, 64, 96] s** (메인 32·64)
* stride: `STRIDE_FRAC×win`(기본 0.25) 또는 `FIXED_STRIDE`(초)
* **길이 버킷 배치**: 예) `32768:1, 24576:2, 16384:4, 12288:6, 8192:10, 6144:16`

**근거**

* 빠른 호흡(>25 bpm) → **24 s**에서 **≥10주기** 확보
* 느린 호흡(5–7 bpm) → **96 s**에서 다주기 안정화
* 버킷 샘플러로 **패딩 낭비↓/OOM 방지**, 긴 시퀀스는 작은 BS

**추가설명**

* 256 Hz → 길이 T는 `[win×256]`; 버킷 키는 대략 **T와 매칭**.

---

## 5) 모델(SeqRegressor)

**결론**

* **입력 16차원** → RNN(LSTM/GRU, bi/uni, L layers, hidden H) → **\[B, T, 1]**
* **LSTM 기본**, **GRU 대안** (설정 동일)
* Dropout(입력/층 사이), AMP 사용 가능

**근거**

* RR 파형은 저주파·장주기 → RNN이 **시간 종속성**과 **부드러움** 유지에 유리
* 양방향은 **과거/미래 문맥** 통합 → 파형 정합 향상

**추가설명**

* 초기화는 PyTorch 기본; 필요 시 LayerNorm/LN-LSTM로 교체 여지.

---

## 6) 손실(학습 목적)

### 6-A. 기본 손실: **SI-MSE + λ·Phase(위상) 손실**

**결론**

* **SI-MSE**: 스케일 불일치 제거 후 파형 MSE
* **Phase 손실**: 주파수별 **위상차 $\Delta\phi_k$** 를 직접 벌점 → **지연/위상 불일치 감소**
* 권장 $\lambda = 0.2$ (`PHASE_LAMBDA`)

**정의(요지)**

* $\displaystyle \mathcal{L}_{\text{SI-MSE}}=\min_{a}\tfrac{1}{T}\|ap-g\|_2^2$
* $\displaystyle \mathcal{L}_{\text{phase}}= 1-\sum_k w_k\cos(\Delta\phi_k),\;\; w_k\propto |X_k||Y_k|$ (RR 대역만 집계)
* 총손실 $\displaystyle \mathcal{L}=\mathcal{L}_{\text{SI-MSE}}+\lambda\,\mathcal{L}_{\text{phase}}$

**추가: 진폭 붕괴 방지(옵션)**

* $a_{\hat{}}=\tfrac{p\cdot g}{p\cdot p+\varepsilon}$, $\mathcal{L}_{\text{amp}}=(a_{\hat{}}-1)^2$ → `AMP_LAMBDA≈1e-3`

### 6-B. 대안 손실: **SI-MSE + λ·(1−corr\@soft-best-lag)**

**결론**

* **라그(±`LAG_MAX_S`)** 내 **soft-max**로 가중 평균한 상관을 최대화
* 위상차/지연에 **직접 민감**하며 **시간영역**에서 구현 간단
* 권장 $\lambda = 0.3$, \*\*온도 `PHASE_BETA`\*\*로 라그 집중도 제어

**비고**

* 두 방식은 **목표 동일(위상/지연 해소)**, 구현/미분 특성만 다름. 둘 중 하나로 고정 가능.

---

## 7) 평가 지표

**결론**

* **scale-align MSE/MAE**, **corr**, **corr\@soft-best-lag**, **RR bpm MAE**, **hit@±2 bpm**
* 리포트: `scale_a_hat_mean`, `valid_bpm_windows`, `num_windows_scored`

**근거/세부**

* **scale 정렬**: $a_{\hat{}}=\tfrac{p\cdot g}{p\cdot p+\varepsilon}$, $\tilde p=a_{\hat{}}p$ 후 MSE/MAE
* **soft-best-lag corr**: $\ell\in[-L,L]$ 에 대해 $\alpha_\ell \propto e^{\beta\,\mathrm{corr}_\ell}$, $\sum \alpha_\ell=1$
* **RR bpm**: Welch-PSD(대역 0.08–0.60 Hz), **prominence=`BPM_MIN_PROM`** 완화, NaN 창 제외

---

## 8) 효율/안정성 팁

**결론**

* **버킷 샘플러** + **pin\_memory/num\_workers** + **FFT 2^n 패딩**(phase 손실 시)
* CPU 스레드/BLAS 환경변수 튜닝, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**추가설명**

* 아주 긴 창(96 s)에서 VRAM 압박 시 **배치 축소** 또는 **96 s 제거**
* 학습 중 `scale_a_hat_mean`이 **수십/수백**이면 **진폭 붕괴** → `AMP_LAMBDA`를 5e-4\~2e-3로 조정

---

## 9) 실패 모드 & 디버깅

**결론/대응**

* **bpm NaN**: `BPM_MIN_PROM` 완화(예: 2.5), 창 길이↑
* **corr는 높은데 MSE 커짐**: 스케일 미스매치 → scale-align 확인
* **val 창 수가 너무 적음**: 검증을 **한 버킷만** 보지 말고 **모든 버킷 평균** 리포트
* **극저주파 드리프트**: `w_trend`만 컨텍스트에 사용(예측/손실에는 RR대역만)

---

## 10) 재현성/로깅

**결론**

* `runs/<model>_<timestamp>/{best_model.pt, metrics.json}` 저장
* 로그: `train_mse, train_phase, train_amp, val{지표}`

**추가설명**

* Early-stop 기준: `val.corr_bestlag` 권장
* Seed: `SPLIT_SEED=42` (추가 seed 고정은 필요 시 확장)