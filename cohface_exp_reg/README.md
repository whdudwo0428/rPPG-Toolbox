# COHFACE RR-only Regression — **V1.6 (Detailed)**

20·40s 멀티윈도우, **미분가능 corr-loss**, **스케일/엔벨로프/분산(variance) 보조항**, **SNR-가중 학습**, **AMP-호환 gradient clipping**, 테스트 지표/플롯 자동 저장

---

## 0) 결론(요약)

- **핵심 아이디어**: rPPG(광학) 대신 **상체 기하(어깨–코) 기반 움직임**으로 호흡수(RR) 파형을 회귀.  
- **입력**: RR 전용 **16채널**(256 Hz) — **파형/에너지/결합/느린 컨텍스트** 4그룹으로 역할 분리 → **자세·시점·개인별 위상차**에 강건.  
- **모델**: Bi-LSTM(2L, H=128) → RR 파형 시계열 출력.  
- **손실**: `z-MSE` + `λ·(1 − corr@soft-best-lag)` + `α·|â−1|` + `γ·L1(envP,envG)` + `δ·|log(std(P)/std(G))|`  
  (상관항은 **Torch 미분가능 버전**, â는 예측↔GT 최소자승 스케일)  
- **학습 가중**: 창별 `snr_rr_hint(h∈[0,1])`로 **SNR-가중 학습** 적용 → 저SNR 창 과벌점/평탄화 완화.  
- **평가**: 스케일·부호 정렬 후 MSE/MAE, corr, corr@soft-best-lag, Welch-BPM(0.08–0.60 Hz, **prominence + argmax 폴백**)  
- **윈도우**: 20 s + 40 s (stride = win×0.25 또는 `FIXED_STRIDE`)  
- **옵션**: 세션 전역 **부호/라그 사전정렬**(`ENABLE_PREALIGN`, `±4 s`)  

> **의도**:  
> ① **파형 정합**(z-MSE) + ② **위상/지연 강건성**(soft-corr) + ③ **진폭/에너지/분산 정합**(스케일·엔벨로프·분산) + ④ **훈련 안정성**(SNR-가중, grad clipping) →  
> **BPM 안정성은 유지**하면서 **파형 corr**과 **전이 내성**을 끌어올림.

---

## 1) V1.6 변경점 (vs V1.5)

1. **corr-loss**를 **미분가능(Torch)** 으로 재구현 → 라그 탐색을 softmax 기대값으로 근사, **역전파 경로 보장**.  
2. **스케일 패널티** `|â−1|` + **엔벨로프 L1(RMS)** + **분산(variance) 패널티** `|log(std(P)/std(G))|` 추가 → **진폭 과대/과소**·**저SNR 평탄화**·**전역 에너지 스케일** 보정.  
3. **SNR-가중 학습**: 창별 `h`로 loss에 `(1−κ)+κ·h` 곱 → 저SNR 창 과벌점 완화.  
4. **AMP-호환 Gradient Clipping**(L2, 기본 1.0) → **LSTM 폭주** 방지.  
5. `run_eval_best.py` 인자 정리 및 문서/프리셋 정합: **val+test 동시 저장**, **Welch-BPM NaN 방지**(피크 미검출 시 **대역 argmax 폴백**, 서브-빈 보간/NFFT 업샘플 옵션 통일).

---

## 2) 파이프라인(개념)

Video → Mediapipe Pose → dW/dY/dD_perp (axis_v2 정규화)
→ Resample 256 Hz → 16ch RR-only stack
→ Windows (20s,40s), stride=win×0.25
→ SeqRegressor(LSTM, bi/2L, H=128) → RR waveform (B,T,1)
→ Loss: z-MSE + λ·(1 − corr@soft-best-lag, ±1.5 s, β=10.0)
+ α·|â−1| + γ·L1(envP,envG) + δ·|log(std(P)/std(G))|
→ Eval: (scale·sign) aligned MSE/MAE, corr, corr@bestlag, Welch-BPM(+fallback)


**좌표계/기하 정의(요점)**  
- **dW**: 좌·우 어깨 간 거리(가로 폭), **W0**=median(dW), **Wslow**=LPF(dW, `fc≈0.05–0.2Hz`)  
- **dY**: 어깨중점의 세로 위치(상하 들썩임)  
- **dD_perp (n⊥)**: 코–어깨중점 벡터를 **느린 어깨축**(v_slow)에 직교 투영한 성분 → **roll/시점 변화**에 둔감  
- 모든 채널은 **RR 대역(0.08–0.60 Hz) bandpass + z-score** 또는 **엔벨로프**로 정규화

---

## 3) **16채널 설계 — 의도·포착·효과·모델 능력**

### A. **파형(위상 민감; RR bandpass→z)**

1) **w_rr** = z(BP_RR(dW/W0 − 1))  
- **의도**: 흉곽 가로 확장/수축의 **순수 파형**  
- **효과**: 흉식 호흡에서 **SNR↑**  
- **능력**: 체격 스케일 제거로 개인 간 진폭 차 둔감

2) **y_rr** = z(BP_RR(dY/(Wslow+ε)))  
- **의도**: 상체의 **세로 들썩임** 파형  
- **효과**: w와 **보완 위상** 제공 → 상쇄 완화  
- **능력**: 카메라 줌/거리 변화에 **스케일 불변**

3) **d_rr** = z(BP_RR(dD_perp/(Wslow+ε)))  
- **의도**: 머리–흉곽 **수직 상호작용**(몸축 기준)  
- **효과**: 전역 y축 대신 **인체 축** 사용 → **roll/시점**에 강함  
- **능력**: 도메인 전이(자세/카메라)에서도 **corr 유지**

4) **dw_rr** = z(BP_RR((d/dt dW)/W0))  
- **의도**: 가로 확장 **속도**(미분) 파형  
- **효과**: w_rr 대비 **≈90° 위상 보강** → 피크 정렬 용이  
- **능력**: 빠른 호흡에서 **전이(edge) 강조**로 검출 민감도↑

### B. **에너지/엔벨로프(위상 불변; |Hilbert|→z 또는 RMS)**

5) **env_w** = env(BP_RR(dW/W0 − 1))  
6) **env_y** = env(BP_RR(dY/(Wslow+ε)))  
7) **env_d** = env(BP_RR(dD_perp/(Wslow+ε)))  
8) **env_dw** = env(BP_RR((d/dt dW)/W0))  
- **의도**: 각 파형의 **강도/세기**(에너지)  
- **효과**: **위상차 불감** → 개인별 위상차/창 내부 지연에도 안정  
- **능력**: **soft-alignment 손실** 및 **엔벨로프 보조항**과 시너지를 내 파형 진폭 드리프트 억제

### C. **결합/서브밴드(양상 분해)**

9) **cross_wy_rr** = z(BP_RR( BP_RR(w) * BP_RR(y) ))  
- **의도**: 가로(w)–세로(y) **동시 변동/동조**  
- **효과**: 흉식·복식 **혼합 양식**에서 상쇄/혼탁 완화  
- **능력**: **호흡 방식 변화**에 대한 **일반화**↑

10) **cross_wd_rr** = z(BP_RR( BP_RR(w) * BP_RR(n⊥) ))  
- **의도**: 흉곽–머리 **기계적 결합**  
- **효과**: **말하기/고개짓** 등 비호흡성 동작 구분 근거  
- **능력**: 비호흡성 아티팩트에 **내성** 향상

11) **env_low_y** = env(bandpass(dY/(Wslow+ε), **0.08–0.25 Hz**))  
12) **env_high_y** = env(bandpass(dY/(Wslow+ε), **0.25–0.60 Hz**))  
- **의도**: **저주기/고주기** 호흡 에너지 분리  
- **효과**: 긴/짧은 윈도우 **상보성** 확보, 과소·과대 추정 방지  
- **능력**: **휴식/수면↔운동/긴장** 상황 모두 추정 유지

### D. **느린 컨텍스트(RR 밴드 금지; LPF<0.3 Hz→z)**

13) **w_trend** = z(LPF(dW/W0 − 1, fc≈0.2 Hz))  
- **의도**: 가로 폭의 **저역 드리프트/자세 변화**  
- **효과**: 오염 구간 **자체 게이팅** 근거  
- **능력**: **거짓 상관** 상승 억제 → 강건성↑

14) **snr_rr_hint** = z(RR-SNR prominence ∈[0,1], **창 내 상수**)  
- **의도**: **신뢰도 힌트**로 저SNR 창 식별  
- **효과**: **가중/주의**에 활용 가능(학습/추론 정책 일치 권장)  
- **능력**: **샘플 선택/가중**으로 일반화↑

15) **corr_hint_wy** = z(|corr(BP_RR(w),BP_RR(y))|, **상수**)  
16) **corr_hint_wd** = z(|corr(BP_RR(w),BP_RR(n⊥))|, **상수**)  
- **의도**: 동조/결합 **안정도 힌트**  
- **효과**: 결합 강할수록 **신뢰↑**, 약하면 **오염 지표**  
- **능력**: **동적 게이팅/주의 전환**의 근거(현재 V1.6에선 상수채널)

**스택 순서(고정)**  
`[ w_rr, y_rr, d_rr, dw_rr, env_w, env_y, env_d, env_dw, cross_wy_rr, cross_wd_rr, env_low_y, env_high_y, w_trend, snr_rr_hint, corr_hint_wy, corr_hint_wd ]`

---

## 4) 손실 함수(설계 철학 → 구현)

### 4.1 메인: **z-MSE**  
- 창 내부 표준화(z) 후 MSE → **파형 형태**에 집중(**스케일/바이어스 제거**)  
- 장점: 도메인 변이에 덜 민감. 단점: **진폭 정보** 반영 약함 → 보조항으로 보완

### 4.2 보조1: **corr@soft-best-lag (미분가능)**  
- 라그 범위(±`LAG_MAX_S`) 내 상관을 계산 → **softmax(β)** 로 기대값 취해 `E[corr]` 산출  
- 손실: `λ·(1 − E[corr])`  
- **효과**: **위상/지연** 불일치에 **강건**. `corr_bestlag − corr` 격차 축소 목표

### 4.3 보조2: **스케일 패널티 |â−1|**  
- `â = ⟨P,G⟩ / ⟨P,P⟩` (최소자승 폐형식)  
- 손실: `α·|â−1|` (L1 권장)  
- **효과**: 평균 **진폭 과대/과소** 경향 교정 (`scale_a_hat_mean → 1.0`)

### 4.4 보조3: **엔벨로프 L1 (RMS)**  
- `env(x) = sqrt(avgpool1d(x², k=ENV_WIN_S·fs))`  
- 손실: `γ·L1(envP,envG)`  
- **효과**: **저SNR 창**에서 파형 평탄화/진폭 드리프트 억제, **에너지 정합** 강화

### 4.5 보조4: **분산(variance) 패널티**  
- 손실: `δ·|log(std(P)/std(G))|`  
- **효과**: **전역 에너지 스케일 직접 정합** → `std(P)≈std(G)` 유도, 진폭 보정 가속

### 4.6 (학습) **SNR-가중 손실**  
- 창별 `h∈[0,1]`(snr_rr_hint 평균)에 대해 **`loss *= (1−κ)+κ·h`**, 기본 `κ=0.30`  
- **효과**: **저SNR 창 과벌점 방지**·과도한 평탄화 완화

> **결합**:  
> `Loss = z-MSE + λ·(1−softcorr) + α·|â−1| + γ·L1(env) + δ·|log(std(P)/std(G))|`  
> *(학습 시 위 식을 per-sample로 계산 후 SNR-가중하여 평균)*

---

## 5) 평가(지표/알고리즘)

- **정렬**: 예측을 **스케일·부호 정렬** 후 지표 산출(align_scale)  
- **지표**: **MSE/MAE**, **corr**, **corr@soft-best-lag**, **RR-BPM**  
- **Welch-BPM**: 0.08–0.60 Hz 대역 PSD에서 prominence 기준 **피크 탐색**, 미검출 시 **대역 argmax 폴백**, 옵션: **서브-빈 포물선 보간**, **NFFT 업샘플**  
- 출력: `metrics.json`(val+test), `metrics_test_full.json`(테스트 창별), **오버레이 4장 PNG**  
- *훈련 종료 시 내부 평가도 동일한 Welch 옵션(BPM_FALLBACK_ARGMAX, BPM_SUBBIN_QUAD, BPM_NFFT_UP)을 사용하도록 코드에서 강제 통일되어 있습니다.*

---

## 6) 윈도우/샘플링/로더

- 입력 리샘플: **256 Hz**  
- 윈도우: **20 s**(T=5120), **40 s**(T=10240), stride = win×0.25 (또는 `FIXED_STRIDE`)  
- 로더: 길이별 **버킷 배치**(`10240:4, 5120:8`) → 메모리/속도 균형, 혼배치 스택 오류 방지  
- 분할: (기본) 60/20/20 (train/val/test) — *subject-wise split 옵션은 후속 버전 권장*

---

## 7) 훈련 안정화

- **AMP-호환 Gradient Clipping**(L2):  
  `backward → scaler.unscale_(opt) → clip_grad_norm_ → step`  
  기본 `GRAD_CLIP_NORM=1.0` (clip 비율 >30% → 2.0, <1% → 0.8)  
- **SNR-weighted Loss**: `w = (1−κ)+κ·h` (기본 `κ=0.30`, `h`는 snr_rr_hint 창 평균)  
- 옵티마이저: AdamW, AMP(autocast+GradScaler)

---

## 8) 실행 방법

### 8.1 요구사항 & 설치
```bash
pip install torch scipy numpy opencv-python mediapipe matplotlib
```

8.2 환경 변수(권장 프리셋)

```bash
# 고정 의도
export LOSS_MODE=corr
export PHASE_LAMBDA=0.3
export LAG_MAX_S=1.5
export PHASE_BETA=10.0

# 보조항/클리핑 (신규)
export SCALE_LAMBDA=0.15         # 0.12~0.15 탐색
export ENV_LAMBDA=0.06
export ENV_WIN_S=0.50
export GRAD_CLIP_NORM=2.0
export VAR_LAMBDA=0.05

# --- SNR 가중(창별 신뢰도) ---
export SNR_CH_IDX=13             # 입력 16채널 중 SNR 힌트 채널(창 내 상수)
export SNR_KAPPA=0.30            # 가중 강도: w=(1-κ)+κ·h ; κ↑→가중 차 커짐
export SNR_KAPPA_WARMUP=5        # (선택) 초기 5epoch 동안 κ 선형 워밍업(0이면 끔)
export SNR_WMIN=0.60             # 가중 하한(너무 나쁜 창도 최소 반영)
export SNR_WMAX=0.98             # 가중 상한(너무 좋은 창 과우대 방지)
export SNR_GAMMA=1.2             # h 비선형(>1이면 상단 눌러 포화 방지)

# --- SNR 힌트 산출 방식(데이터로더에서 창별 계산) ---
export SNR_MODE=crest            # crest | flat | topk  (기본: crest)
export SNR_CREST_LO=2.0          # crest 매핑 하한(나쁜 창 기준)
export SNR_CREST_HI=12.0         # crest 매핑 상한(아주 좋은 창 기준)
export SNR_TOPK_K=4              # SNR_MODE=topk일 때 상위 k 피크 사용
export SNR_SRC=w                 # SNR 소스: w|y|d|mix (기본 w=가로폭 채널, mix=0/1/2 평균)


# 윈도우/스트라이드
export RR_WIN_LIST="20,40"
export STRIDE_FRAC="0.25"
# export FIXED_STRIDE="2.0"

# 경로
export COHFACE_ROOT="/path/to/cohface"
export CACHE_DIR="cohface_exp_reg/cache_cohface_feats"
export RUNS_DIR="cohface_exp_reg/runs"

# 장치/스레드
export CUDA_VISIBLE_DEVICES=0
export CUDA_INDEX=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TORCH_NUM_THREADS=16

# Mediapipe(추출)
export MEDIAPIPE_GL_BACKEND=egl
export MEDIAPIPE_USE_GPU=1
export MP_TASK_PATH="$PWD/cohface_exp_reg/assets/pose_landmarker_full.task"

# BPM/컨텍스트
export BPM_MIN_PROM=3.0
export BPM_FALLBACK_ARGMAX=1
export BPM_SUBBIN_QUAD=1
export BPM_NFFT_UP=4
export SNR_HIT_BPM=2
export W_TREND_FC=0.05
# (선택) 전역 사전정렬
# export ENABLE_PREALIGN=1
# export PREALIGN_MAX_LAG=4.0
```

8.3 캐시 생성
```bash
mkdir -p cohface_exp_reg/assets
wget -O cohface_exp_reg/assets/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

python -m cohface_exp_reg.run_extract_all \
  --root /path/to/cohface \
  --subjects 1-40 --sessions 0-3 \
  --out cohface_exp_reg/cache_cohface_feats
```
    생성: cache_cohface_feats/s<subject>_k<session>.npz(키: t,dW,dY,dD,dD_perp,resp)

8.4 학습(VAL+TEST 저장)
```bash
python -m cohface_exp_reg.run_train_lstm \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --lr 5e-4 \
  --hidden 128 --layers 2 --bidir 1 --dropout 0.1 \
  --num_workers 12 --pin_memory 1
```
    결과: runs/<tag>/{best_model.pt, metrics.json} (metrics에 val+test 동시 기록)

8.5 평가/플롯(체크포인트 재사용)
```bash
# (선택) 이미 8.2에서 설정했다면 아래 3줄은 생략 가능
export BPM_FALLBACK_ARGMAX=1
export BPM_SUBBIN_QUAD=1
export BPM_NFFT_UP=4

python -m cohface_exp_reg.run_eval_best \
  --cache cohface_exp_reg/cache_cohface_feats \
  --model "cohface_exp_reg/runs/lstm_rronly_20250918_135929/best_model.pt" \
  --hidden 128 --layers 2 --bidir 1 --dropout 0.1 \
  --num_workers 12 --pin_memory 1 --n_plots 4 \
  --vis_norm minmax01
```
    출력: {run_dir}/metrics_test_full.json, {run_dir}/plots/test_sessionXX_[s**_k**.npz].png

---

9) 체크리스트(성공 판정)
```
    corr 상승 & (corr_bestlag − corr) 격차 축소 → 위상/지연 강건성 개선

    scale_a_hat_mean → 1.0 근접 → 진폭 보정 정상 동작

    RR-BPM MAE 안정 + 누락 적음(폴백·보간·업샘플)

    클리핑 개입률 정상(초반↑, 안정 후 1–30% 범위)
```
---

10) 트러블슈팅
```
    파형이 평탄: ENV_LAMBDA ↑(0.06→0.10), ENV_WIN_S 0.5–1.0 s 조정

    진폭 과대/과소 지속: SCALE_LAMBDA 0.12→0.15 단계적 ↑, VAR_LAMBDA 0.03→0.05 AB

    지연 흔들림: PHASE_BETA ↑(10→12), LAG_MAX_S 재검토(1.5–2.0 s)

    학습 불안정/NaN: GRAD_CLIP_NORM 2.0↑ 또는 LR↓, 클리핑 위치 확인

    BPM 누락: BPM_MIN_PROM 2.0–2.5로 ↓, BPM_NFFT_UP=4, BPM_SUBBIN_QUAD=1

    도메인 전이 취약: 학습/평가에서 사전정렬 정책 일치, subject-wise split 검토
```
---

11) 폴더 구조(요약)
```
cohface_exp_reg/
  config.py
  utils.py        # 필터/정규화/정렬/PSD-BPM + soft-corr(Torch/NumPy)
  pose_backend.py # Mediapipe 추출(TASKS GPU/CPU 폴백)
  preprocess.py   # 동기화/리샘플/캐시
  data.py         # 16ch 스택 & 윈도우
  models.py       # SeqRegressor(LSTM)
  sampler.py      # 20s/40s 버킷
  train.py        # 손실(SNR-가중 포함)/클리핑/평가(웰치 옵션 통일)
  run_extract_all.py
  run_train_lstm.py
  run_eval_best.py
  runs/, cache_cohface_feats/, assets/
```
---

12) 라이선스

    COHFACE 데이터 정책 및 Mediapipe / PyTorch / SciPy 라이선스 준수


---
13) 빠른 프리셋(한 줄)

```bash
export LOSS_MODE=corr; export PHASE_LAMBDA=0.3; export LAG_MAX_S=1.5; export PHASE_BETA=10.0; \
export SCALE_LAMBDA=0.12; export ENV_LAMBDA=0.06; export VAR_LAMBDA=0.03; export ENV_WIN_S=0.50; \
export GRAD_CLIP_NORM=1.0; export RR_WIN_LIST="20,40"; export STRIDE_FRAC=0.25; \
export BPM_MIN_PROM=3.0; export BPM_FALLBACK_ARGMAX=1; export BPM_SUBBIN_QUAD=1; export BPM_NFFT_UP=4; \
export SNR_HIT_BPM=2; export W_TREND_FC=0.05; export SNR_CH_IDX=13; export SNR_KAPPA=0.30
```