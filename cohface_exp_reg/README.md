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

## 4) # RR 전용 16채널 설명서 (의도 · 포착 · 효과 · 능력)

> **결론**
> 16채널은 **(A) 파형(위상 민감) + (B) 에너지(위상 불변) + (C) 결합/서브밴드(양상 분해) + (D) 느린 컨텍스트(신뢰/게이트)** 로 역할을 분리해, **자세·구도·개인별 위상차**에 강하고 **corr\_rr↑ & rr\_bpm\_mae 유지**를 목표로 설계했습니다.

> **공통 전제(계산 규칙 요약)**
>
> * `W0=median(dW)`, `Wslow=LPF(dW, fc≈0.03Hz)`
> * **n⊥(dD\_perp)**: θ 언랩→저역필터(0.03–0.05Hz)로 얻은 `v_slow`에 `(N−M)` 투영
> * `BP_RR(x)=bandpass(x, 0.08–0.60Hz)` → `z-score`
> * `env(x)= z(|Hilbert(x)|)` *(모바일: rectified RMS+LPF로 대체 가능)*
> * **D그룹**은 RR 밴드 금지, `LPF(<0.3Hz)→z`만 적용

---

## A. RR **원파형** (위상 민감; BP\_RR→z)

1. **w\_rr** = z(BP\_RR(dW/W0 − 1))

* **포착**: 흉곽의 가로 폭 변화(어깨폭) **순수 파형**
* **효과**: 흉식 호흡에서 **SNR 높은 RR** 확보
* **능력**: 체격 스케일 제거로 **개인 간 진폭 차**에 둔감

2. **y\_rr** = z(BP\_RR(dY/(Wslow+ε)))

* **포착**: 어깨중점의 **세로 들썩임** 파형(복식·흉식 공통)
* **효과**: 가로 신호(w)와 **보완적 위상** 제공(상쇄 완화)
* **능력**: 카메라 줌/거리 변화에 **스케일 불변**

3. **d\_rr** = z(BP\_RR(dD\_perp/(Wslow+ε)))  ※ **n⊥**

* **포착**: 코–어깨중점 벡터의 **어깨선 수직 성분** 파형
* **효과**: 전역 y축 대신 **몸축 기준** → **roll/시점 변화**에 강함
* **능력**: **데이터셋/자세 전이**에서 **corr 유지** 기여

4. **dw\_rr** = z(BP\_RR((d/dt dW)/W0))

* **포착**: 가로 확장의 **속도**(시간 미분) 파형
* **효과**: w\_rr 대비 **≈90° 위상 보강** → 피크 정렬 용이
* **능력**: **가쁜 호흡**에서 전이(edge) 강조로 **검출 민감도**↑

---

## B. **에너지/엔벨로프** (위상 불변; |Hilbert|→z)

5. **env\_w** = env(BP\_RR(dW/W0 − 1))

* **포착**: w\_rr의 **크기/세기(에너지)**
* **효과**: 위상차에 **불변** → 개인별 위상차 커도 안정
* **능력**: **soft-alignment 손실**과 시너지로 corr 상승

6. **env\_y** = env(BP\_RR(dY/(Wslow+ε)))

* **포착**: y\_rr의 에너지
* **효과**: 자세 전환·복식/흉식 변화의 **진폭 변동**을 견고 반영
* **능력**: **진폭 드리프트**에 둔감한 RR 강도 피처

7. **env\_d** = env(BP\_RR(dD\_perp/(Wslow+ε)))

* **포착**: n⊥ 파형의 에너지
* **효과**: 머리 상하 **기계적 추종 강도** 정보 추가
* **능력**: **머리 흔들림/말하기** 등 잡동작의 영향 구분

8. **env\_dw** = env(BP\_RR((d/dt dW)/W0))

* **포착**: 확장 속도 성분의 에너지
* **효과**: **호흡이 빠를수록** 강하게 반응 → 고주기 보강
* **능력**: **운동성 상황**에서도 RR 감지 유지

---

## C. **결합/서브밴드** (양상 분해; RR대역 곱→RR BP→z)

9. **cross\_wy\_rr** = z(BP\_RR( BP\_RR(w) \* BP\_RR(y) ))

* **포착**: 가로(w)–세로(y) **동시 변동/동조**
* **효과**: 흉식·복식 **혼합 양식**에서 상쇄/혼탁 완화
* **능력**: **호흡 방식 변화**에 대한 **일반화**↑

10. **cross\_wd\_rr** = z(BP\_RR( BP\_RR(w) \* BP\_RR(n⊥) ))

* **포착**: 가로(w)–머리 수직(n⊥) **결합**
* **효과**: 흉곽–머리 **기계적 연결 강도** 파악 → 말하기/고개짓 구분
* **능력**: **비호흡성 움직임** 필터링에 기여

11. **env\_low\_y** = env(bandpass(dY/(Wslow+ε), 0.08–0.25Hz))

* **포착**: **느린 호흡(저주기)** 에너지
* **효과**: 64 s 같은 긴 윈도우에서 **분해능 확보**
* **능력**: **휴식/수면 호흡**에도 안정 추정

12. **env\_high\_y** = env(bandpass(dY/(Wslow+ε), 0.25–0.60Hz))

* **포착**: **빠른 호흡(고주기)** 에너지
* **효과**: 짧은 보조 윈도우와 **상보** → 과소추정 방지
* **능력**: **운동/긴장** 상황에서 RR 추정 유지

---

## D. **느린 컨텍스트** (LPF<0.3Hz→z; RR 밴드 금지)

13. **w\_trend** = z(LPF(dW/W0 − 1, fc≈0.2Hz))

* **포착**: 가로 폭의 **저역 트렌드**(자세 드리프트, 카메라 요동)
* **효과**: 오염 구간 **자체 게이팅** 근거 제공
* **능력**: **거짓 상관 상승** 방지(강건성↑)

14. **snr\_rr\_hint** = z(RR-SNR prominence ∈\[0,1], 창 내 상수)

* **포착**: 윈도우의 **RR 신뢰도/피크 우세도**
* **효과**: **저SNR** 샘플의 과신 억제(가중·어텐션에 활용)
* **능력**: **샘플 선택/가중 학습**으로 일반화↑

15. **corr\_hint\_wy** = z(|corr(BP\_RR(w), BP\_RR(y))|, 상수)

* **포착**: w↔y **동조도**(창 단위 절대 상관)
* **효과**: 동조 높을수록 **신뢰↑**, 낮으면 **오염 신호** 지표
* **능력**: **동적 게이팅/주의 전환**에 사용 가능

16. **corr\_hint\_wd** = z(|corr(BP\_RR(w), BP\_RR(n⊥))|, 상수)

* **포착**: w↔n⊥ **결합 안정도**
* **효과**: 머리–흉곽 상호작용이 안정적일 때 **추정 신뢰** 향상
* **능력**: **머리 흔들림/발화** 등 비호흡성 인공물에 내성

---

### 스택 순서(고정)

`[ w_rr, y_rr, d_rr, dw_rr,
env_w, env_y, env_d, env_dw,
cross_wy_rr, cross_wd_rr, env_low_y, env_high_y,
w_trend, snr_rr_hint, corr_hint_wy, corr_hint_wd ]`

**요약 한 줄**
- **A(1–4)**: RR **파형 핵심**
- **B(5–8)**: **위상 불변 강도**
- **C(9–12)**: **양식/대역 분해 & 결합**
- **D(13–16)**: **신뢰/게이트 컨텍스트**

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

# ==== (신규) 보조항/클리핑 ====
export SCALE_LAMBDA=0.05      # |â-1| 가중치 (0.03~0.1 튜닝)
export ENV_LAMBDA=0.05        # 엔벨로프 L1 가중치 (0.03~0.1 튜닝)
export ENV_WIN_S=0.75         # RMS 엔벨로프 윈도우(초, 0.5~1.0 권장)
export GRAD_CLIP_NORM=1.0     # grad clip L2 norm (0.8~2.0 튜닝)

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
export BPM_FALLBACK_ARGMAX=1
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
- FULL 지표: {run_dir}/metrics_test_full.json   (전 창 evaluate 결과)
- 샘플 플롯: {run_dir}/plots/test_sessionXX_[s**_k**.npz].png

> 특정 사람(세션)을 지정하고 싶으면 `run_eval_best.py`의 선택 로직을 `sX_kY` 필터로 바꿔드릴 수 있습니다.

---

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