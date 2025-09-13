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
export CUDA_VISIBLE_DEVICES=0
export CUDA_INDEX=0
export MEDIAPIPE_GL_BACKEND=egl
export MEDIAPIPE_USE_GPU=1
export MP_TASK_PATH="$PWD/cohface_exp_reg/assets/pose_landmarker_full.task"

# (선택) 위치/윈도우
export COHFACE_ROOT="/home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/cohface"
export CACHE_DIR="cohface_exp_reg/cache_cohface_feats"
export RUNS_DIR="cohface_exp_reg/runs"
export RR_WIN_LIST="24,32,48,64,96"
export STRIDE_FRAC="0.25"
# export FIXED_STRIDE="2.0"

# 손실 하이퍼
export PHASE_LAMBDA="0.3"
export PHASE_BETA="6.0"
export LAG_MAX_S="0.5"
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
```bash
python -m cohface_exp_reg.run_train_lstm \
  --cache cohface_exp_reg/cache_cohface_feats \
  --epochs 50 --lr 1e-3 --hidden 128 --layers 2 --bidir 1 --dropout 0.1
```
- 출력: `runs/lstm_rronly_<timestamp>/best_model.pt, metrics.json`
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
- 멀티스케일 윈도우: **24,32,48,64,96s**, stride=윈도우×0.25(기본)
- RNN(양방향 LSTM)로 RR 파형 회귀
- **손실**: `MSE + λ*(1 - corr@soft-best-lag)`

---

## 윈도우 조합 근거
- **32·64s**: 일반 성인 호흡(10–20 bpm, 주기 3–6 s)을 **≥8–10주기** 포함.
- **24s**: 빠른 호흡(>25 bpm) 커버(주기~2.4 s → 10주기).
- **96s**: 느린 호흡(~5–7 bpm, 8–12 s 주기)을 다회 주기로 안정 추정.
- **48s**: 32↔64 사이 중간 스케일로 파형 정합 보조.

---

## 트러블슈팅
- `.task` 미인식 → `MP_TASK_PATH` 확인
- CPU 폴백 강제 → `export MEDIAPIPE_USE_GPU=0`
- VRAM OOM → `RR_WIN_LIST`에서 96s 제거 또는 `BUCKET_BS`에서 24576:1 유지
