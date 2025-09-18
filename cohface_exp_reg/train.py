# -*- coding: utf-8 -*-
import json
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler

from .config import (
    DEVICE, EPOCHS, PATIENCE, FS_MODEL,
    PHASE_LAMBDA, PHASE_BETA, LAG_MAX_S, SNR_HIT_BPM,
    LOSS_MODE
)
from .utils import (
    corr_soft_bestlag, welch_psd_rr_bpm,
    align_scale_np, corr_soft_bestlag_torch
)

# ====================================================================================================
# === Loss auxiliaries & eval/unify (env overridable) ===
SCALE_LAMBDA = float(os.getenv("SCALE_LAMBDA", "0.10"))  # |â-1| 가중치
ENV_LAMBDA = float(os.getenv("ENV_LAMBDA", "0.06"))  # L1(envP,envG) 가중치
ENV_WIN_S = float(os.getenv("ENV_WIN_S", "0.50"))  # RMS envelope window (sec)
VAR_LAMBDA = float(os.getenv("VAR_LAMBDA", "0.02"))  # |log(std(P)/std(G))| 가중치

# SNR-weighted loss (저 SNR일수록 가중 완화)
SNR_CH_IDX = int(os.getenv("SNR_CH_IDX", "13"))  # 16ch 중 snr_rr_hint (0-based)
SNR_KAPPA = float(os.getenv("SNR_KAPPA", "0.30"))  # κ∈[0,1], weight = (1-κ)+κ*h
SNR_GAMMA = float(os.getenv("SNR_GAMMA", "1.0"))  # 비선형 강화: h^γ

# AMP grad clipping
GRAD_CLIP_NORM = float(os.getenv("GRAD_CLIP_NORM", "1.0"))

# Welch-BPM 옵션 (학습 평가에서도 run_eval_best와 통일)
BPM_FALLBACK_ARGMAX = int(os.getenv("BPM_FALLBACK_ARGMAX", "1"))
BPM_SUBBIN_QUAD = int(os.getenv("BPM_SUBBIN_QUAD", "1"))
BPM_NFFT_UP = int(os.getenv("BPM_NFFT_UP", "4"))


# ----------------------- Collate: make_batch -----------------------
def make_batch(ds, idxs):
    """
    ds.sessions[sid]['X']:[T,16], ds.sessions[sid]['Y']:[T]
    idxs: list of tuples (sid, a, b, T)  # run_*에서 build_loaders_by_session이 생성
    return X:[B,T,16], Y:[B,T,1]
    """
    Xs, Ys = [], []
    for (sid, a, b, T) in idxs:
        X = ds.sessions[sid]["X"][a:b]  # [T,16]
        Y = ds.sessions[sid]["Y"][a:b]  # [T]
        Xs.append(torch.from_numpy(X))
        Ys.append(torch.from_numpy(Y).unsqueeze(-1))
    X = torch.stack(Xs, 0).to(torch.float32)
    Y = torch.stack(Ys, 0).to(torch.float32)
    return X, Y


# ----------------------- Aux losses (vectorized per-sample) -------------------------------
def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def phase_loss_spectral(pred, gold, fs, band=(0.08, 0.60), eps=1e-8, pad_pow2=True):
    """
    Spectral phase alignment loss (batch scalar):
      L = 1 - Σ_k w_k * cos(Δφ_k),  w_k ∝ |X_k|·|Y_k| (normalized in band)
    pred, gold: [B,T,1] or [B,T]
    """
    use_cuda = (pred.is_cuda or gold.is_cuda)
    ctx = torch.amp.autocast('cuda', enabled=False) if use_cuda else nullcontext()
    with ctx:
        p = pred.squeeze(-1) if (pred.dim() == 3 and pred.size(-1) == 1) else pred
        g = gold.squeeze(-1) if (gold.dim() == 3 and gold.size(-1) == 1) else gold
        p = p.float()
        g = g.float()
        B, T = p.shape
        N = _next_pow2(T) if pad_pow2 else T
        if N != T:
            pad = (0, N - T)
            p = F.pad(p, pad)
            g = F.pad(g, pad)
        X = torch.fft.rfft(p, dim=-1)
        Y = torch.fft.rfft(g, dim=-1)
        freqs = torch.fft.rfftfreq(N, d=1.0 / fs).to(X.device)
        lo, hi = band
        m = (freqs >= lo) & (freqs <= hi)
        Xb = X[:, m]
        Yb = Y[:, m]
        W = (torch.abs(Xb) * torch.abs(Yb)).clamp_min(1e-8)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-8)
        dphi = torch.angle(Xb) - torch.angle(Yb)
        loss = 1.0 - torch.sum(W * torch.cos(dphi), dim=1)
        return loss.mean()


def _mse_z_vec(pred: torch.Tensor, gold: torch.Tensor, eps: float = 1e-6):
    """per-sample z-MSE → [B]"""
    P = (pred - pred.mean(dim=1, keepdim=True)) / (pred.std(dim=1, keepdim=True) + eps)
    G = (gold - gold.mean(dim=1, keepdim=True)) / (gold.std(dim=1, keepdim=True) + eps)
    return ((P - G) ** 2).mean(dim=1)


def _scale_penalty_vec(pred: torch.Tensor, gold: torch.Tensor, eps: float = 1e-6, p: int = 1):
    """|â-1| per-sample → ([B], â:[B])"""
    den = (pred * pred).sum(dim=1) + eps
    num = (pred * gold).sum(dim=1)
    a_hat = num / den  # [B]
    loss = (a_hat - 1.0).abs() if p == 1 else (a_hat - 1.0).pow(2)
    return loss, a_hat


def _rms_envelope(x: torch.Tensor, win_samples: int):
    """RMS envelope → [B,T]"""
    k = max(3, int(win_samples))
    x2 = (x.unsqueeze(1) ** 2)  # [B,1,T]
    env = torch.sqrt(F.avg_pool1d(x2, kernel_size=k, stride=1, padding=k // 2) + 1e-8)
    return env.squeeze(1)  # [B,T]


def _env_l1_vec(pred: torch.Tensor, gold: torch.Tensor, fs: float, win_s: float):
    """L1(envP, envG) per-sample → [B]"""
    env_win = max(3, int(win_s * fs))
    envP, envG = _rms_envelope(pred, env_win), _rms_envelope(gold, env_win)
    return torch.mean(torch.abs(envP - envG), dim=1)


def _var_penalty_vec(pred: torch.Tensor, gold: torch.Tensor, eps: float = 1e-6, p: int = 1):
    """|log(std(P)/std(G))| per-sample → ([B], ratio:[B])"""
    std_p = pred.std(dim=1) + eps
    std_g = gold.std(dim=1) + eps
    ratio = std_p / std_g
    log_r = torch.log(ratio)
    loss = log_r.abs() if p == 1 else (log_r ** 2)
    return loss, ratio


# ----------------------- Evaluation -------------------------------
@torch.no_grad()
def evaluate(model, loader, fs=FS_MODEL):
    """학습 중/종료 후 내부 평가. Welch 옵션은 환경변수에서 통일 적용."""
    # Welch 옵션 통일(함수 내부에서 env를 읽으므로, 안전하게 동일값 강제)
    os.environ["BPM_FALLBACK_ARGMAX"] = str(BPM_FALLBACK_ARGMAX)
    os.environ["BPM_SUBBIN_QUAD"] = str(BPM_SUBBIN_QUAD)
    os.environ["BPM_NFFT_UP"] = str(BPM_NFFT_UP)

    model.eval()
    loaders = loader if isinstance(loader, (list, tuple)) else [loader]

    mse_list, mae_list, corr_vals, corr_bl_vals = [], [], [], []
    bpm_mae_vals, hit_vals, a_hats, valid_bpm = [], [], [], 0

    for ld in loaders:
        for X, Y in ld:
            X = X.to(DEVICE).float()
            Y = Y.to(DEVICE).float()
            P = model(X).squeeze(-1)  # [B,T]
            G = Y.squeeze(-1)
            B, T = P.shape
            for b in range(B):
                pb = P[b].detach().cpu().numpy().astype(np.float32)
                gb = G[b].detach().cpu().numpy().astype(np.float32)
                # scale-align 후 지표
                pb_aligned, a_hat = align_scale_np(pb, gb)
                a_hats.append(a_hat)
                mse_list.append(float(np.mean((pb_aligned - gb) ** 2)))
                mae_list.append(float(np.mean(np.abs(pb_aligned - gb))))
                # corr / soft-best-lag corr
                c = np.corrcoef(pb_aligned, gb)[0, 1]
                if not np.isfinite(c):
                    c = 0.0
                corr_vals.append(float(c))
                c2, _ = corr_soft_bestlag(pb_aligned, gb, fs=fs, lag_s=LAG_MAX_S, beta=PHASE_BETA)
                corr_bl_vals.append(float(c2))
                # BPM
                pbpm = welch_psd_rr_bpm(pb_aligned, fs)
                gbpm = welch_psd_rr_bpm(gb, fs)
                if not (np.isnan(pbpm) or np.isnan(gbpm)):
                    bpm_mae_vals.append(float(abs(pbpm - gbpm)))
                    hit_vals.append(float(abs(pbpm - gbpm) <= SNR_HIT_BPM))
                    valid_bpm += 1

    _m = (lambda x: float(np.mean(x)) if len(x) else float("nan"))
    out = {
        "mse": _m(mse_list),
        "mae": _m(mae_list),
        "corr": _m(corr_vals),
        "corr_bestlag": _m(corr_bl_vals),
        "rr_bpm_mae": _m(bpm_mae_vals),
        f"hit@±{SNR_HIT_BPM}bpm": _m(hit_vals),
        "scale_a_hat_mean": _m(a_hats),
        "valid_bpm_windows": int(valid_bpm),
        "num_windows_scored": int(len(mse_list)),
        # 편의상, 사용된 Welch 옵션도 기록
        "bpm_opts": {
            "FALLBACK_ARGMAX": BPM_FALLBACK_ARGMAX,
            "SUBBIN_QUAD": BPM_SUBBIN_QUAD,
            "NFFT_UP": BPM_NFFT_UP,
        },
    }
    return out


# ----------------------- Train loop -------------------------------
def _normalize_train_val_inputs(train_loaders, val_loaders):
    """train_loaders:[DataLoader,...] 또는 단일 DataLoader
       val_loaders:{name:DataLoader,...} 또는 [DataLoader,...] 또는 단일 DataLoader"""
    # train
    if isinstance(train_loaders, (list, tuple)):
        t_list = list(train_loaders)
    else:
        t_list = [train_loaders]
    # val
    if isinstance(val_loaders, dict):
        v_dict = val_loaders
    elif isinstance(val_loaders, (list, tuple)):
        v_dict = {f"val_{i}": ld for i, ld in enumerate(val_loaders)}
    else:
        v_dict = {"val": val_loaders}
    return t_list, v_dict


def train_loop(model, optimizer, train_loaders, val_loaders, epochs=EPOCHS, device=DEVICE):
    """
    - 'corr' : loss = z-MSE + λ * (1 - corr_soft_bestlag_torch) + α*|â-1| + γ*L1(env) + δ*|log(stdP/stdG)|
               (모두 per-sample로 계산 후 SNR 가중 적용)
    - 'phase': loss = SI-MSE + λ * spectral phase loss (+ 선택: env/var)
    - early-stop: val.corr_bestlag 기준
    """

    scaler = GradScaler()
    best_state, best_metric, patience = None, -1e9, 0

    # train/val 입력 정규화
    t_loaders, v_dict = _normalize_train_val_inputs(train_loaders, val_loaders)
    is_cuda = str(device).startswith('cuda')

    for ep in range(1, int(epochs) + 1):
        model.train()
        loss_mse_list, loss_aux_list, loss_tot_list = [], [], []

        # 로깅 누적
        a_hat_acc, var_ratio_acc, snr_w_acc = 0.0, 0.0, 0.0
        batch_count, clip_events = 0, 0

        # -------------------- Train --------------------
        for ld in t_loaders:
            for X, Y in ld:
                X = X.to(device).float()
                Y = Y.to(device).float()
                optimizer.zero_grad(set_to_none=True)

                # 창별 SNR 힌트(0~1) → 저 SNR일수록 weight 낮춤: w = (1-κ) + κ*h
                h_win = torch.clamp(X[:, :, SNR_CH_IDX].mean(dim=1), 0.0, 1.0)  # [B]
                h_eff = torch.pow(h_win, SNR_GAMMA)  # 비선형 옵션
                w_snr = (1.0 - SNR_KAPPA) + SNR_KAPPA * h_eff  # [B]

                if LOSS_MODE == 'phase':
                    # FFT 기반 손실은 fp16에서 불안정 → autocast 비활성 (fp32)
                    with torch.autocast(device_type=('cuda' if is_cuda else 'cpu'),
                                        dtype=torch.float16, enabled=False):
                        P = model(X).squeeze(-1)  # [B,T]
                        G = Y.squeeze(-1)

                        # 메인/보조 (per-sample 벡터화)
                        # 간단 근사: si-MSE ≈ z-MSE
                        si_mse_vec = _mse_z_vec(P, G)
                        phase_loss = phase_loss_spectral(P, G, fs=FS_MODEL, band=(0.08, 0.60))
                        phase_vec = phase_loss * torch.ones_like(si_mse_vec)

                        # 선택 보조항: env/var 추가(가벼운 정규화)
                        env_vec = _env_l1_vec(P, G, FS_MODEL, ENV_WIN_S)
                        var_vec, var_ratio = _var_penalty_vec(P, G, p=1)

                        loss_vec = si_mse_vec \
                                   + PHASE_LAMBDA * phase_vec \
                                   + ENV_LAMBDA * env_vec \
                                   + VAR_LAMBDA * var_vec

                        loss = torch.mean(w_snr * loss_vec)
                        loss_main = si_mse_vec.mean()
                        loss_aux_total = (PHASE_LAMBDA * phase_vec
                                          + ENV_LAMBDA * env_vec + VAR_LAMBDA * var_vec).mean()

                        # 로깅용
                        a_hat = torch.tensor(float('nan'), device=P.device)  # phase 모드에선 스케일 패널티 없음

                else:  # 'corr'
                    with torch.autocast(device_type=('cuda' if is_cuda else 'cpu'),
                                        dtype=torch.float16, enabled=is_cuda):
                        P = model(X).squeeze(-1)  # [B,T]
                        G = Y.squeeze(-1)

                        # (1) 파형 정합
                        mse_vec = _mse_z_vec(P, G)  # [B]
                        # (2) 위상/지연 강건
                        c_soft, _ = corr_soft_bestlag_torch(P, G, fs=FS_MODEL, lag_s=LAG_MAX_S, beta=PHASE_BETA)
                        corr_vec = (1.0 - c_soft)  # [B]
                        # (3) 스케일 보정
                        scale_vec, a_hat = _scale_penalty_vec(P, G, p=1)  # [B], [B]
                        # (4) 엔벨로프 정합
                        env_vec = _env_l1_vec(P, G, FS_MODEL, ENV_WIN_S)  # [B]
                        # (5) 분산 정합
                        var_vec, var_ratio = _var_penalty_vec(P, G, p=1)  # [B], [B]

                        # SNR 가중 결합
                        loss_vec = mse_vec \
                                   + PHASE_LAMBDA * corr_vec \
                                   + SCALE_LAMBDA * scale_vec \
                                   + ENV_LAMBDA * env_vec \
                                   + VAR_LAMBDA * var_vec

                        loss = torch.mean(w_snr * loss_vec)
                        loss_main = mse_vec.mean()
                        loss_aux_total = (PHASE_LAMBDA * corr_vec + SCALE_LAMBDA * scale_vec
                                          + ENV_LAMBDA * env_vec + VAR_LAMBDA * var_vec).mean()

                # === backward + AMP-safe grad clipping ===
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if torch.isfinite(total_norm) and float(total_norm) > GRAD_CLIP_NORM:
                    clip_events += 1
                scaler.step(optimizer)
                scaler.update()

                # 누적 로깅
                loss_mse_list.append(float(loss_main.detach().cpu()))
                loss_aux_list.append(float(loss_aux_total.detach().cpu()))
                loss_tot_list.append(float(loss.detach().cpu()))
                if torch.is_tensor(a_hat):
                    a_hat_acc += float(torch.nanmean(a_hat).detach().cpu())
                if torch.is_tensor(var_ratio):
                    var_ratio_acc += float(torch.nanmean(var_ratio).detach().cpu())
                snr_w_acc += float(w_snr.mean().detach().cpu())
                batch_count += 1

        # -------------------- Validate --------------------
        with torch.no_grad():
            vouts = []
            for name, vld in v_dict.items():
                metrics = evaluate(model, vld, fs=FS_MODEL)  # dict 반환
                vouts.append(metrics)

        # 평균 지표 집계
        keys = set().union(*[m.keys() for m in vouts]) if len(vouts) else set()

        val_avg = {}
        for k in keys:
            # dict(예: bpm_opts)는 평균에서 제외
            vals = [m[k] for m in vouts if (k in m and not isinstance(m[k], dict))]
            if len(vals):
                val_avg[k] = float(np.nanmean(vals))

        # 메타데이터는 그대로 한 개만 보존
        for m in reversed(vouts):
            if "bpm_opts" in m:
                val_avg["bpm_opts"] = m["bpm_opts"]
                break

        clip_rate = (clip_events / max(1, batch_count)) * 100.0
        print(f"[epoch {ep:03d}] "
              f"train_main={np.mean(loss_mse_list):.6e} | "
              f"train_aux={np.mean(loss_aux_list):.6e} | "
              f"train_total={np.mean(loss_tot_list):.6e} | "
              f"a_hat_mean={a_hat_acc / max(1, batch_count):.3f} | "
              f"var_ratio_mean={var_ratio_acc / max(1, batch_count):.3f} | "
              f"snr_weight_mean={snr_w_acc / max(1, batch_count):.3f} | "
              f"clip_rate={clip_rate:.1f}% | "
              f"val_avg_corr_bestlag={val_avg.get('corr_bestlag', float('nan')):.6f}")

        # Early stop 기준: corr_bestlag
        key = val_avg.get('corr_bestlag', -1e9)
        if key > best_metric:
            best_metric = key
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("[early stop]")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------------- Save -------------------------------
def save_run(run_dir, model, metrics):
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
