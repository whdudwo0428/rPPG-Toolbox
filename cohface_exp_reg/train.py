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

# ---- New: 보조항/클리핑 하이퍼파라미터 ----
SCALE_LAMBDA    = float(os.getenv("SCALE_LAMBDA", "0.05"))  # 스케일 패널티 가중치
ENV_LAMBDA      = float(os.getenv("ENV_LAMBDA",   "0.05"))  # 엔벨로프 MSE 가중치
ENV_WIN_S       = float(os.getenv("ENV_WIN_S",    "0.75"))  # RMS 엔벨로프 윈도우(초)
GRAD_CLIP_NORM  = float(os.getenv("GRAD_CLIP_NORM", "1.0"))  # 클리핑 노름

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

# ----------------------- Aux losses -------------------------------
def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def phase_loss_spectral(pred, gold, fs, band=(0.08, 0.60), eps=1e-8, pad_pow2=True):
    """
    Spectral phase alignment loss:
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

def _si_mse(pred, gold, eps=1e-6):
    p = pred; g = gold
    den = torch.sum(p * p, dim=1, keepdim=True) + eps
    num = torch.sum(p * g, dim=1, keepdim=True)
    a_hat = num / den
    p_aligned = a_hat * p
    return torch.mean((p_aligned - g) ** 2)

def _mse_z(pred, gold):
    p = pred - pred.mean(dim=1, keepdim=True)
    p = p / (p.std(dim=1, keepdim=True) + 1e-6)
    g = gold - gold.mean(dim=1, keepdim=True)
    g = g / (g.std(dim=1, keepdim=True) + 1e-6)
    return torch.mean((p - g) ** 2)

def _scale_penalty(pred, gold, eps=1e-6, p=1):
    """
    a_hat = argmin_a ||a·pred - gold||^2  →  a_hat = <p,g>/<p,p>
    반환: L_scale = |a_hat-1| 의 (평균) (p=1: L1, p=2: L2), a_hat
    """
    den = torch.sum(pred * pred, dim=1, keepdim=True) + eps
    num = torch.sum(pred * gold, dim=1, keepdim=True)
    a_hat = num / den
    if p == 1:
        loss_scale = (a_hat - 1.0).abs().mean()
    else:
        loss_scale = ((a_hat - 1.0) ** 2).mean()
    return loss_scale, a_hat

def _rms_envelope(x, win_samples: int):
    """
    RMS 엔벨로프: sqrt( avgpool1d(x^2, k=win) )
    x: [B,T]
    """
    k = max(3, int(win_samples))
    x2 = (x.unsqueeze(1) ** 2)  # [B,1,T]
    env = torch.sqrt(F.avg_pool1d(x2, kernel_size=k, stride=1, padding=k // 2) + 1e-8)
    return env.squeeze(1)  # [B,T]


# ----------------------- Evaluation -------------------------------
@torch.no_grad()
def evaluate(model, loader, fs=FS_MODEL):
    model.eval()
    # loader는 단일 DataLoader 또는 [DataLoader, ...]일 수 있음 → 단일로 정규화
    loaders = loader if isinstance(loader, (list, tuple)) else [loader]

    mse_list, mae_list, corr_vals, corr_bl_vals = [], [], [], []
    bpm_mae_vals, hit_vals, a_hats, valid_bpm = [], [], [], 0

    for ld in loaders:
        for X, Y in ld:
            X = X.to(DEVICE).float()
            Y = Y.to(DEVICE).float()
            P = model(X)  # [B,T,1]
            P = P.squeeze(-1)
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
                c2, _lag = corr_soft_bestlag(pb_aligned, gb, fs=fs, lag_s=LAG_MAX_S, beta=PHASE_BETA)
                corr_bl_vals.append(float(c2))
                # BPM
                pbpm = welch_psd_rr_bpm(pb_aligned, fs)
                gbpm = welch_psd_rr_bpm(gb, fs)
                if not (np.isnan(pbpm) or np.isnan(gbpm)):
                    bpm_mae_vals.append(float(abs(pbpm - gbpm)))
                    hit_vals.append(float(abs(pbpm - gbpm) <= SNR_HIT_BPM))
                    valid_bpm += 1

    def _m(x):
        return float(np.mean(x)) if len(x) else float("nan")

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
    - 'corr' : loss = z-MSE + λ * (1 - corr_soft_bestlag_torch)
    - 'phase': loss = SI-MSE + λ * spectral phase loss  (fp32 권장)
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

        # -------------------- Train --------------------
        for ld in t_loaders:
            for X, Y in ld:
                X = X.to(device).float()
                Y = Y.to(device).float()
                optimizer.zero_grad(set_to_none=True)

                if LOSS_MODE == 'phase':
                    # FFT 기반 손실은 fp16에서 불안정 → autocast 비활성 (fp32)
                    with torch.autocast(device_type=('cuda' if is_cuda else 'cpu'),
                                        dtype=torch.float16, enabled=False):
                        P = model(X).squeeze(-1)  # [B,T]
                        G = Y.squeeze(-1)
                        loss_main = _si_mse(P, G)
                        loss_aux  = PHASE_LAMBDA * phase_loss_spectral(
                            P, G, fs=FS_MODEL, band=(0.08, 0.60)
                        )
                        loss = loss_main + loss_aux
                        loss_aux_total = loss_aux
                else:  # 'corr'
                    with torch.autocast(device_type=('cuda' if is_cuda else 'cpu'),
                                        dtype=torch.float16, enabled=is_cuda):
                        P = model(X).squeeze(-1)  # [B,T]
                        G = Y.squeeze(-1)

                        # (1) 기본: 파형 정합
                        loss_main = _mse_z(P, G)

                        # (2) 위상/지연 강건: soft-best-lag corr (미분가능)
                        c_soft, _ = corr_soft_bestlag_torch(P, G, fs=FS_MODEL, lag_s=LAG_MAX_S, beta=PHASE_BETA)
                        loss_corr = PHASE_LAMBDA * (1.0 - c_soft.mean())

                        # (3) 진폭 보정: 스케일 패널티 |a_hat - 1|
                        loss_scale, a_hat = _scale_penalty(P, G, p=1)  # L1 권장

                        # (4) 에너지(엔벨로프) 매칭: RMS envelope L1
                        env_win = max(3, int(ENV_WIN_S * FS_MODEL))
                        envP = _rms_envelope(P, env_win)
                        envG = _rms_envelope(G, env_win)
                        loss_env = F.l1_loss(envP, envG)

                        # 최종 손실 결합
                        loss_aux_total = loss_corr + SCALE_LAMBDA * loss_scale + ENV_LAMBDA * loss_env
                        loss = loss_main + loss_aux_total

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # 스케일 해제 → 실제 grad
                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                scaler.step(optimizer)
                scaler.update()

                loss_mse_list.append(float(loss_main.detach().cpu()))
                loss_aux_list.append(float((loss_aux_total if LOSS_MODE != 'phase' else loss_aux).detach().cpu()))
                loss_tot_list.append(float(loss.detach().cpu()))

        # -------------------- Validate --------------------
        with torch.no_grad():
            vouts = []
            for name, vld in v_dict.items():
                metrics = evaluate(model, vld, fs=FS_MODEL)  # dict 반환
                vouts.append(metrics)

        # 평균 지표 집계(키가 없을 수 있으니 안전하게 처리)
        keys = set().union(*[m.keys() for m in vouts]) if len(vouts) else set()
        val_avg = {k: float(np.nanmean([m[k] for m in vouts if k in m])) for k in keys}

        print(f"[epoch {ep:03d}] "
              f"train_main={np.mean(loss_mse_list):.6e} | "
              f"train_aux={np.mean(loss_aux_list):.6e} | "
              f"train_total={np.mean(loss_tot_list):.6e} | "
              f"val_avg_corr_bestlag={val_avg.get('corr_bestlag', float('nan')):.6f} | "
              f"val={{...}}")

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