# -*- coding: utf-8 -*-
import json
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler

# === train.py: evaluate() 교체 ===
from .config import (DEVICE, EPOCHS, PATIENCE, FS_MODEL,
                     PHASE_LAMBDA, PHASE_BETA, LAG_MAX_S, SNR_HIT_BPM,
                     BPM_MIN_PROM)
from .utils import corr_soft_bestlag, welch_psd_rr_bpm

torch.backends.cuda.matmul.allow_tf32 = True  # Ampere(3060 Ti) TF32 허용
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # matmul 성능 ↑
torch.backends.cudnn.benchmark = True  # 입력 길이 고정 시 conv/RNN 선택 최적화


def si_mse_loss(pred, gold, eps=1e-8):
    # [B,T,1] 또는 [B,T]
    if pred.dim() == 3: pred = pred[..., 0]
    if gold.dim() == 3: gold = gold[..., 0]
    num = (pred * gold).sum(dim=-1, keepdim=True)
    den = (pred * pred).sum(dim=-1, keepdim=True) + eps
    a = num / den
    err = a * pred - gold
    return (err * err).mean()


def amp_penalty_loss(pred, gold, eps=1e-8, k_std=0.1):
    if pred.dim() == 3: pred = pred[..., 0]
    if gold.dim() == 3: gold = gold[..., 0]
    num = (pred * gold).sum(dim=-1)
    den = (pred * pred).sum(dim=-1) + eps
    a_hat = torch.clamp(num / den, -100.0, 100.0)
    std_p = pred.std(dim=-1) + eps
    std_g = gold.std(dim=-1) + eps
    loss_scale = (a_hat - 1.0).pow(2)
    loss_std   = (torch.log(std_p / std_g)).pow(2)
    return (loss_scale + k_std * loss_std).mean()


def _next_pow2(n: int) -> int:
    if n <= 1: return 1
    return 1 << (n - 1).bit_length()


def phase_loss_spectral(pred, gold, fs, band=(0.08, 0.60), eps=1e-8, pad_pow2=True):
    """
    위상 정합 손실(주파수 영역):
      L = 1 - Σ_k w_k * cos(Δφ_k),  w_k ∝ |X_k|·|Y_k| (정규화)
    - pred, gold: [B,T,1] or [B,T]
    - 위상차(Δφ)를 직접 벌점 → 지연/위상 오프셋이 있으면 손실 > 0
    - AMP/half 안정: autocast 비활성 + float32 강제
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

        X = torch.fft.rfft(p, dim=-1)  # [B, F]
        Y = torch.fft.rfft(g, dim=-1)  # [B, F]

        # 대역 인덱스 (빈 마스크 방지용 정수화)
        k_lo = int(torch.clamp(torch.floor(torch.tensor(band[0] * N / fs)), min=1).item())
        k_hi = int(torch.clamp(torch.ceil(torch.tensor(band[1] * N / fs)), max=X.size(-1) - 1).item())
        if k_hi <= k_lo:
            return p.new_tensor(1.0)

        Xb = X[:, k_lo:k_hi + 1]
        Yb = Y[:, k_lo:k_hi + 1]

        # 위상차와 가중치
        phi_x = torch.angle(Xb)  # [B,Fb]
        phi_y = torch.angle(Yb)
        dphi = phi_x - phi_y  # [B,Fb]
        w = (Xb.abs() * Yb.abs()) + eps
        w = w / (w.sum(dim=-1, keepdim=True) + eps)  # 각 배치 내 정규화

        # cos(Δφ) 평균 → 1 - ···
        cos_d = torch.cos(dphi)
        score = (w * cos_d).sum(dim=-1)  # [B]
        loss = 1.0 - score
        loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
        return loss.mean()


def pad_collate(batch):
    # V1: 길이 버킷으로 동일 길이만 한 배치로 묶기 때문에 불사용
    raise NotImplementedError("pad_collate not used in V1 (we bucket by length).")


def _window_hints(x_np, fs=FS_MODEL):
    # x_np: [T,16], indices: 0=w_rr,1=y_rr,2=d_rr
    w = x_np[:, 0]
    y = x_np[:, 1]
    d = x_np[:, 2]

    # crude SNR: RR대역 PSD 피크 prominence 근사
    def _snr(sig):
        sig = np.asarray(sig, dtype=np.float32)
        L = len(sig)
        if L < 64: return 0.0
        freqs = np.fft.rfftfreq(L, d=1.0 / fs)
        pxx = np.abs(np.fft.rfft(sig)) ** 2
        m = (freqs >= 0.08) & (freqs <= 0.60)
        if not np.any(m): return 0.0
        p = pxx[m]
        peak = float(np.max(p))
        med = float(np.median(p) + 1e-6)
        s = (peak - med) / (peak + 1e-6)
        return float(np.clip(s, 0.0, 1.0))

    snr = _snr(w)

    # abs corr hints
    def _corr(a, b):
        if len(a) < 4: return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(abs(0.0 if np.isnan(c) else c))

    c_wy = _corr(w, y)
    c_wd = _corr(w, d)
    return snr, c_wy, c_wd


def make_batch(dataset, indices):
    # indices: list of (session_idx, a, b, T)
    Xs, Ys = [], []
    for (i, a, b, T) in indices:
        x = torch.tensor(dataset.X[i][a:b], dtype=torch.float32)  # [T,16]
        y = torch.tensor(dataset.Y[i][a:b], dtype=torch.float32)  # [T,1]
        # 힌트 채널(13,14,15)을 윈도우 상수로 채움
        xn = x.numpy()
        snr, cwy, cwd = _window_hints(xn)
        xn[:, 13] = snr
        xn[:, 14] = cwy
        xn[:, 15] = cwd
        Xs.append(torch.tensor(xn, dtype=torch.float32))
        Ys.append(y)
    X = torch.stack(Xs, dim=0)  # [B,T,C] (동일 길이만 배치)
    Y = torch.stack(Ys, dim=0)  # [B,T,1]
    return X, Y


def _align_scale_np(p: np.ndarray, g: np.ndarray, eps: float = 1e-8):
    """
    최소자승 스케일 정렬: â = argmin_a ||a·p - g||^2 = (p·g)/(p·p)
    반환: p_aligned = â·p, â
    """
    pp = float(np.dot(p, p)) + eps
    pg = float(np.dot(p, g))
    a_hat = pg / pp
    return (a_hat * p), a_hat


def evaluate(model, loader, fs=FS_MODEL, device=DEVICE):
    model.eval()
    mse_list, mae_list = [], []
    corr_vals, corr_bl_vals = [], []
    bpm_mae_vals, hit_vals = [], []
    a_hats = []
    valid_bpm = 0
    any_samples = False

    with torch.no_grad():
        for X, Y in loader:
            any_samples = True
            X = X.to(device);
            Y = Y.to(device)
            pred = model(X)  # [B,T,1] or [B,T]

            # [B,T]로 정규화
            if pred.dim() == 3 and pred.size(-1) == 1:
                pred = pred[:, :, 0]
            if Y.dim() == 3 and Y.size(-1) == 1:
                Y = Y[:, :, 0]

            B, T = pred.shape
            for b in range(B):
                pb = pred[b].detach().float().cpu().numpy()
                gb = Y[b].detach().float().cpu().numpy()

                # 결측/비정상 방지
                if not np.isfinite(pb).all() or not np.isfinite(gb).all():
                    continue
                if len(pb) < 8:
                    continue

                # (1) 스케일 정렬 후 오차
                pb_aligned, a_hat = _align_scale_np(pb, gb)
                a_hats.append(a_hat)
                err = pb_aligned - gb
                mse_list.append(float(np.mean(err ** 2)))
                mae_list.append(float(np.mean(np.abs(err))))

                # (2) 상관 & best-lag 상관
                if len(pb) > 3:
                    c = np.corrcoef(pb, gb)[0, 1]
                    if np.isfinite(c):
                        corr_vals.append(float(c))
                c2, _ = corr_soft_bestlag(pb, gb, fs=fs, lag_s=LAG_MAX_S, beta=PHASE_BETA)
                if np.isfinite(c2):
                    corr_bl_vals.append(float(c2))

                # (3) RR bpm (정렬 파형 사용, 완화된 prominence)
                bpm_p = welch_psd_rr_bpm(pb_aligned, fs, band=(0.08, 0.60), min_prom=BPM_MIN_PROM)
                bpm_g = welch_psd_rr_bpm(gb, fs, band=(0.08, 0.60), min_prom=BPM_MIN_PROM)
                if np.isfinite(bpm_p) and np.isfinite(bpm_g):
                    valid_bpm += 1
                    diff = abs(bpm_p - bpm_g)
                    bpm_mae_vals.append(diff)
                    hit_vals.append(float(diff <= SNR_HIT_BPM))

    def _m(x):
        return float(np.mean(x)) if x else float("nan")

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
    if not any_samples:
        out["warning"] = "loader yielded no batches"
    return out


def train_loop(model, optimizer, train_loaders, val_loaders, epochs=EPOCHS, device=DEVICE):
    scaler = GradScaler()
    best_val, best_state, patience = None, None, 0
    AMP_LAMBDA = float(os.getenv("AMP_LAMBDA", "5e-2"))

    for ep in range(1, epochs + 1):
        model.train()
        loss_mse_list, loss_ph_list, loss_amp_list, loss_tot_list = [], [], [], []

        for loader in train_loaders:
            for X, Y in loader:
                X = X.to(device);
                Y = Y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=('cuda' in device)):
                    pred = model(X)
                    loss_mse = si_mse_loss(pred, Y)
                    loss_ph = phase_loss_spectral(pred, Y, fs=FS_MODEL, band=(0.08, 0.60))
                    loss_amp = amp_penalty_loss(pred, Y)
                    total_loss = loss_mse + PHASE_LAMBDA * loss_ph + AMP_LAMBDA * loss_amp
                scaler.scale(total_loss).backward()
                scaler.step(optimizer);
                scaler.update()
                loss_mse_list.append(float(loss_mse.detach().cpu().item()))
                loss_ph_list.append(float(loss_ph.detach().cpu().item()))
                loss_amp_list.append(float(loss_amp.detach().cpu().item()))
                loss_tot_list.append(float(total_loss.detach().cpu().item()))

        # --- Val: 모든 로더 평균 ---
        val_metrics = {}
        for k, vloader in val_loaders.items():
            val_metrics[k] = evaluate(model, vloader, device=device)

        # 평균 corr_bestlag로 Early-Stop 판단
        vals = [m.get("corr_bestlag", -1e9) for m in val_metrics.values()]
        cur = float(np.mean(vals)) if vals else -1e9

        if (best_val is None) or (cur > best_val):
            best_val = cur
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        print(
            f"[epoch {ep:03d}] "
            f"train_mse={np.mean(loss_mse_list):.6e} "
            f"| train_phase={np.mean(loss_ph_list):.6e} "
            f"| train_amp={np.mean(loss_amp_list):.6e} "
            f"| train_total={np.mean(loss_tot_list):.6e} "
            f"| val_avg_corr_bestlag={cur:.6f} "
            f"| val={val_metrics}"
        )
        if patience >= PATIENCE:
            print("[early stop]")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def save_run(run_dir, model, metrics):
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
