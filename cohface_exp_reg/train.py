# -*- coding: utf-8 -*-
import os, json, math, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .config import (DEVICE, RUNS_DIR, LR, EPOCHS, PATIENCE, FS_MODEL,
                     PHASE_LAMBDA, PHASE_BETA, LAG_MAX_S, SNR_HIT_BPM)
from .utils import corr_soft_bestlag, welch_psd_rr_bpm

def pad_collate(batch):
    # V1: 길이 버킷으로 동일 길이만 한 배치로 묶기 때문에 불사용
    raise NotImplementedError("pad_collate not used in V1 (we bucket by length).")

def _window_hints(x_np, fs=FS_MODEL):
    # x_np: [T,16], indices: 0=w_rr,1=y_rr,2=d_rr
    w = x_np[:,0]; y = x_np[:,1]; d = x_np[:,2]
    # crude SNR: RR대역 PSD 피크 prominence 근사
    def _snr(sig):
        sig = np.asarray(sig, dtype=np.float32)
        L = len(sig)
        if L < 64: return 0.0
        freqs = np.fft.rfftfreq(L, d=1.0/fs)
        pxx = np.abs(np.fft.rfft(sig))**2
        m = (freqs>=0.08) & (freqs<=0.60)
        if not np.any(m): return 0.0
        p = pxx[m]
        peak = float(np.max(p)); med = float(np.median(p) + 1e-6)
        s = (peak - med) / (peak + 1e-6)
        return float(np.clip(s, 0.0, 1.0))
    snr = _snr(w)
    # abs corr hints
    def _corr(a,b):
        if len(a)<4: return 0.0
        c = np.corrcoef(a,b)[0,1]
        return float(abs(0.0 if np.isnan(c) else c))
    c_wy = _corr(w,y)
    c_wd = _corr(w,d)
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
        xn[:,13] = snr; xn[:,14] = cwy; xn[:,15] = cwd
        Xs.append(torch.tensor(xn, dtype=torch.float32))
        Ys.append(y)
    X = torch.stack(Xs, dim=0)  # [B,T,C] (동일 길이만 배치)
    Y = torch.stack(Ys, dim=0)  # [B,T,1]
    return X, Y

def evaluate(model, loader, fs=FS_MODEL, device=DEVICE):
    model.eval()
    mse_list, mae_list, corr_list, corr_bl_list, bpm_mae_list, hit_list = [], [], [], [], [], []
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device); Y = Y.to(device)
            pred = model(X)  # [B,T,1]
            err = pred - Y
            mse = (err**2).mean(dim=[1,2]).cpu().numpy()
            mae = err.abs().mean(dim=[1,2]).cpu().numpy()
            for b in range(pred.size(0)):
                pb = pred[b,:,0].detach().cpu().numpy()
                gb = Y[b,:,0].detach().cpu().numpy()
                c  = np.corrcoef(pb, gb)[0,1] if len(pb)>3 else 0.0
                c2, _lag = corr_soft_bestlag(pb, gb, fs=fs, lag_s=LAG_MAX_S, beta=PHASE_BETA)
                bpm_p = welch_psd_rr_bpm(pb, fs); bpm_g = welch_psd_rr_bpm(gb, fs)
                bpm_mae = abs(bpm_p - bpm_g) if not (np.isnan(bpm_p) or np.isnan(bpm_g)) else np.nan
                hit = float(abs(bpm_p-bpm_g) <= SNR_HIT_BPM) if not (np.isnan(bpm_p) or np.isnan(bpm_g)) else 0.0
                corr_list.append(c if not np.isnan(c) else 0.0)
                corr_bl_list.append(c2)
                bpm_mae_list.append(bpm_mae if not np.isnan(bpm_mae) else 0.0)
                hit_list.append(hit)
            mse_list.extend(mse.tolist()); mae_list.extend(mae.tolist())
    out = {
        "mse": float(np.mean(mse_list)),
        "mae": float(np.mean(mae_list)),
        "corr": float(np.mean(corr_list)),
        "corr_bestlag": float(np.mean(corr_bl_list)),
        "rr_bpm_mae": float(np.mean(bpm_mae_list)),
        f"hit@±{SNR_HIT_BPM}bpm": float(np.mean(hit_list)),
    }
    return out

def train_loop(model, optimizer, train_loaders, val_loaders, epochs=EPOCHS, device=DEVICE):
    scaler = GradScaler()
    best_val = None
    best_state = None
    patience = 0

    for ep in range(1, epochs+1):
        model.train()
        ep_loss = []
        for loader in train_loaders:
            for X, Y in loader:
                X = X.to(device); Y = Y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type='cuda', enabled=('cuda' in device)):
                    pred = model(X)
                    mse = torch.mean((pred - Y)**2)
                    # corr@soft-best-lag은 numpy로 계산(상수 취급), MSE 그래프만 유지
                    loss_phase = 0.0
                    for b in range(pred.size(0)):
                        pb = pred[b,:,0].detach().cpu().numpy()
                        gb = Y[b,:,0].detach().cpu().numpy()
                        c2, _ = corr_soft_bestlag(pb, gb, fs=FS_MODEL, lag_s=LAG_MAX_S, beta=PHASE_BETA)
                        loss_phase += (1.0 - c2)
                    loss_phase = loss_phase / max(1, pred.size(0))
                    # BUGFIX: 그래프 유지 (mse 경로)
                    total_loss = mse + (PHASE_LAMBDA * torch.as_tensor(loss_phase, device=device))
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ep_loss.append(float(mse.detach().cpu().item()))

        # 평가 (현재 Early-Stop은 'val' 첫 그룹 기준)
        val_metrics = {}
        for k, vloader in val_loaders.items():
            val_metrics[k] = evaluate(model, vloader, device=device)

        key = "corr_bestlag"
        cur = val_metrics.get("val", {}).get(key, -1e9)
        if (best_val is None) or (cur > best_val):
            best_val = cur
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        print(f"[epoch {ep:03d}] train_mse={np.mean(ep_loss):.4f} | val={val_metrics.get('val',{})}")
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