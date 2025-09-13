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
    # batch: list of dicts produced in make_batch (see train_loop)
    # we expect already sliced tensors in same length per batch
    raise NotImplementedError("pad_collate not used in V1 (we bucket by length).")

def make_batch(dataset, indices):
    # indices: list of (session_idx, a, b, T)
    Xs, Ys = [], []
    for (i, a, b, T) in indices:
        x = torch.tensor(dataset.X[i][a:b], dtype=torch.float32)  # [T,16]
        y = torch.tensor(dataset.Y[i][a:b], dtype=torch.float32)  # [T,1]
        # 힌트 채널(상수)을 윈도우 단위로 채움: snr/corr
        # 간단 구현: 0으로 두되, 이후 개선 시 여기서 계산하여 채움
        Xs.append(x); Ys.append(y)
    X = torch.stack(Xs, dim=0)  # [B,T,C] (같은 길이만 모아 배치)
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
            # numpy 지표
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
                    # corr@soft-best-lag (numpy ref → torch로 대체 가능하나 ep 마다 배치 샘플링으로 근사)
                    # 여기서는 간소화: 배치 평균으로 numpy 함수 호출
                    loss_phase = 0.0
                    for b in range(pred.size(0)):
                        pb = pred[b,:,0].detach().cpu().numpy()
                        gb = Y[b,:,0].detach().cpu().numpy()
                        c2, _ = corr_soft_bestlag(pb, gb, fs=FS_MODEL, lag_s=LAG_MAX_S, beta=PHASE_BETA)
                        loss_phase += (1.0 - c2)
                    loss_phase = loss_phase / max(1, pred.size(0))
                    loss = mse + PHASE_LAMBDA * loss_phase
                scaler.scale(torch.tensor(loss, dtype=torch.float32, device=device)).backward()
                scaler.step(optimizer)
                scaler.update()
                ep_loss.append(float(mse.detach().cpu().item()))

        # 평가
        val_metrics = {}
        for k, vloader in val_loaders.items():
            val_metrics[k] = evaluate(model, vloader, device=device)

        # 선택 기준: corr_bestlag 최대
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
    # 복원
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def save_run(run_dir, model, metrics):
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
