# cohface_exp_reg/train.py
import json
import os
from typing import Dict

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import DEVICE, RUNS_DIR, LR, EPOCHS, PATIENCE
from .utils import estimate_rr_bpm

def corrcoef_masked(x, y, mask, eps=1e-8):
    m = (mask > 0.5).float()
    L = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)
    xm = (x * m); ym = (y * m)
    mean_x = xm.sum(dim=1, keepdim=True) / L
    mean_y = ym.sum(dim=1, keepdim=True) / L
    xc = (x - mean_x) * m
    yc = (y - mean_y) * m
    num = (xc * yc).sum(dim=1)
    den = torch.sqrt((xc.square().sum(dim=1) + eps) * (yc.square().sum(dim=1) + eps))
    r = num / den
    return r.mean()

def train_loop(model, train_loader: DataLoader, val_loader: DataLoader, tag: str):
    out_dir = os.path.join(RUNS_DIR, tag); os.makedirs(out_dir, exist_ok=True)
    model = model.to(DEVICE)
    torch.backends.cudnn.benchmark = (DEVICE.startswith("cuda"))
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler("cuda", enabled=DEVICE.startswith("cuda"))

    best = {"val_loss": 1e9, "epoch": 0, "corr_rr": 0.0}; patience = 0
    for epoch in range(1, EPOCHS+1):
        model.train(); tr_loss = 0.0
        for X, Y, M, pad_mask, *_ in train_loader:
            X = X.to(DEVICE).float(); Y = Y.to(DEVICE).float()
            M = M.to(DEVICE).float(); P = pad_mask.to(DEVICE).float()

            with autocast("cuda", enabled=DEVICE.startswith("cuda")):
                pred = model(X)  # [B,T,2]
                mask_rr = (M[:,:,0:1] * P); mask_hr = (M[:,:,1:2] * P)
                def masked_mse(p, y, m):
                    num = ((p - y).square() * m).sum()
                    den = torch.clamp(m.sum(), min=1.0)
                    return num / den
                mse_rr = masked_mse(pred[:,:,0:1], Y[:,:,0:1], mask_rr)
                mse_hr = masked_mse(pred[:,:,1:2], Y[:,:,1:2], mask_hr)
                corr_rr = corrcoef_masked(pred[:,:,0], Y[:,:,0], mask_rr.squeeze(-1))
                loss = mse_rr + 0.5*mse_hr + 0.2*(1.0 - corr_rr)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))

        model.eval(); va_loss=0.0; corr_list=[]
        with torch.no_grad():
            for X,Y,M,pad_mask,*_ in val_loader:
                X=X.to(DEVICE).float(); Y=Y.to(DEVICE).float()
                M=M.to(DEVICE).float(); P=pad_mask.to(DEVICE).float()
                with autocast("cuda", enabled=DEVICE.startswith("cuda")):
                    pred = model(X)
                    mask_rr = (M[:,:,0:1] * P); mask_hr = (M[:,:,1:2] * P)
                    def masked_mse(p, y, m):
                        num = ((p - y).square() * m).sum()
                        den = torch.clamp(m.sum(), min=1.0)
                        return num / den
                    va_loss += (masked_mse(pred[:,:,0:1], Y[:,:,0:1], mask_rr) +
                                0.5*masked_mse(pred[:,:,1:2], Y[:,:,1:2], mask_hr)).item()
                    corr_list.append(float(corrcoef_masked(pred[:,:,0], Y[:,:,0], mask_rr.squeeze(-1)).item()))
        va_loss /= max(1,len(val_loader)); corrRR = float(np.mean(corr_list)) if corr_list else 0.0
        print(f"[{epoch:02d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_corrRR={corrRR:.3f}")

        crit = va_loss + (1.0 - corrRR)*0.1
        if crit < best["val_loss"]:
            best.update(val_loss=crit, corr_rr=corrRR, epoch=epoch)
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            with open(os.path.join(out_dir, "best.json"), "w") as f:
                json.dump(best, f, indent=2, ensure_ascii=False)
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break
    return out_dir, best

def evaluate(model, loader: DataLoader, fs_model: float) -> Dict[str,float]:
    model = model.to(DEVICE).eval()
    all_corr_rr, all_rmse_rr, all_mae_rr = [], [], []
    all_corr_hr, all_rmse_hr, all_mae_hr = [], [], []
    rr_pred_bpm_list, rr_gt_bpm_list = [], []
    with torch.no_grad():
        for X,Y,M,pad_mask,*_ in loader:
            X=X.to(DEVICE).float(); Y=Y.to(DEVICE).float()
            M=M.to(DEVICE).float(); P=pad_mask.to(DEVICE).float()
            with autocast("cuda", enabled=DEVICE.startswith("cuda")):
                pred=model(X)
            mask_rr = (M[:,:,0:1] * P); mask_hr = (M[:,:,1:2] * P)
            def masked_rmse(p, y, m):
                num = ((p - y).square() * m).sum()
                den = torch.clamp(m.sum(), min=1.0)
                return torch.sqrt(num/den)

            def masked_mae(p, y, m):
                num = (torch.abs(p - y) * m).sum()
                den = torch.clamp(m.sum(), min=1.0)
                return num / den
            all_rmse_rr.append(masked_rmse(pred[:, :, 0:1], Y[:, :, 0:1], mask_rr).item())
            all_mae_rr.append(masked_mae(pred[:, :, 0:1], Y[:, :, 0:1], mask_rr).item())
            all_corr_rr.append(corrcoef_masked(pred[:,:,0], Y[:,:,0], mask_rr.squeeze(-1)).item())
            # RR bpm per-window
            B = X.shape[0]
            for b in range(B):
                m = mask_rr[b,:,0] > 0.5
                if m.sum() >= int(6*fs_model):
                    y_gt = Y[b,m,0].detach().cpu().numpy()
                    y_pr = pred[b,m,0].detach().cpu().numpy()
                    gt_bpm = estimate_rr_bpm(y_gt, fs_model)
                    pr_bpm = estimate_rr_bpm(y_pr, fs_model)
                    if np.isfinite(gt_bpm) and np.isfinite(pr_bpm):
                        rr_gt_bpm_list.append(float(gt_bpm))
                        rr_pred_bpm_list.append(float(pr_bpm))
            # HR
            if (mask_hr.sum()>0).item():
                all_rmse_hr.append(masked_rmse(pred[:, :, 1:2], Y[:, :, 1:2], mask_hr).item())
                all_mae_hr.append(masked_mae(pred[:, :, 1:2], Y[:, :, 1:2], mask_hr).item())
                m = mask_hr.squeeze(-1)>0.5
                if m.any():
                    gt = Y[:,:,1][m].detach().cpu().numpy()
                    pr = pred[:,:,1][m].detach().cpu().numpy()
                    if gt.size>10 and np.all(np.isfinite(pr)) and np.all(np.isfinite(gt)):
                        all_corr_hr.append(float(np.corrcoef(pr,gt)[0,1]))
    def safemean(v):
        return float(np.mean(v)) if (len(v)>0 and np.all(np.isfinite(v))) else float("nan")

    rr_bpm_mae = safemean([abs(a - b) for a, b in zip(rr_pred_bpm_list, rr_gt_bpm_list)])

    return dict(corr_rr=safemean(all_corr_rr), rmse_rr = safemean(all_rmse_rr), mae_rr = safemean(all_mae_rr),
                corr_hr = safemean(all_corr_hr), rmse_hr = safemean(all_rmse_hr), mae_hr = safemean(all_mae_hr),
                rr_bpm_mae = rr_bpm_mae
                )
