# %% Lag audit & visualization for COHFACE cache (.npz)
# - Uses stored 'lag' if present; else estimates by GCC-PHAT between composite dC and g_resp
# - Compares corr at 0-lag vs best-lag (delta) to judge usefulness of global alignment
# - Saves CSV + histogram under runs/
import os, glob, json, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("CACHE_DIR", "cohface_exp_reg/cache_cohface_feats"))
    ap.add_argument("--fs", type=float, default=float(os.environ.get("FS_RESAMP", 256)))
    ap.add_argument("--max_lag_s", type=float, default=float(os.environ.get("LAG_MAX_S", 0.5)))
    ap.add_argument("--recursive", action="store_true")
    return ap.parse_args()

def gcc_phat_lag(x, y, fs=256.0, max_tau=0.5):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = 1 << (len(x) + len(y) - 2).bit_length()
    X = np.fft.rfft(x, n=n); Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y); R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=n)
    max_shift = int(min(max_tau * fs, (n // 2) - 1))
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    idx = int(np.argmax(cc))
    lag_samples = idx - max_shift
    return float(np.clip(lag_samples / fs, -max_tau, max_tau))

def first_key(d, cand):
    for k in cand:
        if k in d: return k
    return None

def zscore(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-12)

def corr(a, b):
    a = zscore(a); b = zscore(b)
    return float(np.clip(np.mean(a*b), -1.0, 1.0))

def shift_linear(sig, tau_s, fs):
    """sub-sample linear shift (1D), same length, pad with edge values"""
    x = np.asarray(sig, float); T = len(x)
    t = np.arange(T)
    t_src = t - tau_s*fs
    t0 = np.floor(t_src).astype(int); t1 = t0 + 1
    w = t_src - t0
    t0 = np.clip(t0, 0, T-1); t1 = np.clip(t1, 0, T-1)
    return (1-w)*x[t0] + w*x[t1]

def estimate_from_npz(z, fs=256.0, max_lag_s=0.5):
    # prefer stored lag
    if 'lag' in z:
        try: return float(z['lag']), True
        except: pass
    # build composite dC and get g_resp
    k_w = first_key(z, ["dW","w","w_rr","w_hr","dw"])
    k_y = first_key(z, ["dY","y","y_rr","y_hr","dy"])
    k_d = first_key(z, ["dD","d","d_rr","d_hr","dd"])
    k_c = first_key(z, ["dC","c","c_rr","c_hr"])
    k_resp = first_key(z, ["g_resp","resp","gt_rr","gt_resp"])
    if k_resp is None: raise KeyError("No GT respiration (g_resp/resp/gt_rr).")
    gt = np.asarray(z[k_resp]).squeeze()
    comps=[]
    for k in [k_w,k_y,k_d]:
        if k is not None and k in z: comps.append(np.asarray(z[k]).squeeze())
    if not comps and k_c is not None: comps.append(np.asarray(z[k_c]).squeeze())
    if not comps: raise KeyError("No displacement channels (dW/dY/dD or composite).")
    L = min([len(gt)] + [len(c) for c in comps])
    gt = gt[:L]; dc = np.mean([c[:L] for c in comps], axis=0)
    lag_s = gcc_phat_lag(dc - dc.mean(), gt - gt.mean(), fs=fs, max_tau=max_lag_s)
    return float(lag_s), False

def scan(cache_dir, fs=256.0, max_lag_s=0.5, recursive=True):
    pattern = "**/*.npz" if recursive else "*.npz"
    files = sorted(glob.glob(os.path.join(cache_dir, pattern), recursive=recursive))
    rows=[]
    for p in files:
        try:
            z = np.load(p, allow_pickle=True)
        except Exception as e:
            rows.append({"file": p, "ok": False, "error": f"load fail: {e}"}); continue
        try:
            lag_s, used_cached = estimate_from_npz(z, fs, max_lag_s)
            # corr at 0 vs best-lag
            k_resp = first_key(z, ["g_resp","resp","gt_rr","gt_resp"])
            gt = np.asarray(z[k_resp]).squeeze()
            # composite
            k_w = first_key(z, ["dW","w","w_rr","w_hr","dw"])
            k_y = first_key(z, ["dY","y","y_rr","y_hr","dy"])
            k_d = first_key(z, ["dD","d","d_rr","d_hr","dd"])
            k_c = first_key(z, ["dC","c","c_rr","c_hr"])
            comps=[]
            for k in [k_w,k_y,k_d]:
                if k is not None and k in z: comps.append(np.asarray(z[k]).squeeze())
            if not comps and k_c is not None: comps.append(np.asarray(z[k_c]).squeeze())
            L = min([len(gt)] + [len(c) for c in comps])
            gt = gt[:L]; dc = np.mean([c[:L] for c in comps], axis=0)
            c0  = corr(dc, gt)
            cbl = corr(shift_linear(dc, lag_s, fs), gt)
            rows.append({"file": p, "ok": True, "lag_s": lag_s, "lag_ms": lag_s*1000.0,
                         "used_cached": used_cached, "corr0": c0, "corr_best": cbl,
                         "delta_corr": cbl - c0})
        except Exception as e:
            rows.append({"file": p, "ok": False, "error": str(e)})
    return rows

def summarize(vals):
    a = np.asarray(vals, float)
    if a.size==0: return {}
    q = np.quantile(a, [0,0.25,0.5,0.75,1])
    return {
        "count": int(a.size),
        "mean_s": float(a.mean()), "std_s": float(a.std(ddof=1) if a.size>1 else 0.0),
        "min_s": float(q[0]), "q1_s": float(q[1]), "median_s": float(q[2]),
        "q3_s": float(q[3]), "max_s": float(q[4]),
        "mean_ms": float(a.mean()*1000.0), "std_ms": float((a.std(ddof=1) if a.size>1 else 0.0)*1000.0),
    }

def decision(stats, delta_corr_vals):
    if not stats: return "파일 없음 → 결론 불가"
    mean_abs = abs(stats["mean_s"]); std = stats["std_s"]
    delta_mean = float(np.mean(delta_corr_vals)) if len(delta_corr_vals)>0 else 0.0
    if mean_abs <= 0.02 and std <= 0.03 and delta_mean < 0.02:
        return "전역 오프셋/효과 미미 → 전역 보정 OFF 권장"
    elif delta_mean >= 0.03:
        return "정렬 시 상관 크게 개선 → 제한된 범위에서 전역 보정 ON 고려"
    else:
        return "경계 상황 → OFF 기본 + 학습단 지연-불변 손실 권장"

def main():
    args = parse_args()
    cache_dir = args.cache; fs = float(args.fs); max_lag_s = float(args.max_lag_s)
    print(f"[audit] cache={cache_dir} fs={fs}Hz max_lag=±{max_lag_s}s")

    rows = scan(cache_dir, fs, max_lag_s, recursive=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print("[result] No NPZ files found."); return

    ok = df[df.ok==True]
    stats = summarize(ok["lag_s"].values) if not ok.empty else {}
    dec = decision(stats, ok["delta_corr"].values.tolist() if "delta_corr" in ok else [])
    os.makedirs("runs", exist_ok=True)
    out_csv = "cohface_exp_reg/runs/lag_audit_results.csv"
    ok.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    print("\n=== Summary (lag seconds / milliseconds) ===")
    if stats: print(json.dumps(stats, indent=2))
    else: print("No successful entries.")
    if "delta_corr" in ok:
        print(f"mean delta_corr={ok['delta_corr'].mean():.4f}  (corr_best - corr0)")
    print(f"decision: {dec}")

    # histogram
    if not ok.empty:
        plt.figure()
        bins = max(10, min(60, int(math.sqrt(len(ok)))))
        plt.hist(ok["lag_s"].values, bins=bins)
        plt.title("Distribution of lag (seconds)")
        plt.xlabel("lag (s)"); plt.ylabel("count")
        plt.tight_layout()
        out_png = "cohface_exp_reg/runs/lag_hist.png"; plt.savefig(out_png)
        print(f"[saved] {out_png}")
        try: plt.show()
        except Exception: pass

if __name__ == "__main__":
    main()
