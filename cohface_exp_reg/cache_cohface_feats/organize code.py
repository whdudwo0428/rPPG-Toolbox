'''
python - <<'PY'
import os, re, shutil, glob

ROOT = "cohface_exp_reg/cache_cohface_feats"
os.makedirs(ROOT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "_old"), exist_ok=True)

# 0) 하위 폴더(s*_k*/)에 갇힌 npz 위로 끌어올리기
for d in glob.glob(os.path.join(ROOT, "s*_k*")):
    if os.path.isdir(d):
        for f in glob.glob(os.path.join(d, "*.npz")):
            base = os.path.basename(f)
            dst  = os.path.join(ROOT, base)
            if os.path.exists(dst):
                shutil.move(f, os.path.join(ROOT, "_old", base))
            else:
                shutil.move(f, dst)
        # 폴더 비었으면 삭제
        try: os.rmdir(d)
        except: pass

# 1) 파일명 정규화: s{S}_k{K}.npz -> s{S:02d}_k{K:02d}.npz
pat = re.compile(r'^s(\d{1,2})_k(\d{1,2})\.npz$')
for base in os.listdir(ROOT):
    if not base.endswith(".npz"):
        continue
    m = pat.match(base)
    if not m:
        continue
    S, K = int(m.group(1)), int(m.group(2))
    padded = f"s{S:02d}_k{K:02d}.npz"
    src = os.path.join(ROOT, base)
    dst = os.path.join(ROOT, padded)
    if base == padded:
        continue
    if os.path.exists(dst):
        # 이미 새 규격 파일이 있으면 구형은 보관폴더로 이동
        shutil.move(src, os.path.join(ROOT, "_old", base))
    else:
        os.rename(src, dst)

print("[cleanup] done. kept =", len([f for f in os.listdir(ROOT) if f.endswith(".npz")]))
print("[cleanup] moved to _old =", len(os.listdir(os.path.join(ROOT, "_old"))))
PY
'''