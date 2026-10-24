"""After the 10-22 source fill-in: (1) signature cache rows of old targets, old vs new cache; (2) A/B/C effects exported
with the filled stats vs the leaderboard-validated export VCC_gm_vip.npz (G3c / G3v cells)."""
import subprocess, os, sys
import numpy as np
SIG = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"; E = "/home/zxc29051617/VCC/data/stage3/effects"
for n in sorted(f[:-4] for f in os.listdir(f"{SIG}/cache_abc_backup_20261022") if f.endswith(".npz") and not f.startswith("heldout")):
    a = np.load(f"{SIG}/cache_abc_backup_20261022/{n}.npz", allow_pickle=True); b = np.load(f"{SIG}/cache/{n}.npz", allow_pickle=True)
    ta = [str(t) for t in a["targets"]]; tb = {str(t): i for i, t in enumerate(b["targets"])}; Ya, Yb = a["Ysrc"], b["Ysrc"]
    pairs = [(i, tb[t]) for i, t in enumerate(ta) if t in tb]
    x = np.concatenate([Ya[i] for i, _ in pairs]); y = np.concatenate([Yb[j] for _, j in pairs]); ok = np.isfinite(x) & np.isfinite(y)
    print(f"cache {n:13s}: rows {len(ta)} -> {len(tb)}, shared {len(pairs)}, max |diff| {np.abs(x[ok] - y[ok]).max():.2e}, corr {np.corrcoef(x[ok], y[ok])[0, 1]:.5f}", flush=True)
out = "/home/zxc29051617/.claude/jobs/3cf1662b/tmp/abc_after_fill.npz"
subprocess.run(["/home/zxc29051617/VCC/scDFM/.venv/bin/python", "-m", "vcc.gm", "export", "A", "vcc", out, "per_celltype=False", "lam=3.0", "kcap=1.0", "alpha=3.0"],
               cwd="/home/zxc29051617/VCC/scDFM", env={**os.environ, "VCC_DATA_DIR": "/home/zxc29051617/VCC/data/vcc2026"}, check=True, stdout=subprocess.DEVNULL)
z0 = np.load(f"{E}/VCC_gm_vip.npz", allow_pickle=True); z1 = np.load(out, allow_pickle=True)
assert list(z0["targets"]) == list(z1["targets"]); L0, L1 = z0["lfc"].astype(np.float64), z1["lfc"].astype(np.float64)
r = [np.corrcoef(L0[i], L1[i])[0, 1] for i in range(len(L0)) if L0[i].std() > 0]
print(f"A/B/C effects after vs before the fill-in: per-target corr median {np.median(r):.4f} (min {np.min(r):.4f}); overall rel. change {np.linalg.norm(L1 - L0) / np.linalg.norm(L0):.4f}", flush=True)
