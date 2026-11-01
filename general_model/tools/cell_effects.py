"""Cell-level effects for the dual render (2026-10-20): own + cis + AMP x trans x (agreement / median)^K, where agreement is
the fraction of the public training screens whose response to the target moves the gene in the same direction as their
mean (signature cache: kscale-normalised, common-mode-removed log ratios). Genes whose sources agree get larger shifts, so
the cells' significance order follows the model's confidence (reach) while more genes are called (fid).
    python cell_effects.py <full effects.npz> <oc effects.npz> <out.npz> [AMP, default 1.5] [K, default 1]
Uses only the predictions and the public training screens.
"""
import os, sys
import numpy as np
sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013")
import sigexp as S
full_f, oc_f, out = sys.argv[1:4]; AMP = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5; K = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
zf = np.load(full_f, allow_pickle=True); zo = np.load(oc_f, allow_pickle=True)
targets = [str(t) for t in zf["targets"]]; assert targets == [str(t) for t in zo["targets"]], "target order differs"
full = zf["lfc"].astype(np.float64); oc = zo["lfc"].astype(np.float64); tr = full - oc
ti = {t: i for i, t in enumerate(targets)}
n = np.zeros_like(full); s1 = np.zeros_like(full); pos = np.zeros_like(full)
C = S.load_cache([c for c in S.CONTEXTS if c not in set(os.environ.get("VCC_EXCLUDE", "").split(","))])  # VCC_EXCLUDE: mock exams
for c in C:
    Y = C[c]["Ysrc"]
    for j, t in enumerate(C[c]["targets"]):
        i = ti.get(t)
        if i is None: continue
        y = Y[j]; ok = np.isfinite(y); yz = np.where(ok, y, 0); n[i] += ok; s1[i] += yz; pos[i] += yz > 0
mean = s1 / np.maximum(n, 1); agree = np.where(mean > 0, pos / np.maximum(n, 1), 1 - pos / np.maximum(n, 1))
covered = n.sum(1) > 0
med = np.median(agree[np.abs(tr) > 0]) if (np.abs(tr) > 0).any() else 1.0
g = np.clip(agree / med, 0, None) ** K
g[~covered] = 1.0                                   # targets absent from the cache: plain amplification
cells = oc + AMP * tr * g
np.savez_compressed(out, targets=np.array(targets), lfc=cells.astype(np.float32))
print(f"wrote {out}: own + cis + {AMP} x trans x (agreement / {med:.3f})^{K}; {int((~covered).sum())} targets without cache rows")
