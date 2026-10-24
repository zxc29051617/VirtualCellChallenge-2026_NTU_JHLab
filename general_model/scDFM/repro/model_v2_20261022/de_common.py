"""How much of the reference's DE calls is the context's common response, and can we predict it? (2026-10-23)

    .venv/bin/python -I repro/model_v2_20261022/de_common.py <variant> [subset]        (from scDFM/)

real panel mean  = mean over the subset's targets of the real log2 fold change per gene (what the context-mean
                   baseline knows); compared with (a) the mean over targets of our totals input before the pipeline's
                   mean removal (d200 file), (b) the mean common mode of the training contexts (H1 excluded).
Also: of each target's reference-significant genes, the share whose sign equals the sign of the panel mean (habitual
direction) and how often our called sign agrees.
"""
import os, sys
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

HERE = os.path.dirname(os.path.abspath(__file__)); EFF = "/home/zxc29051617/VCC/data/stage3/effects"
v = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
d = f"{HERE}/mock/mock_{v}_s{k}_ce2"
R = pl.read_parquet(f"{d}/de_real.parquet").to_pandas(); P = pl.read_parquet(f"{d}/de_pred.parquet").to_pandas()
G = B.genome(); gi = G.idx
Lr = R.pivot(index="target", columns="feature", values="log2_fold_change"); Sr = R.pivot(index="target", columns="feature", values="p_adj")
Lp = P.pivot(index="target", columns="feature", values="log2_fold_change").reindex(index=Lr.index, columns=Lr.columns)
feat = list(Lr.columns); cols = np.array([gi.get(f, -1) for f in feat]); ok = cols >= 0
print(f"{Lr.shape[0]} targets x {Lr.shape[1]} genes in the DE tables ({ok.sum()} on the model axis)")
pm_real = np.nanmean(Lr.values, 0)

zd = np.load(f"{EFF}/mock_{v}_s{k}_d200.npz", allow_pickle=True); zo = np.load(f"{EFF}/mock_{v}_s{k}_oc.npz", allow_pickle=True)
tl = [str(t) for t in zd["targets"]]; ti = [tl.index(t) for t in Lr.index]
tr = (zd["lfc"].astype(np.float64) - zo["lfc"].astype(np.float64))[ti]
pm_ours = np.full(len(feat), np.nan); pm_ours[ok] = tr[:, cols[ok]].mean(0)
D = B.data(); cms = [np.asarray(D[c]["cm"], np.float64) / np.log(2) for c in B.CONTEXTS if c not in B.FOLDS["H1"]]
pm_cm = np.full(len(feat), np.nan); pm_cm[ok] = np.nanmean(np.stack(cms), 0)[cols[ok]]
for nm, x in (("our panel mean (before removal)", pm_ours), ("training contexts' common mode", pm_cm)):
    m = np.isfinite(x) & np.isfinite(pm_real)
    print(f"corr(real panel mean, {nm}) = {np.corrcoef(x[m], pm_real[m])[0, 1]:+.3f}  (norms {np.linalg.norm(x[m]):.2f} vs {np.linalg.norm(pm_real[m]):.2f})")

sig = Sr.values < 0.05
own = np.array([[f == t for f in feat] for t in Lr.index]); sig &= ~own
hab = np.sign(Lr.values) == np.sign(pm_real)[None]
ours = np.sign(Lp.values) == np.sign(Lr.values)
nR = sig.sum(1)
print(f"reference-significant genes per target: median {np.median(nR):.0f}; share with the habitual sign (sign of panel mean): "
      f"{(hab & sig).sum() / sig.sum():.3f}; our sign correct on them: {(ours & sig).sum() / sig.sum():.3f}")
# per-gene: how often is the gene significant across the panel's targets
freq = sig.mean(0)
for lo, hi in ((0, .05), (.05, .2), (.2, .5), (.5, 1.01)):
    m = (freq >= lo) & (freq < hi)
    if not m.any(): continue
    s = sig[:, m]
    print(f"genes significant in {lo:.0%}-{hi:.0%} of targets: {m.sum():5d} genes, {s.sum():6d} calls ({s.sum() / sig.sum():.1%}); "
          f"habitual sign {(hab[:, m] & s).sum() / max(s.sum(), 1):.3f}, our sign {(ours[:, m] & s).sum() / max(s.sum(), 1):.3f}")
