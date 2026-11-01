"""Where does the DE lfc error (nmae) come from? H1 / KOLF200 V6 DE tables (2026-10-20, analysis of headroom).

Per target: gate = real p_adj < 0.05, own gene excluded, >= 10 genes (scorer rules). nmae_p = mean|pred - real| / mean|real|.
Classes: cis (within 50 kb of the target, the model's cis set) vs trans. Reports the share of the denominator from cis
genes, nmae as is, with cis replaced by truth (headroom), with trans predictions set to 0, and per-class sign accuracy.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay"); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013")
import sigexp as S
from vcc.gm import Genome
gen = Genome(S.GENES)
for name in ["H1", "KOLF200"]:
    R = pd.read_parquet(f"de_sets/{name}/de_real.parquet"); P = pd.read_parquet(f"de_sets/{name}/de_pred.parquet")
    M = R.merge(P, on=["target", "feature"], suffixes=("_r", "_p"))
    rows = []
    for t, d in M.groupby("target"):
        d = d[(d.p_adj_r < 0.05) & (d.feature != t)]
        d = d[np.isfinite(d.log2_fold_change_r) & np.isfinite(d.log2_fold_change_p)]
        if len(d) < 10: continue
        cg, _ = gen.cis(t); cis = set(S.GENES[i] for i in cg)
        isc = d.feature.isin(cis).values; lr = d.log2_fold_change_r.values; lp = d.log2_fold_change_p.values
        den = np.abs(lr).mean()
        e = np.abs(lp - lr)
        rows.append(dict(t=t, n=len(d), ncis=int(isc.sum()), cis_share=np.abs(lr[isc]).sum() / np.abs(lr).sum(),
                         nmae=e.mean() / den, nmae_cis_true=np.where(isc, 0, e).mean() / den,
                         nmae_trans0=np.where(isc, e, np.abs(lr)).mean() / den,
                         sign_cis=np.mean(np.sign(lp[isc]) == np.sign(lr[isc])) if isc.any() else np.nan,
                         sign_trans=np.mean(np.sign(lp[~isc]) == np.sign(lr[~isc])),
                         top10_sign_trans=np.mean((np.sign(lp[~isc]) == np.sign(lr[~isc]))[np.argsort(-np.abs(lp[~isc]))[:10]])))
    df = pd.DataFrame(rows)
    print(f"{name}: {len(df)} targets, median gate {df.n.median():.0f}, cis genes in gate median {df.ncis.median():.0f}; "
          f"cis share of sum|real lfc| mean {df.cis_share.mean():.3f}")
    print(f"   nmae {df.nmae.mean():.3f} | cis -> truth {df.nmae_cis_true.mean():.3f} | trans -> 0 {df.nmae_trans0.mean():.3f} | "
          f"sign acc cis {df.sign_cis.mean():.3f}, trans {df.sign_trans.mean():.3f}, top-10 |pred| trans {df.top10_sign_trans.mean():.3f}")
