"""Bulk-only transformations for target discrimination (pds), screened with the exact noise-free proxy on the three
hold-outs. All variants start from V6 (own + cis + trans with the panel mean and top-10 shared components removed) and
change only the gene totals (cells can stay as V6 through dual rendering, so the DE members are not touched here).

    python pds_screen2.py            (scDFM venv)

  k            shared components removed: 5, 10, 15, 20, 30
  ctrlax m     additionally remove the top-m principal axes of the hold-out's CONTROL cells (log1p CP10k, 2,000 cells) from
               the trans part (cell-state directions that every perturbation response can contain), m = 10, 30, 50
  white        divide each gene's trans change by its across-target standard deviation (equalise gene weights), then
               restore the target's norm
  cisw g       cis neighbours x g in the totals (1.5, 2)
Uses hold-out truth for scoring only (screen); a choice is confirmed with the official scorer afterwards.
"""
import json, sys
import numpy as np, scipy.sparse as sp
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM"); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM/repro/signature_20261013")
import coexp_screen as CS, render_eval as E, sigexp as S
from vcc.gm import Genome
gen = Genome(S.GENES); res = {}


def rm_shared(t, k):
    t = t - t.mean(0, keepdims=True)
    if k == 0: return t
    _, _, vt = np.linalg.svd(t, full_matrices=False); return t - (t @ vt[:k].T) @ vt[:k]


for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); _, oc, full, shuf = E.components(name)
    tr = (full - oc).astype(np.float64)
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True; mp = ~panel & (rate > 0)
    sc = lambda lfc: round(CS.pds_like(CS.pred_deltas(lfc, rate, mu_c), D, mp), 4)
    row = {}
    for k in [5, 10, 15, 20, 30]: row[f"k{k}"] = sc(oc + rm_shared(tr, k))
    base = rm_shared(tr, 10)
    # control principal axes of the hold-out context (no truth used)
    D0, R, C, lib, rate0, ref_mean = E.load_set(name)
    idx = np.random.default_rng(0).choice(C.shape[0], min(2000, C.shape[0]), replace=False)
    X = C[idx].multiply(1e4 / lib[idx][:, None]).tocsr(); X.data = np.log1p(X.data); X = X.toarray(); X -= X.mean(0, keepdims=True)
    _, _, Vc = np.linalg.svd(X, full_matrices=False)
    for m in [10, 30, 50]:
        V = Vc[:m]; row[f"k10_ctrlax{m}"] = sc(oc + (base - (base @ V.T) @ V))
    sd = base.std(0); w = np.where(sd > 0, 1 / np.maximum(sd, 1e-6), 0)
    wb = base * w[None]; wb *= (np.linalg.norm(base, axis=1, keepdims=True) / (np.linalg.norm(wb, axis=1, keepdims=True) + 1e-12))
    row["k10_white"] = sc(oc + wb)
    for g in [1.5, 2.0]:
        o2 = oc.copy()
        for i, t in enumerate(targets):
            gg, _ = gen.cis(t); o2[i, gg] *= g
        row[f"k10_cisw{g}"] = sc(o2 + base)
    res[name] = row; print(name, json.dumps(row), flush=True)
json.dump(res, open("pds_screen2.json", "w"), indent=1)
