"""Fast pds screen (the noise-free proxy equals the official raw pds for exact-total rendering: H1 0.9006, KOLF200
0.5706, HCT116 0.6469 vs official V1 0.9005/0.5706/0.6472). Variants use only predictions (no truth) to build the
change; the hold-out truth is used only to score this screen.
  shared PCs: remove the mean and the top-k principal components of the panel's trans signatures (components shared
              by many targets cannot help tell targets apart)
  cis x g:    scale the cis (neighbour) part by g"""
import sys, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM")
import coexp_screen as CS, sigexp as S
from vcc.gm import Genome
gen = Genome(S.GENES); Cc = S.load_cache(); res = {}
for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); o = S.OUTER[name]; train = [c for c in S.CONTEXTS if c not in o["exclude"]]
    w = {c: 1.0 for c in train}
    sig = np.stack([S.trans_signature(Cc, train, w, t) for t in targets]).astype(np.float64)
    oc = S.build_effects(Cc, train, w, targets, gen, use_trans=False).astype(np.float64)
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True; m = ~panel & (rate > 0)
    def score(lfc):
        P = CS.pred_deltas(lfc, rate, mu_c); return round(CS.pds_like(P, D, m), 4)
    row = {"base": score(oc + 3 * sig / np.log(2))}
    mu = sig.mean(0, keepdims=True); Sc = sig - mu
    U, sv, Vt = np.linalg.svd(Sc, full_matrices=False)
    row["mean_removed"] = score(oc + 3 * Sc / np.log(2))
    for k in [1, 2, 3, 5, 10]:
        Sk = Sc - (Sc @ Vt[:k].T) @ Vt[:k]
        row[f"pc{k}_removed"] = score(oc + 3 * Sk / np.log(2))
    for g in [1.5, 2.0, 3.0]:
        row[f"cis_x{g}"] = score(g * oc + 3 * sig / np.log(2))
    res[name] = row; print(name, row, flush=True)
json.dump(res, open("pds_screen.json", "w"), indent=1)
