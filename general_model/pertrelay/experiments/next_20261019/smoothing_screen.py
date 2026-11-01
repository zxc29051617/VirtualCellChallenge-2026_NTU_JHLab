"""Screen (uses hold-out truth): does borrowing from functionally similar targets help? Target-target similarity
from TRAINING signatures only (cosine of the model's trans signatures over the 3,000 most variable genes); smoothed
s'_p = s_p + lam * mean of its k most similar targets' signatures (similarity > 0). Evaluated with the full prediction
(own + cis + 3 x trans) by the pds-like rank score and the median cosine with the real bulk delta."""
import sys, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay")
import coexp_screen as CS, sigexp as S
from vcc.gm import Genome
gen = Genome(S.GENES); Cc = S.load_cache(); T = S.target_list(); res = {}
for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); o = S.OUTER[name]; train = [c for c in S.CONTEXTS if c not in o["exclude"]]
    w = {c: 1.0 for c in train}
    sig = np.stack([S.trans_signature(Cc, train, w, t) for t in T]); Ti = {t: i for i, t in enumerate(T)}
    var = sig.var(0); top = np.argsort(-var)[:3000]
    Z = sig[:, top] / (np.linalg.norm(sig[:, top], axis=1, keepdims=True) + 1e-12); Sim = Z @ Z.T; np.fill_diagonal(Sim, -1)
    oc = S.build_effects(Cc, train, w, targets, gen, use_trans=False)
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True; m = ~panel & (rate > 0)
    row = {}
    for k, lam in [(0, 0.0), (5, 0.5), (5, 1.0), (10, 0.5), (20, 0.5)]:
        lfc = np.zeros((len(targets), S.G), np.float32)
        for i, t in enumerate(targets):
            j = Ti[t]; s = sig[j].copy()
            if k:
                nb = np.argsort(-Sim[j])[:k]; nb = nb[Sim[j, nb] > 0]
                if len(nb): s = s + lam * sig[nb].mean(0)
            lfc[i] = oc[i] + 3.0 * s / np.log(2)
        P = CS.pred_deltas(lfc, rate, mu_c)
        row[f"k{k}_lam{lam}"] = {"pds_like": round(CS.pds_like(P, D, m), 4), "median_cos": round(float(np.median(CS.cos_rows(P, D, m))), 4)}
    res[name] = row; print(name, row, flush=True)
json.dump(res, open("smoothing_screen.json", "w"), indent=1)
