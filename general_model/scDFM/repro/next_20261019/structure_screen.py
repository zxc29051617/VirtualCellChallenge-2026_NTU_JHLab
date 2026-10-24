"""How low-rank are the real per-target effects, and does our prediction capture the main directions? (screen)"""
import json, sys, numpy as np
sys.path.insert(0, "."); import coexp_screen as C
from vcc.gm import Genome
SIG = C.SIG
for name in sys.argv[1:]:
    targets, D, rate, mu_c = C.real_deltas(name)
    z = np.load(f"{SIG}/effects/{name}_A.npz", allow_pickle=True); P = C.pred_deltas(z["lfc"], rate, mu_c)
    panel = np.zeros(C.G, bool); panel[[C.GI[t] for t in targets if t in C.GI]] = True; m = ~panel & (rate > 0)
    Dm, Pm = D[:, m], P[:, m]
    tot = (Dm ** 2).sum(); mean = Dm.mean(0); Dc = Dm - mean
    U, S, Vt = np.linalg.svd(Dc, full_matrices=False); ev = S ** 2 / (Dc ** 2).sum()
    # global best scaling of our prediction (pooled) -> best achievable ratio for uniform amplitude
    a = (Pm * Dm).sum() / (Pm ** 2).sum(); e_opt = ((a * Pm - Dm) ** 2).sum() / tot; e3 = ((Pm - Dm) ** 2).sum() / tot
    # oracle: per-target best scaling of our prediction
    ap = (Pm * Dm).sum(1) / ((Pm ** 2).sum(1) + 1e-12); e_pt = ((ap[:, None] * Pm - Dm) ** 2).sum() / tot
    # how much of the real effect lies in our prediction's top-k subspace
    print(name, json.dumps({"n": len(targets), "mean_response_share": round(float((mean ** 2).sum() * len(targets) / tot), 4),
          "top_PC_var_share(centred)": [round(float(x), 3) for x in ev[:5]], "cum_top10": round(float(ev[:10].sum()), 3),
          "plain_ratio_alpha3(no noise corr)": round(float(e3), 3), "best_uniform_scale": round(float(a), 3),
          "ratio_at_best_uniform_scale": round(float(e_opt), 4), "ratio_per_target_oracle_scale": round(float(e_pt), 4)}))
