"""How big is the shared (generic) part of the real deltas, and can it be predicted? (2026-10-20, cell-free)

    python generic_check.py          (pertrelay venv)

g_true = mean over targets of the real deltas (scorer space). Share = n |g_true|^2 / sum_p (|r_p|^2 - v_p).
Candidates for g (all from training sources / predictions, no hold-out truth): panel mean of the model's predicted trans
(log2 x3 -> divided by 3), the mean of the training sources' VCC-target responses (ln -> log2). mse_free of the
current-style totals (own + cis + 0.3 x V8 trans) plus gamma x g_hat for a few gamma.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, coexp_screen as CS
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
for name in ["H1", "KOLF200", "HCT116"]:
    B = BE.load(name); D = B["D"]; n = len(D); oc = B["oc"]; tr = B["full"] - oc
    m = np.ones(S.G, bool)
    g_true = D.mean(0); share = n * (g_true ** 2).sum() / ((D ** 2).sum() - B["v"].sum())
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    T = set(B["targets"])
    srcmean = []
    for c in train:
        Y = C[c]["Ysrc"]; rows = [j for j, t in enumerate(C[c]["targets"]) if t in T]
        if len(rows) >= 30: srcmean.append(np.nan_to_num(np.nanmean(Y[rows], 0)))
    g_src = np.mean(srcmean, 0) / np.log(2)                      # log2 ratio
    g_pred = tr.mean(0) / 3.0
    Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    t5 = E.remove_shared(tr).astype(np.float64); t8 = t5 - (t5 @ Vc.T) @ Vc
    base = oc + 0.3 * t8
    gq = lambda lg: CS.pred_deltas(lg[None], B["rate"], B["mu_c"])[0]
    cosv = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"{name}: generic share of real signal {share:.3f}; cos(g_true, g_pred) {cosv(g_true, gq(g_pred)):.3f}, cos(g_true, g_src) {cosv(g_true, gq(g_src)):.3f}", flush=True)
    for lab, g in [("g_pred", g_pred), ("g_src", g_src)]:
        out = []
        for gam in [0.0, 0.5, 1.0, 2.0, 3.0]:
            r = BE.score(B, base + gam * g[None]); out.append(f"x{gam}: mse {r['mse_free']:.4f} pds {r['pds']:.4f}")
        print(f"   {lab}: " + "  ".join(out), flush=True)
