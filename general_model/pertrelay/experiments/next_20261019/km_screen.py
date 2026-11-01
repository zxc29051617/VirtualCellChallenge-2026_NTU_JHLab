"""Shared-component count k and control-axis count m re-checked for the G1 totals (2026-10-20, cell-free): cos (mse at
budget = claim) and pds for depth200 + gamma 3 trans. Approximate leaderboard value: 1.94 x d(pds raw) + 0.95 x d(cos)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); oc = B["oc"]; targets = B["targets"]
    td = W.trans(C, train, targets, 0, W.WEIGHTS["depth200"])
    _, _, Vall = np.linalg.svd((lambda X: X - X.mean(0, keepdims=True))(
        (lambda Xc: (Xc.multiply(1e4 / B["lib"][:, None])).tocsr())(B["C"][np.random.default_rng(0).choice(B["C"].shape[0], min(2000, B["C"].shape[0]), replace=False)]).toarray() if False else E.control_axes(B["C"], B["lib"], 100)), full_matrices=False) if False else (None, None, E.control_axes(B["C"], B["lib"], 100).astype(np.float64))
    bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
    out = []; ref = None
    for k in [0, 5, 10, 20]:
        for m in [0, 20, 50, 100]:
            t = td - td.mean(0, keepdims=True)
            if k > 0:
                _, _, vt = np.linalg.svd(t, full_matrices=False); t = t - (t @ vt[:k].T) @ vt[:k]
            if m > 0: V = Vall[:m]; t = t - (t @ V.T) @ V
            t = t * (np.maximum(np.linalg.norm(t, axis=1), 1e-9) ** 2)[:, None]
            r = BE.score(B, oc + 0.6 * t * (bn / np.linalg.norm(t)))
            if (k, m) == (10, 50): ref = r
            out.append((k, m, r))
    print(name, " | ".join(f"k{k} m{m}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']):+.4f}" for k, m, r in out), flush=True)
