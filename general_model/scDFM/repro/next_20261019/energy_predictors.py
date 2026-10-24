"""Can a target's response strength in a NEW context be predicted from that context's control cells? (2026-10-20)
Spearman of the true noise-corrected energy with: |pooled trans|, the target's expression in the context's controls
(CP10k mean), its expression rank, the ratio of its context expression to its mean source expression, and products."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S
SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"
for name in ["H1", "KOLF200", "HCT116"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; Cc = S.load_cache(train)
    B = BE.load(name); D = B["D"].copy(); oc = B["oc"]; targets = B["targets"]
    for i, k in enumerate(B["own"]):
        if k >= 0: D[i, k] = 0
    dist = (D ** 2).sum(1) - B["v"]
    Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    t5 = E.remove_shared(B["full"] - oc).astype(np.float64); t8 = t5 - (t5 @ Vc.T) @ Vc; nq = np.linalg.norm(t8, axis=1)
    expr = B["rate"] * 1e4                                    # CP10k in the hold-out's controls
    ex_t = np.array([expr[k] if k >= 0 else np.nan for k in B["own"]])
    src = np.array([np.nanmean([Cc[c]["ctrl"][k] for c in train if Cc[c]["measured"][k]]) if k >= 0 else np.nan for k in B["own"]])   # CP50k
    ratio = np.log1p(ex_t * 5) - np.log1p(src)
    ok = np.isfinite(ex_t) & np.isfinite(src)
    sp_ = lambda a: spearmanr(a[ok], dist[ok]).correlation
    print(f"{name}: spearman with true energy: |pooled| {sp_(nq):.3f}, target expr {sp_(ex_t):.3f}, expr ratio ctx/sources {sp_(ratio):.3f}, "
          f"|pooled| x expr^0.5 {sp_(nq * np.sqrt(ex_t)):.3f}, |pooled| x exp(ratio/2) {sp_(nq * np.exp(ratio / 2)):.3f}", flush=True)
    o = np.argsort(-dist)[:8]
    print("   top energy targets (expr CP10k ctx / sources CP10k): " + "  ".join(f"{targets[i]} {ex_t[i]:.1f}/{src[i] / 5:.1f}" for i in o), flush=True)
