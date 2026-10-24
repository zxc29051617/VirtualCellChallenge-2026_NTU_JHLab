"""Which training source's measured responses transfer best to each hold-out? (screen, uses hold-out truth)
Per source c: cosine between c's common-mode-removed response to target p (as cached for the model) and the real
bulk delta of p in the hold-out, genes = all except panel targets, p's own gene and its cis window. Median over the
targets c has, and the same for the uniform average of all allowed sources (the model's trans part)."""
import sys, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM")
import coexp_screen as CS, render_eval as E, sigexp as S
from vcc.gm import Genome
gen = Genome(S.GENES); Cc = S.load_cache(); res = {}
for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); o = S.OUTER[name]; train = [c for c in S.CONTEXTS if c not in o["exclude"]]
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True
    base = ~panel & (rate > 0)
    def cos(a, b, m):
        a, b = a[m], b[m]; d = np.linalg.norm(a) * np.linalg.norm(b); return float(a @ b / d) if d > 0 else np.nan
    masks = []
    for t in targets:
        m = base.copy(); g, _ = gen.cis(t); m[g] = False
        if t in S.GI: m[S.GI[t]] = False
        masks.append(m)
    row = {}
    for c in train:
        v = []
        for i, t in enumerate(targets):
            j = Cc[c]["tidx"].get(t)
            if j is None: continue
            y = np.nan_to_num(Cc[c]["Ysrc"][j]); v.append(cos(y, D[i], masks[i]))
        if v: row[c] = (round(float(np.nanmedian(v)), 4), round(float(np.nanmean(np.array(v) > 0)), 2), len(v))
    w = {c: 1.0 for c in train}; v = []
    for i, t in enumerate(targets):
        s = S.trans_signature(Cc, train, w, t); v.append(cos(s, D[i], masks[i]))
    row["ALL (model)"] = (round(float(np.nanmedian(v)), 4), round(float(np.nanmean(np.array(v) > 0)), 2), len(v))
    res[name] = row
    print(name); [print(f"   {k:12s} median cos {x[0]:+.4f}  frac>0 {x[1]:.2f}  n {x[2]}") for k, x in sorted(row.items(), key=lambda kv: -kv[1][0])]
json.dump(res, open("source_transfer_screen.json", "w"), indent=1)
